"""Signal processing module for computing power spectral density and radial profiles.
"""

import functools
import typing

import numpy as np
import torch

import radpsd.common as _common
import radpsd.torch_util as _torch_util

ArrayLike = typing.TypeVar('ArrayLike', np.ndarray, torch.Tensor)

# -----------------------------------------------------------------------------------------------------------------------------------------------------
# Extension loader


@functools.lru_cache()
def _get_cpp_module(is_cuda: bool = True, with_omp: bool = True) -> typing.Any:
    ext_loader = _torch_util.get_extension_loader()

    name = 'signal'
    sources = [
        'signal.cpp',
    ]

    if is_cuda:
        sources += [
            'signal.cu',
        ]
        name += '_cuda'

    module = ext_loader.load(
        name=name,
        sources=sources,
        debug=_common.GlobalSettings.DEBUG_MODE,
        with_omp=with_omp,
    )

    return module


# -----------------------------------------------------------------------------------------------------------------------------------------------------
# Utility functions


def _calc_radial_psd_impl(
    psd: torch.Tensor,
    n_divs: int,
    n_points: int,
    enable_omp: bool = True,
) -> torch.Tensor:
    """Calculate the radial power spectral density.

    Parameters
    ----------
    psd : torch.Tensor
        Input power spectral density. Must be 4D tensor with shape [batch, channel, height, width].
    n_divs : int
        Number of bins for the radial angle.
    n_points : int
        Number of points for the polar coordinate.
    enable_omp : bool, optional
        Whether to enable OpenMP for parallel processing, by default True.
        This only affects the C++ implementation.

    Returns
    -------
    torch.Tensor
        Radial power spectral density profile. The output will have the shape [batch, channel, n_divs, n_points].
    """

    # Convert to channels last format
    psd = psd.permute(0, 2, 3, 1).contiguous()  # [batch, height, width, channel]

    # Compute the power spectral density
    _module = _get_cpp_module(is_cuda=psd.is_cuda, with_omp=enable_omp)

    radial_profile = _module.calc_radial_psd(
        psd,
        n_divs,
        n_points
    )

    # Convert back to channels first format
    radial_profile = radial_profile.permute(0, 3, 1, 2)  # [batch, channel, n_divs, n_points]

    return radial_profile


# -----------------------------------------------------------------------------------------------------------------------------------------------------
# Public API

def compute_psd(
    img: ArrayLike,
    is_db_scale: bool = False,
    beta: float | None = None,
    padding_factor: int = 1,
) -> ArrayLike:
    """
    Compute the power spectral density of an image.

    Parameters
    ----------
    img : ArrayLike (np.ndarray or torch.Tensor)
        Input image to be processed.
        The input image can have 2, 3, or 4 dimensions.
        - 2D image: [height, width]
        - 3D image: [channels, height, width]
        - 4D image: [batch, channels, height, width]
    is_db_scale : bool, optional
        Whether to convert the power spectral density to dB scale, by default False
    beta : float, optional
        The beta parameter for the Kaiser window. If None, no windowing is applied.
    padding_factor : int, optional
        The padding_factor factor for the windowing, by default 4
        The input image will be padded to `padding_factor * img.shape[-2:]` before applying the FFT.
        If `padding_factor <= 1`, no padding is applied.

    Returns
    -------
    ArrayLike (np.ndarray or torch.Tensor)
        The power spectral density of the image. The output image will have the shape `[..., padding_factor * height, padding_factor * width]`.
    """

    # ----------------------------------------------------------------------------------------------------
    # If the input is a numpy array, convert it to a torch tensor
    is_np = isinstance(img, np.ndarray)
    if is_np:
        assert img.dtype in [np.float16, np.float32, np.float64], 'The input image must be a float type numpy array.'

        # Convert to torch.Tensor
        img = torch.tensor(img)

    # Check the input
    bc_shape = tuple()

    if img.ndim == 2:
        # Add the batch and channel dimensions
        img = img.unsqueeze(0).unsqueeze(0)
    elif img.ndim == 3:
        bc_shape = (img.shape[0],)
        # Add the batch dimension
        img = img.unsqueeze(0)
    elif img.ndim == 4:
        # Already in the batch and channel format
        bc_shape = (img.shape[0], img.shape[1])
    else:
        raise ValueError(f'Input image must have 2, 3, or 4 dimensions, but got {img.ndim}.')

    img_h, img_w = img.shape[-2:]

    # ----------------------------------------------------------------------------------------------------
    # Windowing

    if beta is not None and beta >= 0.0:
        assert padding_factor > 1, 'Interpolation must be greater than 1 when beta is specified.'

        # Prepare kaiser window
        short_side = max(img_h, img_w)

        window = torch.kaiser_window(short_side, periodic=False, beta=beta, device=img.device, dtype=img.dtype)
        window *= window.square().sum().rsqrt()
        window = window.ger(window)  # [short_side, short_side]
        window = window.unsqueeze(0).unsqueeze(0)  # [1, 1, short_side, short_side]

        if short_side != img_h or short_side != img_w:
            padding_h = (short_side - img_h) // 2
            padding_w = (short_side - img_w) // 2
            window = torch.nn.functional.pad(
                window,
                (padding_w, padding_w, padding_h, padding_h)  # [left, right, top, bottom]
            )

        assert window.shape[-2] == img_h
        assert window.shape[-1] == img_w

        # Apply window
        img = img * window

    # ----------------------------------------------------------------------------------------------------
    # Padding

    if padding_factor > 1:
        padding_h = (img_h * padding_factor - img_h)
        padding_w = (img_w * padding_factor - img_w)

        img = torch.nn.functional.pad(img, (0, padding_w, 0, padding_h))

    # ----------------------------------------------------------------------------------------------------
    # Apply FFT

    spectrum = torch.fft.fftn(img, dim=(-2, -1)).abs().square()  # [batch, channel, height, width]
    spectrum = torch.fft.fftshift(spectrum, dim=(-2, -1))  # Shift the zero frequency component to the center

    spectrum = spectrum / (img.shape[-2] * img.shape[-1])  # Normalize

    if is_db_scale:
        spectrum = 10.0 * torch.log10(spectrum + 1e-10)

    # ----------------------------------------------------------------------------------------------------
    # Reshape the output to match the expected shape
    spectrum = spectrum.reshape(*bc_shape, spectrum.shape[-2], spectrum.shape[-1])  # [batch, channel, height, width]

    # Convert back to numpy array if the input was a numpy array
    if is_np:
        # Convert back to numpy array
        spectrum = spectrum.cpu().numpy()

    # ----------------------------------------------------------------------------------------------------
    return spectrum


def compute_radial_psd(
    img: ArrayLike,
    n_angles: int,
    n_radial_bins: int,
    kaiser_beta: float = 8.0,
    padding_factor: int = 4,
) -> ArrayLike:
    """
    Compute the radial power spectral density of images.

    Parameters
    ----------
    img : ArrayLike (np.ndarray or torch.Tensor)
        Input image to be processed.
        The input image can have 2, 3, or 4 dimensions.
        - 2D image: [height, width]
        - 3D image: [channels, height, width]
        - 4D image: [batch, channels, height, width]
    n_angles : int, optional
        Number of divisions for the radial angle.
        The radial profile will be divided into `n_angles` bins.
    n_radial_bins : int, optional
        Number of points in the radial direction.
        The radial profile will have `n_radial_bins` points.
    kaiser_beta : float, optional
        Beta parameter for the Kaiser window, by default 8.0.
    padding_factor : int, optional
        Padding factor for the FFT computation, by default 4.

    Returns
    -------
    ArrayLike (np.ndarray or torch.Tensor)
        Radial power spectral density profile. The output image will have the shape `[..., n_angles, n_radial_bins]`.

    Raises
    ------
    ValueError
        If the input image has an invalid number of dimensions.
    """

    # ----------------------------------------------------------------------------------------------------
    # If the input is a numpy array, convert it to a torch tensor
    is_np = isinstance(img, np.ndarray)
    if is_np:
        # Convert to torch.Tensor
        img = torch.tensor(img).float()

    # Check the input
    bc_shape = tuple()

    if img.ndim == 2:
        # Add the batch and channel dimensions
        img = img.unsqueeze(0).unsqueeze(0)
    elif img.ndim == 3:
        # Add the batch dimension
        bc_shape = (img.shape[0],)
        img = img.unsqueeze(0)
    elif img.ndim == 4:
        # Already in the batch and channel format
        bc_shape = (img.shape[0], img.shape[1])
    else:
        raise ValueError(f'Input image must have 2, 3, or 4 dimensions, but got {img.ndim}.')

    # ----------------------------------------------------------------------------------------------------
    # Execute FFT with Kaiser windowing
    psd = compute_psd(img, is_db_scale=False, beta=kaiser_beta, padding_factor=padding_factor)

    # ----------------------------------------------------------------------------------------------------
    # Compute the radial power spectral density
    radial_psd = _calc_radial_psd_impl(
        psd=psd,
        n_divs=n_angles,
        n_points=n_radial_bins,
        enable_omp=True
    )

    # ----------------------------------------------------------------------------------------------------
    # Reshape the output to match the expected shape
    radial_psd = radial_psd.reshape(*bc_shape, n_angles, n_radial_bins)

    # If the input was a numpy array, convert the output back to numpy
    if is_np:
        # Convert back to numpy array
        radial_psd = radial_psd.cpu().numpy()

    return radial_psd


def radial_freq(
    img_size: int,
    n_radial_bins: int,
    dtype: typing.Type = np.float32,
) -> np.ndarray:
    """Generate radial frequency values for a given image size and number of points.

    Parameters
    ----------
    img_size : int
        The size of the image (assumed to be square).
    n_radial_bins : int
        The number of points in the radial direction.
    dtype : type, optional
        The data type of the output array, by default np.float32.

    Returns
    -------
    np.ndarray
        The radial frequency values for the real parts.
    """

    assert isinstance(img_size, int) and img_size > 0, 'img_size must be a positive integer.'
    assert isinstance(n_radial_bins, int) and n_radial_bins > 0, 'n_radial_bins must be a positive integer.'

    max_index = (img_size - 1) // 2

    # NOTE: Reference for the frequency calculation:
    #     'numpy.fft.fftfreq' in NumPy's API reference (URL: https://numpy.org/doc/2.2/reference/generated/numpy.fft.fftfreq.html#numpy-fft-fftfreq)
    #     'numpy/fft/_helper.py' (URL: https://github.com/numpy/numpy/blob/e7a123b2d3eca9897843791dd698c1803d9a39c2/numpy/fft/_helper.py#L125-L177)
    #
    # f = [0, 1, ...,   n/2-1,     -n/2, ..., -1] / (d*n)   if n is even
    # f = [0, 1, ..., (n-1)/2, -(n-1)/2, ..., -1] / (d*n)   if n is odd

    freq = np.linspace(0.0, float(max_index), n_radial_bins, dtype=dtype, endpoint=True) / (float(img_size) * 1.0)  # d = 1.0

    return freq

# -----------------------------------------------------------------------------------------------------------------------------------------------------
