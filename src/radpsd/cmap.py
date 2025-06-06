"""Utility functions for color maps
"""

import pathlib
import typing

import numpy as np
import torch

# ------------------------------------------------------------------------------------------------------------
# Lookup table for color maps

_cmap_luts: dict[str, np.ndarray] = {}

_cmap_lut_file = pathlib.Path(__file__).parent / 'cmap' / 'cmap_luts.npz'
if _cmap_lut_file.exists():
    with np.load(_cmap_lut_file) as data:
        for key, value in data.items():
            _cmap_luts[key] = value

# ------------------------------------------------------------------------------------------------------------
# Interpolation function
#
# Imported from: https://github.com/pytorch/pytorch/issues/50334#issuecomment-1247611276


def _interp(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor, dim: int = -1, extrapolate: str = 'constant') -> torch.Tensor:
    """One-dimensional linear interpolation between monotonically increasing sample
    points, with extrapolation beyond sample points.

    Returns the one-dimensional piecewise linear interpolant to a function with
    given discrete data points :math:`(xp, fp)`, evaluated at :math:`x`.

    Args:
        x: The :math:`x`-coordinates at which to evaluate the interpolated
            values.
        xp: The :math:`x`-coordinates of the data points, must be increasing.
        fp: The :math:`y`-coordinates of the data points, same shape as `xp`.
        dim: Dimension across which to interpolate.
        extrapolate: How to handle values outside the range of `xp`. Options are:
            - 'linear': Extrapolate linearly beyond range of xp values.
            - 'constant': Use the boundary value of `fp` for `x` values outside `xp`.

    Returns:
        The interpolated values, same size as `x`.
    """
    # Move the interpolation dimension to the last axis
    x = x.movedim(dim, -1)
    xp = xp.movedim(dim, -1)
    fp = fp.movedim(dim, -1)

    m = torch.diff(fp) / torch.diff(xp)  # slope
    b = fp[..., :-1] - m * xp[..., :-1]  # offset
    indices = torch.searchsorted(xp, x, right=False)

    if extrapolate == 'constant':
        # Pad m and b to get constant values outside of xp range
        m = torch.cat([torch.zeros_like(m)[..., :1], m, torch.zeros_like(m)[..., :1]], dim=-1)
        b = torch.cat([fp[..., :1], b, fp[..., -1:]], dim=-1)
    else:  # extrapolate == 'linear'
        indices = torch.clamp(indices - 1, 0, m.shape[-1] - 1)

    values = m.gather(-1, indices) * x + b.gather(-1, indices)

    return values.movedim(-1, dim)


# ------------------------------------------------------------------------------------------------------------
# Converting functions


ArrayLike = typing.TypeVar('ArrayLike', np.ndarray, torch.Tensor)


def apply_color_map(input_img: ArrayLike, color_map_type: str = 'viridis') -> ArrayLike:
    assert input_img.ndim in (2, 3), "Input image must be a 2D (HW) or 3D (BHW) array."
    assert color_map_type in _cmap_luts, f"Color map '{color_map_type}' is not supported. You can choose from: {list(_cmap_luts.keys())}"

    is_np = isinstance(input_img, np.ndarray)
    img = torch.tensor(input_img) if is_np else input_img

    has_batch_dim = (img.ndim == 3)
    if not has_batch_dim:
        img = img.unsqueeze(0)  # Add batch dimension
    assert img.ndim == 3

    n_data, height, width = img.shape
    orig_dtype = img.dtype

    img = img.float()  # Convert to float for processing
    img = img.flatten(1)  # [n_data, height * width]

    orig_min = img.min(dim=1, keepdim=True).values
    orig_max = img.max(dim=1, keepdim=True).values

    # Normalize the image to [0, 1]
    img = (img - orig_min) / (orig_max - orig_min + 1e-8)  # Avoid division by zero

    # Apply the colormap (linear interpolation)
    lut = _cmap_luts[color_map_type]
    lut = torch.tensor(lut, dtype=img.dtype, device=img.device)
    n_points = lut.shape[0]

    xp = torch.linspace(0.0, 1.0, n_points, dtype=img.dtype, device=img.device)
    fp_r = lut[:, 0]  # Red channel
    fp_g = lut[:, 1]  # Green channel
    fp_b = lut[:, 2]  # Blue channel

    flattened_img = img.reshape(-1)
    colorred = torch.stack(
        (
            _interp(flattened_img, xp, fp_r, dim=-1, extrapolate='constant').reshape(n_data, -1),
            _interp(flattened_img, xp, fp_g, dim=-1, extrapolate='constant').reshape(n_data, -1),
            _interp(flattened_img, xp, fp_b, dim=-1, extrapolate='constant').reshape(n_data, -1)
        ),
        dim=-1
    )  # [n_data, height * width, 3]

    # Convert back to original range
    colorred = colorred * (orig_max.unsqueeze(-1) - orig_min.unsqueeze(-1)) + orig_min.unsqueeze(-1)

    # Convert back to original dtype
    colorred = colorred.to(orig_dtype)

    # Reshape back to original shape
    colorred = colorred.reshape(n_data, height, width, 3)  # [n_data, height, width, 3]

    if not has_batch_dim:
        colorred = colorred.squeeze(0)

    if is_np:
        colorred = colorred.detach().cpu().numpy()

    return colorred
