import numpy as np
import torch

import radpsd


def example_compute_psd():
    batch_size = 8
    channels = 3
    img_size = 256

    padding_factor = 4

    # Compute power spectral density (PSD) of a random image

    # ---------------------------------------------------------------------------
    # Numpy, shape: (batch_size, channels, height, width)
    img = np.random.randn(batch_size, channels, img_size, img_size)
    psd = radpsd.compute_psd(img, padding_factor=padding_factor)

    assert psd.shape == (batch_size, channels, img_size * padding_factor, img_size * padding_factor)

    # Numpy, shape: (channels, height, width)
    img = np.random.randn(channels, img_size, img_size)
    psd = radpsd.compute_psd(img, padding_factor=padding_factor)

    assert psd.shape == (channels, img_size * padding_factor, img_size * padding_factor)

    # Numpy, shape: (height, width)
    img = np.random.randn(img_size, img_size)
    psd = radpsd.compute_psd(img, padding_factor=padding_factor)

    assert psd.shape == (img_size * padding_factor, img_size * padding_factor)

    # ---------------------------------------------------------------------------
    # Torch, shape: (batch_size, channels, height, width)
    img = torch.randn(batch_size, channels, img_size, img_size)
    img = radpsd.compute_psd(img, padding_factor=padding_factor)

    assert img.shape == (batch_size, channels, img_size * padding_factor, img_size * padding_factor)

    # Torch, shape: (channels, height, width)
    img = torch.randn(channels, img_size, img_size)
    img = radpsd.compute_psd(img, padding_factor=padding_factor)

    assert img.shape == (channels, img_size * padding_factor, img_size * padding_factor)

    # Torch, shape: (height, width)
    img = torch.randn(img_size, img_size)
    img = radpsd.compute_psd(img, padding_factor=padding_factor)

    # ---------------------------------------------------------------------------
    if torch.cuda.is_available():
        print("Using CUDA for computation.")

        device = torch.device(f'cuda:{torch.cuda.current_device()}')

        # Torch, shape: (batch_size, channels, height, width)
        img = torch.randn(batch_size, channels, img_size, img_size, device=device)
        img = radpsd.compute_psd(img, padding_factor=padding_factor)

        assert img.shape == (batch_size, channels, img_size * padding_factor, img_size * padding_factor)

        # Torch, shape: (channels, height, width)
        img = torch.randn(channels, img_size, img_size, device=device)
        img = radpsd.compute_psd(img, padding_factor=padding_factor)

        assert img.shape == (channels, img_size * padding_factor, img_size * padding_factor)

        # Torch, shape: (height, width)
        img = torch.randn(img_size, img_size, device=device)
        img = radpsd.compute_psd(img, padding_factor=padding_factor)

        assert img.shape == (img_size * padding_factor, img_size * padding_factor)

    assert img.shape == (img_size * padding_factor, img_size * padding_factor)


def example_compute_radial_psd():
    batch_size = 8
    channels = 3
    img_size = 256

    padding_factor = 4
    n_angles = 360 * 2
    n_radial_bins = img_size * 4

    # ---------------------------------------------------------------------------
    # Compute radial power spectral density (PSD) of a random image

    # Numpy, shape: (batch_size, channels, height, width)
    img = np.random.randn(batch_size, channels, img_size, img_size)
    radial_psd = radpsd.compute_radial_psd(img, padding_factor=padding_factor, n_angles=n_angles, n_radial_bins=n_radial_bins)

    assert radial_psd.shape == (batch_size, channels, n_angles, n_radial_bins)

    # Numpy, shape: (channels, height, width)
    img = np.random.randn(channels, img_size, img_size)
    radial_psd = radpsd.compute_radial_psd(img, padding_factor=padding_factor, n_angles=n_angles, n_radial_bins=n_radial_bins)

    assert radial_psd.shape == (channels, n_angles, n_radial_bins)

    # Numpy, shape: (height, width)
    img = np.random.randn(img_size, img_size)
    radial_psd = radpsd.compute_radial_psd(img, padding_factor=padding_factor, n_angles=n_angles, n_radial_bins=n_radial_bins)

    assert radial_psd.shape == (n_angles, n_radial_bins)

    # ---------------------------------------------------------------------------
    # Torch, shape: (batch_size, channels, height, width)
    img = torch.randn(batch_size, channels, img_size, img_size)
    radial_psd = radpsd.compute_radial_psd(img, padding_factor=padding_factor, n_angles=n_angles, n_radial_bins=n_radial_bins)

    assert radial_psd.shape == (batch_size, channels, n_angles, n_radial_bins)

    # Torch, shape: (channels, height, width)
    img = torch.randn(channels, img_size, img_size)
    radial_psd = radpsd.compute_radial_psd(img, padding_factor=padding_factor, n_angles=n_angles, n_radial_bins=n_radial_bins)

    assert radial_psd.shape == (channels, n_angles, n_radial_bins)

    # Torch, shape: (height, width)
    img = torch.randn(img_size, img_size)
    radial_psd = radpsd.compute_radial_psd(img, padding_factor=padding_factor, n_angles=n_angles, n_radial_bins=n_radial_bins)

    assert radial_psd.shape == (n_angles, n_radial_bins)

    # ---------------------------------------------------------------------------
    if torch.cuda.is_available():
        print("Using CUDA for computation.")

        device = torch.device(f'cuda:{torch.cuda.current_device()}')

        # Torch, shape: (batch_size, channels, height, width)
        img = torch.randn(batch_size, channels, img_size, img_size, device=device)
        radial_psd = radpsd.compute_radial_psd(img, padding_factor=padding_factor, n_angles=n_angles, n_radial_bins=n_radial_bins)

        assert radial_psd.shape == (batch_size, channels, n_angles, n_radial_bins)

        # Torch, shape: (channels, height, width)
        img = torch.randn(channels, img_size, img_size, device=device)
        radial_psd = radpsd.compute_radial_psd(img, padding_factor=padding_factor, n_angles=n_angles, n_radial_bins=n_radial_bins)

        assert radial_psd.shape == (channels, n_angles, n_radial_bins)

        # Torch, shape: (height, width)
        img = torch.randn(img_size, img_size, device=device)
        radial_psd = radpsd.compute_radial_psd(img, padding_factor=padding_factor, n_angles=n_angles, n_radial_bins=n_radial_bins)

        assert radial_psd.shape == (n_angles, n_radial_bins)


def main():
    example_compute_psd()
    print("Example compute_psd passed successfully.")

    example_compute_radial_psd()
    print("Example compute_radial_psd passed successfully.")


if __name__ == "__main__":
    main()
