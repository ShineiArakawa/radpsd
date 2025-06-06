import conftest
import pytest
import torch

import radpsd


def test_cpu_cuda_compatibility(seed: int):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available, skipping CUDA tests.")

    conftest.seed_all(seed)

    img = torch.randn((16, 3, 64, 64), dtype=torch.float64)

    # Test CPU implementation
    psd_cpu = radpsd.compute_radial_psd(img, 360, 1024)

    # Test CUDA implementation
    img_cuda = img.to('cuda')
    psd_cuda = radpsd.compute_radial_psd(img_cuda, 360, 1024)

    # Ensure the results are close enough
    assert torch.allclose(psd_cpu, psd_cuda.cpu(), atol=1e-5), "CPU and CUDA results do not match within tolerance."
