import time

import torch

import radpsd

assert torch.cuda.is_available(), "CUDA is not available. Please run on a machine with CUDA support."

NUM_ITERS = 50
DTYPE = torch.float64
DATA_SHAPE = (64, 3, 128, 128)


# ------------------------------------------------------------------------------------
# Dry run to compile the sources

# Compile the OpenMP implementation
radpsd.compute_radial_psd(
    img=torch.randn((128, 128), dtype=DTYPE),
    n_angles=16,
    n_radial_bins=64,
)
print("OpenMP implementation compiled successfully.")

# Compile the CUDA implementation
radpsd.compute_radial_psd(
    img=torch.randn((128, 128), dtype=DTYPE).cuda(),
    n_angles=16,
    n_radial_bins=64,
)
print("CUDA implementation compiled successfully.")


# ------------------------------------------------------------------------------------
# OpenMP implementation

elapsed_time = 0.0

for i_iter in range(NUM_ITERS):
    print(f'Iteration {i_iter + 1}/{NUM_ITERS} for OpenMP implementation...')
    img = torch.randn(DATA_SHAPE, dtype=DTYPE)

    start_time = time.perf_counter()
    radpsd.compute_radial_psd(
        img=img,
        n_angles=360 * 2,
        n_radial_bins=img.shape[-1] * 4,
    )
    end_time = time.perf_counter()

    elapsed_time += end_time - start_time

elapsed_time /= NUM_ITERS
print(f"OpenMP implementation: {elapsed_time:.6f} seconds for {NUM_ITERS} iterations.")

# ------------------------------------------------------------------------------------
# CUDA implementation

elapsed_time = 0.0

for i_iter in range(NUM_ITERS):
    print(f'Iteration {i_iter + 1}/{NUM_ITERS} for CUDA implementation...')
    img = torch.randn(DATA_SHAPE, dtype=DTYPE).cuda()

    start_time = time.perf_counter()
    radpsd.compute_radial_psd(
        img=img,
        n_angles=360 * 2,
        n_radial_bins=img.shape[-1] * 4,
    )
    end_time = time.perf_counter()

    elapsed_time += end_time - start_time

elapsed_time /= NUM_ITERS
print(f"CUDA implementation: {elapsed_time:.6f} seconds for {NUM_ITERS} iterations.")
# ------------------------------------------------------------------------------------
