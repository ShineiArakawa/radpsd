#pragma once

#include <cuda_common.cuh>

// ---------------------------------------------------------------------------------------------------------
// CUDA kernels
// ---------------------------------------------------------------------------------------------------------

#if defined(__CUDACC__)
template <typename scalar_t>
static __device__ __forceinline__ int4 get_neighbor_pixel_ids(
    const scalar_t u,
    const scalar_t v,
    const int input_width,
    const int input_height) {
  const int x_idx_low = floor(u * static_cast<scalar_t>(input_width) - 0.5);   // [-1, input_width - 1]
  const int y_idx_low = floor(v * static_cast<scalar_t>(input_height) - 0.5);  // [-1, input_height - 1]
  const int x_idx_high = x_idx_low + 1;                                        // [0, input_width]
  const int y_idx_high = y_idx_low + 1;                                        // [0, input_height]

  return make_int4(x_idx_low, y_idx_low, x_idx_high, y_idx_high);
}
#endif
