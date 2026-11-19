#include "common.cuh"

void ggml_cuda_op_argsort(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// LLapis slim build: the CUB radix sort (the largest single object after mmvq) only serves ARGSORT
// over more than 1024 columns, and TOP_K when cub::DeviceTopK is missing (CCCL < 3.2). This model
// sorts 128 router logits (bitonic) and takes its vocabulary top-k through DeviceTopK. So with
// DeviceTopK available the radix sort is left out, and the backend declines ARGSORT > 1024 columns
// (ggml_backend_cuda_device_supports_op): it then runs on the CPU instead of aborting.
#if defined(LLAPIS_CUDA_SLIM) && defined(GGML_CUDA_USE_CUB)
#    include <cub/version.cuh>
#    if (CCCL_MAJOR_VERSION >= 3 && CCCL_MINOR_VERSION >= 2)
#        define LLAPIS_NO_ARGSORT_CUB
#    endif
#endif

#if defined(GGML_CUDA_USE_CUB) && !defined(LLAPIS_NO_ARGSORT_CUB)
void argsort_f32_i32_cuda_cub(ggml_cuda_pool & pool,
                              const float *    x,
                              int *            dst,
                              const int        ncols,
                              const int        nrows,
                              ggml_sort_order  order,
                              cudaStream_t     stream);
#endif  // GGML_CUDA_USE_CUB
void argsort_f32_i32_cuda_bitonic(const float *   x,
                                  int *           dst,
                                  const int       ncols,
                                  const int       nrows,
                                  ggml_sort_order order,
                                  cudaStream_t    stream);
