#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

#ifdef GGML_USE_HIP
#define GGML_CUDA_NAME "ROCm"
#define GGML_CUBLAS_NAME "hipBLAS"
#elif defined(GGML_USE_MUSA)
#define GGML_CUDA_NAME "MUSA"
#define GGML_CUBLAS_NAME "muBLAS"
#else
#define GGML_CUDA_NAME "CUDA"
#define GGML_CUBLAS_NAME "cuBLAS"
#endif
#define GGML_CUDA_MAX_DEVICES       16

// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_cuda_init(int device);

GGML_BACKEND_API bool ggml_backend_is_cuda(ggml_backend_t backend);

// device buffer
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_buffer_type(int device);

// conduct allreduce operation between devices
GGML_BACKEND_API bool ggml_backend_cuda_allreduce_tensor(ggml_backend_t * backends, struct ggml_tensor ** tensors, size_t n_backends);

// split tensor buffer that splits matrices by rows across multiple devices
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_split_buffer_type(int main_device, const float * tensor_split);

// pinned host buffer for use with the CPU backend for faster copies between CPU and GPU
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_host_buffer_type(void);

GGML_BACKEND_API int  ggml_backend_cuda_get_device_count(void);
GGML_BACKEND_API void ggml_backend_cuda_get_device_description(int device, char * description, size_t description_size);
GGML_BACKEND_API void ggml_backend_cuda_get_device_memory(int device, size_t * free, size_t * total);

// [moe-dual] scratch-pool occupancy, summed over every live pool on `device`
// (pools are per-device AND per-stream, created lazily, so there is no single owner to ask).
// `reserved` = device memory the pools hold from the driver; `used` = the part currently handed
// out; reserved - used is committed-but-idle scratch, which the VMM pool never returns. `n_pools`
// reports how many contributed. device < 0 sums all devices. Any out-pointer may be NULL.
// Exists because this was the largest block of VRAM the runtime could not name (a ~466 MiB
// residual on an 8 GB card = 4+ hot-expert slots).
GGML_BACKEND_API void ggml_backend_cuda_pool_stats(int device, size_t * reserved, size_t * used, int * n_pools);

GGML_BACKEND_API bool ggml_backend_cuda_register_host_buffer(void * buffer, size_t size);
GGML_BACKEND_API void ggml_backend_cuda_unregister_host_buffer(void * buffer);

// Instant pin: page-lock a host buffer in place (no device mapping) so H2D staging
// from it is async + fast. 1 = pinned (or already registered), 0 = failed. Used to pin the whole
// owned IQ2 cold arena at load, before any prefill. Undo with the unregister below.
GGML_BACKEND_API int  ggml_backend_cuda_host_register_pinned(void * buffer, size_t size);
GGML_BACKEND_API int  ggml_backend_cuda_host_register_mapped(void * buffer, size_t size);   // findings 233
GGML_BACKEND_API void * ggml_backend_cuda_host_map(void * base, size_t size);   // device alias of a host range (claude dump findings 248)
GGML_BACKEND_API void ggml_backend_cuda_host_unregister(void * buffer);

// Device-side routing counters: the fused-op
// partition kernel accumulates [n_layers][n_expert] routed-pair counts + [n_layers][2] hot/cold pair
// counts on device; the host fetches (and resets) them only at cache-refresh points, replacing the
// per-layer eval-callback read of ffn_moe_topk (which forces a backend sync at every layer).
GGML_BACKEND_API int  ggml_backend_cuda_moe_dual_route_counts_enable(int n_layers, int n_expert);
GGML_BACKEND_API int  ggml_backend_cuda_moe_dual_route_counts_fetch(int32_t * dst, int reset);
struct moert_cuda_stats;
GGML_BACKEND_API void ggml_backend_cuda_moe_dual_stats(struct moert_cuda_stats * out);
// Batch transport (moert/cuda.h, findings 216k-216l)
GGML_BACKEND_API int  ggml_backend_cuda_moe_dual_dir_init(int sm_blocks, int sm_ring_mib, int sm_stage_threads);   // sm_blocks 0 = copy engine, >0 = SM copy kernel from a pinned ring
GGML_BACKEND_API int  ggml_backend_cuda_moe_dual_dir_async(void * dst_dev, const void * src, uint64_t nbytes);
GGML_BACKEND_API int64_t ggml_backend_cuda_moe_dual_dir_mark(void);
GGML_BACKEND_API int64_t ggml_backend_cuda_moe_dual_dir_poll(int wait_all);
GGML_BACKEND_API int  ggml_backend_cuda_moe_dual_publish_h2d(int n, void * const * dsts, const void * const * srcs, const uint64_t * sizes);
struct moert_cuda_dir_stats;
GGML_BACKEND_API void ggml_backend_cuda_moe_dual_dir_stats(struct moert_cuda_dir_stats * out);
GGML_BACKEND_API int64_t ggml_backend_cuda_moe_dual_dir_gate(int op, int64_t arg);   // early-decision gate (findings 234)
GGML_BACKEND_API void ggml_backend_cuda_moe_dual_hot_table(int il, int kind, const void * const * dev_ptrs, int n, int type_tag);   // loaned slots (claude dump findings 246)


// Push the moert-cuda island config (moert/cuda.h; the island reads
// no env vars — the host parses the MOE_DUAL_* compat surface and forwards it here before the
// first decode). The island's state lives inside this library (the trampoline links moert-cuda);
// host code must use these forwarders rather than linking moert-cuda itself.
struct moert_cuda_config;
GGML_BACKEND_API void ggml_backend_cuda_moe_dual_configure(const struct moert_cuda_config * cfg);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_cuda_reg(void);

#ifdef  __cplusplus
}
#endif
