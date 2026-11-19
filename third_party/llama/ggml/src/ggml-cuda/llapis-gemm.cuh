#pragma once

#include "common.cuh"

// LLapis: a plain tiled FP32-accumulate GEMM that takes the place of cuBLAS in ggml_cuda_mul_mat.
//
// WHY. cuBLAS (cublas64 + cublasLt64) is ~490 MB of DLLs, 87% of the release zip, and in this
// runtime it serves only the shapes no ggml kernel covers: the MoE router gate in prefill
// (f32 [2816 x 128] x f32 [2816 x N], N > 16), the vision encoder's large matmuls, and the
// odd non-contiguous case. mmf cannot take the gate on every card -- its F32 path needs Ampere
// (RTX 20 has none) -- so the replacement is one generic kernel that works on any architecture.
// It is not cuBLAS-fast and does not need to be: the gate is ~2 GFLOP per layer per prompt.
//
// ON by default. LLAPIS_CUBLAS=1 restores cuBLAS (the DLLs must then be next to ggml-cuda.dll;
// the release zip does not ship them). cuBLAS is delay-loaded, so a process that never calls it
// never maps it.
//
// Returns false (and does nothing) for a shape it does not handle; the caller then falls through
// to cuBLAS as before.
bool llapis_gemm_mul_mat(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);

// true unless LLAPIS_CUBLAS=1 is set (read once).
bool llapis_gemm_enabled();
