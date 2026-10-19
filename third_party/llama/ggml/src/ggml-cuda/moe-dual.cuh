#pragma once

#include "common.cuh"

// Selective dual-precision MoE matmul op (LLapis).
// Computes each routed (token,expert) pair once at its resident tier (hot pool vs cold pool),
// reusing the quantized mul_mat_id path behind a device-count guard, then scatters to pair order.
void ggml_cuda_moe_dual(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
