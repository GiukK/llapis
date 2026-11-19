#include "llapis-gemm.cuh"
#include "convert.cuh"

#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// See llapis-gemm.cuh for why this exists.
//
// dst[m, n, i2, i3] = sum_k src0[k, m, i2/r2, i3/r3] * src1[k, n, i2, i3]   (ggml mul_mat, broadcast over 2,3)
//
// 64x64 output tile per block, 256 threads, 4x4 outputs per thread, K in steps of 16 through shared
// memory. FP32 multiply-add throughout (cuBLAS in this backend ran F32 in TF32 tensor-op mode, so
// the router logits move in the last bits; the gate for that is the router statistics).

#define LG_BM 64
#define LG_BN 64
#define LG_BK 16
#define LG_NT 256

template <typename T> static __device__ __forceinline__ float lg_to_f(T x);
template <> __device__ __forceinline__ float lg_to_f<float>(float x) { return x; }
template <> __device__ __forceinline__ float lg_to_f<half>(half x) { return __half2float(x); }
template <> __device__ __forceinline__ float lg_to_f<nv_bfloat16>(nv_bfloat16 x) { return __bfloat162float(x); }

template <typename T0, typename T1>
static __global__ void __launch_bounds__(LG_NT) llapis_gemm_kernel(
        const T0 * __restrict__ src0, const T1 * __restrict__ src1, float * __restrict__ dst,
        const int K, const int M, const int N,
        const int64_t s00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t s10, const int64_t s11, const int64_t s12, const int64_t s13,
        const int64_t d1,  const int64_t d2,  const int64_t d3,
        const int ne12, const int r2, const int r3, const int ksplit, const int kchunk, float * __restrict__ part) {
    __shared__ float As[LG_BK][LG_BM + 1];
    __shared__ float Bs[LG_BK][LG_BN + 1];

    // blockIdx.z = batch * ksplit + slice: each slice sums K in [kbeg, kend)
    const int slice = blockIdx.z % ksplit;
    const int batch = blockIdx.z / ksplit;
    const int i2 = batch % ne12;
    const int i3 = batch / ne12;
    const int kbeg = slice * kchunk;
    const int kend = min(K, kbeg + kchunk);
    const T0 * A = src0 + (i2 / r2) * s02 + (i3 / r3) * s03;
    const T1 * B = src1 + i2 * s12 + i3 * s13;
    float    * C = dst  + i2 * d2  + i3 * d3;

    const int m0  = blockIdx.x * LG_BM;
    const int n0  = blockIdx.y * LG_BN;
    const int tid = threadIdx.x;
    const int tx  = tid % 16;
    const int ty  = tid / 16;

    float acc[4][4] = {};

    for (int k0 = kbeg; k0 < kend; k0 += LG_BK) {
        // consecutive threads walk k first: k is the contiguous axis of both operands
#pragma unroll
        for (int i = tid; i < LG_BK * LG_BM; i += LG_NT) {
            const int kk = i % LG_BK, mm = i / LG_BK;
            const int k = k0 + kk, m = m0 + mm;
            As[kk][mm] = (k < kend && m < M) ? lg_to_f(A[k * s00 + m * s01]) : 0.0f;
        }
#pragma unroll
        for (int i = tid; i < LG_BK * LG_BN; i += LG_NT) {
            const int kk = i % LG_BK, nn = i / LG_BK;
            const int k = k0 + kk, n = n0 + nn;
            Bs[kk][nn] = (k < kend && n < N) ? lg_to_f(B[k * s10 + n * s11]) : 0.0f;
        }
        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < LG_BK; ++kk) {
            float a[4], b[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) { a[i] = As[kk][ty + 16 * i]; b[i] = Bs[kk][tx + 16 * i]; }
#pragma unroll
            for (int i = 0; i < 4; ++i) {
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    acc[i][j] = fmaf(a[i], b[j], acc[i][j]);
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int m = m0 + ty + 16 * i;
        if (m >= M) continue;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int n = n0 + tx + 16 * j;
            if (n >= N) continue;
            if (ksplit == 1) {
                C[m + n * d1] = acc[i][j];
            } else {
                part[((int64_t) blockIdx.z * N + n) * M + m] = acc[i][j];   // [batch][slice][n][m]
            }
        }
    }
}

// the split-K partials summed in slice order: deterministic, unlike atomics
static __global__ void llapis_gemm_reduce(const float * __restrict__ part, float * __restrict__ dst,
        const int M, const int N, const int ne12, const int ksplit,
        const int64_t d1, const int64_t d2, const int64_t d3) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int batch = blockIdx.y;
    if (i >= (int64_t) M * N) {
        return;
    }
    const int m = (int) (i % M), n = (int) (i / M);
    const float * p = part + (int64_t) batch * ksplit * M * N + i;
    float sum = 0.0f;
    for (int s = 0; s < ksplit; ++s) {
        sum += p[(int64_t) s * M * N];
    }
    dst[(batch % ne12) * d2 + (batch / ne12) * d3 + m + n * d1] = sum;
}

template <typename T0, typename T1>
static void llapis_gemm_launch(ggml_backend_cuda_context & ctx, const T0 * src0, const T1 * src1, float * dst,
        const ggml_tensor * t0, const ggml_tensor * t1, const ggml_tensor * td) {
    cudaStream_t stream = ctx.stream();
    const size_t ts0 = sizeof(T0), ts1 = sizeof(T1);
    const int K = (int) t0->ne[0], M = (int) t0->ne[1], N = (int) t1->ne[1];
    const int ne12 = (int) t1->ne[2], ne13 = (int) t1->ne[3];
    const int r2 = (int) (t1->ne[2] / t0->ne[2]), r3 = (int) (t1->ne[3] / t0->ne[3]);
    const int gx = (M + LG_BM - 1) / LG_BM, gy = (N + LG_BN - 1) / LG_BN, nb = ne12 * ne13;
    const int64_t d1 = td->nb[1] / sizeof(float), d2 = td->nb[2] / sizeof(float), d3 = td->nb[3] / sizeof(float);

    // Split K when the output tiles alone cannot fill the GPU: the router gate is M 128 x N 256
    // per prefill chunk = 8 tiles, each walking K 2816 alone. Aim for ~4 blocks per SM, keep
    // every slice >= 8 K-steps, at most 32 slices.
    const int nsm  = ggml_cuda_info().devices[ctx.device].nsm;
    const int64_t tiles = (int64_t) gx * gy * nb;
    int ksplit = 1;
    if (tiles < 4LL * nsm) {
        ksplit = (int) std::min<int64_t>((4LL * nsm + tiles - 1) / tiles, 32);
        ksplit = std::max(1, std::min(ksplit, K / (8 * LG_BK)));
    }
    if ((int64_t) nb * ksplit > 65535) {
        ksplit = 1;
    }
    const int kchunk = ((K + ksplit - 1) / ksplit + LG_BK - 1) / LG_BK * LG_BK;
    ksplit = (K + kchunk - 1) / kchunk;

    ggml_cuda_pool_alloc<float> part(ctx.pool());
    if (ksplit > 1) {
        part.alloc((size_t) nb * ksplit * M * N);
    }
    const dim3 grid(gx, gy, nb * ksplit);
    llapis_gemm_kernel<T0, T1><<<grid, LG_NT, 0, stream>>>(src0, src1, dst, K, M, N,
        t0->nb[0] / ts0, t0->nb[1] / ts0, t0->nb[2] / ts0, t0->nb[3] / ts0,
        t1->nb[0] / ts1, t1->nb[1] / ts1, t1->nb[2] / ts1, t1->nb[3] / ts1,
        d1, d2, d3, ne12, r2, r3, ksplit, kchunk, part.get());
    CUDA_CHECK(cudaGetLastError());
    if (ksplit > 1) {
        const int64_t mn = (int64_t) M * N;
        const dim3 rgrid((unsigned) ((mn + 255) / 256), nb);
        llapis_gemm_reduce<<<rgrid, 256, 0, stream>>>(part.get(), dst, M, N, ne12, ksplit, d1, d2, d3);
        CUDA_CHECK(cudaGetLastError());
    }
}

template <typename T0>
static bool llapis_gemm_src1(ggml_backend_cuda_context & ctx, const T0 * src0, const ggml_tensor * t0, const ggml_tensor * src1, ggml_tensor * dst) {
    float * d = (float *) dst->data;
    switch (src1->type) {
        case GGML_TYPE_F32:  llapis_gemm_launch(ctx, src0, (const float *)       src1->data, d, t0, src1, dst); return true;
        case GGML_TYPE_F16:  llapis_gemm_launch(ctx, src0, (const half *)        src1->data, d, t0, src1, dst); return true;
        case GGML_TYPE_BF16: llapis_gemm_launch(ctx, src0, (const nv_bfloat16 *) src1->data, d, t0, src1, dst); return true;
        default: return false;
    }
}

static bool lg_strides_ok(const ggml_tensor * t, size_t ts) {
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (t->nb[i] % ts != 0) {
            return false;
        }
    }
    return true;
}

bool llapis_gemm_enabled() {
    static const bool on = [] {
        const char * e = getenv("LLAPIS_CUBLAS");
        return !(e && e[0] && strcmp(e, "0") != 0);
    }();
    return on;
}

// LLAPIS_GEMM_TRACE=<file>: append one line per call (op name, types, shape) to <file>. A file, not
// stderr or the log callback: stderr is lost under the launcher and the callback is not proven safe
// at this frequency (findings 296d). For measurements only.
static void lg_trace(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst) {
    static FILE * f = [] { const char * p = getenv("LLAPIS_GEMM_TRACE"); return p && p[0] ? fopen(p, "a") : (FILE *) nullptr; }();
    if (!f) {
        return;
    }
    fprintf(f, "%s\t%s x %s\tK %lld M %lld N %lld b %lldx%lld\n", dst->name, ggml_type_name(src0->type), ggml_type_name(src1->type),
            (long long) src0->ne[0], (long long) src0->ne[1], (long long) src1->ne[1], (long long) src1->ne[2], (long long) src1->ne[3]);
    fflush(f);
}

bool llapis_gemm_mul_mat(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    if (!llapis_gemm_enabled()) {
        return false;
    }
    lg_trace(src0, src1, dst);
    if (dst->type != GGML_TYPE_F32 || dst->nb[0] != sizeof(float) || !lg_strides_ok(dst, sizeof(float))) {
        return false;
    }
    if (src1->type != GGML_TYPE_F32 && src1->type != GGML_TYPE_F16 && src1->type != GGML_TYPE_BF16) {
        return false;
    }
    if (!lg_strides_ok(src1, ggml_type_size(src1->type))) {
        return false;
    }
    if (src1->ne[2] % src0->ne[2] != 0 || src1->ne[3] % src0->ne[3] != 0) {
        return false;
    }
    if (src0->ne[0] > INT_MAX || src0->ne[1] > INT_MAX || src1->ne[1] > INT_MAX ||
        (src1->ne[1] + LG_BN - 1) / LG_BN > 65535 || src1->ne[2] * src1->ne[3] > 65535) {
        return false;
    }

    cudaStream_t stream = ctx.stream();

    switch (src0->type) {
        case GGML_TYPE_F32:
            return lg_strides_ok(src0, 4) && llapis_gemm_src1(ctx, (const float *)       src0->data, src0, src1, dst);
        case GGML_TYPE_F16:
            return lg_strides_ok(src0, 2) && llapis_gemm_src1(ctx, (const half *)        src0->data, src0, src1, dst);
        case GGML_TYPE_BF16:
            return lg_strides_ok(src0, 2) && llapis_gemm_src1(ctx, (const nv_bfloat16 *) src0->data, src0, src1, dst);
        default:
            break;
    }

    // quantized src0: dequantize to a contiguous F32 copy first (as the cuBLAS path did to F16)
    if (!ggml_is_contiguous(src0)) {
        return false;
    }
    const to_fp32_cuda_t to_fp32 = ggml_get_to_fp32_cuda(src0->type);
    if (to_fp32 == nullptr) {
        return false;
    }
    if (src1->type != GGML_TYPE_F32 && src1->type != GGML_TYPE_F16 && src1->type != GGML_TYPE_BF16) {
        return false;
    }
    ggml_cuda_pool_alloc<float> src0_f32(ctx.pool(), ggml_nelements(src0));
    to_fp32(src0->data, src0_f32.get(), ggml_nelements(src0), stream);
    CUDA_CHECK(cudaGetLastError());

    ggml_tensor t0 = *src0;   // the same shape, contiguous F32 strides
    t0.type  = GGML_TYPE_F32;
    t0.nb[0] = sizeof(float);
    t0.nb[1] = t0.nb[0] * t0.ne[0];
    t0.nb[2] = t0.nb[1] * t0.ne[1];
    t0.nb[3] = t0.nb[2] * t0.ne[2];
    return llapis_gemm_src1(ctx, (const float *) src0_f32.get(), &t0, src1, dst);
}
