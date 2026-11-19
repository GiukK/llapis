// test_llapis_gemm -- the GEMM that replaces cuBLAS in ggml_cuda_mul_mat (ggml-cuda/llapis-gemm.cuh),
// against a double-precision CPU reference, on the shapes that reach it in this runtime:
//   - the MoE router gate in prefill: f32 [2816 x 128] x f32 [2816 x N], N > 16
//   - a vision-encoder-like f16 weight against many patch columns
//   - a batched, broadcast product (ne2 > 1, src0 shared by 3 heads), odd sizes on every edge
//   - bf16 weights, and a quantized (Q8_0) weight (mmq takes it; checked for the dispatch only)
// Every case goes through ggml_mul_mat on the CUDA backend, so it also proves the dispatch.
// Needs a CUDA device; without one it prints SKIP and passes (the CI runner has no GPU).
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif

static int g_fail = 0;

static float round_to(ggml_type t, float x) {
    switch (t) {
        case GGML_TYPE_F16:  return ggml_fp16_to_fp32(ggml_fp32_to_fp16(x));
        case GGML_TYPE_BF16: return ggml_bf16_to_fp32(ggml_fp32_to_bf16(x));
        default:             return x;
    }
}

// a: K x M x A2, b: K x N x B2 (B2 % A2 == 0). Fails when the max scaled error reaches tol.
static void run_case(ggml_backend_t be, const char * name, ggml_type ta, int K, int M, int N, int A2, int B2, double tol = 1e-5) {
    std::mt19937 rng(1234 + K + M + N);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    std::vector<float> a((size_t) K * M * A2), b((size_t) K * N * B2);
    for (auto & x : a) x = u(rng);
    for (auto & x : b) x = u(rng);

    ggml_init_params ip = { 8 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * ta_t = ggml_new_tensor_3d(ctx, ta, K, M, A2);
    ggml_tensor * tb_t = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, N, B2);
    ggml_tensor * tc_t = ggml_mul_mat(ctx, ta_t, tb_t);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, tc_t);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);

    // weights in their storage type; the reference sees exactly what the GPU sees
    std::vector<float> a_ref(a.size());
    if (ggml_is_quantized(ta)) {
        std::vector<uint8_t> q(ggml_row_size(ta, K) * (size_t) M * A2);
        ggml_quantize_chunk(ta, a.data(), q.data(), 0, (int64_t) M * A2, K, nullptr);
        ggml_backend_tensor_set(ta_t, q.data(), 0, q.size());
        const ggml_type_traits * tr = ggml_get_type_traits(ta);
        tr->to_float(q.data(), a_ref.data(), (int64_t) a.size());
    } else if (ta == GGML_TYPE_F32) {
        ggml_backend_tensor_set(ta_t, a.data(), 0, a.size() * 4);
        a_ref = a;
    } else {
        std::vector<uint16_t> h(a.size());
        for (size_t i = 0; i < a.size(); ++i) {
            if (ta == GGML_TYPE_F16) { ggml_fp16_t v = ggml_fp32_to_fp16(a[i]); memcpy(&h[i], &v, 2); }
            else                     { ggml_bf16_t v = ggml_fp32_to_bf16(a[i]); memcpy(&h[i], &v, 2); }
            a_ref[i] = round_to(ta, a[i]);
        }
        ggml_backend_tensor_set(ta_t, h.data(), 0, h.size() * 2);
    }
    ggml_backend_tensor_set(tb_t, b.data(), 0, b.size() * 4);

    if (ggml_backend_graph_compute(be, gf) != GGML_STATUS_SUCCESS) {
        printf("[FAIL] %s: compute failed\n", name); ++g_fail;
    } else {
        std::vector<float> c((size_t) M * N * B2);
        ggml_backend_tensor_get(tc_t, c.data(), 0, c.size() * 4);
        const int r2 = B2 / A2;
        double worst = 0.0;
        for (int i2 = 0; i2 < B2; ++i2)
        for (int n = 0; n < N; ++n)
        for (int m = 0; m < M; ++m) {
            const float * pa = &a_ref[((size_t) (i2 / r2) * M + m) * K];
            const float * pb = &b[((size_t) i2 * N + n) * K];
            double ref = 0.0, mag = 0.0;
            for (int k = 0; k < K; ++k) { ref += (double) pa[k] * pb[k]; mag += fabs((double) pa[k] * pb[k]); }
            const double err = fabs(c[((size_t) i2 * N + n) * M + m] - ref) / (mag + 1e-6);
            if (!(err <= worst)) worst = err;   // also catches NaN
        }
        // FP32 accumulation over K terms: ~K * 2^-24 relative to the magnitude sum, with margin.
        // TF32 (cuBLAS's mode here) would be ~2^-11 and fail the gate cases -- this checks FP32 too.
        const bool ok = worst < tol;
        printf("[%s] %-34s %-5s K %5d M %5d N %5d x%d/%d  max scaled err %.2e\n",
               ok ? " OK " : "FAIL", name, ggml_type_name(ta), K, M, N, A2, B2, worst);
        if (!ok) ++g_fail;
    }
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
}

// `test-llapis-gemm bench`: the router gate of one prefill chunk (K 2816, M 128, N 256), 30 layers per
// graph like the model, timed over many graphs. Run it with and without LLAPIS_CUBLAS=1.
static void bench(ggml_backend_t be) {
    const int K = 2816, M = 128, N = 256, L = 30, reps = 50;
    ggml_init_params ip = { (3 * L + 4) * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    std::vector<ggml_tensor *> ins;
    for (int l = 0; l < L; ++l) {
        ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
        ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N);
        ins.push_back(w); ins.push_back(x);
        ggml_build_forward_expand(gf, ggml_mul_mat(ctx, w, x));
    }
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    std::vector<float> data((size_t) K * N, 0.25f);
    for (ggml_tensor * t : ins) ggml_backend_tensor_set(t, data.data(), 0, ggml_nbytes(t));
    ggml_backend_graph_compute(be, gf);   // warm-up (and, under LLAPIS_CUBLAS, the cuBLAS load)
    const int64_t t0 = ggml_time_us();
    for (int r = 0; r < reps; ++r) ggml_backend_graph_compute(be, gf);
    const double us = (double) (ggml_time_us() - t0) / reps / L;
    // the replays went through a captured CUDA graph (split-K partials from the pool): still right?
    std::vector<float> out((size_t) M * N);
    ggml_backend_tensor_get(ggml_graph_node(gf, -1), out.data(), 0, out.size() * 4);
    double worst = 0.0;
    for (float v : out) worst = std::max(worst, fabs(v - K * 0.0625) / (K * 0.0625));
    printf("[%s] after %d graph replays the output is %s (max rel err %.1e)\n", worst < 1e-6 ? " OK " : "FAIL",
           reps, worst < 1e-6 ? "exact" : "WRONG", worst);
    printf("router gate K %d M %d N %d: %.1f us per call (%.0f GFLOP/s), %s\n", K, M, N, us,
           2.0 * K * M * N / us / 1e3, getenv("LLAPIS_CUBLAS") ? "cuBLAS" : "LLapis GEMM");
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
}

int main(int argc, char ** argv) {
    if (ggml_backend_cuda_get_device_count() < 1) {
        printf("SKIP: no CUDA device\n");
        return 0;
    }
    ggml_backend_t be = ggml_backend_cuda_init(0);
    if (!be) { printf("SKIP: CUDA backend did not start\n"); return 0; }
    if (argc > 1 && strcmp(argv[1], "bench") == 0) {
        ggml_time_init();
        bench(be);
        ggml_backend_free(be);
        return 0;
    }

    run_case(be, "router gate, prefill",            GGML_TYPE_F32,  2816,  128,  300, 1, 1);
    run_case(be, "router gate, 17 columns",         GGML_TYPE_F32,  2816,  128,   17, 1, 1);
    run_case(be, "router gate, odd tile edges",     GGML_TYPE_F32,  2816,  128, 1000, 1, 1);
    run_case(be, "vision-like f16 weight",          GGML_TYPE_F16,  1152,  577,  129, 1, 1);
    run_case(be, "bf16 weight",                     GGML_TYPE_BF16,  320,   96,   70, 1, 1);
    run_case(be, "batched broadcast, odd sizes",    GGML_TYPE_F32,    72,   50,   33, 2, 6);
    run_case(be, "batched f16, K not a tile mult.", GGML_TYPE_F16,    83,   65,   65, 4, 4);
    // mmq takes this one and quantizes the activations to q8_1: a looser bound, it checks the dispatch
    run_case(be, "quantized q8_0 weight",           GGML_TYPE_Q8_0,  512,  100,   40, 1, 1, 2e-2);

#ifdef _WIN32
    // The point of the GEMM is that cuBLAS is never mapped (it is delay-loaded): ask the process.
    // LLAPIS_CUBLAS=1 is the control -- then it must be loaded, or this check proves nothing.
    {
        char mod[32];
        snprintf(mod, sizeof(mod), "cublas64_%d.dll", CUDART_MAJOR_FOR_TEST);
        const bool loaded = GetModuleHandleA(mod) != nullptr;
        const char * e = getenv("LLAPIS_CUBLAS");
        const bool want = e && e[0] && strcmp(e, "0") != 0;
        printf("[%s] %s %s in the process (LLAPIS_CUBLAS %s)\n", loaded == want ? " OK " : "FAIL",
               mod, loaded ? "IS loaded" : "is not loaded", want ? "set" : "unset");
        if (loaded != want) ++g_fail;
    }
#endif
    ggml_backend_free(be);
    if (g_fail) { printf("%d case(s) failed\n", g_fail); return 1; }
    printf("all cases passed\n");
    return 0;
}
