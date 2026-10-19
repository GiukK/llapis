// moert/cuda — the MoE CUDA island. Bodies relocated VERBATIM from the
// pre-M2 ggml-cuda/moe-dual.cu custom op:
// M2 changes the calling convention (device-pointer ABI + host vtable, moert/cuda.h), not the
// kernels or their launch order — the exit gate holds the output bit-identical.
//
// Computes each routed (token,expert) pair once at its resident tier: hot pairs against the
// K-slot pool, cold pairs against the full pool. Slice 1b correctness path + Slice 2
// device-counted guard + Slice 3 shared-activation gate_up, single-warp partition
// (Pmax = n_used*n_tokens <= 32; decode: top_k=8, n_tokens=1 -> Pmax=8), mirroring the
// ballot+popc compaction with deterministic ascending-p ordering.
//
// ggml is reached ONLY through the host vtable (tuned quantized mul_mat_id + the backend's
// LIFO VRAM scratch pool); this file includes no ggml headers.

#include "moert/cuda.h"
#include "moert/policy.h"   // MOERT_SLOT_READY — single source for the residency predicate

#include <cuda_runtime.h>
#include <immintrin.h>

#include <chrono>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>

// Local check macro (the island cannot use ggml's CUDA_CHECK): print + abort, matching the
// fail-fast behavior the op had inside ggml.
#define MOERT_CUDA_CHECK(err)                                                              \
    do {                                                                                   \
        const cudaError_t err_ = (err);                                                    \
        if (err_ != cudaSuccess) {                                                         \
            fprintf(stderr, "moert/cuda: CUDA error %s at %s:%d: %s\n",                    \
                    cudaGetErrorName(err_), __FILE__, __LINE__, cudaGetErrorString(err_)); \
            abort();                                                                       \
        }                                                                                  \
    } while (0)

namespace {

moert_cuda_config g_cfg = [] { moert_cuda_config c; moert_cuda_config_defaults(&c); return c; }();

// Addressing stride of a cold pool of `eb`-byte experts (contiguous packing: == eb).
static inline size_t cold_stride(size_t eb) {
    return eb;
}

// ---------------------------------------------------------------------------------
// Kernels (verbatim from the pre-M2 op)
// ---------------------------------------------------------------------------------

// Owning-lane activation-column copy: one warp lane moves a full D_in-float column. float4 (16 B
// transactions, ~4x fewer than the scalar loop) when D_in is a multiple of 4 (down: n_ff=704) and
// the offsets are 4-float aligned by construction (idx*D_in / src*D_in); scalar fallback otherwise.
// Pure byte copy => bit-identical.
__device__ __forceinline__ void moe_dual_gather_col(float * __restrict__ dst, const float * __restrict__ src, int D_in) {
    if ((D_in & 3) == 0) {
        float4 * __restrict__ d4 = reinterpret_cast<float4 *>(dst);
        const float4 * __restrict__ s4 = reinterpret_cast<const float4 *>(src);
        const int n4 = D_in >> 2;
        for (int k = 0; k < n4; ++k) d4[k] = s4[k];
    } else {
        for (int k = 0; k < D_in; ++k) dst[k] = src[k];
    }
}

// Partition routed pairs by residency, optionally gathering the activation columns (x_hot/x_cold may
// be null when the caller shares a single activation, e.g. gate_up). Outputs are compacted and the
// unused [n_hot,Pmax)/[n_cold,Pmax) tails are left at the caller's zero-pad (ids) / undefined
// (x - those columns are pre-zeroed by the launcher when gathered). counts[0]=n_hot, counts[1]=n_cold.
__global__ void moe_dual_partition_kernel(
        const int32_t * __restrict__ ids, int n_used, int ids_s1, int top_k, int kind,
        const float * __restrict__ cur, int D_in,
        const int32_t * __restrict__ exp_slot, const int32_t * __restrict__ slot_exp,
        const int32_t * __restrict__ slot_state, int n_expert, int K,
        int32_t * __restrict__ ids_hot,  int32_t * __restrict__ ids_cold,
        float   * __restrict__ x_hot,    float   * __restrict__ x_cold,
        int32_t * __restrict__ hot_dst,  int32_t * __restrict__ cold_dst,
        int32_t * __restrict__ counts, int n_pairs,
        int32_t * __restrict__ rc_counts, int32_t * __restrict__ rc_hits, int split_gather) {
    const int  lane  = threadIdx.x;               // 0..31, one warp
    const bool valid = lane < n_pairs;
    const int  p     = lane;
    int32_t e = -1, slot = -1, t = 0;
    bool hot = false;
    if (valid) {
        t = lane / n_used;                        // token index (n_used routes per token)
        e = ids[t * ids_s1 + (p - t * n_used)];   // ids is a STRIDED VIEW (argsort_top_k keeps the
                                                  // argsort's row stride): token t's row starts at
                                                  // t*ids_s1, NOT t*n_used. At n_tokens==1 this
                                                  // reduces to ids[p] (the validated n=1 path).
        if (e >= 0 && e < n_expert) {
            const int s = exp_slot[e];
            if (s >= 0 && s < K && slot_state[s] == MOERT_SLOT_READY && slot_exp[s] == e) { hot = true; slot = s; }
        }
    }
    const unsigned full     = 0xffffffffu;
    const unsigned hot_mask  = __ballot_sync(full, valid &&  hot);
    const unsigned cold_mask = __ballot_sync(full, valid && !hot);
    const unsigned below     = (lane == 0) ? 0u : (full >> (32 - lane));
    const int      src_col   = (kind == 0) ? t : p;   // gate_up: per-token (cur ne1==1) / down: per-pair

    if (valid && hot) {
        const int idx = __popc(hot_mask & below);
        ids_hot[idx]  = slot;  hot_dst[idx] = p;
        if (x_hot && !split_gather) moe_dual_gather_col(x_hot + (size_t) idx * D_in, cur + (size_t) src_col * D_in, D_in);
    } else if (valid) {
        const int idx = __popc(cold_mask & below);
        ids_cold[idx] = e;     cold_dst[idx] = p;
        if (x_cold && !split_gather) moe_dual_gather_col(x_cold + (size_t) idx * D_in, cur + (size_t) src_col * D_in, D_in);
    }
    if (lane == 0) { counts[0] = __popc(hot_mask); counts[1] = __popc(cold_mask); }

    // Device-side routing counters (native observation): accumulate per-expert routed-pair counts and
    // the hot/cold split for this layer while the ids are already in registers — the host then fetches
    // them only at refresh points instead of reading ffn_moe_topk through the eval callback (which
    // forces a per-layer backend sync). Passed only for kind 0 (gate_up and down share ids/residency).
    if (rc_counts && valid && e >= 0 && e < n_expert) atomicAdd(&rc_counts[e], 1);
    if (rc_hits && lane == 0) { atomicAdd(&rc_hits[0], __popc(hot_mask)); atomicAdd(&rc_hits[1], __popc(cold_mask)); }

}

// The activation gather of the partition, spread over the GPU (one block per compact column and side)
// instead of one warp lane copying a whole D_in column: the lone-warp copy was 39 us of every fire,
// on the critical path, in front of the cold-list D2H (findings 202). Same bytes, same places.
__global__ void moe_dual_gather_kernel(
        const float * __restrict__ cur, int D_in, int kind, int n_used, const int32_t * __restrict__ counts,
        const int32_t * __restrict__ hot_dst, const int32_t * __restrict__ cold_dst,
        float * __restrict__ x_hot, float * __restrict__ x_cold) {
    const int i = blockIdx.x, side = blockIdx.y;
    const int n = counts[side];
    if (i >= n) return;
    const int p = side == 0 ? hot_dst[i] : cold_dst[i];
    const int src_col = (kind == 0) ? p / n_used : p;
    float * dst = (side == 0 ? x_hot : x_cold) + (size_t) i * D_in;
    const float * src = cur + (size_t) src_col * D_in;
    if ((D_in & 3) == 0) {
        float4 * d4 = reinterpret_cast<float4 *>(dst);
        const float4 * s4 = reinterpret_cast<const float4 *>(src);
        for (int k = threadIdx.x; k < (D_in >> 2); k += blockDim.x) d4[k] = s4[k];
    } else {
        for (int k = threadIdx.x; k < D_in; k += blockDim.x) dst[k] = src[k];
    }
}

// Scatter both tier outputs back to routed-pair order, applying the down .scale exactly once.
// grid = Pmax; block i handles compact index i on each side (n_hot/n_cold <= Pmax). HOT/COLD
// destinations are disjoint by construction.
__global__ void moe_dual_scatter_kernel(
        const float * __restrict__ y_hot, const float * __restrict__ y_cold, int D_out, int kind,
        const int32_t * __restrict__ counts,
        const int32_t * __restrict__ ids_hot,  const int32_t * __restrict__ hot_dst,
        const int32_t * __restrict__ ids_cold, const int32_t * __restrict__ cold_dst,
        const float * __restrict__ s_hot, const float * __restrict__ s_cold, float * __restrict__ y_full) {
    const int i = blockIdx.x;
    const int n_hot = counts[0], n_cold = counts[1];
    if (i < n_hot) {
        const int   p  = hot_dst[i];
        const float sc = (kind == 1 && s_hot) ? s_hot[ids_hot[i]] : 1.0f;
        for (int o = threadIdx.x; o < D_out; o += blockDim.x) y_full[(size_t) p * D_out + o] = y_hot[(size_t) i * D_out + o] * sc;
    }
    if (i < n_cold) {
        const int   p  = cold_dst[i];
        const float sc = (kind == 1 && s_cold) ? s_cold[ids_cold[i]] : 1.0f;
        for (int o = threadIdx.x; o < D_out; o += blockDim.x) y_full[(size_t) p * D_out + o] = y_cold[(size_t) i * D_out + o] * sc;
    }
}

// Fill p[0..n-1] = 0,1,...,n-1 (ring-slot ids for the staged cold matmul). n <= 32 (Pmax).
__global__ void moe_dual_iota_kernel(int32_t * __restrict__ p, int n) {
    const int i = threadIdx.x;
    if (i < n) p[i] = i;
}

// ---------------------------------------------------------------------------------
// Diagnostics state (config-driven since M2; the printed tags keep the historical
// MOE_DUAL_* names — the host shim still parses those env vars into the config)
// ---------------------------------------------------------------------------------

static inline long long sync_prof_now() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

// --- cold-seam host-block telemetry (2026-08-09) -----------------------------------------
// `legacy_ns` is named in three comments in this file and no longer exists: the counter went
// with the phase profiler in F4.5, and with it the ability to answer "how much of a cold turn
// is the host standing still on the cold-list D2H?". One site since the 2026-10-03 pruning (the
// planner / cross-kind / L2 sites went with their paths). Printed at process exit next to
// [MOE_DUAL_COLDCPU].
struct ColdSeam {
    struct Site { long long ns = 0, n = 0; const char * name; };
    Site sites[1] = { {0,0,"fixed-seam"} };
    ~ColdSeam() {
        long long tn = 0, tns = 0;
        for (const Site & s : sites) { tn += s.n; tns += s.ns; }
        if (!tn) return;
        fprintf(stderr, "[MOE_DUAL_SEAM] host-blocked on the cold-list D2H: total %.1f ms over"
                        " %lld syncs (mean %.1f us)\n", tns / 1e6, tn, tns / 1e3 / tn);
        for (const Site & s : sites)
            if (s.n) fprintf(stderr, "[MOE_DUAL_SEAM]   %-11s %8lld syncs  %9.1f ms  mean %6.1f us\n",
                             s.name, s.n, s.ns / 1e6, s.ns / 1e3 / s.n);
    }
};
static ColdSeam g_seam;
static inline void seam_sync(cudaStream_t s, int site) {
    const long long t0 = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    MOERT_CUDA_CHECK(cudaStreamSynchronize(s));
    g_seam.sites[site].ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count() - t0;
    ++g_seam.sites[site].n;
}

// Pinned header scratch for the per-op cold-list D2H (the RAM-cold path).
// A pageable-destination D2H cudaMemcpyAsync is synchronous w.r.t. the host (the driver blocks
// INSIDE the copy call to stage through its own pinned bounce), so bracketing only the trailing
// cudaStreamSynchronize (legacy_ns) charges the wrong call and under-counts host-blocked time.
// Pinning the destination makes the D2H truly async, moving the block into the timed sync ->
// honest legacy_ns. This matters most under -DevCounts, where the eval callback no longer syncs
// the backend per layer, so the stream is NOT already drained when the op's own sync runs.
// Layout: [0..1]=counts, [2..2+Pmax)=cold ids (Pmax<=32). One
// global instance is safe: the host issues these fires sequentially (fill -> sync -> read before
// the next fire) and there is one site. Graceful pageable fallback
// (== prior behavior) if the one-time pinned alloc fails.
struct LegacyHdrPin {
    int32_t   fallback[2 + 32 + 32];   // counts, cold ids, cold pair indices (cold_dst)
    int32_t * buf   = nullptr;
    bool      tried = false;
    int32_t * get() {
        if (!tried) {
            tried = true;
            if (cudaHostAlloc(&buf, (size_t) (2 + 32 + 32) * sizeof(int32_t), cudaHostAllocDefault) != cudaSuccess) buf = nullptr;
        }
        return buf ? buf : fallback;
    }
};
LegacyHdrPin g_hdrpin;

// CPU cold sidecar state (MOE_DUAL_COLD_CPU, the product's cold executor).
// Pinned host buffers for the small activation D2H and per-pair output H2D, plus telemetry.
// The compute itself lives in the HOST process (src/moe_sidecar.cpp) behind cfg.coldcpu_fn —
// the island stays ggml-free.
struct ColdCpu {
    static const int PMAX = 32, DMAX = 4096;   // >= any gemma4 per-kind d_in/d_out
    float * x_pin = nullptr, * y_pin = nullptr;
    bool tried = false;
    long long calls = 0, pairs = 0, over_max = 0, bytes_avoided = 0, cpu_ns = 0;
    // Down .scale is static per layer (cold-tier metadata, fixed at load): cache the device copy
    // once per layer instead of re-uploading it into scratch every fire.
    float * s_dev_cache[128] = {};
    void ensure_pinned(int Pmax, int d_in, int d_out) {
        (void) Pmax; (void) d_in; (void) d_out;
        if (tried) return;
        tried = true;
        if (cudaHostAlloc(&x_pin, (size_t) PMAX * DMAX * sizeof(float), 0) != cudaSuccess) { x_pin = nullptr; return; }
        if (cudaHostAlloc(&y_pin, (size_t) PMAX * DMAX * sizeof(float), 0) != cudaSuccess) {
            cudaFreeHost(x_pin); x_pin = nullptr; y_pin = nullptr;
        }
    }
    ~ColdCpu() {
        if (calls || over_max)
            fprintf(stderr, "[MOE_DUAL_COLDCPU] sidecar fires=%lld pairs=%lld cpu_ms=%.1f"
                    " | over-max fallback fires=%lld | PCIe bytes avoided=%.1f MiB\n",
                    calls, pairs, cpu_ns / 1e6, over_max, bytes_avoided / 1048576.0);
    }
};
ColdCpu g_coldcpu;

// Routed-only cold copy-stream (MOE_DUAL_COPYSTREAM). The per-op RAM-cold path issues
// the selected cold experts' H2D copies on the COMPUTE stream, behind a full cudaStreamSynchronize
// (stream) that also drains the hot matmul — so even pinned, the copies never overlap compute and the
// host ping-pongs with the GPU ~60x/token. This routes the SAME device-partitioned cold list  onto a dedicated copy stream, gated on the
// partition kernel alone (ev_part, recorded BEFORE the hot matmul is enqueued), so: (1) the host sync
// waits only for partition + the tiny cold-list D2H, not the hot matmul; (2) the pinned H2D copies run
// concurrently with the hot matmul on the compute stream; (3) one cudaStreamWaitEvent(ev_done) fences
// the cold matmul behind the copies. Bytes staged are byte-identical to legacy => logits unchanged.
struct ColdStream {
    bool         inited  = false;
    cudaStream_t st      = nullptr;
    cudaEvent_t  ev_part = nullptr;   // partition-complete fence (recorded on the compute stream)
    cudaEvent_t  ev_done = nullptr;   // cold-outputs/copies-complete fence (joins cs back into the compute stream)
    long long    fires   = 0;         // per-op cold fires on the copy stream
    void lazy_init() {
        if (inited) return;
        MOERT_CUDA_CHECK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
        MOERT_CUDA_CHECK(cudaEventCreateWithFlags(&ev_part, cudaEventDisableTiming));
        MOERT_CUDA_CHECK(cudaEventCreateWithFlags(&ev_done, cudaEventDisableTiming));
        inited = true;
    }
    ~ColdStream() {
        if (fires) fprintf(stderr, "[MOE_DUAL_COPYSTREAM] routed-only cold copy stream: %lld ops staged\n", fires);
    }
};
ColdStream g_coldstream;

// cold_ffn mode (findings 206): the event the host job waits on before reading the pinned D2H data.
struct ColdFfn {
    cudaEvent_t ev = nullptr;
    long long submits = 0;
    void lazy_init() { if (!ev) MOERT_CUDA_CHECK(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming)); }
    ~ColdFfn() { if (submits) fprintf(stderr, "[MOE_DUAL_COLDFFN] asynchronous cold FFN jobs: %lld\n", submits); }
};
ColdFfn g_coldffn;
static void coldffn_wait_ready(void * ev) { MOERT_CUDA_CHECK(cudaEventSynchronize((cudaEvent_t) ev)); }

// Device-side routing counters (native observation). Layout: one flat int32 buffer,
// [n_layers][n_expert] routed-pair counts followed by [n_layers][2] hot/cold pair counts.
// The partition kernel (kind 0 only) atomically accumulates; the host fetches + resets at
// refresh points (every ~64 visible tokens / warmup end / turn end) — replacing the per-layer
// eval-callback read of ffn_moe_topk that forces a backend sync at every layer of every token.
struct RouteCounts {
    int32_t * dev = nullptr;
    int nl = 0, ne = 0;
    bool on = false;
    size_t n_ints() const { return (size_t) nl * ne + (size_t) nl * 2; }
};
RouteCounts g_rc;

}  // namespace


// One cudaMemcpyBatchAsync for n host->device copies from pinned staging, read in stream order. Falls back to
// per-copy cudaMemcpyAsync if the batch call is refused.
static void batch_h2d(void * const * dsts, const void * const * srcs, const size_t * sizes, size_t n, cudaStream_t st) {
    if (!n) return;
    cudaMemcpyAttributes at = {};
    at.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
    size_t idx = 0;
    if (cudaMemcpyBatchAsync(dsts, srcs, sizes, n, &at, &idx, 1, st) == cudaSuccess) return;
    (void) cudaGetLastError();
    for (size_t i = 0; i < n; ++i) MOERT_CUDA_CHECK(cudaMemcpyAsync(dsts[i], srcs[i], sizes[i], cudaMemcpyHostToDevice, st));
}
namespace { cudaStream_t g_pub_st = nullptr; long long g_pub_calls = 0, g_pub_copies = 0; }
// Pageable host memory is NOT a valid batch-copy source on this platform (cudaDevAttrPageableMemoryAccess is false on
// WDDM: the batch treats the pointer as unreadable -> illegal address). Small pageable sources are staged here first.
// The ring is far larger than what can be in flight (a few KB per step), so a slot is never reused while queued.
namespace {
char * g_ring = nullptr; size_t g_ring_cap = 0, g_ring_off = 0;
const void * ring_stage(const void * src, size_t n) {
    if (!g_ring) { g_ring_cap = (size_t) 4 << 20; MOERT_CUDA_CHECK(cudaHostAlloc((void **) &g_ring, g_ring_cap, cudaHostAllocDefault)); }
    n = std::min(n, g_ring_cap);
    if (g_ring_off + n > g_ring_cap) g_ring_off = 0;
    char * p = g_ring + g_ring_off;
    std::memcpy(p, src, n);
    g_ring_off += (n + 63) & ~(size_t) 63;
    return p;
}
char * g_pub_pin = nullptr; size_t g_pub_pin_cap = 0;   // publication staging (synchronized after each call: reused)
}

// Batch transport (findings 216k-216l; contract in moert/cuda.h): the promotion worker issues its copies at once on a
// copy stream and marks each batch of promotions with an event; completions are polled without blocking (events on one
// stream complete in order).
namespace {
struct Director {
    bool on = false;
    cudaStream_t st = nullptr;
    std::mutex mu;
    struct Mark { cudaEvent_t ev; long long id; };
    std::deque<Mark> marks;
    long long next_mark = 0, done_upto = -1;
    std::vector<cudaEvent_t> ev_pool;
    std::atomic<long long> async_copies{0}, async_marks{0}, bytes{0};
    // SM transport (claude dump findings 229-231, opt-in MOE_DUAL_DIR_SM=1): the copies are a small kernel on a
    // lowest-priority stream -- they never queue in the copy engine's single FIFO, where the drafts' small transfers
    // waited behind them (findings 227). Its source is a mapped pinned RING, not the registered arena: registering the
    // arena mapped fills WDDM's shared-memory budget (7.0 of 7.1 GiB, findings 231) and the boot dies. The worker copies
    // each run into the ring (the RAM is idle in the window between verifies), the kernel reads it through its device
    // alias; a ring span is reused only after the event of the kernel that read it. A registered (pinned) source is read
    // in place. Each launch stamps its GPU start and end (globaltimer) into a mapped record, read for the exit line.
    bool sm = false;
    int sm_blocks = 1;
    unsigned long long * sm_rec = nullptr;   // [SM_REC][2] start, end ns (mapped host memory)
    static constexpr long long SM_REC = 1 << 16;
    char * ring = nullptr; char * ring_dev = nullptr; size_t ring_cap = 0, ring_head = 0;
    struct Span { size_t off, len; cudaEvent_t ev; };
    std::deque<Span> live;
    std::vector<cudaEvent_t> span_ev;
    long long ring_waits = 0, ring_copies = 0; double ring_wait_ms = 0.0, stage_ms = 0.0;
    std::atomic<long long> sm_copies{0}, sm_fallback{0};
    // Engine source (findings 240): a run in the registered (non-mapped) arena goes on the copy engine -- no CPU copy, the
    // RAM read once -- paced so at most `pace` runs sit in the engine's single FIFO (a draft's or the logits' small
    // transfer then waits at most ~pace x 0.1 ms behind them). Runs outside the registration still take the ring.
    bool engine = false;
    int pace = 2;
    std::deque<cudaEvent_t> eng_live;
    long long eng_copies = 0, eng_waits = 0; double eng_wait_ms = 0.0;
    cudaEvent_t span_event() {
        if (span_ev.empty()) { cudaEvent_t e; MOERT_CUDA_CHECK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming)); return e; }
        cudaEvent_t e = span_ev.back(); span_ev.pop_back(); return e;
    }
    // A free ring span of n bytes (under mu): retire finished spans, wait for the oldest while the span overlaps one.
    size_t ring_reserve(size_t n) {
        while (!live.empty() && cudaEventQuery(live.front().ev) == cudaSuccess) { span_ev.push_back(live.front().ev); live.pop_front(); }
        (void) cudaGetLastError();
        size_t off = ring_head;
        if (off + n > ring_cap) off = 0;
        for (;;) {
            bool hit = false;
            for (const Span & s : live) if (s.off < off + n && off < s.off + s.len) { hit = true; break; }
            if (!hit) break;
            const auto t0 = std::chrono::steady_clock::now();
            MOERT_CUDA_CHECK(cudaEventSynchronize(live.front().ev));
            ring_wait_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            ++ring_waits;
            span_ev.push_back(live.front().ev); live.pop_front();
        }
        ring_head = off + ((n + 255) & ~(size_t) 255);
        return off;
    }
    static bool is_pinned(const void * p) {
        cudaPointerAttributes at = {};
        const bool ok = cudaPointerGetAttributes(&at, p) == cudaSuccess && at.type == cudaMemoryTypeHost;
        (void) cudaGetLastError();
        return ok;
    }
};
Director g_dir;

// Parallel staging (claude dump findings 233): one thread copies ~10 GB/s, the RAM channel ~29 GB/s and it is idle in the
// window between verifies, so each run is split over the caller + (n-1) helpers. Helpers sleep on a condition variable
// (no spinning: the cold FFN's workers need the cores during the verify).
// Non-temporal copy (findings 233): streaming stores skip the read-for-ownership of the ring line, 2x the bytes on the RAM
// channel instead of 3x -- 0.093 instead of 0.143 ms per 1 MB piece. dst is 32-byte aligned (ring spans, 4 KiB chunks).
static void stage_copy(char * d, const char * s, size_t b) {
    size_t i = 0;
    if (((uintptr_t) d & 31) == 0) {
        for (; i + 64 <= b; i += 64) {
            const __m256i a = _mm256_loadu_si256((const __m256i *) (s + i)), c = _mm256_loadu_si256((const __m256i *) (s + i + 32));
            _mm256_stream_si256((__m256i *) (d + i), a);
            _mm256_stream_si256((__m256i *) (d + i + 32), c);
        }
        _mm_sfence();
    }
    if (i < b) std::memcpy(d + i, s + i, b - i);
}

struct StagePool {
    int n = 1;
    std::vector<std::thread> th;
    std::mutex mu;
    std::condition_variable cv_go, cv_done;
    long long gen = 0;
    int pending = 0;
    bool quit = false;
    char * dst = nullptr; const char * src = nullptr; size_t len = 0, chunk = 0;
    void start(int threads) {
        n = std::max(1, std::min(8, threads));
        for (int i = 1; i < n; ++i) th.emplace_back([this, i] { loop(i); });
    }
    void part(int i) {
        const size_t lo = (size_t) i * chunk, hi = std::min(len, lo + chunk);
        if (lo < hi) stage_copy(dst + lo, src + lo, hi - lo);
    }
    void loop(int i) {
        long long seen = 0;
        std::unique_lock<std::mutex> lk(mu);
        for (;;) {
            cv_go.wait(lk, [&] { return quit || gen != seen; });
            if (quit) return;
            seen = gen;
            lk.unlock();
            part(i);
            lk.lock();
            if (--pending == 0) cv_done.notify_one();
        }
    }
    void copy(char * d, const char * s, size_t bytes) {
        if (n <= 1 || bytes < ((size_t) 256 << 10)) { stage_copy(d, s, bytes); return; }
        {
            std::lock_guard<std::mutex> lk(mu);
            dst = d; src = s; len = bytes;
            chunk = ((bytes + (size_t) n - 1) / (size_t) n + 4095) & ~(size_t) 4095;
            pending = n - 1; ++gen;
        }
        cv_go.notify_all();
        part(0);
        std::unique_lock<std::mutex> lk(mu);
        cv_done.wait(lk, [&] { return pending == 0; });
    }
    ~StagePool() {
        { std::lock_guard<std::mutex> lk(mu); quit = true; }
        cv_go.notify_all();
        for (auto & t : th) t.join();
    }
};
StagePool g_stage;

// Early-decision gate (claude dump findings 234): a promotion decided DURING a verify may only write its slot after that
// verify has read the slab for the last time. The last MoE op of the verify records `ev` on the compute stream and bumps
// `gen`; a gated copy waits (host) until gen reaches its target, then makes the copy stream wait on `ev` (device), so it
// starts exactly when the GPU leaves the last MoE layer. The host also bumps gen when the decode returns (the verify is
// over by then): the gate can never stay shut.
// Per-layer mode (findings 235): every layer's last MoE op records its own event, so a copy into a slot of layer l may
// start as soon as the verify has left layer l -- in the RAM-free pause while the GPU runs the next layer's attention.
struct Gate {
    static constexpr int MAXL = 128;
    bool on = false, per_layer = false;
    int last_layer = -1;
    cudaEvent_t ev[MAXL] = {};
    bool recorded[MAXL] = {};
    long long c[MAXL] = {};
    long long armed = 0;
    std::mutex mu;
    std::condition_variable cv;
    long long waits = 0, timeouts = 0; double wait_ms = 0.0;
};
Gate g_gate;
}  // namespace

extern "C" int32_t moert_cuda_dir_init(int32_t sm_blocks, int32_t sm_ring_mib, int32_t sm_stage_threads) {
    Director & d = g_dir;
    if (d.on) return 1;
    d.sm = sm_blocks > 0;
    if (d.sm) {
        d.sm_blocks = std::min(36, (int) sm_blocks);
        int lo = 0, hi = 0;
        MOERT_CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&lo, &hi));
        MOERT_CUDA_CHECK(cudaStreamCreateWithPriority(&d.st, cudaStreamNonBlocking, lo));
        const size_t rb = (size_t) Director::SM_REC * 2 * sizeof(unsigned long long);
        MOERT_CUDA_CHECK(cudaHostAlloc((void **) &d.sm_rec, rb, cudaHostAllocMapped));
        std::memset(d.sm_rec, 0, rb);
        d.ring_cap = (size_t) std::max(8, (int) sm_ring_mib) << 20;
        MOERT_CUDA_CHECK(cudaHostAlloc((void **) &d.ring, d.ring_cap, cudaHostAllocMapped));
        MOERT_CUDA_CHECK(cudaHostGetDevicePointer((void **) &d.ring_dev, d.ring, 0));
        g_stage.start(sm_stage_threads);
    } else {
        MOERT_CUDA_CHECK(cudaStreamCreateWithFlags(&d.st, cudaStreamNonBlocking));
    }
    d.on = true;
    if (d.sm) fprintf(stderr, "moert/cuda: batch transport ON (SM copy: %d block(s) on a lowest-priority stream from a %zu MiB pinned ring filled by %d thread(s), the copy engine stays free), completion polled\n",
                      d.sm_blocks, d.ring_cap >> 20, g_stage.n);
    else      fprintf(stderr, "moert/cuda: batch transport ON: promotions issued at once on a copy stream, completion polled\n");
    return 1;
}

// SM transport: one launch copies one contiguous run; thread 0 stamps the start and the end (after the block's last
// store) into the mapped record. A misaligned run (the tiny scale pages) goes byte by byte.
static __device__ __forceinline__ unsigned long long dir_globaltimer() {
    unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t;
}
static __global__ void dir_sm_copy(const char * __restrict__ src, char * __restrict__ dst, unsigned long long n,
                                   unsigned long long * rec) {
    if (threadIdx.x == 0 && blockIdx.x == 0 && rec) rec[0] = dir_globaltimer();
    const size_t tid = blockIdx.x * (size_t) blockDim.x + threadIdx.x, step = (size_t) gridDim.x * blockDim.x;
    if ((((uintptr_t) src | (uintptr_t) dst | (uintptr_t) n) & 15) == 0) {
        const uint4 * s = (const uint4 *) src; uint4 * d = (uint4 *) dst;
        for (size_t i = tid; i < n / 16; i += step) d[i] = s[i];
    } else {
        for (size_t i = tid; i < n; i += step) dst[i] = src[i];
    }
    if (rec) { __syncthreads(); if (threadIdx.x == 0 && blockIdx.x == 0) rec[1] = dir_globaltimer(); }
}

extern "C" int32_t moert_cuda_dir_async(void * dst_dev, const void * src, uint64_t nbytes) {
    Director & d = g_dir;
    // registered sources; a tiny pageable one (a 4-byte scale page) is staged into pinned memory first
    if (!d.on || !dst_dev || !src || !nbytes) return 0;
    bool pinned = Director::is_pinned(src);
    if (pinned && d.sm) {
        // read in place only if the WHOLE run is in one mapped registration: a run that starts in the last page of a
        // registered neighbour (ranges are widened to pages) would otherwise be read past the mapping (findings 233)
        char * a0 = nullptr; char * a1 = nullptr;
        const bool ok0 = cudaHostGetDevicePointer((void **) &a0, const_cast<void *>(src), 0) == cudaSuccess;
        const bool ok1 = cudaHostGetDevicePointer((void **) &a1, (char *) const_cast<void *>(src) + nbytes - 1, 0) == cudaSuccess;
        (void) cudaGetLastError();
        if (!(ok0 && ok1 && a0 && a1 == a0 + nbytes - 1)) pinned = false;
    }
    if (d.sm && d.engine && pinned) {   // registered arena: the copy engine, paced
        std::lock_guard<std::mutex> lk(d.mu);
        while (!d.eng_live.empty() && cudaEventQuery(d.eng_live.front()) == cudaSuccess) { d.span_ev.push_back(d.eng_live.front()); d.eng_live.pop_front(); }
        (void) cudaGetLastError();
        while ((int) d.eng_live.size() >= d.pace) {
            const auto t0 = std::chrono::steady_clock::now();
            MOERT_CUDA_CHECK(cudaEventSynchronize(d.eng_live.front()));
            d.eng_wait_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            ++d.eng_waits;
            d.span_ev.push_back(d.eng_live.front()); d.eng_live.pop_front();
        }
        MOERT_CUDA_CHECK(cudaMemcpyAsync(dst_dev, src, (size_t) nbytes, cudaMemcpyHostToDevice, d.st));
        const cudaEvent_t ev = d.span_event();
        MOERT_CUDA_CHECK(cudaEventRecord(ev, d.st));
        d.eng_live.push_back(ev);
        ++d.eng_copies;
        d.async_copies.fetch_add(1, std::memory_order_relaxed);
        d.bytes.fetch_add((long long) nbytes, std::memory_order_relaxed);
        return 1;
    }
    if (d.sm && !pinned && nbytes <= d.ring_cap / 4) {   // SM transport from the ring
        std::lock_guard<std::mutex> lk(d.mu);
        const size_t off = d.ring_reserve((size_t) nbytes);
        const auto t0 = std::chrono::steady_clock::now();
        g_stage.copy(d.ring + off, (const char *) src, (size_t) nbytes);
        d.stage_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        const long long k = d.sm_copies.fetch_add(1, std::memory_order_relaxed);
        unsigned long long * rec = nullptr;
        if (k < Director::SM_REC && cudaHostGetDevicePointer((void **) &rec, d.sm_rec + 2 * k, 0) != cudaSuccess) { (void) cudaGetLastError(); rec = nullptr; }
        dir_sm_copy<<<d.sm_blocks, 512, 0, d.st>>>(d.ring_dev + off, (char *) dst_dev, (unsigned long long) nbytes, rec);
        MOERT_CUDA_CHECK(cudaGetLastError());
        const cudaEvent_t ev = d.span_event();
        MOERT_CUDA_CHECK(cudaEventRecord(ev, d.st));
        d.live.push_back({ off, (size_t) nbytes, ev });
        ++d.ring_copies;
        d.async_copies.fetch_add(1, std::memory_order_relaxed);
        d.bytes.fetch_add((long long) nbytes, std::memory_order_relaxed);
        return 1;
    }
    if (!pinned && nbytes > 4096) return 0;
    std::lock_guard<std::mutex> lk(d.mu);
    if (!pinned) src = ring_stage(src, (size_t) nbytes);
    // one call per copy: cudaMemcpyBatchAsync from the registered arena faulted in our test (findings 216l; the test
    // itself is in doubt, review 2 -- the device-alias probe is in the lab)
    void * alias = nullptr;
    if (d.sm && cudaHostGetDevicePointer(&alias, const_cast<void *>(src), 0) == cudaSuccess && alias) {
        const long long k = d.sm_copies.fetch_add(1, std::memory_order_relaxed);
        unsigned long long * rec = nullptr;
        if (k < Director::SM_REC && cudaHostGetDevicePointer((void **) &rec, d.sm_rec + 2 * k, 0) != cudaSuccess) { (void) cudaGetLastError(); rec = nullptr; }
        dir_sm_copy<<<d.sm_blocks, 512, 0, d.st>>>((const char *) alias, (char *) dst_dev, (unsigned long long) nbytes, rec);
        MOERT_CUDA_CHECK(cudaGetLastError());
    } else {
        if (d.sm) { (void) cudaGetLastError(); d.sm_fallback.fetch_add(1, std::memory_order_relaxed); }
        MOERT_CUDA_CHECK(cudaMemcpyAsync(dst_dev, src, (size_t) nbytes, cudaMemcpyHostToDevice, d.st));
    }
    d.async_copies.fetch_add(1, std::memory_order_relaxed);
    d.bytes.fetch_add((long long) nbytes, std::memory_order_relaxed);
    return 1;
}

extern "C" int64_t moert_cuda_dir_mark(void) {
    Director & d = g_dir;
    if (!d.on) return -1;
    std::lock_guard<std::mutex> lk(d.mu);
    if (d.ev_pool.empty()) { cudaEvent_t e; MOERT_CUDA_CHECK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming)); d.ev_pool.push_back(e); }
    cudaEvent_t e = d.ev_pool.back(); d.ev_pool.pop_back();
    MOERT_CUDA_CHECK(cudaEventRecord(e, d.st));
    d.marks.push_back({ e, d.next_mark });
    d.async_marks.fetch_add(1, std::memory_order_relaxed);
    return d.next_mark++;
}

extern "C" int64_t moert_cuda_dir_poll(int32_t wait_all) {
    Director & d = g_dir;
    if (!d.on) return -1;
    std::lock_guard<std::mutex> lk(d.mu);
    if (wait_all) MOERT_CUDA_CHECK(cudaStreamSynchronize(d.st));
    while (!d.marks.empty()) {
        const cudaError_t e = cudaEventQuery(d.marks.front().ev);
        if (e == cudaErrorNotReady) break;
        MOERT_CUDA_CHECK(e);
        d.done_upto = d.marks.front().id;
        d.ev_pool.push_back(d.marks.front().ev);
        d.marks.pop_front();
    }
    return d.done_upto;
}

extern "C" int32_t moert_cuda_publish_h2d(int32_t n, void * const * dsts, const void * const * srcs, const uint64_t * sizes) {
    if (n <= 0) return 1;
    if (!g_pub_st) MOERT_CUDA_CHECK(cudaStreamCreateWithFlags(&g_pub_st, cudaStreamNonBlocking));
    std::vector<size_t> sz((size_t) n);
    size_t tot = 0;
    for (int i = 0; i < n; ++i) { sz[(size_t) i] = (size_t) sizes[i]; tot += (sz[(size_t) i] + 63) & ~(size_t) 63; }
    if (tot > g_pub_pin_cap) {   // the policy views are pageable: stage them into pinned memory, then one batch
        if (g_pub_pin) MOERT_CUDA_CHECK(cudaFreeHost(g_pub_pin));
        g_pub_pin_cap = std::max(tot, (size_t) 256 << 10);
        MOERT_CUDA_CHECK(cudaHostAlloc((void **) &g_pub_pin, g_pub_pin_cap, cudaHostAllocDefault));
    }
    std::vector<const void *> st((size_t) n);
    size_t off = 0;
    for (int i = 0; i < n; ++i) {
        std::memcpy(g_pub_pin + off, srcs[i], sz[(size_t) i]);
        st[(size_t) i] = g_pub_pin + off;
        off += (sz[(size_t) i] + 63) & ~(size_t) 63;
    }
    batch_h2d(dsts, st.data(), sz.data(), (size_t) n, g_pub_st);   // pinned staging, stream order
    MOERT_CUDA_CHECK(cudaStreamSynchronize(g_pub_st));          // landed before the decode that reads them
    ++g_pub_calls; g_pub_copies += n;
    return 1;
}

// Early-decision gate, one entry point: op 0 enable (arg = index of the last MoE layer), 1 arm (returns the target a copy
// decided from now on must wait for), 2 wait for target arg (then the copy stream waits on the last record), 3 open from
// the host (the decode returned), 4 stats (waits, timeouts, ms*1000).
extern "C" int64_t moert_cuda_dir_gate(int32_t op, int64_t arg) {
    Gate & g = g_gate;
    switch (op) {
        case 0: {   // arg: last MoE layer | (per-layer mode << 16)
            const int ll = (int) (arg & 0xffff);
            if (ll < 0 || ll >= Gate::MAXL) return 0;
            for (int l = 0; l <= ll; ++l) if (!g.ev[l]) MOERT_CUDA_CHECK(cudaEventCreateWithFlags(&g.ev[l], cudaEventDisableTiming));
            g.last_layer = ll; g.per_layer = (arg >> 16) != 0; g.on = true;
            return 1;
        }
        case 1: {   // arm: the next forward's marks
            std::lock_guard<std::mutex> lk(g.mu);
            long long m = 0;
            for (int l = 0; l <= g.last_layer; ++l) if (g.per_layer || l == g.last_layer) m = std::max(m, g.c[l]);
            g.armed = m + 1;
            return g.armed;
        }
        case 2: {   // arg: (layer << 40) | target
            const long long target = arg & ((1LL << 40) - 1);
            int l = (int) (arg >> 40);
            if (!g.on || target <= 0) return 0;
            if (!g.per_layer || l < 0 || l > g.last_layer) l = g.last_layer;
            const auto t0 = std::chrono::steady_clock::now();
            bool rec = false;
            {
                std::unique_lock<std::mutex> lk(g.mu);
                if (!g.cv.wait_for(lk, std::chrono::milliseconds(2000), [&] { return g.c[l] >= target; })) ++g.timeouts;
                rec = g.recorded[l];
                ++g.waits;
                g.wait_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            }
            if (rec && g_dir.st) MOERT_CUDA_CHECK(cudaStreamWaitEvent(g_dir.st, g.ev[l], 0));
            return 1;
        }
        case 3: {   // the decode returned: every layer's gate reaches the armed target
            { std::lock_guard<std::mutex> lk(g.mu); for (int l = 0; l <= g.last_layer; ++l) g.c[l] = std::max(g.c[l], g.armed); }
            g.cv.notify_all();
            return 1;
        }
        case 5:   // engine source on (arg = pace, runs in the engine FIFO at most)
            g_dir.engine = arg > 0; g_dir.pace = (int) std::max<int64_t>(1, arg);
            return 1;
        case 6:   // engine stats: arg 0 copies, 1 pacing waits, 2 pacing wait us
            if (arg == 0) return g_dir.eng_copies;
            if (arg == 1) return g_dir.eng_waits;
            return (int64_t) (g_dir.eng_wait_ms * 1000.0);
        case 4:
            if (arg == 0) return g.waits;
            if (arg == 1) return g.timeouts;
            return (int64_t) (g.wait_ms * 1000.0);
    }
    return 0;
}

extern "C" void moert_cuda_dir_stats_get(moert_cuda_dir_stats * out) {
    if (!out) return;
    const Director & d = g_dir;
    out->async_copies = d.async_copies.load(); out->async_marks = d.async_marks.load(); out->bytes = d.bytes.load();
    out->pub_calls = g_pub_calls; out->pub_copies = g_pub_copies;
    out->sm_copies = d.sm_copies.load(); out->sm_fallback = d.sm_fallback.load();
    out->sm_ring_copies = d.ring_copies; out->sm_ring_waits = d.ring_waits; out->sm_ring_wait_ms = d.ring_wait_ms; out->sm_stage_ms = d.stage_ms;
    out->sm_timed = 0; out->sm_gpu_ns = 0; out->sm_gpu_ns_max = 0;
    if (d.sm_rec) {   // only the launches that already finished have both stamps
        const long long n = std::min(out->sm_copies, Director::SM_REC);
        for (long long i = 0; i < n; ++i) {
            const unsigned long long a = d.sm_rec[2 * i], b = d.sm_rec[2 * i + 1];
            if (!a || b < a) continue;
            out->sm_gpu_ns += (long long) (b - a);
            out->sm_gpu_ns_max = std::max(out->sm_gpu_ns_max, (long long) (b - a));
            ++out->sm_timed;
        }
    }
}

// ---------------------------------------------------------------------------------
// C ABI
// ---------------------------------------------------------------------------------

extern "C" void moert_cuda_configure(const moert_cuda_config * cfg) {
    if (cfg) g_cfg = *cfg;
}

// Enable the device routing counters (idempotent; re-enable with different dims reallocates).
// Returns 1 on success.
extern "C" int32_t moert_cuda_route_counts_enable(int32_t n_layers, int32_t n_expert) {
    if (n_layers <= 0 || n_expert <= 0) return 0;
    if (g_rc.on && g_rc.nl == n_layers && g_rc.ne == n_expert) return 1;
    if (g_rc.dev) { MOERT_CUDA_CHECK(cudaFree(g_rc.dev)); g_rc.dev = nullptr; g_rc.on = false; }
    g_rc.nl = n_layers; g_rc.ne = n_expert;
    const size_t bytes = g_rc.n_ints() * sizeof(int32_t);
    if (cudaMalloc(&g_rc.dev, bytes) != cudaSuccess) { (void) cudaGetLastError(); g_rc.nl = g_rc.ne = 0; return 0; }
    MOERT_CUDA_CHECK(cudaMemset(g_rc.dev, 0, bytes));
    g_rc.on = true;
    return 1;
}

// Fetch (and optionally reset) the device routing counters into `dst` (n_layers*n_expert + n_layers*2
// int32, same layout as above). Synchronous D2H on the legacy default stream — call it only at refresh
// points, after llama_decode returned (the graph is complete). Returns the int32 count copied, 0 if off.
extern "C" int32_t moert_cuda_route_counts_fetch(int32_t * dst, int32_t reset) {
    if (!g_rc.on || !dst) return 0;
    const size_t n = g_rc.n_ints();
    MOERT_CUDA_CHECK(cudaMemcpy(dst, g_rc.dev, n * sizeof(int32_t), cudaMemcpyDeviceToHost));
    if (reset) MOERT_CUDA_CHECK(cudaMemset(g_rc.dev, 0, n * sizeof(int32_t)));
    return (int32_t) n;
}

// 3b instant pin: page-lock an existing host buffer IN PLACE (cudaHostRegisterDefault, no device
// mapping) so H2D staging copies from it are async + ~2.2x. Used to pin the whole owned IQ2 cold
// arena once at load.
// 1 = pinned (or already registered), 0 = failed (buffer stays pageable; caller continues).
extern "C" int32_t moert_cuda_host_register_pinned(void * base, uint64_t size) {
    if (!base || size == 0) return 0;
    cudaError_t e = cudaHostRegister(base, (size_t) size, cudaHostRegisterDefault);
    if (e == cudaSuccess || e == cudaErrorHostMemoryAlreadyRegistered) { (void) cudaGetLastError(); return 1; }
    (void) cudaGetLastError();
    // a read-only mapping (the mapped pack) can only be registered read-only (findings 213g)
    e = cudaHostRegister(base, (size_t) size, cudaHostRegisterReadOnly);
    if (e == cudaSuccess || e == cudaErrorHostMemoryAlreadyRegistered) { (void) cudaGetLastError(); return 1; }
    (void) cudaGetLastError();
    return 0;
}

// Mapped registration (findings 233, the SM transport's direct source): a kernel can read the range through its device
// alias. Counts in WDDM's shared-memory budget: register a part of the arena only (7 GiB killed the boot, findings 231).
extern "C" int32_t moert_cuda_host_register_mapped(void * base, uint64_t size) {
    if (!base || size == 0) return 0;
    cudaError_t e = cudaHostRegister(base, (size_t) size, cudaHostRegisterMapped);
    if (e == cudaSuccess || e == cudaErrorHostMemoryAlreadyRegistered) { (void) cudaGetLastError(); return 1; }
    (void) cudaGetLastError();
    e = cudaHostRegister(base, (size_t) size, cudaHostRegisterMapped | cudaHostRegisterReadOnly);   // the mapped pack
    if (e == cudaSuccess || e == cudaErrorHostMemoryAlreadyRegistered) { (void) cudaGetLastError(); return 1; }
    (void) cudaGetLastError();
    return 0;
}

// Device alias of [base, base + size) (findings 248: the drafter's head read by kernels from host RAM): the range is
// widened to whole 64 KiB pages and registered mapped (read-only if it is a read-only file mapping), and the alias of
// `base` itself is returned; nullptr if the driver refuses.
extern "C" void * moert_cuda_host_map(void * base, uint64_t size) {
    if (!base || size == 0) return nullptr;
    {   // already pinned host memory (cudaMallocHost: ggml's CUDA_Host buffers) has a device alias of its own
        void * d0 = nullptr;
        if (cudaHostGetDevicePointer(&d0, base, 0) == cudaSuccess && d0) return d0;
        (void) cudaGetLastError();
    }
    const uintptr_t pg = 65536, lo = (uintptr_t) base & ~(pg - 1), hi = ((uintptr_t) base + size + pg - 1) & ~(pg - 1);
    if (!moert_cuda_host_register_mapped((void *) lo, (uint64_t) (hi - lo))) {
        fprintf(stderr, "moert/cuda: host_map: cudaHostRegister(mapped) refused %p + %llu MiB\n", (void *) lo,
                (unsigned long long) ((hi - lo) >> 20));
        return nullptr;
    }
    void * d = nullptr;
    const cudaError_t e = cudaHostGetDevicePointer(&d, (void *) lo, 0);
    if (e != cudaSuccess || !d) {
        fprintf(stderr, "moert/cuda: host_map: no device alias (%s)\n", cudaGetErrorString(e));
        (void) cudaGetLastError(); return nullptr;
    }
    return (char *) d + ((uintptr_t) base - lo);
}

// Undo a moert_cuda_host_register_pinned (cudaHostUnregister); safe on any base (errors cleared).
extern "C" void moert_cuda_host_unregister(void * base) {
    if (base) { (void) cudaHostUnregister(base); (void) cudaGetLastError(); }
}

// ---------------------------------------------------------------------------------
// The fused op (orchestration verbatim from the pre-M2 ggml_cuda_moe_dual, with the two
// ggml services — quantized mul_mat_id + LIFO scratch pool — behind the host vtable)
// ---------------------------------------------------------------------------------

// host time inside the op, for moert_cuda_stats (one RAII stamp per call)
static long long g_op_fires = 0, g_op_ns = 0;
struct OpTimer {
    long long t0 = sync_prof_now();
    ~OpTimer() { g_op_ns += sync_prof_now() - t0; ++g_op_fires; }
};

extern "C" void moert_cuda_stats_get(moert_cuda_stats * out) {
    if (!out) return;
    out->fires = g_op_fires; out->op_host_ns = g_op_ns;
    out->seam_syncs = g_seam.sites[0].n; out->seam_ns = g_seam.sites[0].ns;
    out->coldcpu_calls = g_coldcpu.calls; out->coldcpu_pairs = g_coldcpu.pairs;
    out->coldcpu_ns = g_coldcpu.cpu_ns; out->coldcpu_over_max = g_coldcpu.over_max;
}

// Loaned slots (findings 246): per (layer, kind) slot -> device pointer table of the hot pool, or none.
namespace {
struct HotTable { const void * const * ptrs = nullptr; int32_t n = 0, type = 0; };
HotTable g_hot_tab[128][2];
}
extern "C" void moert_cuda_set_hot_table(int32_t il, int32_t kind, const void * const * dev_ptrs, int32_t n, int32_t type_tag) {
    if (il < 0 || il >= 128 || kind < 0 || kind > 1) return;
    g_hot_tab[il][kind].ptrs = n > 0 ? dev_ptrs : nullptr;
    g_hot_tab[il][kind].n    = n > 0 ? n : 0;
    g_hot_tab[il][kind].type = type_tag;
}

extern "C" void moert_ffn_moe(const moert_ffn_moe_args * a, const moert_ffn_moe_host * h) {
    const OpTimer op_timer;
    const int kind       = a->kind;
    const int top_k      = a->top_k;
    const int planc_mode = a->planc_mode;
    const int cold_type  = a->cold_type_tag;

    cudaStream_t stream = (cudaStream_t) a->stream;

    const int n_used   = a->n_used;
    const int n_tokens = a->n_tokens;
    const int Pmax     = n_used * n_tokens;
    const int D_in     = a->d_in;
    const int D_out    = a->d_out;
    const int n_expert = a->n_expert;
    const int K        = a->k;
    const size_t expert_bytes = (size_t) a->expert_bytes;

    // Slice 3: gate_up (kind 0) feeds one shared activation column (a single token vector -> all routed
    // experts). mul_mat_id picks the activation column as channel_dst % nchannels_y, so passing cur
    // directly with nchannels_y==1 makes every channel read that one column - no x-gather, and the
    // quantize covers one column instead of the Pmax-fold replica. down (kind 1) has a distinct
    // activation per routed pair, so it still gathers into x_hot/x_cold.
    // n_tokens>1 (K=2 speculative batched decode) also needs the gather: the Slice-3 shortcut
    // shares ONE activation column, valid only for a single token. With gather_x, the partition
    // fills x_hot/x_cold with each pair's per-token activation (src_col = t = lane/n_used), the
    // cold matmul reads x_cold, and the sidecar quantizes per-pair (sc.x_per_pair = gather_x).
    const bool gather_x = (kind != 0 || n_tokens > 1);

    const int il_param = a->il;
    // Routed-only copy stream (RAM cold, MOE_DUAL_COPYSTREAM): its ev_part must be recorded before the
    // hot matmul, so the decision is made here, before the partition kernel is enqueued.
    const bool use_cstream = g_cfg.copy_stream && planc_mode == 2;

    // Scratch from the host's pool: SAME allocation order as the pre-M2 pool_alloc declarations
    // (the ggml pool frees LIFO — the epilogue below releases in exact reverse). x_hot/x_cold are
    // always allocated to keep that order; they stay unused (no memset, no gather) on the shared-cur
    // path. The ring is Plan C staging scratch (mode 2 only, allocated unconditionally likewise).
    void * scratch[16];
    int    n_scratch = 0;
    auto pool_alloc = [&](uint64_t nbytes) -> void * {
        void * p = h->pool_alloc(h->user, nbytes);
        scratch[n_scratch++] = p;
        return p;
    };
    float   * x_hot        = (float *)   pool_alloc((uint64_t) D_in * Pmax * sizeof(float));
    float   * x_cold       = (float *)   pool_alloc((uint64_t) D_in * Pmax * sizeof(float));
    int32_t * ids_hot      = (int32_t *) pool_alloc((uint64_t) Pmax * sizeof(int32_t));
    int32_t * ids_cold     = (int32_t *) pool_alloc((uint64_t) Pmax * sizeof(int32_t));
    int32_t * hot_dst      = (int32_t *) pool_alloc((uint64_t) Pmax * sizeof(int32_t));
    int32_t * cold_dst     = (int32_t *) pool_alloc((uint64_t) Pmax * sizeof(int32_t));
    int32_t * counts       = (int32_t *) pool_alloc(2 * sizeof(int32_t));
    float   * y_hot        = (float *)   pool_alloc((uint64_t) D_out * Pmax * sizeof(float));
    float   * y_cold       = (float *)   pool_alloc((uint64_t) D_out * Pmax * sizeof(float));
    char    * ring         = (char *)    pool_alloc((uint64_t) Pmax * expert_bytes);
    int32_t * ids_ring     = (int32_t *) pool_alloc((uint64_t) Pmax * sizeof(int32_t));
    float   * s_cold_stage = (float *)   pool_alloc((uint64_t) (n_expert > 0 ? n_expert : 1) * sizeof(float));


    // Padded matmul columns must hold an in-range expert id (0); the scatter ignores the padded tail
    // (and the device-counted guard skips it entirely). Gathered activation columns are zeroed so a
    // padded id=0 reads x=0; the shared-cur path reads the real cur on padded channels but its output
    // is likewise ignored.
    // DEAD under the Slice-2 counted guard (default ON): mmvq's ne_live guard skips the padded
    // [n_live,Pmax) channels before any id/x read, and the scatter only touches [0,n_hot)/[0,n_cold)
    // (counts-guarded, see moe_dual_scatter_kernel), so nothing reads the padding. Only the counted=0
    // A/B path (matmul over all Pmax columns) needs these zeroed. ~180 memsets/tok (~0.56 ms/tok).
    if (!g_cfg.counted) {
        MOERT_CUDA_CHECK(cudaMemsetAsync(ids_hot,  0, (size_t) Pmax * sizeof(int32_t), stream));
        MOERT_CUDA_CHECK(cudaMemsetAsync(ids_cold, 0, (size_t) Pmax * sizeof(int32_t), stream));
        if (gather_x) {
            MOERT_CUDA_CHECK(cudaMemsetAsync(x_hot,  0, (size_t) D_in * Pmax * sizeof(float), stream));
            MOERT_CUDA_CHECK(cudaMemsetAsync(x_cold, 0, (size_t) D_in * Pmax * sizeof(float), stream));
        }
    }

    // device routing counters: only kind 0 counts (gate_up and down share ids/residency per layer)
    int32_t * rc_counts = nullptr, * rc_hits = nullptr;
    if (g_rc.on && kind == 0 && il_param >= 0 && il_param < g_rc.nl) {
        rc_counts = g_rc.dev + (size_t) il_param * g_rc.ne;
        rc_hits   = g_rc.dev + (size_t) g_rc.nl * g_rc.ne + (size_t) il_param * 2;
    }

    moe_dual_partition_kernel<<<1, 32, 0, stream>>>(
        (const int32_t *) a->ids, n_used, a->ids_s1, top_k, kind,
        (const float *) a->cur, D_in,
        (const int32_t *) a->exp_slot, (const int32_t *) a->slot_exp, (const int32_t *) a->slot_state,
        n_expert, K,
        ids_hot, ids_cold,
        gather_x ? x_hot  : nullptr,
        gather_x ? x_cold : nullptr,
        hot_dst, cold_dst, counts, Pmax, rc_counts, rc_hits, gather_x ? 1 : 0);
    if (gather_x)
        moe_dual_gather_kernel<<<dim3(Pmax, 2), 128, 0, stream>>>((const float *) a->cur, D_in, kind, n_used, counts,
                                                                   hot_dst, cold_dst, x_hot, x_cold);
    if (use_cstream) {                                         // copy-stream cold list waits on partition, NOT the hot mm below
        g_coldstream.lazy_init();
        MOERT_CUDA_CHECK(cudaEventRecord(g_coldstream.ev_part, stream));
    }

    // Slice 2: device-counted guard. counts[0]=n_hot, counts[1]=n_cold (written by the partition).
    // Passing them as ne_live makes the quantized matmul skip the padded [n_live,Pmax) channels before
    // any dequant/vec_dot, so each routed pair is computed once (not over the padded Pmax). The launch
    // shape stays Pmax (static => CUDA-graph-capturable); only the per-channel work is gated on device.
    // counted=0 forces the Slice 1 padded path (for A/B). Scatter reads only the live columns either
    // way, so output is identical; the guard only removes the padded matmul work.
    const int32_t * ne_hot  = g_cfg.counted ? counts     : nullptr;
    const int32_t * ne_cold = g_cfg.counted ? counts + 1 : nullptr;

    // hot matmul first (independent of the cold source, so cold staging can overlap it)
    {
        moert_mm_call c = {};
        c.w_sel   = MOERT_MM_W_HOT;
        if (il_param >= 0 && il_param < 128 && (kind == 0 || kind == 1)) {
            const HotTable & ht = g_hot_tab[il_param][kind];
            if (ht.ptrs && ht.n == K) {   // loaned slots: every slot through its pointer
                c.w_sel = MOERT_MM_W_RAW; c.w_type_tag = ht.type; c.w_n = K; c.w_ptrs = ht.ptrs;
                c.w_data = (const void *) ht.ptrs;   // ignored by the matmul when w_ptrs is set
            }
        }
        c.b       = gather_x ? (const void *) x_hot : a->cur;
        c.b_ncols = gather_x ? Pmax : 1;
        c.ids     = ids_hot;
        c.n_ids   = Pmax;
        c.y       = y_hot;
        c.ne_live = ne_hot;
        h->mm(h->user, &c);
    }

    // Cold weight source per planc_mode.
    bool skip_cold_mm = false;   // set by the CPU cold sidecar (y_cold authored on the host)
    moert_mm_call cc = {};
    cc.w_sel   = a->has_w_cold ? MOERT_MM_W_COLD_SRC : MOERT_MM_W_RAW;   // mode 0: GPU-resident cold
    cc.b       = gather_x ? (const void *) x_cold : a->cur;
    cc.b_ncols = gather_x ? Pmax : 1;
    cc.ids     = ids_cold;                                   // partition expert ids (mode 0/1)
    cc.n_ids   = Pmax;
    cc.y       = y_cold;
    cc.ne_live = ne_cold;
    const float * s_cold_ptr = (const float *) a->s_cold;    // device .scale (mode 0), or null
    if (planc_mode == 2) {                            // RAM cold: CPU sidecar, or the staging ring over max
        // Per-op cold path. With the copy stream the cold-list D2H (+ activation D2H, + the H2D copies or
        // the sidecar's output upload) run on `cs` gated on ev_part (partition only), so the host sync
        // excludes the hot matmul (which runs concurrently) and ev_done joins cs back before the scatter.
        // cs == the compute stream reproduces the single-stream behavior (full drain).
        cudaStream_t cs = use_cstream ? g_coldstream.st : stream;
        if (use_cstream) { ++g_coldstream.fires; MOERT_CUDA_CHECK(cudaStreamWaitEvent(cs, g_coldstream.ev_part, 0)); }
        // cold_ffn (findings 206): ONE asynchronous cold job per layer. The gate_up op queues the cold
        // list, the pair map and the token columns D2H, records an event and hands the whole cold FFN to
        // the host without blocking; its own cold columns are zeros (GLU of zeros is zero, and the down
        // op recomputes nothing for them). The down op waits for the job and uploads its raw outputs.
        // Same routing in both ops (same ids, residency only changes between decodes), so the down
        // op's partition compacts the cold pairs in the job's order.
        const bool ffn = g_cfg.cold_ffn && g_cfg.coldcpu && g_cfg.coldffn_submit_fn && g_cfg.coldffn_wait_fn &&
                         Pmax <= 32 && (kind == 0 || kind == 1);
        if (ffn && kind == 0) {
            g_coldcpu.ensure_pinned(Pmax, D_in, D_out);
            int32_t * h_hdr = g_hdrpin.get();
            MOERT_CUDA_CHECK(cudaMemcpyAsync(h_hdr,          counts,   2 * sizeof(int32_t),             cudaMemcpyDeviceToHost, cs));
            MOERT_CUDA_CHECK(cudaMemcpyAsync(h_hdr + 2,      ids_cold, (size_t) Pmax * sizeof(int32_t), cudaMemcpyDeviceToHost, cs));
            MOERT_CUDA_CHECK(cudaMemcpyAsync(h_hdr + 2 + 32, cold_dst, (size_t) Pmax * sizeof(int32_t), cudaMemcpyDeviceToHost, cs));
            MOERT_CUDA_CHECK(cudaMemcpyAsync(g_coldcpu.x_pin, a->cur, (size_t) D_in * n_tokens * sizeof(float),
                                             cudaMemcpyDeviceToHost, cs));
            g_coldffn.lazy_init();
            MOERT_CUDA_CHECK(cudaEventRecord(g_coldffn.ev, cs));
            // the op returns without waiting for these D2H: with the copy stream, the compute stream must not run
            // ahead of them, or the next ops reuse a->cur and the pool scratch (ids_cold, cold_dst, counts) before
            // they are read -- the host job then computes on garbage (claude dump findings 219)
            if (use_cstream) MOERT_CUDA_CHECK(cudaStreamWaitEvent(stream, g_coldffn.ev, 0));
            moert_coldffn_job job = {};
            job.il = il_param; job.n_used = n_used; job.n_tokens = n_tokens;
            job.h_counts = h_hdr; job.h_ids = h_hdr + 2; job.h_cdst = h_hdr + 2 + 32;
            job.x = g_coldcpu.x_pin;
            job.w_gu = (const void *) (uintptr_t) a->w_cold_dev; job.gu_bytes = expert_bytes;
            job.gu_stride = cold_stride(expert_bytes); job.gu_type = cold_type; job.d_in = D_in; job.d_gu = D_out;
            job.y = g_coldcpu.y_pin;
            job.wait_ready = &coldffn_wait_ready; job.ev = (void *) g_coldffn.ev;
            ((moert_coldffn_submit_fn) (uintptr_t) g_cfg.coldffn_submit_fn)(&job);
            ++g_coldffn.submits;
            MOERT_CUDA_CHECK(cudaMemsetAsync(y_cold, 0, (size_t) D_out * Pmax * sizeof(float), stream));
            skip_cold_mm = true;
            s_cold_ptr = nullptr;
        } else if (ffn && kind == 1) {
            g_coldcpu.ensure_pinned(Pmax, D_in, D_out);
            int64_t cpu_ns = 0;
            const int32_t n_cold = ((moert_coldffn_wait_fn) (uintptr_t) g_cfg.coldffn_wait_fn)(il_param, &cpu_ns);
            ++g_coldcpu.calls;
            g_coldcpu.pairs += n_cold;
            g_coldcpu.cpu_ns += cpu_ns;
            if (n_cold > 0)
                MOERT_CUDA_CHECK(cudaMemcpyAsync(y_cold, g_coldcpu.y_pin, (size_t) n_cold * D_out * sizeof(float),
                                                 cudaMemcpyHostToDevice, cs));
            skip_cold_mm = true;
            s_cold_ptr = nullptr;
            if (a->s_cold_dev) {
                if (il_param >= 0 && il_param < 128) {
                    float *& sdc = g_coldcpu.s_dev_cache[il_param];
                    if (!sdc) {
                        MOERT_CUDA_CHECK(cudaMalloc(&sdc, (size_t) n_expert * sizeof(float)));
                        MOERT_CUDA_CHECK(cudaMemcpyAsync(sdc, (const void *) (uintptr_t) a->s_cold_dev,
                                                         (size_t) n_expert * sizeof(float), cudaMemcpyHostToDevice, cs));
                    }
                    s_cold_ptr = sdc;
                } else {
                    MOERT_CUDA_CHECK(cudaMemcpyAsync(s_cold_stage, (const void *) (uintptr_t) a->s_cold_dev,
                                                     (size_t) n_expert * sizeof(float), cudaMemcpyHostToDevice, cs));
                    s_cold_ptr = s_cold_stage;
                }
            }
            if (use_cstream) {
                MOERT_CUDA_CHECK(cudaEventRecord(g_coldstream.ev_done, cs));
                MOERT_CUDA_CHECK(cudaStreamWaitEvent(stream, g_coldstream.ev_done, 0));
            }
        } else {
        // CPU cold sidecar: the activation D2H rides the SAME sync the cold-list D2H already pays; the
        // decision (n_cold <= max) is taken after the sync. With the copy stream that sync drains ONLY
        // cs, so the hot matmul runs concurrently with the host wait + the sidecar's CPU cold compute.
        const bool try_coldcpu = g_cfg.coldcpu && g_cfg.coldcpu_fn && kind >= 0 && kind < 2;
        if (try_coldcpu) g_coldcpu.ensure_pinned(Pmax, D_in, D_out);
        int32_t * h_hdr  = g_hdrpin.get();   // pinned {counts[2], cold_ids[Pmax]} -> honest seam timing
        int32_t * h_cnt  = h_hdr;
        int32_t * h_list = h_hdr + 2;
        int32_t * h_cdst = h_hdr + 2 + 32;   // cold pair index p (token = p / n_used), gate_up under MTP
        // gate_up with several tokens: the sidecar needs only the token columns (a->cur, n_tokens x D_in)
        // and which token each cold pair belongs to, not the Pmax gathered copies (findings 202)
        const bool tok_cols = kind == 0 && n_tokens > 1 && n_tokens <= 32;
        // The FIXED SEAM, paid before n_cold is known. Whichever sub-branch runs below, these
        // copies and the host block have already happened.
        MOERT_CUDA_CHECK(cudaMemcpyAsync(h_cnt,  counts,   2 * sizeof(int32_t),             cudaMemcpyDeviceToHost, cs));
        MOERT_CUDA_CHECK(cudaMemcpyAsync(h_list, ids_cold, (size_t) Pmax * sizeof(int32_t), cudaMemcpyDeviceToHost, cs));
        if (try_coldcpu && g_coldcpu.x_pin) {
            // gate_up decode shares ONE activation column (a->cur); down has per-pair columns,
            // already compacted into x_cold[:,0..n_cold-1] by the partition kernel.
            if (tok_cols) {
                MOERT_CUDA_CHECK(cudaMemcpyAsync(h_cdst, cold_dst, (size_t) Pmax * sizeof(int32_t), cudaMemcpyDeviceToHost, cs));
                MOERT_CUDA_CHECK(cudaMemcpyAsync(g_coldcpu.x_pin, a->cur,
                                                 (size_t) D_in * n_tokens * sizeof(float), cudaMemcpyDeviceToHost, cs));
            } else if (gather_x) MOERT_CUDA_CHECK(cudaMemcpyAsync(g_coldcpu.x_pin, x_cold,
                                            (size_t) D_in * Pmax * sizeof(float), cudaMemcpyDeviceToHost, cs));
            else          MOERT_CUDA_CHECK(cudaMemcpyAsync(g_coldcpu.x_pin, a->cur,
                                            (size_t) D_in * sizeof(float), cudaMemcpyDeviceToHost, cs));
        }
        seam_sync(cs, 0);   // the host needs the cold list (and the activation) to go on
        const int n_cold = h_cnt[1];
        if (g_cfg.cold_guard_fn && n_cold > 0)   // exclusive arena: every cold page resident before any read
            ((moert_cold_guard_fn) (uintptr_t) g_cfg.cold_guard_fn)(il_param, h_list, n_cold);
        if (try_coldcpu && g_coldcpu.x_pin && n_cold <= g_cfg.coldcpu_max) {
            // Sidecar path: weights never cross PCIe. Compute the n_cold pairs on the host from
            // the RAM pool and upload ONLY the outputs into y_cold[:,0..n_cold-1] (the exact
            // columns the GPU cold matmul would have written; scatter/scale/graph unchanged).
            ++g_coldcpu.calls;
            g_coldcpu.pairs += n_cold;
            g_coldcpu.bytes_avoided += (long long) n_cold * (long long) expert_bytes;
            if (n_cold > 0) {
                moert_coldcpu_call sc = {};
                sc.il = il_param; sc.kind = kind; sc.n_cold = n_cold;
                sc.cold_ids = h_list;
                sc.x = g_coldcpu.x_pin; sc.x_per_pair = gather_x ? 1 : 0;
                int32_t xcol[32];
                if (tok_cols) {
                    for (int i = 0; i < n_cold && i < 32; ++i) xcol[i] = h_cdst[i] / n_used;
                    sc.x_col = xcol;
                }
                sc.w_host = (const void *) (uintptr_t) a->w_cold_dev;
                sc.expert_bytes = expert_bytes; sc.stride = cold_stride(expert_bytes);
                sc.w_type_tag = cold_type;
                sc.d_in = D_in; sc.d_out = D_out;
                sc.y = g_coldcpu.y_pin;
                const long long c0 = sync_prof_now();
                ((moert_coldcpu_fn) (uintptr_t) g_cfg.coldcpu_fn)(&sc);
                const long long c1 = sync_prof_now();
                g_coldcpu.cpu_ns += c1 - c0;
                // y H2D goes on cs, NOT the compute stream: the next fire's cs sync is what
                // fences the pinned y_pin against host reuse (on the compute stream it would
                // never be drained by the copy-stream sync -> use-while-in-flight race).
                MOERT_CUDA_CHECK(cudaMemcpyAsync(y_cold, g_coldcpu.y_pin,
                                 (size_t) n_cold * D_out * sizeof(float), cudaMemcpyHostToDevice, cs));
            }
            skip_cold_mm = true;                      // y_cold is authored by the sidecar
            if (a->s_cold_dev) {                      // scatter still applies the down .scale
                // Static per-layer content: upload once, reuse the cached device copy on
                // every later fire for this layer (was: re-copied into scratch every fire).
                if (il_param >= 0 && il_param < 128) {
                    float *& sdc = g_coldcpu.s_dev_cache[il_param];
                    if (!sdc) {
                        MOERT_CUDA_CHECK(cudaMalloc(&sdc, (size_t) n_expert * sizeof(float)));
                        MOERT_CUDA_CHECK(cudaMemcpyAsync(sdc, (const void *) (uintptr_t) a->s_cold_dev,
                                                         (size_t) n_expert * sizeof(float), cudaMemcpyHostToDevice, cs));
                    }
                    s_cold_ptr = sdc;
                } else {
                    MOERT_CUDA_CHECK(cudaMemcpyAsync(s_cold_stage, (const void *) (uintptr_t) a->s_cold_dev,
                                                     (size_t) n_expert * sizeof(float), cudaMemcpyHostToDevice, cs));
                    s_cold_ptr = s_cold_stage;
                }
            } else {
                s_cold_ptr = nullptr;
            }
            // Join: the scatter on the compute stream reads y_cold + s_cold_stage, both
            // authored on cs.
            if (use_cstream) {
                MOERT_CUDA_CHECK(cudaEventRecord(g_coldstream.ev_done, cs));
                MOERT_CUDA_CHECK(cudaStreamWaitEvent(stream, g_coldstream.ev_done, 0));
            }
        } else {
            if (try_coldcpu) ++g_coldcpu.over_max;        // armed but n_cold > max (or pin failed): weight staging
            const char * host_pool = (const char *) (uintptr_t) a->w_cold_dev;   // HOST base pointer in mode 2
            const size_t as = cold_stride(expert_bytes);
            for (int i = 0; i < n_cold; ++i) {
                int e = h_list[i];
                if (e < 0 || e >= n_expert) e = 0;   // defensive in-range host-pool read
                MOERT_CUDA_CHECK(cudaMemcpyAsync(ring + (size_t) i * expert_bytes,
                                                 host_pool + (size_t) e * as, expert_bytes,
                                                 cudaMemcpyHostToDevice, cs));
            }
            moe_dual_iota_kernel<<<1, 32, 0, stream>>>(ids_ring, Pmax);  // ring slot ids 0..Pmax-1 (compute stream: independent)
            cc.w_sel = MOERT_MM_W_RAW; cc.w_type_tag = cold_type; cc.w_data = ring; cc.w_n = Pmax;
            cc.ids = ids_ring;                            // matmul selects ring slots; scatter still uses expert ids
            if (a->s_cold_dev) {
                MOERT_CUDA_CHECK(cudaMemcpyAsync(s_cold_stage, (const void *) (uintptr_t) a->s_cold_dev,
                                                 (size_t) n_expert * sizeof(float), cudaMemcpyHostToDevice, cs));
                s_cold_ptr = s_cold_stage;
            } else {
                s_cold_ptr = nullptr;
            }
            // Join the copy stream back into the compute stream: the cold matmul (and the scatter's read of
            // s_cold_stage) must not start until the ring + .scale copies complete.
            if (use_cstream) {
                MOERT_CUDA_CHECK(cudaEventRecord(g_coldstream.ev_done, cs));
                MOERT_CUDA_CHECK(cudaStreamWaitEvent(stream, g_coldstream.ev_done, 0));
            }
        }
        }   // cold_ffn else
    }

    if (!skip_cold_mm) h->mm(h->user, &cc);

    moe_dual_scatter_kernel<<<Pmax, 64, 0, stream>>>(
        y_hot, y_cold, D_out, kind, counts,
        ids_hot, hot_dst, ids_cold, cold_dst,
        (const float *) a->s_hot,
        s_cold_ptr,
        (float *) a->dst);

    // early-decision gate: the last MoE layer's slab reads are enqueued -- copies decided during this verify may follow
    if (g_gate.on && kind == 1 && il_param >= 0 && il_param <= g_gate.last_layer &&
        (g_gate.per_layer || il_param == g_gate.last_layer)) {
        MOERT_CUDA_CHECK(cudaEventRecord(g_gate.ev[il_param], stream));
        { std::lock_guard<std::mutex> lk(g_gate.mu); g_gate.recorded[il_param] = true; ++g_gate.c[il_param]; }
        g_gate.cv.notify_all();
    }

    // release scratch in exact reverse allocation order (the ggml pool contract)
    while (n_scratch > 0) h->pool_free(h->user, scratch[--n_scratch]);
}
