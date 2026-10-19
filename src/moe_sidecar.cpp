#include "moe_sidecar.hpp"
#include "sidecar_kernels.hpp"   // the decode-once kernels (tested alone in tests/test_sidecar_kernels.cpp)

#include "ggml.h"
#include "ggml-cpu.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>
#include <algorithm>
#include <vector>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

#if defined(_MSC_VER)
#include <intrin.h>
#define SIDECAR_PAUSE() _mm_pause()
#else
#include <immintrin.h>
#define SIDECAR_PAUSE() _mm_pause()
#endif

namespace {

using namespace sck;

constexpr int PMAX = 32;      // matches the fused op's Pmax ceiling
constexpr int DMAX = 4096;    // >= any gemma4 per-kind d_in/d_out
constexpr int TILE_ROWS = 64;

// Our kernels (sidecar_kernels.hpp, decode once per expert row for all the pairs that share it) are the only path
// for the two expert types the product runs; the measured-negative C5.4 prefetch / address-order variants and the
// old-kernel A/B switch were removed on 2026-10-11 (tag pre-elision-2026-10-11).

static inline long long stats_now() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

// Per-kind live telemetry for the SYNC per-kind sidecar (
// slice A). Counters only — no behavior change; every write is on the single caller thread
// except wake_ns/wake_n (worker threads, atomic). Printed from the global destructor at
// process exit (shutdown() is never called on the live path), next to [MOE_DUAL_COLDCPU].
struct Stats {
    static const int MAX_IL = 64;
    // caller-thread counters, indexed by kind (0 gate_up / 1 down)
    long long calls[2] = {}, pairs[2] = {}, bytes[2] = {}, wall_ns[2] = {},
              quant_ns[2] = {}, tail_ns[2] = {}, repeats[2] = {};
    int       d_out[2] = {};                    // last seen (fixed per kind for gemma4)
    // worker wake latency: now - job_publish_ts, once per worker per fresh job
    std::atomic<long long> wake_ns{0}, wake_n{0};
    long long pub_ns = 0;                       // written under mu before the gen bump
    // previous fire's cold ids per (il, kind) == previous token (each fires once/position)
    int32_t last_ids[MAX_IL][2][PMAX];
    int32_t last_n[MAX_IL][2] = {};
    void note_ids(int il, int kind, const int32_t * ids, int n) {
        if (il < 0 || il >= MAX_IL || n > PMAX) return;
        int32_t (&prev)[PMAX] = last_ids[il][kind];
        const int pn = last_n[il][kind];
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < pn; ++j)
                if (ids[i] == prev[j]) { ++repeats[kind]; break; }
        memcpy(prev, ids, (size_t) n * sizeof(int32_t));
        last_n[il][kind] = n;
    }
    // WITHIN-fire duplicates (/
    // annotations): under MTP one fire's cold_ids can name the same expert twice if two
    // DIFFERENT tokens in the same verify batch route cold to it -- a question that did not
    // exist pre-MTP (one fire == one token then, top-k distinct by construction) and that
    // note_ids() above does NOT answer (it compares across fires/positions, not within one).
    // Measurement only, O(n^2) over n<=Pmax=32 (trivial, single call/fire).
    long long within_dup_pairs[2] = {}, within_dup_calls[2] = {};
    // calls that actually took the grouped decode-once path, and the weight-row decodes it
    // eliminated (n_cold - n_groups). saved_decodes SHOULD track within_dup_pairs closely on
    // gate_up; a gap means the fast-path gate is rejecting calls the counter counted.
    long long grouped_calls[2] = {}, saved_decodes[2] = {};
    void note_within_dup(int kind, const int32_t * ids, int n) {
        if (n > PMAX) n = PMAX;
        bool any = false;
        for (int i = 1; i < n; ++i)
            for (int j = 0; j < i; ++j)
                if (ids[i] == ids[j]) { ++within_dup_pairs[kind]; any = true; break; }
        if (any) ++within_dup_calls[kind];
    }
    ~Stats() {
        if (!calls[0] && !calls[1]) return;
        const char * kn[2] = { "gate_up", "down" };
        for (int k = 0; k < 2; ++k) {
            if (!calls[k]) continue;
            const double wall_ms = wall_ns[k] / 1e6;
            fprintf(stderr, "[MOE_SIDECAR_STATS] %-7s calls=%lld pairs=%lld (%.2f/call)"
                    " bytes=%.1f MiB | wall=%.1f ms quant=%.1f ms tail=%.1f ms"
                    " | eff %.1f GB/s | y-upload=%.1f MiB | repeat-pairs=%lld (%.1f%%)"
                    " | within-fire-dup=%lld (%.1f%% of pairs, %.1f%% of calls)"
                    " | grouped-calls=%lld saved-decodes=%lld (%.1f%% of pairs)\n",
                    kn[k], calls[k], pairs[k], (double) pairs[k] / calls[k],
                    bytes[k] / 1048576.0, wall_ms, quant_ns[k] / 1e6, tail_ns[k] / 1e6,
                    wall_ns[k] > 0 ? (double) bytes[k] / wall_ns[k] : 0.0,   // B/ns == GB/s
                    (double) pairs[k] * d_out[k] * sizeof(float) / 1048576.0,
                    repeats[k], pairs[k] ? 100.0 * repeats[k] / pairs[k] : 0.0,
                    within_dup_pairs[k], pairs[k] ? 100.0 * within_dup_pairs[k] / pairs[k] : 0.0,
                    calls[k] ? 100.0 * within_dup_calls[k] / calls[k] : 0.0,
                    grouped_calls[k], saved_decodes[k],
                    pairs[k] ? 100.0 * saved_decodes[k] / pairs[k] : 0.0);
        }
        const long long wn = wake_n.load(), wns = wake_ns.load();
        const long long tc = calls[0] + calls[1];
        fprintf(stderr, "[MOE_SIDECAR_STATS] worker wakes=%lld mean-latency=%.1f us"
                " | mean/call: wall=%.0f us quant=%.1f us tail=%.0f us (%lld calls)\n",
                wn, wn ? wns / 1e3 / wn : 0.0,
                (wall_ns[0] + wall_ns[1]) / 1e3 / tc,
                (quant_ns[0] + quant_ns[1]) / 1e3 / tc,
                (tail_ns[0] + tail_ns[1]) / 1e3 / tc, tc);
    }
};
Stats g_stats;

struct Pool {
    // pool
    std::vector<std::thread> threads;
    std::mutex mu;
    std::condition_variable cv;
    uint64_t gen = 0;
    bool quit = false;
    // sleepers: notify only when someone is asleep. (A spin-before-sleep variant, findings 206, measured
    // negative and was removed at the tag pre-elision-mtp-step-2026-10-09.)
    int sleepers = 0;
    int n_workers = 1;                       // total participants incl. caller

    // cache prefetch (claude dump findings 258): mode 1 = the workers read the planned byte ranges into their caches
    // while the GPU runs the next layer's attention; any compute call aborts it first
    int mode = 0;
    std::atomic<bool> pf_abort{false};
    std::atomic<int>  pf_ctr{0};
    std::vector<const char *> pf_ptr;        // one 64 KiB chunk each
    std::vector<int>          pf_len, pf_owner;
    std::atomic<int>          pf_done_by[64];   // chunks completed per planned expert
    void run_prefetch();

    // current job (valid while a phase runs)
    std::atomic<int> tile_ctr{0};
    std::atomic<int> tiles_done{0};
    std::atomic<int> active{0};              // workers currently inside run_tiles
    int n_tiles = 0;
    const moert_coldcpu_call * call = nullptr;
    const char * xq = nullptr;               // quantized activation(s)
    size_t xq_stride = 0;                    // bytes per activation column (0 = shared)
    const char * xp[PMAX] = {};              // pair i's quantized activation (distinct columns quantized once)
    ggml_vec_dot_t vec_dot = nullptr;
    size_t w_row = 0;                        // weight row bytes
    int tiles_per_group = 0;   // row-tiles per GROUP (was per pair; a group == one distinct expert)

    /// expert grouping for the classic per-kind job: pairs sharing one cold expert are
    // processed together so the weight row is decoded once for all of them. grp_pair holds pair
    // indices grouped contiguously; group g occupies [grp_start[g], grp_start[g+1]); grp_order
    // is the order groups are swept in (first appearance).
    int  grp_pair[PMAX] = {};
    int  grp_start[PMAX + 1] = {};
    int  grp_order[PMAX] = {};
    int  n_groups = 0;
    bool grp_fast = false;      // decodeN kernel applicable to this call (type/shape/AVX2 gated)
    int  grp_nsb  = 0;          // weight blocks per row the kernel loops over (256-superblocks / 32-blocks)
    ggml_type grp_type = GGML_TYPE_COUNT;

    // Build groups preserving FIRST-APPEARANCE order, so with no duplicates the sweep order is
    // byte-identical to the pre- identity order (the default path stays unchanged).
    void build_groups(const int32_t * cold_ids, int n_cold) {
        if (n_cold > PMAX) n_cold = PMAX;
        bool used[PMAX] = {};
        int n = 0, gi = 0;
        grp_start[0] = 0;
        for (int i = 0; i < n_cold; ++i) {
            if (used[i]) continue;
            grp_pair[n++] = i; used[i] = true;
            for (int j = i + 1; j < n_cold; ++j)
                if (!used[j] && cold_ids[j] == cold_ids[i]) { grp_pair[n++] = j; used[j] = true; }
            grp_start[++gi] = n;
        }
        n_groups = gi;
        for (int g = 0; g < n_groups; ++g) grp_order[g] = g;
    }

    // scratch (preallocated)
    std::vector<char> xq_buf;

    // One tile = (group, row range). Every pair in a group shares the SAME expert, so the tile
    // reads one weight row range and serves all of the group's pairs from it -- decoding once
    // when the fast kernel applies, or falling back to per-pair vec_dot when it does not.
    // Tiles are now NON-UNIFORM (a 3-pair group is ~3x a 1-pair group): the existing atomic
    // work-stealing loop absorbs that dynamically, which is why the tile counter is claimed per
    // tile rather than statically partitioned.
    void run_tiles() {
        const moert_coldcpu_call * c = call;
        const size_t as = c->stride ? (size_t) c->stride : (size_t) c->expert_bytes;   // v3 slot stride
        for (;;) {
            const int t = tile_ctr.fetch_add(1, std::memory_order_relaxed);
            if (t >= n_tiles) break;
            const int g  = grp_order[t / tiles_per_group];
            const int r0 = (t % tiles_per_group) * TILE_ROWS;
            const int r1 = r0 + TILE_ROWS < c->d_out ? r0 + TILE_ROWS : c->d_out;
            const int gs = grp_start[g], ge = grp_start[g + 1];
            const char * w = (const char *) c->w_host + (size_t) c->cold_ids[grp_pair[gs]] * as;

            if (!grp_fast) {
                for (int k = gs; k < ge; ++k) {            // generic path: ggml's vec_dot per pair and row
                    const int i = grp_pair[k];
                    const char * x = xp[i];   // stride 0 = shared column
                    float * y = c->y + (size_t) i * c->d_out;
                    for (int r = r0; r < r1; ++r) {
                        vec_dot(c->d_in, &y[r], 0, w + (size_t) r * w_row, 0, x, 0, 1);
                    }
                }
            } else {
                // our kernels (sidecar_kernels.hpp, bit-exact with ggml's): the group's pairs in chunks
                // of <= 4 activations, each weight row decoded once per chunk; a 1-pair chunk runs the
                // same kernel at N=1 (inlined, no function pointer per row).
                for (int b = gs; b < ge; b += 4) {
                    const int n = ge - b < 4 ? ge - b : 4;
                    const char * ys[4];
                    float      * yo[4];
                    for (int k = 0; k < n; ++k) {
                        const int i = grp_pair[b + k];
                        ys[k] = xp[i];
                        yo[k] = c->y + (size_t) i * c->d_out;
                    }
                    if (grp_type == GGML_TYPE_IQ2_XXS) {
                        const blk_q8_K * const * yk = (const blk_q8_K * const *) ys;
                        for (int r = r0; r < r1; ++r) {
                            const char * wr = w + (size_t) r * w_row;
                            float outs[4];
                            if      (n == 1) iq2xxs_row_decodeN<1>(wr, yk, outs, grp_nsb);
                            else if (n == 2) iq2xxs_row_decodeN<2>(wr, yk, outs, grp_nsb);
                            else if (n == 3) iq2xxs_row_decodeN<3>(wr, yk, outs, grp_nsb);
                            else             iq2xxs_row_decodeN<4>(wr, yk, outs, grp_nsb);
                            for (int k = 0; k < n; ++k) yo[k][r] = outs[k];
                        }
                    } else {
                        const blk_q8_0 * const * yk = (const blk_q8_0 * const *) ys;
                        for (int r = r0; r < r1; ++r) {
                            const char * wr = w + (size_t) r * w_row;
                            float outs[4];
                            if      (n == 1) iq4nl_row_decodeN<1>(wr, yk, outs, grp_nsb);
                            else if (n == 2) iq4nl_row_decodeN<2>(wr, yk, outs, grp_nsb);
                            else if (n == 3) iq4nl_row_decodeN<3>(wr, yk, outs, grp_nsb);
                            else             iq4nl_row_decodeN<4>(wr, yk, outs, grp_nsb);
                            for (int k = 0; k < n; ++k) yo[k][r] = outs[k];
                        }
                    }
                }
            }
            tiles_done.fetch_add(1, std::memory_order_acq_rel);
        }
    }

    void worker() {
        // P1.5: MOE_SIDECAR_PRIO=1|2 raises worker priority
        // (ABOVE_NORMAL|HIGHEST). The live gate_up straggler tail inflates ~6x under
        // ambient load (caller spin-waits the LAST worker; equal-priority browser/OS
        // threads preempt workers mid-tile). Workers only -- the caller (llama.cpp
        // compute thread) is not ours to retune. Default 0 = unchanged.
#if defined(_WIN32)
        {
            const char * e = std::getenv("MOE_SIDECAR_PRIO");
            const int prio = e ? atoi(e) : 0;
            if (prio == 1) SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_ABOVE_NORMAL);
            else if (prio == 2) SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);
        }
#endif
        uint64_t seen = 0;
        std::unique_lock<std::mutex> lk(mu);
        for (;;) {
            if (!quit && gen == seen) { ++sleepers; cv.wait(lk, [&] { return quit || gen != seen; }); --sleepers; }
            if (quit) return;
            seen = gen;
            const int m = mode;
            if (g_stats.pub_ns && m == 0) {            // stats: wake latency vs the job publish stamp
                g_stats.wake_ns.fetch_add(stats_now() - g_stats.pub_ns, std::memory_order_relaxed);
                g_stats.wake_n.fetch_add(1, std::memory_order_relaxed);
            }
            lk.unlock();
            active.fetch_add(1, std::memory_order_acq_rel);
            if (m == 1) run_prefetch(); else run_tiles();
            active.fetch_sub(1, std::memory_order_acq_rel);
            lk.lock();
        }
    }
};

// Read every planned chunk once (one 32-byte load per 64-byte line pulls the line into this core's L2, then the CCD's
// L3); stop at the first chunk after an abort. The loads feed a sink so the compiler keeps them.
void Pool::run_prefetch() {
    __m256i acc = _mm256_setzero_si256();
    const int n = (int) pf_ptr.size();
    for (;;) {
        if (pf_abort.load(std::memory_order_relaxed)) break;
        const int i = pf_ctr.fetch_add(1, std::memory_order_relaxed);
        if (i >= n) break;
        const char * b = pf_ptr[(size_t) i];
        const int len = pf_len[(size_t) i];
        for (int o = 0; o + 32 <= len; o += 64)
            acc = _mm256_xor_si256(acc, _mm256_loadu_si256((const __m256i *) (b + o)));
        pf_done_by[pf_owner[(size_t) i]].fetch_add(1, std::memory_order_relaxed);
    }
    static thread_local volatile int sink;
    sink = _mm256_extract_epi32(acc, 0) ^ _mm256_extract_epi32(acc, 5);
}

Pool * g_pool = nullptr;

}  // namespace

namespace moe_sidecar {

bool init(int n_workers) {
    if (g_pool) return true;
    ggml_cpu_init();
    build_tables();          // decodeN grid/sign tables; must follow ggml_cpu_init (uses to_float)
    Pool * p = new Pool();
    p->n_workers = n_workers < 1 ? 1 : (n_workers > 16 ? 16 : n_workers);
    p->xq_buf.resize((size_t) PMAX * ggml_row_size(GGML_TYPE_Q8_K, DMAX));
    for (int i = 1; i < p->n_workers; ++i) p->threads.emplace_back([p] { p->worker(); });
    {
        const char * e = std::getenv("MOE_SIDECAR_PRIO");
        if (e && atoi(e) != 0)
            fprintf(stderr, "moe_sidecar: worker priority raised (MOE_SIDECAR_PRIO=%s)\n", e);
    }
    g_pool = p;
    return true;
}

void shutdown() {
    Pool * p = g_pool;
    if (!p) return;
    g_pool = nullptr;
    { std::lock_guard<std::mutex> lk(p->mu); p->quit = true; }
    p->cv.notify_all();
    for (auto & t : p->threads) t.join();
    delete p;
}

}  // namespace moe_sidecar

extern "C" void moe_sidecar_compute(const moert_coldcpu_call * c) {
    Pool * p = g_pool;
    if (!p || !c || c->n_cold <= 0) return;
    const int  sk = (c->kind == 1) ? 1 : 0;                // stats kind bucket
    const long long s0 = stats_now();

    // A straggler from the previous call may still be draining its (already-empty) tile
    // loop; the job fields and xq scratch must not change under it. A cache prefetch still running stops at its next
    // 64 KiB chunk (findings 258).
    p->pf_abort.store(true, std::memory_order_relaxed);
    while (p->active.load(std::memory_order_acquire) != 0) SIDECAR_PAUSE();
    p->mode = 0;

    const auto * trw = ggml_get_type_traits_cpu((ggml_type) c->w_type_tag);
    if (!trw || !trw->vec_dot) {                           // unsupported type: zero contribution, loud in compare gates
        memset(c->y, 0, (size_t) c->n_cold * c->d_out * sizeof(float));
        return;
    }
    const ggml_type vdt = trw->vec_dot_type;
    const auto * trq = ggml_get_type_traits_cpu(vdt);
    const size_t qrow = ggml_row_size(vdt, c->d_in);

    // quantize the activation(s) once per call into preallocated scratch
    const long long q0 = stats_now();
    char * xq = p->xq_buf.data();
    if (c->x_col) {
        // x holds the distinct activation columns (the verify batch's tokens) and pair i uses column
        // x_col[i]: quantize each column once, not once per pair (gate_up: ~14-25 pairs, <= 4 tokens)
        bool done[PMAX] = {};
        for (int i = 0; i < c->n_cold; ++i) {
            const int col = c->x_col[i] < 0 ? 0 : (c->x_col[i] >= PMAX ? PMAX - 1 : c->x_col[i]);
            if (!done[col]) { trq->from_float(c->x + (size_t) col * c->d_in, xq + (size_t) col * qrow, c->d_in); done[col] = true; }
            p->xp[i] = xq + (size_t) col * qrow;
        }
        p->xq_stride = qrow;
    } else if (c->x_per_pair) {
        for (int i = 0; i < c->n_cold; ++i) {
            trq->from_float(c->x + (size_t) i * c->d_in, xq + (size_t) i * qrow, c->d_in);
            p->xp[i] = xq + (size_t) i * qrow;
        }
        p->xq_stride = qrow;
    } else {
        trq->from_float(c->x, xq, c->d_in);
        for (int i = 0; i < c->n_cold; ++i) p->xp[i] = xq;
        p->xq_stride = 0;                                  // all pairs share column 0
    }
    g_stats.quant_ns[sk] += stats_now() - q0;

    p->call = c;
    p->xq = xq;
    p->vec_dot = trw->vec_dot;
    p->w_row = ggml_row_size((ggml_type) c->w_type_tag, c->d_in);
    p->tiles_per_group = (c->d_out + TILE_ROWS - 1) / TILE_ROWS;
    p->build_groups(c->cold_ids, c->n_cold);
    p->n_tiles = p->n_groups * p->tiles_per_group;
    // fast path gate. x_per_pair==0 (one shared activation column) is the single-token
    // gate_up layout, where duplicates cannot arise (top-k is distinct within one token) --
    // excluded rather than special-cased, since decodeN's whole premise is N DIFFERENT
    // activations. Everything else falls back to the unchanged per-pair vec_dot loop.
    // Our kernels cover the two expert types the product runs: IQ2_XXS x Q8_K (gate_up) and IQ4_NL x
    // Q8_0 (down, an even block count); ggml's vec_dot for anything else.
    const ggml_type wt = (ggml_type) c->w_type_tag;
    const bool iq2 = wt == GGML_TYPE_IQ2_XXS && vdt == GGML_TYPE_Q8_K && (c->d_in % 256) == 0;
    const bool iq4 = wt == GGML_TYPE_IQ4_NL  && vdt == GGML_TYPE_Q8_0 && (c->d_in % 64) == 0;
    p->grp_type = wt;
    p->grp_nsb  = iq2 ? c->d_in / 256 : c->d_in / 32;
    p->grp_fast = g_tables_ok && (iq2 || iq4);
    p->tile_ctr.store(0, std::memory_order_relaxed);
    p->tiles_done.store(0, std::memory_order_relaxed);

    bool wake = false;
    { std::lock_guard<std::mutex> lk(p->mu); g_stats.pub_ns = stats_now(); ++p->gen;
      wake = p->sleepers > 0; }
    if (wake) p->cv.notify_all();
    p->run_tiles();                                        // the caller participates
    const long long t0 = stats_now();                      // stats: straggler tail the caller eats
    while (p->tiles_done.load(std::memory_order_acquire) < p->n_tiles) SIDECAR_PAUSE();
    const long long t1 = stats_now();
    g_stats.tail_ns[sk] += t1 - t0;
    g_stats.wall_ns[sk] += t1 - s0;
    ++g_stats.calls[sk];
    g_stats.pairs[sk]   += c->n_cold;
    g_stats.bytes[sk]   += (long long) c->n_cold * (long long) c->expert_bytes;
    g_stats.d_out[sk]    = c->d_out;
    g_stats.note_ids(c->il, sk, c->cold_ids, c->n_cold);
    g_stats.note_within_dup(sk, c->cold_ids, c->n_cold);
    if (p->grp_fast) {                       // decodes actually saved by grouping
        g_stats.grouped_calls[sk] += 1;
        g_stats.saved_decodes[sk] += c->n_cold - p->n_groups;
    }
}


// ---------------------------------------------------------------------------------------------------
// cold_ffn mode (findings 206): ONE asynchronous cold job per layer. The island's gate_up op submits it
// and returns; a dedicated coordinator thread waits for the D2H copies, runs the cold gate_up through the
// pool above (this thread as the caller), applies GELU-GLU exactly as ggml-cuda writes it, runs the cold
// down, and signals. The server thread is not involved until the down op waits -- by then the GPU has
// its hot gate_up, the GLU and its hot down queued, which is the overlap this mode exists for.
// One job in flight: the down op of layer il waits before the gate_up op of layer il+1 can submit.
// ---------------------------------------------------------------------------------------------------
namespace {

struct FfnCoordinator {
    std::thread th;
    std::mutex mu;
    std::condition_variable cv_job, cv_done;
    bool has_job = false, busy = false, quit = false;
    moert_coldffn_job job{};
    int32_t n_cold = 0;
    int64_t ns = 0;
    std::vector<float>   ygu, act;
    std::vector<int32_t> xcol;
};
// where a job's time goes, printed at exit (the coordinator itself is never destroyed: its thread lives
// for the process): waiting for the D2H, gate_up, GLU, down, and how long the down op waited on it.
struct FfnStats {
    long long jobs = 0, t_ready = 0, t_gu = 0, t_glu = 0, t_down = 0, pairs = 0, waits = 0, t_wait = 0;
    ~FfnStats() {
        if (jobs) fprintf(stderr, "[MOE_COLDFFN_STATS] jobs=%lld pairs=%lld (%.1f/job) | mean us: wait-D2H %.1f gate_up %.1f "
                                  "GLU %.1f down %.1f | down op blocked %.1f us/job\n", jobs, pairs, (double) pairs / jobs,
                          t_ready / 1e3 / jobs, t_gu / 1e3 / jobs, t_glu / 1e3 / jobs, t_down / 1e3 / jobs,
                          waits ? t_wait / 1e3 / waits : 0.0);
    }
};
FfnStats g_ffn_stats;

FfnCoordinator *            g_ffn = nullptr;
moe_sidecar::down_resolver_fn g_down_res = nullptr;
void (*g_ffn_guard)(int32_t, const int32_t *, int32_t) = nullptr;

// --- Cache prefetch of the next layer's likely cold experts (claude dump findings 258) --------------------------------
// MOE_DUAL_L3PF=<M>: when a layer's cold job ends, the pool reads the M experts most likely to be cold in the NEXT layer
// (the ones its last 4 jobs used, by frequency and recency) into the CPU caches, while the GPU runs that layer's attention
// and router -- the RAM is idle there. MOE_DUAL_L3PF_DRY=1 plans and measures without reading anything (the A arm).
// Exact: only reads. Measured at exit: the prediction (precision, coverage), the window and how much of the plan it read,
// and the cold job's time per expert bucketed by how many of its experts were fully prefetched -- the mechanism itself.
struct L3pf {
    static constexpr int NLX = 64, HIST = 4, B = 6;
    bool init = false, dry = false;
    int M = 0, n_layers = 0;
    std::vector<int32_t> hist[NLX][HIST];
    int hpos[NLX] = {};
    struct Src { const char * gu = nullptr; int64_t stride = 0, bytes = 0; } src[NLX];
    int plan_il = -1;
    std::vector<int32_t> plan;
    std::vector<int> plan_chunks;
    long long t_post = 0;
    long long posts = 0, planned = 0, chunks_planned = 0, chunks_done = 0, experts_full = 0, aborted = 0, window_ns = 0;
    long long jobs = 0, cold_distinct = 0, pred_hits = 0, hits_full = 0, hits_part = 0;
    long long b_n[B] = {}, b_ns[B] = {}, b_exp[B] = {}, b_gu_ns[B] = {}, b_dn_ns[B] = {};
    ~L3pf() {
        if (!init || !M || !jobs) return;
        fprintf(stderr, "[MOE_L3PF] %s, M %d | %lld plans for %lld jobs, %.2f experts each | window %.0f us mean | read %.1f%% of the "
                        "planned chunks (%.0f MiB/s in the window), %.2f experts fully read per plan, %lld plans cut short\n",
                dry ? "DRY (plan only)" : "ON", M, posts, jobs, posts ? (double) planned / posts : 0.0, posts ? window_ns / 1e3 / posts : 0.0,
                chunks_planned ? 100.0 * chunks_done / chunks_planned : 0.0,
                window_ns ? chunks_done * 65536.0 / 1048576.0 / (window_ns / 1e9) : 0.0, posts ? (double) experts_full / posts : 0.0, aborted);
        fprintf(stderr, "[MOE_L3PF] prediction: precision %.1f%% (planned experts the job used), coverage %.1f%% (cold experts that were "
                        "planned) | fully prefetched and used %.1f%% of the cold experts, partly %.1f%%\n",
                planned ? 100.0 * pred_hits / planned : 0.0, cold_distinct ? 100.0 * pred_hits / cold_distinct : 0.0,
                cold_distinct ? 100.0 * hits_full / cold_distinct : 0.0, cold_distinct ? 100.0 * hits_part / cold_distinct : 0.0);
        for (int b = 0; b < B; ++b)
            if (b_n[b])
                fprintf(stderr, "[MOE_L3PF] jobs with %d%s fully prefetched cold experts: %lld jobs, %.1f distinct cold, gate_up %.0f + down %.0f us, "
                                "%.1f us per cold expert\n", b, b == B - 1 ? "+" : "", b_n[b], (double) b_exp[b] / b_n[b],
                        b_gu_ns[b] / 1e3 / b_n[b], b_dn_ns[b] / 1e3 / b_n[b], b_exp[b] ? b_ns[b] / 1e3 / b_exp[b] : 0.0);
    }
};
L3pf g_pf;

static void pf_init() {
    if (g_pf.init) return;
    g_pf.init = true;
    const char * m = std::getenv("MOE_DUAL_L3PF");
    g_pf.M = m ? (std::max)(0, (std::min)(32, atoi(m))) : 0;
    const char * d = std::getenv("MOE_DUAL_L3PF_DRY");
    g_pf.dry = d && d[0] == '1';
    if (g_pf.M) fprintf(stderr, "moe_sidecar: cache prefetch of the next layer's cold experts %s, %d experts per layer (findings 258)\n",
                        g_pf.dry ? "DRY (plan + measure only)" : "ON", g_pf.M);
}

// before a job of layer il computes: settle the plan made for it and count what it got
static void pf_settle(int il, const int32_t * ids, int n, int32_t * full_hits) {
    *full_hits = 0;
    if (!g_pf.M) return;
    Pool * p = g_pool;
    if (p) {
        p->pf_abort.store(true, std::memory_order_relaxed);
        while (p->active.load(std::memory_order_acquire) != 0) SIDECAR_PAUSE();
    }
    int32_t d[PMAX]; int nd = 0;
    for (int i = 0; i < n && i < PMAX; ++i) {
        bool seen = false;
        for (int k = 0; k < nd; ++k) seen |= d[k] == ids[i];
        if (!seen) d[nd++] = ids[i];
    }
    ++g_pf.jobs; g_pf.cold_distinct += nd;
    if (g_pf.plan_il == il && !g_pf.plan.empty()) {
        g_pf.window_ns += stats_now() - g_pf.t_post;
        bool cut = false;
        for (size_t k = 0; k < g_pf.plan.size(); ++k) {
            const int done = (p && !g_pf.dry) ? p->pf_done_by[k].load(std::memory_order_relaxed) : 0;
            const bool full = done >= g_pf.plan_chunks[k];
            cut |= !full && !g_pf.dry;
            g_pf.chunks_done += done; g_pf.experts_full += full;
            bool used = false;
            for (int i = 0; i < nd; ++i) used |= d[i] == g_pf.plan[k];
            if (used) { ++g_pf.pred_hits; if (full) { ++g_pf.hits_full; ++*full_hits; } else if (done > 0) ++g_pf.hits_part; }
        }
        g_pf.aborted += cut;
    }
    g_pf.plan_il = -1;
    if (il >= 0 && il < L3pf::NLX) {   // history of this layer's distinct cold experts
        g_pf.hist[il][g_pf.hpos[il]].assign(d, d + nd);
        g_pf.hpos[il] = (g_pf.hpos[il] + 1) % L3pf::HIST;
        g_pf.n_layers = (std::max)(g_pf.n_layers, il + 1);
    }
}

// after a job of layer il: plan the next layer and, unless dry, start the pool reading it
static void pf_post(int il) {
    if (!g_pf.M || g_pf.n_layers < 2) return;
    const int nl = (il + 1) % g_pf.n_layers;
    const L3pf::Src & s = g_pf.src[nl];
    if (!s.gu) return;
    moe_sidecar::DownPool dn = {};
    if (!g_down_res || !g_down_res(nl, &dn)) return;
    // score: frequency over the last 4 jobs of that layer, the most recent weighing most
    static const float w[L3pf::HIST] = { 1.0f, 0.7f, 0.5f, 0.35f };
    std::vector<std::pair<float, int32_t>> sc;
    for (int a = 0; a < L3pf::HIST; ++a) {
        const int slot = (g_pf.hpos[nl] - 1 - a + 2 * L3pf::HIST) % L3pf::HIST;
        for (int32_t e : g_pf.hist[nl][slot]) {
            bool f = false;
            for (auto & x : sc) if (x.second == e) { x.first += w[a]; f = true; break; }
            if (!f) sc.push_back({ w[a], e });
        }
    }
    if (sc.empty()) return;
    std::stable_sort(sc.begin(), sc.end(), [](const auto & a, const auto & b) { return a.first > b.first; });
    const int m = (std::min<int>)(g_pf.M, (int) sc.size());
    Pool * p = g_pool;
    if (!p) return;
    while (p->active.load(std::memory_order_acquire) != 0) SIDECAR_PAUSE();   // no straggler reads the old plan
    g_pf.plan.clear(); g_pf.plan_chunks.assign((size_t) m, 0);
    p->pf_ptr.clear(); p->pf_len.clear(); p->pf_owner.clear();
    const int CH = 65536;
    auto add = [&](const char * b, int64_t bytes, int k) {
        for (int64_t o = 0; o < bytes; o += CH) {
            p->pf_ptr.push_back(b + o); p->pf_len.push_back((int) (std::min<int64_t>)(CH, bytes - o)); p->pf_owner.push_back(k);
            ++g_pf.plan_chunks[(size_t) k];
        }
    };
    for (int k = 0; k < m; ++k) { g_pf.plan.push_back(sc[(size_t) k].second); add(s.gu + (int64_t) sc[(size_t) k].second * s.stride, s.bytes, k); }
    const int64_t dn_stride = (int64_t) (dn.stride ? dn.stride : dn.bytes);
    for (int k = 0; k < m; ++k) add((const char *) dn.base + (int64_t) sc[(size_t) k].second * dn_stride, (int64_t) dn.bytes, k);
    for (int k = 0; k < m; ++k) p->pf_done_by[k].store(0, std::memory_order_relaxed);
    g_pf.plan_il = nl; g_pf.t_post = stats_now();
    ++g_pf.posts; g_pf.planned += m; g_pf.chunks_planned += (long long) p->pf_ptr.size();
    if (g_pf.dry) return;
    p->pf_ctr.store(0, std::memory_order_relaxed);
    p->pf_abort.store(false, std::memory_order_relaxed);
    bool wake = false;
    { std::lock_guard<std::mutex> lk(p->mu); p->mode = 1; ++p->gen; wake = p->sleepers > 0; }
    if (wake) p->cv.notify_all();
}


// ggml_cuda_op_gelu_single, written the same way (the GPU GLU it replaces for the cold pairs)
static inline float ffn_gelu(float x) {
    const float GELU_COEF_A    = 0.044715f;
    const float SQRT_2_OVER_PI = 0.79788456080286535587989211986876f;
    return 0.5f * x * (1.0f + tanhf(SQRT_2_OVER_PI * x * (1.0f + GELU_COEF_A * x * x)));
}

// 8-wide exp (Cephes polynomial, as in avx_mathfun: ~1 ulp on the range used here) for the GELU's tanh.
static inline __m256 ffn_exp8(__m256 x) {
    x = _mm256_min_ps(x, _mm256_set1_ps(88.3762626647949f));
    x = _mm256_max_ps(x, _mm256_set1_ps(-88.3762626647949f));
    __m256 fx = _mm256_fmadd_ps(x, _mm256_set1_ps(1.44269504088896341f), _mm256_set1_ps(0.5f));
    fx = _mm256_floor_ps(fx);
    x = _mm256_fnmadd_ps(fx, _mm256_set1_ps(0.693359375f), x);
    x = _mm256_fnmadd_ps(fx, _mm256_set1_ps(-2.12194440e-4f), x);
    const __m256 z = _mm256_mul_ps(x, x);
    __m256 y = _mm256_set1_ps(1.9875691500e-4f);
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(1.3981999507e-3f));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(8.3334519073e-3f));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(4.1665795894e-2f));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(1.6666665459e-1f));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(5.0000001201e-1f));
    y = _mm256_fmadd_ps(y, z, _mm256_add_ps(x, _mm256_set1_ps(1.0f)));
    const __m256i e = _mm256_slli_epi32(_mm256_add_epi32(_mm256_cvttps_epi32(fx), _mm256_set1_epi32(0x7f)), 23);
    return _mm256_mul_ps(y, _mm256_castsi256_ps(e));
}

// gelu(g) * u for 8 lanes, the tanh form ggml-cuda uses; tanh(v) = sign(v) (1 - e) / (1 + e), e = exp(-2|v|)
static inline __m256 ffn_geglu8(__m256 g, __m256 u) {
    const __m256 v = _mm256_mul_ps(_mm256_mul_ps(_mm256_set1_ps(0.79788456080286535587989211986876f), g),
                                   _mm256_fmadd_ps(_mm256_mul_ps(_mm256_set1_ps(0.044715f), g), g, _mm256_set1_ps(1.0f)));
    const __m256 sign = _mm256_and_ps(v, _mm256_castsi256_ps(_mm256_set1_epi32((int) 0x80000000u)));
    const __m256 av = _mm256_andnot_ps(_mm256_castsi256_ps(_mm256_set1_epi32((int) 0x80000000u)), v);
    const __m256 e = ffn_exp8(_mm256_mul_ps(_mm256_set1_ps(-2.0f), av));
    const __m256 th = _mm256_or_ps(sign, _mm256_div_ps(_mm256_sub_ps(_mm256_set1_ps(1.0f), e), _mm256_add_ps(_mm256_set1_ps(1.0f), e)));
    const __m256 gl = _mm256_mul_ps(_mm256_mul_ps(_mm256_set1_ps(0.5f), g), _mm256_add_ps(_mm256_set1_ps(1.0f), th));
    return _mm256_mul_ps(gl, u);
}

void ffn_run(FfnCoordinator * f) {
    moert_coldffn_job & j = f->job;
    const long long t0 = stats_now();
    j.wait_ready(j.ev);
    const long long t1 = stats_now();
    g_ffn_stats.jobs++; g_ffn_stats.t_ready += t1 - t0;
    int n = j.h_counts[1];
    n = n < 0 ? 0 : (n > PMAX ? PMAX : n);
    f->n_cold = n;
    pf_init();
    if (g_pf.M && j.il >= 0 && j.il < L3pf::NLX)
        g_pf.src[j.il] = { (const char *) j.w_gu, (int64_t) (j.gu_stride ? j.gu_stride : j.gu_bytes), (int64_t) j.gu_bytes };
    int32_t full_hits = 0;
    pf_settle(j.il, j.h_ids, n, &full_hits);
    if (n == 0) return;
    if (g_ffn_guard) g_ffn_guard(j.il, j.h_ids, n);
    f->xcol.assign(PMAX, 0);
    for (int i = 0; i < n; ++i) f->xcol[i] = j.n_used > 0 ? j.h_cdst[i] / j.n_used : 0;
    f->ygu.resize((size_t) n * j.d_gu);
    moert_coldcpu_call c0 = {};
    c0.il = j.il; c0.kind = 0; c0.n_cold = n; c0.cold_ids = j.h_ids;
    c0.x = j.x; c0.x_per_pair = 1; c0.x_col = f->xcol.data();
    c0.w_host = j.w_gu; c0.expert_bytes = j.gu_bytes; c0.stride = j.gu_stride; c0.w_type_tag = j.gu_type;
    c0.d_in = j.d_in; c0.d_out = j.d_gu; c0.y = f->ygu.data();
    moe_sidecar_compute(&c0);
    const long long t2 = stats_now();
    g_ffn_stats.t_gu += t2 - t1; g_ffn_stats.pairs += n;
    const int nff = j.d_gu / 2;   // gate rows [0, nff), up rows [nff, 2 nff) -- build_moe_ffn's two views
    f->act.resize((size_t) n * nff);
    for (int i = 0; i < n; ++i) {
        const float * g = f->ygu.data() + (size_t) i * j.d_gu;
        float * a = f->act.data() + (size_t) i * nff;
        int k = 0;
        for (; k + 8 <= nff; k += 8)   // 8-wide: the scalar tanhf was ~120 us per layer at 25 cold pairs
            _mm256_storeu_ps(a + k, ffn_geglu8(_mm256_loadu_ps(g + k), _mm256_loadu_ps(g + nff + k)));
        for (; k < nff; ++k) a[k] = ffn_gelu(g[k]) * g[nff + k];
    }
    const long long t3 = stats_now();
    g_ffn_stats.t_glu += t3 - t2;
    moe_sidecar::DownPool d = {};
    if (!g_down_res || !g_down_res(j.il, &d) || d.d_in != nff) {
        static bool warned = false;
        if (!warned) { warned = true; fprintf(stderr, "moe_sidecar: CRITICAL: cold FFN has no down pool for layer %d -- zeros\n", j.il); }
        std::memset(j.y, 0, (size_t) n * 4096 * sizeof(float));
        return;
    }
    moert_coldcpu_call c1 = {};
    c1.il = j.il; c1.kind = 1; c1.n_cold = n; c1.cold_ids = j.h_ids;
    c1.x = f->act.data(); c1.x_per_pair = 1;
    c1.w_host = d.base; c1.expert_bytes = d.bytes; c1.stride = d.stride; c1.w_type_tag = d.type;
    c1.d_in = d.d_in; c1.d_out = d.d_out; c1.y = j.y;
    moe_sidecar_compute(&c1);
    const long long t4 = stats_now();
    g_ffn_stats.t_down += t4 - t3;
    if (g_pf.M) {   // the mechanism: job time per cold expert, by how many of them were fully prefetched
        int nd = 0;
        for (int i = 0; i < n; ++i) { bool s = false; for (int k = 0; k < i; ++k) s |= j.h_ids[k] == j.h_ids[i]; nd += !s; }
        const int b = (std::min)(full_hits, L3pf::B - 1);
        ++g_pf.b_n[b]; g_pf.b_exp[b] += nd; g_pf.b_ns[b] += (t2 - t1) + (t4 - t3);
        g_pf.b_gu_ns[b] += t2 - t1; g_pf.b_dn_ns[b] += t4 - t3;
    }
}

void ffn_loop(FfnCoordinator * f) {
    std::unique_lock<std::mutex> lk(f->mu);
    for (;;) {
        f->cv_job.wait(lk, [&] { return f->quit || f->has_job; });
        if (f->quit) return;
        f->has_job = false;
        lk.unlock();
        const long long t0 = stats_now();
        ffn_run(f);
        const long long t1 = stats_now();
        lk.lock();
        f->ns = t1 - t0;
        f->busy = false;
        f->cv_done.notify_all();
        if (g_pf.M && !f->has_job) {   // findings 258: read the next layer's likely cold experts while the GPU works
            const int il = f->job.il;
            lk.unlock();
            pf_post(il);
            lk.lock();
        }
    }
}

}  // namespace

namespace moe_sidecar {
void set_down_resolver(down_resolver_fn fn) { g_down_res = fn; }
void set_cold_guard(void (*fn)(int32_t, const int32_t *, int32_t)) { g_ffn_guard = fn; }
}  // namespace moe_sidecar

extern "C" void moe_sidecar_ffn_submit(const moert_coldffn_job * job) {
    if (!g_ffn) {
        g_ffn = new FfnCoordinator();
        g_ffn->th = std::thread([f = g_ffn] { ffn_loop(f); });
#if defined(_WIN32)
        SetThreadPriority(g_ffn->th.native_handle(), THREAD_PRIORITY_ABOVE_NORMAL);
#endif
    }
    FfnCoordinator * f = g_ffn;
    std::unique_lock<std::mutex> lk(f->mu);
    f->cv_done.wait(lk, [&] { return !f->busy; });   // never waits in practice: the down op waited
    f->job = *job;
    f->has_job = true;
    f->busy = true;
    // findings 258: a running prefetch is NOT stopped here -- the job first waits for its D2H (~0.2 ms); pf_settle stops
    // it when the activations are ready
    f->cv_job.notify_one();
}

extern "C" int32_t moe_sidecar_ffn_wait(int32_t /*il*/, int64_t * cpu_ns) {
    FfnCoordinator * f = g_ffn;
    if (!f) return 0;
    const long long w0 = stats_now();
    std::unique_lock<std::mutex> lk(f->mu);
    f->cv_done.wait(lk, [&] { return !f->busy; });
    g_ffn_stats.waits++; g_ffn_stats.t_wait += stats_now() - w0;
    if (cpu_ns) *cpu_ns = f->ns;
    return f->n_cold;
}
