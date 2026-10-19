// moert/cuda — the MoE CUDA island.
//
// The selective dual-precision MoE matmul that used to live inside the ggml custom op
// (ggml-cuda/moe-dual.cu): partition routed pairs by residency, run the hot (K-slot Q4
// pool) and cold (full low-precision pool) matmuls each over its own compacted pair
// list, scatter back to routed-pair order with the down .scale applied exactly once.
// Plus the island's host services: the device routing counters (native observation) and
// the in-place page-lock of the cold arena.
//
// This header is the device-pointer ABI: plain C structs, raw device pointers, an
// opaque stream — no ggml types. The two things the island still borrows from ggml
// (the tuned quantized mul_mat_id kernel and the backend's VRAM scratch pool) are
// reached through the moert_ffn_moe_host vtable, implemented by the GGML_OP_MOE_DUAL
// trampoline (the only remaining in-tree code). The island is linked INTO ggml-cuda
// (one copy of its state, next to the kernels that feed it); host-process callers
// (moe_dual.cpp) reach it through ggml-cuda's exported forwarders, never by linking
// this library directly.

#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------------------------
// Configuration: the island reads NO environment variables; the host shim
// parses the MOE_DUAL_* compat surface once and pushes it here before the first fire).
// Zero-initialized defaults == the historical env-unset behavior, except `counted`
// (the Slice-2 device-counted guard) which defaults ON — use moert_cuda_config_defaults.
// ---------------------------------------------------------------------------------

// Hard architectural ceiling on routed pairs per fire: the partition is ONE 32-lane warp,
// one lane per routed pair (moe_dual_partition_kernel), so Pmax = top_k * n_tokens can never
// exceed 32 — spec_kmax is clamped to 4 in the host shim for exactly this reason, the
// trampoline asserts it, and every pinned/scratch/telemetry buffer in the cold path is sized
// from it. It is the maximum a STATIC allocation must cover.
#define MOERT_COLDCPU_GS_PMAX_MAX 32

typedef struct moert_cuda_config {
    int32_t counted;        // device-counted live-channel guard (skip padded [n_live,Pmax)
                            // matmul channels). Default ON; 0 = Slice-1 padded path. (MOE_DUAL_COUNTED)
    int32_t copy_stream;    // Plan-C RAM-cold path: route the SAME device-partitioned
                            // cold list onto a dedicated copy stream, with the cold-list D2H sync
                            // gated on the partition kernel alone (not the hot matmul), so the pinned
                            // H2D copies overlap compute and the host stops draining the compute
                            // stream every op. Bit-identical bytes. 0 = off (DEFAULT). (MOE_DUAL_COPYSTREAM)
    int32_t coldcpu;        // CPU cold sidecar (the product's cold executor): in the Plan-C
                            // staging path, fires with n_cold <= coldcpu_max compute their cold pairs
                            // ON THE HOST from the RAM cold pool (weights never cross PCIe; the small
                            // activation goes D2H inside the sync the path already pays, and the
                            // per-pair outputs come back H2D into y_cold). Above the threshold the op
                            // falls back to the weight-staging ring (correct, counted). Requires
                            // coldcpu_fn. 0 = off (DEFAULT). (MOE_DUAL_COLD_CPU)
    int32_t coldcpu_max;    // per-op n_cold sidecar threshold (MOE_DUAL_COLD_CPU_MAX, default 3)
    uint64_t coldcpu_fn;    // host callback (moert_coldcpu_fn as uintptr), 0 = off
    uint64_t cold_guard_fn; // host callback (moert_cold_guard_fn as uintptr), 0 = off. Called on the
                            // RAM-cold path right after the cold list reaches the host and BEFORE any
                            // read of the cold pool (CPU sidecar or staging ring), with the layer and
                            // its cold expert ids: the exclusive arena blocks there until every one of
                            // them has a resident page (full-IQ2 plan phase 2, MOE_DUAL_ARENA_EXCLUSIVE).
    int32_t cold_ffn;       // ONE asynchronous cold job per layer (findings 206, the host sets it with coldcpu): the gate_up
                            // op sends the cold list + token columns D2H and submits the whole cold FFN
                            // (gate_up -> GELU-GLU -> down) to the host WITHOUT blocking; the down op waits
                            // for it. Requires coldcpu and the two callbacks below. 0 = the per-kind synchronous sidecar.
    uint64_t coldffn_submit_fn;   // host callback (moert_coldffn_submit_fn as uintptr)
    uint64_t coldffn_wait_fn;     // host callback (moert_coldffn_wait_fn as uintptr)
} moert_cuda_config;
typedef void (*moert_cold_guard_fn)(int32_t il, const int32_t * ids, int32_t n);

// CPU cold sidecar callback: compute y[:,i] = W[cold_ids[i]] · x_i on the HOST for the
// n_cold compacted cold pairs of one (layer, kind) fire. `x` is a pinned host buffer:
// x_per_pair==0 -> one shared column [d_in] (gate_up decode); ==1 -> compacted per-pair
// columns [d_in, n_cold] (down). `w_host` is the RAM cold pool base (uniform expert_bytes
// stride — the same source the staging ring copies from). Output `y` is a pinned host
// buffer [d_out, n_cold], RAW matmul outputs (no .scale, no router weight — the scatter
// and the downstream graph apply those exactly as for the GPU cold matmul). Synchronous;
// internally threaded by the host implementation.
typedef struct moert_coldcpu_call {
    int32_t         il, kind;
    int32_t         n_cold;
    const int32_t * cold_ids;   // host, compaction order
    const float   * x;          // host activations (see x_per_pair)
    int32_t         x_per_pair;
    const void    * w_host;     // cold pool host base
    uint64_t        expert_bytes;
    uint64_t        stride;     // arena v3 addressing stride (base + e*stride); 0 => use expert_bytes
    int32_t         w_type_tag; // opaque ggml type of the pool
    int32_t         d_in, d_out;
    float         * y;          // out [d_out, n_cold], host
    const int32_t * x_col;      // optional: pair i reads column x_col[i] of x (x = the distinct token
                                // columns; gate_up under MTP). NULL -> x_per_pair as above.
} moert_coldcpu_call;
typedef void (*moert_coldcpu_fn)(const moert_coldcpu_call * call);

// One layer's cold FFN for the host (cold_ffn mode). Everything pinned is valid once wait_ready(ev)
// returns (the D2H copies the gate_up op queued). The host computes, per cold pair p (pair p belongs to
// token p / n_used): g = W_gu[id] x_tok (d_gu = 2*n_ff rows: gate then up), a = gelu(gate) * up,
// y = W_down[id] a -- RAW down outputs [d_out_down, n_cold] in `y`, no .scale and no router weight (the
// down op's scatter and the graph apply those). The down pool is resolved on the host by layer.
typedef struct moert_coldffn_job {
    int32_t         il;
    int32_t         n_used, n_tokens;
    const int32_t * h_counts;          // [0] n_hot, [1] n_cold
    const int32_t * h_ids;             // cold expert ids, compaction order
    const int32_t * h_cdst;            // cold pair index p, same order
    const float   * x;                 // token columns [d_in, n_tokens]
    const void    * w_gu;              // gate_up cold pool host base
    uint64_t        gu_bytes, gu_stride;
    int32_t         gu_type, d_in, d_gu;
    float         * y;                 // out
    void          (*wait_ready)(void * ev);
    void          * ev;
} moert_coldffn_job;
typedef void    (*moert_coldffn_submit_fn)(const moert_coldffn_job * job);     // copies the job, returns at once
typedef int32_t (*moert_coldffn_wait_fn)(int32_t il, int64_t * cpu_ns);         // blocks; n_cold computed

// Header-inline so the host process can fill defaults WITHOUT linking moert-cuda (the
// island's single copy of state lives inside ggml-cuda; see the note at the top).
static inline void moert_cuda_config_defaults(moert_cuda_config * cfg) {
    cfg->counted   = 1;   // device-counted live-channel guard: on unless explicitly disabled
    cfg->copy_stream = 0;
    cfg->coldcpu = 0;
    cfg->coldcpu_max = 3;
    cfg->coldcpu_fn = 0;
    cfg->cold_guard_fn = 0;
    cfg->cold_ffn = 0;
    cfg->coldffn_submit_fn = 0;
    cfg->coldffn_wait_fn = 0;
}

// Push the config (copied). Call before the first decode; later calls apply immediately
// except where state was already created from the old value (counters, the copy stream).
void moert_cuda_configure(const moert_cuda_config * cfg);

// ---------------------------------------------------------------------------------
// The fused op. Called per (layer, kind) fire by the GGML_OP_MOE_DUAL trampoline.
// ---------------------------------------------------------------------------------

// Which weights the host-side matmul wrapper should bind.
enum {
    MOERT_MM_W_HOT      = 0,   // the op's hot pool tensor (K slots)
    MOERT_MM_W_COLD_SRC = 1,   // the op's cold source tensor (planc_mode 0: device-resident pool)
    MOERT_MM_W_RAW      = 2,   // a raw device view: {w_type_tag, w_data, w_n} below
};

// One quantized mul_mat_id: y[d_out, i] = W[ids[i]] · b[:, col(i)] for the n_ids channels,
// with the live-channel guard `ne_live` (device int32, may be NULL) skipping padded columns.
typedef struct moert_mm_call {
    int32_t      w_sel;        // MOERT_MM_W_*
    int32_t      w_type_tag;   // MOERT_MM_W_RAW only: opaque ggml type tag of the weights
    const void * w_data;       // MOERT_MM_W_RAW only: device base of the expert pool
    int64_t      w_n;          // MOERT_MM_W_RAW only: expert slices in that pool
    const void * b;            // F32 activations [d_in, b_ncols], contiguous, device
    int64_t      b_ncols;      // 1 (single shared column) or Pmax (gathered per-pair columns)
    const void * ids;          // I32 [n_ids] channel -> expert/slot selector, device
    int64_t      n_ids;        // Pmax (launch shape; padded tail carries in-range id 0)
    void *       y;            // F32 [d_out, n_ids] output, device
    const int32_t * ne_live;   // device live-channel count (counted guard) or NULL
    // Variant-F de-dup (MOERT_MM_W_RAW only): optional per-expert weight-base table (device array
    // indexed by expert id == ids[i]), or NULL. When set, the matmul reads each channel's weights
    // from w_ptrs[ids[i]] instead of w_data + ids[i]*expert_bytes — lets the cold pool be a set of
    // individually mapped (scattered-devptr) regions. w_data is ignored when w_ptrs is non-NULL.
    const void * const * w_ptrs;
} moert_mm_call;

// Host services the island borrows from the embedding backend (ggml): VRAM scratch with
// strict LIFO free order (the ggml pool contract), and the tuned quantized mul_mat_id.
typedef struct moert_ffn_moe_host {
    void *  user;
    void * (*pool_alloc)(void * user, uint64_t nbytes);   // device scratch; freed LIFO
    void   (*pool_free)(void * user, void * ptr);         // reverse alloc order, same fire
    void   (*mm)(void * user, const moert_mm_call * call);
} moert_ffn_moe_host;

typedef struct moert_ffn_moe_args {
    int32_t kind;            // 0 = merged gate_up, 1 = down
    int32_t top_k;           // router top-k (metadata; n_used is the launch-shaping count)
    int32_t il;              // layer index (planner/counters/trace)
    int32_t planc_mode;      // cold source: 0 = cold tensor (device pool) / 1 = zero-copy
                             // mapped device ptr / 2 = staging ring (host base ptr)
    int32_t n_used;          // routes per token (ids->ne[0])
    int32_t n_tokens;        // tokens in this fire (decode: 1); Pmax = n_used*n_tokens <= 32
    int32_t ids_s1;          // ids ROW STRIDE in int32 elements (ids->nb[1]/4). NOT n_used:
                             // ggml_argsort_top_k returns a VIEW over the argsort result with
                             // the parent's row stride (n_expert), so token t's ids start at
                             // t*ids_s1. Flat ids[p] indexing is correct only at n_tokens==1.
    int32_t d_in, d_out;     // per-expert matmul shape
    int32_t n_expert;        // cold pool expert count
    int32_t k;               // hot pool slot count
    int32_t cold_type_tag;   // opaque ggml type of the cold pool (raw views + ring stride)
    uint64_t expert_bytes;   // bytes per cold expert (host computed: row_size(type,d_in)*d_out)
    const void * cur;        // F32 activations, device
    const void * ids;        // I32 [n_used, n_tokens] routed expert ids, device (row stride ids_s1)
    const void * exp_slot;   // I32 [n_expert] expert -> slot or -1, device
    const void * slot_exp;   // I32 [k] slot -> expert or -1, device
    const void * slot_state; // I32 [k] moert_slot_state, device
    const void * s_hot;      // F32 [k] down per-slot .scale, device, or NULL
    const void * s_cold;     // F32 [n_expert] down per-expert .scale, device (mode 0), or NULL
    uint64_t w_cold_dev;     // mode 1: mapped device base; mode 2: HOST base of the cold pool
    uint64_t s_cold_dev;     // ditto for the cold .scale pool (0 = none)
    int32_t  has_w_cold;     // mode 0: the trampoline holds a cold source tensor
    void *   dst;            // F32 [d_out, n_used*n_tokens] output, device
    void *   stream;         // cudaStream_t the surrounding graph runs on
} moert_ffn_moe_args;

void moert_ffn_moe(const moert_ffn_moe_args * args, const moert_ffn_moe_host * host);

// ---------------------------------------------------------------------------------
// Host services (reached through ggml-cuda's exported forwarders; same contracts as
// the pre-M2 ggml_backend_cuda_moe_dual_* functions they back).
// ---------------------------------------------------------------------------------

// Device routing counters (native observation). Layout of `dst` for fetch:
// [n_layers][n_expert] routed-pair counts then [n_layers][2] hot/cold pair counts.
// fetch is a synchronous D2H on the legacy default stream — refresh points only.
int32_t moert_cuda_route_counts_enable(int32_t n_layers, int32_t n_expert);
int32_t moert_cuda_route_counts_fetch(int32_t * dst, int32_t reset);

// 3b instant pin: page-lock a host buffer in place (cudaHostRegisterDefault, no mapping) so H2D
// staging from it is async + fast. 1 = pinned (or already registered), 0 = failed. Undo with the
// unregister below. Used to pin the whole owned IQ2 cold arena at load, before any prefill.
int32_t moert_cuda_host_register_pinned(void * base, uint64_t size);
void    moert_cuda_host_unregister(void * base);
// Mapped registration (a kernel reads the range through its device alias; counts in WDDM's shared budget). Findings 233.
int32_t moert_cuda_host_register_mapped(void * base, uint64_t size);
// Register [base, base + size) mapped (widened to 64 KiB pages) and return the DEVICE alias of base, or NULL. Findings 248.
void *  moert_cuda_host_map(void * base, uint64_t size);

// Cumulative counters of the fused op since process start (2026-10-03, the decode-step breakdown of
// the system model): the same numbers the [MOE_DUAL_COLDCPU] / [MOE_DUAL_SEAM] exit lines print,
// readable at any time (a server stopped from outside never prints its exit lines).
typedef struct moert_cuda_stats {
    long long fires;            // fused-op calls (one per layer and kind per forward)
    long long op_host_ns;       // host wall time spent inside moert_ffn_moe
    long long seam_syncs;       // host blocks on the cold-list D2H (RAM-cold path)
    long long seam_ns;
    long long coldcpu_calls;    // fires served by the CPU cold sidecar
    long long coldcpu_pairs;
    long long coldcpu_ns;       // host wall time inside the sidecar compute
    long long coldcpu_over_max; // fires above the sidecar threshold (weight-staging ring)
} moert_cuda_stats;
void moert_cuda_stats_get(moert_cuda_stats * out);

// BATCH TRANSPORT (findings 216k-216l, opt-in MOE_DUAL_RAM_DIR=1). The asynchronous promotions' host->VRAM copies
// go on a copy stream of their own and are issued at once; completion is tracked by events, never waited for.
//   async: issue one copy now (registered sources; a pageable source <= 4 KiB is staged into pinned memory; 0 =
//          refused, the caller copies itself); mark: an event after the copies issued so far, returns its id
//          (increasing); poll: the highest id whose copies have landed (wait_all: synchronize first). Thread-safe.
//   SM transport (sm_blocks > 0, claude dump findings 231): the copies are a kernel with sm_blocks blocks on a
//          lowest-priority stream, out of the copy engine's FIFO, reading a pinned ring of sm_ring_mib the worker fills
//          (a registered source is read in place) with sm_stage_threads threads (findings 233).
int32_t moert_cuda_dir_init(int32_t sm_blocks, int32_t sm_ring_mib, int32_t sm_stage_threads);   // sm_blocks 0 = copy engine
int32_t moert_cuda_dir_async(void * dst_dev, const void * src, uint64_t nbytes);
int64_t moert_cuda_dir_mark(void);
int64_t moert_cuda_dir_poll(int32_t wait_all);
// Residency publication in one call (findings 216l): n host->device copies from pageable host views (staged into
// pinned memory during the call: the views may change right after), then synchronized.
int32_t moert_cuda_publish_h2d(int32_t n, void * const * dsts, const void * const * srcs, const uint64_t * sizes);
typedef struct moert_cuda_dir_stats {
    long long async_copies, async_marks, bytes;         // copies issued, completion marks, bytes
    long long pub_calls, pub_copies;                    // publication calls and their copies
    long long sm_copies, sm_fallback;                   // SM transport (MOE_DUAL_DIR_SM=1): kernel copies, engine fallbacks
    long long sm_timed, sm_gpu_ns, sm_gpu_ns_max;       // launches with both GPU stamps, their summed and max duration
    long long sm_ring_copies, sm_ring_waits; double sm_ring_wait_ms, sm_stage_ms;   // worker waits for a ring span; host time copying into it
} moert_cuda_dir_stats;
void    moert_cuda_dir_stats_get(moert_cuda_dir_stats * out);
// Early-decision gate (findings 234-235): 0 enable (arg = last MoE layer | per_layer << 16), 1 arm -> target,
// 2 wait((layer << 40) | target), 3 open (host),
// 4 stats (arg 0 waits, 1 timeouts, 2 wait us), 5 engine source on (arg = pace), 6 engine stats (0 copies, 1 waits, 2 us).
int64_t moert_cuda_dir_gate(int32_t op, int64_t arg);
// Loaned slots (claude dump findings 246): the hot pool of (il, kind) is a per-slot table of n device pointers (the
// slab slots, then the loaned ones lent from another buffer); the hot matmul reads each slot through it. n = 0
// clears it (the hot matmul reads the contiguous pool tensor). `dev_ptrs` is a DEVICE array owned by the host.
void    moert_cuda_set_hot_table(int32_t il, int32_t kind, const void * const * dev_ptrs, int32_t n, int32_t type_tag);

#ifdef __cplusplus
}
#endif
