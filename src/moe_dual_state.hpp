#pragma once

// ---------------------------------------------------------------------------
// moe_dual INTERNAL shared state + cross-TU plumbing declarations.
// ---------------------------------------------------------------------------
// The moe_dual module is split into three translation units that all operate on
// the single process-lifetime `State g`:
//   moe_dual.cpp           the graph hook + decode-time runtime (refresh / telemetry)
//                          + the ggml/file plumbing callbacks
//   moe_dual_init.cpp      init_dynamic (pool + memmgr + policy construction, env knob
//                          parsing, admission preflight)
//   moe_dual_q4stream.cpp  the full-layer Q4Stream prefill staging

// This header is private to those files — the public API is moe_dual.hpp.
// ---------------------------------------------------------------------------

#include "moert/policy.h"
#include "moert/memmgr.h"
#include "moert/store.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace moe_dual {
void forecast_init();   // moe_dual.cpp: route trace, batch transport, MTP forecaster (all opt-in), after the policy exists

struct State {
    ggml_context *        pack_ctx = nullptr;  // hot expert tensors (data in pack_buf)
    ggml_context *        aux_ctx  = nullptr;  // hotmap / residency tensors
    ggml_backend_t        backend  = nullptr;
    ggml_backend_buffer_t pack_buf = nullptr;
    ggml_backend_buffer_t aux_buf  = nullptr;
    int n_layers = 0, n_expert = 0, k = 0;   // k = the uniform hot capacity of every layer IN THE SLAB
    // Loaned slots (claude dump findings 246, opt-in MOE_DUAL_LOAN=1): `loan` more slots per layer whose weights
    // live in the prefill staging reserve (idle between prompts). The policy and the residency mirrors see
    // k + loan slots; the slab holds k. Before a streamed prefill the loaned slots are revoked (evicted, the
    // policy limited to k), after it they are lent again. `loan_live` = the policy may use them now.
    int  loan = 0;
    bool loan_live = false;
    ggml_context *        hot_tab_ctx = nullptr;   // slot -> device pointer tables of the hot pools (with a loan)
    ggml_backend_buffer_t hot_tab_buf = nullptr;
    // Seam telemetry companions: the per-fire
    // n_cold histogram is only interpretable against the capacity that produced it, so the
    // hot-slab ladder's own events are counted here and reported alongside it.
    long long slab_rebuilds = 0;   // slab_rebuild_at_k() calls that actually rebuilt
    long long k_changes     = 0;   // subset of the above where the uniform K value changed
    // kind: 0 = merged gate_up, 1 = down. (kinds 2/3 — separate gate/up — were dead code on
    // gemma4 and were deleted together with their llama-graph call sites.)
    std::vector<std::array<ggml_tensor *, 2>> w_hot;   // [il][kind] hot weight
    std::vector<std::array<ggml_tensor *, 2>> s_hot;   // [il][kind] hot per-expert scale (may be null)
    std::vector<ggml_tensor *> hotmap;                 // [il] i32[n_expert]: global id -> local hot slot (0 if cold)
    std::vector<ggml_tensor *> resid;                  // [il] f32[n_expert]: 1.0 hot / 0.0 cold

    // Residency mirror for the fused selective op (ggml_moe_dual; Phase 3b). i32 device tensors in
    // the same backend. exp_slot[e]=slot or -1; slot_exp[slot]=expert or -1; slot_state[slot]=SlotState.
    // Kept in step with the policy's page table by push_residency at every commit.
    std::vector<ggml_tensor *> es_t, se_t, ss_t;


    // --- dynamic cache: a capacity-K VRAM pool per layer, filled causally from observed decode
    //     routing. Hot weights are streamed on promotion from a Q4 source. The
    //     residency/scoring POLICY (page table, routing scores, the cost-aware decision,
    //     telemetry) lives in the moert policy library (moert/include/moert/policy.h);
    //     this struct keeps only what touches ggml or the filesystem. ---
    bool   dynamic = false;
    moert_policy *      pol = nullptr;                  // owns residency + scores + planner + telemetry
    moert_policy_config pcfg{};                         // knobs (env-parsed at init; kept for banner/getters)
    // the promotion SOURCE (page catalog, file handles, reads — expert store or
    // legacy GGUF-offset shim) and the hot-slab LAYOUT/lifetime live in the moert memory manager;
    // this struct keeps only the ggml view of the slab. The typed pool tensors in w_hot/s_hot are
    // manually placed at the memmgr's region offsets inside slab_buf (q4stream aliasing pattern).
    moert_memmgr *        mm = nullptr;
    ggml_backend_buffer_t slab_buf     = nullptr;       // device allocation backing the memmgr slab
    ggml_tensor *         slab_bytes_t = nullptr;       // I8 view of the whole slab (offset-addressed uploads)
    size_t                slab_total_bytes = 0;         // exact finalized slab size (captured in cb_slab_alloc;
                                                        // slab_restore re-allocates at exactly this size)
    // Device routing counters (the only observation since elision E1): routing counts accumulate in
    // the fused op's partition kernel ON DEVICE; the host fetches them only at refresh points. The
    // eval callback then never asks for ffn_moe_topk => the sched computes each graph as ONE async
    // submission instead of ~31 synchronous segments per token. STAGED-only; q4stream prefill is
    // incompatible (its staging rides the same callback).
    bool dev_counts = false;        // device counters active (env + STAGED + CUDA enable succeeded)


    // 3b: OWNED cold arena as the LIVE cold source. MOE_DUAL_COLD_ARENA=<experts.index> builds the
    // arena in memmgr and the fused op sources cold from it instead of llama's borrowed tensor —
    // the step before cutting the loader. `cold_arena_verify_*` accumulate the reg-time byte-match
    // (arena vs borrowed) that proves the swap is bit-identical.
    bool  cold_arena_on = false;
    long long cold_arena_verify_ok = 0, cold_arena_verify_bad = 0;

    // 3b instant pin: the owned IQ2 arena is page-locked in place (cudaHostRegister) once at load, so
    // H2D staging from it is async + fast from the very first token -- no waiting for the ColdPinner's
    // greedy per-fire locking to warm up during the first decodes. Record of the pinned block bases:

    // VRAM report fields (RuntimeInfo / JSON). All zero unless the pool backend is a GPU device.
    int       k_requested        = 0;
    long long vram_total_mib     = 0;
    long long vram_free_pre_mib  = 0;
    long long slab_mib           = 0;
    long long vram_used_post_mib = 0;

    // --- Q4Stream prefill: one FULL layer of Q4 experts staged into a single reused VRAM buffer,
    //     layer by layer, while a batched prompt chunk flows through the graph (see moe_dual.hpp).
    //     Deliberately independent of the decode hot pool (own file handle + offset tables + staging
    //     tensors). ---
    bool   q4s_enabled  = false;                // staging infrastructure loaded (q4stream_init)
    bool   q4s_on       = false;                // inside q4stream_begin/end (prompt phase)
    bool   q4s_error    = false;                // armed fault: abort the graph at the next topk callback
    std::vector<FILE *> q4s_f;                  // own handle(s) to the source, one per GGUF split shard
    int    q4s_layers_n = 0, q4s_expert_n = 0;  // geometry covered by the offset tables
    ggml_context *        q4s_ctx = nullptr;
    ggml_backend_buffer_t q4s_buf = nullptr;
    // Staging tensors: the UD source mixes quant types across layers (e.g. blk.29 down = Q8_0 while
    // the rest are Q5_1), so there is one staging VARIANT tensor per distinct (kind, scale?, type,
    // shape). All variants of the same (kind, scale?) slot ALIAS the same buffer region (manual
    // placement): only one layer's variant is live at a time, so the region just needs the max
    // variant size. q4s_wt/q4s_st map each layer to its variant.
    std::vector<std::array<ggml_tensor *, 2>> q4s_wt, q4s_st;  // [il][kind] staging tensor (null = absent)
    std::vector<std::array<int64_t, 2>> q4s_off,   q4s_off_s;   // [il][kind] file offset of expert 0 (-1 = absent)
    std::vector<std::array<size_t,  2>> q4s_blk,   q4s_blk_s;   // [il][kind] bytes per expert
    std::vector<std::array<int16_t, 2>> q4s_shard, q4s_shard_s; // [il][kind] which q4s_f[] shard owns the offset
    std::vector<char> q4s_pending;              // [il] built into the current graph -> stage at its topk
    std::vector<char> q4s_rbuf;                 // fallback host read buffer (pageable, synchronous path)
    // Overlapped staging pipeline: two PINNED host slices; while slice i uploads (true async H2D on
    // g.backend's own stream), the disk read of slice i+1 runs on the CPU. Events fence slice reuse;
    // a full backend sync before returning to the graph keeps the callback contract (staging done).
    ggml_backend_buffer_t q4s_pin_buf = nullptr;         // pinned host buffer holding both slices
    char *                q4s_slice[2] = { nullptr, nullptr };
    size_t                q4s_slice_sz = 0;
    ggml_backend_event_t  q4s_ev[2] = { nullptr, nullptr };   // per-slice upload-completion fence
    bool                  q4s_ev_live[2] = { false, false };
    std::vector<void *>   q4s_hfile;             // per-shard Win32 NO_BUFFERING handle (pipelined reads);
                                                  // an entry is null => that shard falls back to q4s_f[shard]
    long long q4s_bytes = 0, q4s_staged = 0;    // telemetry: cumulative streamed bytes / layer stagings
    double    q4s_stage_ms = 0.0;
    double    q4s_read_ms  = 0.0;               // fread slice of stage_ms (read-bound vs upload-bound)

    // --- ARENA-SOURCED staging (MOE_DUAL_Q4S_ARENA). When the staging source is the model's own file,
    //     the bytes the prefill reads are also in the owned cold arena: the staging copies come from RAM
    //     through the same pinned-slice pipeline; only (layer,kind,ws) groups PROVEN byte-identical to the
    //     file are mapped. `ar_str` is the arena's addressing stride per expert. ---
    int q4s_arena_mode = 0;                                            // 0 = file reads, 1 = arena-sourced
    std::vector<std::array<const void *, 2>> q4s_ar,     q4s_ar_s;     // [il][kind] arena base (null = file)
    std::vector<std::array<uint64_t,     2>> q4s_ar_str, q4s_ar_str_s; // [il][kind] addressing stride
    long long q4s_arena_bytes = 0;              // telemetry: bytes staged from RAM (subset of q4s_bytes)
    long long q4s_ar_ok = 0, q4s_ar_bad = 0;    // arm-time byte-compare of the arena against the file
    // Residency evidence for the arena source. "Already in RAM" is a claim about PHYSICAL residency,
    // and a committed arena can be trimmed to the pagefile without anyone noticing; these two make
    // the difference measurable instead of assumed. faults = process page faults charged while the
    // arena copies ran (~1 per 4 KiB touched => nothing was resident); ws_mib = working set after.
    long long q4s_faults = 0;
    long long q4s_ws_mib = 0;

    // --- R1: cross-layer double-buffered staging. The legacy path above reads +
    //     uploads layer `il` SYNCHRONOUSLY inside the ffn_moe_topk eval callback, so the GPU idles
    //     for the whole ~165-205 ms/layer SSD read. With a ring of q4s_ring_n full-layer VRAM buffers
    //     a background thread reads layer il+1 while the graph computes layer il, and the callback
    //     becomes a fence-wait. The graph hook (moe_dual.cpp) picks slot `il % q4s_ring_n`, so each
    //     layer's expert matmuls read the slot the prefetcher filled.
    //     q4s_ring_n == 1 keeps the ORIGINAL synchronous path byte-for-byte (q4stream_init picks the
    //     depth). Depth 2 suffices: the reader, not buffer availability, is the
    //     sustained bottleneck — a deeper ring only absorbs disk jitter. ---
    int q4s_ring_n = 1;                                    // 1 = legacy synchronous staging
    std::vector<ggml_backend_buffer_t>                 q4s_buf_r;   // [slot] one full-layer VRAM buffer
    std::vector<std::vector<std::array<ggml_tensor *, 2>>> q4s_wt_r;  // [slot][il][kind] staging weight
    std::vector<std::vector<std::array<ggml_tensor *, 2>>> q4s_st_r;  // [slot][il][kind] staging scale
    // prefetch thread + host-side fence protocol. The eval callback runs with the backend
    // SYNCHRONIZED, so at callback(il) every layer < il has finished computing — that is the
    // "consumed" signal, no device event needed for the reuse direction.
    std::thread             q4s_pf_thread;
    std::mutex              q4s_ring_mtx;
    std::condition_variable q4s_ring_cv;
    std::vector<int>        q4s_slot_layer;    // [slot] layer currently staged there (-1 = empty)
    std::vector<char>       q4s_slot_ready;    // [slot] contents fully uploaded and usable
    int                     q4s_consumed_upto = -1;   // highest layer whose compute is known done
    bool                    q4s_pf_run   = false;     // thread should keep going
    bool                    q4s_pf_fail  = false;     // reader hit an I/O error -> abort the graph
    double                  q4s_wait_ms  = 0.0;       // telemetry: callback time spent WAITING on the ring
                                                       // (exposed read; ~0 = the overlap fully hid the SSD)
    int                     q4s_pf_start_il = 0;      // first layer the prefetch thread stages

    // --- P2: prefill WITHOUT slab eviction (claude dump
    //     docs/design-prefill-transfer-2026-10-04.md section 2.3). The staging ring is a FIXED reserve:
    //     allocated with the boot slab and kept across prompts, so every refit
    //     measures free VRAM with it resident and K pays for it once instead of the slab being evicted
    //     and rebuilt from the file on every prompt. Each layer is staged from two sources: the experts
    //     already READY in the hot slab are copied device-to-device from their slot (the same bytes,
    //     proven once per process), and only the cold ones are read from the source (file or arena). ---
    std::string dyn_src_path;                    // the dynamic cache's promotion source (init_dynamic)
    std::string q4s_src_path;                    // the q4stream staging source (q4stream_init)
    bool q4s_lean    = false;                    // the current staging was armed lean (graph-reserve only)
    std::vector<ggml_tensor *> q4s_bytes_r;      // [slot] I8 byte view of the whole staging slot (D2D target)
    ggml_backend_event_t q4s_ev_d2d = nullptr;   // fence for the slab -> staging copies of one layer
    bool                 q4s_ev_d2d_live = false;
    long long q4s_slab_bytes = 0;                // telemetry: bytes staged D2D from the hot slab
    long long q4s_slab_experts = 0;              // telemetry: expert pages staged D2D
    std::atomic<bool> q4s_arena_trip{false};     // this prompt's arena copies were slower than the file
    long long q4s_ar_p_bytes = 0;                // current rate window: bytes copied from the arena (staging thread)
    double    q4s_ar_p_ms    = 0.0;              // current rate window: time those copies took

    // --- P3: residency decided at the boundary, inside the prefill (design
    //     section 2.4). The prompt's routing of layer il is counted from the ffn_moe_topk callback
    //     (every ubatch of the layer-major merged graph); once layer il is computed -- the first
    //     callback of layer il+1, or the end of the phase for the last layer -- and while the staging
    //     still holds all of layer il's experts, the policy re-decides layer il with the prompt counts
    //     as the forecast's prior and a near-zero price, and every expert that enters is copied D2D
    //     from the staging into its slot (no promotion, no SSD, no arena). ---
    float bnd_w  = 0.5f;                         // weight of the prompt counts in the forecast prior
    float bnd_r  = 0.5f;                         // price of one boundary copy, in forecast uses
    std::vector<std::vector<int32_t>> bnd_cnt;   // [il][e] routes over the prompt so far
    std::vector<long long> bnd_ntok;             // [il] prompt tokens counted
    std::vector<char>      bnd_pending;          // [il] counted in this decode call, not yet decided
    int  bnd_il = -1, bnd_rs = 0;                // layer being decided + the staging slot holding it
    long long bnd_promos = 0, bnd_layers = 0, bnd_fail = 0;   // telemetry (cumulative)
    double    bnd_ms = 0.0;

    // Hook-path knobs, parsed ONCE from the MOE_DUAL_* env compat surface at init (the
    // debt "hook getenv statics", retired in M2 — no more lazy `static int` env reads per call
    // site). The CUDA island's own knobs are pushed to it in the same pass.
    struct HookCfg {
        bool parsed       = false;
        int  fused        = 0;    // MOE_DUAL_FUSED: the selective fused op (vs compute-both blend)
        int  planc        = 0;    // MOE_DUAL_PLANC: RAM-cold fused delivery (1 ring / 2 zero-copy)
        int  spec_kmax    = 1;    // MOE_DUAL_SPEC_K: speculative batched-decode ceiling (1 = off;
                                  // 2..4 = fused op runs n_tokens<=spec_kmax via the gathered path,
                                  // Pmax = top_k(8)*n_tokens <= 32 at the 4 ceiling)
        int  arena_exclusive = 0; // MOE_DUAL_ARENA_EXCLUSIVE: full-IQ2 plan phase 2 -- a hot expert's arena
                                  // page is released and rebuilt FROM ITS SLOT when it goes cold (no SSD).
                                  // Requires MOE_DUAL_ASYNC_PROMOTE.
    } hcfg;
};

// The one process-lifetime instance (defined in moe_dual.cpp).
extern State g;

// SlotState mirror values: single-sourced from moert (which must in turn match the
// MOE_DUAL_SLOT_* values compiled into the ggml_moe_dual patch's partition kernel).
constexpr int32_t SLOT_FREE = MOERT_SLOT_FREE, SLOT_READY = MOERT_SLOT_READY;

// Hot capacity of layer il. Uniform across layers; kept as a per-layer accessor because that
// is what the call sites mean, and because variable-KV re-keys every layer through the memmgr.
inline int klayer(int) { return g.k + g.loan; }
// Loaned slots (findings 246; moe_dual.cpp)
int  loan_slots_for(int k_slab);   // how many slots the staging reserve can lend at this K (0 = off / no reserve)
void loan_tables_build();          // per-layer slot -> pointer tables for the hot matmul (cleared when loan == 0)
void loan_revoke(const char * why);   // before the staging is written or freed
void loan_lend();                     // after a streamed prompt
// P3 (moe_dual_q4stream.cpp / moe_dual.cpp)
int32_t q4stream_boundary_fill(int32_t il, int32_t e, int32_t slot);
void    xfer_settle();

// --- shared plumbing (defined in moe_dual.cpp) ------------------------------

// Parse the MOE_DUAL_* env compat surface once and push the CUDA island's slice of it
// through the ggml-cuda forwarder. Idempotent; called from every init entry point.
void parse_hook_env();

// Pool backend: GPU from the registry when available (MOE_DUAL_CPU=1 forces CPU).
ggml_backend_t pick_pack_backend();

bool ends_with(const std::string & s, const char * suf);

// Device memory of the pool backend's device; false when the pool is not on a GPU.
bool pool_dev_memory(size_t * free_b, size_t * total_b);

// 3b instant pin: page-lock the whole owned IQ2 cold arena at load (best-effort).

// The dual-gather graph hook registered with llama_set_moe_dual_mm_id().
ggml_tensor * hook(ggml_context * ctx, int il, int kind,
                   ggml_tensor * w_cold, ggml_tensor * cur,
                   ggml_tensor * ids, ggml_tensor * w_cold_s);

// moert memmgr device services (slab alloc/upload/free through ggml) and policy
// callbacks (promotion stream + residency publish + external-counter fetch).
int32_t cb_slab_alloc(void * user, uint64_t nbytes);
int32_t cb_slab_upload(void * user, uint64_t off, const void * src, uint64_t nbytes);
void early_join();   // early decision (findings 234): the policy is idle
void    cb_slab_free(void * user);
int32_t cb_slab_download(void * user, uint64_t off, void * dst, uint64_t nbytes);
int32_t cb_stream_expert(void * user, int32_t il, int32_t e, int32_t slot);
void    cb_publish_layer(void * user, int32_t il);
void    cb_fetch_external(void * user);

// Stream expert e's high-precision pages into pool slot `slot` of layer il (memmgr promote).
bool stream_expert(int il, int e, int slot);

// Variable-KV: clear/set the evicted flag from slab_rebuild_at_k (a rebuild while evicted IS a
// restore — it re-allocates the slab at the new K). Internal to the moe_dual TUs.
void slab_set_evicted(bool v);
// the C5.2 boot-deferral flag, cleared by the SAME event that clears `evicted` when
// the restore comes from slab_rebuild_at_k instead of slab_restore. The two call sites used to
// disagree — slab_restore cleared both, slab_rebuild_at_k only the first — so under variable-KV
// the deferral was never redeemed and the graph hook kept its lazy "restore instead of serving
// stock" permission for the whole session. Internal to the moe_dual TUs.

// Push layer il's residency mirrors (policy views) to the device tensors read by the graph.
void push_residency(int il);


}  // namespace moe_dual
