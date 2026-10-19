#pragma once

#include <cstdint>
#include <vector>

// ---------------------------------------------------------------------------
// moe_dual — the dual-precision MoE expert runtime (host side).
// ---------------------------------------------------------------------------
// A per-layer hot pool of high-precision experts in VRAM, promoted per session from observed routing,
// over a cold low-precision (IQ2) tier of ALL experts in RAM. A hook registered with llama.cpp
// (llama_set_moe_dual_mm_id) replaces each MoE expert matmul; on CUDA the fused selective op
// (GGML_OP_MOE_DUAL, moert/cuda) computes every routed pair ONCE at its tier.
//   moert/policy   WHAT is hot: page table, routing scores, the cost-aware decision, telemetry.
//   moert/memmgr   WHERE bytes live: hot-slab layout, promotion source (GGUF offsets), owned cold arena.
//   moert/cuda     the CUDA island behind a device-pointer ABI.
//   this module    the ggml/llama.cpp glue: pools as ggml tensors, the hook, promotion callbacks,
//                  the prefill staging, the env knob surface (parsed once at init).
// Split: moe_dual.cpp (hook + decode-time runtime), moe_dual_init.cpp (construction),
// moe_dual_q4stream.cpp (prefill staging), moe_dual_state.hpp (shared internal state).
struct ggml_tensor;   // q4stream_count_topk (P3)

namespace moe_dual {

// The dynamic cache: a capacity-K VRAM pool per layer, filled from observed routing, hot weights
// promoted from the GGUF `q4_path`, plus the fused-op residency mirror. Routing is counted on the device
// by the fused op; the host calls on_token_end() per generated token. The pool starts empty: the boot
// slab (server_main on_ready) fits K and the first streamed prefill fills it (boundary residency).
// Promotion happens ONLY at controlled points: prompt end, every visible token, turn end, the prefill
// boundary.
bool init_dynamic(const char * q4_path, int k);

// Measure device used/total after the model+context load, for runtime_info()/JSON only.
void measure_vram_post_load();


// Diagnostic (route-trace, cache-aware top-k gate): true iff a routed (il, e) pair would be served
// from the hot pool right now — the fused-op partition predicate (exp_slot/slot_exp match + READY).
// Dynamic cache only; always false for stock runs.
bool expert_is_hot(int il, int e);

// Device routing counters (always on since elision E1): routing is counted on device by the fused
// op's partition kernel and fetched only at refresh points, so the decode graph runs as one async
// submission. False only if the CUDA counter allocation failed -- the server then refuses to run.
bool dev_counts_active();


// PREFILL (prompt) progress: fold routing counts into the session/recent scores. Never promotes
// (prefill only observes). No-op unless the dynamic cache is active. `n_tokens` = how many prompt
// tokens the accumulated counts cover: 1 for token-by-token prefill, the chunk size for batched
// prefill (the fold is decay-corrected so a chunk carries the same total weight as per-token folds).
void on_prompt_token(long long n_tokens = 1);

// One VISIBLE (decode) token ended: fold scores + age slots, then apply the mode's promotion
// decision (the cost-aware rule, every visible token).
void on_token_end();
// Before a generating step's decode: mark the promotions whose copies already finished READY and publish
// them, so the verify that follows uses them (decided after step t, hot from step t+1 instead of t+2).
// Never waits. No-op unless MOE_DUAL_EARLY_READY=1.
void apply_ready_before_decode();
// Early decision (claude dump findings 234): posts the per-step decision to run during the verify about to start.
void after_publish_before_decode();
void publish_fence();   // batch transport: publish the deferred residency mirrors in one batch before a decode
void mtp_draft(int i, const float * h, int n);   // MTP forecaster (findings 216k): a draft's h_nextn
void before_decode();   // MTP forecaster: the drafts this verify uses become its input

// The cost-aware decision across ALL layers (moert/policy.h), batching the promotions.
// `reason` is a RefreshReason; REFRESH_VISIBLE_INTERVAL is capped at the per-token promotions.
enum RefreshReason { REFRESH_PROMPT_END = 0, REFRESH_VISIBLE_INTERVAL = 1, REFRESH_TURN_END = 2 };
void refresh_cache(int reason);

// End of an assistant turn: the turn-end decision (uncapped). No-op unless the dynamic cache is
// active. Convenience wrapper over refresh_cache(REFRESH_TURN_END).
void on_turn_end();

// C5.2 admission preflight: true when init_dynamic refused because free VRAM/RAM was
// clearly short of the load's needs (no device/host allocation was attempted). The host
// exits with code 86 so the router's spawn gate stops respawning into the same wall.
bool preflight_failed();

bool is_dynamic();
bool is_staged();          // the dynamic cache is loaded (the only promotion mode is staged)
int  cost_per_token();     // promotions per visible token, at most; 0 if not loaded
// Fused-op counters since process start (moert_cuda_stats); false when the CUDA island is absent.
struct IslandStats { long long fires = 0, op_host_ns = 0, seam_syncs = 0, seam_ns = 0,
                     coldcpu_calls = 0, coldcpu_pairs = 0, coldcpu_ns = 0, coldcpu_over_max = 0; };
bool island_stats(IslandStats * out);

// Reset only the live telemetry counters (routes/promotions/demotions/tokens) — used by the
// chat /reset so a fresh conversation shows its own hit-rate curve. The learned residency is kept.
void reset_stats();

// Drain the pending on-device routing counters WITHOUT folding them (fetch+drop). Server path
// only: neutralizes llama-server's startup can_seq_rm probe (2 dummy tokens through the runtime)
// so the first request's refresh sees exactly the app-path scores. No-op unless dev counts.
void discard_pending_routes();

// --- Q4Stream prefill ---------------------------------------------------------------------------
// Prefill routing is near-uniform, so the sparse hot cache is a weak fit: the prompt streams ONE FULL
// LAYER of source experts at a time through a reused VRAM staging buffer (the fixed reserve), uploaded
// just in time at the ffn_moe_topk sync point, so the prompt KV is built at the source precision at
// batched speed. Type-generic; follows GGUF split files (an expert can straddle a shard boundary).
// Loud failure (false + stderr) on a missing shard, an incomplete layer or a staging that does not fit
// — never a silent downgrade. `lean_ring` = an arm that only keeps graph_reserve on the streamed path
// (load time, ctx rebuild): depth 1, since free VRAM there is inflated; the boot-slab arm sizes the ring
// for real (depth 2 only for an SSD source with K to spare).
bool q4stream_init(const char * src_path, const char * label = "Q4STREAM", bool lean_ring = false);
bool q4stream_available();

// Free the staging buffer/context/file handles and reset all q4stream state (safe no-op if not
// initialized). Q4Stream's normal multi-turn use never calls this — the buffer is reused turn to
// turn; the shutdown path exists for a context rebuild, which must not leave the old staging
// buffer in VRAM alongside the new one.
void q4stream_shutdown();

// --- Vision-transient slab evict/restore (C4c) -------------------------------------------------
// A non-causal mtmd image chunk needs a ~2.7 GiB transient compute buffer that cannot coexist with
// the resident hot slab on 8 GB. slab_evict frees the slab's VRAM (bookkeeping — policy page table,
// residency mirrors — is untouched); slab_restore re-allocates it and re-streams every resident
// expert from the Q4 source at its original slot = a bit-identical restore. Wired to the fork's
// llama_set_vision_room callback around the vision-transient decode.
// slab_evict(min_free_mib): no-op (returns true, arms nothing) when the device already has
// min_free_mib free — small-K configs skip the ~2s restore cost entirely.
bool slab_evict(long long min_free_mib);
bool slab_restore();
bool slab_evicted();

// --- Variable-KV S6a: rebuild the hot slab at a NEW uniform K at runtime ------------------------
// Quiescent-point operation (between turns, no in-flight graph). Policy ranks each layer's
// residents by its own blended eviction score and keeps the top-K_new (learned scores/clocks
// preserved); the memmgr re-layouts the hot slab WITHOUT touching the K-independent cold arena;
// the K-dependent stack (pool tensors + slab + aux mirrors) is rebuilt by the same helpers init
// uses; survivors re-stream into their new compacted slots. Refuses while the slab is evicted or
// q4stream is mid-prefill. ~2-4 s at product K.
bool slab_rebuild_at_k(int k_new);

// --- Variable-KV Phase 3: the MEASURED K clamp -------------------------------------------------
// vkv_per_k_bytes(): exact hot-slab bytes per K unit, from the memmgr page catalog (0 = no runtime).
// vkv_refit_k(): grow/shrink K so `keep_free_mib` VRAM stays free afterwards — measurement-driven,
// so it self-corrects around whatever the current context actually costs. ±1-slot hysteresis.
// Returns the K in force after the call. Quiescent-point only (it calls slab_rebuild_at_k).
uint64_t vkv_per_k_bytes();
int      vkv_refit_k(long long keep_free_mib);

// Arms the measured refit for vkv_restore_or_refit() (variable-KV on). Set once at server init.
void vkv_set_boot_policy(bool armed, long long keep_free_mib);
// The one function every such call site should use instead of a bare slab_restore(): refits to the
// MEASURED K when armed, else falls back to plain slab_restore() (byte-identical when unarmed).
bool vkv_restore_or_refit();

// enforces THE invariant of a mid-session context swap — the hot slab must be SMALL
// BUT RESIDENT when the new context runs graph_reserve. Its token-generation graph (n_tokens == 1)
// does NOT go through the q4stream bracket (that branch is `n_tokens > 1 && g.q4s_on`), so its only
// cheap path is the hook's fused branch on the resident `w_hot` pool tensors; with the slab evicted
// the hook returns nullptr and llama's stock graph sources the loader-cut expert SHELLS, whose
// cross-backend copies saturate GGML_SCHED_MAX_SPLIT_INPUTS -> a ~2734 MiB PERMANENT compute buffer
// for the life of that context (~25 slots of K). Shrinks a resident slab to the quality floor, or
// RESTORES an evicted one at the floor; no-op when already resident at/below the floor. Must be
// called BEFORE q4stream arms (slab_rebuild_at_k refuses under an armed bracket). Does not change
// the final K: moe_ctx_rebuild_done -> vkv_refit_k remains the single true recompute per resize.
//
// `slab_budget_mib` = the VRAM the slab may occupy DURING the swap (host-computed: measured old-ctx
// footprint + vkv::ctx_cost_predict_mib for the new one, minus q4stream staging and the safety
// margin). With it, the shrink removes exactly the K_old - K_new experts that no longer fit, by the
// LRU rank — the user's spec, literally. Pass -1 when no measurement is available and the blunt
// collapse-to-the-quality-floor fallback should be used instead.
bool vkv_fit_slab_for_ctx_swap(long long slab_budget_mib);

// Free VRAM on the pool device, MiB (0 if there is no GPU pool). The host uses this to measure the
// old context's real footprint across its teardown — the one input to the ctx-cost model that must
// never be modelled.
long long pool_free_mib();

// Deferred trigger for the server path: SCHEDULE a rebuild from any thread (the HTTP handler);
// on_turn_end applies it on the slot thread at the next turn boundary (the quiescent point).
// 1 = scheduled; 0 = rejected with a reason written into err (no runtime / K out of range).
int request_rebuild_k(int k, char * err, int errcap);

// this process's WDDM shared-segment usage sampled BEFORE the run's first device
// allocation — the only reading that can serve as a spill baseline, because anything taken later
// has already absorbed whatever spilled during load. -1 when DXGI is unavailable.
long long nonlocal_boot_mib();

// Vision: the transient image decode reads routed experts straight from the cold arena on the CPU, so
// every arena page must be resident first. Under the exclusive arena this rebuilds every released page
// from VRAM and holds them until the turn ends; otherwise a no-op. Call before each image decode.
void arena_vision_full_commit();

// Bracket the prompt phase: inside begin/end, batched hook calls (n_tokens > 1) take the staged
// full-layer Q4 path. Decode (n_tokens == 1) is never affected. The staging rides the server's
// dedicated ffn_moe_topk eval callback (the only sync point the streamed prefill needs).
void q4stream_begin();
void q4stream_end();

// True while inside q4stream_begin/end with the staging infrastructure loaded — the server's
// staging-only eval callback keys on this (asks for nothing outside a streamed prompt phase).
bool q4stream_prompt_active();

// Re-arm staging for the next prompt-chunk llama_decode. MUST be called before every multi-token
// decode inside begin/end: llama.cpp reuses the previous graph when consecutive chunks have the
// same shape (chunks < the 256-token kq-mask pad quantum), skipping the hook's build-time pending
// marks — the buffer would then keep the LAST staged layer's bytes for the whole chunk (wrong
// type variant under the aliased regions => NaN KV). Idempotent; a no-op outside begin/end.
void q4stream_chunk_begin();

// Called from the ffn_moe_topk eval callback (ask=false, backend synced) with the layer index:
// streams layer `il`'s full Q4 expert set from disk into the staging buffer. Returns false on a
// stream error or an armed q4stream fault — the caller must abort the graph (loud, no IQ2 KV).
bool q4stream_on_topk(int il);

struct Q4StreamStats {
    long long bytes_streamed = 0;   // cumulative bytes read from the Q4 source (== H2D upload bytes)
    long long layers_staged  = 0;   // cumulative full-layer stagings
    double    stage_ms       = 0.0; // cumulative wall time spent reading + uploading
    double    read_ms        = 0.0; // fread slice of stage_ms (stage_ms - read_ms ~= exposed upload/sync)
    long long buf_bytes      = 0;   // staging buffer size (one full expert layer, all kinds)
    // R1 cross-layer prefetch (ring_n > 1): time the eval callback spent BLOCKED waiting for the
    // prefetcher. This is the staging cost still EXPOSED to the graph -- ~0 means the reader stayed
    // ahead and the SSD read was fully hidden behind compute. At ring_n == 1 it stays 0 (the whole
    // read is inline in stage_ms instead).
    double    wait_ms        = 0.0;
    int       ring_n         = 1;   // staging ring depth actually in use
    // Arena-sourced staging (MOE_DUAL_Q4S_ARENA): the slice of bytes_streamed that came from the
    // resident owned cold arena instead of the source file. == bytes_streamed when fully armed, 0
    // when staging from disk; anything in between means only some groups mapped.
    long long arena_bytes    = 0;
    bool      arena_mode     = false;   // arena sourcing armed for this prompt
    // P2: bytes / expert pages staged device-to-device from the hot slab (not part of bytes_streamed)
    long long slab_bytes     = 0;
    long long slab_experts   = 0;
    // P3: boundary decisions (cumulative): layers decided, experts copied staging -> slab, time
    long long bnd_layers     = 0;
    long long bnd_promos     = 0;
    long long bnd_fail       = 0;
    double    bnd_ms         = 0.0;
    // Residency evidence: page faults charged while the arena copies ran and the working set after.
    // ~1 fault per 4 KiB of arena_bytes means the arena was NOT physically resident (committed pages
    // trimmed to the pagefile), which makes "the bytes are already in RAM" false and the copy slower
    // than re-reading the source file.
    long long arena_faults   = 0;
    long long ws_mib         = 0;
};
Q4StreamStats q4stream_stats();

// P2 (MOE_DUAL_PREFILL_NOEVICT=1): true when the staging must SURVIVE the end of the prompt -- it is
// the fixed reserve the prompt-end refit fits K against, so the next prompt needs no slab eviction.
// False for a lean (graph-reserve-only) arm, which is always released.
bool q4stream_keep_after_prompt();

// P3 (MOE_DUAL_BOUNDARY=1): count the prompt routing of layer il from its ffn_moe_topk result (called
// from the staging eval callback for every ubatch, after q4stream_on_topk). No-op when off.
void q4stream_count_topk(int il, const ggml_tensor * topk);

// Read-only runtime summary for the CLI startup banner / `/stats`. `active` is false when no pack or
// dynamic cache is loaded (stock path). `fused`/`planc` mirror the MOE_DUAL_FUSED/MOE_DUAL_PLANC env.
struct RuntimeInfo {
    bool      active     = false;  // the dynamic hot pool is loaded
    bool      dynamic    = false;  // dynamic online cache (vs static oracle pack)
    int       hot_k      = 0;      // hot experts kept per layer (K)
    int       n_expert   = 0;      // total experts per layer
    int       n_layers   = 0;      // MoE layers the hook covers
    int       total_slots = 0;     // n_layers * hot_k
    bool      fused      = false;  // fused selective op active (MOE_DUAL_FUSED=1)
    int       planc      = 0;      // RAM-cold delivery: 0 none / 1 staging ring / 2 zero-copy
    long long promotions = 0;      // dynamic-cache streaming promotions (0 for static)
    long long demotions  = 0;      // dynamic-cache evictions (0 for static)
    long long tokens     = 0;      // dynamic-cache tokens observed (0 for static)
    long long hot_routes = 0;      // cumulative routed pairs served from the Q4 hot pool
    long long cold_routes= 0;      // cumulative routed pairs served cold (staged IQ2)
    double    occupancy  = 0.0;    // mean filled hot slots / K across layers (0..1)

    // windowed rates over the last ~32 visible tokens (M2 R9/R10; zeros until 2+ samples).
    // NOTE: with device counters the window hit-rate advances in refresh-interval steps
    // (counts are poured at refresh points, never per token); tok/s is per-token exact.
    int       win_tokens     = 0;    // tokens the window actually covers
    double    win_tok_per_s  = 0.0;  // instantaneous decode rate
    double    win_ms_per_tok = 0.0;
    double    win_hit_rate   = 0.0;  // realized hit-rate over the window

    // staged-cache promotion policy + refresh telemetry (0/"" for online or static)
    const char * cache_mode = "";  // "staged" when the dynamic cache is loaded
    long long visible_tokens        = 0;   // decode tokens observed (excludes prefill)
    long long refresh_count         = 0;   // number of refresh_cache() calls
    long long refresh_promotions    = 0;   // promotions applied at refresh points (staged)
    long long refresh_demotions     = 0;   // demotions applied at refresh points (staged)
    double    refresh_pause_ms_total= 0.0; // cumulative host wall time spent in refresh_cache()
    double    refresh_pause_ms_max  = 0.0; // worst single refresh pause
    long long last_refresh_tok      = 0;   // visible-token index of the last refresh
    long long last_refresh_promos   = 0;   // promotions applied by the last refresh
    double    last_refresh_pause_ms = 0.0; // wall time of the last refresh

    // M3a dynamic-K / VRAM budget guardrail (all 0 unless the dynamic cache is on a GPU device;
    // *_target/_requested are 0 when the guardrail is not engaged)
    int       k_requested             = 0;   // K asked for before any guardrail clamp
    long long vram_total_mib          = 0;   // device total memory
    long long vram_free_pre_slab_mib  = 0;   // device free just before the hot-slab allocation
    long long slab_mib                = 0;   // hot-slab device allocation (all layers, K experts each)
    long long vram_used_post_load_mib = 0;   // measured usage after model+context load (measure_vram_post_load)

    // 3b: OWNED cold arena as the live cold source (MOE_DUAL_COLD_ARENA). verify_* = the reg-time
    // byte-match arena-vs-borrowed (bad must be 0 — proves the cold-source swap is bit-identical).
    bool      cold_arena_active       = false;
    long long cold_arena_verify_ok    = 0;
    long long cold_arena_verify_bad   = 0;

    // --- live memory map (for /moe/telemetry + the bench mem-map block; all MiB). These are the moe_dual-OWNED
    // regions only — KV cache + non-expert weights + ggml compute buffers are llama-owned and probed
    // separately (see mem_probe.hpp), since runtime_info() has no llama_context/model handle. Device
    // totals above (vram_total_mib / vram_used_post_load_mib / slab_mib) frame these on the VRAM side.
    long long cold_pool_mib   = 0;  // borrowed IQ2 cold expert pool in host RAM (sum of cold views)
    long long arena_committed_mib = 0;  // C5.2: currently-committed arena RAM (total minus RAM-dedup released)
    long long cold_arena_mib  = 0;  // moert-OWNED cold arena RAM (0 unless MOE_DUAL_COLD_ARENA; a
                                    // duplicate of the borrowed pool until the loader is cut)
    long long staging_mib     = 0;  // Q4Stream/BF16 full-layer staging buffer in VRAM (0 if none)

    // VRAM L2 cold cache: REMOVED 2026-10-03 (tag pre-prune-cold-research-2026-10-03). The fields stay,
    // permanently inactive / 0, so the /moe/telemetry JSON keeps its shape for existing readers.
    bool      l2_active       = false;
    long long l2_budget_mib   = 0;
    long long l2_slab_mib     = 0;
    long long l2_capacity     = 0;
    long long l2_resident     = 0;
    long long l2_lookups      = 0;
    long long l2_hits         = 0;
    long long l2_misses       = 0;
    long long l2_routes       = 0;
    long long l2_bytes_avoided= 0;
};
RuntimeInfo runtime_info();

// Cumulative hit-rate over all routes observed so far (hot / (hot+cold)); 0 if none.
double hit_rate();

// --- Seam-telemetry companions ---------------
// The per-fire n_cold histogram is only interpretable against the capacity that produced it:
// a distribution recorded across a K change or a slab rebuild is two populations, not one.
// These are the cache-side facts the seam JSON reports next to the histogram. All zero /
// empty when the dynamic runtime is not loaded.
struct SeamCacheState {
    int       n_layers   = 0;
    int       k_by_layer[128] = {};   // per-layer hot capacity (uniform K unless layerwise)
    long long k_changes  = 0;         // uniform-K changes applied by the variable-KV ladder
    long long rebuilds   = 0;         // hot-slab rebuilds performed (>= k_changes)
    long long promotions = 0;
    long long demotions  = 0;
};
void seam_cache_state(SeamCacheState * out);

// Live device-wide VRAM (the hot pool's device): fills MiB values and returns true, or
// returns false when the runtime is inactive / the pool is not on a GPU (values untouched).
bool vram_now(long long * free_mib, long long * total_mib);

// --- C5.6 memory accounting: the owned cold arena's RAM blocks -------------------------------
// The partition probe needs the arena's ADDRESS RANGES, not just its size, because "allocated"
// and "resident" are different questions and only the second one belongs in a memory picture
// (most of a 7.9 GiB arena is paged out on a 16 GB box). These are thin pass-throughs to the
// memmgr's block enumeration; read-only, no allocation, safe from any thread that is not
// concurrently rebuilding the arena (the probe runs at turn boundaries and on the idle timer).
int  cold_arena_block_count();
bool cold_arena_block(int i, void ** base, unsigned long long * bytes);
// Currently-committed arena bytes (allocated minus whatever the exclusive arena released).
long long cold_arena_committed_mib();

}  // namespace moe_dual
