// ===========================================================================
// moert/policy.h — the expert residency policy.
// ===========================================================================
// The backend-agnostic state & policy: the page table (exp_slot / slot_exp / slot_state), the
// routing scores, the cost-aware decision and the telemetry counters.
//
// HARD RULE: this header and its implementation
// include NO ggml and NO CUDA headers. Device/file specifics reach the policy only
// through the callbacks below. C ABI so both C++ hosts and CUDA TUs can call it.
//
// One rule (since elision E3, 2026-10-05): promote on PRICED, FORECAST reuse, decided at every visible
// token (claude dump findings §159-§161, §173). The policy keeps an EMA of each expert's per-visible-
// token use rate (prompt routes excluded) and forecasts its use over the next H visible tokens,
// f = H * min(1, rate * (1 - decay)). Per layer, the best non-resident is paired with a free slot while
// f_in > r and then with the worst resident while f_in - f_out > r; at most cost_per_token promotions
// per visible token (the copy engine's capacity), uncapped at the prompt end / turn end / boundary.
// The V1 rule (score ratio + margin, min-residency, cooldown, layer cap, refresh budgets) and the
// decision-telemetry sink are at the tag pre-e3-2026-10-05.
// Any intentional change to promotion/eviction/scoring semantics needs a new gate run, not just review.
#ifndef MOERT_POLICY_H
#define MOERT_POLICY_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct moert_policy moert_policy;   // opaque

// Values must match moe_dual::RefreshReason (and the spec's refresh taxonomy).
enum moert_refresh_reason {
    MOERT_REFRESH_PROMPT_END       = 0,
    MOERT_REFRESH_VISIBLE_INTERVAL = 1,   // the per-visible-token decision (capped at cost_per_token)
    MOERT_REFRESH_TURN_END         = 2,
    // 3 was the hidden-warmup refresh (removed 2026-10-05, elision E1: never armed on the server)
    MOERT_REFRESH_BOUNDARY         = 4,   // one layer, decided while its prompt routing is known (P3)
};

// Slot lifecycle mirror values (must match MOE_DUAL_SLOT_* in the ggml_moe_dual patch:
// the device partition kernel reads the same encoding via the published mirrors).
// FILLING (asynchronous promotion, 2026-10-03): the slot is OCCUPIED by its new expert (exp_slot /
// slot_exp already point at it, so no other decision can take the slot) but the bytes are still in
// flight, so the device partition serves that expert COLD until moert_policy_complete_fill marks the
// slot READY. Any value other than READY is "not hot" for the partition kernel.
enum moert_slot_state { MOERT_SLOT_FREE = 0, MOERT_SLOT_FILLING = 1, MOERT_SLOT_READY = 2 };

// All policy knobs, fixed at construction (the host owns whatever config surface fills this).
typedef struct moert_policy_config {
    int32_t n_layers;                 // MoE layers covered
    int32_t n_expert;                 // experts per layer
    int32_t k;                        // hot-pool capacity per layer (slots)

    // The resize ranking (moert_policy_resize_k keeps the residents with the highest blended score).
    float   seed_prior;               // score of a commit_seed'ed expert
    float   session_decay;            // slow EMA (topic/session locality)
    float   recent_decay;             // fast EMA (current window)
    float   blend_a, blend_b, blend_c;// blended = a*recent + b*session (+ c*seed_prior if resident)

    // OPTIONAL static layerwise capacity (NULL = uniform k for all layers). When non-NULL it points
    // to n_layers int32 slot counts, each in [1, n_expert]; create() copies them, so the array need
    // only outlive the create() call. Uniform table (every entry == k) reproduces the scalar-k path.
    const int32_t * k_per_layer;

    // The decision.
    int32_t cost_horizon;             // H: visible tokens the forecast looks ahead
    float   cost_reuse;               // r: cold pairs one promotion must save to pay
    float   cost_rate_decay;          // per-visible-token decay of the rate EMA (default 0.5^(1/32))
    int32_t cost_per_token;           // promotions per visible token, at most (>= 1)
} moert_policy_config;

// Fill `cfg` with the product defaults (H 16, r 2, 4 promotions/token, findings §173).
// Geometry (n_layers/n_expert/k) is zeroed — the host must set it.
void moert_policy_config_defaults(moert_policy_config * cfg);

// Host services the policy drives. The policy is single-threaded and re-entrant only in
// the sense that callbacks may query views/stats; they must NOT call mutating policy API.
typedef struct moert_policy_callbacks {
    // Stream expert `e`'s high-precision weights into hot-pool slot (il, slot).
    // Return 1 when the bytes are in place (the slot becomes READY), 2 when the copy was
    // ACCEPTED ASYNCHRONOUSLY (the slot becomes FILLING; the host later calls
    // moert_policy_complete_fill or moert_policy_abort_fill), 0 on failure (the policy then
    // skips the promotion; on an eviction-reuse the slot stays FREE).
    int32_t (*stream_expert)(void * user, int32_t il, int32_t e, int32_t slot);
    // Layer `il`'s residency changed and ALL streaming for it completed: publish the
    // device mirrors (read them back via the moert_policy_*_view accessors). Called
    // once per touched layer, ascending il — the fused op must never see a
    // half-written slot as READY, which this ordering guarantees.
    void (*publish_layer)(void * user, int32_t il);
    // OPTIONAL (external/device observation only, else NULL): pour any pending device
    // routing counts into the policy via moert_policy_add_external_counts. Invoked at
    // refresh points and stats/hit-rate reads.
    void (*fetch_external)(void * user);
    void * user;
} moert_policy_callbacks;

// Live telemetry snapshot (the policy slice of moe_dual::RuntimeInfo).
typedef struct moert_policy_stats {
    int64_t promotions, demotions, tokens, visible_tokens;
    int64_t hot_routes, cold_routes;              // realized hit/miss, visible phase
    int64_t refresh_count, refresh_promotions, refresh_demotions;
    double  refresh_pause_ms_total, refresh_pause_ms_max;
    int64_t last_refresh_tok, last_refresh_promos;
    double  last_refresh_pause_ms;
    double  occupancy;                            // mean READY slots / K across layers (0..1)
    int64_t fills_async, fills_completed, fills_stale, fills_aborted;   // asynchronous promotion ledger
} moert_policy_stats;

// Lifecycle. create() copies cfg/cbs; all state starts cold (empty pool, zero scores).
// Returns NULL on invalid geometry.
moert_policy * moert_policy_create(const moert_policy_config * cfg,
                                   const moert_policy_callbacks * cbs);
void moert_policy_free(moert_policy * p);

// External (device) observation: per-token folds are skipped, scores fold at refresh
// points from counts delivered by fetch_external/add_external_counts.
void moert_policy_set_external_observation(moert_policy * p, int32_t on);

// --- observation & clocks -----------------------------------------------------------
// Routing observed for layer il (host observation; the product uses external counts): `ids` =
// n_used*n_tok expert ids (token-major). Folds per-expert counts + realized hot/cold accounting.
void moert_policy_observe(moert_policy * p, int32_t il, const int32_t * ids,
                          int32_t n_used, int64_t n_tok);
// Prompt progress: 1 token, or a whole batched chunk (fold is decay-corrected so a
// chunk carries the same total weight as per-token folds). Never promotes.
void moert_policy_on_prompt_token(moert_policy * p, int64_t n_tokens);
// One VISIBLE decode token ended: fold + age clocks, then the per-token decision.
void moert_policy_on_token_end(moert_policy * p);

// Run the decision now (see the top of this file). `reason` MOERT_REFRESH_VISIBLE_INTERVAL caps it at
// cost_per_token promotions; every other reason is uncapped. Streams are batched and residency is
// published once per touched layer.
void moert_policy_refresh(moert_policy * p, int32_t reason);

// Variable-KV: resize the hot capacity to a UNIFORM k_new
// at runtime, preserving every learned score/clock (only the page table is rebuilt, not the routing
// statistics). Per layer: residents are ranked by the blended score; the top-k_new survive, the rest
// are evict-accounted (demotions); survivors COMPACT into slots 0..n-1 in rank order and stay READY.
// HOST CONTRACT: the caller rebuilds the slab and re-streams every surviving resident into its NEW slot
// (slot_exp_view) before the next decode — exactly the slab_restore contract — then publishes every
// layer. A layerwise capacity table is collapsed to the uniform k_new.
// Slot-array view pointers (slot_exp/slot_state) are INVALIDATED by this call — re-fetch them.
// 1 = ok, 0 = invalid handle/k.
int32_t moert_policy_resize_k(moert_policy * p, int32_t k_new);

// Loaned slots (claude dump findings 246): limit every layer to its first k slots (k <= the allocated
// capacity) or give them all back (k >= capacity). Residents of the slots above k are evicted (READY and
// FILLING alike: the HOST must have drained every in-flight copy first) and the decision never uses those
// slots until the capacity is restored; the slot arrays keep their size. The caller publishes every layer.
int32_t moert_policy_set_active_k(moert_policy * p, int32_t k);

// Test seam (the product no longer seeds): put expert e into (il, slot) READY with the seed prior as
// its blended score. can_seed = e valid and not already resident. commit_seed returns 0 if can_seed fails.
int32_t moert_policy_can_seed(const moert_policy * p, int32_t il, int32_t e);
int32_t moert_policy_commit_seed(moert_policy * p, int32_t il, int32_t e, int32_t slot);

// Asynchronous promotion (stream_expert returned 2). complete_fill: the copy of expert e into (il, slot)
// finished — if the slot still holds e in FILLING it becomes READY and 1 is returned (the caller then
// publishes layer il); 0 means the completion is STALE (the slot was evicted or re-assigned meanwhile)
// and nothing changes. abort_fill: the copy failed — a matching FILLING slot is released to FREE
// (1 = released, 0 = stale). Neither calls back into the host.
int32_t moert_policy_complete_fill(moert_policy * p, int32_t il, int32_t slot, int32_t e);
int32_t moert_policy_abort_fill(moert_policy * p, int32_t il, int32_t slot, int32_t e);

// P3 boundary residency (claude dump docs/design-prefill-transfer-2026-10-04.md section 2.4): layer
// il's routing over the n_tok tokens of the prompt just prefilled (`counts`, len n_expert, each <= n_tok)
// becomes the prior of the decode forecast -- the use-rate EMA moves toward the prompt's per-token rate,
// rate = (1 - weight) * rate + weight * (count / n_tok) / (1 - decay) -- and the decision then runs on
// THIS layer only at price `r` (the copy comes from the prefill staging, ~free). Promotions go through
// stream_expert as usual (the host copies from its staging and returns 1); the layer is published.
// Returns the number of promotions, -1 on invalid input.
int32_t moert_policy_boundary_layer(moert_policy * p, int32_t il, const int32_t * counts, int64_t n_tok,
                                    float weight, float r);

// External observation input: add per-expert route counts (len n_expert) and realized
// hot/cold pair counts for layer il.
void moert_policy_add_external_counts(moert_policy * p, int32_t il,
                                      const int32_t * counts, int64_t hot, int64_t cold);

// Reset live telemetry counters only (learned residency + scores kept) — chat /reset.
void moert_policy_reset_stats(moert_policy * p);

// Capacity of layer il (the uniform k unless a per-layer table was given at create). -1 invalid.
int32_t moert_policy_layer_k(const moert_policy * p, int32_t il);

// External forecast (claude dump findings 210): when f is non-NULL the cost-aware pairing ranks and prices
// on f[il * n_expert + e] (expected uses over the forecaster's horizon, in its own units) at price r, with
// at most cap promotions per per-token decision (cap <= 0: the config's cost_per_token). The buffer is
// owned by the host and may be rewritten from the fetch_external callback. The P3 boundary pass keeps its
// own count-based forecast. NULL restores the rate EMA.
void moert_policy_set_external_forecast(moert_policy * p, const float * f, float r, int32_t cap);

// Telemetry. Both pull pending external counts first.
double moert_policy_hit_rate(moert_policy * p);
void   moert_policy_get_stats(moert_policy * p, moert_policy_stats * out);

// --- windowed rates -----------------------------------------
// Instantaneous view over the last `n_tokens` VISIBLE decode tokens (clamped to the ring
// capacity of 128 and to what has been observed): decode tok/s from per-token steady-clock
// timestamps, and the realized hit-rate over the window from cumulative route snapshots.
// Pulls pending external counts first, like the reads above. Cleared by reset_stats.
typedef struct moert_policy_window {
    int32_t tokens;                 // tokens actually covered (<= requested; 0 if <2 samples)
    double  span_ms;                // wall time covered by the window
    double  tok_per_s, ms_per_tok;  // decode rate over the window (0 when tokens == 0)
    double  hit_rate;               // realized hit-rate over the window (0 when no routes)
    int64_t hot_routes, cold_routes;// routes attributed to the window
} moert_policy_window;
void moert_policy_get_window(moert_policy * p, int32_t n_tokens, moert_policy_window * out);

// --- residency views (the page table; device-mirror sources) ------------------------
// Pointers stay valid until free(); contents change only inside the mutating calls
// above. Lengths: n_expert for exp_slot/hotmap/resid, k for slot_exp/slot_state.
const int32_t * moert_policy_exp_slot_view  (const moert_policy * p, int32_t il); // e -> slot | -1
const int32_t * moert_policy_slot_exp_view  (const moert_policy * p, int32_t il); // slot -> e | -1
const int32_t * moert_policy_slot_state_view(const moert_policy * p, int32_t il); // slot -> moert_slot_state
const int32_t * moert_policy_hotmap_view    (const moert_policy * p, int32_t il); // e -> slot (0 if cold; blend path)
const float   * moert_policy_resid_view     (const moert_policy * p, int32_t il); // e -> 1.0 hot / 0.0 cold

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // MOERT_POLICY_H
