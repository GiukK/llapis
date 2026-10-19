#pragma once

// moe_dual trampolines (LLapis fork addition — trampoline-only, no MoE logic here,
// same discipline as the llama-moe-dual-hook patch). A single-slot MoE runtime that embeds
// llama-server-impl (moert-server) registers per-phase callbacks; every pointer is null by
// default and every call site is a no-op when unset.
//
// The phase model mirrors the reference decode driver:
//   prompt chunk decoded  -> on_prompt_tokens(n)   (routing scores folded, no promotion)
//   prompt -> generation  -> on_prompt_end()       (prompt-end cache refresh)
//   generated token decoded -> on_token_end()      (per-token cache upkeep / interval refresh)
//   generating slot released -> on_turn_end()      (turn-end refresh for the next turn)
//
// NOTE: the call sites classify the whole batch by the single processing slot's state, which
// is only correct with n_parallel = 1 — the host app must enforce a single slot.

extern void (*g_srv_moe_on_prompt_tokens)(int n_tokens);
extern void (*g_srv_moe_on_prompt_end)();
extern void (*g_srv_moe_on_token_end)();
extern void (*g_srv_moe_on_turn_end)();
// [moe-hooks] C5.6: the slot was released WITHOUT ever reaching the generating state — a client
// that disconnected mid-prefill, a cancelled task, a send_error path, abort_all_slots. on_turn_end
// deliberately does NOT fire for these (no turn happened, so no cache upkeep should be charged),
// which left the host's PROMPT-phase state uncleaned: the q4stream bracket stays open and its
// ~589 MiB staging buffer stays allocated for as long as the server then sits idle, silently
// shrinking the K the next prompt-end refit can afford. Null by default, like every hook here.
extern void (*g_srv_moe_on_turn_abort)();
void llama_server_set_moe_turn_abort(void (*fn)());
// fired once, after load_model (incl. the can_seq_rm startup probe) and right before the
// server starts accepting tasks — the host discards probe-polluted routing state here.
extern void (*g_srv_moe_on_ready)();

// Observability provider (M1): when set, the server registers GET /moe/telemetry and
// GET /moe/info, answering with whatever JSON the host writes into buf (C-string style —
// only POD crosses the boundary). `what` is "telemetry" or "info". Returns the number of
// bytes written (> 0), or <= 0 for "unavailable" (the route answers 503).
extern int (*g_srv_moe_json)(const char * what, char * buf, int cap);

// Speculative draft cap (spec-decode K=2 wiring): when > 0, every generated draft is
// truncated to this many tokens right after common_speculative_draft(), before the batch
// is built and before n_draft_total counts it. The moert-server host sets 1: its fused
// dual-tier verifier (MOE_DUAL_SPEC_K=2) handles n_tokens <= 2 — a longer verify batch
// would silently fall through to the cold-precision stock expert path — and under the
// loader cut's n_ubatch=2 a wider batch would split across sub-batches (post_decode
// rejects spec indices spanning sub-batches). 0 = uncapped (stock behavior). Data-only
// trampoline: the policy that picks the value lives in the host.
extern int g_srv_moe_spec_draft_max;

// Adaptive (alpha-gated) speculative draft width. When set, it OVERRIDES the fixed
// g_srv_moe_spec_draft_max: called once per fresh-draft decode step, BEFORE the drafter runs,
// with the previous verify's (accepted, proposed) token counts; it returns the draft width to
// use THIS step (clamped to [1, spec-draft-n-max]). Because the cap is applied to the drafter's
// n_max, a narrower return also cuts the drafter's autoregressive forwards (not just a post-hoc
// truncation) — the point of the gate. Return <= 0 to fall back to the fixed cap this step.
// Data-only trampoline: the policy (an EMA of acceptance -> width ladder) lives in the host.
extern int (*g_srv_moe_spec_draft_fn)(int prev_accepted, int prev_proposed);

// C5: cumulative EMITTED generation tokens (monotonic, process-lifetime). on_token_end fires
// once per decode CYCLE, so under speculative decoding it undercounts the user-visible rate
// (a spec_k-wide verify emits up to spec_k tokens in one cycle). Every emitted generation
// token increments this: +1 per generating-slot cycle, +accepted per speculative verify.
// The host derives a display tok/s from deltas of this counter. Read via the accessor.
extern long long g_srv_moe_gen_tokens;
long long llama_server_moe_gen_tokens();

// Exported from llama-server-impl (defined in server.cpp) so a host executable can set the
// hooks across the DLL boundary (WINDOWS_EXPORT_ALL_SYMBOLS exports functions, not data).
void llama_server_set_moe_hooks(void (*on_prompt_tokens)(int),
                                void (*on_prompt_end)(),
                                void (*on_token_end)(),
                                void (*on_turn_end)(),
                                void (*on_ready)());
void llama_server_set_moe_json_provider(int (*moe_json)(const char *, char *, int));
void llama_server_set_moe_spec_draft_max(int n);
void llama_server_set_moe_spec_draft_fn(int (*fn)(int prev_accepted, int prev_proposed));

// Variable-KV S6a (POST /moe/rebuild_k, child side; the router proxies it): schedule a hot-slab
// rebuild at a new uniform K. The handler only SCHEDULES (data-only trampoline into the host);
// the host applies it at the next turn boundary on the slot thread (quiescent point). Returns
// > 0 = scheduled, <= 0 = rejected with a reason written into err.
extern int (*g_srv_moe_rebuild_k)(int k, char * err, int errcap);
void llama_server_set_moe_rebuild_k(int (*fn)(int, char *, int));

// Variable-KV Phase 2 / S6b (POST /moe/rebuild_ctx, child side; the router proxies it): rebuild
// the target+draft contexts at a new n_ctx, replaying the active conversation's KV. Three-pointer
// protocol, all null-by-default:
//   g_srv_moe_rebuild_ctx(n, err, cap)  HTTP thread -> host: validate + SCHEDULE. >0 = scheduled.
//   g_srv_moe_ctx_rebuild_take()        slot thread -> host, polled at the all-idle quiescent
//                                       point of update_slots: consume the pending n_ctx (0 =
//                                       none). The host uses this hook to evict the expert slab
//                                       BEFORE the rebuild (no transient VRAM double-occupancy).
//   g_srv_moe_ctx_rebuild_done(ok, n)   fork -> host, after the rebuild: restore the slab + log.
// Variable-KV Phase 3: the automatic rebalance trigger. Fired at TASK LAUNCH (quiescent: the task
// is being assigned to the single slot, nothing decoded yet this iteration) with the incoming
// turn's need. The host runs its capacity policy and returns the n_ctx the turn should run at, or
// 0 to keep the current one. A nonzero return makes the fork rebalance the context RIGHT THERE,
// before the task starts — so the KV replay carries the previous conversation into the new size.
// The host may also prepare (evict/shrink the expert slab, re-arm staging) before returning.
extern int  (*g_srv_moe_plan_ctx)(int n_prompt_tokens, int n_predict, int n_ctx_cur);
void llama_server_set_moe_plan_ctx(int (*fn)(int, int, int));

// Variable-KV: fired by rebalance_ctx right after the OLD target context is
// freed and right BEFORE the new one is created — the point of maximum available VRAM during the
// swap, and the earliest point real free-VRAM measurement is meaningful. `n_ctx_new` lets the host
// size the hot slab for the INCOMING context's predictable KV cost in this one step, instead of
// reacting to it after the fact (without this, the slab stayed at its old, larger K
// through the swap and briefly overlapped the new context's KV, measured hitting 0 MiB free).
// The host also uses this call to arm q4stream staging for the reserve (mirroring the load path's
// own "begin before load_model" trick) at the moment least likely to force it to evict anything.
extern void (*g_srv_moe_pre_ctx_create)(int n_ctx_new);
void llama_server_set_moe_pre_ctx_create(void (*fn)(int));

extern int  (*g_srv_moe_rebuild_ctx)(int n_ctx, char * err, int errcap);
extern int  (*g_srv_moe_ctx_rebuild_take)();
extern void (*g_srv_moe_ctx_rebuild_done)(int ok, int n_ctx);
void llama_server_set_moe_rebuild_ctx(int (*schedule)(int, char *, int),
                                      int (*take)(),
                                      void (*done)(int, int));

// M4 lifecycle (router mode). When the plan provider is set AND the server runs as a
// ROUTER (no model), it registers:
//   GET  /moe/plan  -> provider JSON (the host spawns `moert-server plan` — the CUDA
//                      probe lives in that short-lived subprocess, keeping the router's
//                      no-GPU invariant). M6b1: `ctx` = the ?ctx=N query param parsed by
//                      the route (0 = host default) — the planner's KV budget scales;
//   POST /moe/exit  -> graceful full shutdown (children unloaded by clean_up);
//   GET  /moe/telemetry|/moe/info -> NO-AUTOLOAD proxy to the single running child, or
//                      {"state":"unloaded"|"loading"} — the stock proxy would autoload
//                      the model on a telemetry poll, which would fight auto-unload.
extern int (*g_srv_moe_plan)(int ctx, char * buf, int cap);
void llama_server_set_moe_plan_provider(int (*plan_json)(int, char *, int));

// M4 plan->profile save (router mode): POST /moe/profiles hands the request body to
// the host (which owns moert-profiles.json + the generated preset INI). Returns > 0
// on success (the fork then re-runs load_models() so the new profile appears in
// /models immediately), <= 0 on failure with a reason written into err.
extern int (*g_srv_moe_profiles_save)(const char * body_json, char * err, int errcap);
void llama_server_set_moe_profiles_save(int (*save)(const char *, char *, int));

// Ask the running server (router or child) to shut down cleanly — same path as SIGINT,
// safe to call from any thread. The child-side auto-unload watchdog uses this.
void llama_server_request_shutdown();

// Observability context sink (M2): fired once right after the target model+context are
// created (before any task). Hands the host the llama handles for read-only probes
// (live KV bytes, weight sizes) that feed /moe/telemetry's memory partition — the host
// must never mutate through them. Single-model server (no draft handles passed).
struct llama_context;
struct llama_model;
extern void (*g_srv_moe_on_ctx)(struct llama_context * ctx, const struct llama_model * model);
void llama_server_set_moe_ctx_sink(void (*on_ctx)(struct llama_context *, const struct llama_model *));

// [moe-dual] the SPECULATIVE side's handles, delivered separately because the sink
// above necessarily fires BEFORE the drafter exists (the drafter is built against ctx_tgt, so it
// cannot precede it). Without this the host's VRAM accounting could not name the drafter's weights
// or its compute buffer, and had to lump them into an unexplained residual alongside the CUDA
// context and allocator slack — which is exactly what made a ~900 MiB block unattributable.
// Fired after the draft context is created, on BOTH construction paths and on every ctx rebuild.
//   model_dft == nullptr means the MTP path that runs against the TARGET model: there are no
//   separate drafter weights to account for (ctx_dft is still a real, separate context).
// Read-only, same contract as the sink above: the host must never mutate through these.
extern void (*g_srv_moe_on_spec_ctx)(struct llama_context * ctx_dft, const struct llama_model * model_dft);
void llama_server_set_moe_spec_ctx_sink(void (*on_spec_ctx)(struct llama_context *, const struct llama_model *));

// --- streamed (q4stream) prefill trampolines (C3) --------------------------------------
// Host eval-callback installed on the TARGET context's params before common_init_from_params.
// The moert host uses it as the Q4Stream staging sync point (asks only for ffn_moe_topk, and
// only during a streamed prompt phase — outside it the callback asks for nothing and the
// sched runs single-submission as if no callback were set).
struct ggml_tensor;
typedef bool (*srv_moe_cb_eval_t)(struct ggml_tensor * t, bool ask, void * user_data);
extern srv_moe_cb_eval_t g_srv_moe_cb_eval;
extern void *            g_srv_moe_cb_eval_ud;
void llama_server_set_moe_cb_eval(srv_moe_cb_eval_t cb, void * user_data);

// Prompt-batch token cap: called while filling a decode batch with prompt tokens;
// `n_remaining` = uncached prompt tokens left for the slot. The returned value caps how many
// go into THIS batch (values < 1 are treated as 1; return INT32_MAX for no cap). The moert
// host returns spec_k below the q4stream break-even so short prompt suffixes keep the fused
// per-token path, and no cap above it so the whole prompt lands in one streamed decode.
extern int (*g_srv_moe_prompt_cap_fn)(int n_remaining);
void llama_server_set_moe_prompt_cap(int (*fn)(int n_remaining));

// Prefix reuse for the streamed prefill (claude dump design-prefill-transfer P4): when > 0, a
// "stream whole" prompt still stops this many tokens before its end, so the SWA context checkpoint
// upstream creates before the last batch covers the prompt minus that tail and the next turn can
// restore it instead of re-prefilling the whole history. The tail runs on the fused path (it is
// <= spec_k tokens, the phase closes) against the resident slab. 0 = off (the whole suffix in one
// decode, no checkpoint: the pre-P4 behavior). Upstream's own break ({4 + n_ubatch, 4}) is kept for
// prompts that are not streamed whole.
extern int g_srv_moe_ckpt_tail;
void llama_server_set_moe_ckpt_tail(int n_tail);

// Pre-decode gate: fired right before every llama_decode on the target context, with the
// batch classified by the processing slot's phase (n_parallel = 1 invariant, as above).
// Returning false REFUSES the batch: the server errors the processing slots cleanly instead
// of decoding (the moert host refuses a batched prompt when q4stream staging cannot be
// (re)initialized — never a silent precision downgrade). Unset = always allowed.
extern bool (*g_srv_moe_pre_decode)(bool is_prompt, int n_tokens);
void llama_server_set_moe_pre_decode(bool (*fn)(bool is_prompt, int n_tokens));
// [llapis] bench content trace: the token ids of a batch the pre-decode hook accepted (unset = not called)
extern void (*g_srv_moe_batch_tokens)(const int32_t * tokens, int n_tokens);
void llama_server_set_moe_batch_tokens(void (*fn)(const int32_t * tokens, int n_tokens));

// --- C5.2 router admission + child-exit observation (trampoline-only) -------------------
// Spawn gate: fired by the ROUTER inside server_models::load(), under the mapping lock,
// right before a child process is spawned. Return false to REFUSE the load; the gate may
// write a human-readable reason into `reason` (C string, `reason_cap` bytes) which lands in
// the thrown error (and thus the HTTP response / router log). The moert host uses it for
// memory-admission + crash-backoff so a failing child is never respawned blindly.
extern bool (*g_srv_moe_spawn_gate)(const char * name, char * reason, int reason_cap);
void llama_server_set_moe_spawn_gate(bool (*fn)(const char *, char *, int));

// Child-exit observation: fired by the ROUTER's child-monitor thread once a child process
// has exited and its exit code is known (after status flips back to UNLOADED). The host
// records {name, exit_code, time} to drive the spawn gate's backoff. Data-only trampoline.
extern void (*g_srv_moe_child_exit)(const char * name, int exit_code);
void llama_server_set_moe_child_exit(void (*fn)(const char *, int));
