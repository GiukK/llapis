#pragma once

// ===========================================================================
// vkv_policy — THE single home of the variable-KV capacity/resizing policy.
// ===========================================================================
// User rule: the resizing policy must be CENTRALIZED so future changes
// happen in exactly one place. Everything that decides WHEN to resize the context
// and TO WHAT size lives here; the K side is deliberately NOT here — K_new is
// always K_max(C_new) from the runtime's MEASURED VRAM clamp, never
// a policy choice. Header-only, no deps beyond <cstdlib>, CPU-unit-testable
// (test-vkv-policy).
//
// Algorithm: Candidate B "minimal-overshoot grow-to-fit" (
// Part B). The ctx ladder is LINEAR (multiples of `gran`), never exponential — the
// measured KV slope is ~1 K per ~5540 tokens, so linear steps cost a gentle 1-3 K
// each where chats actually live; a doubling ladder would collapse K by up to 12 in
// one step (rejected by the user's directional spec).
//
//   need        = history_tokens + max_generation + slack
//   grow when   need > C_cur:      C_new = roundup(need + headroom, gran)
//   shrink when depth + headroom fits a FULL band below C_cur (hysteresis: the
//               freed space must be >= gran + headroom, so a boundary-straddling
//               conversation never oscillates), and only opportunistically
//               (the caller additionally spaces shrinks by turns/cooldown).

#include <cstdlib>

namespace vkv {

struct Params {
    int min_ctx  = 8192;   // the boot allocation and the ladder floor
    int max_ctx  = 262144; // ceiling (the model/train limit; the 8 GB physics wall binds earlier)
    int gran     = 8192;   // ladder granularity (aligned with the q4stream single-pass envelope)
    int headroom = 8192;   // extra tokens allocated past `need` (~8 organic turns before the next
                           // resize at the observed ~1k tok/turn growth; amortizes the rebuild)
    int slack    = 512;    // per-turn safety margin inside `need`
    // Planning reserve for the turn's GENERATION, in tokens. This is NOT the client's
    // `max_tokens`: a harness sends its configured ceiling on every single request, so
    // planning on that number makes an agent's first two-line answer allocate a context
    // sized for its longest possible one. With the provider JSON now declaring
    // maxTokens 65536 that is not a rounding error — it would grow the
    // context to the ceiling on request #1 and undo the whole point of booting at 8192.
    // So `plan_grow` charges min(max_gen, gen_reserve). 8192 = one full ladder rung, and
    // also the maxTokens the provider JSON declared before this change, so every harness
    // that worked yesterday plans identically today.
    // Cost of being wrong low: a turn that really does generate past the reserve meets the
    // context wall and ends with finish_reason=length instead of silently corrupting
    // anything; the NEXT turn sees the longer history and grows for it. Cost of being
    // wrong high: every request pays for a context it will not use. The asymmetry is why
    // this is a reserve and not a forecast.
    int gen_reserve = 8192;
};

// Env overrides (MOE_DUAL_VKV_MIN_CTX / _MAX_CTX / _GRAN / _HEADROOM / _SLACK) on top of the
// defaults above — parsed once by the host at init, like every other MOE_DUAL_* knob.
inline Params params_from_env() {
    Params p;
    // Both prefixes: MOE_DUAL_VKV_* is canonical, LLAPIS_VKV_* is the one that survives the
    // launcher→router→child hops (the launcher clears MOE_* for the router; LLAPIS_* is kept).
    // HEADROOM accepts 0, so it is read separately from the >0-only knobs.
    auto one = [](const char * k, int & v, bool allow_zero) {
        if (const char * s = std::getenv(k); s && s[0]) {
            const int n = std::atoi(s);
            if (n > 0 || (allow_zero && n == 0)) v = n;
        }
    };
    auto ov = [&one](const char * a, const char * b, int & v, bool allow_zero = false) {
        one(a, v, allow_zero);
        one(b, v, allow_zero);
    };
    ov("MOE_DUAL_VKV_MIN_CTX",  "LLAPIS_VKV_MIN_CTX",  p.min_ctx);
    ov("MOE_DUAL_VKV_MAX_CTX",  "LLAPIS_VKV_MAX_CTX",  p.max_ctx);
    ov("MOE_DUAL_VKV_GRAN",     "LLAPIS_VKV_GRAN",     p.gran);
    ov("MOE_DUAL_VKV_HEADROOM", "LLAPIS_VKV_HEADROOM", p.headroom, /*allow_zero=*/true);
    ov("MOE_DUAL_VKV_SLACK",    "LLAPIS_VKV_SLACK",    p.slack);
    ov("MOE_DUAL_VKV_GEN_RESERVE", "LLAPIS_VKV_GEN_RESERVE", p.gen_reserve);
    if (p.min_ctx > p.max_ctx) p.min_ctx = p.max_ctx;
    return p;
}

inline int round_up(int v, int gran) { return ((v + gran - 1) / gran) * gran; }

// KV cost model from llapis/preset.ps1's static K_max(ctx) clamp, REHABILITATED
// by declared it broken for the live server build — "predicted 319 MiB at
// ctx=16384, real cost was ~1168 MiB" — and banned it from the resize path. That measurement was
// taken THROUGH the 2734 MiB compute-reserve bug: most of the "real cost" was the reserve,
// not the KV. On a clean build the per-context footprint is exactly affine in n_ctx — measured
// 601 / 763 / 925 MiB at 16384 / 24576 / 32768, i.e. slope 324/16384 = 0.0198 MiB/token, which
// agrees with the six-point sweep's 0.0195 to 1.5%. The slope is now the most independently
// validated number in the project. See ctx_cost_predict_mib below for how it is (and is not) used.
constexpr double KV_MIB_PER_TOKEN = 0.0195;
inline long long kv_mib_for_ctx(int n_ctx) { return (long long) (KV_MIB_PER_TOKEN * n_ctx + 0.5); }

// --- ctx-swap cost model ---------------------------------------------------------
// Predicts what the NEW context will occupy, so the pre-swap slab fit can remove EXACTLY the
// experts that no longer fit — the user's spec, literally: "il numero di K possibili in VRAM viene
// ricalcolato e ... vengono rimossi i K_old - K_new esperti che non entrano piu". Without a
// prediction the swap has to assume the worst and collapse the slab to the quality floor, which
// throws away the learned hot set on every resize and then refills it lazily over the next turn.
//
// What is modelled and what is MEASURED is the whole point, and it is where went wrong: it
// modelled the ABSOLUTE cost. The intercept (SWA sub-cache, output buffers, per-context overhead)
// is build- and box-specific — ~277 MiB here — and getting it wrong is what left the old slab and
// the new context resident at once, at 0 MiB free. So the intercept is NEVER modelled: the caller
// MEASURES the OLD context's real footprint (free VRAM before its teardown vs after) and only the
// DELTA between the two ctx sizes is extrapolated, with the validated slope above.
//
// Validated against the sweep, all three tiers: predicted 636 / 806 / 968 MiB vs real
// 601 / 763 / 925 — conservative by a consistent ~40 MiB. That is the SAFE direction: an
// over-estimate shrinks marginally too much and moe_ctx_rebuild_done's post-swap vkv_refit_k grows
// it back from a real measurement, whereas an under-estimate would OOM the context creation or
// spill it into WDDM shared memory. Callers must keep that asymmetry (never bias this down).
inline long long ctx_cost_predict_mib(int n_ctx_old, long long footprint_old_mib, int n_ctx_new) {
    // Sign-aware rounding: a plain `+ 0.5` truncates toward zero on the SHRINK direction (C_new <
    // C_old), which biases the predicted cost UP on grow and also up on shrink — harmless here but
    // wrong, and Phase 4's /compact down-resize runs entirely on the negative branch.
    const double    d     = KV_MIB_PER_TOKEN * (double) (n_ctx_new - n_ctx_old);
    const long long delta = (long long) (d >= 0.0 ? d + 0.5 : d - 0.5);
    const long long cost  = footprint_old_mib + delta;
    return cost > 0 ? cost : 0;
}

// The boot allocation: the ladder floor.
inline int boot_ctx(const Params & p) { return round_up(p.min_ctx, p.gran); }

// GROW decision at turn start. history = conversation tokens already held; max_gen = the turn's
// generation budget. Returns the new ctx (> c_cur, gran-aligned, clamped to max_ctx) or 0 = keep
// c_cur (the common short-chat case — zero cost). Returns 0 also when the need exceeds max_ctx
// (the caller must refuse/compact — never silently truncate).
inline int plan_grow(long long history, int max_gen, int c_cur, const Params & p) {
    // See Params::gen_reserve — `max_gen` is a client-declared CEILING, not a forecast, so it
    // is charged only up to the reserve. History is charged in full: it is already real.
    const int gen = (p.gen_reserve > 0 && max_gen > p.gen_reserve) ? p.gen_reserve : max_gen;
    const long long need = history + gen + p.slack;
    if (need <= c_cur) return 0;
    if (need > p.max_ctx) return 0;                    // caller refuses/compacts, loudly
    long long c_new = round_up((int) (need + p.headroom > p.max_ctx ? need : need + p.headroom), p.gran);
    if (c_new > p.max_ctx) c_new = p.max_ctx;
    return c_new > c_cur ? (int) c_new : 0;
}

// SHRINK decision (opportunistic — after /compact or history truncation). Hysteresis: only
// shrink when the depth fits a full band below, i.e. the reclaimed space is >= gran + headroom;
// a conversation sitting on a boundary never oscillates. Returns the new ctx or 0 = keep.
inline int plan_shrink(long long depth, int c_cur, const Params & p) {
    // a deep compact goes straight to the floor (max K reclaim); otherwise keep full headroom
    // above the remaining depth so the next grow is not immediate (anti-thrash, with the
    // full-band requirement below as the hysteresis)
    long long keep = depth + p.slack;
    if (keep > p.min_ctx) keep += p.headroom;
    int c_new = round_up((int) (keep < p.min_ctx ? p.min_ctx : keep), p.gran);
    if (c_new < p.min_ctx) c_new = p.min_ctx;
    return (c_cur - c_new >= p.gran) ? c_new : 0;
}

}  // namespace vkv
