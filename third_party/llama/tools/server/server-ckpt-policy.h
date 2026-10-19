#pragma once

// ===========================================================================
// server-ckpt-policy — THE single home of "may this context checkpoint be restored?"
// ===========================================================================
// Header-only, no dependencies, no llama types: five ints in, a decision out. That is the
// whole point. This arithmetic decides whether a returning turn reuses its prefix or
// re-prefills it from scratch, and it has now been diagnosed WRONG in writing three times,
// each time costing a 15-minute GPU A/B to settle:
//
//   findings 124  "the cause is --checkpoint-min-step spacing"      -> it is not; no checkpoint
//                                                                      is created at all
//   findings 128  "fix: also checkpoint at SLOT_STATE_DONE_PROMPT"  -> refuted by head_is_usable()
//                                                                      below, in one line
//   findings 129  "take the checkpoint after the decode"            -> same refutation, measured
//                                                                      the hard way instead
//
// It is five integers. It never needed a GPU. Extracted 2026-08-11 so the next question about
// it costs a second (test-ckpt-policy) instead of an afternoon.
//
// Behaviour is IDENTICAL to the inline code it replaces (server-context.cpp update_slots);
// the golden transcript is the gate on that.

namespace ckpt_policy {

// The largest pos_min a checkpoint may have and still be useful.
//
// `pos_next` is where the new prompt diverges from the cached one. `n_swa` is the sliding
// window (0 when --swa-full builds a unified cache). `has_new_tokens` is false when the
// prompt is fully contained in the cache, which needs one extra token of slack so that at
// least one token is actually processed [TAG_PROMPT_LOGITS].
inline int restore_threshold(int pos_next, int n_swa, bool has_new_tokens) {
    const int t = pos_next - n_swa - (has_new_tokens ? 0 : 1);
    return t > 0 ? t : 0;
}

// Is a checkpoint needed at all? With a unified KV (n_swa == 0, i.e. --swa-full) the cache
// reports pos_min == 0 while the threshold sits at ~pos_next, so this is false and the whole
// checkpoint machinery is bypassed -- which is exactly why stock llama.cpp reuses a prefix
// where we re-prefill it (findings 131). An iSWA cache reports the window start, which is
// >= the threshold, so it always needs one.
inline bool needs_checkpoint(int cache_pos_min, int restore_thold) {
    return cache_pos_min >= restore_thold;
}

// May THIS checkpoint be restored?
//   - a checkpoint that already covers past the divergence point is useless (it would have to
//     be rolled back itself) [TAG_CHECKPOINTS_FIX_POS_MIN];
//   - otherwise its window must start STRICTLY BEFORE the threshold, so that restoring it can
//     rebuild the window from before the resume point. pos_min == 0 is the special case of a
//     checkpoint that holds the whole sequence.
inline bool usable(int ckpt_pos_min, int ckpt_pos_max, int pos_next, int restore_thold) {
    if (ckpt_pos_max > pos_next) {
        return false;
    }
    return ckpt_pos_min < restore_thold || ckpt_pos_min == 0;
}

// Where processing resumes from a restored checkpoint.
inline int resume_pos(int ckpt_pos_min, int ckpt_pos_max, int pos_next) {
    const int lo = ckpt_pos_min + 1 > ckpt_pos_max ? ckpt_pos_min + 1 : ckpt_pos_max;
    return pos_next < lo ? pos_next : lo;
}

// ---------------------------------------------------------------------------
// THE RULE THAT KEEPS BEING REDISCOVERED, stated as code so it can be TESTED instead of
// re-derived on a GPU. A checkpoint is born when the prompt batch is cut, `born_early`
// tokens before the end of a prompt that ends at `pos_max`. By construction its window is
// [pos_max - born_early - n_swa, pos_max - born_early]. The NEXT turn diverges `divergence`
// tokens before that same end, so its threshold is pos_max - divergence - n_swa.
//
// usable() then reduces to:   pos_max - born_early - n_swa  <  pos_max - divergence - n_swa
// i.e. simply:                born_early > divergence
//
// n_swa cancels. pos_max cancels. **A checkpoint is restorable if and only if it was born
// STRICTLY EARLIER than the point where the next turn diverges.** That is the entire
// mechanism, and it is why upstream cuts the prompt batch at 4 + n_ubatch tokens from the
// end: it buys born_early = 68 against a chat template's divergence of ~16.
//
// Corollary, which is what findings 128 and 129 both proposed and neither could work:
// a checkpoint minted AFTER the whole prompt is decoded has born_early = 0, so it is
// unusable for EVERY divergence >= 0 -- including 0, where the strict `<` still refuses it.
inline bool usable_when_born(int born_early, int divergence, int pos_max, int n_swa) {
    const int ckpt_pos_max = pos_max - born_early;
    const int ckpt_pos_min = ckpt_pos_max - n_swa;
    const int pos_next     = pos_max - divergence;
    return usable(ckpt_pos_min, ckpt_pos_max, pos_next,
                  restore_threshold(pos_next, n_swa, /*has_new_tokens=*/true));
}

// The special case above: born at the head, i.e. after the prompt has been decoded.
inline bool head_is_usable(int pos_max, int n_swa, int divergence) {
    return usable_when_born(/*born_early=*/0, divergence, pos_max, n_swa);
}

} // namespace ckpt_policy
