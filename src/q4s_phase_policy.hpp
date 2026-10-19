#pragma once

// ===========================================================================
// q4s_phase_policy — THE single home of "may this decode batch run with the
// q4stream phase ARMED?"
// ===========================================================================
// Header-only, no deps: three inputs and an action out. Extracted 2026-08-11, last of the four
// in findings 133's list, and deliberately last because it is the one where being wrong is
// WORST and the golden transcript is WEAKEST.
//
// WHAT GOES WRONG WHEN THIS IS WRONG (the C3.3 corruption class). While the phase is OPEN the
// graph hook builds expert ops against the STAGED tensors. A decode whose graph is REUSED --
// same shape, no rebuild, therefore no pending re-arm -- computes against whatever layer was
// staged LAST. That is garbage KV, and it does not crash: it produces confident, well-formed,
// wrong text. It was seen live, as `draft 0/231` and `<unused49>` in the output.
//
// This is why the extraction matters more here than anywhere else in the list. The other three
// decisions fail loudly or slowly; this one fails QUIETLY, and a golden transcript recorded on
// a build that already had the bug would happily reproduce it character for character.
//
// THE THREE REGIMES, each individually validated on the GPU before this file existed:
//
//   n <= spec_k          CLOSE the phase, pass. The fused dual-tier op on arena/slab,
//                        bit-identical to the shipped fused default (the reserve covers it).
//                        This is the DANGEROUS shape: it is also the shape a generation decode
//                        has, so its graph is exactly the one that gets reused.
//   n >  spec_k, prompt  ARM (open the phase if needed) + chunk_begin. A fresh graph against
//                        just-in-time staging. Never the loader-cut stock shell path -- its
//                        batched reserve is ~2.7 GiB, measured OOM on 8 GB.
//   generation           CLOSE the phase, pass. Same as n <= spec_k in practice.
//
// The phase re-opens transparently on the next >spec_k prompt chunk (multi-part prompts).

namespace q4s {

enum class action {
    close_and_pass,  // close the phase (no-op if already closed), then let the batch run fused
    arm_and_pass,    // ensure the phase is open, re-arm staging for this chunk, then run streamed
};

// The decision. `spec_k` is the fused-path ceiling = the speculative verify width, so a batch of
// at most spec_k tokens is shape-identical to a generation decode.
inline action decide(bool is_prompt, int n_tokens, int spec_k) {
    if (!is_prompt || n_tokens <= spec_k) {
        return action::close_and_pass;
    }
    return action::arm_and_pass;
}

// An armed batch REQUIRES staging. If it cannot be obtained the host must refuse the batch
// loudly (the fork errors the request) -- never compute the prompt's experts against a stale
// staging buffer, and never silently downgrade precision.
inline bool requires_staging(action a) {
    return a == action::arm_and_pass;
}

// ---------------------------------------------------------------------------
// THE SAFETY INVARIANT, as a predicate so a test can check it exhaustively rather than by
// example. A batch may only be armed if it is a prompt batch STRICTLY WIDER than spec_k --
// i.e. never a shape that a generation decode could also produce, because that is the shape
// whose graph gets reused without a re-arm.
//
// Every corruption in this class is a violation of this one line.
inline bool safe(action a, bool is_prompt, int n_tokens, int spec_k) {
    if (a != action::arm_and_pass) {
        return true;             // a closed phase is always safe
    }
    return is_prompt && n_tokens > spec_k;
}

} // namespace q4s
