#pragma once

// ===========================================================================
// prefill_policy — THE single home of "does this prompt suffix stream, or fuse?"
// ===========================================================================
// Header-only, no deps, no CUDA, no llama types: three inputs and a path out. Extracted
// 2026-08-11 (findings 133's criterion: a decision is extracted once it has already cost GPU
// time to verify). This one cost three four-boot A/Bs on 2026-08-09 alone — and the entry
// that sent me looking, findings 124, described it wrongly:
//
//   "MOE_DUAL_PREFILL_MIN decides from the whole prompt; it must decide from the uncached
//    length. This is an `if`."
//
// The `if` was already there. `n_remaining` is the UNCACHED suffix, because
// server-context.cpp trims the slot to the reused prefix (keep_first(n_past)) before calling
// the hook, so a 2680-token prompt with 2663 cached arrives as 17. What was wrong was the
// VALUE, and no single value can be right — see below.
//
// THE TWO REGIMES, measured (findings 125, `speed`, 5120 MiB pinned):
//
//   streamed   FLAT ~2.9-3.9 s at any length. It is the 7.67 GiB pool stream, not the prompt:
//              3.9 s at n=19 and 2.9 s at n=272.
//   fused      ~25 ms/token with the hot slab RESIDENT   -> crossing at ~125 tokens
//              ~418 ms/token with the slab EVICTED       -> crossing at ~13 tokens
//
// A 17x spread on one condition nobody had declared. A 42-token first prompt costs 17.6 s
// fused and 5.4 s streamed; the same size warm costs 0.5 s fused and 3.9 s streamed. So the
// gate consults the slab, and a cold server streams whatever the length — which is also
// exactly the 2026-08-08 shipped behaviour, kept bit-for-bit where its own measurement was
// taken. The constant has now been wrong twice in opposite directions (384, then 1) and both
// times for the same reason: one number fitted to one regime, then applied to both.

namespace prefill {

enum class path {
    fused,      // spec_k tokens per decode, dual-tier op, q4stream phase CLOSED
    streamed,   // the whole suffix in one decode, pool streamed once
};

struct params {
    // The streamed/fused crossing for a WARM server. 128 is the measured value; 1 restores the
    // 2026-08-08 always-stream behaviour in both regimes. It cannot force fused on a COLD
    // server — that costs 418 ms/token and is not a knob.
    int warm_min = 128;
};

// The decision. `slab_resident` is the warm/cold discriminator: it is false only between the
// C5 boot release and the first prompt end, i.e. precisely on the first prompt of a session.
inline path decide(int n_remaining, bool slab_resident, const params & p) {
    if (slab_resident && n_remaining < p.warm_min) {
        return path::fused;
    }
    return path::streamed;
}

// Graceful degradation (C3.6). The staging buffer (~589 MiB) is (re)allocated per prompt, and
// the host probes it only when `decide` asked to stream. An alloc failure (a browser eating
// VRAM, KV growth) must send the prompt down the fused path, never error the request —
// pre_decode's hard refusal stays as the last-resort invariant behind this.
inline path on_staging_unavailable() {
    return path::fused;
}

// How the chosen path is encoded for the server's prompt-batch hook: the number of prompt
// tokens allowed into THIS batch. INT32_MAX means "no cap", which the fork reads as
// `moe_stream_whole`.
inline int batch_cap(path pth, int spec_k) {
    return pth == path::fused ? spec_k : 0x7fffffff;
}

} // namespace prefill
