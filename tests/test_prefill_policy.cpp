// test-prefill-policy — the streamed-vs-fused prefill gate, on the CPU, in milliseconds.
//
// WHY THIS FILE EXISTS. This decision picks between a flat ~3.2 s pool stream and a per-token
// fused path whose cost swings 17x with one condition (is the hot slab resident). Getting it
// wrong is worth seconds on every turn, and it has been wrong twice in opposite directions:
// 384 (never measured) and 1 (measured on a COLD server, applied to every prompt).
//
// Reading it off a GPU run took three four-boot A/Bs on 2026-08-09. The cases below are those
// runs, as assertions. No GPU, no model, no server. Run it: ctest -R prefill-policy

#include "prefill_policy.hpp"

#include <cstdio>

static int g_fail = 0;

static void check(bool ok, const char * what) {
    std::printf("%s  %s\n", ok ? "  ok  " : "  FAIL", what);
    if (!ok) ++g_fail;
}

static void check_eq(int got, int want, const char * what) {
    const bool ok = got == want;
    std::printf("%s  %s (got %d, want %d)\n", ok ? "  ok  " : "  FAIL", what, got, want);
    if (!ok) ++g_fail;
}

int main() {
    using namespace prefill;
    const params dflt;                       // warm_min = 128, the measured crossing
    const bool WARM = true, COLD = false;

    check_eq(dflt.warm_min, 128, "shipped default is the measured warm crossing");

    // ---------------------------------------------------------------- the warm regime
    // findings 125: fused ~25 ms/token warm vs a flat ~3.2 s stream -> cross at ~125 tokens.
    std::printf("warm server (hot slab resident)\n");
    check(decide(17,  WARM, dflt) == path::fused,
          "17-token suffix fuses: 0.9 s measured against 3.2 s streamed");
    check(decide(66,  WARM, dflt) == path::fused,   "66 tokens still fuses (1.6 s vs 3.8 s)");
    check(decide(127, WARM, dflt) == path::fused,   "one below the threshold");
    check(decide(128, WARM, dflt) == path::streamed, "at the threshold it streams");
    check(decide(134, WARM, dflt) == path::streamed,
          "134 streams: measured 3.0 s streamed against 3.3 s fused, the crossing");
    check(decide(2678, WARM, dflt) == path::streamed, "a full prompt always streams");

    // ---------------------------------------------------------------- the cold regime
    // The whole reason the gate consults the slab. Fused costs ~418 ms/token here: a 42-token
    // first prompt is 17.6 s fused against 5.4 s streamed.
    std::printf("cold server (slab still evicted -- the first prompt of a session)\n");
    check(decide(1,    COLD, dflt) == path::streamed, "even ONE token streams when cold");
    check(decide(42,   COLD, dflt) == path::streamed, "42 tokens: 5.4 s streamed vs 17.6 s fused");
    check(decide(127,  COLD, dflt) == path::streamed, "below the warm threshold, still streams");
    check(decide(2678, COLD, dflt) == path::streamed, "and a full cold prompt, obviously");

    // ---------------------------------------------------------------- the knob's span
    // MOE_DUAL_PREFILL_MIN must still reach both shipped behaviours, and must NOT be able to
    // force the fused path onto a cold server -- 418 ms/token is not a knob.
    std::printf("MOE_DUAL_PREFILL_MIN\n");
    const params legacy{ 1 };
    check(decide(17, WARM, legacy) == path::streamed,
          "MIN=1 restores the 2026-08-08 always-stream behaviour when warm");
    check(decide(1,  WARM, legacy) == path::streamed, "...at every length");
    const params force_fused{ 100000 };
    check(decide(2678, WARM, force_fused) == path::fused,
          "a large MIN forces fused while the slab is resident (the A/B arm)");
    check(decide(2678, COLD, force_fused) == path::streamed,
          "but it CANNOT force fused on a cold server, at any value");
    check(decide(1,    COLD, force_fused) == path::streamed, "...not even for one token");

    // ---------------------------------------------------------------- degradation + encoding
    std::printf("graceful degradation and the wire encoding\n");
    check(on_staging_unavailable() == path::fused,
          "staging alloc failure falls back to fused, never errors the request");
    check_eq(batch_cap(path::fused, 4), 4, "fused caps the batch at spec_k");
    check_eq(batch_cap(path::streamed, 4), 0x7fffffff,
             "streamed is INT32_MAX, which the fork reads as moe_stream_whole");

    // ---------------------------------------------------------------- the regression guard
    // findings 124 said the gate compared against the WHOLE prompt and the fix was an `if`.
    // It already compared against the uncached suffix. Encode the case that settles it: a
    // 2680-token prompt with 2663 cached arrives here as 17, and 17 is what decides.
    std::printf("findings 124's claim, as a guard\n");
    check(decide(/*n_remaining=*/17, WARM, dflt) == path::fused,
          "a 2680-token prompt with 2663 cached is a 17-token prompt, and fuses");

    std::printf(g_fail == 0 ? "\nPASS\n" : "\nFAIL: %d check(s)\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
