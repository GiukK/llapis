// test-q4s-phase-policy — when a decode batch may run with the q4stream phase ARMED.
//
// WHY THIS FILE EXISTS, and why it is the most valuable of the four. The other extracted
// decisions fail loudly (an errored request) or slowly (seconds of TTFT). This one fails
// QUIETLY: an armed phase over a reused graph computes against whatever layer was staged last,
// which is garbage KV, which comes out as confident, well-formed, wrong text. It was seen live
// as `draft 0/231` and `<unused49>`.
//
// That also means the golden transcript is the WEAKEST gate here: a transcript recorded on a
// build that already had the bug would reproduce it character for character. The invariant has
// to be checked directly, and it can be checked EXHAUSTIVELY -- something no GPU run could do.
//
// No GPU, no model, no server. Run it: ctest -R q4s-phase-policy

#include "q4s_phase_policy.hpp"

#include <cstdio>

static int g_fail = 0;

static void check(bool ok, const char * what) {
    std::printf("%s  %s\n", ok ? "  ok  " : "  FAIL", what);
    if (!ok) ++g_fail;
}

int main() {
    using namespace q4s;
    const bool PROMPT = true, GEN = false;

    std::printf("the three regimes\n");
    check(decide(PROMPT, 4, 4) == action::close_and_pass,
          "prompt batch of exactly spec_k: CLOSE (it is generation-shaped)");
    check(decide(PROMPT, 1, 4) == action::close_and_pass, "prompt batch below spec_k: CLOSE");
    check(decide(PROMPT, 5, 4) == action::arm_and_pass,
          "prompt batch above spec_k: ARM, a fresh graph against just-in-time staging");
    check(decide(PROMPT, 2678, 4) == action::arm_and_pass, "a whole prompt: ARM");
    check(decide(GEN, 1, 4)  == action::close_and_pass, "generation: CLOSE");
    check(decide(GEN, 64, 4) == action::close_and_pass,
          "generation is CLOSED at ANY width -- is_prompt decides first");

    std::printf("staging is required exactly when armed\n");
    check(requires_staging(action::arm_and_pass),  "armed needs staging, or the batch is refused");
    check(!requires_staging(action::close_and_pass), "closed needs nothing");

    // ---------------------------------------------------------------- the case that corrupted
    // findings C3.3: the fused prompt chunks the spec_k cap feeds through were run with the
    // phase still open. Same shape as a generation decode -> graph reused -> no re-arm ->
    // computed against the last staged layer. This is that exact batch.
    std::printf("the batch that actually corrupted output\n");
    check(decide(PROMPT, 4, 4) == action::close_and_pass,
          "the spec_k-capped prompt chunk is CLOSED, which is what fixed C3.3");
    check(safe(decide(PROMPT, 4, 4), PROMPT, 4, 4), "...and it satisfies the invariant");
    check(!safe(action::arm_and_pass, PROMPT, 4, 4),
          "arming that same batch VIOLATES the invariant -- the bug, stated as a predicate");
    check(!safe(action::arm_and_pass, GEN, 64, 4),
          "arming a generation batch violates it too, at any width");

    // ---------------------------------------------------------------- exhaustive
    // A GPU run can sample this space; a test can cover it. Every spec_k a config can resolve
    // to, every batch width up to a wide prefill split, both phases.
    std::printf("EXHAUSTIVE: the invariant holds for every input the runtime can present\n");
    {
        long long checked = 0;
        bool all_safe = true, all_armed_are_wide_prompts = true;
        for (int spec_k = 1; spec_k <= 8; ++spec_k) {
            for (int n = 0; n <= 4096; ++n) {
                for (int p = 0; p <= 1; ++p) {
                    const bool is_prompt = (p == 1);
                    const action a = decide(is_prompt, n, spec_k);
                    if (!safe(a, is_prompt, n, spec_k)) all_safe = false;
                    if (a == action::arm_and_pass && !(is_prompt && n > spec_k)) {
                        all_armed_are_wide_prompts = false;
                    }
                    ++checked;
                }
            }
        }
        std::printf("        (%lld combinations)\n", checked);
        check(all_safe, "no input produces an unsafe action");
        check(all_armed_are_wide_prompts,
              "every ARMED action is a prompt batch strictly wider than spec_k -- never a "
              "generation-reusable shape");
    }

    // The converse, so the test cannot pass by never arming anything at all.
    std::printf("and it still arms when it should (not vacuously safe)\n");
    {
        int armed = 0;
        for (int spec_k = 1; spec_k <= 8; ++spec_k) {
            for (int n = 0; n <= 4096; ++n) {
                if (decide(PROMPT, n, spec_k) == action::arm_and_pass) ++armed;
            }
        }
        check(armed > 30000, "the wide-prompt path is reached across the space");
    }

    std::printf(g_fail == 0 ? "\nPASS\n" : "\nFAIL: %d check(s)\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
