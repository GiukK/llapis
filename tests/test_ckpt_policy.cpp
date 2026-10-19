// test-ckpt-policy — the context-checkpoint restore decision, on the CPU, in milliseconds.
//
// WHY THIS FILE EXISTS. This arithmetic decides whether a returning turn reuses its prefix
// or re-prefills 2680 tokens from scratch. It was diagnosed wrong in writing three times
// (findings 124, 128, 129), and every time the only way anyone had to ask was a four-boot
// GPU A/B costing ~15 minutes. It is five integers. The cases below are exactly the ones
// those runs were spending an afternoon to discover.
//
// No GPU, no model, no server. Run it: ctest -R ckpt-policy

#include "server-ckpt-policy.h"

#include <cstdio>
#include <cstdlib>

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
    using namespace ckpt_policy;

    // ---------------------------------------------------------------- threshold
    std::printf("restore_threshold\n");
    check_eq(restore_threshold(2663, 1024, true),  1639, "iSWA, new tokens: pos_next - n_swa");
    check_eq(restore_threshold(2663, 1024, false), 1638, "fully cached: one extra token of slack");
    check_eq(restore_threshold(500,  1024, true),     0, "window longer than the sequence clamps to 0");
    check_eq(restore_threshold(2663, 0,    true),  2663, "swa_full (n_swa = 0): threshold is pos_next");

    // ---------------------------------------------------------------- is one needed at all
    std::printf("needs_checkpoint\n");
    // findings 131: with --swa-full the KV is unified and reports pos_min = 0, so the whole
    // checkpoint branch is skipped and reuse just works -- stock llama.cpp's behaviour, and
    // the reason it reuses 2663 tokens where we re-prefill them.
    check(!needs_checkpoint(/*cache_pos_min=*/0, restore_threshold(2663, 0, true)),
          "swa_full: unified cache never needs a checkpoint");
    // iSWA: the cache reports the window start, which is at or past the threshold.
    check(needs_checkpoint(/*cache_pos_min=*/1639, restore_threshold(2663, 1024, true)),
          "iSWA: sliding cache always needs one");

    // ---------------------------------------------------------------- usable
    std::printf("usable\n");
    const int thold = restore_threshold(2663, 1024, true);   // 1639
    // The checkpoint that WORKED in findings 129, measured: taken 68 tokens before the end of a
    // 2680-token prompt, restored, and left prompt_n = 67 instead of 2680.
    check(usable(/*ckpt_pos_min=*/1334, /*ckpt_pos_max=*/2613, /*pos_next=*/2663, thold),
          "findings 129's real checkpoint [1334, 2613] restores");
    // A checkpoint that reaches past the divergence point cannot be used: it would have to be
    // rolled back itself. [TAG_CHECKPOINTS_FIX_POS_MIN]
    check(!usable(/*ckpt_pos_min=*/1334, /*ckpt_pos_max=*/2700, /*pos_next=*/2663, thold),
          "a checkpoint covering past pos_next is refused");
    // Strictness matters: equality fails. This one token is the entire difference between
    // "reuse the prefix" and "re-prefill 2680 tokens".
    check(!usable(/*ckpt_pos_min=*/thold, /*ckpt_pos_max=*/2000, /*pos_next=*/2663, thold),
          "pos_min == threshold is REFUSED (the guard is strict)");
    check(usable(/*ckpt_pos_min=*/thold - 1, /*ckpt_pos_max=*/2000, /*pos_next=*/2663, thold),
          "pos_min == threshold - 1 is accepted");
    // A checkpoint holding the whole sequence is always fine.
    check(usable(/*ckpt_pos_min=*/0, /*ckpt_pos_max=*/2000, /*pos_next=*/2663, /*thold=*/0),
          "pos_min == 0 is the special case and always passes");

    // ---------------------------------------------------------------- resume point
    std::printf("resume_pos\n");
    check_eq(resume_pos(/*min=*/1334, /*max=*/2613, /*pos_next=*/2663), 2613, "resume at the checkpoint's end");
    check_eq(resume_pos(/*min=*/1334, /*max=*/2613, /*pos_next=*/2000), 2000, "never past where the prompt diverges");
    check_eq(resume_pos(/*min=*/5,    /*max=*/3,    /*pos_next=*/900),     6, "degenerate max < min: pos_min + 1 wins");

    // ---------------------------------------------------------------- THE THEOREM
    // findings 128 proposed "also create a checkpoint at SLOT_STATE_DONE_PROMPT, when the KV is
    // no longer empty" as a ~3-line fix. findings 129 nearly built the same thing. Both are
    // refuted here, in microseconds, for every divergence a real turn can have.
    std::printf("head_is_usable -- the fix that keeps being proposed\n");
    check(!head_is_usable(/*pos_max=*/2679, /*n_swa=*/1024, /*divergence=*/16),
          "checkpoint minted after the whole prompt: USELESS at 16 tokens of divergence");
    check(!head_is_usable(2679, 1024, 1),
          "still useless at ONE token of divergence");
    check(!head_is_usable(5262, 1024, 31),
          "and at the shape server-context.cpp:1748 recorded measured ([4239,5262] vs 4208)");
    // Written expecting this to be the one usable case. It is NOT: the guard is strict, so
    // born_early == divergence == 0 fails too. The head checkpoint is useless at EVERY
    // divergence, with no exception -- a stronger statement than any findings entry made.
    check(!head_is_usable(2679, 1024, 0),
          "useless even at ZERO divergence: the strict guard refuses equality");

    // ---------------------------------------------------------------- the rule itself
    // usable() reduces to born_early > divergence: n_swa cancels, pos_max cancels.
    std::printf("usable_when_born -- born_early > divergence, and nothing else\n");
    check(usable_when_born(/*born_early=*/68, /*divergence=*/16, 2679, 1024),
          "upstream's cut (68) beats a chat template's divergence (16)");
    check(!usable_when_born(16, 16, 2679, 1024), "equal is not enough (strict)");
    check(usable_when_born(17, 16, 2679, 1024),  "one token earlier is enough");
    check(usable_when_born(68, 16, 2679, 0),     "holds with n_swa = 0: the window cancels out");
    check(usable_when_born(68, 16, 99999, 1024), "holds at any prompt length: pos_max cancels out");
    check(!usable_when_born(68, 100, 2679, 1024),
          "and fails when the next turn diverges further back than the cut");

    std::printf(g_fail == 0 ? "\nPASS\n" : "\nFAIL: %d check(s)\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
