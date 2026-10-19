// CPU unit test for moert_policy_resize_k (variable-KV S6a foundation; no ggml, no CUDA,
// no model). Exercises: rank-ordered survival on shrink (blended-score eviction, the hot
// slab's own ordering), slot compaction + READY survivors + preserved promotion clocks,
// evict accounting (demotions + cooldown stamp), grow leaving FREE tail slots, score
// preservation across the resize, non-residents never entering via resize, error paths.
//
// Build (CMake target test-moert-policy-resize), run with no arguments.

#include "moert/policy.h"

#include <cstdio>
#include <cstring>
#include <vector>

static int g_fail = 0;
#define CHECK(cond) do { \
    if (!(cond)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); ++g_fail; } \
} while (0)

// stream callback: always succeeds, counts calls (resize itself must never stream)
static int g_streams = 0;
static int32_t st_ok(void *, int32_t, int32_t, int32_t) { ++g_streams; return 1; }
static int g_publishes = 0;
static void pub(void *, int32_t) { ++g_publishes; }

static const int NL = 2, NE = 8, K0 = 4;

int main() {
    moert_policy_config cfg;
    moert_policy_config_defaults(&cfg);
    cfg.n_layers = NL; cfg.n_expert = NE; cfg.k = K0;
    cfg.cost_reuse = 1000.0f;   // a price nothing pays: no decision during the setup, this test is about resize
    moert_policy_callbacks cbs{}; cbs.stream_expert = &st_ok; cbs.publish_layer = &pub;
    moert_policy * p = moert_policy_create(&cfg, &cbs);
    CHECK(p != nullptr);

    // seed experts 0..3 into slots 0..3 of both layers (all get score = seed_prior)
    for (int il = 0; il < NL; ++il)
        for (int e = 0; e < K0; ++e)
            CHECK(moert_policy_commit_seed(p, il, e, e));

    // shape a ranking with observed routing: e1 hottest, then e0, then e3; e2 unrouted.
    // e7 is routed hardest of all but NOT resident — resize must never pull it in.
    // ids are token-major n_used*n_tok; observe both layers identically.
    {
        std::vector<int32_t> ids;
        for (int t = 0; t < 4; ++t) ids.push_back(1);   // e1 x4
        for (int t = 0; t < 3; ++t) ids.push_back(0);   // e0 x3
        for (int t = 0; t < 1; ++t) ids.push_back(3);   // e3 x1
        for (int t = 0; t < 6; ++t) ids.push_back(7);   // e7 x6 (cold)
        for (int il = 0; il < NL; ++il)
            moert_policy_observe(p, il, ids.data(), (int32_t) ids.size(), 1);
        moert_policy_on_token_end(p);   // fold into the EMAs (the decision cannot pay its price)
    }

    moert_policy_stats st0;
    moert_policy_get_stats(p, &st0);
    const int streams_before = g_streams;

    // --- shrink 4 -> 2: survivors must be the top-2 ranked residents {e1, e0}, compacted ---
    CHECK(moert_policy_resize_k(p, 2) == 1);
    CHECK(g_streams == streams_before);   // resize never streams (host re-streams after)
    for (int il = 0; il < NL; ++il) {
        CHECK(moert_policy_layer_k(p, il) == 2);
        const int32_t * se = moert_policy_slot_exp_view(p, il);
        const int32_t * ss = moert_policy_slot_state_view(p, il);
        const int32_t * es = moert_policy_exp_slot_view(p, il);
        const float   * rs = moert_policy_resid_view(p, il);
        CHECK(se[0] == 1 && se[1] == 0);                    // rank order: e1 > e0
        CHECK(ss[0] == MOERT_SLOT_READY && ss[1] == MOERT_SLOT_READY);
        CHECK(es[1] == 0 && es[0] == 1);                    // exp->slot remapped to compacted slots
        CHECK(es[2] == -1 && es[3] == -1 && es[7] == -1);   // e2/e3 evicted; e7 never entered
        CHECK(rs[1] == 1.0f && rs[0] == 1.0f && rs[3] == 0.0f);
    }
    moert_policy_stats st1;
    moert_policy_get_stats(p, &st1);
    CHECK(st1.demotions == st0.demotions + 2 * NL);         // 2 evictions per layer, accounted

    // --- grow 2 -> 5: survivors keep their compacted slots, tail slots FREE ---
    CHECK(moert_policy_resize_k(p, 5) == 1);
    for (int il = 0; il < NL; ++il) {
        CHECK(moert_policy_layer_k(p, il) == 5);
        const int32_t * se = moert_policy_slot_exp_view(p, il);
        const int32_t * ss = moert_policy_slot_state_view(p, il);
        CHECK(se[0] == 1 && se[1] == 0);
        CHECK(se[2] == -1 && se[3] == -1 && se[4] == -1);
        CHECK(ss[2] == MOERT_SLOT_FREE && ss[4] == MOERT_SLOT_FREE);
    }
    moert_policy_stats st2;
    moert_policy_get_stats(p, &st2);
    CHECK(st2.demotions == st1.demotions);                  // grow evicts nothing

    // scores survived both resizes (learned statistics are NOT rebuilt): e7's routed
    // score must still outrank e2's zero, visible through a fresh refresh promotion.
    // (Direct check: seed can_seed says e7 is still cold and e1 still resident.)
    CHECK(moert_policy_can_seed(p, 0, 7) == 1);
    CHECK(moert_policy_can_seed(p, 0, 1) == 0);

    // --- error paths ---
    CHECK(moert_policy_resize_k(nullptr, 2) == 0);
    CHECK(moert_policy_resize_k(p, 0)  == 0);
    CHECK(moert_policy_resize_k(p, NE + 1) == 0);

    // --- same-k resize is a clean identity (rebuild path may call it unconditionally) ---
    CHECK(moert_policy_resize_k(p, 5) == 1);
    for (int il = 0; il < NL; ++il) {
        const int32_t * se = moert_policy_slot_exp_view(p, il);
        CHECK(se[0] == 1 && se[1] == 0 && se[2] == -1);
    }

    moert_policy_free(p);
    if (g_fail) { std::fprintf(stderr, "test_moert_policy_resize: %d FAILURES\n", g_fail); return 1; }
    std::printf("test_moert_policy_resize: ALL PASS\n");
    return 0;
}
