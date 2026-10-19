// CPU unit test for the policy's cost-aware rule (no ggml, no CUDA, no model). Exercises:
//   * the price binds: a candidate whose forecast gain over the victim exceeds r is promoted,
//     the same candidate under a price it cannot pay is not;
//   * forecasts come from VISIBLE decode routing only: prompt routing never promotes;
//   * decisions run at every visible token with at most cost_per_token promotions per token;
//   * the prompt-end decision is uncapped;
//   * the P3 boundary pass (moert_policy_boundary_layer): the prompt's counts seed the forecast of one
//     layer and decide it at a near-zero price; idempotent.
//
// Build (CMake target test-moert-policy-cost), run with no arguments.

#include "moert/policy.h"

#include <cstdio>
#include <vector>

static int g_fail = 0;
#define CHECK(cond) do { \
    if (!(cond)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); ++g_fail; } \
} while (0)

static int g_streams = 0;
static int32_t st_ok(void *, int32_t, int32_t, int32_t) { ++g_streams; return 1; }
static void pub(void *, int32_t) {}

static const int NL = 1, NE = 16, K = 4;

static moert_policy * make(float reuse, int per_token, int horizon) {
    moert_policy_config cfg;
    moert_policy_config_defaults(&cfg);
    cfg.n_layers = NL; cfg.n_expert = NE; cfg.k = K;
    cfg.cost_horizon = horizon; cfg.cost_reuse = reuse; cfg.cost_per_token = per_token;
    moert_policy_callbacks cbs{}; cbs.stream_expert = &st_ok; cbs.publish_layer = &pub;
    moert_policy * p = moert_policy_create(&cfg, &cbs);
    for (int e = 0; e < K; ++e) moert_policy_commit_seed(p, 0, e, e);   // seeds 0..3
    return p;
}

// one visible token routing experts `ids` (top-k = ids.size())
static void tok(moert_policy * p, const std::vector<int32_t> & ids) {
    moert_policy_observe(p, 0, ids.data(), (int32_t) ids.size(), 1);
    moert_policy_on_token_end(p);
}

static bool resident(moert_policy * p, int e) { return moert_policy_exp_slot_view(p, 0)[e] >= 0; }

int main() {
    // --- 1. the price binds -------------------------------------------------------------------
    // experts 10..13 used on every token for 64 tokens, the seeds 0..3 never: forecast(10..13)
    // approaches H = 128 uses, forecast(seeds) = 0, so the gain reaches ~96 by token 64.
    for (float reuse : {51.0f, 500.0f}) {
        moert_policy * p = make(reuse, /*per_token*/ 4, /*H*/ 128);
        g_streams = 0;
        for (int t = 0; t < 64; ++t) tok(p, {10, 11, 12, 13});
        if (reuse < 100.0f) {
            CHECK(resident(p, 10) && resident(p, 11) && resident(p, 12) && resident(p, 13));
            CHECK(g_streams == 4);                                    // and nothing churns afterwards
        } else {
            CHECK(!resident(p, 10) && resident(p, 0));               // gain < price 500: nothing moves
            CHECK(g_streams == 0);
        }
        moert_policy_free(p);
    }

    // --- 2. prompt routing never feeds the forecast -------------------------------------------
    {
        moert_policy * p = make(51.0f, 4, 128);
        std::vector<int32_t> ids;
        for (int t = 0; t < 200; ++t) ids.push_back(9);          // a long prompt hammering e9
        moert_policy_observe(p, 0, ids.data(), 1, 200);
        moert_policy_on_prompt_token(p, 200);
        g_streams = 0;
        moert_policy_refresh(p, MOERT_REFRESH_PROMPT_END);
        CHECK(!resident(p, 9));
        CHECK(g_streams == 0);
        moert_policy_free(p);
    }

    // --- 3. per-token decisions, at most N promotions per token --------------------------------
    // H = 16, r = 2: forecast(10..13) = 16 * (1 - d^t) passes the price after ~5 tokens; with N = 1
    // the four arrive on four consecutive tokens.
    {
        moert_policy * p = make(2.0f, /*per_token*/ 1, /*H*/ 16);
        g_streams = 0;
        int prev = 0, max_step = 0, first = -1;
        for (int t = 1; t <= 32; ++t) {
            tok(p, {10, 11, 12, 13});
            if (g_streams - prev > max_step) max_step = g_streams - prev;
            if (first < 0 && g_streams > 0) first = t;
            prev = g_streams;
        }
        CHECK(first > 0 && first < 16);
        CHECK(max_step == 1);                                         // the cap binds
        CHECK(g_streams == 4);                                        // and nothing churns afterwards
        CHECK(resident(p, 10) && resident(p, 11) && resident(p, 12) && resident(p, 13));
        moert_policy_free(p);
    }

    // --- 4. the prompt-end decision is uncapped ------------------------------------------------
    // The forecast is built through the boundary pass's prior at a price it cannot pay (so nothing
    // moves), then the prompt-end decision runs at the real price: all four enter at once although
    // the per-token cap is 1.
    {
        moert_policy * p = make(2.0f, /*per_token*/ 1, /*H*/ 16);
        std::vector<int32_t> cnt(NE, 0);
        for (int e = 10; e <= 13; ++e) cnt[e] = 100;
        CHECK(moert_policy_boundary_layer(p, 0, cnt.data(), 100, /*w*/ 1.0f, /*r*/ 1000.0f) == 0);
        g_streams = 0;
        moert_policy_refresh(p, MOERT_REFRESH_PROMPT_END);
        CHECK(g_streams == 4);
        CHECK(resident(p, 10) && resident(p, 11) && resident(p, 12) && resident(p, 13));
        moert_policy_free(p);
    }

    // --- 5. P3 boundary: the prompt's counts decide ONE layer at a near-zero price -------------
    // A prompt of 100 tokens routed e9 on every token and e7 on half of them; the seeds 0..3 never.
    // With weight 1 the forecast becomes the prompt's rate: f(9) = H, f(7) = H/2, seeds 0: both
    // enter at r = 0.5, two seeds leave, and the decode price r = 51 is not consulted.
    {
        moert_policy * p = make(51.0f, /*per_token*/ 4, /*H*/ 16);
        std::vector<int32_t> cnt(NE, 0);
        cnt[9] = 100; cnt[7] = 50;
        g_streams = 0;
        const int32_t np = moert_policy_boundary_layer(p, 0, cnt.data(), 100, /*w*/ 1.0f, /*r*/ 0.5f);
        CHECK(np == 2 && g_streams == 2);
        CHECK(resident(p, 9) && resident(p, 7));
        int seeds = 0;
        for (int e = 0; e < K; ++e) seeds += resident(p, e) ? 1 : 0;
        CHECK(seeds == 2);
        // the same counts again: nothing left to gain, nothing moves
        g_streams = 0;
        CHECK(moert_policy_boundary_layer(p, 0, cnt.data(), 100, 1.0f, 0.5f) == 0 && g_streams == 0);
        // weight 0: the prompt does not move the forecast; a non-resident with no history stays out
        std::vector<int32_t> c2(NE, 0);
        c2[12] = 100;
        CHECK(moert_policy_boundary_layer(p, 0, c2.data(), 100, 0.0f, 0.5f) == 0 && !resident(p, 12));
        // invalid input / out-of-range layer: -1, nothing changes
        CHECK(moert_policy_boundary_layer(p, 5, cnt.data(), 100, 1.0f, 0.5f) == -1);
        CHECK(moert_policy_boundary_layer(p, 0, cnt.data(), 0, 1.0f, 0.5f) == -1);
        moert_policy_free(p);
    }

    if (g_fail) { std::fprintf(stderr, "test-moert-policy-cost: %d FAILED\n", g_fail); return 1; }
    std::printf("test-moert-policy-cost: OK\n");
    return 0;
}
