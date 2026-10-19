// CPU unit test for asynchronous promotion in moert-policy (no ggml, no CUDA, no model). Exercises:
//   * a stream_expert that returns 2 leaves the slot FILLING: occupied by the new expert (exp_slot /
//     slot_exp set, so no other decision can take it) but NOT READY, so the device partition would
//     serve it cold; complete_fill makes it READY;
//   * a completion that arrives after the slot was evicted / re-assigned is STALE and changes nothing;
//   * abort_fill releases a FILLING slot to FREE; a stale abort changes nothing;
//   * a synchronous stream_expert (return 1) behaves exactly as before (READY at once, no ledger);
//   * the stats ledger counts async / completed / stale / aborted fills.
//
// Build (CMake target test-moert-policy-async), run with no arguments.

#include "moert/policy.h"

#include <cstdio>
#include <vector>

static int g_fail = 0;
#define CHECK(cond) do { \
    if (!(cond)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); ++g_fail; } \
} while (0)

struct Job { int il, e, slot; };
static std::vector<Job> g_jobs;
static int32_t g_ret = 2;
static int32_t st_cb(void *, int32_t il, int32_t e, int32_t slot) {
    if (g_ret == 2) g_jobs.push_back({il, e, slot});
    return g_ret;
}
static void pub(void *, int32_t) {}

static const int NL = 1, NE = 16, K = 4;

static moert_policy * make() {
    moert_policy_config cfg;
    moert_policy_config_defaults(&cfg);
    cfg.n_layers = NL; cfg.n_expert = NE; cfg.k = K;
    cfg.cost_horizon = 128; cfg.cost_reuse = 51.0f; cfg.cost_per_token = 4;
    moert_policy_callbacks cbs{}; cbs.stream_expert = &st_cb; cbs.publish_layer = &pub;
    moert_policy * p = moert_policy_create(&cfg, &cbs);
    for (int e = 0; e < K; ++e) moert_policy_commit_seed(p, 0, e, e);   // seeds 0..3, READY
    return p;
}

static void tok(moert_policy * p, const std::vector<int32_t> & ids) {
    moert_policy_observe(p, 0, ids.data(), (int32_t) ids.size(), 1);
    moert_policy_on_token_end(p);
}

static int slot_of(moert_policy * p, int e) { return moert_policy_exp_slot_view(p, 0)[e]; }
static int state_of(moert_policy * p, int slot) { return moert_policy_slot_state_view(p, 0)[slot]; }

int main() {
    // --- 1. async accept -> FILLING, then complete -> READY --------------------------------------
    {
        g_ret = 2; g_jobs.clear();
        moert_policy * p = make();
        for (int t = 0; t < 64; ++t) tok(p, {10, 11, 12, 13});   // the per-token decision promotes 10..13 (~token 24)
        CHECK(g_jobs.size() == 4);
        for (const Job & j : g_jobs) {
            CHECK(slot_of(p, j.e) == j.slot);                     // occupied by the new expert...
            CHECK(state_of(p, j.slot) == MOERT_SLOT_FILLING);     // ...but not hot yet
        }
        moert_policy_stats st; moert_policy_get_stats(p, &st);
        CHECK(st.fills_async == 4 && st.fills_completed == 0 && st.occupancy == 0.0);
        for (const Job & j : g_jobs) CHECK(moert_policy_complete_fill(p, j.il, j.slot, j.e) == 1);
        for (const Job & j : g_jobs) CHECK(state_of(p, j.slot) == MOERT_SLOT_READY);
        moert_policy_get_stats(p, &st);
        CHECK(st.fills_completed == 4 && st.occupancy == 1.0);
        // completing twice is stale
        CHECK(moert_policy_complete_fill(p, 0, g_jobs[0].slot, g_jobs[0].e) == 0);
        moert_policy_free(p);
    }

    // --- 2. a completion for an expert that no longer owns the slot is stale -----------------
    {
        g_ret = 2; g_jobs.clear();
        moert_policy * p = make();
        for (int t = 0; t < 64; ++t) tok(p, {10, 11, 12, 13});
        CHECK(g_jobs.size() == 4);
        const Job j = g_jobs[0];
        CHECK(moert_policy_complete_fill(p, 0, j.slot, 9) == 0);   // wrong expert: stale
        CHECK(state_of(p, j.slot) == MOERT_SLOT_FILLING);          // untouched
        CHECK(moert_policy_complete_fill(p, 0, j.slot, j.e) == 1);
        moert_policy_stats st; moert_policy_get_stats(p, &st);
        CHECK(st.fills_stale == 1);
        moert_policy_free(p);
    }

    // --- 3. abort releases the slot; a stale abort does nothing ------------------------------
    {
        g_ret = 2; g_jobs.clear();
        moert_policy * p = make();
        for (int t = 0; t < 64; ++t) tok(p, {10, 11, 12, 13});
        const Job j = g_jobs[1];
        CHECK(moert_policy_abort_fill(p, 0, j.slot, j.e) == 1);
        CHECK(state_of(p, j.slot) == MOERT_SLOT_FREE);
        CHECK(slot_of(p, j.e) < 0);
        CHECK(moert_policy_abort_fill(p, 0, j.slot, j.e) == 0);   // already released
        CHECK(moert_policy_complete_fill(p, 0, j.slot, j.e) == 0); // and cannot be completed
        moert_policy_stats st; moert_policy_get_stats(p, &st);
        CHECK(st.fills_aborted == 1 && st.fills_stale == 2);
        moert_policy_free(p);
    }

    // --- 4. synchronous streaming is unchanged -----------------------------------------------
    {
        g_ret = 1; g_jobs.clear();
        moert_policy * p = make();
        for (int t = 0; t < 64; ++t) tok(p, {10, 11, 12, 13});
        for (int e = 10; e <= 13; ++e) {
            CHECK(slot_of(p, e) >= 0);
            CHECK(state_of(p, slot_of(p, e)) == MOERT_SLOT_READY);
        }
        moert_policy_stats st; moert_policy_get_stats(p, &st);
        CHECK(st.fills_async == 0 && st.occupancy == 1.0);
        moert_policy_free(p);
    }

    if (g_fail) { std::fprintf(stderr, "test-moert-policy-async: %d FAILED\n", g_fail); return 1; }
    std::printf("test-moert-policy-async: all checks passed\n");
    return 0;
}
