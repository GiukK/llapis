// CPU unit test for vkv_policy (the CENTRALIZED variable-KV resizing policy). Seconds, no deps.
#include "../src/vkv_policy.hpp"
#include <cstdio>

static int g_fail = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++g_fail; } } while (0)

int main() {
    vkv::Params p;   // defaults: min 8192, gran 8192, headroom 8192, slack 512, max 262144

    CHECK(vkv::boot_ctx(p) == 8192);

    // short chat: fits -> never resize (the zero-cost common case)
    CHECK(vkv::plan_grow(2000, 1024, 8192, p) == 0);
    // crossing: 7000+1024+512 = 8536 > 8192 -> roundup(8536+8192) = 24576? no: 16728 -> 24576
    CHECK(vkv::plan_grow(7000, 1024, 8192, p) == 24576);
    // just over, small headroom effect: linear ladder, not doubling
    CHECK(vkv::plan_grow(8000, 512, 8192, p) == 24576);
    CHECK(vkv::plan_grow(20000, 1024, 24576, p) == 0);       // still fits
    CHECK(vkv::plan_grow(24000, 1024, 24576, p) == 40960);   // next rungs stay gran-multiples
    // ceiling: need beyond max -> 0 (caller refuses loudly, never truncates)
    p.max_ctx = 32768;
    CHECK(vkv::plan_grow(40000, 1024, 8192, p) == 0);
    // need fits max but need+headroom would exceed: clamp to max, still grows
    CHECK(vkv::plan_grow(30000, 1024, 8192, p) == 32768);
    p.max_ctx = 262144;

    // --- generation RESERVE --------------------------------------------------------
    // max_gen is the client's declared max_tokens, which an agent harness repeats verbatim on
    // every request. It must be charged only up to gen_reserve, or the provider JSON's declared
    // ceiling alone decides the context size. These are the cases that would regress that:
    // 8192 is chosen to be BEHAVIOUR-PRESERVING: it is exactly the maxTokens the provider JSON
    // declared before the 65536 change, so every harness that worked yesterday plans identically.
    CHECK(p.gen_reserve == 8192);
    // THE case this exists for: a 65536 declaration must plan exactly like an 8192 one, at every
    // conversation depth — the declared ceiling stops being an input to the context size.
    CHECK(vkv::plan_grow( 2000, 65536, 8192,  p) == vkv::plan_grow( 2000, 8192, 8192,  p));
    CHECK(vkv::plan_grow( 7000, 65536, 8192,  p) == vkv::plan_grow( 7000, 8192, 8192,  p));
    CHECK(vkv::plan_grow(30000, 65536, 24576, p) == vkv::plan_grow(30000, 8192, 24576, p));
    // history is ALWAYS charged in full — the reserve caps generation only
    CHECK(vkv::plan_grow(30000, 65536, 8192, p) == 49152);
    // below the reserve the ceiling is still charged in full (no silent under-planning)
    CHECK(vkv::plan_grow(2000, 7000, 8192, p) == 24576);
    // reserve disabled (0) restores the pre-2026-07-28 arithmetic exactly: 2000+65536+512
    // +8192 headroom = 76240 -> the 81920 rung, i.e. the ceiling really would have decided it
    {
        vkv::Params q = p; q.gen_reserve = 0;
        CHECK(vkv::plan_grow(2000, 65536, 8192, q) == 81920);
    }

    // shrink hysteresis: reclaiming less than a full band -> keep
    CHECK(vkv::plan_shrink(20000, 32768, p) == 0);           // keep=28712 -> 32768, no gain
    CHECK(vkv::plan_shrink(4000, 32768, p) == 8192);         // deep compact -> straight to floor
    CHECK(vkv::plan_shrink(100, 16384, p) == 8192);          // floor
    CHECK(vkv::plan_shrink(100, 8192, p) == 0);              // already at floor
    CHECK(vkv::plan_shrink(12000, 32768, p) == 24576);       // mid depth keeps full headroom

    // KV cost model: near-linear, validated slope
    CHECK(vkv::kv_mib_for_ctx(8192)  == 160);
    CHECK(vkv::kv_mib_for_ctx(16384) == 319);
    CHECK(vkv::kv_mib_for_ctx(0)     == 0);

    // --- ctx-swap cost model -----------------------------------------------------
    // Pinned to the real sweep measurements. Each tier measured the OLD context's footprint
    // across its teardown and the NEW context's real footprint after creation; the model must
    // predict the latter from the former, and must land ON THE SAFE SIDE (>= real). If a future
    // change makes any of these predict BELOW the measured real cost, the ctx creation can OOM or
    // spill into WDDM shared memory — that is the whole reason the asymmetry is asserted here.
    struct { int c_old; long long fp_old; int c_new; long long real_new; } tiers[] = {
        {  8192, 476, 16384, 601 },   // tier 1: predicted 636
        { 16384, 646, 24576, 763 },   // tier 2: predicted 806
        { 24576, 808, 32768, 925 },   // tier 3: predicted 968
    };
    for (const auto & t : tiers) {
        const long long pred = vkv::ctx_cost_predict_mib(t.c_old, t.fp_old, t.c_new);
        CHECK(pred >= t.real_new);                 // never under-predict: an OOM, not a slow turn
        CHECK(pred - t.real_new < 120);            // but stay tight enough to be worth predicting
    }
    CHECK(vkv::ctx_cost_predict_mib( 8192, 476, 16384) == 636);
    CHECK(vkv::ctx_cost_predict_mib(16384, 646, 24576) == 806);
    CHECK(vkv::ctx_cost_predict_mib(24576, 808, 32768) == 968);
    // same size in and out = no extrapolation at all, just the measurement
    CHECK(vkv::ctx_cost_predict_mib(16384, 646, 16384) == 646);
    // shrink direction is symmetric (Phase 4 /compact will use it)
    CHECK(vkv::ctx_cost_predict_mib(16384, 646,  8192) == 486);
    // a nonsense measurement can never produce a negative budget input
    CHECK(vkv::ctx_cost_predict_mib(32768,   0,  8192) == 0);

    if (g_fail) { std::fprintf(stderr, "test_vkv_policy: %d FAILURES\n", g_fail); return 1; }
    std::printf("test_vkv_policy: ALL PASS\n");
    return 0;
}
