// CPU unit test for moert/registry.h: the cheapest valid source of an expert's bytes, in both regimes.
//   * full IQ2 (hot == cold bytes): slot < arena < file for either representation; a slot serves the
//     cold representation; a released arena page with no slot falls back to the file;
//   * Q4 hot / IQ2 cold: the arena never serves the hot representation and a slot never serves the
//     cold one; a hot promotion comes from the file unless a slot already holds it.
// Build (CMake target test-moert-registry), run with no arguments.

#include "moert/registry.h"

#include <cstdio>

static int g_fail = 0;
#define CHECK(cond) do { \
    if (!(cond)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); ++g_fail; } \
} while (0)

int main() {
    // --- full IQ2: same bytes -----------------------------------------------------------------
    {
        moert_expert_where w = { -1, 1, 1, 1, 1 };            // cold expert, page resident
        CHECK(moert_source_for(&w, MOERT_REP_HOT)  == MOERT_SRC_ARENA);   // promotion: from RAM
        CHECK(moert_source_for(&w, MOERT_REP_COLD) == MOERT_SRC_ARENA);
        w.slot = 5;                                            // hot: the slot beats the arena
        CHECK(moert_source_for(&w, MOERT_REP_HOT)  == MOERT_SRC_SLOT);    // prefill staging: D2D
        CHECK(moert_source_for(&w, MOERT_REP_COLD) == MOERT_SRC_SLOT);    // exclusive-arena restore
        w.arena_valid = 0;                                     // released while hot: still the slot
        CHECK(moert_source_for(&w, MOERT_REP_COLD) == MOERT_SRC_SLOT);
        w.slot = -1;                                           // no slot, no page: the file
        CHECK(moert_source_for(&w, MOERT_REP_HOT)  == MOERT_SRC_FILE);
        w.file_hot = w.file_cold = 0;
        CHECK(moert_source_for(&w, MOERT_REP_HOT)  == MOERT_SRC_NONE);
    }
    // --- Q4 hot / IQ2 cold: different bytes -------------------------------------------------------
    {
        moert_expert_where w = { -1, 1, 0, 1, 1 };
        CHECK(moert_source_for(&w, MOERT_REP_HOT)  == MOERT_SRC_FILE);    // the arena is not Q4
        CHECK(moert_source_for(&w, MOERT_REP_COLD) == MOERT_SRC_ARENA);
        w.slot = 2;
        CHECK(moert_source_for(&w, MOERT_REP_HOT)  == MOERT_SRC_SLOT);
        CHECK(moert_source_for(&w, MOERT_REP_COLD) == MOERT_SRC_ARENA);   // the slot is not IQ2
        w.arena_valid = 0;
        CHECK(moert_source_for(&w, MOERT_REP_COLD) == MOERT_SRC_FILE);
    }
    CHECK(moert_source_for(nullptr, MOERT_REP_HOT) == MOERT_SRC_NONE);

    if (g_fail) { std::fprintf(stderr, "test-moert-registry: %d FAILED\n", g_fail); return 1; }
    std::printf("test-moert-registry: all checks passed\n");
    return 0;
}
