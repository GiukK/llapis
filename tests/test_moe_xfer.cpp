// CPU unit test for the transfer engine (src/moe_xfer): no GPU, no model; a fake slab and a legacy
// catalog file stand in for the device and the promotion source. Exercises:
//   * a submitted job stays PENDING (invisible to the worker) until flush();
//   * promotions from the FILE source land in the slab byte-for-byte like a synchronous
//     moert_memmgr_promote into the same slots (the path the Q4-hot regime will use);
//   * a promotion without a valid source, or from an arena that does not exist, fails cleanly
//     (Done.ok = false, nothing written) instead of reading anything;
//   * the completion list and the per-source counters.
// Build (CMake target test-moe-xfer), run with a scratch directory as the one argument.

#include "moe_xfer.hpp"
#include "moert/memmgr.h"
#include "moert/registry.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

static int g_fail = 0;
#define CHECK(cond) do { \
    if (!(cond)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); ++g_fail; } \
} while (0)

struct FakeSlab { std::vector<uint8_t> bytes; std::mutex mu; };
static int32_t fk_alloc(void * u, uint64_t n) { ((FakeSlab *) u)->bytes.assign(n, 0); return 1; }
static int32_t fk_upload(void * u, uint64_t off, const void * src, uint64_t n) {
    FakeSlab * s = (FakeSlab *) u;
    std::lock_guard<std::mutex> lk(s->mu);   // the engine uploads from its worker thread
    if (off + n > s->bytes.size()) return 0;
    std::memcpy(s->bytes.data() + off, src, n);
    return 1;
}
static void fk_free(void *) {}

static const int NL = 2, NE = 4, K = 2;
struct GroupSpec { int kind, ws; uint64_t bpe; int64_t s0, s1; };
static const GroupSpec SPECS[2] = { { 0, 0, 96, 8, 3 }, { 1, 0, 48, 4, 3 } };

static void fill_page(std::vector<uint8_t> & buf, int L, int e, int gi, size_t n) {
    buf.resize(n);
    for (size_t j = 0; j < n; ++j) buf[j] = (uint8_t) (L * 61 + e * 17 + gi * 5 + j * 3 + 1);
}

// a memmgr over the legacy catalog file, finalized onto `slab`
static moert_memmgr * make(const std::string & path, const std::vector<int64_t> & off, FakeSlab * slab) {
    moert_memmgr_device dev = { fk_alloc, fk_upload, fk_free, slab, nullptr };
    moert_memmgr * m = moert_memmgr_create(NL, NE, K, &dev);
    const int32_t sid = moert_memmgr_add_source(m, path.c_str());
    for (int L = 0; L < NL; ++L)
        for (int gi = 0; gi < 2; ++gi) {
            int64_t ne[4] = { SPECS[gi].s0, SPECS[gi].s1, NE, 1 };
            moert_memmgr_set_page(m, L, SPECS[gi].kind, SPECS[gi].ws, sid, off[L * 2 + gi], SPECS[gi].bpe, 40 + gi, ne, 2);
        }
    moert_memmgr_finalize(m);
    return m;
}

int main(int argc, char ** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: test-moe-xfer <scratch-dir>\n"); return 2; }
    const std::string dir = argv[1];
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    const std::string path = dir + "/xfer-legacy.bin";
    std::vector<int64_t> off(NL * 2);
    {
        FILE * f = std::fopen(path.c_str(), "wb");
        CHECK(f != nullptr);
        if (!f) return 1;
        std::vector<uint8_t> page;
        int64_t pos = 0;
        for (int L = 0; L < NL; ++L)
            for (int gi = 0; gi < 2; ++gi) {
                off[L * 2 + gi] = pos;
                for (int e = 0; e < NE; ++e) {
                    fill_page(page, L, e, gi, SPECS[gi].bpe);
                    std::fwrite(page.data(), 1, page.size(), f);
                    pos += (int64_t) page.size();
                }
            }
        std::fclose(f);
    }

    // reference: synchronous promotions
    FakeSlab ref;
    moert_memmgr * mref = make(path, off, &ref);
    CHECK(moert_memmgr_promote(mref, 0, 3, 1) == 1);
    CHECK(moert_memmgr_promote(mref, 1, 2, 0) == 1);

    FakeSlab slab;
    moert_memmgr * m = make(path, off, &slab);
    CHECK(moe_xfer::start(m));
    CHECK(moe_xfer::running());

    // --- 1. pending until flush ------------------------------------------------------------------
    moe_xfer::submit(0, 3, 1, MOERT_SRC_FILE);
    moe_xfer::submit(1, 2, 0, MOERT_SRC_FILE);
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    std::vector<moe_xfer::Done> done;
    CHECK(moe_xfer::take_done(done) == 0);                // the worker cannot see PENDING jobs
    {
        std::lock_guard<std::mutex> lk(slab.mu);
        CHECK(slab.bytes != ref.bytes);                    // nothing written yet
    }

    // --- 2. file promotions land byte-for-byte ---------------------------------------------------
    moe_xfer::flush();
    moe_xfer::wait_idle();
    done.clear();
    CHECK(moe_xfer::take_done(done) == 2);
    for (const auto & d : done) CHECK(d.ok);
    {
        std::lock_guard<std::mutex> lk(slab.mu);
        CHECK(slab.bytes == ref.bytes);
    }

    // --- 3. no valid source / no arena: clean failures, nothing written --------------------------
    const std::vector<uint8_t> before = slab.bytes;
    moe_xfer::submit(0, 1, 0, MOERT_SRC_NONE);
    moe_xfer::submit(0, 2, 0, MOERT_SRC_ARENA);           // this memmgr has no cold arena
    moe_xfer::flush();
    moe_xfer::wait_idle();
    done.clear();
    CHECK(moe_xfer::take_done(done) == 2);
    for (const auto & d : done) CHECK(!d.ok);
    {
        std::lock_guard<std::mutex> lk(slab.mu);
        CHECK(slab.bytes == before);
    }

    const moe_xfer::Stats st = moe_xfer::stats();
    CHECK(st.submitted == 4 && st.completed == 2 && st.failed == 2);
    CHECK(st.from_file == 2 && st.from_arena == 0);

    moert_memmgr_free(mref);
    // the engine is a process-lifetime singleton; it is joined at exit, before the memmgr would be
    // used again, so `m` is intentionally left alive here
    if (g_fail) { std::fprintf(stderr, "test-moe-xfer: %d FAILED\n", g_fail); return 1; }
    std::printf("test-moe-xfer: all checks passed\n");
    return 0;
}
