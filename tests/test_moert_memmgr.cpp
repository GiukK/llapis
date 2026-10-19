// CPU unit test for moert-memmgr + expert-store v0 (no ggml, no CUDA, no model).
// Exercises: legacy uniform-stride catalog, store writer -> probe -> verify ->
// open_store -> promote byte-exactness, slab layout invariants, error paths
// (geometry mismatch, corrupted page, wrong source_uuid, truncated source),
// cold borrow views. Malloc-backed device callbacks stand in for VRAM.
//
// Build (CMake target test-moert-memmgr) and run with a scratch dir argument:
//   test-moert-memmgr <empty-scratch-dir>

#include "moert/memmgr.h"
#include "moert/store.h"

#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

static int g_fail = 0;
#define CHECK(cond) do { \
    if (!(cond)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); ++g_fail; } \
} while (0)

// --- malloc-backed device ------------------------------------------------------------
struct FakeSlab {
    std::vector<uint8_t> bytes;
    int allocs = 0, frees = 0, uploads = 0;
    bool fail_alloc = false;
};
static int32_t fk_alloc(void * u, uint64_t n) {
    FakeSlab * s = (FakeSlab *) u;
    if (s->fail_alloc) return 0;
    s->bytes.assign(n, 0);
    s->allocs++;
    return 1;
}
static int32_t fk_upload(void * u, uint64_t off, const void * src, uint64_t n) {
    FakeSlab * s = (FakeSlab *) u;
    if (off + n > s->bytes.size()) return 0;
    std::memcpy(s->bytes.data() + off, src, n);
    s->uploads++;
    return 1;
}
static void fk_free(void * u) { ((FakeSlab *) u)->frees++; }
static int32_t fk_download(void * u, uint64_t off, void * dst, uint64_t n) {
    FakeSlab * s = (FakeSlab *) u;
    if (off + n > s->bytes.size()) return 0;
    std::memcpy(dst, s->bytes.data() + off, n);
    return 1;
}

// deterministic page bytes: byte j of page (L,e,kind,ws) = hash mix
static void fill_page(std::vector<uint8_t> & buf, int L, int e, int kind, int ws, size_t n) {
    buf.resize(n);
    for (size_t j = 0; j < n; ++j) {
        buf[j] = (uint8_t) (0x5A ^ (L * 131) ^ (e * 31) ^ (kind * 7) ^ (ws * 191) ^ (int) (j * 2654435761u >> 3));
    }
}

// Geometry used throughout: 2 layers x 4 experts, K=2.
// Groups: kind0 weight (bpe 96, per-expert shape [8,3]), kind1 weight (bpe 48, [4,3]),
//         kind1 scale (bpe 4, scalar).
static const int NL = 2, NE = 4, K = 2;
struct GroupSpec { int kind, ws; uint64_t bpe; int64_t s0, s1; };
static const GroupSpec SPECS[3] = { { 0, 0, 96, 8, 3 }, { 1, 0, 48, 4, 3 }, { 1, 1, 4, 1, 1 } };

int main(int argc, char ** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: test-moert-memmgr <scratch-dir>\n"); return 2; }
    const std::string dir = argv[1];
    // Create it here rather than relying on the caller: on a fresh checkout `ctest` passes a
    // path under the build tree that nothing has made yet, and the first fopen would fail
    // with a CHECK that says nothing about the real cause.
    {
        std::error_code ec;
        std::filesystem::create_directories(dir, ec);
        if (ec) {
            std::fprintf(stderr, "test-moert-memmgr: cannot create scratch dir '%s': %s\n",
                         dir.c_str(), ec.message().c_str());
            return 2;
        }
    }

    // ---------------------------------------------------------------- legacy catalog --
    // Source file: for each (L, kind, ws) a stacked run of NE pages (uniform stride).
    const std::string legacy_path = dir + "/legacy.bin";
    std::vector<int64_t> group_off(NL * 3);
    {
        FILE * f = fopen(legacy_path.c_str(), "wb");
        CHECK(f != nullptr);
        std::vector<uint8_t> page;
        int64_t pos = 0;
        for (int L = 0; L < NL; ++L) {
            for (int gi = 0; gi < 3; ++gi) {
                group_off[L * 3 + gi] = pos;
                for (int e = 0; e < NE; ++e) {
                    fill_page(page, L, e, SPECS[gi].kind, SPECS[gi].ws, SPECS[gi].bpe);
                    fwrite(page.data(), 1, page.size(), f);
                    pos += (int64_t) page.size();
                }
            }
        }
        fclose(f);
    }

    FakeSlab slab1;
    moert_memmgr_device dev1 = { fk_alloc, fk_upload, fk_free, &slab1 };
    moert_memmgr * m1 = moert_memmgr_create(NL, NE, K, &dev1);
    CHECK(m1 != nullptr);
    CHECK(moert_memmgr_create(0, NE, K, &dev1) == nullptr);       // bad geometry
    CHECK(moert_memmgr_create(NL, NE, NE + 1, &dev1) == nullptr); // K > n_expert

    const int32_t sid = moert_memmgr_add_source(m1, legacy_path.c_str());
    CHECK(sid == 0);
    CHECK(moert_memmgr_add_source(m1, (dir + "/missing.bin").c_str()) == -1);

    for (int L = 0; L < NL; ++L) {
        for (int gi = 0; gi < 3; ++gi) {
            const GroupSpec & s = SPECS[gi];
            int64_t ne[4];
            int axis;
            if (s.ws) { ne[0] = NE; ne[1] = ne[2] = ne[3] = 1; axis = 0; }
            else      { ne[0] = s.s0; ne[1] = s.s1; ne[2] = NE; ne[3] = 1; axis = 2; }
            CHECK(moert_memmgr_set_page(m1, L, s.kind, s.ws, sid, group_off[L * 3 + gi],
                                        s.bpe, /*type*/ 40 + gi, ne, axis) == 1);
        }
    }
    // duplicate + bad axis rejected
    {
        int64_t ne[4] = { 8, 3, NE, 1 };
        CHECK(moert_memmgr_set_page(m1, 0, 0, 0, sid, 0, 96, 40, ne, 2) == 0);  // dup
        ne[2] = NE + 1;
        CHECK(moert_memmgr_set_page(m1, 1, 2, 0, sid, 0, 96, 40, ne, 2) == 0);  // ne[axis] != n_expert
    }
    CHECK(moert_memmgr_promote(m1, 0, 0, 0) == 0);   // before finalize
    CHECK(moert_memmgr_finalize(m1) == 1);
    CHECK(moert_memmgr_finalize(m1) == 0);           // double finalize
    CHECK(slab1.allocs == 1);

    // layout: regions 1024-aligned, slab covers all groups
    uint64_t expect = 0;
    for (int L = 0; L < NL; ++L) {
        for (int gi = 0; gi < 3; ++gi) {
            expect = (expect + 1023) & ~1023ull;
            moert_memmgr_page pg;
            CHECK(moert_memmgr_page_info(m1, L, SPECS[gi].kind, SPECS[gi].ws, &pg) == 1);
            CHECK(pg.region_off == expect);
            CHECK(pg.bytes_per_expert == SPECS[gi].bpe);
            if (SPECS[gi].ws) { CHECK(pg.ndim == 1 && pg.pool_ne[0] == K && pg.expert_axis == 0); }
            else { CHECK(pg.ndim == 3 && pg.pool_ne[0] == SPECS[gi].s0 && pg.pool_ne[1] == SPECS[gi].s1 && pg.pool_ne[2] == K); }
            expect += (uint64_t) K * SPECS[gi].bpe;
        }
    }
    CHECK(moert_memmgr_slab_bytes(m1) == expect);
    CHECK(slab1.bytes.size() == expect);
    moert_memmgr_page missing;
    CHECK(moert_memmgr_page_info(m1, 0, 2, 0, &missing) == 0);   // absent group

    // promote e=3 -> slot 1 and e=1 -> slot 0; verify slab bytes
    CHECK(moert_memmgr_promote(m1, 0, 3, 1) == 1);
    CHECK(moert_memmgr_promote(m1, 1, 1, 0) == 1);
    CHECK(moert_memmgr_promote(m1, 0, NE, 0) == 0);  // e out of range
    CHECK(moert_memmgr_promote(m1, 0, 0, K) == 0);   // slot out of range
    {
        std::vector<uint8_t> page;
        for (int gi = 0; gi < 3; ++gi) {
            moert_memmgr_page pg;
            CHECK(moert_memmgr_page_info(m1, 0, SPECS[gi].kind, SPECS[gi].ws, &pg) == 1);
            fill_page(page, 0, 3, SPECS[gi].kind, SPECS[gi].ws, SPECS[gi].bpe);
            CHECK(std::memcmp(slab1.bytes.data() + pg.region_off + 1 * pg.bytes_per_expert,
                              page.data(), page.size()) == 0);
            CHECK(moert_memmgr_page_info(m1, 1, SPECS[gi].kind, SPECS[gi].ws, &pg) == 1);
            fill_page(page, 1, 1, SPECS[gi].kind, SPECS[gi].ws, SPECS[gi].bpe);
            CHECK(std::memcmp(slab1.bytes.data() + pg.region_off + 0 * pg.bytes_per_expert,
                              page.data(), page.size()) == 0);
        }
    }

    // cold borrow views
    {
        static uint8_t fake_cold[64];
        moert_cold_view v = { fake_cold, 16, 40, NE };
        CHECK(moert_memmgr_set_cold_view(m1, 0, 0, 0, &v) == 1);
        CHECK(moert_memmgr_cold_view_count(m1) == 1);
        moert_cold_view got;
        CHECK(moert_memmgr_get_cold_view(m1, 0, 0, 0, &got) == 1);
        CHECK(got.base == fake_cold && got.stride == 16 && got.n_expert == NE);
        CHECK(moert_memmgr_get_cold_view(m1, 1, 0, 0, &got) == 0);   // not registered
        v.n_expert = NE + 1;
        CHECK(moert_memmgr_set_cold_view(m1, 0, 1, 0, &v) == 0);     // wrong geometry
        v.n_expert = NE;
        CHECK(moert_memmgr_set_cold_view(m1, 0, 0, 0, &v) == 1);     // re-register same slot
        CHECK(moert_memmgr_cold_view_count(m1) == 1);                // not double-counted
    }
    moert_memmgr_free(m1);
    CHECK(slab1.frees == 1);

    // ---------------------------------------------------------------- expert store ----
    const std::string idx = dir + "/t" MOERT_STORE_INDEX_SUFFIX;
    const std::string pak = dir + "/t" MOERT_STORE_PACK_SUFFIX;
    const uint64_t SRC_UUID = 0xDEADBEEFCAFEF00Dull;
    {
        moert_store_info info{};
        info.source_uuid = SRC_UUID;
        std::strcpy(info.arch, "testarch");
        info.n_layers = NL;
        info.n_expert = NE;
        info.n_embd   = 8;
        info.n_ff_exp = 3;
        moert_store_writer * w = moert_store_writer_open(idx.c_str(), pak.c_str(), &info);
        CHECK(w != nullptr);
        std::vector<uint8_t> page;
        // store page order = (L, e, kind, ws): per-expert locality (the runtime promotion unit)
        for (int L = 0; L < NL; ++L) {
            for (int e = 0; e < NE; ++e) {
                for (int gi = 0; gi < 3; ++gi) {
                    const GroupSpec & s = SPECS[gi];
                    fill_page(page, L, e, s.kind, s.ws, s.bpe);
                    const int64_t shape[2] = { s.ws ? 1 : s.s0, s.ws ? 1 : s.s1 };
                    CHECK(moert_store_writer_add_page(w, L, e, s.kind, s.ws,
                                                      /*quant*/ 40 + gi, /*src quant*/ 40 + gi, 0,
                                                      shape, 2, page.data(), page.size()) == 1);
                }
            }
        }
        CHECK(moert_store_writer_close(w) == 1);
    }

    moert_store_info probed{};
    CHECK(moert_store_probe(idx.c_str(), &probed) == 1);
    CHECK(probed.n_layers == NL && probed.n_expert == NE);
    CHECK(probed.source_uuid == SRC_UUID);
    CHECK(probed.record_count == (uint64_t) NL * NE * 3);
    CHECK(std::strcmp(probed.arch, "testarch") == 0);
    CHECK(moert_store_probe((dir + "/nope.index").c_str(), &probed) == 0);
    CHECK(moert_store_verify(idx.c_str()) == 0);

    // --- store reader (host page reads): every page reads back byte-exact + checksum-clean --
    {
        moert_store_reader * rd = moert_store_reader_open(idx.c_str(), SRC_UUID);
        CHECK(rd != nullptr);
        CHECK(moert_store_reader_open(idx.c_str(), 0x1234) == nullptr);   // wrong uuid refused
        int32_t rl = 0, re = 0;
        CHECK(moert_store_reader_geometry(rd, &rl, &re) == 1 && rl == NL && re == NE);
        std::vector<uint8_t> got, want;
        for (int L = 0; L < NL; ++L)
            for (int e = 0; e < NE; ++e)
                for (int gi = 0; gi < 3; ++gi) {
                    const GroupSpec & s = SPECS[gi];
                    const uint64_t nb = moert_store_reader_page_nbytes(rd, L, s.kind, s.ws, e);
                    CHECK(nb == s.bpe);
                    got.assign(nb, 0);
                    CHECK(moert_store_reader_read(rd, L, s.kind, s.ws, e, got.data(), nb, /*verify*/ 1) == nb);
                    fill_page(want, L, e, s.kind, s.ws, s.bpe);
                    CHECK(std::memcmp(got.data(), want.data(), nb) == 0);
                }
        CHECK(moert_store_reader_page_nbytes(rd, 0, 2, 0, 0) == 0);              // absent group (kind 2)
        CHECK(moert_store_reader_read(rd, 0, 0, 0, 0, got.data(), 1, 0) == 0);   // dst too small
        moert_store_reader_free(rd);
    }

    // --- OWNED cold arena (3b): reconstruct the cold tensors from the store, byte-exact ---
    {
        FakeSlab as; moert_memmgr_device ad = { fk_alloc, fk_upload, fk_free, &as };
        moert_memmgr * ma = moert_memmgr_create(NL, NE, K, &ad);
        CHECK(moert_memmgr_build_cold_arena(ma, idx.c_str(), 0x1234) == 0);   // wrong uuid refused
        CHECK(moert_memmgr_build_cold_arena(ma, idx.c_str(), SRC_UUID) == 1);
        // contiguous slots: addressing stride == payload
        auto slot = [&](uint64_t b, int ws) -> uint64_t { (void) ws; return b; };
        uint64_t exp_bytes = 0;
        for (int L = 0; L < NL; ++L) for (int gi = 0; gi < 3; ++gi) exp_bytes += (uint64_t) NE * slot(SPECS[gi].bpe, SPECS[gi].ws);
        CHECK(moert_memmgr_cold_arena_bytes(ma) == exp_bytes);
        std::vector<uint8_t> want;
        for (int L = 0; L < NL; ++L)
            for (int gi = 0; gi < 3; ++gi) {
                const GroupSpec & s = SPECS[gi];
                moert_cold_arena_view av;
                CHECK(moert_memmgr_get_cold_arena(ma, L, s.kind, s.ws, &av) == 1);
                CHECK(av.stride == s.bpe && av.arena_stride == slot(s.bpe, s.ws) &&
                      av.type_tag == 40 + gi && av.n_expert == NE);
                if (s.ws) { CHECK(av.ndim == 1 && av.ne[0] == NE && av.expert_axis == 0); }
                else      { CHECK(av.ndim == 3 && av.ne[0] == s.s0 && av.ne[1] == s.s1 &&
                                  av.ne[2] == NE && av.expert_axis == 2); }
                for (int e = 0; e < NE; ++e) {   // base + e*arena_stride == the store page (payload = stride bytes)
                    fill_page(want, L, e, s.kind, s.ws, s.bpe);
                    CHECK(std::memcmp((const uint8_t *) av.base + (size_t) e * av.arena_stride, want.data(), s.bpe) == 0);
                }
            }
        moert_cold_arena_view absent;
        CHECK(moert_memmgr_get_cold_arena(ma, 0, 2, 0, &absent) == 0);   // absent group
        moert_memmgr_free(ma);
    }

    // open_store + promote must reproduce the exact same slab bytes as the legacy path
    FakeSlab slab2;
    moert_memmgr_device dev2 = { fk_alloc, fk_upload, fk_free, &slab2, fk_download };
    moert_memmgr * m2 = moert_memmgr_create(NL, NE, K, &dev2);
    CHECK(moert_memmgr_open_store(m2, idx.c_str(), /*wrong uuid*/ 0x1234) == 0);
    CHECK(moert_memmgr_open_store(m2, idx.c_str(), SRC_UUID) == 1);
    CHECK(moert_memmgr_finalize(m2) == 1);
    CHECK(moert_memmgr_slab_bytes(m2) == expect);   // same layout as legacy
    CHECK(moert_memmgr_promote(m2, 0, 3, 1) == 1);
    CHECK(moert_memmgr_promote(m2, 1, 1, 0) == 1);
    {
        std::vector<uint8_t> page;
        for (int gi = 0; gi < 3; ++gi) {
            moert_memmgr_page pg;
            CHECK(moert_memmgr_page_info(m2, 0, SPECS[gi].kind, SPECS[gi].ws, &pg) == 1);
            CHECK(pg.type_tag == 40 + gi);
            fill_page(page, 0, 3, SPECS[gi].kind, SPECS[gi].ws, SPECS[gi].bpe);
            CHECK(std::memcmp(slab2.bytes.data() + pg.region_off + 1 * pg.bytes_per_expert,
                              page.data(), page.size()) == 0);
        }
        CHECK(slab1.bytes.size() == slab2.bytes.size());
    }
    // promote_from_arena: the same memmgr also owns the cold arena built from the same store, so an
    // arena promotion must write exactly the bytes the file promotion wrote, at the same slot address
    CHECK(moert_memmgr_promote_from_arena(m2, 0, 3, 1) == 0);           // no arena yet: refused
    CHECK(moert_memmgr_arena_matches_hot(m2) == 0);
    CHECK(moert_memmgr_build_cold_arena(m2, idx.c_str(), SRC_UUID) == 1);
    CHECK(moert_memmgr_arena_matches_hot(m2) == 1);
    {
        const std::vector<uint8_t> after_file = slab2.bytes;
        std::fill(slab2.bytes.begin(), slab2.bytes.end(), (uint8_t) 0);
        CHECK(moert_memmgr_promote_from_arena(m2, 0, 3, 1) == 1);
        CHECK(moert_memmgr_promote_from_arena(m2, 1, 1, 0) == 1);
        CHECK(slab2.bytes == after_file);                               // byte-identical to the file path
        CHECK(moert_memmgr_promote_from_arena(m2, 0, NE, 0) == 0);      // e out of range
        CHECK(moert_memmgr_promote_from_arena(m2, 0, 0, K) == 0);       // slot out of range
    }
    // exclusive arena: release + restore-from-slot rebuilds the arena page byte for byte. The test
    // pages are smaller than an OS page, so release frees nothing and the page stays writable: we
    // scribble on it to prove that restore really rewrites it from the slot.
    {
        // expert 3 of layer 0 sits in slot 1 (promoted above)
        moert_memmgr_arena_release(m2, 0, 3);
        std::vector<std::vector<uint8_t>> want;
        std::vector<uint8_t *> pages;
        for (int gi = 0; gi < 3; ++gi) {
            const GroupSpec & s = SPECS[gi];
            moert_cold_arena_view av;
            if (!moert_memmgr_get_cold_arena(m2, 0, s.kind, s.ws, &av)) continue;
            uint8_t * pg = (uint8_t *) av.base + (size_t) 3 * av.arena_stride;
            want.emplace_back(pg, pg + av.stride);
            pages.push_back(pg);
            if (s.ws == 0) std::memset(pg, 0xAB, av.stride);              // weights: restore must rewrite
        }
        CHECK(moert_memmgr_arena_restore(m2, 0, 3, 1) == 1);
        for (size_t i = 0; i < pages.size(); ++i)
            CHECK(std::memcmp(pages[i], want[i].data(), want[i].size()) == 0);
        CHECK(moert_memmgr_arena_restore(m2, 0, 3, K) == 0);              // slot out of range
        CHECK(moert_memmgr_arena_restore(m2, NL, 3, 1) == 0);             // layer out of range
        CHECK(moert_memmgr_arena_committed_bytes(m2) == moert_memmgr_cold_arena_bytes(m2));
    }
    moert_memmgr_free(m2);

    // region alloc override (host-measured device padding): page_info works pre-finalize,
    // inflating the LAST region grows the slab, slot addressing is unchanged
    {
        FakeSlab s5;
        moert_memmgr_device d5 = { fk_alloc, fk_upload, fk_free, &s5 };
        moert_memmgr * m5 = moert_memmgr_create(NL, NE, K, &d5);
        CHECK(moert_memmgr_open_store(m5, idx.c_str(), 0) == 1);
        moert_memmgr_page pg;
        CHECK(moert_memmgr_page_info(m5, 1, 1, 1, &pg) == 1);   // pre-finalize: catalog facts
        CHECK(pg.bytes_per_expert == SPECS[2].bpe);
        CHECK(moert_memmgr_set_region_alloc(m5, 1, 1, 1, (uint64_t) K * pg.bytes_per_expert + 100) == 1);
        CHECK(moert_memmgr_finalize(m5) == 1);
        CHECK(moert_memmgr_slab_bytes(m5) == expect + 100);
        CHECK(moert_memmgr_promote(m5, 1, 2, 0) == 1);
        moert_memmgr_page pg2;
        CHECK(moert_memmgr_page_info(m5, 1, 1, 1, &pg2) == 1);
        std::vector<uint8_t> page;
        fill_page(page, 1, 2, 1, 1, SPECS[2].bpe);
        CHECK(std::memcmp(s5.bytes.data() + pg2.region_off, page.data(), page.size()) == 0);
        moert_memmgr_free(m5);
    }

    // geometry mismatch refused
    {
        FakeSlab s3;
        moert_memmgr_device d3 = { fk_alloc, fk_upload, fk_free, &s3 };
        moert_memmgr * m3 = moert_memmgr_create(NL + 1, NE, K, &d3);
        CHECK(moert_memmgr_open_store(m3, idx.c_str(), 0) == 0);
        moert_memmgr_free(m3);
    }

    // corrupt one pack byte -> verify flags exactly 1 bad page
    {
        moert_store_info info2{};
        CHECK(moert_store_probe(idx.c_str(), &info2) == 1);
        FILE * f = fopen(pak.c_str(), "rb+");
        CHECK(f != nullptr);
        fseek(f, 300, SEEK_SET);   // inside the first page (pages start at 256)
        int c = fgetc(f);
        fseek(f, 300, SEEK_SET);
        fputc(c ^ 0xFF, f);
        fclose(f);
        CHECK(moert_store_verify(idx.c_str()) == 1);
        // un-corrupt for cleanliness
        f = fopen(pak.c_str(), "rb+");
        fseek(f, 300, SEEK_SET);
        fputc(c, f);
        fclose(f);
        CHECK(moert_store_verify(idx.c_str()) == 0);
    }

    // failed slab alloc propagates out of finalize
    {
        FakeSlab s4;
        s4.fail_alloc = true;
        moert_memmgr_device d4 = { fk_alloc, fk_upload, fk_free, &s4 };
        moert_memmgr * m4 = moert_memmgr_create(NL, NE, K, &d4);
        CHECK(moert_memmgr_open_store(m4, idx.c_str(), 0) == 1);
        CHECK(moert_memmgr_finalize(m4) == 0);
        moert_memmgr_free(m4);
        CHECK(s4.frees == 0);   // never allocated -> never freed
    }

    // optional: exercise the reader on a REAL store (argv[2] = experts.index), sampling pages
    // and re-checking their checksums through the reader path. Proves the reader works on the
    // actual IQ2 store the exclusive path will fill owned pages from.
    if (argc > 2) {
        moert_store_reader * rd = moert_store_reader_open(argv[2], 0);
        CHECK(rd != nullptr);
        if (rd) {
            int32_t rl = 0, re = 0;
            moert_store_reader_geometry(rd, &rl, &re);
            std::printf("real store %s: %d layers x %d experts\n", argv[2], rl, re);
            std::vector<uint8_t> buf;
            int sampled = 0, ok = 0;
            const int lstep = rl > 4 ? rl / 4 : 1, estep = re > 8 ? re / 8 : 1;
            for (int L = 0; L < rl; L += lstep)
                for (int e = 0; e < re; e += estep)
                    for (int kind = 0; kind < 4; ++kind)
                        for (int ws = 0; ws < 2; ++ws) {
                            const uint64_t nb = moert_store_reader_page_nbytes(rd, L, kind, ws, e);
                            if (!nb) continue;
                            buf.assign(nb, 0);
                            ++sampled;
                            if (moert_store_reader_read(rd, L, kind, ws, e, buf.data(), nb, /*verify*/ 1) == nb) ++ok;
                        }
            std::printf("real store reader: %d/%d sampled pages read + checksum OK\n", ok, sampled);
            CHECK(sampled > 0 && ok == sampled);
            moert_store_reader_free(rd);
        }
    }

    // ---------------------------------------------------------- variable-KV: reset_hot_layout --
    // A fresh, self-contained memmgr (no interference with the checks above): finalize at K=2,
    // reset_hot_layout, re-layout at K=3, and verify the slab is rebuilt bigger, the page catalog
    // survives, offsets/slot-counts are correct at the new K, and promotion works into the new slab.
    {
        FakeSlab slabRv;
        moert_memmgr_device devRv = { fk_alloc, fk_upload, fk_free, &slabRv };
        moert_memmgr * mrv = moert_memmgr_create(NL, NE, K, &devRv);   // K=2
        CHECK(mrv != nullptr);
        const int32_t sidRv = moert_memmgr_add_source(mrv, legacy_path.c_str());
        CHECK(sidRv == 0);
        for (int L = 0; L < NL; ++L) {
            for (int gi = 0; gi < 3; ++gi) {
                const GroupSpec & s = SPECS[gi];
                int64_t ne[4];
                int axis;
                if (s.ws) { ne[0] = NE; ne[1] = ne[2] = ne[3] = 1; axis = 0; }
                else      { ne[0] = s.s0; ne[1] = s.s1; ne[2] = NE; ne[3] = 1; axis = 2; }
                CHECK(moert_memmgr_set_page(mrv, L, s.kind, s.ws, sidRv, group_off[L * 3 + gi],
                                            s.bpe, /*type*/ 40 + gi, ne, axis) == 1);
            }
        }
        CHECK(moert_memmgr_reset_hot_layout(mrv) == 0);       // not finalized yet -> refused
        CHECK(moert_memmgr_finalize(mrv) == 1);
        CHECK(slabRv.allocs == 1 && slabRv.frees == 0);
        const uint64_t bytes_k2 = moert_memmgr_slab_bytes(mrv);
        CHECK(bytes_k2 > 0);

        // reset: releases the old slab, clears finalized/offsets, keeps the catalog
        CHECK(moert_memmgr_reset_hot_layout(mrv) == 1);
        CHECK(slabRv.frees == 1);                              // old slab released exactly once
        CHECK(moert_memmgr_slab_bytes(mrv) == 0);
        moert_memmgr_page pgc;                                 // catalog survived the reset
        CHECK(moert_memmgr_page_info(mrv, 0, SPECS[0].kind, SPECS[0].ws, &pgc) == 1);
        CHECK(pgc.bytes_per_expert == SPECS[0].bpe);
        CHECK(moert_memmgr_promote(mrv, 0, 0, 0) == 0);        // promote refused while unfinalized

        // re-layout at K=3 and re-finalize
        for (int L = 0; L < NL; ++L) CHECK(moert_memmgr_set_layer_k(mrv, L, 3) == 1);
        CHECK(moert_memmgr_finalize(mrv) == 1);
        CHECK(slabRv.allocs == 2);                             // a second, independent slab alloc
        const uint64_t bytes_k3 = moert_memmgr_slab_bytes(mrv);
        CHECK(bytes_k3 > bytes_k2);                            // K 2->3 grows every region
        moert_memmgr_page pg3;
        CHECK(moert_memmgr_page_info(mrv, 1, SPECS[0].kind, SPECS[0].ws, &pg3) == 1);
        CHECK(pg3.pool_ne[pg3.expert_axis] == 3);             // slot count follows the new K
        CHECK(moert_memmgr_promote(mrv, 1, 2, 2) == 1);       // slot 2 is valid only at K>=3
        moert_memmgr_free(mrv);
        CHECK(slabRv.frees == 2);                              // final free releases the K=3 slab
    }

    if (g_fail == 0) {
        std::printf("test-moert-memmgr: ALL PASS\n");
        return 0;
    }
    std::printf("test-moert-memmgr: %d FAILURES\n", g_fail);
    return 1;
}
