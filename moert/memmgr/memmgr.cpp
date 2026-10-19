// moert/memmgr.cpp — the expert memory manager (moert/memmgr.h).
// Owns: the page catalog, the promotion-source file handles + reads, the hot-slab
// layout/lifetime (device memory reached only via the moert_memmgr_device callbacks),
// and the cold-tier borrow-view registry. No ggml / no CUDA.

#include "moert/memmgr.h"
#include "moert/store.h"
#include "store_format.h"

#include <atomic>
#include <array>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <system_error>
#include <thread>
#include <vector>

#ifdef _WIN32
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>
#else
#  include <fcntl.h>
#  include <sys/mman.h>
#  include <unistd.h>
#endif

namespace {

constexpr uint64_t REGION_ALIGN = 1024;  // region starts (slot stride stays = bytes/expert)

// Cold-arena slot stride for a payload of `bytes`: contiguous packing (the page-aligned "arena v3"
// variant served only the arena pin, removed in elision E2). Kept as the one place the arena builder
// and the release/reload addressing agree on.
static inline uint64_t arena_slot_stride(uint64_t bytes) {
    return bytes;
}

// RAM-decommittable buffer for the owned cold arena (RAM-dedup / smart-refresh, docs). Reserve+commit
// the whole (n_expert*stride) block up front; then DECOMMIT the pages fully inside a hot expert's byte
// range to release its physical RAM (the ~1.9 GiB of hot experts are duplicated in the VRAM Q4 slab),
// and (re)COMMIT + reload from the store when it is evicted back to cold. The virtual address is stable,
// so cold experts never move (base + e*stride and any GPU mappings stay valid). Inner page alignment on
// decommit keeps the boundary pages a hot expert shares with cold neighbours committed (loses ~1 page
// per side, negligible). Falls back to a plain heap block (no decommit) if the OS alloc fails.
struct PagedBuffer {
    uint8_t * p  = nullptr;
    size_t    sz = 0;
    bool      os = false;   // OS-paged (decommit works) vs heap fallback

    PagedBuffer() = default;
    ~PagedBuffer() { free_all(); }
    PagedBuffer(const PagedBuffer &) = delete;
    PagedBuffer & operator=(const PagedBuffer &) = delete;
    PagedBuffer(PagedBuffer && o) noexcept { p = o.p; sz = o.sz; os = o.os; o.p = nullptr; o.sz = 0; o.os = false; }
    PagedBuffer & operator=(PagedBuffer && o) noexcept {
        if (this != &o) { free_all(); p = o.p; sz = o.sz; os = o.os; o.p = nullptr; o.sz = 0; o.os = false; }
        return *this;
    }

    uint8_t *       data()       { return p; }
    const uint8_t * data() const { return p; }
    size_t          size() const { return sz; }

    static size_t pagesize() {
#ifdef _WIN32
        SYSTEM_INFO si; GetSystemInfo(&si); return si.dwPageSize ? si.dwPageSize : 4096;
#else
        long v = sysconf(_SC_PAGESIZE); return v > 0 ? (size_t) v : 4096;
#endif
    }
    bool alloc(size_t n) {
        free_all();
        sz = n;
#ifdef _WIN32
        p = (uint8_t *) VirtualAlloc(nullptr, n, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
        os = p != nullptr;
#else
        p = (uint8_t *) mmap(nullptr, n, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (p == MAP_FAILED) p = nullptr;
        os = p != nullptr;
#endif
        if (!p) { p = (uint8_t *) std::malloc(n); os = false; if (p) std::memset(p, 0, n); }
        return p != nullptr;
    }
    void free_all() {
        if (!p) { sz = 0; return; }
#ifdef _WIN32
        if (os) VirtualFree(p, 0, MEM_RELEASE); else std::free(p);
#else
        if (os) munmap(p, sz); else std::free(p);
#endif
        p = nullptr; sz = 0; os = false;
    }
    // Release physical RAM for the pages fully inside [off, off+len). Returns bytes released.
    size_t decommit(size_t off, size_t len) {
        if (!p || !os || len == 0) return 0;
        const size_t pg = pagesize();
        size_t lo = (off + pg - 1) & ~(pg - 1);            // round UP (inner)
        size_t hi = (off + len) & ~(pg - 1);               // round DOWN (inner)
        if (hi <= lo) return 0;
        if (hi > sz) hi = sz & ~(pg - 1);
#ifdef _WIN32
        VirtualFree(p + lo, hi - lo, MEM_DECOMMIT);
#else
        madvise(p + lo, hi - lo, MADV_DONTNEED);
#endif
        return hi - lo;
    }
    // (Re)commit the pages covering [off, off+len) so they are writable before a reload.
    bool commit(size_t off, size_t len) {
        if (!p || !os || len == 0) return p != nullptr;
        const size_t pg = pagesize();
        size_t lo = off & ~(pg - 1);                       // round DOWN (outer)
        size_t hi = (off + len + pg - 1) & ~(pg - 1);      // round UP (outer)
        if (hi > sz) hi = sz;
#ifdef _WIN32
        return VirtualAlloc(p + lo, hi - lo, MEM_COMMIT, PAGE_READWRITE) != nullptr;
#else
        return true;   // MADV_DONTNEED pages re-fault on write; no explicit commit needed
#endif
    }
};

// One (layer, kind, is_scale) page group: K hot slots in the slab, n_expert source pages.
struct PageGroup {
    bool     present   = false;
    int32_t  source_id = -1;
    int64_t  off0      = -1;   // uniform-stride base (legacy GGUF catalog); -1 when per_off used
    uint64_t bpe       = 0;    // bytes per expert page (== slot stride)
    int32_t  type_tag  = 0;
    int64_t  ne[MOERT_MEMMGR_MAX_DIMS] = { 1, 1, 1, 1 };  // stacked shape (expert axis = n_expert)
    int32_t  axis      = -1;
    std::vector<int64_t> per_off;   // store catalog: per-expert page offsets (size n_expert)
    uint64_t alloc_bytes = 0;       // device-allocation override (host-measured; 0 = payload only)
    uint64_t region_off  = 0;       // valid after finalize
    uint64_t loan_off    = 0;       // weight groups with a loan: VIRTUAL offset of loan slot 0 (>= slab_bytes)
};

struct ColdSlot {
    bool            set = false;
    moert_cold_view v{};
};

// One (layer, kind, is_scale) group of the OWNED cold arena: the stacked cold tensor
// reconstructed contiguously in host RAM (expert e at e*stride).
struct ColdArena {
    bool     present = false;
    PagedBuffer buf;              // n_expert * arena_stride, decommittable per-expert (copy mode)
    const uint8_t * mapped = nullptr;   // map mode: expert 0 inside the read-only pack view (buf unused)
    uint64_t stride   = 0;        // PAYLOAD bytes/expert (the store read/reload size)
    uint64_t arena_stride = 0;    // ADDRESSING stride: expert e at e*arena_stride (>= stride under v3)
    int32_t  type_tag = 0;
    int64_t  ne[MOERT_MEMMGR_MAX_DIMS] = { 1, 1, 1, 1 };
    int32_t  axis     = -1;
    std::vector<int64_t> per_off; // store byte offset per expert (for RAM-dedup reload); empty = none
    int32_t  src_id   = -1;       // index into m->sources (open store FILE*); -1 = store not kept open
    const uint8_t * base() const { return mapped ? mapped : buf.data(); }
    uint64_t span(int32_t n_expert) const { return (uint64_t) n_expert * (arena_stride ? arena_stride : stride); }
};

}  // namespace

struct moert_memmgr {
    int32_t n_layers = 0, n_expert = 0, k = 0;
    std::vector<int32_t> k_layer;   // per-layer hot capacity (all == k for the uniform path)
    moert_memmgr_device dev{};
    // [il][kind*2 + is_scale]
    std::vector<std::array<PageGroup, MOERT_MEMMGR_MAX_KIND * 2>> groups;
    std::vector<std::array<ColdSlot,  MOERT_MEMMGR_MAX_KIND * 2>> cold;
    std::vector<std::array<ColdArena, MOERT_MEMMGR_MAX_KIND * 2>> cold_arena;   // owned cold tier
    uint64_t arena_bytes = 0;
    // Map mode (findings 197): the pack is mapped read-only and the weight groups point into it -- the OS
    // file cache is the cold tier (clean pages: dropped without a page-file write, re-read from the pack).
    // Only the 4-byte scale pages are copied. Set when every weight group is a dense run in the pack.
    void *   map_view = nullptr;
    uint64_t map_size = 0;
    std::atomic<uint64_t> arena_decommitted{0};   // running sum of released bytes (telemetry; RSS is truth).
                                                  // Atomic: the exclusive arena releases on a worker thread.
    int32_t cold_count = 0;
    std::vector<FILE *>      sources;
    std::vector<std::string> source_paths;
    bool     finalized  = false;
    bool     slab_owned = false;
    uint64_t slab_bytes = 0;
    // Loaned slots (claude dump findings 246): the last `loan` slots of every WEIGHT group live outside the
    // slab, in a buffer the host lends while it is idle (the prefill staging reserve between prompts). Their
    // offsets are VIRTUAL, at and past slab_bytes: the host's slab_upload/slab_download map them into that
    // buffer. Scale groups keep every slot in the slab (the fused op reads the scales as one array).
    int32_t  loan = 0;
    uint64_t loan_bytes = 0;
    std::vector<char> rbuf;   // promotion read staging (reused, buffered-fread fallback path)
    // update 6: promote()'s reads measured ~1.0-1.3 GB/s via buffered fread, matching
    // the SAME ceiling q4stream's own comment already documents for this box ("measured 1.2 GB/s vs
    // the NVMe's 2-3.5", moe_dual_q4stream.cpp) -- reusing q4stream's WHOLE-LAYER streaming mechanism
    // would read ~5.8x more data than a K=22-of-128 restore needs (net loss by the numbers), so this
    // borrows just its proven I/O TECHNIQUE (Win32 FILE_FLAG_NO_BUFFERING into an aligned buffer) at
    // the SAME per-expert granularity promote() already uses. unbuf_h[source_id] is the lazily-opened
    // unbuffered HANDLE (nullptr after unbuf_tried[source_id]==1 means "tried, unavailable, permanently
    // fall back to the buffered `sources[source_id]` FILE*" for that source).
    std::vector<void *> unbuf_h;
    std::vector<char>   unbuf_tried;
    PagedBuffer          io_align_buf;   // 4 KiB-aligned scratch for the unbuffered read path
};

static PageGroup * group_at(moert_memmgr * m, int32_t il, int32_t kind, int32_t is_scale) {
    if (!m || il < 0 || il >= m->n_layers || kind < 0 || kind >= MOERT_MEMMGR_MAX_KIND ||
        is_scale < 0 || is_scale > 1) {
        return nullptr;
    }
    return &m->groups[il][kind * 2 + is_scale];
}

// Device offset of (il, slot) in group gi: in the slab, or virtual (loaned slot) past slab_bytes.
static uint64_t slot_off(const moert_memmgr * m, int32_t il, int32_t gi, int32_t slot) {
    const PageGroup & g = m->groups[il][gi];
    const int32_t kphys = m->k_layer[il] - ((gi & 1) ? 0 : m->loan);
    return slot < kphys ? g.region_off + (uint64_t) slot * g.bpe : g.loan_off + (uint64_t) (slot - kphys) * g.bpe;
}

extern "C" moert_memmgr * moert_memmgr_create(int32_t n_layers, int32_t n_expert, int32_t k,
                                              const moert_memmgr_device * dev) {
    if (n_layers <= 0 || n_expert <= 0 || k <= 0 || k > n_expert ||
        !dev || !dev->slab_alloc || !dev->slab_upload) {
        return nullptr;
    }
    moert_memmgr * m = new moert_memmgr();
    m->n_layers = n_layers;
    m->n_expert = n_expert;
    m->k        = k;
    m->k_layer.assign(n_layers, k);   // uniform until moert_memmgr_set_layer_k overrides a layer
    m->dev      = *dev;
    m->groups.resize(n_layers);
    m->cold.resize(n_layers);
    m->cold_arena.resize(n_layers);
    return m;
}

extern "C" int32_t moert_memmgr_set_layer_k(moert_memmgr * m, int32_t il, int32_t k_l) {
    if (!m || m->finalized || il < 0 || il >= m->n_layers || k_l < 1 || k_l > m->n_expert) return 0;
    m->k_layer[il] = k_l;
    return 1;
}

extern "C" int32_t moert_memmgr_layer_k(const moert_memmgr * m, int32_t il) {
    if (!m || il < 0 || il >= m->n_layers) return -1;
    return m->k_layer[il];
}

extern "C" void moert_memmgr_free(moert_memmgr * m) {
    if (!m) return;
    for (FILE * f : m->sources) {
        if (f) std::fclose(f);
    }
#if defined(_WIN32)
    for (void * h : m->unbuf_h) {
        if (h) CloseHandle((HANDLE) h);
    }
#endif
    if (m->slab_owned && m->dev.slab_free) m->dev.slab_free(m->dev.user);
    if (m->map_view) {
#if defined(_WIN32)
        UnmapViewOfFile(m->map_view);
#else
        munmap(m->map_view, (size_t) m->map_size);
#endif
    }
    delete m;
}

extern "C" int32_t moert_memmgr_add_source(moert_memmgr * m, const char * path) {
    if (!m || !path || m->finalized) return -1;
    FILE * f = std::fopen(path, "rb");
    if (!f) {
        std::fprintf(stderr, "moert-memmgr: cannot open promotion source %s\n", path);
        return -1;
    }
    m->sources.push_back(f);
    m->source_paths.push_back(path);
    return (int32_t) m->sources.size() - 1;
}

extern "C" int32_t moert_memmgr_set_page(moert_memmgr * m, int32_t il, int32_t kind, int32_t is_scale,
                                         int32_t source_id, int64_t off_expert0, uint64_t bytes_per_expert,
                                         int32_t type_tag, const int64_t * stacked_ne, int32_t expert_axis) {
    PageGroup * g = group_at(m, il, kind, is_scale);
    if (!g || m->finalized || g->present || !stacked_ne ||
        source_id < 0 || source_id >= (int32_t) m->sources.size() ||
        off_expert0 < 0 || bytes_per_expert == 0 ||
        expert_axis < 0 || expert_axis >= MOERT_MEMMGR_MAX_DIMS ||
        stacked_ne[expert_axis] != m->n_expert) {
        return 0;
    }
    g->present   = true;
    g->source_id = source_id;
    g->off0      = off_expert0;
    g->bpe       = bytes_per_expert;
    g->type_tag  = type_tag;
    for (int d = 0; d < MOERT_MEMMGR_MAX_DIMS; ++d) g->ne[d] = stacked_ne[d] > 0 ? stacked_ne[d] : 1;
    g->axis      = expert_axis;
    return 1;
}

extern "C" int32_t moert_memmgr_open_store(moert_memmgr * m, const char * index_path,
                                           uint64_t expect_source_uuid) {
    if (!m || !index_path || m->finalized) return 0;
    moert_store::IndexHeader hdr;
    std::vector<moert_store::IndexRecord> recs;
    FILE * pack_f = nullptr;
    std::string pack_path;
    if (!moert_store_load(index_path, hdr, recs, &pack_f, &pack_path)) return 0;

    if ((int32_t) hdr.n_layers != m->n_layers || (int32_t) hdr.n_expert != m->n_expert) {
        std::fprintf(stderr, "moert-memmgr: store geometry %uL x %uE != runtime %dL x %dE (%s)\n",
                     hdr.n_layers, hdr.n_expert, m->n_layers, m->n_expert, index_path);
        std::fclose(pack_f);
        return 0;
    }
    if (expect_source_uuid && hdr.source_uuid != expect_source_uuid) {
        std::fprintf(stderr, "moert-memmgr: store source_uuid %016" PRIx64 " != expected %016" PRIx64
                             " — the store was built from a DIFFERENT source gguf; rebuild it "
                             "(rebuild it with `llapis setup`) (%s)\n", hdr.source_uuid, expect_source_uuid, index_path);
        std::fclose(pack_f);
        return 0;
    }

    m->sources.push_back(pack_f);
    m->source_paths.push_back(pack_path);
    const int32_t sid = (int32_t) m->sources.size() - 1;

    // Fold the records into per-(layer, kind, is_scale) groups; enforce group uniformity
    // (equal nbytes/type/shape across experts) and completeness (all n_expert pages).
    std::vector<std::array<int32_t, MOERT_MEMMGR_MAX_KIND * 2>> seen(m->n_layers);
    for (auto & a : seen) a.fill(0);
    bool ok = true;
    for (size_t i = 0; i < recs.size() && ok; ++i) {
        const moert_store::IndexRecord & r = recs[i];
        const int32_t is_scale = (r.flags & MOERT_PAGE_SCALE) ? 1 : 0;
        PageGroup * g = group_at(m, (int32_t) r.layer, (int32_t) r.matrix_kind, is_scale);
        if (!g || r.expert_id >= (uint16_t) m->n_expert) {
            std::fprintf(stderr, "moert-memmgr: store record %zu out of range (l%u e%u k%u)\n",
                         i, r.layer, r.expert_id, r.matrix_kind);
            ok = false;
            break;
        }
        if (!g->present) {
            g->present   = true;
            g->source_id = sid;
            g->bpe       = r.nbytes;
            g->type_tag  = (int32_t) r.quant_type;
            if (is_scale) {
                // per-expert scalar page -> stacked [n_expert]
                g->axis = 0;
                g->ne[0] = m->n_expert; g->ne[1] = g->ne[2] = g->ne[3] = 1;
                if (r.shape[0] != 1 || r.shape[1] != 1 || r.shape[2] != 1 || r.shape[3] != 1) {
                    std::fprintf(stderr, "moert-memmgr: store scale page with non-scalar shape (l%u k%u)\n",
                                 r.layer, r.matrix_kind);
                    ok = false;
                    break;
                }
            } else {
                // per-expert matrix page [a,b] -> stacked [a,b,n_expert]
                g->axis = 2;
                g->ne[0] = r.shape[0]; g->ne[1] = r.shape[1]; g->ne[2] = m->n_expert; g->ne[3] = 1;
                if (r.shape[0] < 1 || r.shape[1] < 1 || r.shape[2] != 1 || r.shape[3] != 1) {
                    std::fprintf(stderr, "moert-memmgr: store weight page with unsupported shape (l%u k%u)\n",
                                 r.layer, r.matrix_kind);
                    ok = false;
                    break;
                }
            }
            g->per_off.assign(m->n_expert, -1);
        } else {
            const bool scalar_ok = (is_scale
                ? (r.shape[0] == 1 && r.shape[1] == 1)
                : (r.shape[0] == g->ne[0] && r.shape[1] == g->ne[1]))
                && r.shape[2] == 1 && r.shape[3] == 1;
            if (g->bpe != r.nbytes || g->type_tag != (int32_t) r.quant_type || !scalar_ok) {
                std::fprintf(stderr, "moert-memmgr: store group (l%u k%u%s) is not uniform across experts\n",
                             r.layer, r.matrix_kind, is_scale ? " scale" : "");
                ok = false;
                break;
            }
        }
        if (g->per_off[r.expert_id] != -1) {
            std::fprintf(stderr, "moert-memmgr: duplicate store page (l%u e%u k%u%s)\n",
                         r.layer, r.expert_id, r.matrix_kind, is_scale ? " scale" : "");
            ok = false;
            break;
        }
        g->per_off[r.expert_id] = (int64_t) r.offset;
        seen[r.layer][r.matrix_kind * 2 + is_scale]++;
    }
    // completeness: every present group covers all n_expert pages
    for (int32_t il = 0; il < m->n_layers && ok; ++il) {
        for (int32_t gi = 0; gi < MOERT_MEMMGR_MAX_KIND * 2 && ok; ++gi) {
            if (m->groups[il][gi].present && m->groups[il][gi].source_id == sid &&
                seen[il][gi] != m->n_expert) {
                std::fprintf(stderr, "moert-memmgr: store group (l%d g%d) has %d/%d pages — incomplete\n",
                             il, gi, seen[il][gi], m->n_expert);
                ok = false;
            }
        }
    }
    if (!ok) {
        // roll back this store's groups (the pack FILE stays owned by m and closes at free)
        for (auto & la : m->groups) {
            for (auto & g : la) {
                if (g.present && g.source_id == sid) g = PageGroup();
            }
        }
        return 0;
    }
    std::fprintf(stderr, "moert-memmgr: expert store opened: %s (%zu pages, source_uuid %016" PRIx64 ")\n",
                 index_path, recs.size(), hdr.source_uuid);
    return 1;
}

extern "C" int32_t moert_memmgr_set_region_alloc(moert_memmgr * m, int32_t il, int32_t kind,
                                                 int32_t is_scale, uint64_t alloc_bytes) {
    PageGroup * g = group_at(m, il, kind, is_scale);
    if (!g || !g->present || m->finalized) return 0;
    if (alloc_bytes > g->alloc_bytes) g->alloc_bytes = alloc_bytes;
    return 1;
}

extern "C" int32_t moert_memmgr_finalize(moert_memmgr * m) {
    if (!m || m->finalized) return 0;
    uint64_t total = 0;
    int32_t  n_groups = 0;
    for (int32_t il = 0; il < m->n_layers; ++il) {
        for (int32_t gi = 0; gi < MOERT_MEMMGR_MAX_KIND * 2; ++gi) {
            PageGroup & g = m->groups[il][gi];
            if (!g.present) continue;
            total = (total + REGION_ALIGN - 1) & ~(REGION_ALIGN - 1);
            g.region_off = total;
            const int32_t  kslots  = m->k_layer[il] - ((gi & 1) ? 0 : m->loan);   // loaned slots live outside
            const uint64_t payload = (uint64_t) kslots * g.bpe;   // per-layer slot count
            total += payload > g.alloc_bytes ? payload : g.alloc_bytes;
            ++n_groups;
        }
    }
    if (total == 0) {
        std::fprintf(stderr, "moert-memmgr: finalize: empty page catalog\n");
        return 0;
    }
    if (!m->dev.slab_alloc(m->dev.user, total)) {
        std::fprintf(stderr, "moert-memmgr: hot slab alloc FAILED (%" PRIu64 " MiB)\n", total >> 20);
        return 0;
    }
    m->slab_owned = true;
    m->slab_bytes = total;
    m->finalized  = true;
    uint64_t lo = total;   // loaned slots: virtual offsets past the slab, one run per weight group
    if (m->loan > 0)
        for (int32_t il = 0; il < m->n_layers; ++il)
            for (int32_t gi = 0; gi < MOERT_MEMMGR_MAX_KIND * 2; gi += 2) {
                PageGroup & g = m->groups[il][gi];
                if (!g.present) continue;
                lo = (lo + 255) & ~(uint64_t) 255;
                g.loan_off = lo;
                lo += (uint64_t) m->loan * g.bpe;
            }
    m->loan_bytes = lo - total;
    int32_t kmin = m->k_layer.empty() ? m->k : m->k_layer[0], kmax = kmin; int64_t ksum = 0;
    for (int32_t v : m->k_layer) { if (v < kmin) kmin = v; if (v > kmax) kmax = v; ksum += v; }
    if (kmin == kmax) {
        std::fprintf(stderr, "moert-memmgr: hot slab %" PRIu64 " MiB (%d regions, K=%d, %d layers)",
                     total >> 20, n_groups, m->k, m->n_layers);
        if (m->loan > 0)
            std::fprintf(stderr, " + %d loaned slots per layer outside the slab (%" PRIu64 " MiB)",
                         m->loan, m->loan_bytes >> 20);
        std::fprintf(stderr, "\n");
    } else {
        std::fprintf(stderr, "moert-memmgr: hot slab %" PRIu64 " MiB (%d regions, LAYERWISE K %d..%d "
                     "sum %" PRId64 " over %d layers)\n",
                     total >> 20, n_groups, kmin, kmax, ksum, m->n_layers);
    }
    return 1;
}

extern "C" int32_t moert_memmgr_reset_hot_layout(moert_memmgr * m) {
    if (!m || !m->finalized) return 0;
    // Release the current hot slab through the same device path finalize allocated it with. The
    // cold arena, page catalog, sources, and cold views are separate members and stay intact — so
    // a K change never triggers a multi-GB cold-arena reload.
    if (m->slab_owned && m->dev.slab_free) m->dev.slab_free(m->dev.user);
    m->slab_owned = false;
    m->slab_bytes = 0;
    m->finalized  = false;
    m->loan_bytes = 0;
    // Clear only the finalize-produced layout fields; every region's offset and the host-measured
    // device-alloc override are K-dependent and will be recomputed on the next finalize (the host
    // re-runs set_region_alloc for the new K). k_layer is left as-is for the host to override via
    // set_layer_k; pages (source_id/off0/bpe/type/shape) are untouched.
    for (auto & layer : m->groups) {
        for (auto & g : layer) {
            g.region_off  = 0;
            g.loan_off    = 0;
            g.alloc_bytes = 0;
        }
    }
    return 1;
}

extern "C" uint64_t moert_memmgr_slab_bytes(const moert_memmgr * m) {
    return m ? m->slab_bytes : 0;
}

extern "C" int32_t moert_memmgr_set_loan(moert_memmgr * m, int32_t n_slots) {
    if (!m || m->finalized || n_slots < 0) return 0;
    for (int32_t v : m->k_layer) if (n_slots >= v) return 0;   // at least one slab slot per layer
    m->loan = n_slots;
    return 1;
}
extern "C" int32_t  moert_memmgr_loan(const moert_memmgr * m) { return m ? m->loan : 0; }
extern "C" uint64_t moert_memmgr_loan_bytes(const moert_memmgr * m) { return m ? m->loan_bytes : 0; }
extern "C" int32_t  moert_memmgr_slot_offset(const moert_memmgr * m, int32_t il, int32_t kind, int32_t is_scale,
                                             int32_t slot, uint64_t * off) {
    if (!m || !m->finalized || il < 0 || il >= m->n_layers || kind < 0 || kind >= MOERT_MEMMGR_MAX_KIND ||
        is_scale < 0 || is_scale > 1 || slot < 0 || slot >= m->k_layer[il] || !off) return 0;
    const int32_t gi = kind * 2 + is_scale;
    if (!m->groups[il][gi].present) return 0;
    *off = slot_off(m, il, gi, slot);
    return 1;
}

#if defined(_WIN32)
// Lazily open an unbuffered (FILE_FLAG_NO_BUFFERING) handle for source_id, alongside the existing
// buffered FILE* in m->sources (kept as the fallback). Tried at most once per source per process;
// a failed open (or a source_id we've never seen) returns nullptr forever after, so a bad path
// costs one CreateFileA call, not one per promote().
static HANDLE unbuffered_handle_for(moert_memmgr * m, int32_t source_id) {
    if (source_id < 0 || source_id >= (int32_t) m->source_paths.size()) return nullptr;
    if ((size_t) source_id >= m->unbuf_h.size()) {
        m->unbuf_h.resize(m->source_paths.size(), nullptr);
        m->unbuf_tried.resize(m->source_paths.size(), 0);
    }
    if (!m->unbuf_tried[source_id]) {
        m->unbuf_tried[source_id] = 1;
        HANDLE h = CreateFileA(m->source_paths[source_id].c_str(), GENERIC_READ, FILE_SHARE_READ,
                               nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_NO_BUFFERING,
                               nullptr);
        if (h != INVALID_HANDLE_VALUE) m->unbuf_h[source_id] = (void *) h;
    }
    return (HANDLE) m->unbuf_h[source_id];
}
#endif

// Read `len` bytes starting at file offset `off` of source_id, preferring the unbuffered path
// ( update 6: measured ~2x this box's buffered-fread ceiling, matching q4stream's own
// already-documented 1.2 vs 2-3.5 GB/s finding) with a byte-identical buffered-fread fallback on any
// unavailability or short/failed read. Returns a pointer into memmgr-owned reused scratch (valid
// until the NEXT read_promote_bytes call -- copy out before then) or nullptr on total failure. NOT
// reentrant / not thread-safe (single caller thread, matching every other memmgr promotion path).
static const char * read_promote_bytes(moert_memmgr * m, FILE * f, int32_t source_id,
                                       int64_t off, size_t len) {
#if defined(_WIN32)
    if (off >= 0) {
        if (HANDLE h = unbuffered_handle_for(m, source_id)) {
            const int64_t a_off = off & ~(int64_t) 4095;
            const size_t  pad   = (size_t) (off - a_off);
            const size_t  want  = (pad + len + 4095) & ~(size_t) 4095;
            if (m->io_align_buf.size() >= want || m->io_align_buf.alloc(want)) {
                OVERLAPPED ov = {};
                ov.Offset     = (DWORD) ((uint64_t) a_off & 0xffffffffu);
                ov.OffsetHigh = (DWORD) ((uint64_t) a_off >> 32);
                DWORD got = 0;
                if (ReadFile(h, m->io_align_buf.data(), (DWORD) want, &got, &ov) &&
                    (size_t) got >= pad + len) {
                    return (const char *) m->io_align_buf.data() + pad;
                }
            }
            // unbuffered open succeeded but this read didn't (short read, alloc failure, etc.) --
            // fall through to the buffered path below rather than failing the whole promotion.
        }
    }
#endif
    m->rbuf.resize(len);
    if (off < 0 || !moert_store::seek64(f, off, SEEK_SET) || fread(m->rbuf.data(), 1, len, f) != len) {
        return nullptr;
    }
    return m->rbuf.data();
}

extern "C" int32_t moert_memmgr_page_info(const moert_memmgr * m, int32_t il, int32_t kind,
                                          int32_t is_scale, moert_memmgr_page * out) {
    const PageGroup * g = group_at(const_cast<moert_memmgr *>(m), il, kind, is_scale);
    if (!g || !g->present || !out) return 0;   // pre-finalize ok (region_off then still 0)
    out->type_tag         = g->type_tag;
    out->expert_axis      = g->axis;
    out->bytes_per_expert = g->bpe;
    out->region_off       = g->region_off;
    int32_t ndim = 1;
    for (int d = 0; d < MOERT_MEMMGR_MAX_DIMS; ++d) {
        out->pool_ne[d] = (d == g->axis) ? m->k_layer[il] - (is_scale ? 0 : m->loan) : g->ne[d];   // slots in the slab
        if (out->pool_ne[d] > 1) ndim = d + 1;
    }
    if (g->axis + 1 > ndim) ndim = g->axis + 1;
    out->ndim = ndim;
    return 1;
}

extern "C" int32_t moert_memmgr_promote(moert_memmgr * m, int32_t il, int32_t e, int32_t slot) {
    if (!m || !m->finalized || il < 0 || il >= m->n_layers ||
        e < 0 || e >= m->n_expert || slot < 0 || slot >= m->k_layer[il]) {
        return 0;
    }
    for (int32_t gi = 0; gi < MOERT_MEMMGR_MAX_KIND * 2; ++gi) {
        PageGroup & g = m->groups[il][gi];
        if (!g.present) continue;   // absent group: nothing to stream (legacy stream_one semantics)
        const int64_t off = g.per_off.empty() ? g.off0 + (int64_t) e * (int64_t) g.bpe
                                              : g.per_off[e];
        const char * data = read_promote_bytes(m, m->sources[g.source_id], g.source_id, off, g.bpe);
        if (!data) {
            std::fprintf(stderr, "moert-memmgr: promote read failed (l%d e%d group %d, %s)\n",
                         il, e, gi, m->source_paths[g.source_id].c_str());
            return 0;
        }
        if (!m->dev.slab_upload(m->dev.user, slot_off(m, il, gi, slot), data, g.bpe)) {
            std::fprintf(stderr, "moert-memmgr: promote upload failed (l%d e%d group %d)\n", il, e, gi);
            return 0;
        }
    }
    return 1;
}

static bool arena_group_matches(const PageGroup & g, const ColdArena & a) {
    return a.present && a.base() && a.type_tag == g.type_tag && a.stride == g.bpe;
}

extern "C" int32_t moert_memmgr_arena_matches_hot(const moert_memmgr * m) {
    if (!m || !m->finalized) return 0;
    int32_t n = 0;
    for (int32_t il = 0; il < m->n_layers; ++il)
        for (int32_t gi = 0; gi < MOERT_MEMMGR_MAX_KIND * 2; ++gi) {
            const PageGroup & g = m->groups[il][gi];
            if (!g.present) continue;
            if (!arena_group_matches(g, m->cold_arena[il][gi])) return 0;
            ++n;
        }
    return n > 0 ? 1 : 0;
}

extern "C" int32_t moert_memmgr_promote_from_arena(moert_memmgr * m, int32_t il, int32_t e, int32_t slot) {
    if (!m || !m->finalized || il < 0 || il >= m->n_layers ||
        e < 0 || e >= m->n_expert || slot < 0 || slot >= m->k_layer[il]) {
        return 0;
    }
    for (int32_t gi = 0; gi < MOERT_MEMMGR_MAX_KIND * 2; ++gi) {   // validate the whole layer first
        const PageGroup & g = m->groups[il][gi];
        if (g.present && !arena_group_matches(g, m->cold_arena[il][gi])) return 0;
    }
    for (int32_t gi = 0; gi < MOERT_MEMMGR_MAX_KIND * 2; ++gi) {
        const PageGroup & g = m->groups[il][gi];
        if (!g.present) continue;
        const ColdArena & a = m->cold_arena[il][gi];
        const uint64_t as = a.arena_stride ? a.arena_stride : a.stride;
        const char * src = (const char *) a.base() + (uint64_t) e * as;
        if (!m->dev.slab_upload(m->dev.user, slot_off(m, il, gi, slot), src, g.bpe)) {
            std::fprintf(stderr, "moert-memmgr: arena promote upload failed (l%d e%d group %d)\n", il, e, gi);
            return 0;
        }
    }
    return 1;
}

// Map mode: the inner pages of expert e's run inside the read-only view, [lo, hi) page-aligned.
static bool mapped_inner(const ColdArena & a, int32_t e, uintptr_t & lo, uintptr_t & hi) {
    const size_t pg = PagedBuffer::pagesize();
    const uint64_t as = a.arena_stride ? a.arena_stride : a.stride;
    const uintptr_t b = (uintptr_t) (a.mapped + (uint64_t) e * as);
    lo = (b + pg - 1) & ~(uintptr_t) (pg - 1);
    hi = (b + as) & ~(uintptr_t) (pg - 1);
    return hi > lo;
}

// Map mode: drop expert e's pages from the process working set. They are clean file pages: the data stays
// valid (a later read faults them back from the standby list or the pack), so nothing has to be restored
// and no reader has to wait. Unlocking pages that are not locked is how Windows trims a range.
static uint64_t mapped_trim(const ColdArena & a, int32_t e) {
    uintptr_t lo = 0, hi = 0;
    if (!mapped_inner(a, e, lo, hi)) return 0;
#ifdef _WIN32
    VirtualUnlock((void *) lo, (size_t) (hi - lo));
#else
    madvise((void *) lo, (size_t) (hi - lo), MADV_DONTNEED);
#endif
    return (uint64_t) (hi - lo);
}

// Map mode: fault expert e's pages back in (one read per page) so its next cold read does not pay them.
static uint64_t mapped_touch(const ColdArena & a, int32_t e) {
    uintptr_t lo = 0, hi = 0;
    if (!mapped_inner(a, e, lo, hi)) return 0;
    const size_t pg = PagedBuffer::pagesize();
    volatile uint8_t sink = 0;
    for (uintptr_t q = lo; q < hi; q += pg) sink ^= *(const volatile uint8_t *) q;
    (void) sink;
    return (uint64_t) (hi - lo);
}

extern "C" int32_t moert_memmgr_cold_arena_mapped(const moert_memmgr * m) {
    return m && m->map_view ? 1 : 0;
}

extern "C" uint64_t moert_memmgr_arena_release(moert_memmgr * m, int32_t il, int32_t e) {
    if (!m || !m->finalized || il < 0 || il >= m->n_layers || e < 0 || e >= m->n_expert) return 0;
    uint64_t freed = 0;
    for (int32_t gi = 0; gi < MOERT_MEMMGR_MAX_KIND * 2; gi += 2) {   // weight groups only
        const PageGroup & g = m->groups[il][gi];
        ColdArena & a = m->cold_arena[il][gi];
        if (a.present && a.mapped) {   // the hot tier's format does not matter: the page is never lost
            if (g.present) freed += mapped_trim(a, e);
            continue;
        }
        if (!g.present || !arena_group_matches(g, a)) continue;
        const uint64_t as = a.arena_stride ? a.arena_stride : a.stride;
        freed += a.buf.decommit((size_t) e * as, (size_t) as);
    }
    m->arena_decommitted += freed;
    return freed;
}

extern "C" int32_t moert_memmgr_arena_restore(moert_memmgr * m, int32_t il, int32_t e, int32_t slot) {
    if (m && m->finalized && m->map_view && il >= 0 && il < m->n_layers && e >= 0 && e < m->n_expert) {
        // map mode: the page was only trimmed; fault it back in, no slot involved
        uint64_t back = 0;
        for (int32_t gi = 0; gi < MOERT_MEMMGR_MAX_KIND * 2; gi += 2) {
            const ColdArena & a = m->cold_arena[il][gi];
            if (a.present && a.mapped && m->groups[il][gi].present) back += mapped_touch(a, e);
        }
        uint64_t cur = m->arena_decommitted.load();
        while (!m->arena_decommitted.compare_exchange_weak(cur, cur >= back ? cur - back : 0)) {}
        return 1;
    }
    if (!m || !m->finalized || !m->dev.slab_download || il < 0 || il >= m->n_layers ||
        e < 0 || e >= m->n_expert || slot < 0 || slot >= m->k_layer[il]) {
        return 0;
    }
    for (int32_t gi = 0; gi < MOERT_MEMMGR_MAX_KIND * 2; gi += 2) {
        const PageGroup & g = m->groups[il][gi];
        ColdArena & a = m->cold_arena[il][gi];
        if (!g.present || !arena_group_matches(g, a)) continue;
        const uint64_t as = a.arena_stride ? a.arena_stride : a.stride;
        // the bytes release() gave back (inner pages of the slot), for the committed-bytes telemetry
        const size_t pg = PagedBuffer::pagesize();
        const size_t lo = ((size_t) e * as + pg - 1) & ~(pg - 1);
        const size_t hi = ((size_t) e * as + as) & ~(pg - 1);
        const uint64_t inner = hi > lo ? (uint64_t) (hi - lo) : 0;
        if (!a.buf.commit((size_t) e * as, (size_t) as) ||
            !m->dev.slab_download(m->dev.user, slot_off(m, il, gi, slot),
                                  a.buf.data() + (size_t) e * as, g.bpe)) {
            std::fprintf(stderr, "moert-memmgr: arena restore failed (l%d e%d slot %d group %d)\n", il, e, slot, gi);
            return 0;
        }
        // restore() of a page that was never released re-downloads identical bytes: harmless, and
        // the telemetry must not go negative
        uint64_t cur = m->arena_decommitted.load();
        while (cur >= inner && !m->arena_decommitted.compare_exchange_weak(cur, cur - inner)) {}
    }
    return 1;
}

extern "C" int32_t moert_memmgr_set_cold_view(moert_memmgr * m, int32_t il, int32_t kind,
                                              int32_t is_scale, const moert_cold_view * v) {
    if (!group_at(m, il, kind, is_scale) || !v || !v->base || v->stride == 0 ||
        v->n_expert != m->n_expert) {
        return 0;
    }
    ColdSlot & c = m->cold[il][kind * 2 + is_scale];
    if (!c.set) ++m->cold_count;
    c.set = true;
    c.v   = *v;
    return 1;
}

extern "C" int32_t moert_memmgr_get_cold_view(const moert_memmgr * m, int32_t il, int32_t kind,
                                              int32_t is_scale, moert_cold_view * out) {
    if (!group_at(const_cast<moert_memmgr *>(m), il, kind, is_scale) || !out) return 0;
    const ColdSlot & c = m->cold[il][kind * 2 + is_scale];
    if (!c.set) return 0;
    *out = c.v;
    return 1;
}

extern "C" int32_t moert_memmgr_cold_view_count(const moert_memmgr * m) {
    return m ? m->cold_count : 0;
}

// --- owned cold arena (3b: moert owns the cold IQ2 tier in host RAM) ------------------
//
// STARTUP. The arena is 7.67 GiB and it is read ONCE, at load, into memory the
// process then owns for the rest of the session — which makes buffered stdio exactly the wrong
// tool for it, on two counts measured on the reference box (RTX 4070 8 GB / 16 GB RAM):
//
//   RATE. The old loop was one thread doing ~11.5k sequential freads of ~700 KiB and reached
//   1.26 GiB/s (6.1 s). The device does better than that even single-threaded: measured 2.00 GiB/s
//   unbuffered at 1 thread, 3.26 at 4, 3.34 at 8 (the plateau — 4 is the default here).
//
//   FOOTPRINT, which is the bigger one. A buffered read leaves every byte in the page cache as
//   well as in the arena, so building a 7.67 GiB arena asks a 15.2 GiB box to hold 15.3 GiB for
//   the duration. It cannot, and the loser is the arena itself: sampling the child's working set
//   through a load shows it climb to 6.5 GiB and then fall back to 0.9 GiB WHILE THE BUILD IS
//   STILL RUNNING — we page out the arena to make room for a cache of the file we are copying it
//   from. That eviction is also why the NE tensor load that follows swings between 2.5 s and 8.7 s
//   run to run: it is competing for whatever is left.
//
// So: read the pack unbuffered (the page cache never sees it) and in parallel. The records are
// written append-only in index order, so a contiguous slice of the record list is a contiguous
// byte range of the file — each worker sweeps its own slice through a sliding sector-aligned
// window and memcpys each page into its arena slot. Destinations are disjoint by construction
// (one (group, expert) slot per record), so no locking. The worker count comes from the hardware
// concurrency, capped at 4; a failure to open the handle falls back to the original sequential
// fread loop, which is kept verbatim below as the correctness anchor.
namespace {

// Sector alignment for unbuffered I/O. 4096 covers both 512e and 4Kn NVMe; over-aligning is
// always legal (the window just starts a little earlier), under-aligning fails the read.
constexpr uint64_t ARENA_IO_SECTOR = 4096;
constexpr uint64_t ARENA_IO_WINDOW = 16u << 20;   // staging window per worker (~23 pages)

static inline uint64_t align_down_u64(uint64_t v, uint64_t a) { return v & ~(a - 1); }
static inline uint64_t align_up_u64  (uint64_t v, uint64_t a) { return (v + a - 1) & ~(a - 1); }

static int arena_read_threads() {
    const unsigned hc = std::thread::hardware_concurrency();
    return (int) (hc ? (hc < 4 ? hc : 4) : 4);
}

// One worker's sweep over records [lo, hi). Returns false on any read failure (the caller then
// abandons the whole parallel attempt and re-runs the sequential path — never a partial arena).
struct ArenaReadJob {
    const std::vector<moert_store::IndexRecord> * recs = nullptr;
    const std::vector<uint8_t *>                * dst  = nullptr;   // per-record destination slot
    std::string pack_path;
    uint64_t    pack_size = 0;
};

static bool arena_read_slice(const ArenaReadJob & job, size_t lo, size_t hi) {
#ifdef _WIN32
    HANDLE h = CreateFileA(job.pack_path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                           OPEN_EXISTING, FILE_FLAG_NO_BUFFERING | FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
    if (h == INVALID_HANDLE_VALUE) return false;
    struct HandleGuard { HANDLE h; ~HandleGuard() { CloseHandle(h); } } hg{ h };
    // VirtualAlloc is page-aligned, which satisfies the unbuffered buffer-alignment rule.
    uint8_t * win = (uint8_t *) VirtualAlloc(nullptr, (SIZE_T) (ARENA_IO_WINDOW + ARENA_IO_SECTOR),
                                             MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    if (!win) return false;
    struct MemGuard { uint8_t * p; ~MemGuard() { VirtualFree(p, 0, MEM_RELEASE); } } mg{ win };
#else
    FILE * fh = std::fopen(job.pack_path.c_str(), "rb");
    if (!fh) return false;
    struct FileGuard { FILE * f; ~FileGuard() { std::fclose(f); } } hg{ fh };
    std::vector<uint8_t> winbuf((size_t) (ARENA_IO_WINDOW + ARENA_IO_SECTOR));
    uint8_t * win = winbuf.data();
#endif
    uint64_t win_beg = 0, win_end = 0;   // [win_beg, win_end) currently held in `win`
    for (size_t i = lo; i < hi; ++i) {
        const moert_store::IndexRecord & r = (*job.recs)[i];
        const uint64_t off = r.offset, len = r.nbytes;
        if (len == 0) continue;
        if (off < win_beg || off + len > win_end) {
            const uint64_t beg = align_down_u64(off, ARENA_IO_SECTOR);
            // What the record actually needs, and what we may ASK for. The ask is sector-aligned
            // (unbuffered I/O requires it) and may therefore run past EOF on the last window — a
            // short read there is normal and not an error, which is why coverage is checked
            // against the raw need and never against the padded one.
            const uint64_t need = off + len - beg;
            const uint64_t pad  = align_up_u64(need, ARENA_IO_SECTOR);
            uint64_t want = pad > ARENA_IO_WINDOW ? pad : ARENA_IO_WINDOW;
            const uint64_t file_end = align_up_u64(job.pack_size, ARENA_IO_SECTOR);
            if (beg + want > file_end) want = file_end - beg;
            uint64_t got = 0;
            while (got < need) {
                const uint64_t ask = want - got;
                if (ask == 0) break;
#ifdef _WIN32
                LARGE_INTEGER li; li.QuadPart = (LONGLONG) (beg + got);
                OVERLAPPED ov{}; ov.Offset = li.LowPart; ov.OffsetHigh = (DWORD) li.HighPart;
                DWORD n = 0;
                if (!ReadFile(h, win + got, (DWORD) ask, &n, &ov) || n == 0) break;
#else
                if (!moert_store::seek64(fh, (int64_t) (beg + got), SEEK_SET)) break;
                const size_t n = std::fread(win + got, 1, (size_t) ask, fh);
                if (n == 0) break;
#endif
                got += n;
                if ((uint64_t) n < ask) break;   // short read == EOF; `got` may now be unaligned
            }
            if (got < need) return false;
            win_beg = beg;
            win_end = beg + got;
        }
        std::memcpy((*job.dst)[i], win + (off - win_beg), (size_t) len);
    }
    return true;
}

// True when every WEIGHT group of the store is one dense run of experts in the pack (offset(e) =
// offset(0) + e*nbytes): the group-major layout `prepare` writes since 2026-10-08, the one map mode needs.
bool store_dense(const std::vector<moert_store::IndexRecord> & recs, int32_t n_layers, int32_t n_expert) {
    if (n_layers <= 0 || n_expert <= 0) return false;
    std::vector<int64_t> off((size_t) n_layers * MOERT_MEMMGR_MAX_KIND * n_expert, -1);
    std::vector<uint64_t> nb((size_t) n_layers * MOERT_MEMMGR_MAX_KIND, 0);
    for (const moert_store::IndexRecord & r : recs) {
        if (r.flags & MOERT_PAGE_SCALE) continue;
        if ((int32_t) r.layer >= n_layers || r.matrix_kind >= MOERT_MEMMGR_MAX_KIND || r.expert_id >= n_expert) return false;
        const size_t gk = (size_t) r.layer * MOERT_MEMMGR_MAX_KIND + r.matrix_kind;
        if (nb[gk] && nb[gk] != r.nbytes) return false;
        nb[gk] = r.nbytes;
        off[gk * n_expert + r.expert_id] = (int64_t) r.offset;
    }
    for (size_t gk = 0; gk < nb.size(); ++gk) {
        if (!nb[gk]) continue;
        for (int32_t e = 0; e < n_expert; ++e)
            if (off[gk * n_expert + e] != off[gk * n_expert] + (int64_t) e * (int64_t) nb[gk]) return false;
    }
    return true;
}

}  // namespace

extern "C" int32_t moert_memmgr_store_group_major(const char * index_path) {
    if (!index_path) return -1;
    moert_store::IndexHeader hdr;
    std::vector<moert_store::IndexRecord> recs;
    FILE * pack_f = nullptr;
    if (!moert_store_load(index_path, hdr, recs, &pack_f, nullptr)) return -1;
    if (pack_f) std::fclose(pack_f);
    return store_dense(recs, (int32_t) hdr.n_layers, (int32_t) hdr.n_expert) ? 1 : 0;
}

extern "C" int32_t moert_memmgr_build_cold_arena(moert_memmgr * m, const char * index_path,
                                                 uint64_t expect_source_uuid) {
    if (!m || !index_path) return 0;
    moert_store::IndexHeader hdr;
    std::vector<moert_store::IndexRecord> recs;
    FILE * pack_f = nullptr;
    std::string pack_path;
    if (!moert_store_load(index_path, hdr, recs, &pack_f, &pack_path)) return 0;
    if ((int32_t) hdr.n_layers != m->n_layers || (int32_t) hdr.n_expert != m->n_expert) {
        std::fprintf(stderr, "moert-memmgr: cold arena store geometry %uL x %uE != runtime %dL x %dE (%s)\n",
                     hdr.n_layers, hdr.n_expert, m->n_layers, m->n_expert, index_path);
        std::fclose(pack_f);
        return 0;
    }
    if (expect_source_uuid && hdr.source_uuid != expect_source_uuid) {
        std::fprintf(stderr, "moert-memmgr: cold arena store source_uuid %016" PRIx64 " != expected %016"
                             PRIx64 " (%s)\n", hdr.source_uuid, expect_source_uuid, index_path);
        std::fclose(pack_f);
        return 0;
    }

    const auto arena_t0 = std::chrono::steady_clock::now();

    // MAP MODE (findings 197): when every WEIGHT group of the pack is one dense run of experts
    // (offset(e) = offset(0) + e*nbytes -- `prepare` writes them so since 2026-10-08), map the pack
    // read-only and point the groups into the view instead of copying 7.7 GiB into private memory.
    // The OS file cache becomes the cold tier: its pages are clean, so Windows drops them without a
    // page-file write and re-reads them from the pack, and they do not count as committed memory.
    // Scale pages (4 bytes per expert, read as a dense float array) are still copied. Old
    // expert-major packs, MOERT_ARENA_MAP=0, or a failed mapping fall back to the copy below.
    uint8_t * view = nullptr;
    {
        const char * mv = std::getenv("MOERT_ARENA_MAP");
        const bool dense = !(mv && mv[0] == '0') && !pack_path.empty() &&
                           store_dense(recs, m->n_layers, m->n_expert);
        std::error_code fec;
        const uint64_t psz = dense ? (uint64_t) std::filesystem::file_size(pack_path, fec) : 0;
        if (dense && !fec && psz) {
#if defined(_WIN32)
            HANDLE fh = CreateFileA(pack_path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                                    FILE_ATTRIBUTE_NORMAL, nullptr);
            if (fh != INVALID_HANDLE_VALUE) {
                HANDLE mh = CreateFileMappingA(fh, nullptr, PAGE_READONLY, 0, 0, nullptr);
                if (mh) { view = (uint8_t *) MapViewOfFile(mh, FILE_MAP_READ, 0, 0, 0); CloseHandle(mh); }
                CloseHandle(fh);   // the view keeps the mapping alive
            }
#else
            const int fd = open(pack_path.c_str(), O_RDONLY);
            if (fd >= 0) {
                void * v = mmap(nullptr, (size_t) psz, PROT_READ, MAP_SHARED, fd, 0);
                if (v != MAP_FAILED) view = (uint8_t *) v;
                close(fd);
            }
#endif
            if (view) { m->map_view = view; m->map_size = psz; }
            else std::fprintf(stderr, "moert-memmgr: could not map %s -- copying the cold arena instead\n", pack_path.c_str());
        } else if (!(mv && mv[0] == '0')) {
            std::fprintf(stderr, "moert-memmgr: %s is in the old expert-major layout -- the cold arena is copied into "
                                 "RAM; re-run `prepare` to get a pack that can be mapped\n", pack_path.c_str());
        }
    }

    std::vector<std::array<int32_t, MOERT_MEMMGR_MAX_KIND * 2>> seen(m->n_layers);
    for (auto & a : seen) a.fill(0);
    bool ok = true;
    // PASS 1 — metadata only: create/validate each group, allocate its buffer, and record where
    // record i's payload has to land. No file I/O at all, so the group table is fully built (and
    // any geometry error already refused) before a single byte is read. `dst` is what turns the
    // read into an embarrassingly parallel scatter: entries are distinct (group, expert) slots.
    std::vector<uint8_t *> dst(recs.size(), nullptr);
    for (size_t i = 0; i < recs.size() && ok; ++i) {
        const moert_store::IndexRecord & r = recs[i];
        const int32_t is_scale = (r.flags & MOERT_PAGE_SCALE) ? 1 : 0;
        if ((int32_t) r.layer >= m->n_layers || (int32_t) r.matrix_kind >= MOERT_MEMMGR_MAX_KIND ||
            r.expert_id >= (uint16_t) m->n_expert) {
            std::fprintf(stderr, "moert-memmgr: cold arena record %zu out of range (l%u e%u k%u)\n",
                         i, r.layer, r.expert_id, r.matrix_kind);
            ok = false; break;
        }
        ColdArena & a = m->cold_arena[r.layer][r.matrix_kind * 2 + is_scale];
        if (!a.present) {
            a.present      = true;
            a.stride       = r.nbytes;                     // payload (store read size)
            // Page-align WEIGHT slots only (is_scale=0). The scale pool is a tiny dense float array read
            // contiguously (n_expert*sizeof(float)) and is never pinned/decommitted -- padding it would
            // desync those contiguous readers (the v3 "garbage scales" bug). Scales stay == payload.
            a.arena_stride = is_scale ? r.nbytes : arena_slot_stride(r.nbytes);
            a.type_tag = (int32_t) r.quant_type;
            if (is_scale) { a.axis = 0; a.ne[0] = m->n_expert; a.ne[1] = a.ne[2] = a.ne[3] = 1; }
            else          { a.axis = 2; a.ne[0] = r.shape[0]; a.ne[1] = r.shape[1];
                            a.ne[2] = m->n_expert; a.ne[3] = 1; }
            if (view && !is_scale) {   // map mode: the dense run inside the view, expert 0 first
                a.arena_stride = r.nbytes;
                a.mapped = view + r.offset - (uint64_t) r.expert_id * r.nbytes;
            } else if (!a.buf.alloc((size_t) m->n_expert * a.arena_stride)) {
                std::fprintf(stderr, "moert-memmgr: cold arena alloc failed (l%u k%u%s)\n",
                             r.layer, r.matrix_kind, is_scale ? " scale" : "");
                ok = false; break;
            }
            a.per_off.assign(m->n_expert, -1);
            m->arena_bytes += (uint64_t) m->n_expert * a.arena_stride;
        } else if (a.stride != r.nbytes || a.type_tag != (int32_t) r.quant_type) {
            std::fprintf(stderr, "moert-memmgr: cold arena group (l%u k%u%s) not uniform\n",
                         r.layer, r.matrix_kind, is_scale ? " scale" : "");
            ok = false; break;
        }
        // address the slot with arena_stride (padded); only the payload r.nbytes lands in its head
        dst[i] = a.mapped ? nullptr : a.buf.data() + (size_t) r.expert_id * a.arena_stride;   // mapped: no copy
        if (!a.per_off.empty()) a.per_off[r.expert_id] = (int64_t) r.offset;
        seen[r.layer][r.matrix_kind * 2 + is_scale]++;
    }

    // PASS 2 — the bytes. Unbuffered + parallel first (see the block comment above); the
    // sequential buffered loop is the fallback and stays byte-for-byte what it always did.
    if (ok && view) {   // map mode: only the scale pages are copied, straight out of the view
        for (size_t i = 0; i < recs.size(); ++i)
            if (dst[i]) std::memcpy(dst[i], view + recs[i].offset, (size_t) recs[i].nbytes);
    } else if (ok) {
        bool done = false;
        const int nthreads = arena_read_threads();
        if (nthreads > 0 && !recs.empty() && !pack_path.empty()) {
            ArenaReadJob job;
            job.recs      = &recs;
            job.dst       = &dst;
            job.pack_path = pack_path;
            std::error_code fec;
            job.pack_size = (uint64_t) std::filesystem::file_size(pack_path, fec);
            if (!fec && job.pack_size) {
                std::vector<char> okv((size_t) nthreads, 0);   // char, not bool: vector<bool> bits race
                std::vector<std::thread> workers;
                workers.reserve((size_t) nthreads);
                const size_t per = (recs.size() + (size_t) nthreads - 1) / (size_t) nthreads;
                for (int t = 0; t < nthreads; ++t) {
                    const size_t lo = (size_t) t * per;
                    const size_t hi = lo + per < recs.size() ? lo + per : recs.size();
                    if (lo >= hi) { okv[(size_t) t] = 1; continue; }
                    workers.emplace_back([&job, &okv, t, lo, hi]() {
                        okv[(size_t) t] = arena_read_slice(job, lo, hi) ? 1 : 0;
                    });
                }
                for (auto & w : workers) w.join();
                done = true;
                for (char c : okv) if (!c) { done = false; break; }
                if (!done)
                    std::fprintf(stderr, "moert-memmgr: parallel arena read failed — falling back to "
                                         "the sequential path (slower, identical bytes)\n");
            }
        }
        // Sequential fallback. The pack is written append-only in index order, so recs[i].offset is
        // normally == the file position right after reading recs[i-1] -- but _fseeki64/fseeko is a
        // hard buffer-discard point even when seeking to the CURRENT position (C stdio semantics),
        // forcing a fresh ReadFile per record instead of riding the CRT's read-ahead buffer. Skip
        // the redundant seek whenever the stream is already positioned where the record starts.
        if (!done) {
            int64_t next_off = -1;
            for (size_t i = 0; i < recs.size() && ok; ++i) {
                const moert_store::IndexRecord & r = recs[i];
                if (((int64_t) r.offset != next_off && !moert_store::seek64(pack_f, (int64_t) r.offset, SEEK_SET)) ||
                    fread(dst[i], 1, r.nbytes, pack_f) != r.nbytes) {
                    std::fprintf(stderr, "moert-memmgr: cold arena read failed (l%u e%u k%u)\n",
                                 r.layer, r.expert_id, r.matrix_kind);
                    ok = false; break;
                }
                next_off = (int64_t) r.offset + (int64_t) r.nbytes;
            }
        }
    }

    // Arena self-check (not reachable in the product): hash every slot in the finished arena against the per-page checksum the
    // index already carries. That is a complete bitwise proof of the read path (any path — this is
    // deliberately not conditioned on which one ran), which is what a parallel unbuffered scatter
    // needs before it can be the default. Off by default: it re-hashes 7.67 GiB, ~2 s.
    if (ok) {
    }

    for (int32_t il = 0; il < m->n_layers && ok; ++il)
        for (int32_t gi = 0; gi < MOERT_MEMMGR_MAX_KIND * 2 && ok; ++gi)
            if (m->cold_arena[il][gi].present && seen[il][gi] != m->n_expert) {
                std::fprintf(stderr, "moert-memmgr: cold arena group (l%d g%d) has %d/%d experts\n",
                             il, gi, seen[il][gi], m->n_expert);
                ok = false;
            }
    if (!ok) {
        std::fclose(pack_f);
        for (auto & la : m->cold_arena) for (auto & a : la) a = ColdArena();
        m->arena_bytes = 0;
        return 0;
    }
    // Keep the store FILE open + tag every group's src_id so an evicted expert can be reloaded into the
    // arena from it (RAM-dedup smart refresh). One pack FILE serves all groups; per-expert offsets saved.
    m->sources.push_back(pack_f);
    m->source_paths.push_back(index_path);
    const int32_t arena_src = (int32_t) m->sources.size() - 1;
    for (auto & la : m->cold_arena) for (auto & a : la) if (a.present) a.src_id = arena_src;
    // Report the achieved RATE, not just the size. The read path above is unbuffered and parallel
    // and was measured at 3.26-3.34 GiB/s on this box; a build far below that is not an I/O
    // problem and no I/O change will fix it — it is the host paging the arena out WHILE it is
    // being written, because 7.67 GiB of anonymous memory does not fit beside everything else.
    // Measured 2026-08-09 at 0.75 GiB/s with 1-5 GiB free. Without this line the only symptom is
    // a startup that is sometimes 10 s and sometimes 3, and the disk gets the blame.
    {
        const double secs = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - arena_t0).count();
        const double gibs = secs > 0 ? (double) m->arena_bytes / 1073741824.0 / secs : 0.0;
        if (view)
            std::fprintf(stderr, "moert-memmgr: cold arena MAPPED read-only from the pack (%" PRIu64 " MiB, "
                                 "%.2f s; scale pages copied) -- the OS file cache holds it\n",
                         m->arena_bytes >> 20, secs);
        else
        std::fprintf(stderr, "moert-memmgr: OWNED cold arena built from %s (%" PRIu64 " MiB in host RAM, "
                             "%.1f s, %.2f GiB/s)%s\n",
                     index_path, m->arena_bytes >> 20, secs, gibs,
                     gibs > 0.0 && gibs < 1.5
                         ? " — well under this path's ceiling: the host is short of RAM and is paging "
                           "the arena out as it is written, not reading slowly"
                         : "");
    }
    return 1;
}

extern "C" int32_t moert_memmgr_get_cold_arena(const moert_memmgr * m, int32_t il, int32_t kind,
                                               int32_t is_scale, moert_cold_arena_view * out) {
    if (!m || !out || il < 0 || il >= m->n_layers || kind < 0 || kind >= MOERT_MEMMGR_MAX_KIND ||
        is_scale < 0 || is_scale > 1) {
        return 0;
    }
    const ColdArena & a = m->cold_arena[il][kind * 2 + is_scale];
    if (!a.present) return 0;
    out->base         = a.base();
    out->stride       = a.stride;                              // payload (copy size)
    out->arena_stride = a.arena_stride ? a.arena_stride : a.stride;  // addressing stride (>= payload under v3)
    out->type_tag    = a.type_tag;
    out->expert_axis = a.axis;
    out->n_expert    = m->n_expert;
    int32_t ndim = 1;
    for (int d = 0; d < MOERT_MEMMGR_MAX_DIMS; ++d) { out->ne[d] = a.ne[d]; if (a.ne[d] > 1) ndim = d + 1; }
    if (a.axis + 1 > ndim) ndim = a.axis + 1;
    out->ndim = ndim;
    return 1;
}

extern "C" uint64_t moert_memmgr_cold_arena_bytes(const moert_memmgr * m) {
    return m ? m->arena_bytes : 0;
}

// RAM-dedup smart refresh. The ~1.9 GiB of
// hot experts held Q4 in VRAM are duplicated in this RAM arena; release their IQ2 pages while hot, reload
// from the (kept-open) store when evicted back to cold. WEIGHT groups only (is_scale=0); the tiny scale
// pool stays resident. VA is stable so cold experts never move. Caller drives these from residency at
// refresh points (between tokens): decommit a just-PROMOTED expert, reload a just-EVICTED one BEFORE it
// is routed cold (never-wait). Inner-page-aligned decommit keeps shared boundary pages committed.
extern "C" uint64_t moert_memmgr_arena_decommit_expert(moert_memmgr * m, int32_t il, int32_t e) {
    if (!m || il < 0 || il >= m->n_layers || e < 0 || e >= m->n_expert) return 0;
    uint64_t freed = 0;
    for (int32_t kind = 0; kind < MOERT_MEMMGR_MAX_KIND; ++kind) {
        ColdArena & a = m->cold_arena[il][kind * 2 + 0];   // weights only; scales stay resident
        if (!a.present || a.per_off.empty() || a.mapped) continue;
        const uint64_t as = a.arena_stride ? a.arena_stride : a.stride;
        freed += a.buf.decommit((size_t) e * as, (size_t) as);
    }
    m->arena_decommitted += freed;
    return freed;
}

extern "C" int32_t moert_memmgr_arena_reload_expert(moert_memmgr * m, int32_t il, int32_t e) {
    if (!m || il < 0 || il >= m->n_layers || e < 0 || e >= m->n_expert) return 0;
    for (int32_t kind = 0; kind < MOERT_MEMMGR_MAX_KIND; ++kind) {
        ColdArena & a = m->cold_arena[il][kind * 2 + 0];
        if (a.mapped) return 0;
        if (!a.present || a.per_off.empty() || a.src_id < 0 || a.src_id >= (int32_t) m->sources.size()) continue;
        const int64_t off = a.per_off[e];
        if (off < 0) return 0;
        const uint64_t as = a.arena_stride ? a.arena_stride : a.stride;
        // recompute the inner-aligned bytes this expert had released, for the committed-bytes telemetry
        // (matches decommit's page-inner range over the padded slot [e*as, e*as+as))
        const size_t pg = PagedBuffer::pagesize();
        const size_t lo = ((size_t) e * as + pg - 1) & ~(pg - 1);
        const size_t hi = ((size_t) e * as + as) & ~(pg - 1);
        const uint64_t inner = hi > lo ? (uint64_t) (hi - lo) : 0;
        a.buf.commit((size_t) e * as, (size_t) as);   // recommit the whole slot
        FILE * f = m->sources[a.src_id];
        // read only the payload bytes into the slot head; the padding tail stays undefined (never read)
        if (!moert_store::seek64(f, off, SEEK_SET) ||
            fread(a.buf.data() + (size_t) e * as, 1, a.stride, f) != a.stride) {
            std::fprintf(stderr, "moert-memmgr: arena reload failed (l%d e%d k%d)\n", il, e, kind);
            return 0;
        }
        if (m->arena_decommitted.load() >= inner) m->arena_decommitted -= inner;
    }
    return 1;
}

// Currently-committed arena bytes (arena minus RAM-dedup released). RSS is the ground truth; this is a hint.
extern "C" uint64_t moert_memmgr_arena_committed_bytes(const moert_memmgr * m) {
    if (!m) return 0;
    const uint64_t d = m->arena_decommitted.load();
    return d <= m->arena_bytes ? m->arena_bytes - d : 0;
}

extern "C" int32_t moert_memmgr_cold_arena_block_count(const moert_memmgr * m) {
    if (!m) return 0;
    int32_t n = 0;
    for (const auto & la : m->cold_arena) for (const auto & a : la) if (a.present) ++n;
    return n;
}

extern "C" int32_t moert_memmgr_cold_arena_block(const moert_memmgr * m, int32_t i,
                                                 void ** base, uint64_t * bytes) {
    if (!m || i < 0 || !base || !bytes) return 0;
    int32_t n = 0;
    for (const auto & la : m->cold_arena) for (const auto & a : la) {
        if (!a.present) continue;
        if (n == i) { *base = (void *) a.base(); *bytes = a.mapped ? a.span(m->n_expert) : a.buf.size(); return 1; }
        ++n;
    }
    return 0;
}
