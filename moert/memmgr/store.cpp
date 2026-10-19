// moert/store.cpp — expert-store v0 writer, prober and verifier (moert/store.h).
// No ggml / no CUDA: raw bytes in, raw bytes out; the host interprets
// quant_type tags. Reader-side catalog loading lives in memmgr.cpp (open_store).

#include "moert/store.h"
#include "store_format.h"

#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>

namespace moert_store {

bool seek64(FILE * f, int64_t off, int origin) {
#if defined(_WIN32)
    return _fseeki64(f, off, origin) == 0;
#else
    return fseeko(f, (off_t) off, origin) == 0;
#endif
}

}  // namespace moert_store

using namespace moert_store;

extern "C" uint64_t moert_store_fnv1a64(const void * data, uint64_t nbytes, uint64_t seed) {
    const uint8_t * p = (const uint8_t *) data;
    uint64_t h = seed;
    for (uint64_t i = 0; i < nbytes; ++i) {
        h ^= p[i];
        h *= 1099511628211ull;
    }
    return h;
}

// --- probe ---------------------------------------------------------------------------

static int32_t read_header(FILE * f, const char * index_path, IndexHeader & h) {
    if (fread(&h, 1, sizeof(h), f) != sizeof(h)) {
        std::fprintf(stderr, "moert-store: %s: truncated index header\n", index_path);
        return 0;
    }
    if (std::memcmp(h.magic, INDEX_MAGIC, sizeof(INDEX_MAGIC)) != 0) {
        std::fprintf(stderr, "moert-store: %s: bad index magic\n", index_path);
        return 0;
    }
    if (h.format_version != FORMAT_VERSION) {
        std::fprintf(stderr, "moert-store: %s: unsupported format version %u (want %u)\n",
                     index_path, h.format_version, FORMAT_VERSION);
        return 0;
    }
    if (h.record_stride < sizeof(IndexRecord) || h.header_bytes < sizeof(IndexHeader)) {
        std::fprintf(stderr, "moert-store: %s: record stride %u / header bytes %u too small\n",
                     index_path, h.record_stride, h.header_bytes);
        return 0;
    }
    if (h.checksum_algo != CHECKSUM_FNV1A64) {
        std::fprintf(stderr, "moert-store: %s: unknown checksum algo %u\n", index_path, h.checksum_algo);
        return 0;
    }
    return 1;
}

extern "C" int32_t moert_store_probe(const char * index_path, moert_store_info * out) {
    if (!index_path || !out) return 0;
    FILE * f = std::fopen(index_path, "rb");
    if (!f) return 0;   // silent: probe is used for sibling auto-detection
    IndexHeader h;
    const int32_t ok = read_header(f, index_path, h);
    std::fclose(f);
    if (!ok) return 0;
    out->format_version = h.format_version;
    out->source_uuid    = h.source_uuid;
    std::memcpy(out->arch, h.arch, sizeof(out->arch));
    out->arch[sizeof(out->arch) - 1] = '\0';
    out->n_layers     = (int32_t) h.n_layers;
    out->n_expert     = (int32_t) h.n_expert;
    out->n_embd       = h.n_embd;
    out->n_ff_exp     = h.n_ff_exp;
    out->pack_uuid    = h.pack_uuid;
    out->record_count = h.record_count;
    return 1;
}

// index_path -> sibling pack path ("<x>.experts.index" -> "<x>.experts.pack"; any other
// name: append MOERT_STORE_PACK_SUFFIX-with-index-stripped fallback = replace extension).
static std::string sibling_pack_path(const std::string & index_path) {
    const std::string isuf = MOERT_STORE_INDEX_SUFFIX;
    if (index_path.size() > isuf.size() &&
        index_path.compare(index_path.size() - isuf.size(), isuf.size(), isuf) == 0) {
        return index_path.substr(0, index_path.size() - isuf.size()) + MOERT_STORE_PACK_SUFFIX;
    }
    return index_path + ".pack";
}

// Shared record-walk used by verify and memmgr open_store: opens+checks both files,
// yields each record. Returns 0 on format error; 1 ok. `pack_f` handed to the caller.
int32_t moert_store_load(const char * index_path, IndexHeader & hdr,
                         std::vector<IndexRecord> & records, FILE ** pack_f,
                         std::string * pack_path_out) {
    FILE * f = std::fopen(index_path, "rb");
    if (!f) {
        std::fprintf(stderr, "moert-store: cannot open index %s\n", index_path);
        return 0;
    }
    if (!read_header(f, index_path, hdr)) { std::fclose(f); return 0; }
    if (!moert_store::seek64(f, (int64_t) hdr.header_bytes, SEEK_SET)) { std::fclose(f); return 0; }
    records.resize(hdr.record_count);
    for (uint64_t i = 0; i < hdr.record_count; ++i) {
        if (fread(&records[i], 1, sizeof(IndexRecord), f) != sizeof(IndexRecord)) {
            std::fprintf(stderr, "moert-store: %s: truncated at record %" PRIu64 "/%" PRIu64 "\n",
                         index_path, i, hdr.record_count);
            std::fclose(f);
            return 0;
        }
        const uint32_t tail = hdr.record_stride - (uint32_t) sizeof(IndexRecord);
        if (tail && !moert_store::seek64(f, tail, SEEK_CUR)) { std::fclose(f); return 0; }
    }
    std::fclose(f);

    const std::string pack_path = sibling_pack_path(index_path);
    FILE * pf = std::fopen(pack_path.c_str(), "rb");
    if (!pf) {
        std::fprintf(stderr, "moert-store: cannot open pack %s\n", pack_path.c_str());
        return 0;
    }
    PackHeader ph;
    if (fread(&ph, 1, sizeof(ph), pf) != sizeof(ph) ||
        std::memcmp(ph.magic, PACK_MAGIC, sizeof(PACK_MAGIC)) != 0 ||
        ph.format_version != FORMAT_VERSION) {
        std::fprintf(stderr, "moert-store: %s: bad pack header\n", pack_path.c_str());
        std::fclose(pf);
        return 0;
    }
    if (ph.pack_uuid != hdr.pack_uuid) {
        std::fprintf(stderr, "moert-store: index/pack uuid mismatch (%016" PRIx64 " vs %016" PRIx64
                             ") — mixed store files?\n", hdr.pack_uuid, ph.pack_uuid);
        std::fclose(pf);
        return 0;
    }
    *pack_f = pf;
    if (pack_path_out) *pack_path_out = pack_path;
    return 1;
}

extern "C" int64_t moert_store_verify(const char * index_path) {
    IndexHeader hdr;
    std::vector<IndexRecord> recs;
    FILE * pf = nullptr;
    if (!moert_store_load(index_path, hdr, recs, &pf, nullptr)) return -1;

    std::vector<uint8_t> buf;
    int64_t bad = 0;
    uint64_t bytes = 0;
    for (size_t i = 0; i < recs.size(); ++i) {
        const IndexRecord & r = recs[i];
        buf.resize(r.nbytes);
        if (!moert_store::seek64(pf, (int64_t) r.offset, SEEK_SET) ||
            fread(buf.data(), 1, r.nbytes, pf) != r.nbytes) {
            std::fprintf(stderr, "moert-store: verify: read failed (record %zu: l%u e%u k%u%s)\n",
                         i, r.layer, r.expert_id, r.matrix_kind,
                         (r.flags & MOERT_PAGE_SCALE) ? " scale" : "");
            ++bad;
            continue;
        }
        if (moert_store_fnv1a64(buf.data(), r.nbytes, MOERT_STORE_FNV_SEED) != r.checksum) {
            std::fprintf(stderr, "moert-store: verify: CHECKSUM MISMATCH (record %zu: l%u e%u k%u%s)\n",
                         i, r.layer, r.expert_id, r.matrix_kind,
                         (r.flags & MOERT_PAGE_SCALE) ? " scale" : "");
            ++bad;
        }
        bytes += r.nbytes;
    }
    std::fclose(pf);
    std::fprintf(stderr, "moert-store: verify: %zu pages, %" PRIu64 " MiB, %" PRId64 " bad\n",
                 recs.size(), bytes >> 20, bad);
    return bad;
}

// --- reader (per-expert host page reads) ---------------------------------------------

namespace { constexpr int RD_MAX_KIND = 4; }   // kinds 0..3 (gemma4 uses 0,1); is_scale 0..1

struct moert_store_reader {
    FILE * pack_f = nullptr;
    IndexHeader hdr{};
    std::vector<IndexRecord> recs;
    int32_t n_layers = 0, n_expert = 0;
    std::vector<int64_t> slot;   // [(layer*RD_MAX_KIND*2 + kind*2 + is_scale)*n_expert + e] -> rec idx | -1
};

static int64_t rd_key(const moert_store_reader * r, int layer, int kind, int is_scale, int expert) {
    if (layer < 0 || layer >= r->n_layers || kind < 0 || kind >= RD_MAX_KIND ||
        is_scale < 0 || is_scale > 1 || expert < 0 || expert >= r->n_expert) {
        return -1;
    }
    return ((int64_t)(layer * RD_MAX_KIND * 2 + kind * 2 + is_scale)) * r->n_expert + expert;
}

extern "C" moert_store_reader * moert_store_reader_open(const char * index_path, uint64_t expect_source_uuid) {
    if (!index_path) return nullptr;
    moert_store_reader * r = new moert_store_reader();
    FILE * pf = nullptr;
    if (!moert_store_load(index_path, r->hdr, r->recs, &pf, nullptr)) { delete r; return nullptr; }
    if (expect_source_uuid && r->hdr.source_uuid != expect_source_uuid) {
        std::fprintf(stderr, "moert-store: reader: source_uuid %016" PRIx64 " != expected %016" PRIx64
                             " (%s)\n", r->hdr.source_uuid, expect_source_uuid, index_path);
        std::fclose(pf); delete r; return nullptr;
    }
    r->pack_f   = pf;
    r->n_layers = (int32_t) r->hdr.n_layers;
    r->n_expert = (int32_t) r->hdr.n_expert;
    r->slot.assign((size_t) r->n_layers * RD_MAX_KIND * 2 * r->n_expert, -1);
    for (size_t i = 0; i < r->recs.size(); ++i) {
        const IndexRecord & rec = r->recs[i];
        const int is_scale = (rec.flags & MOERT_PAGE_SCALE) ? 1 : 0;
        const int64_t k = rd_key(r, rec.layer, rec.matrix_kind, is_scale, rec.expert_id);
        if (k >= 0) r->slot[k] = (int64_t) i;
    }
    return r;
}

extern "C" void moert_store_reader_free(moert_store_reader * r) {
    if (!r) return;
    if (r->pack_f) std::fclose(r->pack_f);
    delete r;
}

extern "C" int32_t moert_store_reader_geometry(const moert_store_reader * r, int32_t * n_layers, int32_t * n_expert) {
    if (!r) return 0;
    if (n_layers) *n_layers = r->n_layers;
    if (n_expert) *n_expert = r->n_expert;
    return 1;
}

extern "C" uint64_t moert_store_reader_page_nbytes(const moert_store_reader * r,
                                                   int32_t layer, int32_t kind, int32_t is_scale, int32_t expert) {
    if (!r) return 0;
    const int64_t k = rd_key(r, layer, kind, is_scale, expert);
    if (k < 0 || r->slot[k] < 0) return 0;
    return r->recs[r->slot[k]].nbytes;
}

extern "C" int32_t moert_store_reader_page_info(const moert_store_reader * r,
                                                int32_t layer, int32_t kind, int32_t is_scale, int32_t expert,
                                                int32_t * quant_type, uint64_t * nbytes, int64_t shape[4]) {
    if (!r) return 0;
    const int64_t k = rd_key(r, layer, kind, is_scale, expert);
    if (k < 0 || r->slot[k] < 0) return 0;
    const IndexRecord & rec = r->recs[r->slot[k]];
    if (quant_type) *quant_type = (int32_t) rec.quant_type;
    if (nbytes)     *nbytes     = rec.nbytes;
    if (shape)      for (int i = 0; i < 4; i++) shape[i] = rec.shape[i];
    return 1;
}

extern "C" uint64_t moert_store_reader_read(moert_store_reader * r,
                                            int32_t layer, int32_t kind, int32_t is_scale, int32_t expert,
                                            void * dst, uint64_t dst_cap, int32_t verify) {
    if (!r || !dst) return 0;
    const int64_t k = rd_key(r, layer, kind, is_scale, expert);
    if (k < 0 || r->slot[k] < 0) return 0;
    const IndexRecord & rec = r->recs[r->slot[k]];
    if (rec.nbytes > dst_cap) {
        std::fprintf(stderr, "moert-store: reader: dst too small (%" PRIu64 " < page %" PRIu64
                             " l%u e%u k%u%s)\n", dst_cap, rec.nbytes, rec.layer, rec.expert_id,
                             rec.matrix_kind, is_scale ? " scale" : "");
        return 0;
    }
    if (!moert_store::seek64(r->pack_f, (int64_t) rec.offset, SEEK_SET) ||
        fread(dst, 1, rec.nbytes, r->pack_f) != rec.nbytes) {
        std::fprintf(stderr, "moert-store: reader: read failed (l%u e%u k%u%s)\n",
                     rec.layer, rec.expert_id, rec.matrix_kind, is_scale ? " scale" : "");
        return 0;
    }
    if (verify && moert_store_fnv1a64(dst, rec.nbytes, MOERT_STORE_FNV_SEED) != rec.checksum) {
        std::fprintf(stderr, "moert-store: reader: CHECKSUM MISMATCH (l%u e%u k%u%s)\n",
                     rec.layer, rec.expert_id, rec.matrix_kind, is_scale ? " scale" : "");
        return 0;
    }
    return rec.nbytes;
}

// --- writer --------------------------------------------------------------------------

struct moert_store_writer {
    std::string index_path, pack_path;
    FILE * pack_f = nullptr;
    IndexHeader hdr{};
    std::vector<IndexRecord> records;
    uint64_t pack_pos = 0;   // next unaligned write position
    bool failed = false;
};

extern "C" moert_store_writer * moert_store_writer_open(const char * index_path, const char * pack_path,
                                                        const moert_store_info * info) {
    if (!index_path || !pack_path || !info || info->n_layers <= 0 || info->n_expert <= 0) return nullptr;
    moert_store_writer * w = new moert_store_writer();
    w->index_path = index_path;
    w->pack_path  = pack_path;
    w->pack_f = std::fopen(pack_path, "wb");
    if (!w->pack_f) {
        std::fprintf(stderr, "moert-store: cannot create pack %s\n", pack_path);
        delete w;
        return nullptr;
    }
    std::memcpy(w->hdr.magic, INDEX_MAGIC, sizeof(INDEX_MAGIC));
    w->hdr.format_version = FORMAT_VERSION;
    w->hdr.header_bytes   = sizeof(IndexHeader);
    w->hdr.source_uuid    = info->source_uuid;
    std::memset(w->hdr.arch, 0, sizeof(w->hdr.arch));
    std::strncpy(w->hdr.arch, info->arch, sizeof(w->hdr.arch) - 1);
    w->hdr.n_layers      = (uint32_t) info->n_layers;
    w->hdr.n_expert      = (uint32_t) info->n_expert;
    w->hdr.n_embd        = info->n_embd;
    w->hdr.n_ff_exp      = info->n_ff_exp;
    w->hdr.pack_uuid     = info->pack_uuid ? info->pack_uuid
                         : moert_store_fnv1a64(index_path, (uint64_t) std::strlen(index_path),
                                               MOERT_STORE_FNV_SEED ^ (uint64_t) std::time(nullptr));
    w->hdr.record_stride = sizeof(IndexRecord);
    w->hdr.checksum_algo = CHECKSUM_FNV1A64;

    PackHeader ph{};
    std::memcpy(ph.magic, PACK_MAGIC, sizeof(PACK_MAGIC));
    ph.format_version = FORMAT_VERSION;
    ph.pack_uuid      = w->hdr.pack_uuid;
    if (fwrite(&ph, 1, sizeof(ph), w->pack_f) != sizeof(ph)) {
        std::fprintf(stderr, "moert-store: pack header write failed (%s)\n", pack_path);
        std::fclose(w->pack_f);
        delete w;
        return nullptr;
    }
    w->pack_pos = sizeof(ph);
    return w;
}

extern "C" int32_t moert_store_writer_add_page(moert_store_writer * w,
                                               int32_t layer, int32_t expert_id, int32_t kind, int32_t is_scale,
                                               int32_t quant_type, int32_t source_quant, uint32_t extra_flags,
                                               const int64_t * shape, int32_t ndim,
                                               const void * bytes, uint64_t nbytes) {
    if (!w || w->failed || !bytes || nbytes == 0 || ndim < 1 || ndim > 4 ||
        layer < 0 || layer >= (int32_t) w->hdr.n_layers ||
        expert_id < 0 || expert_id >= (int32_t) w->hdr.n_expert || kind < 0 || kind > 255) {
        if (w) w->failed = true;
        return 0;
    }
    // align the page start
    const uint64_t off = (w->pack_pos + PAGE_ALIGN - 1) & ~(PAGE_ALIGN - 1);
    if (off != w->pack_pos) {
        static const uint8_t zeros[PAGE_ALIGN] = {};
        if (fwrite(zeros, 1, off - w->pack_pos, w->pack_f) != off - w->pack_pos) { w->failed = true; return 0; }
    }
    if (fwrite(bytes, 1, nbytes, w->pack_f) != nbytes) { w->failed = true; return 0; }
    w->pack_pos = off + nbytes;

    IndexRecord r{};
    r.layer        = (uint16_t) layer;
    r.expert_id    = (uint16_t) expert_id;
    r.matrix_kind  = (uint8_t) kind;
    r.quant_type   = (uint16_t) quant_type;
    r.offset       = off;
    r.nbytes       = nbytes;
    for (int d = 0; d < 4; ++d) r.shape[d] = d < ndim ? shape[d] : 1;
    r.checksum     = moert_store_fnv1a64(bytes, nbytes, MOERT_STORE_FNV_SEED);
    r.source_quant = (uint16_t) source_quant;
    r.flags        = extra_flags | (is_scale ? MOERT_PAGE_SCALE : 0) |
                     (quant_type != source_quant ? MOERT_PAGE_REQUANTIZED : 0);
    w->records.push_back(r);
    return 1;
}

extern "C" int32_t moert_store_writer_close(moert_store_writer * w) {
    if (!w) return 0;
    bool ok = !w->failed;
    // checked-flush discipline (gguf_tools): fflush + ferror + fclose all checked.
    if (w->pack_f) {
        if (fflush(w->pack_f) != 0 || ferror(w->pack_f)) ok = false;
        if (std::fclose(w->pack_f) != 0) ok = false;
    }
    if (ok) {
        w->hdr.record_count = w->records.size();
        FILE * f = std::fopen(w->index_path.c_str(), "wb");
        if (!f) {
            std::fprintf(stderr, "moert-store: cannot create index %s\n", w->index_path.c_str());
            ok = false;
        } else {
            if (fwrite(&w->hdr, 1, sizeof(w->hdr), f) != sizeof(w->hdr)) ok = false;
            if (ok && !w->records.empty() &&
                fwrite(w->records.data(), sizeof(IndexRecord), w->records.size(), f) != w->records.size()) {
                ok = false;
            }
            if (fflush(f) != 0 || ferror(f)) ok = false;
            if (std::fclose(f) != 0) ok = false;
        }
    }
    if (!ok) {
        std::fprintf(stderr, "moert-store: writer FAILED — %s / %s are not usable\n",
                     w->index_path.c_str(), w->pack_path.c_str());
    }
    delete w;
    return ok ? 1 : 0;
}
