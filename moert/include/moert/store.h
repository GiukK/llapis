// ===========================================================================
// moert/store.h — the expert-store v0 side format (experts.pack + experts.index).
// ===========================================================================
// the lateral per-expert page store designed in,
// finally built. GGUF stays the model format; the pack/index exist only to hold
// per-(layer, expert, matrix, precision) pages the runtime promotes from SSD.
//
// HARD RULE: no ggml and no CUDA headers. The
// `quant_type` tags reuse ggml_type enum VALUES but are opaque int32 here — the
// host interprets them.
//
// v0 format notes (deviations from the design doc, both recorded in the header):
//  - checksum is FNV-1a 64 (checksum_algo = 1), not xxh3-64 — integrity only, no
//    dedup in v0, so a small dependency-free hash wins.
//  - the compat key binds the index to the SOURCE gguf the pages were sliced from
//    (field `source_uuid`), not to the runtime base model: the same Q4 promotion
//    source is legitimately paired with different bases (native IQ2, ne-Q4 splice).
#ifndef MOERT_STORE_H
#define MOERT_STORE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Conventional sibling names: "<source.gguf>" + suffix (what the runtime auto-detects).
#define MOERT_STORE_INDEX_SUFFIX ".experts.index"
#define MOERT_STORE_PACK_SUFFIX  ".experts.pack"

// Page record flags.
enum moert_page_flags {
    MOERT_PAGE_REQUANTIZED = 1 << 0,  // quant_type != source_quant lineage (1.5c-style requant)
    MOERT_PAGE_IMATRIX     = 1 << 1,  // imatrix-calibrated source
    MOERT_PAGE_PADDED      = 1 << 2,  // page bytes padded beyond the block payload
    MOERT_PAGE_SCALE       = 1 << 3,  // page is a per-expert SCALE (e.g. ffn_down_exps.scale slice)
};

// Index header facts (probe result / writer input).
typedef struct moert_store_info {
    uint32_t format_version;   // writer: ignored (always current); probe: as read
    uint64_t source_uuid;      // identity hash of the source GGUF's expert-tensor metadata
    char     arch[16];         // architecture string, NUL-terminated/padded ("gemma4", ...)
    int32_t  n_layers;
    int32_t  n_expert;
    uint64_t n_embd;           // 0 = unknown (sanity metadata only)
    uint64_t n_ff_exp;         // 0 = unknown
    uint64_t pack_uuid;        // links index <-> pack; writer: 0 = generate
    uint64_t record_count;     // probe only (writer counts internally)
} moert_store_info;

// Read an experts.index header. 1 = ok, 0 = missing/invalid (reason on stderr).
int32_t moert_store_probe(const char * index_path, moert_store_info * out);

// Re-read every page from the sibling pack and verify its checksum (and the pack
// header's uuid). Returns the number of bad pages (0 = clean), -1 on I/O/format
// error. Progress + per-page failures go to stderr.
int64_t moert_store_verify(const char * index_path);

// --- writer (offline build tool; the host feeds raw page bytes) --------------------
typedef struct moert_store_writer moert_store_writer;

// Create both files (truncating). `info` supplies source_uuid/arch/geometry;
// pack_uuid 0 = generated. NULL on failure (reason on stderr).
moert_store_writer * moert_store_writer_open(const char * index_path, const char * pack_path,
                                             const moert_store_info * info);

// Append one page. `kind` is the host's matrix-kind enum (0 gate_up / 1 down / 2 up /
// 3 gate — gemma4 uses 0,1); `is_scale` sets MOERT_PAGE_SCALE. `shape` = per-page dims
// (per-expert matrix dims for a weight, {1} for a scale), ndim <= 4. `quant_type` /
// `source_quant` are ggml_type values (opaque here). 1 = ok, 0 = write error.
int32_t moert_store_writer_add_page(moert_store_writer * w,
                                    int32_t layer, int32_t expert_id, int32_t kind, int32_t is_scale,
                                    int32_t quant_type, int32_t source_quant, uint32_t extra_flags,
                                    const int64_t * shape, int32_t ndim,
                                    const void * bytes, uint64_t nbytes);

// Flush records + header, fsync-equivalent checked close of both files, free the
// writer. 1 = ok, 0 = a write/flush failed (files are unusable). Always frees `w`.
int32_t moert_store_writer_close(moert_store_writer * w);

// FNV-1a 64 over a byte range (checksum_algo 1) — exposed so build tools can hash
// source metadata into `source_uuid` with the same function family.
uint64_t moert_store_fnv1a64(const void * data, uint64_t nbytes, uint64_t seed);
#define MOERT_STORE_FNV_SEED 14695981039346656037ull

// --- reader (per-expert page reads into HOST RAM) ------------------------------------
// The read side the OWNED cold-page path needs: unlike memmgr open_store (which streams
// pages into the device slab), this reads a single (layer, kind, is_scale, expert) page
// into host memory — the source moert/pages.h EXCLUSIVE fills owned IQ2 pages from. The
// pack stays open for the reader's lifetime. Single-threaded.
typedef struct moert_store_reader moert_store_reader;

// Open a store for host-side page reads. `expect_source_uuid` != 0 also requires the
// header's source_uuid to match (bind to the gguf the pages were sliced from). NULL on
// failure (reason on stderr).
moert_store_reader * moert_store_reader_open(const char * index_path, uint64_t expect_source_uuid);
void moert_store_reader_free(moert_store_reader * r);

// Geometry read from the header (either out may be NULL). 1 = ok.
int32_t moert_store_reader_geometry(const moert_store_reader * r, int32_t * n_layers, int32_t * n_expert);

// Byte length of page (layer, kind, is_scale, expert), or 0 if that page is absent.
uint64_t moert_store_reader_page_nbytes(const moert_store_reader * r,
                                        int32_t layer, int32_t kind, int32_t is_scale, int32_t expert);

// Page metadata for (layer, kind, is_scale, expert): quant_type (ggml_type value, opaque
// here), nbytes and shape (trailing 1s). Any out may be NULL. 1 = page present, 0 = absent.
int32_t moert_store_reader_page_info(const moert_store_reader * r,
                                     int32_t layer, int32_t kind, int32_t is_scale, int32_t expert,
                                     int32_t * quant_type, uint64_t * nbytes, int64_t shape[4]);

// Read page (layer, kind, is_scale, expert) into `dst` (capacity `dst_cap`). If `verify`
// is nonzero the page checksum is re-checked. Returns bytes read (0 = absent / too small /
// I/O error / checksum mismatch; reason on stderr for real errors).
uint64_t moert_store_reader_read(moert_store_reader * r,
                                 int32_t layer, int32_t kind, int32_t is_scale, int32_t expert,
                                 void * dst, uint64_t dst_cap, int32_t verify);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // MOERT_STORE_H
