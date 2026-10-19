// Private on-disk layout of expert-store v0 (shared by the writer, the verifier and
// the memmgr reader — the ONE place that knows the byte layout). See moert/store.h
// for the public ABI and for the design.
//
// All integers little-endian; structs are packed and written/read as raw bytes
// (this project only targets little-endian hosts — x86_64).
#ifndef MOERT_STORE_FORMAT_H
#define MOERT_STORE_FORMAT_H

#include <stdint.h>
#include <stdio.h>

namespace moert_store {

constexpr char     INDEX_MAGIC[8]  = { 'E','X','I','D','X','1','\0','\0' };
constexpr char     PACK_MAGIC[8]   = { 'E','X','P','A','C','K','1','\0' };
constexpr uint32_t FORMAT_VERSION  = 1;
constexpr uint32_t CHECKSUM_FNV1A64 = 1;   // the only algo in v0
constexpr uint64_t PAGE_ALIGN      = 256;  // pack page start alignment (mmap/DMA-friendly)

#pragma pack(push, 1)
struct IndexHeader {                 // 96 bytes
    char     magic[8];
    uint32_t format_version;
    uint32_t header_bytes;           // offset of the first record (= sizeof(IndexHeader))
    uint64_t source_uuid;            // identity of the source GGUF (moert_store_fnv1a64 family)
    char     arch[16];               // NUL-padded architecture string
    uint32_t n_layers;
    uint32_t n_expert;
    uint64_t n_embd;                 // 0 = unknown
    uint64_t n_ff_exp;               // 0 = unknown
    uint64_t pack_uuid;              // must match the pack header
    uint64_t record_count;
    uint32_t record_stride;          // = sizeof(IndexRecord); readers skip unknown tails
    uint32_t checksum_algo;          // CHECKSUM_FNV1A64
    uint64_t reserved;
};
static_assert(sizeof(IndexHeader) == 96, "index header layout");

struct IndexRecord {                 // 72 bytes
    uint16_t layer;
    uint16_t expert_id;
    uint8_t  matrix_kind;            // host kind enum (0 gate_up / 1 down / 2 up / 3 gate)
    uint8_t  reserved8;
    uint16_t quant_type;             // ggml_type value (storage)
    uint64_t offset;                 // absolute byte offset of the page in experts.pack
    uint64_t nbytes;                 // page length
    int64_t  shape[4];               // per-page dims, trailing 1s (weight: per-expert matrix; scale: {1,1,1,1})
    uint64_t checksum;               // FNV-1a 64 of the page bytes
    uint16_t source_quant;           // ggml_type value (provenance)
    uint16_t reserved16;
    uint32_t flags;                  // moert_page_flags
};
static_assert(sizeof(IndexRecord) == 72, "index record layout");

struct PackHeader {                  // 32 bytes
    char     magic[8];
    uint32_t format_version;
    uint32_t reserved32;
    uint64_t pack_uuid;
    uint64_t reserved64;
};
static_assert(sizeof(PackHeader) == 32, "pack header layout");
#pragma pack(pop)

// 64-bit FILE* seek that works on Win32 and POSIX (moert must not depend on the
// harness's io_util). Returns true on success.
bool seek64(FILE * f, int64_t off, int origin);

}  // namespace moert_store

#ifdef __cplusplus
#include <string>
#include <vector>

// Open + validate an index and its sibling pack, read all records. On success (1)
// the pack FILE* (positioned arbitrarily) is handed to the caller. Shared by
// moert_store_verify and moert_memmgr_open_store. Format errors -> 0 + stderr.
int32_t moert_store_load(const char * index_path, moert_store::IndexHeader & hdr,
                         std::vector<moert_store::IndexRecord> & records, FILE ** pack_f,
                         std::string * pack_path_out);
#endif

#endif  // MOERT_STORE_FORMAT_H
