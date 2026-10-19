// ===========================================================================
// moert/memmgr.h — the expert memory manager.
// ===========================================================================
// `moert` owns the hot-slab LAYOUT + lifetime and the SSD promotion
// source (page catalog, file handles, reads), replacing moe_dual.cpp's ad hoc
// per-tensor ggml allocation + raw Q4-GGUF file handles. Tier ownership:
//   hot  Q4 VRAM slab  — OWNED (layout + promotion writes; device memory reached
//                        only through the moert_memmgr_device callbacks below)
//   cold IQ2 RAM tier  — BORROWED (read-only base+stride views registered by the
//                        host once per load; llama.cpp's loader stays the owner)
//   SSD  Q4 pages      — OWNED (experts.pack/index store, or a legacy catalog the
//                        host fills from GGUF metadata as a compat shim)
//
// HARD RULE: no ggml and no CUDA headers —
// standard library only, CPU-unit-testable (device callbacks can be malloc).
// Single-threaded, like moert/policy.h.
//
// Page addressing convention shared with the host: a page is identified by
// (layer, kind, is_scale) × expert. `kind` = 0 gate_up / 1 down / 2 up / 3 gate
// (gemma4 uses the merged 0 and 1(+scale)); `type_tag` values are ggml_type
// enum values, opaque to this library.
#ifndef MOERT_MEMMGR_H
#define MOERT_MEMMGR_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct moert_memmgr moert_memmgr;

#define MOERT_MEMMGR_MAX_KIND 4
#define MOERT_MEMMGR_MAX_DIMS 4

// Device services the memmgr drives (the host implements them with ggml/CUDA;
// unit tests with malloc). slab_alloc is called exactly once, from finalize().
// Callbacks must NOT call mutating memmgr API.
typedef struct moert_memmgr_device {
    // Allocate THE hot slab (one contiguous device allocation, zero-initialized so
    // never-promoted slots hold defined bytes). Nonzero = ok.
    int32_t (*slab_alloc)(void * user, uint64_t nbytes);
    // Copy `nbytes` from host memory `src` into the slab at byte offset `slab_off`.
    // Nonzero = ok. Synchronous (the caller may reuse `src` on return).
    int32_t (*slab_upload)(void * user, uint64_t slab_off, const void * src, uint64_t nbytes);
    // Release the slab (paired with a successful slab_alloc; called from free()).
    void (*slab_free)(void * user);
    void * user;
    // OPTIONAL (may be NULL): copy `nbytes` from the slab at byte offset `slab_off` into host memory
    // `dst`. Synchronous. Needed only by moert_memmgr_arena_restore (the exclusive arena, below).
    int32_t (*slab_download)(void * user, uint64_t slab_off, void * dst, uint64_t nbytes);
} moert_memmgr_device;

// Lifecycle. Geometry fixed at creation; NULL on invalid geometry/callbacks.
moert_memmgr * moert_memmgr_create(int32_t n_layers, int32_t n_expert, int32_t k,
                                   const moert_memmgr_device * dev);
void moert_memmgr_free(moert_memmgr * m);

// --- page catalog (before finalize) --------------------------------------------------
// Legacy compat shim: the host parses a GGUF's metadata and registers each stacked
// expert tensor as a uniform-stride page group: expert e's page = off_expert0 +
// e*bytes_per_expert in source `source_id`. `stacked_ne` (length MOERT_MEMMGR_MAX_DIMS,
// trailing 1s) is the FULL stacked tensor shape; `expert_axis` is the dim holding
// n_expert. 1 = ok, 0 = invalid/duplicate.
int32_t moert_memmgr_add_source(moert_memmgr * m, const char * path);  // >=0 source id, -1 fail
int32_t moert_memmgr_set_page(moert_memmgr * m, int32_t il, int32_t kind, int32_t is_scale,
                              int32_t source_id, int64_t off_expert0, uint64_t bytes_per_expert,
                              int32_t type_tag, const int64_t * stacked_ne, int32_t expert_axis);

// Expert-store path: read an experts.index, open its sibling pack as a source, and
// fill the catalog from the page records (per-page offsets — no stride assumption).
// Validates: header magic/version, geometry == this memmgr's, per-(layer,kind,is_scale)
// group uniformity (all n_expert pages present, equal nbytes/type/shape), pack uuid.
// `expect_source_uuid` != 0 additionally requires the header's source_uuid to match
// (host computes it from the GGUF it resolved the store from). 1 = ok, 0 = fail (stderr).
int32_t moert_memmgr_open_store(moert_memmgr * m, const char * index_path,
                                uint64_t expect_source_uuid);

// Static layerwise capacity (opt-in, pre-finalize). Override the uniform `k` from create() for
// layer `il` with `k_l` in [1, n_expert]; K stays FIXED for the whole run (not dynamic). Layers
// never set keep the uniform k, so a run that calls this for no layer — or with k_l == k for every
// layer — produces a byte-identical slab layout to the pure-uniform path. 1 = ok, 0 = invalid/finalized.
int32_t moert_memmgr_set_layer_k(moert_memmgr * m, int32_t il, int32_t k_l);
// Capacity of layer il (the uniform k unless overridden above). -1 on invalid handle/layer.
int32_t moert_memmgr_layer_k(const moert_memmgr * m, int32_t il);

// Device-allocation size override for one page group's region (pre-finalize, optional).
// Device backends may need MORE than the K*bytes_per_expert payload for a region (e.g.
// ggml-cuda pads quantized tensor allocations so block loads past the last row are safe);
// the host measures the true requirement (ggml_backend_buft_get_alloc_size on the pool
// tensor it will place there) and reports it here. finalize() sizes the region to
// max(K*bytes_per_expert, alloc_bytes) — so padded reads never land in the next region.
int32_t moert_memmgr_set_region_alloc(moert_memmgr * m, int32_t il, int32_t kind,
                                      int32_t is_scale, uint64_t alloc_bytes);

// --- layout + slab --------------------------------------------------------------------
// Compute the slab layout (one region per registered page group, region starts 1 KiB-
// aligned, region span = max(K*bytes_per_expert, region_alloc), slot stride =
// bytes_per_expert) and call slab_alloc. 1 = ok.
int32_t moert_memmgr_finalize(moert_memmgr * m);
uint64_t moert_memmgr_slab_bytes(const moert_memmgr * m);   // 0 before finalize

// Loaned slots (claude dump findings 246; pre-finalize, optional): the last n_slots of every layer's capacity
// (k_layer) are NOT in the slab for the WEIGHT groups -- they live in a device buffer the host lends (the
// prefill staging reserve while no prompt streams). Their offsets are virtual, at and past slab_bytes, in one
// run of n_slots * bytes_per_expert per weight group; the host's slab_upload/slab_download must map
// [slab_bytes, slab_bytes + loan_bytes) into the lent buffer. Scale groups keep all k_layer slots in the slab.
// page_info's pool_ne counts only the slots in the slab. n_slots = 0 (default) is the plain layout. 1 = ok.
int32_t  moert_memmgr_set_loan(moert_memmgr * m, int32_t n_slots);
int32_t  moert_memmgr_loan(const moert_memmgr * m);
uint64_t moert_memmgr_loan_bytes(const moert_memmgr * m);   // bytes of the loaned runs (0 = none / before finalize)
// Device offset of (il, kind, is_scale, slot): in the slab, or virtual for a loaned slot. 1 = ok.
int32_t  moert_memmgr_slot_offset(const moert_memmgr * m, int32_t il, int32_t kind, int32_t is_scale,
                                  int32_t slot, uint64_t * off);

// Variable-KV: re-layout the hot slab at a NEW K WITHOUT
// disturbing the page catalog, sources, cold views, or the (K-independent, ~GB-scale) owned cold
// arena. Releases the current slab via the device slab_free callback, then clears the finalize
// outputs (per-region offsets/alloc overrides, slab_bytes) and the `finalized` flag — so the host
// can call moert_memmgr_set_layer_k / moert_memmgr_set_region_alloc again with the new K and
// re-finalize (which re-allocs the slab at the new size). This exists so a runtime K change costs
// only the hot-slab rebuild, never a cold-arena reload. Requires a finalized memmgr. 1 = ok, 0 =
// not finalized / invalid handle. CALLER CONTRACT: reset any device tensor views into the old slab
// BEFORE calling this — the slab memory is released here.
int32_t moert_memmgr_reset_hot_layout(moert_memmgr * m);

typedef struct moert_memmgr_page {
    int32_t  type_tag;                        // ggml_type value of the pages
    int32_t  ndim;                            // dims of the pool tensor
    int64_t  pool_ne[MOERT_MEMMGR_MAX_DIMS];  // stacked shape with the expert axis narrowed to K
    int32_t  expert_axis;
    uint64_t bytes_per_expert;                // == slot stride inside the region
    uint64_t region_off;                      // slab byte offset of slot 0
} moert_memmgr_page;
// 1 = page group exists (out filled), 0 = absent. Valid as soon as the group is in the
// catalog (the host uses it to build its typed device views BEFORE finalize);
// `region_off` is meaningful only after finalize.
int32_t moert_memmgr_page_info(const moert_memmgr * m, int32_t il, int32_t kind,
                               int32_t is_scale, moert_memmgr_page * out);

// --- promotion (the SSD -> hot path) --------------------------------------------------
// Stream every page group's expert `e` (all kinds, weight + scale) into hot slot
// (il, slot): read from the owning source, upload at region_off + slot*stride.
// Absent groups are skipped (legacy stream_expert semantics). 1 = ok, 0 = a read or
// upload failed (the policy then leaves the slot FREE; partial bytes are never
// published as READY).
int32_t moert_memmgr_promote(moert_memmgr * m, int32_t il, int32_t e, int32_t slot);

// Promotion from the OWNED COLD ARENA instead of the promotion source (full-IQ2 mode, 2026-10-03):
// when the hot and cold tiers hold the same quantization, expert e's hot page IS its arena page, so it
// is uploaded straight from host RAM -- no file read. Every present hot group of layer il must have an
// arena group of the same (kind, is_scale) with the same type and payload size, else 0 is returned and
// nothing is uploaded. Read-only on memmgr state (thread-safe against other readers; the slab_upload
// callback decides its own stream). The arena pages of e must be committed (no RAM-dedup decommit).
int32_t moert_memmgr_promote_from_arena(moert_memmgr * m, int32_t il, int32_t e, int32_t slot);
// 1 iff every present hot group (all layers) has a type- and size-matching arena group, i.e. iff
// promote_from_arena can serve every promotion. Call once after finalize + build_cold_arena.
int32_t moert_memmgr_arena_matches_hot(const moert_memmgr * m);

// EXCLUSIVE ARENA (full-IQ2 plan, phase 2): while expert e is hot, its slot holds the same bytes as its
// arena page, so the page can be released and later rebuilt FROM THE SLOT (no file read).
//   arena_release: decommit the WEIGHT pages of e in every arena group that matches its hot group
//     (the same groups promote_from_arena serves; scale groups stay resident). Returns bytes released.
//   arena_restore: recommit those pages and download the slot (il, slot) into them through the
//     device's slab_download. The slot MUST still hold e's bytes. 1 = ok, 0 = no slab_download /
//     invalid arguments / a group mismatch / a commit or download failure.
// Both are safe to call from a worker thread as long as nothing reads e's arena page meanwhile
// (the host guarantees that); the arena must not be page-locked (no pin hooks are called).
// MAP MODE (moert_memmgr_cold_arena_mapped): release only drops e's pages from the working set and
// restore faults them back in (slot ignored). The data is never lost, so a reader never has to wait and
// the hot tier's format does not matter (quality's Q4 slab included).
uint64_t moert_memmgr_arena_release(moert_memmgr * m, int32_t il, int32_t e);
int32_t  moert_memmgr_arena_restore(moert_memmgr * m, int32_t il, int32_t e, int32_t slot);
// 1 iff the cold arena is the read-only mapping of the pack (findings 197), 0 if copied or absent.
int32_t  moert_memmgr_cold_arena_mapped(const moert_memmgr * m);

// --- cold-tier borrow views (registered once per load; consumers arrive in M3) --------
typedef struct moert_cold_view {
    const void * base;      // expert 0 of the stacked cold tensor (host RAM)
    uint64_t     stride;    // bytes per expert
    int32_t      type_tag;  // ggml_type value
    int64_t      n_expert;  // experts covered (== geometry n_expert)
} moert_cold_view;
int32_t moert_memmgr_set_cold_view(moert_memmgr * m, int32_t il, int32_t kind, int32_t is_scale,
                                   const moert_cold_view * v);   // 1 ok, 0 invalid
int32_t moert_memmgr_get_cold_view(const moert_memmgr * m, int32_t il, int32_t kind,
                                   int32_t is_scale, moert_cold_view * out);  // 1 present
int32_t moert_memmgr_cold_view_count(const moert_memmgr * m);

// --- OWNED cold arena (the 3b migration: moert OWNS the cold IQ2 tier in host RAM instead of
// borrowing llama.cpp's expert tensors). Reconstruct the per-(layer, kind, is_scale) cold tensors
// CONTIGUOUSLY (expert 0..n_expert-1 at stride) in memmgr-owned RAM from an IQ2 expert store, so the
// fused op can source cold from moert memory and llama need not allocate the ~8 GiB of *_exps. This
// is the read-only full-pool arena ("own the serve"); freeable/compacting residency layers on top.
// Reuses the store's page geometry + checksums. 1 = ok, 0 = open/geometry/read failure (stderr).
int32_t moert_memmgr_build_cold_arena(moert_memmgr * m, const char * index_path,
                                      uint64_t expect_source_uuid);
// 1 = the store's pack is group-major (every weight group one dense run: build_cold_arena can map it),
// 0 = the old expert-major layout (copied into RAM; `prepare` rebuilds it), -1 = unreadable store.
int32_t moert_memmgr_store_group_major(const char * index_path);

typedef struct moert_cold_arena_view {
    const void * base;                        // expert 0 of the owned stacked cold tensor (host RAM)
    uint64_t     stride;                      // PAYLOAD bytes per expert (the real quantized size; == copy size)
    uint64_t     arena_stride;                // ADDRESSING stride: expert e is at base + e*arena_stride (== stride
                                              // today; consumers address with arena_stride, copy stride).
    int32_t      type_tag;                    // ggml_type value of the pages
    int32_t      ndim;                        // dims of the stacked tensor
    int64_t      ne[MOERT_MEMMGR_MAX_DIMS];   // stacked shape (expert axis = n_expert)
    int32_t      expert_axis;
    int64_t      n_expert;
} moert_cold_arena_view;
// Owned cold tensor for (layer, kind, is_scale): base + e*stride is expert e. 1 = present.
int32_t moert_memmgr_get_cold_arena(const moert_memmgr * m, int32_t il, int32_t kind, int32_t is_scale,
                                    moert_cold_arena_view * out);
uint64_t moert_memmgr_cold_arena_bytes(const moert_memmgr * m);   // total owned cold RAM (0 = none)

// Release/reload a WEIGHT expert's IQ2 pages in the owned arena from its source file (VA stays, cold
// experts never move). reload is the exclusive arena's last resort when a restore from the slot fails.
// committed_bytes = arena minus released (telemetry; process RSS is truth).
uint64_t moert_memmgr_arena_decommit_expert(moert_memmgr * m, int32_t il, int32_t e);   // returns bytes freed
int32_t  moert_memmgr_arena_reload_expert(moert_memmgr * m, int32_t il, int32_t e);      // 1 ok, 0 fail
uint64_t moert_memmgr_arena_committed_bytes(const moert_memmgr * m);

// Enumerate the owned cold-arena RAM blocks (one per present (layer,kind,is_scale) group) so the host
// can page-lock the whole arena in one batch right after build (cudaHostRegister lives in the host —
// no CUDA in this lib). Order is stable for a given build.
int32_t moert_memmgr_cold_arena_block_count(const moert_memmgr * m);   // 0 = none
// The i-th present block: *base = host RAM start, *bytes = its size. 1 = ok, 0 = out of range.
int32_t moert_memmgr_cold_arena_block(const moert_memmgr * m, int32_t i, void ** base, uint64_t * bytes);


#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // MOERT_MEMMGR_H
