// moert/registry.h -- where an expert's bytes are, and the cheapest place to get them from.
//
// Pure (no CUDA, no allocation, header-only). Each representation of a (layer, expert) has one owner:
// the residency policy owns the VRAM slots, the transfer engine owns the RAM arena pages (released or
// resident under the exclusive arena), the memory manager owns the files. The host composes a
// moert_expert_where from those owners at decision time and asks which source to copy from. Nothing
// here knows the name of the mode: the regime enters only through `same_bytes` and the file flags
// (claude dump docs/system-model.md §5, docs/design-prefill-transfer-2026-10-04.md §2.1).
//
// Cost order, measured on the reference box: a VRAM slot (device-to-device, ~87 GB/s on the copy
// engine, findings §177) < the RAM arena (host-to-device, ~9-12 GB/s, §177) < a file (SSD,
// ~3 GB/s with 4 readers, §165). A slot can serve the COLD representation, and the arena the HOT one,
// only when the two representations are the same bytes (the full-IQ2 regime).
#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum moert_rep    { MOERT_REP_HOT = 0, MOERT_REP_COLD = 1 };
enum moert_source { MOERT_SRC_NONE = 0, MOERT_SRC_SLOT = 1, MOERT_SRC_ARENA = 2, MOERT_SRC_FILE = 3 };

typedef struct moert_expert_where {
    int32_t slot;          // VRAM slot holding the expert's HOT bytes, READY; -1 = none
    int32_t arena_valid;   // the RAM arena page will hold the expert's COLD bytes when a job queued
                           // now runs (resident, or released with a restore queued ahead of it)
    int32_t same_bytes;    // the hot and cold representations are byte-identical
    int32_t file_hot;      // a file source holds the HOT representation
    int32_t file_cold;     // a file source holds the COLD representation
} moert_expert_where;

// Cheapest valid source of representation `rep` (moert_rep) of one expert; MOERT_SRC_NONE if none.
static inline int32_t moert_source_for(const moert_expert_where * w, int32_t rep) {
    if (!w) return MOERT_SRC_NONE;
    const int32_t hot = rep == MOERT_REP_HOT;
    if (w->slot >= 0 && (hot || w->same_bytes))  return MOERT_SRC_SLOT;
    if (w->arena_valid && (!hot || w->same_bytes)) return MOERT_SRC_ARENA;
    if (hot ? w->file_hot : w->file_cold)          return MOERT_SRC_FILE;
    return MOERT_SRC_NONE;
}

static inline const char * moert_source_name(int32_t src) {
    switch (src) {
        case MOERT_SRC_SLOT:  return "slot";
        case MOERT_SRC_ARENA: return "arena";
        case MOERT_SRC_FILE:  return "file";
        default:              return "none";
    }
}

#ifdef __cplusplus
}
#endif
