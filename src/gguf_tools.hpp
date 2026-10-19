#pragma once

#include <filesystem>
#include <string>

// ---------------------------------------------------------------------------
// GGUF tensor-splicing tools (Phase 1.5b / 1.5c model construction).
// ---------------------------------------------------------------------------
// Both build a derived GGUF by streaming tensor bytes between source files
// without holding the multi-GB result in RAM (documented gguf "meta-then-append"
// pattern). They are pure offline file transforms — no model load / backend.
namespace gguf_tools {

// 1.5b — byte-exact splice: take the *expert* tensors (name contains "_exps")
// from `experts_from` and every other tensor from `rest_from`. No requant.
// take_low (optional, GPU-floor program): comma list of NON-expert tensor names ALSO
// taken from `experts_from` (e.g. "token_embd.weight" for the Q5_K-head rung); every
// listed name must match a tensor or the splice fails loudly.
int make_hybrid(const std::filesystem::path & experts_from,
                const std::filesystem::path & rest_from,
                const std::filesystem::path & out_path,
                const std::string & take_low = "");

// EXPERT STORE v0.
// Slice every per-expert page (weights + scales) of `src_path`'s stacked *_exps
// tensors into `experts.pack` + `experts.index`, byte-exact (no requant), page
// order (layer, expert, kind, scale?) = the runtime promotion unit. The store is
// the moert memory manager's SSD promotion source; auto-detected by the runtime
// as "<src.gguf>" + MOERT_STORE_INDEX/PACK_SUFFIX siblings. Single-file sources
// only in v0 (split GGUFs refused). Completeness-gated like q4stream_init.
int make_expert_store(const std::filesystem::path & src_path,
                      const std::filesystem::path & out_index,
                      const std::filesystem::path & out_pack);

// Identity hash of `src_path`'s expert-tensor metadata (arch + each *_exps name/
// type/shape/offset), stored as the index header's source_uuid and re-derived at
// runtime to refuse a store built from a different source. 0 on failure.
uint64_t expert_source_uuid(const std::filesystem::path & src_path);

// Two-stage drafter head (claude dump findings 291): write `out_path` = the drafter + the shipped basis Q
// (draft_head_proj.weight) + its approximate head E.Q (draft_head_low.weight, Q8_0) + llapis.draft_head.candidates.
// 0 = written, 2 = the drafter is not the one the basis was fit on (nothing written), 1 = error.
int make_draft_head(const std::filesystem::path & drafter,
                    const std::filesystem::path & basis,
                    const std::filesystem::path & out_path);

}  // namespace gguf_tools
