#pragma once

// ===========================================================================
// model_plan —: model folders, pre-load hardware probe, auto-K.
// ===========================================================================
// Scans models/high + models/low (one GGUF each), validates the pair (same
// model geometry, high = hot/promotion source, low = cold tier), probes free
// VRAM/RAM before any model is loaded, and computes the K plan:
//
//   need(K)     = NE_vram + K*per_k_step + kv_alloc(ctx) + compute_reserve
//   headroom(K) = vram_free_now − need(K)
//   verdict     = SAFE (headroom ≥ safe), WARN (≥ tight), REFUSE (< tight)
//
// Every constant is derived from the GGUF tensor table / metadata except
// compute_reserve, a CALIBRATED residual (see PlanOptions) — echoed in the
// output so it is never an invisible assumption.
//
// NE_vram INCLUDES token_embd: gemma ties the output head to the embedding
// table, so those 748 MiB are IN VRAM (M2 partition: nonexpert_weights=2459).
//
// STRICT ERROR POLICY (user directive 2026-07-07): any missing file, missing
// metadata key, unknown architecture with SWA, or pair mismatch THROWS
// std::runtime_error with a precise reason. No silent fallbacks.
// Calibration anchors (reference 8 GB box, ctx 4096, cut config, cuda-view
// free_pre=7054): headroom K32=824, K36=394, K38=179 MiB — reproduced EXACTLY.

#include <cstdint>
#include <string>
#include <vector>

namespace model_plan {

struct ModelInfo {
    std::string path;
    std::string arch;                 // general.architecture
    int32_t     file_type      = -1;  // general.file_type (raw enum)
    long long   file_bytes     = 0;
    int32_t n_layer        = 0;
    int32_t n_moe_layer    = 0;       // layers that actually carry ffn_*_exps tensors
    int32_t n_expert       = 0;
    int32_t n_expert_used  = 0;
    // kv geometry (for the kv_alloc(ctx) term) — PER LAYER, exact.
    // gemma4 reality check: head_count_kv is a non-uniform per-layer array (8 on SWA
    // layers, 2 on full-attention layers) and SWA layers use separate key/value
    // lengths (*_swa keys). A single-scalar model would silently mis-size KV.
    std::vector<int32_t> kv_heads;    // per layer (scalar keys broadcast)
    std::vector<uint8_t> is_swa;      // per layer: 1 = sliding-window, 0 = full attention
    int32_t key_length     = 0;       // full-attention layers
    int32_t value_length   = 0;
    int32_t key_length_swa   = 0;     // SWA layers (defaults to full lengths if absent)
    int32_t value_length_swa = 0;
    int32_t sliding_window = 0;       // 0 = full attention on every layer
    int32_t shared_kv_layers = 0;     // last N layers reuse an earlier layer's KV (gemma4: 4); NOT subtracted
                                      // from kv_alloc, so the KV terms are an upper bound
    // tensor-table sums (bytes)
    long long expert_bytes    = 0;    // all *_exps tensors
    long long max_layer_expert_bytes = 0;  // the largest single layer's *_exps bytes (the streamed prefill
                                           // stages one full expert layer at a time)
    long long nonexpert_bytes = 0;    // everything else EXCLUDING token_embd
    long long token_embd_bytes= 0;    // token_embd.weight (stays host-side under our config)
    double per_k_step_mib() const;    // expert_bytes / n_expert, MiB (one hot slot across all MoE layers)
};

// Read one GGUF's metadata + tensor table (no data load). Throws on any miss.
ModelInfo read_model(const std::string & gguf_path);

// List every metadata key (debug aid for STRICT key errors).
std::vector<std::string> dump_keys(const std::string & gguf_path);

struct ModelPair { ModelInfo high, low; };

// The ONE model .gguf in dir (mmproj-* projectors ignored). Throws when dir is missing or holds 0 or >1.
std::string find_model_gguf(const std::string & dir);

// models_dir must contain high/ and low/ with EXACTLY one .gguf each.
ModelPair scan_models_dir(const std::string & models_dir);

// Throws listing EVERY mismatch at once (arch/layers/experts/kv geometry,
// high must have more expert bytes than low).
void validate_pair(const ModelPair & p);

struct HwProbe {
    long long vram_total_mib = 0, vram_free_mib = 0;
    long long ram_total_mib  = 0, ram_avail_mib = 0;
    std::string gpu_desc;
};
// Pre-load probe (no model): ggml CUDA device enumeration + GlobalMemoryStatusEx.
// Throws if no GPU backend device is found.
HwProbe probe_hw();

enum class Verdict : int { SAFE = 0, WARN = 1, REFUSE = 2 };
const char * verdict_name(Verdict v);

struct PlanRow {
    int32_t   k = 0;
    long long slab_mib = 0, need_mib = 0, headroom_mib = 0;
    Verdict   verdict = Verdict::REFUSE;
};

struct PlanOptions {
    int32_t   ctx                 = 4096;
    int32_t   k_min               = 8;
    int32_t   k_max               = 64;
    // CALIBRATED residual (M2 partition + product-gate, reference box): with
    // NE = non-expert + token_embd and kv_alloc = FULL allocation at ctx, reserve = 54
    // reproduces the measured device usage EXACTLY at all three anchors —
    // headroom K32=824, K36=394, K38=179 MiB (cuda-view frame: the CUDA context is
    // largely drawn from the WDDM driver paging reserve, which cudaMemGetInfo already
    // excludes from `free`, so the residual is small; non-WDDM boxes lean on safe_mib).
    long long compute_reserve_mib = 54;
    long long safe_mib            = 300;  // verdict bands (K38 lesson: 180 = pressure)
    long long tight_mib           = 100;
};

struct Plan {
    PlanOptions opt;
    // derived, all echoed
    double    per_k_step_mib = 0.0;
    long long ne_vram_mib    = 0;   // nonexpert_bytes of HIGH (token_embd excluded)
    long long kv_alloc_mib   = 0;   // full pre-allocation at opt.ctx (iSWA-aware)
    long long ram_low_arena_mib = 0;// low-model expert bytes (host cold arena)
    std::vector<PlanRow> rows;
    int32_t k_max_safe = 0;         // recommendation: highest SAFE K
    int32_t k_default  = 0;         // min(32, k_max_safe)
    bool ram_fits = true;           // cold arena + working set vs ram_avail (informational)
};

Plan compute_plan(const ModelPair & p, const HwProbe & hw, const PlanOptions & opt);

// Full report (models, hw, constants, rows) as JSON.
std::string plan_to_json(const ModelPair & mp, const HwProbe & hw, const Plan & p);

// --- launcher terms (`moert-server plan --terms`) ---------------------------
// The quant-dependent terms of the launcher's VRAM model (llapis/cli/preset.ps1), read from the GGUF
// headers of the files the user actually installed instead of constants measured on one quant pair.
// No hardware probe and no model load: the launcher calls it before every boot.
//   speed   : both tiers are the LOW file (hot slots and non-experts at its precision)
//   quality : hot slots and non-experts from HIGH, cold tier from LOW
struct Terms {
    std::string mode;
    std::string hot_path, cold_path, drafter_path;
    double    slab_mib_per_k   = 0;   // one hot slot across every MoE layer, hot source precision
    long long ne_vram_mib      = 0;   // non-expert weights + token_embd (gemma ties the output head to it)
    long long drafter_mib      = 0;   // the drafter file's tensors (upper bound of its VRAM share)
    double    kv_mib_per_tok   = 0;   // KV growth per context token: full-attention layers only
    long long kv_swa_mib       = 0;   // fixed KV of the sliding-window layers (window-capped)
    long long cold_arena_mib   = 0;   // LOW file expert bytes: the host RAM cold arena
    long long staging_mib      = 0;   // one full expert layer of the hot source (streamed prefill reserve)
    int32_t   n_expert         = 0;   // the largest meaningful K
    int32_t   hot_file_type = -1, cold_file_type = -1;
};
// high_path may be empty in speed mode, drafter_path when no drafter is used. Throws (STRICT) on a
// missing or unreadable file and, in quality, on an invalid high/low pair.
Terms compute_terms(const std::string & mode, const std::string & high_path, const std::string & low_path,
                    const std::string & drafter_path);
std::string terms_to_json(const Terms & t);

}  // namespace model_plan
