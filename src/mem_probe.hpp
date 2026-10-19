#pragma once

// ===========================================================================
// mem_probe — llama-owned memory breakdown (feeds mem_partition + bench mem-map).
// ===========================================================================
// moe_dual::runtime_info() reports the moe_dual-OWNED regions (hot Q4 slab, VRAM
// L2, staging, cold IQ2 pool/arena) but has no llama_context/model handle, so the
// llama-owned pieces — the model's non-expert weights and the KV cache — are probed
// here, where cmd_runtime holds the context. Together they give the full VRAM/RAM
// map. Public llama API only (no internals): non-expert weight
// bytes are derived from llama_model_size minus the IQ2 expert bytes the caller
// already computed, and KV is the REAL serialized size of the live sequence (not an
// analytic estimate — Gemma's head_dim is not publicly exposed, so an analytic KV
// would be wrong; llama_state_seq_get_size is exact for the cells actually present).

#include <cstdint>

struct llama_context;
struct llama_model;

namespace mem_probe {

struct Breakdown {
    long long model_total_mib = 0;   // llama_model_size: all weights as loaded (non-expert + IQ2 experts)
    long long cold_expert_mib = 0;   // caller-supplied IQ2 cold expert bytes (runtime_info.cold_pool_mib)
    long long nonexpert_mib   = 0;   // model_total - cold_expert, clamped >=0 (hot Q4 slab is a separate copy)
    long long kv_current_mib  = 0;   // REAL KV bytes for seq_id 0 at the current fill (llama_state_seq_get_size)
    long long ctx_tokens      = 0;   // llama_n_ctx: KV capacity in tokens (context for the current fill)

    // --- the pieces that used to hide inside one unattributable residual ---------
    // Before this, mem_partition could only say "ours minus everything named", which on an 8 GB
    // box was a ~900 MiB block containing at least four unrelated things. Each is now read from
    // the engine rather than inferred, so a change in any one of them is visible on its own.
    long long compute_tgt_mib   = 0; // target context's scheduler/compute buffer, DEVICE part only
    long long compute_dft_mib   = 0; // draft context's scheduler/compute buffer, DEVICE part only
    long long compute_host_mib  = 0; // both contexts' pinned CUDA_Host compute staging — this is RAM,
                                     // reported so it is never silently counted as VRAM
    long long drafter_wts_mib   = 0; // separately-loaded MTP drafter weights (0 on the MTP-against-
                                     // target path, where no second model exists)
    bool      spec_active       = false; // a draft context exists right now
    bool      drafter_kv_shared = true;  // Gemma4 MTP: shared_kv_layers=4, the drafter aliases the
                                     // target's KV buffers, so its KV costs no INCREMENTAL VRAM
                                     // Probing it with llama_state_seq_get_size would double-count
                                     // the target's cells, so it is deliberately not summed.
};

// Fill a Breakdown from the live handles. `cold_expert_mib` is the IQ2 cold-pool size in MiB that
// moe_dual::runtime_info() already reports (cold_pool_mib) — passed in so the non-expert figure nets
// it out of the total model size. `ctx_dft`/`model_dft` are the speculative side;
// both may be null (no speculative decoding, or MTP running against the target model, in which case
// ctx_dft is non-null and model_dft is null). Safe with any handle null (those rows read 0).
Breakdown probe(llama_context * ctx, const llama_model * model, long long cold_expert_mib,
                llama_context * ctx_dft = nullptr, const llama_model * model_dft = nullptr);

}  // namespace mem_probe
