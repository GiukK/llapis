// ===========================================================================
// mem_probe — see mem_probe.hpp.
// ===========================================================================

#include "mem_probe.hpp"

#include "llama.h"

namespace mem_probe {

Breakdown probe(llama_context * ctx, const llama_model * model, long long cold_expert_mib,
                llama_context * ctx_dft, const llama_model * model_dft) {
    Breakdown b;
    b.cold_expert_mib = cold_expert_mib;

    // the compute buffers and the drafter, read from the engine instead of being
    // left inside an unattributable residual. Done before the early return below so they are
    // still reported when only the context handles are available.
    const auto to_mib = [](size_t bytes) { return (long long) (bytes >> 20); };
    if (ctx) {
        b.compute_tgt_mib  = to_mib(llama_context_compute_buffer_size(ctx, /*device_only=*/true));
        b.compute_host_mib = to_mib(llama_context_compute_buffer_size(ctx, /*device_only=*/false) -
                                    llama_context_compute_buffer_size(ctx, /*device_only=*/true));
    }
    if (ctx_dft) {
        b.spec_active      = true;
        b.compute_dft_mib  = to_mib(llama_context_compute_buffer_size(ctx_dft, /*device_only=*/true));
        b.compute_host_mib += to_mib(llama_context_compute_buffer_size(ctx_dft, /*device_only=*/false) -
                                     llama_context_compute_buffer_size(ctx_dft, /*device_only=*/true));
    }
    // Only a SEPARATELY loaded drafter adds weights. On the MTP-against-target path model_dft is
    // null by contract (the hook passes nullptr) because the drafter reuses the target's weights —
    // counting them again would double-book the non-expert row below.
    if (model_dft && model_dft != model) {
        b.drafter_wts_mib = to_mib(llama_model_size(model_dft));
    }

    if (!model) return b;

    b.model_total_mib = (long long) (llama_model_size(model) >> 20);
    // Non-expert weights = everything the model loaded minus the IQ2 experts. The hot Q4 slab is a
    // SEPARATE allocation (moe_dual-owned, reported by runtime_info().slab_mib) and is not part of
    // llama_model_size, so it must not be subtracted here.
    b.nonexpert_mib = b.model_total_mib - cold_expert_mib;
    if (b.nonexpert_mib < 0) b.nonexpert_mib = 0;   // guards a stale/oversized cold estimate

    if (ctx) {
        b.ctx_tokens = (long long) llama_n_ctx(ctx);
        // REAL KV footprint for the chat's single sequence (seq_id 0), at the current fill. This is
        // the exact byte count llama would serialize for those cells — dominated by the K/V rows,
        // plus a small per-cell metadata header. Reflects occupancy (n_past), not full capacity.
        b.kv_current_mib = (long long) (llama_state_seq_get_size(ctx, /*seq_id*/ 0) >> 20);
    }
    return b;
}

}  // namespace mem_probe
