// moe_sidecar — host-side CPU cold executor for the LLapis V2 live seam (research,
// default off; armed by MOE_DUAL_COLD_CPU=1 in moe_dual's env shim).
//
// Computes y[:,i] = W[cold_ids[i]] · x_i for the few cold pairs of one (layer, kind)
// fused-op fire, on the host, straight from the RAM cold pool (owned arena / borrowed
// tensor) — the weights never cross PCIe. Uses ggml-cpu's own quantized vec_dot kernels
// (activation quantized once per call to the type's vec_dot_type), a persistent worker
// pool created at init (NO thread spawn / malloc in the decode loop), and preallocated
// scratch. Deterministic: per-row outputs are independent (no cross-thread reduction).
//
// NOT product code: the seam exists to measure the sidecar live.
#pragma once

#include "moert/cuda.h"   // moert_coldcpu_call

namespace moe_sidecar {

// Create the persistent pool (n_workers total participants incl. the calling thread;
// clamped to [1, 16]). Idempotent. Returns false if ggml-cpu init failed.
bool init(int n_workers);

// Join + destroy the pool (model/profile unload). Safe when never inited.
void shutdown();

// cold_ffn mode (findings 206): where layer il's DOWN cold pool lives (the gate_up one travels in the job).
struct DownPool { const void * base; uint64_t bytes, stride; int32_t type, d_in, d_out; };
typedef bool (*down_resolver_fn)(int32_t il, DownPool * out);
void set_down_resolver(down_resolver_fn fn);
// Called with a layer's cold ids before the job reads the pool (the exclusive arena's guard), or null.
void set_cold_guard(void (*fn)(int32_t il, const int32_t * ids, int32_t n));

}  // namespace moe_sidecar

// The C callback registered into moert_cuda_config::coldcpu_fn (island -> host).
extern "C" void moe_sidecar_compute(const moert_coldcpu_call * call);


// cold_ffn mode: the island submits one layer's cold FFN (returns at once) and later waits for it.
extern "C" void    moe_sidecar_ffn_submit(const moert_coldffn_job * job);
extern "C" int32_t moe_sidecar_ffn_wait(int32_t il, int64_t * cpu_ns);
