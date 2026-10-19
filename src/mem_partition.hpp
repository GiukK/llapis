#pragma once

// ===========================================================================
// mem_partition — the M2 memory-partition model.
// ===========================================================================
// One struct, every byte accounted, both devices. Composes the existing sources:
// moe_dual::runtime_info() (hot slab / L2 / staging / cold arena / pinned),
// mem_probe (llama-owned non-expert weights + live KV), moe_dual::vram_now()
// (device-wide free/total), DXGI QueryVideoMemoryInfo (THIS process's VRAM usage,
// the WDDM ground truth nvidia-smi shows per process), GlobalMemoryStatusEx +
// GetProcessMemoryInfo (system/process RAM). Derived categories close the ledger:
//   VRAM: other_procs = total − free − ours;  compute_other = ours − Σ(known ours)
//   RAM:  other_procs = total − avail − rss;  nonarena_rss  = rss − cold_arena
// The ledger is measured, not estimated: every row comes from one of the sources above and
// the derived rows are what closes it, so the totals reconcile with what the OS reports for
// this process rather than approximating it. All values MiB. Windows-only (DXGI/psapi) — the
// reference box; non-Windows builds get zeros for the OS-derived fields.

struct llama_context;
struct llama_model;

namespace mem_partition {

struct Partition {
    bool vram_valid = false;        // device queries succeeded (GPU pool active)
    // --- VRAM (MiB) ---
    long long vram_total        = 0;   // device-wide
    long long vram_free         = 0;   // device-wide free (driver view)
    long long vram_ours         = 0;   // THIS process (DXGI local-segment usage; 0 if query failed)
    long long vram_shared_ours  = 0;   // THIS process's WDDM SHARED (system-memory) segment usage —
                                       // the overcommit ground truth. A growing value means our
                                       // "VRAM" allocations are spilling into system RAM (silent,
                                       // slow, and it dirties every measurement). Flagged upstream.
    long long vram_nonexpert    = 0;   // high-model non-expert weights (GPU tier)
    long long vram_hot_slab     = 0;   // hot Q4 expert slab
    long long vram_l2           = 0;   // VRAM L2 cold cache slab (0 unless on)
    long long vram_staging      = 0;   // prefill staging buffer (0 unless a stream mode is armed)
    long long vram_kv           = 0;   // live KV bytes (current fill, exact serialize size)
    // --- broken out of what used to be one opaque `vram_compute_other` block ------
    long long vram_compute_tgt  = 0;   // target context's scheduler/compute buffer (device part)
    long long vram_compute_dft  = 0;   // draft context's scheduler/compute buffer (device part)
    long long vram_drafter_wts  = 0;   // separately-loaded MTP drafter weights (0 if it reuses the target)
    bool      spec_active       = false; // a draft context exists (so the two rows above are live)
    // ggml-cuda's scratch pools, the biggest single piece of what's residual
    // still could not name. Reserved = held from the driver; used = currently handed out; the gap
    // is committed-but-idle scratch the VMM pool never returns.
    long long vram_pool_reserved= 0;
    long long vram_pool_used    = 0;
    int       vram_pool_count   = 0;
    long long vram_compute_other= 0;   // ours − Σ(every named row above). After and this is
                                       // the CUDA primary context + cuBLAS's own internal workspace
                                       // + allocator slack — no longer a dumping ground for the
                                       // drafter, the graph reserves, or the scratch pools.
    long long vram_other_procs  = 0;   // total − free − ours (everyone else incl. WDDM/desktop)
    // --- RAM (MiB) ---
    long long ram_total         = 0;
    long long ram_avail         = 0;   // "truly free" (available)
    long long ram_rss           = 0;   // this process working set
    long long ram_cold_arena    = 0;   // owned IQ2 cold expert arena, ALLOCATED size (see below)
    // --- C5.6: allocated is not resident, and this ledger used to conflate them ----------------
    // `ram_cold_arena` is how big the arena IS. On a 16 GB box holding a 7.9 GiB arena, most of
    // it is NOT in the working set — a full sweep faults ~98% of its pages, and only the routed
    // hot-ish subset stays physically present. The old derivation `nonarena_rss = rss −
    // cold_arena` assumed the whole arena sat inside RSS, so whenever the arena was bigger than
    // the working set (the normal case) the subtraction went negative, clamped to 0, and the UI
    // drew a RAM bar in which the runtime, the llama weights and the KV spill all occupied
    // exactly nothing. Both numbers are now measured rather than assumed:
    long long ram_arena_committed = 0; // arena bytes currently COMMITTED (== ram_cold_arena unless
                                       // the exclusive arena has released the pages of hot experts)
    long long ram_arena_resident  = 0; // arena bytes actually in this process's working set,
                                       // SAMPLED (see arena_resident_mib in mem_partition.cpp).
                                       // This is the one that belongs in a memory picture.
    long long ram_nonarena_rss  = 0;   // rss − arena_resident: llama weights in RAM, KV spill, runtime
    long long ram_other_procs   = 0;   // total − avail − rss
};

// Fill from the live process. ctx/model may be null (llama-owned rows read 0 — the
// server host passes the handles it received via the context trampoline).
// ctx_dft/model_dft are the speculative handles from the g_srv_moe_on_spec_ctx sink;
// both may be null (no speculative decoding; or MTP against the target model, where ctx_dft is
// live but model_dft is null because there are no separate drafter weights).
Partition probe(llama_context * ctx, const llama_model * model,
                llama_context * ctx_dft = nullptr, const llama_model * model_dft = nullptr);

// C5: refresh only the SYSTEM-level rows (device free/total, DXGI process VRAM, RAM totals/RSS,
// moe_dual-owned buffer sizes) and re-derive the ledger rows, keeping the llama-owned values
// (non-expert weights, live KV) from `prev`. Touches NO llama handle, so it is safe from a
// background timer while a slot decodes — the idle "memory probing" cadence the UI mem panel
// expects (the full probe stays slot-thread-only). KV only changes during decode, when the slot
// thread's full probe keeps it fresh anyway.
Partition refresh_system(const Partition & prev);

// --- the WDDM per-process budget, i.e. the number the allocator actually obeys ----
//
// Windows hands THIS process a budget on each memory segment. Allocating past it does NOT fail:
// WDDM demotes pages to system RAM and keeps going, so the runtime degrades onto PCIe silently.
// Device-wide free VRAM (nvidia-smi / cudaMemGetInfo) does not bound the budget and cannot be used
// as a proxy for it — measured on the reference box: 1.1 GiB device-free while our LOCAL budget was
// already exhausted, and free VRAM then *rose* as the driver evicted our own resident buffers.
// Every sizing decision must therefore fit under min(device free, headroom).
//
// All three are best-effort: false / -1 when DXGI is unavailable or the query failed, and callers
// MUST fall back to the device-free number rather than treating -1 as "no room".
bool      wddm_local(long long * budget_mib, long long * usage_mib);
long long wddm_local_headroom_mib();    // budget − usage, clamped at 0; -1 when unavailable
long long wddm_nonlocal_usage_mib();    // this process's shared/system-RAM segment usage; -1 if n/a

// Diagnostic: every adapter, both segments, usage AND budget, tagged. The aggregates above pick one
// adapter with a max(); this is what you read when you need to know WHICH one and how close to the
// wall it is. Prints to stderr (so it lands in LLAPIS_LOG_FILE like every other load-stage line).
void wddm_dump(const char * tag);

}  // namespace mem_partition
