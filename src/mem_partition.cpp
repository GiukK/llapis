// mem_partition — implementation. See mem_partition.hpp for the model and the
// reconciliation contract. Windows-specific sources are compiled only on _WIN32.
#include "mem_partition.hpp"

#include "mem_probe.hpp"
#include "moe_dual.hpp"

#include "ggml-cuda.h"   // ggml_backend_cuda_pool_stats (scratch-pool occupancy)

#include <algorithm>
#include <mutex>
#include <vector>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <psapi.h>
#include <dxgi1_4.h>
#endif

namespace mem_partition {

namespace {

#ifdef _WIN32
// THIS process's VRAM usage on the local (dedicated) segment — the WDDM number
// nvidia-smi reports per process. Best-effort: 0 when DXGI or the query fails.
// THIS process's usage on a WDDM memory segment group. LOCAL = dedicated VRAM (the number
// nvidia-smi shows per process); NON_LOCAL = the shared system-memory segment — a nonzero,
// GROWING value there is the overcommit signature. Best-effort: 0 when DXGI fails.
// `Budget` is the field that actually governs the allocator, and reading only
// `CurrentUsage` is why the runtime could not see the wall it was hitting. WDDM gives THIS process
// a per-segment budget; allocations past it do not fail, they get DEMOTED to system RAM (silently,
// over PCIe). Device-wide free VRAM does not bound that — measured: 1.1 GiB device-free with our
// LOCAL budget already exhausted, so 754 MiB of already-resident buffers were evicted to the
// NON_LOCAL segment at drafter load. Both fields now come back from ONE query on ONE adapter, so
// budget and usage can never be read off two different GPUs.
bool dxgi_process_seg(DXGI_MEMORY_SEGMENT_GROUP seg, long long * budget_mib, long long * usage_mib) {
    IDXGIFactory4 * factory = nullptr;
    if (FAILED(CreateDXGIFactory1(__uuidof(IDXGIFactory4), (void **) &factory)) || !factory) return false;
    bool      found      = false;
    long long best_usage = -1;
    long long best_budget = 0;
    IDXGIAdapter1 * adapter = nullptr;
    for (UINT i = 0; factory->EnumAdapters1(i, &adapter) != DXGI_ERROR_NOT_FOUND; ++i) {
        DXGI_ADAPTER_DESC1 desc{};
        adapter->GetDesc1(&desc);
        IDXGIAdapter3 * a3 = nullptr;
        if (!(desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) &&
            SUCCEEDED(adapter->QueryInterface(__uuidof(IDXGIAdapter3), (void **) &a3)) && a3) {
            DXGI_QUERY_VIDEO_MEMORY_INFO info{};
            if (SUCCEEDED(a3->QueryVideoMemoryInfo(0, seg, &info))) {
                // multiple adapters (iGPU + dGPU): the CUDA pool lives on the one with the
                // largest usage from this process — take the max
                const long long u = (long long) (info.CurrentUsage >> 20);
                if (u > best_usage) {
                    best_usage  = u;
                    best_budget = (long long) (info.Budget >> 20);
                    found       = true;
                }
            }
            a3->Release();
        }
        adapter->Release();
    }
    factory->Release();
    if (!found) return false;
    if (budget_mib) *budget_mib = best_budget;
    if (usage_mib)  *usage_mib  = best_usage < 0 ? 0 : best_usage;
    return true;
}
long long dxgi_process_seg_mib(DXGI_MEMORY_SEGMENT_GROUP seg) {
    long long usage = 0;
    return dxgi_process_seg(seg, nullptr, &usage) ? usage : 0;
}
long long dxgi_process_vram_mib() { return dxgi_process_seg_mib(DXGI_MEMORY_SEGMENT_GROUP_LOCAL); }
#endif

long long clamp0(long long v) { return v < 0 ? 0 : v; }

#ifdef _WIN32
// C5.6 — how much of the owned cold arena is ACTUALLY in the working set.
//
// The question matters because the arena is the single biggest number in the RAM picture
// (7.9 GiB on the reference box) and it is mostly NOT resident: it is committed address space
// that the OS backs on demand, and a full sweep of it faults ~98% of its pages. Reporting the
// allocated size as if it were occupancy made the RAM bar say "cold experts: 7.9 GiB" while the
// process working set was a fraction of that, and drove ram_nonarena_rss to a clamped 0.
//
// Method, stated because it is an ESTIMATE and the caller should know its shape:
// QueryWorkingSetEx answers "is this address resident?" exactly, but the arena is ~2 M pages and
// asking about all of them every second is not free. So we ask about a fixed, evenly spaced
// SAMPLE (MAX_PROBES pages spread across the blocks) and scale the hit fraction by the arena
// size. Sampling error on a uniform stride is well under a percent at this sample count, and the
// number is a memory picture, not an allocator input. Cost: one syscall over a 64 KiB buffer.
//
// Exact at both ends by construction: an arena the OS has fully evicted samples 0 hits, a fully
// resident one samples all of them.
long long arena_resident_mib() {
    // Memoized for 2 s. refresh_partition() runs on the SLOT thread every 16 visible tokens
    // (~3 Hz while decoding) as well as on the 5 s idle timer, and residency does not move fast
    // enough to be worth a syscall on the decode path three times a second.
    static std::mutex   memo_mtx;
    static long long    memo_val = -1;
    static long long    memo_ms  = 0;
    const long long now_ms = (long long) GetTickCount64();
    {
        std::lock_guard<std::mutex> lk(memo_mtx);
        if (memo_ms != 0 && now_ms - memo_ms < 2000) return memo_val;
    }
    auto publish = [&](long long v) {
        std::lock_guard<std::mutex> lk(memo_mtx);
        memo_val = v; memo_ms = now_ms;
        return v;
    };

    const int nblocks = moe_dual::cold_arena_block_count();
    if (nblocks <= 0) return publish(0);

    struct Block { char * base; unsigned long long bytes; };
    std::vector<Block> blocks;
    blocks.reserve((size_t) nblocks);
    unsigned long long total_bytes = 0;
    for (int i = 0; i < nblocks; ++i) {
        void * b = nullptr; unsigned long long n = 0;
        if (!moe_dual::cold_arena_block(i, &b, &n) || !b || n == 0) continue;
        blocks.push_back(Block{ (char *) b, n });
        total_bytes += n;
    }
    if (blocks.empty() || total_bytes == 0) return publish(0);

    constexpr size_t PAGE       = 4096;
    constexpr size_t MAX_PROBES = 4096;
    const unsigned long long total_pages = total_bytes / PAGE;
    if (total_pages == 0) return publish(0);
    const unsigned long long stride_pages = total_pages > MAX_PROBES ? (total_pages / MAX_PROBES) : 1;

    std::vector<PSAPI_WORKING_SET_EX_INFORMATION> probes;
    probes.reserve(MAX_PROBES + 8);
    for (const Block & bl : blocks) {
        const unsigned long long pages = bl.bytes / PAGE;
        for (unsigned long long p = 0; p < pages && probes.size() < MAX_PROBES + 8; p += stride_pages) {
            PSAPI_WORKING_SET_EX_INFORMATION e{};
            e.VirtualAddress = bl.base + p * PAGE;
            probes.push_back(e);
        }
    }
    if (probes.empty()) return publish(0);
    if (!QueryWorkingSetEx(GetCurrentProcess(), probes.data(),
                           (DWORD) (probes.size() * sizeof(PSAPI_WORKING_SET_EX_INFORMATION)))) {
        return publish(-1);   // query unavailable: the caller reports a floor, never a guess
    }
    size_t resident = 0;
    for (const auto & e : probes) if (e.VirtualAttributes.Valid) ++resident;
    const double frac = (double) resident / (double) probes.size();
    return publish((long long) ((double) (total_bytes >> 20) * frac + 0.5));
}
#else
long long arena_resident_mib() { return -1; }
#endif

}  // namespace

// Shared core: everything except the llama-owned rows (non-expert weights, live KV, and — since
// the two compute buffers and the drafter weights), which the caller supplies:
// probe() reads them via mem_probe (slot-thread-only), refresh_system() carries them over from the
// previous snapshot (safe from any thread).
struct LlamaRows {
    long long nonexpert_mib  = 0;
    long long kv_mib         = 0;
    long long compute_tgt    = 0;
    long long compute_dft    = 0;
    long long drafter_wts    = 0;
    bool      spec_active    = false;
};

static Partition probe_core(const LlamaRows & lr) {
    const long long nonexpert_mib = lr.nonexpert_mib;
    const long long kv_mib        = lr.kv_mib;
    Partition p;
    const moe_dual::RuntimeInfo r = moe_dual::runtime_info();

    // --- VRAM ---
    long long vfree = 0, vtot = 0;
    p.vram_valid = moe_dual::vram_now(&vfree, &vtot);
    if (p.vram_valid) {
        p.vram_total = vtot;
        p.vram_free  = vfree;
    }
    p.vram_hot_slab  = r.slab_mib;
    p.vram_l2        = r.l2_slab_mib;
    p.vram_staging   = r.staging_mib;
    p.vram_nonexpert = nonexpert_mib;   // World C: the non-expert tier lives on the GPU
    p.vram_kv        = kv_mib;
    p.vram_compute_tgt = lr.compute_tgt;   //
    p.vram_compute_dft = lr.compute_dft;
    p.vram_drafter_wts = lr.drafter_wts;
    p.spec_active      = lr.spec_active;
    // ggml-cuda scratch pools. Registry walk only — no CUDA calls, safe from the
    // background refresh thread as well as the slot thread.
    {
        size_t pr = 0, pu = 0; int pn = 0;
        ggml_backend_cuda_pool_stats(/*device=*/-1, &pr, &pu, &pn);
        p.vram_pool_reserved = (long long) (pr >> 20);
        p.vram_pool_used     = (long long) (pu >> 20);
        p.vram_pool_count    = pn;
    }
#ifdef _WIN32
    p.vram_ours        = dxgi_process_vram_mib();
    p.vram_shared_ours = dxgi_process_seg_mib(DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL);
#endif
    if (p.vram_ours > 0) {
        p.vram_compute_other = clamp0(p.vram_ours - (p.vram_nonexpert + p.vram_hot_slab +
                                                     p.vram_l2 + p.vram_staging + p.vram_kv +
                                                     p.vram_compute_tgt + p.vram_compute_dft +
                                                     p.vram_drafter_wts + p.vram_pool_reserved));
    }
    // C5.6: `other_procs` is total − free − ours and is meaningful even when the DXGI query
    // fails (ours = 0 then attributes everything on the device to "everyone else", which is at
    // least a true statement about what we could measure). It used to be computed only INSIDE
    // the `ours > 0` branch, so a failed DXGI query produced a partition in which the device was
    // simultaneously 90% full and 100% unattributed — and the UI drew that as free VRAM.
    if (p.vram_valid) {
        p.vram_other_procs = clamp0(p.vram_total - p.vram_free - p.vram_ours);
    }

    // --- RAM ---
#ifdef _WIN32
    MEMORYSTATUSEX ms{};
    ms.dwLength = sizeof(ms);
    if (GlobalMemoryStatusEx(&ms)) {
        p.ram_total = (long long) (ms.ullTotalPhys >> 20);
        p.ram_avail = (long long) (ms.ullAvailPhys >> 20);
    }
    PROCESS_MEMORY_COUNTERS_EX pmc{};
    if (GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS *) &pmc, sizeof(pmc))) {
        p.ram_rss = (long long) (pmc.WorkingSetSize >> 20);
    }
#endif
    p.ram_cold_arena     = r.cold_arena_mib;
    p.ram_arena_committed = moe_dual::cold_arena_committed_mib();
    if (p.ram_arena_committed <= 0) p.ram_arena_committed = p.ram_cold_arena;
    // Resident, measured. -1 = the query failed: report 0 rather than a guess.
    const long long res = arena_resident_mib();
    p.ram_arena_resident = res >= 0 ? res : 0;
    if (p.ram_arena_resident > p.ram_arena_committed) p.ram_arena_resident = p.ram_arena_committed;
    // The runtime's own RAM = working set minus the RESIDENT part of the arena. Subtracting the
    // ALLOCATED arena here (the pre-C5.6 formula) went negative on every normal run and clamped
    // to zero, which is how the memory panel came to claim the runtime used no RAM at all.
    p.ram_nonarena_rss = clamp0(p.ram_rss - p.ram_arena_resident);
    p.ram_other_procs  = clamp0(p.ram_total - p.ram_avail - p.ram_rss);
    return p;
}

Partition probe(llama_context * ctx, const llama_model * model,
                llama_context * ctx_dft, const llama_model * model_dft) {
    const moe_dual::RuntimeInfo r = moe_dual::runtime_info();
    const mem_probe::Breakdown mb = mem_probe::probe(ctx, model, r.cold_pool_mib, ctx_dft, model_dft);
    LlamaRows lr;
    lr.nonexpert_mib = mb.nonexpert_mib;
    lr.kv_mib        = mb.kv_current_mib;
    lr.compute_tgt   = mb.compute_tgt_mib;
    lr.compute_dft   = mb.compute_dft_mib;
    lr.drafter_wts   = mb.drafter_wts_mib;
    lr.spec_active   = mb.spec_active;
    return probe_core(lr);
}

Partition refresh_system(const Partition & prev) {
    LlamaRows lr;
    lr.nonexpert_mib = prev.vram_nonexpert;
    lr.kv_mib        = prev.vram_kv;
    lr.compute_tgt   = prev.vram_compute_tgt;
    lr.compute_dft   = prev.vram_compute_dft;
    lr.drafter_wts   = prev.vram_drafter_wts;
    lr.spec_active   = prev.spec_active;
    return probe_core(lr);
}

// --- the numbers that actually govern the allocator ------------------------------
bool wddm_local(long long * budget_mib, long long * usage_mib) {
#ifdef _WIN32
    return dxgi_process_seg(DXGI_MEMORY_SEGMENT_GROUP_LOCAL, budget_mib, usage_mib);
#else
    (void) budget_mib; (void) usage_mib;
    return false;
#endif
}

long long wddm_local_headroom_mib() {
    long long budget = 0, usage = 0;
    if (!wddm_local(&budget, &usage)) return -1;
    if (budget <= 0) return -1;              // a zero budget is a failed/unsupported query, not "no room"
    return clamp0(budget - usage);
}

// the full DXGI picture, per adapter, both segments, budget AND usage. Needed
// because every aggregate above collapses the adapters with a max() — and this box has TWO (the
// RTX 4070 and the CPU's integrated Radeon), for which "non-local" means completely different
// things: on a discrete GPU it is the system-RAM spill segment, on an integrated one it is simply
// where all of its memory lives. Any conclusion about a spill has to name the adapter it saw.
void wddm_dump(const char * tag) {
#ifdef _WIN32
    IDXGIFactory4 * factory = nullptr;
    if (FAILED(CreateDXGIFactory1(__uuidof(IDXGIFactory4), (void **) &factory)) || !factory) return;
    IDXGIAdapter1 * adapter = nullptr;
    for (UINT i = 0; factory->EnumAdapters1(i, &adapter) != DXGI_ERROR_NOT_FOUND; ++i) {
        DXGI_ADAPTER_DESC1 desc{};
        adapter->GetDesc1(&desc);
        IDXGIAdapter3 * a3 = nullptr;
        if (!(desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) &&
            SUCCEEDED(adapter->QueryInterface(__uuidof(IDXGIAdapter3), (void **) &a3)) && a3) {
            DXGI_QUERY_VIDEO_MEMORY_INFO loc{}, non{};
            const bool okl = SUCCEEDED(a3->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &loc));
            const bool okn = SUCCEEDED(a3->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL, &non));
            char name[128] = {0};
            WideCharToMultiByte(CP_UTF8, 0, desc.Description, -1, name, sizeof(name) - 1, nullptr, nullptr);
            std::fprintf(stderr, "[wddm %s] adapter %u '%s' dedicated=%lld MiB | LOCAL use %lld / budget %lld"
                                 " | NON_LOCAL use %lld / budget %lld\n",
                         tag, i, name, (long long) (desc.DedicatedVideoMemory >> 20),
                         okl ? (long long) (loc.CurrentUsage >> 20) : -1,
                         okl ? (long long) (loc.Budget       >> 20) : -1,
                         okn ? (long long) (non.CurrentUsage >> 20) : -1,
                         okn ? (long long) (non.Budget       >> 20) : -1);
            a3->Release();
        }
        adapter->Release();
    }
    factory->Release();
    std::fflush(stderr);
#else
    (void) tag;
#endif
}

long long wddm_nonlocal_usage_mib() {
#ifdef _WIN32
    long long usage = 0;
    if (!dxgi_process_seg(DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL, nullptr, &usage)) return -1;
    return usage;
#else
    return -1;
#endif
}

}  // namespace mem_partition
