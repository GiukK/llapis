#include "moe_dual.hpp"
#include "moe_dual_state.hpp"
#include "moe_xfer.hpp"
#include "kfit_policy.hpp"   // how many experts fit in VRAM right now (the single place it lives)

#include "io_util.hpp"
#include "gguf_tools.hpp"
#include "mem_partition.hpp"       // the WDDM per-process budget the allocator obeys
#include "llapis_log_filter.hpp"   // C5 phase tracing (llapis_phase)

#include "gguf.h"
#include "llama.h"

#ifdef MOE_DUAL_HAVE_CUDA_MAP
#include "ggml-cuda.h"
#endif

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>   // C5.2 RAM preflight (GlobalMemoryStatusEx)
#endif

// ===========================================================================
// moe_dual initialization: construct the hot pool + memmgr + policy and register the graph hook.
// Entry point (public API in moe_dual.hpp): init_dynamic() -- the per-session dual-precision cache:
// capacity-K VRAM pool per layer (empty), moert policy/memmgr, the admission preflight, the optional
// owned cold arena. All MOE_DUAL_* env knobs are parsed HERE; the moert libraries never read the env.
// ===========================================================================

namespace moe_dual {

namespace {

// this process's WDDM shared-segment usage BEFORE the run's first device allocation.
// -1 when DXGI is unavailable (the host then falls back to its own post-load reading).
long long g_nonlocal_boot_mib = -1;

// C5.2 admission preflight: set when init refused because free VRAM/RAM was clearly short
// of the load's needs. The host maps it to a dedicated exit code (86) so the router's
// spawn gate can tell "won't fit" from a crash and stop respawning into the same wall.
bool g_preflight_failed = false;

// --- variable-KV S6a: the K-DEPENDENT slab stack (typed pool tensors + slab allocation +
// placement + aux residency mirrors) is built through these three helpers so BOTH
// init_dynamic and slab_rebuild_at_k construct it identically (: the K-dependent
// pieces are built together and must be rebuilt together). All read the CURRENT per-layer
// capacity from the memmgr via klayer()/page_info. ----------------------------------------

// pass 1 (pre-finalize): fresh pack_ctx + typed pool tensors at the current per-layer K,
// verifying each ggml view agrees with the memmgr layout — contiguous K_l slots of
// bytes_per_expert each — and reporting each region's TRUE device allocation size to the
// memmgr (backends can require more than the payload: ggml-cuda pads quantized tensors so
// block loads past the last row stay in bounds; regions must be sized so those padded
// reads never cross into the next one). `slab_planned` (optional) receives the planned
// total device bytes (the C5.2 preflight input).
static bool build_pool_tensors(uint64_t * slab_planned) {
    ggml_init_params pp;
    pp.mem_size   = ((size_t) g.n_layers * 4 + 2) * ggml_tensor_overhead() + (1u << 20);
    pp.mem_buffer = nullptr; pp.no_alloc = true;
    g.pack_ctx = ggml_init(pp);
    static const char * kind_name[2] = { "gate_up", "down" };
    g.w_hot.assign(g.n_layers, {nullptr, nullptr});
    g.s_hot.assign(g.n_layers, {nullptr, nullptr});
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(g.backend);
    uint64_t planned = 0;
    for (int L = 0; L < g.n_layers; ++L) {
        for (int kind = 0; kind < 2; ++kind) {
            for (int ws = 0; ws < 2; ++ws) {
                moert_memmgr_page pg;
                if (!moert_memmgr_page_info(g.mm, L, kind, ws, &pg)) continue;
                ggml_tensor * ot = ggml_new_tensor(g.pack_ctx, (ggml_type) pg.type_tag, pg.ndim, pg.pool_ne);
                const std::string nm = "blk." + std::to_string(L) + ".ffn_" + kind_name[kind]
                                     + "_exps" + (ws ? ".scale" : ".weight");
                ggml_set_name(ot, nm.c_str());
                const int64_t kL = pg.pool_ne[pg.expert_axis];
                if (ggml_nbytes(ot) != (size_t) kL * pg.bytes_per_expert ||
                    ot->nb[pg.expert_axis] != pg.bytes_per_expert) {
                    std::cerr << "moe_dual: pool tensor " << nm << " disagrees with memmgr layout ("
                              << ggml_nbytes(ot) << " B vs K*" << pg.bytes_per_expert << " B)\n";
                    return false;
                }
                const uint64_t reg_alloc = (uint64_t) ggml_backend_buft_get_alloc_size(buft, ot);
                moert_memmgr_set_region_alloc(g.mm, L, kind, ws, reg_alloc);
                planned += reg_alloc;
                (ws ? g.s_hot : g.w_hot)[L][kind] = ot;
            }
        }
    }
    if (slab_planned) *slab_planned = planned;
    return true;
}

// finalize (drives cb_slab_alloc at the computed layout size) + pass 2: place the typed
// pool tensors at the finalized region offsets inside the slab.
static bool alloc_and_place_slab() {
    llapis_phase("Allocating hot expert cache in VRAM ... (K=%d, %d layers)", g.k, g.n_layers);
    if (!moert_memmgr_finalize(g.mm)) {
        std::cerr << "moe_dual: alloc dynamic pool failed (VRAM budget?)\n";
        return false;
    }
    g.slab_mib = (long long) (ggml_backend_buffer_get_size(g.slab_buf) >> 20);
    llapis_phase("Hot expert cache slab allocated: %lld MiB VRAM (empty — experts stream in later)", g.slab_mib);
    char * slab_base = (char *) ggml_backend_buffer_get_base(g.slab_buf);
    for (int L = 0; L < g.n_layers; ++L) {
        for (int kind = 0; kind < 2; ++kind) {
            for (int ws = 0; ws < 2; ++ws) {
                ggml_tensor * ot = (ws ? g.s_hot : g.w_hot)[L][kind];
                if (!ot) continue;
                moert_memmgr_page pg;
                moert_memmgr_page_info(g.mm, L, kind, ws, &pg);
                if (ggml_backend_tensor_alloc(g.slab_buf, ot, slab_base + pg.region_off) != GGML_STATUS_SUCCESS) {
                    std::cerr << "moe_dual: pool tensor placement failed (" << ot->name << ")\n";
                    return false;
                }
            }
        }
    }
    return true;
}

// aux residency mirrors: hotmap + residency + the fused-op page-table mirrors
// (exp_slot/slot_exp/slot_state), fresh aux_ctx, all-cold until push_residency.
// se_t/ss_t are sized to the CURRENT klayer(L) — the K-dependent piece.
static bool build_aux_mirrors() {
    ggml_init_params ap;
    ap.mem_size = (size_t) g.n_layers * 5 * ggml_tensor_overhead() + 4096;
    ap.mem_buffer = nullptr; ap.no_alloc = true;
    g.aux_ctx = ggml_init(ap);
    g.hotmap.assign(g.n_layers, nullptr); g.resid.assign(g.n_layers, nullptr);
    g.es_t.assign(g.n_layers, nullptr); g.se_t.assign(g.n_layers, nullptr); g.ss_t.assign(g.n_layers, nullptr);
    for (int L = 0; L < g.n_layers; ++L) {
        g.hotmap[L] = ggml_new_tensor_1d(g.aux_ctx, GGML_TYPE_I32, g.n_expert);
        g.resid[L]  = ggml_new_tensor_1d(g.aux_ctx, GGML_TYPE_F32, g.n_expert);
        g.es_t[L]   = ggml_new_tensor_1d(g.aux_ctx, GGML_TYPE_I32, g.n_expert);   // expert -> slot or -1
        g.se_t[L]   = ggml_new_tensor_1d(g.aux_ctx, GGML_TYPE_I32, klayer(L));  // slot   -> expert or -1 (per-layer K)
        g.ss_t[L]   = ggml_new_tensor_1d(g.aux_ctx, GGML_TYPE_I32, klayer(L));  // slot   -> SlotState (per-layer K)
        ggml_set_name(g.hotmap[L], ("moe_dual.hotmap." + std::to_string(L)).c_str());
        ggml_set_name(g.resid[L],  ("moe_dual.resid."  + std::to_string(L)).c_str());
        ggml_set_name(g.es_t[L],   ("moe_dual.expslot." + std::to_string(L)).c_str());
        ggml_set_name(g.se_t[L],   ("moe_dual.slotexp." + std::to_string(L)).c_str());
        ggml_set_name(g.ss_t[L],   ("moe_dual.slotstate." + std::to_string(L)).c_str());
    }
    g.aux_buf = ggml_backend_alloc_ctx_tensors(g.aux_ctx, g.backend);
    return g.aux_buf != nullptr;
}

}  // namespace

bool init_dynamic(const char * q4_path, int k) {
    g.dyn_src_path = q4_path ? q4_path : "";   // the boundary proof compares it with the staging source
    // C5.2 single-instance choke point: EVERY full dual-runtime load passes here.
    // One instance per session:
    // a second one would double the ~7.9 GiB arena + slab and OOM the box. Idempotent
    // for a process that already holds the lock (the moert-server entry point acquires
    // it first). Refusal counts as an admission failure (host exits 86/87, never falls
    // back to a stock full-model load).
    if (!llapis_singleton_acquire("Local\\LLapis-model")) {
        g_preflight_failed = true;
        std::cerr << "moe_dual: FATAL: another LLapis process already holds a model loaded — "
                     "one runtime instance per session (stop the other one first: run "
                     "`llapis stop`, or close the other terminal)\n";
        llapis_phase("PREFLIGHT FAILED: another LLapis runtime instance is already running — refusing to load");
        return false;
    }
    parse_hook_env();
    // policy knobs: validated defaults from moert, then the MOE_DUAL_* env compat surface on top.
    // (The env parsing stays HERE — the moert policy takes a Config at construction and never
    // reads the environment)
    moert_policy_config & pc = g.pcfg;
    moert_policy_config_defaults(&pc);
    // The rule's prices (A/B arms only; the product runs the library defaults, findings §173).
    if (const char * v = std::getenv("MOE_DUAL_COST_H"))           pc.cost_horizon           = std::atoi(v);
    if (const char * v = std::getenv("MOE_DUAL_COST_R"))           pc.cost_reuse             = (float) std::atof(v);
    if (const char * v = std::getenv("MOE_DUAL_COST_PER_TOKEN"))   pc.cost_per_token         = std::atoi(v);

    // --- geometry ---
    int n_layers = 0, n_expert = 0;
    ggml_context * ctx = nullptr;
    gguf_context * gg  = nullptr;
    // the promotion source is the source GGUF's own expert tensors (the "legacy" catalog; the expert-
    // store promotion source was never reached by the product and left in elision E2)
    {
        gguf_init_params ip; ip.no_alloc = true; ip.ctx = &ctx;
        gg = gguf_init_from_file(q4_path, ip);
        if (!gg) { std::cerr << "moe_dual: cannot open the promotion source " << q4_path << "\n"; return false; }
        // discover n_layers / n_expert from the *_exps.weight shapes
        const int64_t nt = gguf_get_n_tensors(gg);
        for (int64_t i = 0; i < nt; ++i) {
            const std::string nm = gguf_get_tensor_name(gg, i);
            const size_t p = nm.find("blk.");
            if (p != std::string::npos) { const int L = std::atoi(nm.c_str() + p + 4); if (L + 1 > n_layers) n_layers = L + 1; }
            if (nm.find("_exps.weight") != std::string::npos) {
                ggml_tensor * t = ggml_get_tensor(ctx, nm.c_str());
                if ((int) t->ne[2] > n_expert) n_expert = (int) t->ne[2];  // expert axis of a 3D expert weight
            }
        }
        if (n_layers <= 0 || n_expert <= 0) { std::cerr << "moe_dual: no experts in the promotion source\n"; gguf_free(gg); return false; }
    }
    if (k <= 0 || k > n_expert) k = std::min(48, n_expert);
    g.dynamic = true; g.n_layers = n_layers; g.n_expert = n_expert;
    g.backend = pick_pack_backend();

    g.k_requested = k;

    // --- memory manager (M1): page catalog + promotion source + hot-slab layout/lifetime.
    //     Built through a creator lambda. ---
    moert_memmgr_device mdev;
    mdev.slab_alloc  = &cb_slab_alloc;
    mdev.slab_upload = &cb_slab_upload;
    mdev.slab_free   = &cb_slab_free;
    mdev.user        = nullptr;
    mdev.slab_download = &cb_slab_download;
    auto create_mm = [&](int kk) -> moert_memmgr * {
        moert_memmgr * mm = moert_memmgr_create(n_layers, n_expert, kk, &mdev);
        if (!mm) { std::cerr << "moe_dual: memmgr create failed (bad geometry?)\n"; return nullptr; }
        // legacy compat shim: register the Q4 GGUF's stacked *_exps tensors as uniform-stride
        // page groups (offset of expert 0 + bytes/expert straight from the meta) — the same
        // offsets the pre-M1 stream_one computed, now owned by the memmgr.
        const int32_t sid = moert_memmgr_add_source(mm, q4_path);
        if (sid < 0) { moert_memmgr_free(mm); return nullptr; }
        const int64_t data_off = (int64_t) gguf_get_data_offset(gg);
        const int64_t nt = gguf_get_n_tensors(gg);
        for (int64_t i = 0; i < nt; ++i) {
            const std::string nm = gguf_get_tensor_name(gg, i);
            if (nm.find("_exps") == std::string::npos) continue;
            const size_t p = nm.find("blk."); if (p == std::string::npos) continue;
            const int L = std::atoi(nm.c_str() + p + 4); if (L < 0 || L >= n_layers) continue;
            const int kind = (nm.find("gate_up_exps") != std::string::npos) ? 0
                           : (nm.find("down_exps")    != std::string::npos) ? 1 : -1;
            if (kind < 0) continue;
            ggml_tensor * st = ggml_get_tensor(ctx, nm.c_str());
            int axis = -1; for (int d = GGML_MAX_DIMS - 1; d >= 0; --d) if ((int) st->ne[d] == n_expert) { axis = d; break; }
            if (axis < 0) continue;
            const int64_t soff = data_off + (int64_t) gguf_get_tensor_offset(gg, i);
            if (!moert_memmgr_set_page(mm, L, kind, ends_with(nm, ".scale") ? 1 : 0, sid,
                                       soff, st->nb[axis], (int32_t) st->type, st->ne, axis)) {
                std::cerr << "moe_dual: memmgr rejected page group " << nm << "\n";
                moert_memmgr_free(mm);
                return nullptr;
            }
        }
        return mm;
    };
    g.mm = create_mm(k);
    if (!g.mm) { if (gg) gguf_free(gg); return false; }

    g.k = k;
    if (gg) { gguf_free(gg); gg = nullptr; }

    // --- hot slab + typed pool tensors placed at the memmgr's region offsets (q4stream
    //     manual-placement pattern; finalize() drives cb_slab_alloc). The K-dependent stack
    //     is built by the shared S6a helpers above (also used by slab_rebuild_at_k). ---
    uint64_t slab_planned = 0;   // C5.2 preflight: the slab's total device bytes, known pre-alloc
    if (!build_pool_tensors(&slab_planned)) return false;
    // M3a report fields: device free/total right before the slab lands (recorded even with the
    // guardrail off, so every bench JSON documents where its K sat in the VRAM envelope).
    {
        size_t free_b = 0, total_b = 0;
        if (pool_dev_memory(&free_b, &total_b)) {
            g.vram_total_mib    = (long long) (total_b >> 20);
            g.vram_free_pre_mib = (long long) (free_b  >> 20);
        }
    }
    // C5.2 VRAM admission preflight: refuse BEFORE touching device memory when the load clearly cannot fit
    // (need = planned slab + a headroom for NE weights + KV + compute + drafter). Conservative on purpose:
    // it catches the hopeless cases (a second instance, another app holding VRAM) with a clean exit 86
    // instead of a CUDA OOM mid-load. LLAPIS_PREFLIGHT_HEADROOM_MIB overrides, and the launcher always sets it
    // from the installed models (their non-expert MiB + 941, + any staging above Q4_K_XL's); 0 disables. 3400 is
    // that sum for the Q4_K_XL non-experts, the default for a launch that bypasses the launcher.
    // (history: docs/code-comments-archive-2026-10-05.md, "preflight")
    {
        long long headroom_mib = 3400;
        if (const char * hv = std::getenv("LLAPIS_PREFLIGHT_HEADROOM_MIB"); hv && hv[0]) headroom_mib = std::atoll(hv);
        size_t free_b = 0, total_b = 0;
        if (headroom_mib > 0 && pool_dev_memory(&free_b, &total_b)) {
            const long long free_mib = (long long) (free_b  >> 20);
            const long long need_mib = (long long) (slab_planned >> 20) + headroom_mib;
            if (free_mib < need_mib) {
                g_preflight_failed = true;
                llapis_phase("PREFLIGHT FAILED: need ~%lld MiB VRAM (hot slab %lld + headroom %lld) "
                             "but only %lld / %lld MiB free — refusing to load (another instance or "
                             "app is holding VRAM; close it, or lower K/ctx)",
                             need_mib, (long long) (slab_planned >> 20), headroom_mib,
                             free_mib, (long long) (total_b >> 20));
                std::cerr << "moe_dual: PREFLIGHT FAILED: VRAM free " << free_mib << " MiB < need ~"
                          << need_mib << " MiB (slab " << (slab_planned >> 20) << " + headroom "
                          << headroom_mib << ") — refusing before any device allocation\n";
                return false;
            }
        }
    }
    mem_partition::wddm_dump("stage1-pre-slab");
    // capture the shared-segment baseline HERE, before the first device allocation
    // of the run. It used to be taken at on_ready — after the slab, the NE weights, the context and
    // the drafter had all been allocated — so a spill that happened during load was baked INTO the
    // baseline and the "growth over baseline" alarm could never fire on it. Measured: 844 MiB of
    // boot spill reported as a healthy baseline, versus 74 MiB read at this point with the whole
    // 7.9 GiB cold arena already in RAM (which is also the direct measurement that kills the
    // "it's just the pinned arena" reading of that number).
    g_nonlocal_boot_mib = mem_partition::wddm_nonlocal_usage_mib();
    // finalize + pass-2 placement, then the residency mirrors (all-cold; the mirror lets the
    // fused Plan C staging op fire for the dynamic cache).
    if (!alloc_and_place_slab()) return false;
    // stage 2 of the load-peak curve. proposed loading the drafter BEFORE the
    // slab to lower the load-time peak; measuring that needs the peak's actual shape, not a
    // guess about it. Stages: (1) pre-slab = g.vram_free_pre_mib, (2) here, (3) after model+ctx
    // (fork's own trace), (4) after the drafter (server_main's spec sink), (5) after the boot
    // release. The PEAK is what gates the preflight and the K clamp.
    {
        size_t fb = 0, tb = 0;
        if (pool_dev_memory(&fb, &tb)) {
            // free VRAM alone cannot tell "we allocated" from "the driver demoted our
            // pages to system RAM to let us allocate" — the second one RAISES free VRAM. Every
            // load-peak stage therefore carries the shared-segment reading next to it.
            std::cerr << "moe_dual: [load-peak] stage 2 slab allocated (" << g.slab_mib
                      << " MiB, K=" << g.k << "): " << (fb >> 20) << " / " << (tb >> 20)
                      << " MiB free (was " << g.vram_free_pre_mib << " pre-slab), shared "
                      << mem_partition::wddm_nonlocal_usage_mib() << " MiB (boot "
                      << g_nonlocal_boot_mib << ")\n";
        }
    }
    if (!build_aux_mirrors()) {
        std::cerr << "moe_dual: aux mirror alloc failed\n";
        return false;
    }

    // moert policy (page table + scores + planner, all cold): the callbacks route its promotion
    // streams and residency publishes back through this file's ggml/file plumbing.
    g.pcfg.n_layers = n_layers; g.pcfg.n_expert = n_expert; g.pcfg.k = k;
    g.pcfg.k_per_layer = nullptr;   // uniform K: the per-layer table is not a product feature
    moert_policy_callbacks cbs;
    cbs.stream_expert  = &cb_stream_expert;
    cbs.publish_layer  = &cb_publish_layer;
    cbs.fetch_external = &cb_fetch_external;
    cbs.user           = nullptr;
    g.pol = moert_policy_create(&g.pcfg, &cbs);
    if (!g.pol) { std::cerr << "moe_dual: moert policy create failed (bad geometry?)\n"; return false; }

    // 3b: OWNED cold arena as the live cold source. Build the per-(layer,kind,is_scale) cold
    // tensors in memmgr-owned host RAM from the store; the hook then sources cold from the arena
    // instead of llama's borrowed tensor (reg_cold byte-matches the two to prove it's bit-identical).
    if (const char * ap = std::getenv("MOE_DUAL_COLD_ARENA"); ap && ap[0] && g.mm) {
        const bool map_off = [] { const char * v = std::getenv("MOERT_ARENA_MAP"); return v && v[0] == '0'; }();
#ifdef _WIN32
        // RAM admission preflight. A second instance is refused by the singleton above, so this asks
        // only whether THIS machine can hold the arena. FREE physical RAM is the wrong question on
        // 16 GB: Windows makes room by trimming other processes, and "available >= arena" refused
        // every launch here (findings 189). Two checks that do fail a hopeless load:
        //   (1) installed RAM >= pack + 4 GiB -- below it the cold experts cannot stay resident next to
        //       the OS and the runtime, and decode would read them from the SSD every token;
        //   (2) committable memory (RAM + page file not yet committed) >= what the arena commits + 1 GiB
        //       -- the whole pack when it is copied (old layout, MOERT_ARENA_MAP=0), ~nothing when it is
        //       mapped (file pages are not commit). Running out of commit fails an allocation mid-load.
        // LLAPIS_PREFLIGHT_RAM=0 disables both (an LLAPIS_ name: the preset wipes unknown MOE_DUAL_* knobs).
        const bool mappable = !map_off && moert_memmgr_store_group_major(ap) == 1;
        {
            const char * mv = std::getenv("LLAPIS_PREFLIGHT_RAM");
            const bool check = !(mv && mv[0] == '0' && mv[1] == '\0');
            std::string pack = ap;
            if (pack.size() > 6 && pack.compare(pack.size() - 6, 6, ".index") == 0) pack.replace(pack.size() - 6, 6, ".pack");
            std::error_code fec;
            const long long pack_mib = (long long) (std::filesystem::file_size(pack, fec) >> 20);
            MEMORYSTATUSEX ms; ms.dwLength = sizeof(ms);
            if (check && !fec && pack_mib > 0 && GlobalMemoryStatusEx(&ms)) {
                const long long total_mib  = (long long) (ms.ullTotalPhys >> 20);
                const long long commit_mib = (long long) (ms.ullAvailPageFile >> 20);   // commit limit - charge
                const long long floor_mib  = pack_mib + 4096;
                const long long commit_need = (mappable ? 0 : pack_mib) + 1024;
                const char * why = nullptr;
                long long need = 0, have = 0;
                if (total_mib < floor_mib)        { why = "installed RAM";     need = floor_mib;   have = total_mib; }
                else if (commit_mib < commit_need) { why = "committable memory (RAM + page file)"; need = commit_need; have = commit_mib; }
                if (why) {
                    g_preflight_failed = true;
                    llapis_phase("PREFLIGHT FAILED: the cold expert arena (%lld MiB, %s) needs ~%lld MiB of %s, "
                                 "this machine has %lld MiB -- refusing to load",
                                 pack_mib, mappable ? "mapped" : "copied", need, why, have);
                    std::cerr << "moe_dual: PREFLIGHT FAILED: " << why << " " << have << " MiB < need ~" << need
                              << " MiB for the cold arena (" << (mappable ? "mapped" : "copied") << ")\n";
                    return false;
                }
                std::cerr << "moe_dual: RAM preflight ok -- installed " << total_mib << " MiB (floor " << floor_mib
                          << "), committable " << commit_mib << " MiB (need " << commit_need << ", arena "
                          << (mappable ? "mapped" : "copied") << ")\n";
            }
        }
#endif
        llapis_phase("Loading the low model's experts into RAM (cold arena) ...");
        const auto arena_t0 = std::chrono::steady_clock::now();
        if (moert_memmgr_build_cold_arena(g.mm, ap, 0)) {
            if (!map_off && !moert_memmgr_cold_arena_mapped(g.mm))
                llapis_phase("note: models/cache/low.experts.pack predates 2026-10-08 and is copied into RAM "
                             "(slower start, twice the private memory) -- run 'moert-server prepare' once to rebuild it");
            g.cold_arena_on = true;
            if (g.hcfg.arena_exclusive && !moert_memmgr_cold_arena_mapped(g.mm) && !std::getenv("MOE_DUAL_ARENA_EXCLUSIVE")) {
                g.hcfg.arena_exclusive = 0;   // the default is for the mapped pack only (findings 283, 200)
                std::cerr << "moe_dual: exclusive arena OFF (default): the arena is a copy, not the mapped pack\n";
            }
            llapis_phase("Cold arena loaded: %lld MiB host RAM in %.1f s",
                         (long long) (moert_memmgr_cold_arena_bytes(g.mm) >> 20),
                         std::chrono::duration<double>(std::chrono::steady_clock::now() - arena_t0).count());
            std::cerr << "moe_dual: 3b OWNED cold arena ACTIVE (" << (moert_memmgr_cold_arena_bytes(g.mm) >> 20)
                      << " MiB host RAM) -- fused op sources cold from moert, not the borrowed tensor\n";
        } else {
            std::cerr << "moe_dual: MOE_DUAL_COLD_ARENA build FAILED: " << ap << " -- staying on borrowed cold\n";
        }
    }

    for (int L = 0; L < n_layers; ++L) push_residency(L);   // publish initial residency (all cold)

    llama_set_moe_dual_mm_id(&hook);
#ifdef MOE_DUAL_HAVE_CUDA_MAP
    llama_set_host_map(&ggml_backend_cuda_host_map);   // findings 248: LLAPIS_DRAFT_HEAD_HOST (the drafter loads after this)
#endif
    std::cerr << "moe_dual: DYNAMIC cache (" << n_layers << " layers, K=" << k << "/" << n_expert << ", "
              << "STAGED mode, "
              << "COLD start" << "); source "
              << q4_path << " [gguf offsets]\n";
    std::cerr << "moe_dual:   cost-aware rule: horizon " << g.pcfg.cost_horizon << " tok, reuse price "
              << g.pcfg.cost_reuse << " cold pairs, rate decay " << g.pcfg.cost_rate_decay
              << ", decides every visible token (cap " << g.pcfg.cost_per_token << " promotions/token)\n";

    // Device routing counters: the ONLY routing observation (the eval-callback read per layer -- ~31
    // synchronous segments per token -- was removed in elision E1). The server refuses to run without.
#ifdef MOE_DUAL_HAVE_CUDA_MAP
    g.dev_counts = ggml_backend_cuda_moe_dual_route_counts_enable(n_layers, n_expert) == 1;
#endif
    moert_policy_set_external_observation(g.pol, g.dev_counts ? 1 : 0);
    forecast_init();   // route trace, batch transport and MTP forecaster, all opt-in
    std::cerr << (g.dev_counts
        ? "moe_dual:   DEVICE routing counters ON: scores observed on GPU, fetched at refresh points\n"
        : "moe_dual: CUDA routing-counter alloc FAILED - the server will refuse to run (blind routing)\n");
    // Hand the (empty) slab back DOWN TO THE FLOOR before llama loads anything: holding 2.9 GiB of
    // never-written VRAM through the NE load made WDDM demote our real allocations (756 MiB spilled in
    // that window). Shrunk, not evicted: graph_reserve's token graph needs a RESIDENT slab (else the
    // loader-cut shells inflate the reserve to ~2.7 GiB). on_ready fits K to the measured VRAM.
    // (history: docs/code-comments-archive-2026-10-05.md, "init-shrink")
    {
        const long long free_before = pool_free_mib();
        const long long nl_before   = mem_partition::wddm_nonlocal_usage_mib();
        vkv_fit_slab_for_ctx_swap(-1);
        std::cerr << "moe_dual: [load-peak] stage 2b boot slab shrunk to the floor before the NE load "
                     "(it would sit empty through the whole model load): " << free_before
                  << " -> " << pool_free_mib() << " MiB free, shared " << nl_before << " -> "
                  << mem_partition::wddm_nonlocal_usage_mib() << " MiB\n";
        mem_partition::wddm_dump("stage2b-post-shrink");
    }
    return true;
}

// C5.2: true when init refused on the admission preflight (VRAM or RAM clearly short).
// The host maps this to exit code 86 so the router can tell "won't fit" from a crash.
bool preflight_failed() { return g_preflight_failed; }

// --- variable-KV: armed state of vkv_restore_or_refit() (server_main sets it once at init)
static bool      g_vkv_armed          = false;
static long long g_vkv_boot_keep_mib  = 300;

void vkv_set_boot_policy(bool armed, long long keep_free_mib) {
    g_vkv_armed         = armed;
    g_vkv_boot_keep_mib = keep_free_mib;
}

// --- variable-KV Phase 3: the MEASURED K clamp ------------------------------------------------
// Exact per-K byte cost of the hot slab: one more K = one more expert slot in EVERY registered
// page group (all layers x kinds x weight/scale). Straight from the memmgr catalog — no model.
uint64_t vkv_per_k_bytes() {
    if (!g.mm) return 0;
    uint64_t per_k = 0;
    for (int L = 0; L < g.n_layers; ++L)
        for (int kind = 0; kind < 2; ++kind)
            for (int ws = 0; ws < 2; ++ws) {
                moert_memmgr_page pg;
                if (moert_memmgr_page_info(g.mm, L, kind, ws, &pg)) per_k += pg.bytes_per_expert;
            }
    return per_k;
}

// Quality floor: never let the hot tier collapse to nothing if we can help it — below this the
// cache stops being worth its rebuild cost. Fixed, not tunable. Shared
// by vkv_refit_k's floor and vkv_fit_slab_for_ctx_swap's pre-swap target, so both agree on "small".
static long long vkv_kmin() {
    long long k_min = 4;
    if (k_min < 1) k_min = 1;
    return k_min;
}

// FAIL CLOSED: the fit is a prediction, this is the verification. WDDM never fails an allocation, it
// demotes to system RAM, so a NON_LOCAL growth at the budget wall means the K just built does not fit:
// step K down, never run a slab over PCIe. Bounded: a step that buys nothing means the growth was not
// ours (demoted pages return lazily), so stop. (history: docs/code-comments-archive-2026-10-05.md, "spill-guard")
long long nonlocal_boot_mib() { return g_nonlocal_boot_mib; }

static void vkv_spill_guard(long long nl_before, long long per_k_mib) {
    if (nl_before < 0 || per_k_mib <= 0) return;          // no DXGI: nothing to verify against
    // (round 2): growth on the NON_LOCAL segment is NOT by itself evidence of a spill,
    // and an earlier version of this guard that assumed it was would have thrown away experts for
    // nothing. Measured on the reference box: the segment goes 74 -> 830 MiB during the model load
    // while LOCAL usage sits at 3442 of a 7180 MiB budget — less than half. That memory is host
    // memory the CUDA driver registers for the mapped model, not demoted VRAM; nothing is spilling.
    // A demotion can only happen when LOCAL usage is AT the budget, so that is the precondition:
    // no budget pressure, no guard, whatever the shared segment does.
    long long budget = 0, usage = 0;
    if (!mem_partition::wddm_local(&budget, &usage) || budget <= 0) return;
    if (usage < budget - 128) return;                     // not at the wall: the growth is not ours
    const long long tol_mib  = 32;
    const long long k_min    = vkv_kmin();
    long long       nl_prev  = nl_before;
    for (int attempt = 0; attempt < 3; ++attempt) {
        const long long nl_now = mem_partition::wddm_nonlocal_usage_mib();
        if (nl_now < 0) return;
        if (nl_now <= nl_before + tol_mib) {
            if (attempt > 0)
                std::cerr << "moe_dual: SPILL GUARD: cleared at K=" << g.k << " (shared segment back to "
                          << nl_now << " MiB, baseline " << nl_before << ")\n";
            return;
        }
        if (attempt > 0 && nl_now >= nl_prev) {
            std::cerr << "moe_dual: SPILL GUARD: stepping down stopped helping (" << nl_now
                      << " MiB shared, was " << nl_prev << ") — the growth is not ours to reclaim; "
                      << "holding K=" << g.k << ". Close other GPU apps or lower -VramBudget.\n";
            return;
        }
        const long long over  = nl_now - nl_before;
        long long       k_new = g.k - std::max<long long>(1, (over + per_k_mib - 1) / per_k_mib);
        if (k_new < k_min) k_new = k_min;
        if (k_new >= g.k)  k_new = g.k - 1;
        if (k_new < 1) {
            std::cerr << "moe_dual: SPILL GUARD: " << over << " MiB spilled to system RAM and K is "
                         "already at the floor — cannot shrink further\n";
            return;
        }
        std::cerr << "moe_dual: SPILL GUARD: " << over << " MiB of this process's VRAM is on the WDDM "
                     "SHARED segment (" << nl_now << " vs " << nl_before << " MiB before the rebuild) — "
                     "the fit overcommitted; stepping K " << g.k << " -> " << k_new << " rather than "
                     "running experts over PCIe\n";
        nl_prev = nl_now;
        if (!slab_rebuild_at_k((int) k_new)) return;      // refused (q4stream armed / no runtime)
    }
}

// Re-fit K to the MEASURED VRAM situation: grow or shrink the hot slab so that `keep_free_mib`
// stays free afterwards. This is the "K always max for the current ctx" half of variable-KV, and
// it is deliberately measurement-driven rather than modelled — so it self-corrects around whatever
// the context actually costs.
// Hysteresis: a delta of a single slot is ignored (never thrash for ~100 MiB). Returns the K in
// force after the call (unchanged on no-op/failure); 0 if there is no dynamic runtime.
int vkv_refit_k(long long keep_free_mib) {
    if (!g.dynamic || !g.mm || !g.pol) return 0;
    const uint64_t per_k = vkv_per_k_bytes();
    if (per_k == 0) return g.k;
    size_t free_b = 0, total_b = 0;
    if (!pool_dev_memory(&free_b, &total_b)) return g.k;
    const long long free_mib  = (long long) (free_b >> 20);
    const long long per_k_mib = (long long) (per_k >> 20);
    if (per_k_mib <= 0) return g.k;
    // The decision lives in src/kfit_policy.hpp (test-kfit-policy); here is what reads or moves the
    // machine. RESIDENT: the slab's bytes are inside `free`; EVICTED (an image's transient window): not.
    // (history: docs/code-comments-archive-2026-10-05.md, "refit-regimes")
    const bool evicted = slab_evicted();
    kfit::inputs in;
    in.free_mib      = free_mib;
    in.slab_mib_now  = evicted ? 0 : (long long) (g.slab_total_bytes >> 20);
    in.headroom_mib  = mem_partition::wddm_local_headroom_mib();
    in.per_k_mib     = per_k_mib;
    in.keep_free_mib = keep_free_mib;
    in.k_cur         = g.k;
    in.k_min         = (int) vkv_kmin();
    in.n_expert      = g.n_expert;
    in.evicted       = evicted;

    const kfit::decision fit = kfit::fit(in);

    if (fit.wddm_bound) {
        std::cerr << "moe_dual: vkv refit: WDDM budget is the binding limit (headroom "
                  << in.headroom_mib << " MiB < " << free_mib << " MiB device-free) — fitting K to "
                  << fit.pool_mib << " MiB, not " << (in.slab_mib_now + free_mib) << "\n";
    }
    if (fit.floor_capped) {
        std::cerr << "moe_dual: vkv refit: quality floor K=" << in.k_min << " does not fit ("
                  << free_mib << " MiB free, " << per_k_mib << " MiB/K) — capping to what "
                  << "physically fits, K=" << fit.k << " (degraded, not overcommitted)\n";
    }
    if (!fit.rebuild) {
        // claude dump findings 292: the loan is armed only by a rebuild, and a rebuild only ran when K changed. Where K
        // cannot grow (quality at 5 GiB: Q4 experts, the Q8 non-experts and a 589 MiB staging reserve leave no room) the
        // reserve then sat idle for the whole decode. It costs no VRAM: rebuild at the same K to lend it.
        if (!evicted && !g.q4s_on && loan_slots_for(g.k) != g.loan) {
            std::cerr << "moe_dual: vkv refit K " << g.k << " unchanged -- rebuilt to lend the staging reserve ("
                      << loan_slots_for(g.k) << " slots per layer)\n";
            slab_rebuild_at_k(g.k);
        }
        return g.k;
    }

    const long long k_new = fit.k;
    std::cerr << "moe_dual: vkv refit K " << g.k << " -> " << k_new << " (pool " << fit.pool_mib
              << " MiB = " << in.slab_mib_now << " slab + " << free_mib << " free, target free "
              << keep_free_mib << ", " << per_k_mib << " MiB per K, slab "
              << (evicted ? "EVICTED (restore at the new K)" : "resident") << ")\n";
    const long long nl_before = mem_partition::wddm_nonlocal_usage_mib();
    slab_rebuild_at_k((int) k_new);
    vkv_spill_guard(nl_before, per_k_mib);
    return g.k;
}

// THE invariant of a mid-session context swap: graph_reserve's token graph (n==1) reaches the hook only
// through the RESIDENT slab (the q4stream bracket guards only the prompt graph); with the slab out the
// reserve inflates to ~2.7 GiB of permanent buffer (ctx-independent, ~25 K per resize). So the slab is
// made SMALL BUT RESIDENT, never evicted: resident above the target -> shrink; evicted -> rebuild at the
// target (a restore); else nothing. Must run BEFORE q4stream arms. `slab_budget_mib` is the SWAP-PEAK
// budget (slab + staging + new context, measured by the host); < 0 = collapse to the K floor. The
// shrink removes exactly K_old - K_new experts by rank; moe_ctx_rebuild_done's refit sets the final K.
// (history: docs/code-comments-archive-2026-10-05.md, "fit-for-swap")
bool vkv_fit_slab_for_ctx_swap(long long slab_budget_mib) {
    if (!g.dynamic || !g.mm || !g.pol) return true;   // no dynamic runtime — nothing to fit
    if (g.q4s_on) return false;                        // must run before the bracket arms
    const long long k_min   = vkv_kmin();
    const bool      evicted = slab_evicted();
    const uint64_t  per_k   = vkv_per_k_bytes();
    const long long per_k_mib = per_k ? (long long) (per_k >> 20) : 0;

    long long k_target = k_min;
    const char * how   = "floor (no budget measurement available)";
    if (slab_budget_mib >= 0 && per_k_mib > 0) {
        k_target = slab_budget_mib / per_k_mib;
        how      = "measured budget";
        if (k_target < k_min)      k_target = k_min;   // quality floor; the post-swap refit caps
        if (k_target > g.k)        k_target = g.k;     // a swap never GROWS the slab (peak safety)
        if (k_target > g.n_expert) k_target = g.n_expert;
    }

    // Nothing to do only when the slab is resident AND already at/below the target. An evicted
    // slab must always be rebuilt here: residency is the invariant, not the size.
    if (!evicted && g.k <= (int) k_target) {
        std::cerr << "moe_dual: vkv ctx-swap fit: K " << g.k << " already fits the swap budget ("
                  << slab_budget_mib << " MiB -> K<=" << k_target << ") — NO shrink, the hot set "
                     "survives this resize untouched\n";
        return true;
    }
    std::cerr << "moe_dual: vkv ctx-swap fit: K " << g.k << " -> " << k_target << " (" << how
              << ", budget " << slab_budget_mib << " MiB, " << per_k_mib << " MiB per K), slab "
              << (evicted ? "EVICTED -> restoring at the new K" : "resident -> shrinking")
              << " — removing exactly " << (evicted ? 0 : (long long) g.k - k_target)
              << " experts/layer by the LRU rank; the new context's token-generation reserve needs "
                 "a RESIDENT slab or it falls to the stock shell path (~2.7 GiB permanent)\n";
    return slab_rebuild_at_k((int) k_target);
}

// The ONE function every "the slab is evicted, bring it back" call site should use. When the
// policy is armed, this is where "massimizza da SUBITO il numero di K possibili" actually happens
// restoring at a MEASURED K instead of whatever static value the slab last held (the boot
// static clamp, or a prior tier's K). When not armed, falls back to the plain slab_restore(),
// byte-identical to pre-variable-KV behavior. Returns true if it did something (matches
// slab_restore's return convention loosely; check slab_evicted() after if precision matters).
bool vkv_restore_or_refit() {
    // Armed: refit whether or not the slab is evicted -- vkv_refit_k handles both regimes (absolute fit
    // when evicted, delta fit when resident) with its own hysteresis.
    if (g_vkv_armed) {
        vkv_refit_k(g_vkv_boot_keep_mib);
        return !slab_evicted();
    }
    if (!slab_evicted()) return true;
    return slab_restore();
}

// --- variable-KV S6a: rebuild the hot slab at a NEW uniform K at runtime -----------------------
// Quiescent-point operation (call between turns, no in-flight graph): the policy ranks each
// layer's residents by its own blended eviction score, keeps the top-K_new (learned scores and
// clocks preserved — only the page table is rebuilt), the memmgr re-layouts the hot slab WITHOUT
// touching the K-independent cold arena (reset_hot_layout), the K-dependent stack (pool tensors +
// slab + aux mirrors) is rebuilt by the same helpers init used, and every surviving resident is
// re-streamed into its new compacted slot (the slab_restore contract). Cost = one slab realloc +
// K_new×n_layers expert reads from the promotion source (~2-4 s at product K).
bool slab_rebuild_at_k(int k_new) {
    if (!g.dynamic || !g.mm || !g.pol) {
        std::cerr << "moe_dual: rebuild_k: no dynamic runtime loaded\n";
        return false;
    }
    if (k_new < 1 || k_new > g.n_expert) {
        std::cerr << "moe_dual: rebuild_k: K=" << k_new << " out of range [1," << g.n_expert << "]\n";
        return false;
    }
    // A rebuild while the slab is EVICTED is legal and is exactly a RESTORE AT A NEW K: the
    // layout is rebuilt and the slab re-allocated below, so the evicted flag is cleared at the
    // end. This is the path variable-KV uses when q4stream staging pushed the slab out.
    const bool was_evicted = slab_evicted();
    if (g.q4s_on) {
        std::cerr << "moe_dual: rebuild_k: refused mid-prefill (q4stream active)\n";
        return false;
    }
    moe_xfer::wait_idle();   // the resize below re-lays every slot: no copy may still be in flight
    // no-op — but ONLY when the slab is actually resident at that K.: while EVICTED
    // this call IS the restore, so the old guard (which ignored was_evicted) silently skipped it.
    // Measured live in the multi-tier sweep: "vkv refit K 1 -> 1 ... slab EVICTED (absolute fit +
    // restore)" printed, no "hot slab REBUILT" line followed, and the slab stayed out with ZERO
    // expert VRAM for the whole next turn. vkv_refit_k deliberately lets the equal-K evicted case
    // through to here (`if (k_new == g.k && !evicted) return g.k;`) precisely to get the restore.
    // claude dump findings 292: an equal-K rebuild still runs when it changes the loan (the first one arms it)
    if (k_new == g.k && !was_evicted && loan_slots_for(k_new) == g.loan) return true;

    const auto t0 = std::chrono::steady_clock::now();
    const int k_old = g.k;
    ++g.slab_rebuilds;
    if (k_new != k_old) ++g.k_changes;
    ggml_backend_synchronize(g.backend);

    // 1) policy: rank + evict the residents that no longer fit + compact survivors (scores kept)
    early_join();
    const int loan_new = loan_slots_for(k_new);   // findings 246: slots lent by the idle staging reserve
    if (!moert_policy_resize_k(g.pol, k_new + loan_new)) {
        std::cerr << "moe_dual: rebuild_k: policy resize failed\n";
        return false;
    }

    // 2) release the old K-dependent stack. reset_hot_layout frees the slab through cb_slab_free
    //    and clears the finalize outputs while keeping pages/sources/cold-arena intact (never a
    //    cold-arena reload). The old pool tensors + slab byte-view die with pack_ctx.
    if (!moert_memmgr_reset_hot_layout(g.mm)) {
        std::cerr << "moe_dual: rebuild_k: memmgr reset_hot_layout failed\n";
        return false;
    }
    if (g.aux_buf)  { ggml_backend_buffer_free(g.aux_buf); g.aux_buf = nullptr; }
    if (g.aux_ctx)  { ggml_free(g.aux_ctx); g.aux_ctx = nullptr; }
    if (g.pack_ctx) { ggml_free(g.pack_ctx); g.pack_ctx = nullptr; }

    // 3) re-key + rebuild the stack at K_new through the SAME helpers init used
    g.k = k_new;
    g.loan = loan_new;
    g.loan_live = loan_new > 0 && !g.q4s_on;
    g.pcfg.k = k_new + loan_new;
    for (int L = 0; L < g.n_layers; ++L) {
        if (!moert_memmgr_set_layer_k(g.mm, L, k_new + loan_new)) {
            std::cerr << "moe_dual: rebuild_k: memmgr rejected K=" << k_new << " for layer " << L << "\n";
            return false;
        }
    }
    moert_memmgr_set_loan(g.mm, loan_new);
    if (!build_pool_tensors(nullptr) || !alloc_and_place_slab() || !build_aux_mirrors()) {
        std::cerr << "moe_dual: CRITICAL: rebuild_k(" << k_new << ") failed mid-rebuild — "
                     "hot tier is unusable; restart the runtime\n";
        return false;
    }

    // 4) re-stream every surviving resident into its NEW compacted slot (policy page table =
    //    ground truth, exactly slab_restore's replay) and publish every layer's mirrors.
    loan_tables_build();
    if (g.loan > 0 && !g.loan_live) moert_policy_set_active_k(g.pol, g.k);
    if (g.loan > 0)
        std::cerr << "moe_dual: loan: " << g.loan << " more hot slots per layer in the staging reserve ("
                  << (moert_memmgr_loan_bytes(g.mm) >> 20) << " MiB), K " << g.k << " + " << g.loan << "\n";
    int n_ok = 0, n_fail = 0;
    for (int L = 0; L < g.n_layers; ++L) {
        const int32_t * se = moert_policy_slot_exp_view(g.pol, L);
        for (int s = 0; s < k_new + loan_new && se; ++s) {
            if (se[s] < 0) continue;
            if (moert_memmgr_promote(g.mm, L, se[s], s)) ++n_ok; else ++n_fail;
        }
        push_residency(L);
    }
    if (was_evicted) {
        slab_set_evicted(false);   // the rebuild re-allocated the slab = a restore
    }
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::cerr << "moe_dual: hot slab REBUILT K=" << k_old << " -> " << k_new << " (" << g.slab_mib
              << " MiB): " << n_ok << " residents re-streamed"
              << (n_fail ? (" (" + std::to_string(n_fail) + " FAILED — those slots are zeroed)") : std::string())
              << " in " << secs << " s\n";
    llapis_phase("Hot expert cache rebuilt at K=%d (was %d): %lld MiB VRAM, %d residents re-streamed in %.1f s",
                 k_new, k_old, g.slab_mib, n_ok, secs);
    return n_fail == 0;
}

}  // namespace moe_dual
