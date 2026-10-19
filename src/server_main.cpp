// ===========================================================================
// moert-server : the product server.
// ===========================================================================
// llama-server (the fork's llama-server-impl library: HTTP routes, OpenAI-compat
// API, slots) with the moe_dual dual-precision expert runtime wired in:
//   - env-driven init BEFORE the model loads (the launcher's resolved
//     configuration registers the expert hook; unset => a stock llama-server)
//   - per-phase trampolines on the server decode loop (server-moe-hooks.h):
//     prompt-chunk fold / prompt-end refresh / per-token upkeep / turn-end refresh
//   - guardrails injected on the command line when the dual runtime is active:
//     -np 1 (single slot: the cache premise is a per-SESSION hot-set; parallel
//     slots would interleave routing streams), --no-warmup (a warmup forward
//     would pollute routing scores), --ubatch-size 1 under the loader cut
//     (graph_reserve must build the lean fused path).
//
// Routing is observed with ON-DEVICE counters, not a host eval-callback: this
// process does not own the llama_context params that would install one. The
// q4stream streamed prefill is wired here instead, against the same counters.
// ===========================================================================

#include "moe_dual.hpp"
#include "mem_partition.hpp"
#include "vkv_policy.hpp"     // variable-KV capacity/resizing policy (the single place it lives)
#include "prefill_policy.hpp" // streamed-vs-fused prefill gate (ditto)
#include "q4s_phase_policy.hpp" // when a batch may run with the q4stream phase ARMED (ditto)
#include "model_plan.hpp"
#include "gguf_tools.hpp"
#include "moert/memmgr.h"   // prepare: the store layout check

#include "llama.h"
#include "gguf.h"
#include "ggml.h"

#include "server-moe-hooks.h"
#include "mtmd.h"             // the lazy projector's VRAM callback (findings 257)
#include "llapis_log_filter.hpp"  // C0: normal-mode quiet + full raw log

#include <nlohmann/json.hpp>   // fork-vendored; moert-profiles.json parsing (M4 lifecycle)
#include <cpp-httplib/httplib.h>   // fork-vendored; child->router unload request (M4 watchdog)

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <dbghelp.h>
#pragma comment(lib, "dbghelp.lib")
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

int llama_server(int argc, char ** argv);                 // llama-server-impl (server.cpp)
void llama_server_set_post_common_init(void (*)());       // C0: re-assert the log filter post common_init

// Decode-start snapshot for the turn-end parity print: run_turn rebaselines its route
// counters right after the prompt-end refresh (visible-decode-only hit-rate); mirror that
// so the [moert-server] turn line is field-comparable with the bench JSON.
static moe_dual::RuntimeInfo g_dec0;

// C5 phase-tracing turn state (single slot, -np 1): first-prompt-chunk detection,
// prompt token tally, and the last staged-refresh count seen by the token hook.
static bool      g_prompt_started    = false;
static long long g_prompt_tok        = 0;
static long long g_last_refresh_seen = 0;
// emitted-token baseline at the last refresh/prompt-end — the phase lines' tok/s counts
// EMITTED tokens (incl. accepted MTP drafts), not decode cycles.
static long long g_ref_gen_tok = 0;
static long long g_ref_gen_ms  = 0;
// per-token cost-aware regime (one refresh per token): the refresh line is printed once per
// REFRESH_LINE_EVERY visible tokens and sums the refreshes since the last line
static constexpr long long REFRESH_LINE_EVERY = 64;
static long long g_ref_line_vt     = 0;
static long long g_ref_line_promos = 0;
static double    g_ref_line_pause  = 0.0;

// llama handles for read-only observability probes (M2 memory partition) — delivered by
// the fork's g_srv_moe_on_ctx sink right after load_model; never used to mutate.
static llama_context     * g_obs_ctx   = nullptr;
static const llama_model * g_obs_model = nullptr;
// the speculative side, delivered by the g_srv_moe_on_spec_ctx sink (which fires
// after the drafter is built — it cannot come through the sink above, that one necessarily runs
// before the drafter exists). Re-delivered on every ctx rebuild, including as nullptr when the
// rebuild drops speculative decoding, so these can never dangle onto a freed context.
static llama_context     * g_obs_ctx_dft   = nullptr;
static const llama_model * g_obs_model_dft = nullptr;

// Cached partition snapshot. mem_partition::probe touches the live llama_context (KV size),
// which is NOT safe from the HTTP thread while a slot decodes — so the SLOT thread refreshes
// this cache (post-load, then every N visible tokens, then at turn end) and the JSON provider
// only reads it. Field-level tearing across a concurrent read is possible but each value is
// an aligned 64-bit store (per-field plausible) — acceptable for telemetry.
static mem_partition::Partition g_part;
static long long g_part_tok = 0;   // visible-token count at the last refresh
static int       g_obs_n_ctx = 0;  // context size, cached at the ctx sink for /moe/info
// Prompt-end -> first-token window. The server stops the prompt clock before this
// work and starts the decode clock after it, so `timings.prompt_ms` reports a TTFT that excludes it
// even though the user is waiting. Written once per turn by the prompt-end hook, read by /moe/info.
// Atomics because the HTTP thread reads them while the slot thread writes.
static std::atomic<double> g_pe_slab_ms{0.0};      // vkv restore / measured refit of the hot slab
static std::atomic<double> g_pe_refresh_ms{0.0};   // prompt-end residency refresh
static std::atomic<double> g_pe_total_ms{0.0};     // the whole window, including the q4stream close

// VRAM-overcommit sentinel: flags (grep [VRAM-OVERCOMMIT]) when device-wide free VRAM drops below a
// small floor -- dedicated (LOCAL) headroom is what precedes a spill into the WDDM shared segment; the
// shared segment's own size is NOT a spill signal (it tracks host memory the driver registers).
// One flag per further 64 MiB step down. (history: docs/code-comments-archive-2026-10-05.md, "vram-sentinel")
static long long g_low_vram_flagged_mib = -1;
// C5.6: this process's WDDM shared-segment usage at boot, captured once at on_ready. It is the
// only number against which a later reading means anything (see the capture site).
static std::atomic<long long> g_vram_shared_baseline_mib{-1};

static void check_vram_overcommit() {
    if (!g_part.vram_valid) return;
    long long floor_mib = 150;
    if (floor_mib <= 0) return;
    const long long free_mib = g_part.vram_free;
    if (free_mib > floor_mib) { g_low_vram_flagged_mib = -1; return; }   // recovered — re-arm
    if (g_low_vram_flagged_mib >= 0 && free_mib >= g_low_vram_flagged_mib - 64) return;   // already flagged this deep
    g_low_vram_flagged_mib = free_mib;
    std::fprintf(stderr, "[VRAM-OVERCOMMIT] dedicated VRAM free is critically low: %lld MiB (floor %lld) — "
                         "further allocations have nowhere dedicated left to go and may silently spill "
                         "into system RAM (this process's WDDM shared-segment usage right now: %lld MiB — "
                         "ground truth for whether it actually did)\n",
                         free_mib, floor_mib, g_part.vram_shared_ours);
    std::fflush(stderr);
    llapis_phase("WARNING: dedicated VRAM free critically low (%lld MiB) — allocations may spill to "
                 "system RAM", free_mib);
}

static void refresh_partition() {
    g_part = mem_partition::probe(g_obs_ctx, g_obs_model, g_obs_ctx_dft, g_obs_model_dft);
    check_vram_overcommit();
}

// Keep the cold arena in this process's working set (findings 192). The first turn after boot was
// slow because of SOFT page faults: Windows trims the arena out of the working set during the model
// load and while the server idles (pages stay in RAM, ~5-6 us to fault each back), so the first decode
// re-faulted ~1.2 M of its 2 M pages. Two parts: a minimum working set of arena + 0.5 GiB (was 1.5: the box stalled at zero free RAM, findings 196), so the
// trimmer leaves it alone (other processes pay instead -- a deliberate trade, speed first), and a
// background pass that faults the arena in while the server waits for its first prompt.
// LLAPIS_ARENA_RESIDENT: soft (the default: a minimum the trimmer honours unless the system is in real
// trouble), hard (QUOTA_LIMITS_HARDWS_MIN_ENABLE: never trimmed below it), 0 = off. Gated A,S,H,H,S,A in
// both modes (findings 193): turn-1 decode 21 -> 36 tok/s (quality), 28 -> 50 (speed), speed TTFT
// -2..3 s on every turn; soft and hard measured equal, but hard made back-to-back server restarts fail
// to connect (the box at zero free RAM), so soft ships. Skipped when physical RAM cannot hold the arena
// plus 6 GiB for everything else.
#ifdef _WIN32
// [llapis] findings 211: a RAM floor under the arena hold. With the arena held in the working set the box
// fell to 30-230 MiB of available RAM when other processes grew (3 days of uptime: a 1.8 GiB nonpaged
// pool, ~7 GiB "other processes"), and at that point WDDM could not make the GPU's allocations resident:
// cudaErrorMemoryAllocation at a stream sync, the child crashed (ffn_moe.cu:211/533, 2026-10-09). The
// hold now yields: the boot touch stops at the floor, and a guard lowers the working-set minimum whenever
// available RAM falls under it, so Windows can trim arena pages (file-backed, they fault back) instead of
// starving the driver. LLAPIS_RAM_FLOOR_MIB (default 640: the crashes came at 72-226 MiB; 1536 released the whole hold on a box with ~1 GiB to spare, findings 211; 0 = no floor).
static long long ram_avail_mib() {
    MEMORYSTATUSEX ms; ms.dwLength = sizeof(ms);
    return GlobalMemoryStatusEx(&ms) ? (long long) (ms.ullAvailPhys >> 20) : -1;
}
static long long ram_floor_mib() {
    static const long long f = [] { const char * e = std::getenv("LLAPIS_RAM_FLOOR_MIB"); return e && e[0] ? std::atoll(e) : 640LL; }();
    return f;
}
static std::atomic<unsigned long long> g_ws_min{0};   // current working-set minimum we asked for (bytes), 0 = none
static SIZE_T g_ws_max = 0;
static DWORD  g_ws_flags = 0;
static bool set_ws_min(unsigned long long mn) {
    const BOOL ok = SetProcessWorkingSetSizeEx(GetCurrentProcess(), (SIZE_T) mn, g_ws_max, g_ws_flags);
    if (ok) g_ws_min.store(mn);
    return ok != 0;
}
static void ram_floor_guard_start() {
    static std::atomic<bool> started{false};
    if (ram_floor_mib() <= 0 || started.exchange(true)) return;
    std::thread([]() {
        long long releases = 0;
        for (;;) {
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            const unsigned long long mn = g_ws_min.load();
            if (!mn) continue;
            const long long avail = ram_avail_mib(), floor = ram_floor_mib();
            if (avail < 0 || avail >= floor) continue;
            const unsigned long long give = (unsigned long long) (floor - avail + 512) << 20;
            const unsigned long long keep = mn > give + (256ull << 20) ? mn - give : 0;
            if (keep ? set_ws_min(keep) : set_ws_min(256ull << 20)) {
                if (!keep) g_ws_min.store(0);   // the hold is gone; Windows manages the arena like any mapping
                if (releases++ < 20)
                    std::fprintf(stderr, "[moert-server] RAM floor: %lld MiB available < %lld -- arena hold lowered to %llu MiB\n",
                                 avail, floor, keep >> 20);
            }
        }
    }).detach();
}
#endif

static void arena_ws_boot() {
#ifdef _WIN32
    const char * env_mode = std::getenv("LLAPIS_ARENA_RESIDENT");
    const char * mode = (env_mode && env_mode[0]) ? env_mode : "soft";
    if (std::strcmp(mode, "0") == 0) return;
    const bool hard = std::strcmp(mode, "hard") == 0;
    const unsigned long long arena_mib = (unsigned long long) moe_dual::cold_arena_committed_mib();
    MEMORYSTATUSEX ms; ms.dwLength = sizeof(ms);
    if (!arena_mib || !GlobalMemoryStatusEx(&ms)) return;
    const unsigned long long phys_mib = ms.ullTotalPhys >> 20;
    if (phys_mib < arena_mib + 6144) {
        std::fprintf(stderr, "[moert-server] arena residency: skipped (RAM %llu MiB < arena %llu + 6144)\n", phys_mib, arena_mib);
        return;
    }
    const SIZE_T mn = (SIZE_T) (arena_mib + 512) << 20, mx = mn + ((SIZE_T) 4096 << 20);
    const DWORD flags = (hard ? QUOTA_LIMITS_HARDWS_MIN_ENABLE : QUOTA_LIMITS_HARDWS_MIN_DISABLE) | QUOTA_LIMITS_HARDWS_MAX_DISABLE;
    g_ws_max = mx; g_ws_flags = flags;
    const BOOL ok = set_ws_min(mn) ? TRUE : FALSE;
    std::fprintf(stderr, "[moert-server] arena residency: working set minimum %llu MiB (%s): %s (err %lu)\n",
                 (unsigned long long) (mn >> 20), hard ? "hard" : "soft", ok ? "set" : "FAILED", ok ? 0ul : GetLastError());
    if (!ok) return;
    std::vector<std::pair<const volatile uint8_t *, size_t>> blocks;
    for (int i = 0, n = moe_dual::cold_arena_block_count(); i < n; ++i) {
        void * base = nullptr; unsigned long long bytes = 0;
        if (moe_dual::cold_arena_block(i, &base, &bytes) && base && bytes)
            blocks.push_back({ (const volatile uint8_t *) base, (size_t) bytes });
    }
    // Blocks dealt round-robin; a fault on resident-but-trimmed memory is kernel CPU work, so it
    // parallelizes. [llapis] findings 211: done BEFORE the server announces itself ready (this runs in
    // on_ready, ahead of the router notification): a prompt sent at «ready» used to decode at 25-34
    // instead of ~45 tok/s in speed while it faulted the arena itself. 8 normal-priority threads, since
    // nothing else runs yet. LLAPIS_ARENA_TOUCH=async restores the background pass (4 below-normal threads).
    const char * tm = std::getenv("LLAPIS_ARENA_TOUCH");
    const bool async_touch = tm && std::strcmp(tm, "async") == 0;
    ram_floor_guard_start();
    std::thread toucher([blocks = std::move(blocks), async_touch]() {
        const auto t0 = std::chrono::steady_clock::now();
        const int nt = async_touch ? 4 : 8;
        std::atomic<unsigned long long> pages{0};
        std::atomic<bool> floor_hit{false};
        std::vector<std::thread> ws;
        for (int t = 0; t < nt; ++t) ws.emplace_back([&, t]() {
            if (async_touch) SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
            unsigned long long n = 0; uint8_t sink = 0;
            for (size_t i = (size_t) t; i < blocks.size() && !floor_hit.load(); i += nt)
                for (size_t off = 0; off < blocks[i].second; off += 4096, ++n) {
                    sink ^= blocks[i].first[off];
                    if ((n & 4095) == 0 && ram_floor_mib() > 0 && ram_avail_mib() < ram_floor_mib()) { floor_hit.store(true); break; }
                }
            pages += n + (sink & 0);
        });
        for (auto & w : ws) w.join();
        std::fprintf(stderr, "[moert-server] arena residency: %llu pages faulted in, %.0f ms (%s)\n", pages.load(),
                     std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(),
                     async_touch ? "background" : "before ready");
        if (floor_hit.load())
            std::fprintf(stderr, "[moert-server] arena residency: stopped at the RAM floor (%lld MiB available < %lld); "
                                 "the rest of the arena faults in on use\n", ram_avail_mib(), ram_floor_mib());
    });
    if (async_touch) toucher.detach(); else toucher.join();
#endif
}

// full VRAM/RAM accounting at the moment the user asked for — the FIRST decode
// token of the session, with the hot slab already resident and re-streamed at Q4 (the prompt-end
// refresh runs just before the first token is generated; verified in's log:
// "vkv refit" -> "hot slab REBUILT" -> "Decoding started"). This is a formatted
// dump of mem_partition::Partition — no new probing mechanism, just every field named and shown
// together in one place, which nothing else in the runtime does.
static void dump_vram_breakdown_once() {
    static std::atomic<bool> done{false};
    if (done.exchange(true)) return;
    refresh_partition();   // one more probe right here so this dump reflects THIS instant, not a
                            // stale cached g_part from 16 tokens ago (the usual decode cadence)
    const mem_partition::Partition & p = g_part;
    const moe_dual::RuntimeInfo r = moe_dual::runtime_info();
    auto pct = [&](long long v) { return p.vram_total > 0 ? 100.0 * (double) v / (double) p.vram_total : 0.0; };
    // `ram_cold_arena` is the arena's committed size, not its touched pages, so at boot it can exceed RSS.
    // cuda_baseline: total - free_pre_slab - other processes = a measured floor for what a CUDA process
    // costs to exist; the residual beyond it is cuBLAS workspaces + the VMM pool + slack. (history: docs/code-comments-archive-2026-10-05.md, "cuda-baseline")
    long long cuda_baseline_mib = 0;
    if (p.vram_total > 0 && r.vram_free_pre_slab_mib > 0) {
        cuda_baseline_mib = p.vram_total - r.vram_free_pre_slab_mib - p.vram_other_procs;
        if (cuda_baseline_mib < 0) cuda_baseline_mib = 0;
        if (cuda_baseline_mib > p.vram_compute_other) cuda_baseline_mib = p.vram_compute_other;
    }
    const long long residual_rest = p.vram_compute_other - cuda_baseline_mib;
    // committed-but-idle scratch, expressed in hot-expert slots so the size of the
    // opportunity is legible next to the thing it competes with.
    const long long pool_idle_mib = p.vram_pool_reserved - p.vram_pool_used;
    const long long per_k_mib     = (long long) (moe_dual::vkv_per_k_bytes() >> 20);
    const bool arena_exceeds_rss = p.ram_cold_arena > p.ram_rss;
    char nonarena_str[32];
    if (arena_exceeds_rss) std::snprintf(nonarena_str, sizeof(nonarena_str), "n/a");
    else                   std::snprintf(nonarena_str, sizeof(nonarena_str), "%lld", p.ram_nonarena_rss);
    std::fprintf(stderr,
        "\n"
        "[moert-server] ================ VRAM/RAM BREAKDOWN — first decode token, hot slab resident at Q4 ================\n"
        "[moert-server] context: n_ctx=%d  K(hot experts/layer)=%d/%d  cache fill=%.1f%%  layers=%d\n"
        "[moert-server] --- VRAM (device total %lld MiB) ---\n"
        "[moert-server]   non-expert model weights (NE, GPU-resident)  %6lld MiB  (%.1f%%)\n"
        "[moert-server]   hot Q4 expert slab (K=%d, this process)      %6lld MiB  (%.1f%%)\n"
        "[moert-server]   live KV cache (target ctx, current fill)     %6lld MiB  (%.1f%%)\n"
        "[moert-server]   VRAM L2 cold-cache slab                     %6lld MiB  (%.1f%%)  %s\n"
        "[moert-server]   prefill staging buffer                      %6lld MiB  (%.1f%%)  %s\n"
        "[moert-server]   target ctx compute/scheduler buffer         %6lld MiB  (%.1f%%)\n"
        "[moert-server]   MTP drafter weights (separate model)        %6lld MiB  (%.1f%%)  %s\n"
        "[moert-server]   MTP draft ctx compute/scheduler buffer      %6lld MiB  (%.1f%%)  %s\n"
        "[moert-server]   MTP draft KV                                     0 MiB  (0.0%%)  (aliases the\n"
        "[moert-server]     target's KV — Gemma4 shared_kv_layers=4, boot log \"layer 0: sharing with\n"
        "[moert-server]     layer 28\"; zero INCREMENTAL VRAM, already counted in the KV row above)\n"
        "[moert-server]   ggml-cuda scratch pools (%d live)           %6lld MiB  (%.1f%%)\n"
        "[moert-server]     of which handed out right now:            %6lld MiB\n"
        "[moert-server]     of which committed but IDLE:              %6lld MiB  (the VMM pool only\n"
        "[moert-server]       ever grows — it never returns this to the driver. This is the\n"
        "[moert-server]       reclaimable candidate: %.1f K slots at %lld MiB per K)\n"
        "[moert-server]   CUDA primary ctx + cuBLAS internal workspace\n"
        "[moert-server]     + allocator slack (RESIDUAL)              %6lld MiB  (%.1f%%)\n"
        "[moert-server]     of which CUDA/driver baseline, measured:  %6lld MiB  (device-wide usage\n"
        "[moert-server]       before the hot slab existed, minus other processes — the floor a CUDA\n"
        "[moert-server]       process pays just to exist on WDDM)\n"
        "[moert-server]     of which cuBLAS internal workspace + slack:%5lld MiB  (cuBLAS allocates\n"
        "[moert-server]       its own workspace per handle and exposes no query for it; this is now\n"
        "[moert-server]       the only unnamed block left)\n"
        "[moert-server]   ---------------------------------------------------------------\n"
        "[moert-server]   THIS PROCESS, total (DXGI dedicated-segment)  %6lld MiB  (%.1f%%)\n"
        "[moert-server]   other processes on this GPU (driver/desktop) %6lld MiB  (%.1f%%)\n"
        "[moert-server]   FREE (dedicated, driver view)                %6lld MiB  (%.1f%%)\n"
        "[moert-server]   ================================================================\n"
        "[moert-server]   device total                                 %6lld MiB  (100.0%%)\n"
        "[moert-server]   THIS PROCESS, WDDM SHARED (system-RAM) segment %5lld MiB  (ground truth for\n"
        "[moert-server]     overcommit — NOT part of the dedicated total above; a growing value there\n"
        "[moert-server]     independent of this snapshot means allocations are spilling into system RAM)\n"
        "[moert-server] --- RAM (system total %lld MiB) ---\n"
        "[moert-server]   this process, working set (RSS, physically resident pages)  %6lld MiB\n"
        "[moert-server]   cold IQ2 expert arena, OWNED/COMMITTED size    %6lld MiB%s\n"
        "[moert-server]   everything else in RSS (weights/RT/KV spill)   %6s MiB\n"
        "[moert-server]   other processes                                %6lld MiB\n"
        "[moert-server]   available (truly free)                        %6lld MiB\n"
        "[moert-server] =====================================================================================\n\n",
        g_obs_n_ctx, r.hot_k, r.n_expert, 100.0 * r.occupancy, r.n_layers,
        p.vram_total,
        p.vram_nonexpert, pct(p.vram_nonexpert),
        r.hot_k, p.vram_hot_slab, pct(p.vram_hot_slab),
        p.vram_kv, pct(p.vram_kv),
        p.vram_l2, pct(p.vram_l2), p.vram_l2 > 0 ? "" : "(off)",
        p.vram_staging, pct(p.vram_staging), p.vram_staging > 0 ? "" : "(off — decode phase, staging is prefill-only)",
        p.vram_compute_tgt, pct(p.vram_compute_tgt),
        p.vram_drafter_wts, pct(p.vram_drafter_wts),
        p.vram_drafter_wts > 0 ? "" : "(MTP runs against the target model — no second copy)",
        p.vram_compute_dft, pct(p.vram_compute_dft),
        p.spec_active ? "" : "(speculative decoding off)",
        p.vram_pool_count, p.vram_pool_reserved, pct(p.vram_pool_reserved),
        p.vram_pool_used, pool_idle_mib,
        per_k_mib > 0 ? (double) pool_idle_mib / (double) per_k_mib : 0.0, per_k_mib,
        p.vram_compute_other, pct(p.vram_compute_other),
        cuda_baseline_mib, residual_rest,
        p.vram_ours, pct(p.vram_ours),
        p.vram_other_procs, pct(p.vram_other_procs),
        p.vram_free, pct(p.vram_free),
        p.vram_total,
        p.vram_shared_ours,
        p.ram_total,
        p.ram_rss,
        p.ram_cold_arena,
        arena_exceeds_rss ? "  <- LARGER than RSS above: normal at boot, the arena's committed pages "
                             "are not all faulted-in yet (lazy page-in on first read)" : "",
        nonarena_str,
        p.ram_other_procs,
        p.ram_avail);
    std::fflush(stderr);
    llapis_phase("VRAM breakdown at first decode token: NE %lld / hot slab %lld (K=%d) / KV %lld / "
                 "compute tgt %lld + dft %lld / drafter %lld / CUDA ctx+slack %lld / free %lld of "
                 "%lld MiB",
                 p.vram_nonexpert, p.vram_hot_slab, r.hot_k, p.vram_kv, p.vram_compute_tgt,
                 p.vram_compute_dft, p.vram_drafter_wts, p.vram_compute_other,
                 p.vram_free, p.vram_total);
}

// --- Variable-KV Phase 2 (S6b) host side of the ctx-rebuild protocol (server-moe-hooks.h).
// The HTTP thread SCHEDULES (validated); the fork polls take() at the all-idle quiescent point
// of update_slots on the slot thread — take() also evicts the expert slab there (so the new,
// possibly larger KV + graph reserve never transiently coexists with the resident experts on
// the 8 GB box); done() restores it after the rebuild.
static void vkv_prepare_ctx_swap(int n_ctx_new);   // defined with the q4stream helpers below
extern std::string g_q4s_src;              // MOE_DUAL_PREFILL_Q4SRC; defined with the q4stream helpers below

static std::atomic<int> g_pending_rebuild_ctx{0};

static int moe_ctx_rebuild_schedule(int n_ctx, char * err, int errcap) {
    auto fail = [&](const char * m) {
        if (err && errcap > 0) std::snprintf(err, (size_t) errcap, "%s", m);
        return 0;
    };
    if (!g_obs_ctx)                          return fail("no context loaded yet");
    if (n_ctx < 2048 || n_ctx > 262144)      return fail("n_ctx out of range [2048, 262144]");
    g_pending_rebuild_ctx.store(n_ctx);
    std::fprintf(stderr, "[moert-server] ctx rebuild to n_ctx=%d SCHEDULED (applies at the next "
                         "all-idle quiescent point)\n", n_ctx);
    std::fflush(stderr);
    return 1;
}
static int moe_ctx_rebuild_take() {
    const int n = g_pending_rebuild_ctx.exchange(0);
    if (n > 0) {
        // The new context's graph_reserve must see the load-time conditions or its permanent compute buffer
        // explodes (~2.7 GiB): q4stream armed (prompt graph) and the slab resident (token graph) -- see
        // vkv_prepare_ctx_swap. (history: docs/code-comments-archive-2026-10-05.md, "rebuild-reserve")
        llapis_phase("Rebuilding context to %d tokens ... (KV saved; experts stay resident)", n);
    }
    return n;
}
// --- Variable-KV: the automatic rebalance policy -------------------------------------------------
// The capacity algorithm lives in src/vkv_policy.hpp; here is the wiring. K is re-fitted to the
// MEASURED post-rebuild VRAM exactly once, in moe_ctx_rebuild_done.
// keep_post: MiB of dedicated VRAM the refit leaves free -- a safety margin only (the prefill staging
// is a fixed reserve already resident when K is fitted). 250 from a measured ladder (500/350/250/150:
// no spill at any rung; 150 sits on the overcommit sentinel's floor and trips it every deep turn).
// (history: docs/code-comments-archive-2026-10-05.md, "keep-post")
static long long g_vkv_keep_post = 250;
static bool      g_vkv_enabled   = false;   // MOE_DUAL_VKV=1 arms the automatic policy
static vkv::Params g_vkv;

// round 2: the ONE input to the ctx-cost model that is measured rather than modelled.
// Sampled at turn start (vkv_plan_ctx, old context still alive) and read again in
// vkv_prepare_ctx_swap (fired by the fork right after the old context is torn down) — the
// difference IS that context's true footprint on this box, this build, right now. Only the DELTA
// to the new ctx size is then extrapolated (vkv::ctx_cost_predict_mib). -1 = no sample, the swap
// falls back to collapsing the slab to the quality floor.
static long long g_vkv_free_at_turn_start = -1;
static int       g_vkv_n_ctx_at_turn_start = 0;
// q4stream's staging buffer must be RESIDENT through the swap (it is what keeps the new context's
// prompt reserve graph lean,/), so the swap-peak budget has to make room for it. Probed
// from the runtime rather than hardcoded; 589 MiB on this model/config.
static long long g_vkv_staging_mib = 589;

static void vkv_init_from_env() {
    g_vkv = vkv::params_from_env();
    // Arming flag, set by the launcher. The LLAPIS_VKV / LLAPIS_VKV_KEEP_POST_MIB aliases that
    // used to be accepted here were never written by anything: the launcher wipes inherited
    // MOE_DUAL_* and then sets its own, so the canonical names already survive every hop.
    if (const char * v = std::getenv("MOE_DUAL_VKV"); v && v[0] == '1') g_vkv_enabled = true;
    if (const char * v = std::getenv("MOE_DUAL_VKV_KEEP_POST_MIB"); v && v[0]) g_vkv_keep_post = std::atoll(v);
    ///: arms the measured boot-K refit for server_main's own on-prompt-end restore
    // (below) — the one call site proven safe for it (quiescent, q4stream guaranteed off there).
    // The graph hook has its own SEPARATE lazy boot-deferred restore that intentionally stays on
    // the plain, unconditional slab_restore() — routing it through this same policy crashed
    // (0xC0000005, round 2); see the comment at that call site in moe_dual.cpp.
    moe_dual::vkv_set_boot_policy(g_vkv_enabled, g_vkv_keep_post);
    if (g_vkv_enabled)
        std::fprintf(stderr, "[moert-server] variable-KV policy ARMED: min_ctx=%d gran=%d headroom=%d "
                             "slack=%d max_ctx=%d (keep free post-resize: %lld MiB)\n",
                     g_vkv.min_ctx, g_vkv.gran, g_vkv.headroom, g_vkv.slack, g_vkv.max_ctx,
                     g_vkv_keep_post);
}

// Fired by the fork at task launch (quiescent). Returns the n_ctx this turn should run at, 0 = keep.
static int vkv_plan_ctx(int n_prompt_tokens, int n_predict, int n_ctx_cur) {
    if (!g_vkv_enabled || n_ctx_cur <= 0) return 0;
    // The OpenAI chat API re-sends the whole conversation, so the prompt IS the history.
    // `n_predict` is the request's max_tokens: a CEILING a harness repeats on every call, not a
    // prediction of this turn's length. vkv::Params::gen_reserve caps what it may contribute to
    // the plan; without that cap the provider JSON's declared maxTokens alone would decide the
    // context size, and a 65536 declaration would grow to the ceiling on the first request.
    const int max_gen = n_predict > 0 ? n_predict : 512;
    const int gen_planned = (g_vkv.gen_reserve > 0 && max_gen > g_vkv.gen_reserve) ? g_vkv.gen_reserve : max_gen;
    const int c_new = vkv::plan_grow(n_prompt_tokens, max_gen, n_ctx_cur, g_vkv);
    if (c_new <= 0) {
        // Report against the PLANNED generation, so the "beyond max_ctx" line means what it says
        // (charging the un-clamped ceiling here would warn on every request from a harness that
        // declares a large max_tokens, while nothing was actually refused).
        if ((long long) n_prompt_tokens + gen_planned + g_vkv.slack > g_vkv.max_ctx)
            std::fprintf(stderr, "[moert-server] variable-KV: turn needs %d+%d tokens — beyond max_ctx %d; "
                                 "the request will be truncated/refused by the server\n",
                         n_prompt_tokens, gen_planned, g_vkv.max_ctx);
        return 0;
    }
    std::fprintf(stderr, "[moert-server] variable-KV GROW: need %d prompt + %d gen (of %d requested) "
                         "> ctx %d -> %d (gran %d, headroom %d, gen reserve %d)\n",
                 n_prompt_tokens, gen_planned, max_gen, n_ctx_cur, c_new, g_vkv.gran, g_vkv.headroom,
                 g_vkv.gen_reserve);
    llapis_phase("Context too small for this turn (%d tokens) — growing %d -> %d and re-fitting the "
                 "expert cache", n_prompt_tokens, n_ctx_cur, c_new);
    // round 2: sample pre-teardown VRAM so the swap can MEASURE what the old context
    // is really worth on this box instead of modelling it (the part got wrong). Nothing between
    // here and the teardown allocates on the device — the KV serialization goes to host RAM.
    // Paired read in vkv_prepare_ctx_swap.
    g_vkv_free_at_turn_start  = moe_dual::pool_free_mib();
    g_vkv_n_ctx_at_turn_start = n_ctx_cur;
    return c_new;
}

static void moe_ctx_rebuild_done(int ok, int n_ctx) {
    // Close the reserve-only q4stream bracket vkv_prepare_ctx_swap opened — mirroring the
    // load path's ctx sink, which does the same right after graph_reserve is done.
    // Single-slot host: no real prompt phase can be concurrently using q4stream here. The staging
    // kept as the fixed reserve stays, and the refit below fits K around it; a lean arm is released.
    if (!g_q4s_src.empty()) {
        moe_dual::q4stream_end();
        if (!moe_dual::q4stream_keep_after_prompt()) moe_dual::q4stream_shutdown();
    }
    // Re-fit K to what the NEW context actually left free — the "K max at every instant" half of
    // variable-KV, and the reason the policy is robust to a context costing more than modelled.
    if (ok && g_vkv_enabled) {
        const int k_now = moe_dual::vkv_refit_k(g_vkv_keep_post);
        std::fprintf(stderr, "[moert-server] variable-KV: after ctx=%d, K re-fitted to %d\n", n_ctx, k_now);
    }
    llapis_phase("Context rebuild %s: n_ctx=%d", ok ? "DONE" : "FAILED", n_ctx);
    std::fprintf(stderr, "[moert-server] ctx rebuild %s (n_ctx=%d)\n", ok ? "done" : "FAILED", n_ctx);
    std::fflush(stderr);
}

static long long lifecycle_now_ms();   // defined with the M4 heartbeat below

// C5 (UI mem panel "static graph" fix): the slot thread refreshes g_part only every 16 visible
// tokens / at turn end, so at REST the panel froze on the last decode's snapshot (other apps'
// VRAM/RAM changes never showed). This timer re-probes the SYSTEM-level rows every 5 s via
// mem_partition::refresh_system — llama handles untouched (safe alongside a decoding slot);
// the llama-owned rows (KV, non-expert) stay from the last slot-thread probe, which is the
// only time they change anyway. Started once at on_ready.
static void start_mem_timer() {
    static std::atomic<bool> started{false};
    if (started.exchange(true)) return;
    std::thread([]() {
        for (;;) {
            std::this_thread::sleep_for(std::chrono::seconds(5));
            g_part = mem_partition::refresh_system(g_part);
            check_vram_overcommit();   // the sentinel also watches the idle 5 s cadence
        }
    }).detach();
}

// Display rate: EMITTED tokens (incl. accepted MTP drafts) over a fixed recent window. A ring of
// (t, cumulative tokens) samples, so every observer computes over the same window and 0 means exactly
// "nothing emitted in the window" (the old consume-the-interval version split the delta among
// pollers and fell to 0). 1500 ms rides over one MTP verify burst. (history: docs/code-comments-archive-2026-10-05.md, "gen-rate-ring")
static double moe_gen_rate_recent() {
    static constexpr long long WINDOW_MS = 1500;
    static constexpr int       CAP       = 64;
    struct Sample { long long ms; long long tok; };
    static std::mutex mtx;
    static Sample ring[CAP] = {};
    static int    n = 0, head = 0;
    std::lock_guard<std::mutex> lk(mtx);

    const long long now = lifecycle_now_ms();
    const long long tok = llama_server_moe_gen_tokens();
    // Coalesce bursts of near-simultaneous polls into one sample so a chatty observer cannot
    // evict the whole window; the ring then spans >= WINDOW_MS even at CAP samples.
    const int newest = n > 0 ? (head + CAP - 1) % CAP : -1;
    if (newest < 0 || now - ring[newest].ms >= 40) {
        ring[head] = Sample{ now, tok };
        head = (head + 1) % CAP;
        if (n < CAP) ++n;
    } else {
        ring[newest].tok = tok;   // same instant: keep the freshest count
    }
    if (n < 2) return 0.0;

    // Oldest sample still inside the window; if every sample is older (a long idle gap), fall
    // back to the oldest we have so a resumed stream reports immediately instead of after a
    // full window of silence.
    const long long cutoff = now - WINDOW_MS;
    int   base  = -1;
    for (int i = 0; i < n; ++i) {
        const Sample & s = ring[(head + CAP - n + i) % CAP];
        if (s.ms <= cutoff) base = (head + CAP - n + i) % CAP; else break;
    }
    if (base < 0) base = (head + CAP - n) % CAP;   // window is shorter than the ring's span

    // Trim leading idle: walk `base` past sample pairs that produced no tokens, so dt spans only time spent
    // generating (else a fresh turn read a third of its real rate for ~4 s). (history: docs/code-comments-archive-2026-10-05.md, "gen-rate-trim")
    {
        const int newest_ix = (head + CAP - 1) % CAP;
        while (base != newest_ix) {
            const int nxt = (base + 1) % CAP;
            if (ring[nxt].tok != ring[base].tok) break;   // tokens appeared in this sub-interval
            base = nxt;
        }
    }
    const double dt = (now - ring[base].ms) / 1000.0;
    if (dt <= 0.05) return 0.0;
    const long long dtok = tok - ring[base].tok;
    return dtok > 0 ? (double) dtok / dt : 0.0;
}

// C5 phase-telemetry MTP observer: the fork feeds every fresh-draft step's previous verify
// (accepted, proposed) to g_srv_moe_spec_draft_fn. We always register a thin wrapper so the
// phase lines (">>> Refresh ... MTP xx%") can report live acceptance; it returns the fixed
// max width (behavior-identical to the plain g_srv_moe_spec_draft_max cap) unless the
// adaptive controller below is explicitly enabled, in which case it delegates.
static std::atomic<long long> g_mtp_proposed{0}, g_mtp_accepted{0};
static int g_spec_fixed_w = 0;
// Verify width as configured (spec_k), copied out of main so the hook lambdas can stamp it onto
// the policy telemetry. 1 = no speculation. Read-only after startup.
static int g_spec_k_tel = 1;
void common_speculative_set_draft_h_hook(void (*fn)(int i, const float * h, int n));   // common/speculative.h
static int moe_spec_observe(int prev_accepted, int prev_proposed) {
    static const bool hooked = (common_speculative_set_draft_h_hook(&moe_dual::mtp_draft), true);   // findings 216k
    (void) hooked;
    if (prev_proposed > 0) {
        g_mtp_proposed.fetch_add(prev_proposed);
        g_mtp_accepted.fetch_add(prev_accepted);
    }
    return g_spec_fixed_w;
}

// Diagnostic (MOE_DIAG_ABORT_STACK=1): GGML_ABORT on Windows prints no backtrace (the ggml
// gdb/lldb path is POSIX-only), which made the layer-major hash abort unattributable. This
// callback symbolizes the aborting thread's stack via dbghelp before dying — dev aid only.
#ifdef _WIN32
static void moe_diag_abort_stack(const char * message) {
    std::fprintf(stderr, "%s\n[abort-stack] symbolizing...\n", message);
    void * frames[62];
    const USHORT n = CaptureStackBackTrace(0, 62, frames, nullptr);
    const HANDLE proc = GetCurrentProcess();
    SymSetOptions(SYMOPT_DEFERRED_LOADS | SYMOPT_UNDNAME | SYMOPT_LOAD_LINES);
    SymInitialize(proc, nullptr, TRUE);
    char buf[sizeof(SYMBOL_INFO) + 512];
    for (USHORT i = 0; i < n; ++i) {
        auto * sym = (SYMBOL_INFO *) buf;
        sym->SizeOfStruct = sizeof(SYMBOL_INFO);
        sym->MaxNameLen   = 512;
        DWORD64 disp64 = 0;
        DWORD   disp32 = 0;
        IMAGEHLP_LINE64 line; line.SizeOfStruct = sizeof(line);
        if (SymFromAddr(proc, (DWORD64) frames[i], &disp64, sym)) {
            if (SymGetLineFromAddr64(proc, (DWORD64) frames[i], &disp32, &line))
                std::fprintf(stderr, "[abort-stack] #%02u %s +0x%llx (%s:%lu)\n",
                             i, sym->Name, (unsigned long long) disp64, line.FileName, line.LineNumber);
            else
                std::fprintf(stderr, "[abort-stack] #%02u %s +0x%llx\n", i, sym->Name, (unsigned long long) disp64);
        } else {
            std::fprintf(stderr, "[abort-stack] #%02u %p\n", i, frames[i]);
        }
    }
    std::fflush(stderr);
}

// Diagnostic (same MOE_DIAG_ABORT_STACK=1 gate): a hardware access violation (0xC0000005 —
// e.g. reading a RAM-dedup-decommitted arena page) is an SEH exception, NOT a ggml abort, so the
// callback above never sees it. This vectored handler logs the faulting address + read/write + a
// symbolized stack, then lets the process die normally (CONTINUE_SEARCH). Dev aid only.
static LONG CALLBACK moe_diag_veh(EXCEPTION_POINTERS * ep) {
    if (!ep || !ep->ExceptionRecord || ep->ExceptionRecord->ExceptionCode != EXCEPTION_ACCESS_VIOLATION)
        return EXCEPTION_CONTINUE_SEARCH;
    const ULONG_PTR kind = ep->ExceptionRecord->ExceptionInformation[0];   // 0=read 1=write 8=exec
    const ULONG_PTR addr = ep->ExceptionRecord->ExceptionInformation[1];
    std::fprintf(stderr, "\n[veh] ACCESS VIOLATION on %s at 0x%llx\n",
                 kind == 0 ? "READ" : kind == 1 ? "WRITE" : "EXEC", (unsigned long long) addr);
    MEMORY_BASIC_INFORMATION mbi;
    if (VirtualQuery((LPCVOID) addr, &mbi, sizeof(mbi))) {
        const char * st = mbi.State == MEM_COMMIT ? "COMMIT" : mbi.State == MEM_RESERVE ? "RESERVE(decommitted)"
                        : mbi.State == MEM_FREE ? "FREE" : "?";
        std::fprintf(stderr, "[veh] page state=%s alloc_base=%p region_base=%p region_size=0x%llx protect=0x%lx\n",
                     st, mbi.AllocationBase, mbi.BaseAddress, (unsigned long long) mbi.RegionSize, (unsigned long) mbi.Protect);
    }
    std::fprintf(stderr, "[veh] symbolizing...\n");
    void * frames[62];
    const USHORT n = CaptureStackBackTrace(0, 62, frames, nullptr);
    const HANDLE proc = GetCurrentProcess();
    SymSetOptions(SYMOPT_DEFERRED_LOADS | SYMOPT_UNDNAME | SYMOPT_LOAD_LINES);
    SymInitialize(proc, nullptr, TRUE);
    char buf[sizeof(SYMBOL_INFO) + 512];
    for (USHORT i = 0; i < n; ++i) {
        auto * sym = (SYMBOL_INFO *) buf;
        sym->SizeOfStruct = sizeof(SYMBOL_INFO);
        sym->MaxNameLen   = 512;
        DWORD64 disp64 = 0;
        DWORD   disp32 = 0;
        IMAGEHLP_LINE64 line; line.SizeOfStruct = sizeof(line);
        if (SymFromAddr(proc, (DWORD64) frames[i], &disp64, sym)) {
            if (SymGetLineFromAddr64(proc, (DWORD64) frames[i], &disp32, &line))
                std::fprintf(stderr, "[veh] #%02u %s +0x%llx (%s:%lu)\n",
                             i, sym->Name, (unsigned long long) disp64, line.FileName, line.LineNumber);
            else
                std::fprintf(stderr, "[veh] #%02u %s +0x%llx\n", i, sym->Name, (unsigned long long) disp64);
        } else {
            std::fprintf(stderr, "[veh] #%02u %p\n", i, frames[i]);
        }
    }
    std::fflush(stderr);
    return EXCEPTION_CONTINUE_SEARCH;
}
#endif

// --- Q4Stream server prefill ---------------------------------------------------------------------
// moe_dual_q4stream.cpp wired into the server decode loop: a prompt streams the full-layer source
// (layer-major merged graph, the pool read once per prompt) through the fixed staging reserve; a
// staging-only eval callback gives the per-layer sync point; outside a streamed phase it asks for
// nothing. Armed through model load (lean) so graph_reserve stays on the staged tensors. A suffix
// streams when it is long or the slab is not warm (moe_q4s_prompt_cap); a short one on a warm slab
// runs fused. Requires the speculative path (backend sampling off). (history: docs/code-comments-archive-2026-10-05.md, "q4stream-server")
std::string g_q4s_src;                        // MOE_DUAL_PREFILL_Q4SRC (empty = q4stream off)
static bool        g_q4s_open   = false;      // inside a streamed prompt phase
static int         g_q4s_spec_k = 1;          // fused-path ceiling (verify width)
static int         g_q4s_prefill_ub = 0;      // prefill-only ubatch split (0 = tied to n_ubatch);
                                              // applied at the ctx sink, see the DECOUPLE block
static int         g_q4s_min    = 128;        // MOE_DUAL_PREFILL_MIN: min suffix tokens to stream when the
                                              // hot slab is RESIDENT (cold, the cap streams regardless).
                                              // The two regimes (findings 125): streamed ~3-4 s FLAT in n; fused ~25 ms/token warm (crossing ~128)
                                              // but ~418 ms/token cold -- no single threshold serves both, so the cap also consults the slab.
                                              // (history: docs/code-comments-archive-2026-10-05.md, "prefill-min")
static moe_dual::Q4StreamStats g_q4s_t0;      // stats snapshot at phase begin (for the turn line)
// P3 gate: hit over the first EARLY_TOKENS visible tokens of each turn (set from the token-end hook)
static constexpr int EARLY_TOKENS = 32;
static bool      g_early_done = false;
static long long g_early_hot = 0, g_early_cold = 0;
static bool g_has_mmproj = false;             // a multimodal projector is loaded (set in main; the
                                              // ctx sink enables the fork's vision-transient decode)

// Content trace (bench, findings 213e, MOE_DUAL_CONTENT_TRACE=<file>): during DECODE only, what a content-aware
// expert forecaster would see -- the batch's token ids, every layer's router input (attn_out, f16) and the real
// top-k per position. Capture-only: each observed tensor is a sync, decode runs slower. Off = no change.
// File "LCT1", then chunks: u8 tag (0 decode call, 1 token ids i32, 2 top-k ids i32 [n_tok][k], 3 attn_out f16
// [n_tok][n_embd], 4 router top-16 i32 ids then f32 scores [n_tok][16], findings 244), u8 0, i16 layer, i32 n_tok, i32 nbytes,
// data. Flushed at every decode call.
static FILE * g_ct = nullptr;
static bool   g_ct_decode = false;
static void content_trace_init() {
    const char * p = std::getenv("MOE_DUAL_CONTENT_TRACE");
    if (!p || !*p) return;
    g_ct = std::fopen(p, "wb");
    if (!g_ct) { std::fprintf(stderr, "moert-server: content trace: cannot open %s\n", p); return; }
    std::fwrite("LCT1", 1, 4, g_ct); std::fflush(g_ct);
    std::fprintf(stderr, "moert-server: content trace ON: %s (decode syncs per layer: capture only)\n", p);
}
static void content_chunk(uint8_t tag, int layer, int n_tok, const void * data, int32_t nbytes) {
    const uint8_t h0[2] = { tag, 0 };
    const int16_t il = (int16_t) layer;
    const int32_t h1[2] = { n_tok, nbytes };
    std::fwrite(h0, 1, 2, g_ct); std::fwrite(&il, 2, 1, g_ct); std::fwrite(h1, 4, 2, g_ct);
    if (nbytes) std::fwrite(data, 1, (size_t) nbytes, g_ct);
}
static int layer_of(const char * name) { const char * d = std::strrchr(name, '-'); return d ? std::atoi(d + 1) : -1; }
// the batch's token ids come from the server (the embedding is an input of the graph, not a node)
static void content_trace_tokens(const int32_t * tokens, int n) {
    if (g_ct && g_ct_decode) content_chunk(1, -1, n, tokens, (int32_t) (n * 4));
}
static bool content_trace_cb(ggml_tensor * t, bool ask) {
    // MOE_DUAL_CONTENT_SKIP_EMBD=<n>: leave out attn_out tensors of that width (the target's: a light capture that
    // keeps the MTP drafter's state, ~15x smaller)
    static const int64_t skip_embd = [] { const char * e = std::getenv("MOE_DUAL_CONTENT_SKIP_EMBD"); return e ? (int64_t) std::atoll(e) : (int64_t) 0; }();
    const bool attn = std::strncmp(t->name, "attn_out-", 9) == 0 && t->ne[0] != skip_embd;
    const bool topk = std::strncmp(t->name, "ffn_moe_topk-", 13) == 0;
    // claude dump findings 244: the router's 16 best scores per position (the experts that "almost" made the top 8)
    const bool probs = std::strncmp(t->name, "ffn_moe_probs-", 14) == 0 && t->ne[0] >= 16 && t->ne[2] == 1;   // [n_expert, n_tok], not its 3-d reshape
    if (ask) return attn || topk || probs;
    static std::vector<char> raw;
    if (probs && t->type == GGML_TYPE_F32) {   // tag 4: i32 ids [n_tok][16], then f32 scores [n_tok][16]
        const int64_t ne = t->ne[0], n = t->ne[1];
        raw.resize(ggml_nbytes(t));
        ggml_backend_tensor_get(t, raw.data(), 0, raw.size());
        const int K16 = (int) std::min<int64_t>(16, ne);
        std::vector<int32_t> ids((size_t) (n * 16), -1);
        std::vector<float> sc((size_t) (n * 16), 0.0f);
        std::vector<int32_t> ord((size_t) ne);
        for (int64_t r = 0; r < n; ++r) {
            const char * row = raw.data() + r * t->nb[1];
            auto v = [&](int32_t e) { return *(const float *) (row + e * t->nb[0]); };
            for (int32_t e = 0; e < (int32_t) ne; ++e) ord[(size_t) e] = e;
            std::partial_sort(ord.begin(), ord.begin() + K16, ord.end(), [&](int32_t a, int32_t b) { return v(a) > v(b); });
            for (int j = 0; j < K16; ++j) { ids[(size_t) (r * 16 + j)] = ord[(size_t) j]; sc[(size_t) (r * 16 + j)] = v(ord[(size_t) j]); }
        }
        std::vector<char> out(ids.size() * 4 + sc.size() * 4);
        std::memcpy(out.data(), ids.data(), ids.size() * 4);
        std::memcpy(out.data() + ids.size() * 4, sc.data(), sc.size() * 4);
        content_chunk(4, layer_of(t->name), (int) n, out.data(), (int32_t) out.size());
    } else if (topk && t->type == GGML_TYPE_I32) {
        const int64_t k = t->ne[0], n = t->ne[1];
        raw.resize(ggml_nbytes(t));
        ggml_backend_tensor_get(t, raw.data(), 0, raw.size());
        std::vector<int32_t> out((size_t) (k * n));
        for (int64_t r = 0; r < n; ++r)
            for (int64_t j = 0; j < k; ++j) out[(size_t) (r * k + j)] = *(const int32_t *) (raw.data() + r * t->nb[1] + j * t->nb[0]);
        content_chunk(2, layer_of(t->name), (int) n, out.data(), (int32_t) (out.size() * 4));
    } else if (attn && t->type == GGML_TYPE_F32 && ggml_is_contiguous(t)) {
        const int64_t ne0 = t->ne[0], n = ggml_nelements(t) / ne0;
        raw.resize(ggml_nbytes(t));
        ggml_backend_tensor_get(t, raw.data(), 0, raw.size());
        std::vector<ggml_fp16_t> h((size_t) (ne0 * n));
        ggml_fp32_to_fp16_row((const float *) raw.data(), h.data(), ne0 * n);
        content_chunk(3, layer_of(t->name), (int) n, h.data(), (int32_t) (h.size() * 2));
    }
    return true;
}

static bool moe_srv_q4s_eval_cb(ggml_tensor * t, bool ask, void * /*user*/) {
    if (g_ct && g_ct_decode && !moe_dual::q4stream_prompt_active()) return content_trace_cb(t, ask);
    if (!moe_dual::q4stream_prompt_active()) return false;   // outside prompt: zero syncs
    const bool is_topk = std::strncmp(t->name, "ffn_moe_topk", 12) == 0;
    if (ask) return is_topk;
    if (is_topk && t->type == GGML_TYPE_I32) {
        const char * dash = std::strrchr(t->name, '-');
        const int il = dash ? std::atoi(dash + 1) : -1;
        if (!moe_dual::q4stream_on_topk(il)) return false;   // stream error -> abort graph (loud)
        moe_dual::q4stream_count_topk(il, t);                 // P3: the prompt's routing (no-op when off)
    }
    return true;
}

static void moe_q4s_close(bool log_stats, bool force_release = false) {
    if (!g_q4s_open) return;
    moe_dual::q4stream_end();
    if (log_stats) {
        const moe_dual::Q4StreamStats s = moe_dual::q4stream_stats();
        // R1: with the staging ring the read runs on the prefetch thread, so stage_ms is no longer
        // the cost the graph PAYS -- wait_ms is (time the eval callback blocked on the ring). Print
        // both: wait ~0 with a full stage_ms means the overlap hid the SSD read behind compute.
        // "source": where the staged bytes came from. RAM = the resident owned cold arena
        // (MOE_DUAL_Q4S_ARENA, no per-prompt disk re-read), SSD = the source GGUF, mixed = only
        // some groups mapped onto the arena. The read_ms term keeps its meaning either way: the
        // time spent OBTAINING the bytes (positioned read, or arena->pinned-slice copy).
        const double gib   = (double) (s.bytes_streamed - g_q4s_t0.bytes_streamed) / (1024.0 * 1024.0 * 1024.0);
        const long long ab = s.arena_bytes - g_q4s_t0.arena_bytes;
        const long long tb = s.bytes_streamed - g_q4s_t0.bytes_streamed;
        const char * src   = (tb > 0 && ab == tb) ? "RAM arena" : (ab > 0 ? "mixed RAM/SSD" : "SSD source");
        const double slab_gib = (double) (s.slab_bytes - g_q4s_t0.slab_bytes) / (1024.0 * 1024.0 * 1024.0);
        std::fprintf(stderr,
            "[moert-server] q4stream prefill: %lld layer stagings, %.2f GiB streamed from %s + %.2f GiB "
            "D2D from the hot slab (%lld expert pages), stage %.0f ms (read %.0f ms), ring %d, exposed wait %.0f ms\n",
            s.layers_staged - g_q4s_t0.layers_staged, gib, src, slab_gib,
            s.slab_experts - g_q4s_t0.slab_experts,
            s.stage_ms - g_q4s_t0.stage_ms, s.read_ms - g_q4s_t0.read_ms,
            s.ring_n, s.wait_ms - g_q4s_t0.wait_ms);
        if (s.bnd_layers > g_q4s_t0.bnd_layers)
            std::fprintf(stderr, "[moert-server] boundary residency: %lld layers decided in the prefill, %lld experts "
                                 "copied staging -> slab%s, %.0f ms\n",
                         s.bnd_layers - g_q4s_t0.bnd_layers, s.bnd_promos - g_q4s_t0.bnd_promos,
                         s.bnd_fail > g_q4s_t0.bnd_fail ? " (SOME FILLS REFUSED)" : "", s.bnd_ms - g_q4s_t0.bnd_ms);
        if (ab > 0) {
            // 4 KiB pages: faults ~= arena bytes / 4096 means nothing was physically resident.
            const long long pages  = ab / 4096;
            const long long faults = s.arena_faults - g_q4s_t0.arena_faults;
            std::fprintf(stderr,
                "[moert-server] q4stream arena source: %lld page faults over %lld pages touched (%.0f%%), "
                "working set %lld MiB\n", faults, pages,
                pages > 0 ? 100.0 * (double) faults / (double) pages : 0.0, s.ws_mib);
        }
        std::fflush(stderr);
        llapis_phase("Q4 batched prefill done: %.2f GiB streamed from %s (%lld layer stagings, stage %.0f ms, read %.0f ms, ring %d, exposed wait %.0f ms)",
            gib, src,
            s.layers_staged - g_q4s_t0.layers_staged,
            s.stage_ms - g_q4s_t0.stage_ms, s.read_ms - g_q4s_t0.read_ms,
            s.ring_n, s.wait_ms - g_q4s_t0.wait_ms);
    }
    // a lean (graph-reserve) arm or a forced release frees the staging; the fixed reserve stays
    if (force_release || !moe_dual::q4stream_keep_after_prompt()) moe_dual::q4stream_shutdown();
    g_q4s_open = false;
}

// Variable-KV ctx swap. graph_reserve sizes the new context's PERMANENT compute buffer on two graphs
// reaching the hook through different doors: the prompt graph (n>1) takes the q4stream branch while
// the bracket is armed; the token graph (n==1) needs the RESIDENT slab. Either door open = the hook
// returns nullptr, llama sources the loader-cut shells and the reserve inflates to ~2.7 GiB
// (ctx-independent). So the slab is made SMALL BUT RESIDENT (vkv_fit_slab_for_ctx_swap), never
// evicted, and K is re-fitted once afterwards. Fired by the fork right after the old context is
// freed (max free VRAM). Never PREDICT the new context's KV cost to size this: measured wrong.
// (history: docs/code-comments-archive-2026-10-05.md, "ctx-swap")
static void vkv_prepare_ctx_swap(int n_ctx_new) {
    (void) n_ctx_new;
    // First, while q4s_on is guaranteed false. The dimensioned shrink: footprint_old = free now - free at
    // turn start (MEASURED); cost_new = footprint_old + slope x (C_new - C_old); budget = (free now + slab)
    // - cost_new - staging - keep_post. (history: docs/code-comments-archive-2026-10-05.md, "ctx-swap-budget")
    if (g_vkv_enabled && moe_dual::is_dynamic()) {
        // Prefer the runtime's own reported staging size over the compiled-in default.
        if (const long long b = moe_dual::q4stream_stats().buf_bytes >> 20; b > 0) g_vkv_staging_mib = b;
        long long budget = -1;   // -1 = no measurement, fall back to the quality floor
        if (g_vkv_free_at_turn_start >= 0 && g_vkv_n_ctx_at_turn_start > 0 && n_ctx_new > 0) {
            const long long free_now      = moe_dual::pool_free_mib();
            const long long footprint_old = free_now - g_vkv_free_at_turn_start;
            if (footprint_old > 0) {
                const long long cost_new = vkv::ctx_cost_predict_mib(
                    g_vkv_n_ctx_at_turn_start, footprint_old, n_ctx_new);
                const long long slab_now = moe_dual::slab_evicted() ? 0 : moe_dual::runtime_info().slab_mib;
                budget = free_now + slab_now - cost_new - g_vkv_staging_mib - g_vkv_keep_post;
                std::fprintf(stderr,
                    "[moert-server] variable-KV swap budget: old ctx=%d measured %lld MiB "
                    "(free %lld -> %lld across teardown); new ctx=%d predicted %lld MiB; "
                    "pool %lld (%lld free + %lld slab) - %lld ctx - %lld staging - %lld keep "
                    "= %lld MiB for the slab\n",
                    g_vkv_n_ctx_at_turn_start, footprint_old, g_vkv_free_at_turn_start, free_now,
                    n_ctx_new, cost_new, free_now + slab_now, free_now, slab_now, cost_new,
                    g_vkv_staging_mib, g_vkv_keep_post, budget);
                std::fflush(stderr);
            }
        }
        moe_dual::vkv_fit_slab_for_ctx_swap(budget);
        g_vkv_free_at_turn_start = -1;   // one sample per swap; never reuse a stale one
    }
    if (!g_q4s_src.empty()) {
        // lean_ring: like the load-time arm, this one exists only so the REBUILD's graph_reserve
        // sees the streamed path -- no prompt streams through it, and the free-VRAM reading here is
        // inflated because the new (larger) context's KV has not been allocated yet. Measured: with
        // an AUTO decision made here, the 16384 -> 24576 rebuild read 1596 MiB of "headroom", armed
        // depth 2 (1179 MiB of staging), and the process died at 0 MiB free with an access
        // violation. Only an arm that is about to
        // stream a prompt, with its context already built, may size the ring.
        if (!moe_dual::q4stream_available() &&
            !moe_dual::q4stream_init(g_q4s_src.c_str(), "Q4STREAM", /*lean_ring=*/true)) {
            std::fprintf(stderr, "[moert-server] ctx rebuild: q4stream staging could not be armed — "
                                 "the new context's compute reserve will be LARGE\n");
            std::fflush(stderr);
        } else {
            // Bracket ONLY the reserve, not the real per-prompt phase: begin here (before the two
            // llama_init_from_model calls in rebalance_ctx), end in moe_ctx_rebuild_done —
            // mirroring the load path's own begin-before-load_model / end-at-ctx-sink shape.
            moe_dual::q4stream_begin();
        }
    }
    // The tg-door invariant, asserted at the last host-visible point before graph_reserve runs.
    // The host cannot see inside the reserve, so if this ever fires the ~2.7 GiB is already
    // unavoidable for the context about to be built — but it will be NAMED instead of silently
    // charged to "8 GB physics" for a third time.
    if (g_vkv_enabled && moe_dual::is_dynamic() && moe_dual::slab_evicted()) {
        std::fprintf(stderr, "[moert-server] variable-KV INVARIANT VIOLATED: the hot slab is EVICTED "
                             "entering the ctx swap — the new context's token-generation graph will "
                             "fall to the stock loader-cut shell path and reserve ~2.7 GiB PERMANENTLY. "
                             "Expect K to collapse after this resize.\n");
        std::fflush(stderr);
    }
}

static int moe_q4s_prompt_cap(int n_remaining) {
    // The decision itself lives in src/prefill_policy.hpp and is covered by test-prefill-policy;
    // the two measured regimes and why one threshold cannot serve both are documented there.
    // What stays HERE is everything that touches the machine: the slab's state and the staging
    // probe, which allocates.
    const prefill::params pp{ g_q4s_min };
    // "warm" = the hot slab is resident AND mostly filled. A resident but empty slab (the boot slab
    // before its first prompt, findings 183) is as cold as an evicted one: fused there is ~418 ms/token.
    const bool slab_warm = !moe_dual::slab_evicted() && moe_dual::runtime_info().occupancy >= 0.5;
    prefill::path pth = prefill::decide(n_remaining, slab_warm, pp);

    // Graceful degradation (C3.6): the staging buffer (~589 MiB) is (re)allocated per prompt;
    // probe it HERE, at batch-build time, so an alloc failure (browser eating VRAM, KV growth)
    // sends the prompt down the correct fused per-token path instead of erroring the request.
    // pre_decode's hard refusal stays as the last-resort invariant (an armed-phase batch must
    // never compute against a stale staging buffer). Note this probe runs for the COLD case
    // too, which is why `decide` is not allowed to short-circuit it.
    if (pth == prefill::path::streamed &&
        !moe_dual::q4stream_available() &&
        !moe_dual::q4stream_init(g_q4s_src.c_str(), "Q4STREAM")) {
        std::fprintf(stderr, "[moert-server] q4stream: staging unavailable for this prompt "
                             "(VRAM pressure?) — falling back to the fused per-token prefill\n");
        std::fflush(stderr);
        pth = prefill::on_staging_unavailable();
    }
    return prefill::batch_cap(pth, g_q4s_spec_k);
}

static bool moe_q4s_pre_decode(bool is_prompt, int n_tokens) {
    // C3.4 phase discipline — the C3.3 corruption root cause. The three regimes, why a batch of
    // at most spec_k tokens must NEVER run with the phase armed, and the safety invariant itself
    // live in src/q4s_phase_policy.hpp, exhaustively checked by test-q4s-phase-policy. Read that
    // file before changing anything here: this failure mode does not crash, it produces
    // confident wrong text.
    const q4s::action act = q4s::decide(is_prompt, n_tokens, g_q4s_spec_k);
    if (act == q4s::action::close_and_pass) {
        moe_q4s_close(/*log_stats=*/true);       // no-op when already closed
        return true;
    }
    // streamed prompt decode: staging is REQUIRED (an armed-phase batch without it would compute
    // the prompt experts against a stale staging buffer — never a silent downgrade).
    if (q4s::requires_staging(act) &&
        !moe_dual::q4stream_available() &&
        !moe_dual::q4stream_init(g_q4s_src.c_str(), "Q4STREAM")) {
        return false;                            // fork errors the request, loudly
    }
    if (!g_q4s_open) {
        g_q4s_t0 = moe_dual::q4stream_stats();
        moe_dual::q4stream_begin();
        g_q4s_open = true;
        llapis_phase("Q4 batched prefill ... (%d prompt tokens in this streamed decode; full-layer Q4 staging)", n_tokens);
    }
    moe_dual::q4stream_chunk_begin();
    return true;
}

// Set once the q4stream pre-decode gate is wired; the seam wrapper below delegates to it.
static bool g_q4s_pre_decode_wired = false;

// Pre-decode trampoline, registered whenever the dual runtime is up (with or without
// q4stream). It delegates to the q4stream phase gate, which owns the REFUSAL semantics.
static bool moe_srv_pre_decode(bool is_prompt, int n_tokens) {
    if (!is_prompt) { moe_dual::apply_ready_before_decode(); moe_dual::before_decode(); }
    moe_dual::publish_fence();   // findings 216k: mirrors published asynchronously must land before any decode
    if (!is_prompt) moe_dual::after_publish_before_decode();   // findings 234: early decision, runs during this verify
    if (g_ct) {   // content trace: a decode call starts (flush the previous one)
        g_ct_decode = !is_prompt;
        if (!is_prompt) { content_chunk(0, -1, n_tokens, nullptr, 0); std::fflush(g_ct); }
    }
    return g_q4s_pre_decode_wired ? moe_q4s_pre_decode(is_prompt, n_tokens) : true;
}

// M4 heartbeat: any /moe poll or decode activity
// counts as "someone is watching this model". 0 = watchdog off (standalone / router).
static long long lifecycle_now_ms() {
    return (long long) std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}
static std::atomic<long long> g_last_activity_ms{0};
static long long  g_auto_unload_s = 0;
static int        g_router_port   = 0;   // LLAMA_SERVER_ROUTER_PORT when router-spawned
static std::string g_child_alias;         // our profile name (--alias) in the router

// --- /moe/telemetry + /moe/info JSON provider (M1) --------------------------------------
// Everything here is moe_dual-OWNED state (RuntimeInfo). The llama-owned pieces of the
// memory map (KV cache, compute buffers — probed via mem_probe with the
// llama_context handle, which llama-server owns internally) are the M2 memory-partition
// work, not reachable from this glue. C-string contract: see server-moe-hooks.h.
static int moe_json_provider(const char * what, char * buf, int cap) {
    g_last_activity_ms.store(lifecycle_now_ms());   // M4 heartbeat: a /moe poll = an observer
    const moe_dual::RuntimeInfo r = moe_dual::runtime_info();
    if (!r.active || !what || !buf || cap <= 0) return 0;

    auto esc = [](const char * s) {
        std::string o;
        for (; s && *s; ++s) { if (*s == '\\' || *s == '"') o += '\\'; o += *s; }
        return o;
    };
    auto env = [](const char * k) { const char * v = std::getenv(k); return v ? v : ""; };

    char tmp[65536];
    int n = 0;
    if (std::strcmp(what, "info") == 0) {
        const bool cut = env("MOE_DUAL_NO_EXPERT_ALLOC")[0] && env("MOE_DUAL_NO_EXPERT_ALLOC")[0] != '0';
        // CPU cold sidecar: cold pairs are computed on the host so their weights never cross
        // PCIe. Default-ON for the product profiles. Reflected from env, like loader_cut above.
        const bool ccpu       = env("MOE_DUAL_COLD_CPU")[0] && env("MOE_DUAL_COLD_CPU")[0] != '0';
        const bool cs         = env("MOE_DUAL_COPYSTREAM")[0] == '1';
        const int  ccpu_max   = env("MOE_DUAL_COLD_CPU_MAX")[0]     ? std::atoi(env("MOE_DUAL_COLD_CPU_MAX"))     : 0;
        const int  ccpu_wrk   = env("MOE_DUAL_COLD_CPU_WORKERS")[0] ? std::atoi(env("MOE_DUAL_COLD_CPU_WORKERS")) : 0;
        // speculative decode K=2 (opt-in; wiring in main): enabled iff MOE_DUAL_SPEC_K >= 2
        const int  spec_k     = env("MOE_DUAL_SPEC_K")[0] ? std::atoi(env("MOE_DUAL_SPEC_K")) : 1;
        // MOE_DUAL_SPEC_NODRAFT: the fused WIDTH without a drafter (-NoMtp). spec_k sizes the dual
        // runtime's fused ceiling and the loader cut's decode ubatch; arming a drafter is a separate
        // decision, and conflating them made `enabled` report speculation that -NoMtp had turned off.
        const bool spec_nodraft = env("MOE_DUAL_SPEC_NODRAFT")[0] == '1';
        const char * spec_dr  = env("MOE_DUAL_SPEC_DRAFTER");
        n = std::snprintf(tmp, sizeof(tmp),
            "{\"preset\":\"%s\",\"ctx\":%d,\"prefill\":\"%s\",\"vision\":%s,\"arena_verify_ok\":%lld,\"arena_verify_bad\":%lld,"
            "\"dynamic\":%s,\"cache_mode\":\"%s\",\"k\":%d,\"k_requested\":%d,"
            "\"n_expert\":%d,\"n_layers\":%d,\"total_slots\":%d,"
            "\"fused\":%s,\"planc\":%d,\"loader_cut\":%s,\"cold_arena\":%s,"
            "\"promotions_per_token\":%d,"
            "\"cold_cpu\":{\"enabled\":%s,\"max\":%d,\"workers\":%d,\"copy_stream\":%s},"
            "\"spec\":{\"enabled\":%s,\"k\":%d,\"drafter\":\"%s\"},"
            "\"model_high\":\"%s\",\"model_low_arena\":\"%s\","
            "\"prompt_end\":{\"total_ms\":%.1f,\"slab_refit_ms\":%.1f,\"refresh_ms\":%.1f},"
            "\"vram_total_mib\":%lld,\"vram_used_post_load_mib\":%lld,\"slab_mib\":%lld}",
            esc(env("MOE_PRESET")).c_str(),   // launch-preset name, set by `llapis serve` ("" = launched bare)
            g_obs_n_ctx,
            env("MOE_DUAL_PREFILL_Q4SRC")[0] ? "q4stream" : "per-token",
            env("MOE_VISION")[0] ? "true" : "false",
            (long long) r.cold_arena_verify_ok, (long long) r.cold_arena_verify_bad,
            r.dynamic ? "true" : "false", r.cache_mode, r.hot_k, r.k_requested,
            r.n_expert, r.n_layers, r.total_slots,
            r.fused ? "true" : "false", r.planc,
            cut ? "true" : "false", r.cold_arena_active ? "true" : "false",
            moe_dual::cost_per_token(),
            ccpu ? "true" : "false", ccpu_max, ccpu_wrk, cs ? "true" : "false",
            (spec_k >= 2 && !spec_nodraft) ? "true" : "false", spec_k, esc(spec_dr[0] ? spec_dr : "none").c_str(),
            esc(env("MOE_DUAL_DYNAMIC")).c_str(),
            esc(env("MOE_DUAL_COLD_ARENA")).c_str(),
            g_pe_total_ms.load(), g_pe_slab_ms.load(), g_pe_refresh_ms.load(),
            r.vram_total_mib, r.vram_used_post_load_mib, r.slab_mib);
    } else if (std::strcmp(what, "telemetry") == 0) {
        // M2 memory-partition block: every byte accounted on both devices, derived rows close
        // the ledger. Read the slot-thread-maintained cache — see refresh_partition() for why.
        const mem_partition::Partition & mp = g_part;
        // queried here rather than cached in the partition — one DXGI call on an
        // HTTP handler, and the budget moves with what OTHER processes are doing (a browser
        // opening costs us ~1 GiB of it), so a stale copy would be worse than no copy.
        long long g_budget_ours_mib = -1, g_budget_headroom_mib = -1;
        {
            long long b = 0, u = 0;
            if (mem_partition::wddm_local(&b, &u) && b > 0) {
                g_budget_ours_mib     = b;
                g_budget_headroom_mib = (b - u) < 0 ? 0 : (b - u);
            }
        }
        char part[2048];
        // C5.6: ship the WHOLE ledger -- a dropped row is charged to "free" by the UI's memory bar.
        // `unaccounted` = ours minus every named row = CUDA context + cuBLAS workspace + allocator slack.
        // (history: docs/code-comments-archive-2026-10-05.md, "ledger")
        std::snprintf(part, sizeof(part),
            "{\"vram\":{\"total\":%lld,\"free\":%lld,\"ours\":%lld,\"shared_ours\":%lld,"
            "\"shared_baseline\":%lld,\"budget_ours\":%lld,\"budget_headroom\":%lld,"
            "\"nonexpert_weights\":%lld,\"hot_slab\":%lld,\"l2\":%lld,\"staging\":%lld,\"kv\":%lld,"
            "\"compute_tgt\":%lld,\"compute_dft\":%lld,\"drafter_weights\":%lld,\"spec_active\":%s,"
            "\"pool_reserved\":%lld,\"pool_used\":%lld,"
            "\"unaccounted\":%lld,\"compute_other\":%lld,\"other_procs\":%lld},"
            "\"ram\":{\"total\":%lld,\"avail\":%lld,\"rss\":%lld,\"cold_arena\":%lld,"
            "\"arena_committed\":%lld,\"arena_resident\":%lld,"
            "\"nonarena_rss\":%lld,\"other_procs\":%lld}}",
            mp.vram_total, mp.vram_free, mp.vram_ours, mp.vram_shared_ours,
            g_vram_shared_baseline_mib.load(),
            // the per-process WDDM budget and what is left of it. `free` is
            // device-wide and says nothing about when THIS process starts getting demoted; this
            // pair does, and it is the only reading that distinguishes "the GPU is full" from
            // "we are full". Both -1 when DXGI is unavailable.
            g_budget_ours_mib, g_budget_headroom_mib,
            mp.vram_nonexpert, mp.vram_hot_slab, mp.vram_l2, mp.vram_staging, mp.vram_kv,
            mp.vram_compute_tgt, mp.vram_compute_dft, mp.vram_drafter_wts,
            mp.spec_active ? "true" : "false",
            mp.vram_pool_reserved, mp.vram_pool_used,
            // compute_other is kept as an ALIAS of unaccounted for one release, so a consumer
            // written against the older field name keeps reading the value it expects.
            mp.vram_compute_other, mp.vram_compute_other, mp.vram_other_procs,
            mp.ram_total, mp.ram_avail, mp.ram_rss, mp.ram_cold_arena,
            mp.ram_arena_committed, mp.ram_arena_resident,
            mp.ram_nonarena_rss, mp.ram_other_procs);
        moe_dual::IslandStats is_;
        moe_dual::island_stats(&is_);
        n = std::snprintf(tmp, sizeof(tmp),
            "{\"tokens\":%lld,\"visible_tokens\":%lld,"
            "\"window\":{\"tokens\":%d,\"tok_per_s\":%.2f,\"gen_tok_per_s\":%.2f,\"ms_per_tok\":%.2f,\"hit_rate\":%.6f},"
            "\"hot_routes\":%lld,\"cold_routes\":%lld,\"hit_rate\":%.6f,\"occupancy\":%.4f,"
            "\"promotions\":%lld,\"demotions\":%lld,"
            "\"refresh\":{\"count\":%lld,\"promotions\":%lld,\"demotions\":%lld,"
            "\"pause_ms_total\":%.2f,\"pause_ms_max\":%.2f,"
            "\"last_tok\":%lld,\"last_promos\":%lld,\"last_pause_ms\":%.2f},"
            "\"mem\":{\"vram_total_mib\":%lld,\"vram_used_post_load_mib\":%lld,"
            "\"vram_free_pre_slab_mib\":%lld,\"slab_mib\":%lld,\"staging_mib\":%lld,"
            "\"cold_pool_mib\":%lld,\"cold_arena_mib\":%lld},"
            "\"l2\":{\"active\":%s,\"budget_mib\":%lld,\"slab_mib\":%lld,\"capacity\":%lld,"
            "\"resident\":%lld,\"lookups\":%lld,\"hits\":%lld,\"routes\":%lld,"
            "\"bytes_avoided\":%lld},"
            "\"island\":{\"fires\":%lld,\"op_host_ms\":%.1f,\"seam_syncs\":%lld,\"seam_ms\":%.1f,"
            "\"coldcpu_calls\":%lld,\"coldcpu_pairs\":%lld,\"coldcpu_ms\":%.1f,\"coldcpu_over_max\":%lld},"
            "\"partition\":%s}",
            r.tokens, r.visible_tokens,
            r.win_tokens, r.win_tok_per_s, moe_gen_rate_recent(), r.win_ms_per_tok, r.win_hit_rate,
            r.hot_routes, r.cold_routes, moe_dual::hit_rate(), r.occupancy,
            r.promotions, r.demotions,
            r.refresh_count, r.refresh_promotions, r.refresh_demotions,
            r.refresh_pause_ms_total, r.refresh_pause_ms_max,
            r.last_refresh_tok, r.last_refresh_promos, r.last_refresh_pause_ms,
            r.vram_total_mib, r.vram_used_post_load_mib,
            r.vram_free_pre_slab_mib, r.slab_mib, r.staging_mib,
            r.cold_pool_mib, r.cold_arena_mib,
            r.l2_active ? "true" : "false", r.l2_budget_mib, r.l2_slab_mib, r.l2_capacity,
            r.l2_resident, r.l2_lookups, r.l2_hits, r.l2_routes,
            r.l2_bytes_avoided,
            is_.fires, is_.op_host_ns / 1e6, is_.seam_syncs, is_.seam_ns / 1e6,
            is_.coldcpu_calls, is_.coldcpu_pairs, is_.coldcpu_ns / 1e6, is_.coldcpu_over_max, part);
    } else {
        return 0;
    }
    if (n <= 0 || n >= cap || n >= (int) sizeof(tmp)) return 0;
    std::memcpy(buf, tmp, (size_t) n + 1);
    return n;
}

// ===========================================================================
// M4 lifecycle glue: router-mode profiles + plan provider,
// child-mode profile application + auto-K + heartbeat auto-unload watchdog.
// STRICT policy everywhere: malformed profiles refuse to start, a missing alias
// kills the child with a precise reason (visible in the router log + /models status).
// ===========================================================================

using nlohmann::json;
namespace fs = std::filesystem;

static std::string exe_path() {
#ifdef _WIN32
    char buf[MAX_PATH];
    const DWORD n = GetModuleFileNameA(nullptr, buf, MAX_PATH);
    if (n == 0 || n >= MAX_PATH) throw std::runtime_error("moert-server: GetModuleFileNameA failed");
    return std::string(buf, n);
#else
    return "moert-server";
#endif
}

// moert-profiles.json, next to the exe
struct MoertProfiles {
    std::string path;        // the json file itself
    std::string models_dir;  // high|low pair for the planner (may be relative to exe dir)
    long long   auto_unload_s = 180;
    json        profiles;    // name -> { model, args{}, env{}, auto_k }
};

static std::string profiles_path() {
    return (fs::path(exe_path()).parent_path() / "moert-profiles.json").string();
}

static MoertProfiles load_profiles() {
    MoertProfiles p;
    p.path = profiles_path();
    std::ifstream f(p.path, std::ios::binary);
    if (!f) throw std::runtime_error("moert-server: cannot open " + p.path);
    json root;
    try {
        root = json::parse(f);
    } catch (const std::exception & e) {
        throw std::runtime_error("moert-server: " + p.path + " is not valid JSON: " + e.what());
    }
    if (root.value("version", 0) != 1)
        throw std::runtime_error("moert-server: " + p.path + ": \"version\" must be 1");
    if (!root.contains("profiles") || !root["profiles"].is_object() || root["profiles"].empty())
        throw std::runtime_error("moert-server: " + p.path + ": \"profiles\" must be a non-empty object");
    p.models_dir    = root.value("models_dir", std::string("models"));
    p.auto_unload_s = root.value("auto_unload_s", (long long) 180);
    // resolve models_dir relative to the exe dir (portable-zip friendly)
    if (fs::path(p.models_dir).is_relative())
        p.models_dir = (fs::path(exe_path()).parent_path() / p.models_dir).string();
    p.profiles = root["profiles"];
    for (const auto & [name, prof] : p.profiles.items()) {
        if (!prof.is_object() || !prof.contains("model") || !prof["model"].is_string())
            throw std::runtime_error("moert-server: " + p.path + ": profile '" + name +
                                     "' needs a string \"model\" (path to the base GGUF)");
        const std::string mp = prof["model"].get<std::string>();
        if (!fs::exists(mp))
            throw std::runtime_error("moert-server: " + p.path + ": profile '" + name +
                                     "': model file not found: " + mp);
        for (const auto & key : {"args", "env"})
            if (prof.contains(key) && !prof[key].is_object())
                throw std::runtime_error("moert-server: " + p.path + ": profile '" + name +
                                         "': \"" + key + "\" must be an object of strings");
    }
    return p;
}

// Generate the fork preset INI from the profiles (the router's model list). The fork
// itself is strict about unknown keys (load_from_ini throws), so bad "args" keys refuse
// the router at startup — exactly what we want.
static std::string generate_preset_ini(const MoertProfiles & p) {
    const std::string ini_path =
        (fs::path(exe_path()).parent_path() / "moert-profiles.gen.ini").string();
    std::string out = "; GENERATED from moert-profiles.json at every router start - do not edit\n";
    auto check_val = [](const std::string & k, const std::string & v) {
        if (v.find_first_of(";#\r\n") != std::string::npos)
            throw std::runtime_error("moert-server: profile value for '" + k +
                                     "' contains ';', '#' or a newline - not representable in the preset INI: " + v);
        return v;
    };
    for (const auto & [name, prof] : p.profiles.items()) {
        out += "\n[" + name + "]\n";
        out += "model = " + check_val("model", prof["model"].get<std::string>()) + "\n";
        if (prof.contains("args"))
            for (const auto & [k, v] : prof["args"].items()) {
                if (!v.is_string())
                    throw std::runtime_error("moert-server: profile '" + name + "' args." + k +
                                             " must be a string");
                out += k + " = " + check_val(k, v.get<std::string>()) + "\n";
            }
    }
    std::ofstream f(ini_path, std::ios::binary | std::ios::trunc);
    if (!f) throw std::runtime_error("moert-server: cannot write " + ini_path);
    f << out;
    return ini_path;
}

#ifdef _WIN32
// Run a command line, capture its stdout (stderr inherits ours => plan errors land in
// the router log). Returns the child's exit code, -1 on spawn failure.
static int run_capture_stdout(const std::string & cmdline, std::string & out) {
    SECURITY_ATTRIBUTES sa{};
    sa.nLength = sizeof(sa);
    sa.bInheritHandle = TRUE;
    HANDLE rd = nullptr, wr = nullptr;
    if (!CreatePipe(&rd, &wr, &sa, 0)) return -1;
    SetHandleInformation(rd, HANDLE_FLAG_INHERIT, 0);

    STARTUPINFOA si{};
    si.cb = sizeof(si);
    si.dwFlags    = STARTF_USESTDHANDLES;
    si.hStdOutput = wr;
    si.hStdError  = GetStdHandle(STD_ERROR_HANDLE);
    si.hStdInput  = GetStdHandle(STD_INPUT_HANDLE);
    PROCESS_INFORMATION pi{};
    std::vector<char> cmd(cmdline.begin(), cmdline.end());
    cmd.push_back('\0');
    if (!CreateProcessA(nullptr, cmd.data(), nullptr, nullptr, TRUE,
                        CREATE_NO_WINDOW, nullptr, nullptr, &si, &pi)) {
        CloseHandle(rd); CloseHandle(wr);
        return -1;
    }
    CloseHandle(wr);   // ours; the child holds the write end now
    char buf[4096];
    DWORD got = 0;
    while (ReadFile(rd, buf, sizeof(buf), &got, nullptr) && got > 0) out.append(buf, got);
    CloseHandle(rd);
    WaitForSingleObject(pi.hProcess, INFINITE);
    DWORD code = 1;
    GetExitCodeProcess(pi.hProcess, &code);
    CloseHandle(pi.hProcess); CloseHandle(pi.hThread);
    return (int) code;
}
#endif

// GET /moe/plan provider (router process): the CUDA probe runs in a short-lived
// `moert-server plan` subprocess so the router never touches the GPU. M6b1: ctx > 0
// (the route's ?ctx=N) is forwarded as `--ctx N` so the KV budget scales; the plan
// subcommand clamps/validates it like any CLI --ctx.
static std::string g_plan_cmdline;   // set once at router init
static int moe_plan_provider(int ctx, char * buf, int cap) {
#ifdef _WIN32
    static std::mutex mtx;           // serialize concurrent /moe/plan requests
    std::lock_guard<std::mutex> lock(mtx);
    std::string cmd = g_plan_cmdline;
    if (ctx > 0) cmd += " --ctx " + std::to_string(ctx);
    std::string out;
    const int rc = run_capture_stdout(cmd, out);
    if (rc != 0 || out.empty()) {
        std::fprintf(stderr, "moert-server: /moe/plan subprocess failed (exit %d) - cmd: %s\n",
                     rc, cmd.c_str());
        std::fflush(stderr);
        return 0;                    // the fork route answers 500
    }
    while (!out.empty() && (out.back() == '\n' || out.back() == '\r')) out.pop_back();
    if ((int) out.size() + 1 > cap) return 0;
    std::memcpy(buf, out.c_str(), out.size() + 1);
    return (int) out.size();
#else
    (void) ctx; (void) buf; (void) cap;
    return 0;
#endif
}

// --- C5.2 router admission gate + child-exit ledger --------------------------------------
// The fork calls moe_child_exit_note from its child-monitor thread whenever a model child
// exits, and moe_spawn_gate right before spawning a new one. Together they end the blind
// respawn storm: a child that died keeps the gate closed for a backoff window (longer when
// it refused on the memory preflight, exit 86), so autoload/poll-driven load requests get a
// cheap, explained refusal instead of another 7.8 GB arena reload into the same wall.
// Clean exits: 0, and 99 (the child's graceful stop after an Unload — NOT a crash).
static std::mutex g_spawn_mtx;
static std::string g_last_exit_name;
static int         g_last_exit_code  = 0;
static long long   g_last_exit_ms    = 0;
static int         g_consec_fails    = 0;

static void moe_child_exit_note(const char * name, int exit_code) {
    std::lock_guard<std::mutex> lk(g_spawn_mtx);
    if (exit_code == 0 || exit_code == 99) {
        g_consec_fails = 0;
        g_last_exit_code = exit_code;
        return;
    }
    g_last_exit_name = name ? name : "";
    g_last_exit_code = exit_code;
    g_last_exit_ms   = lifecycle_now_ms();
    g_consec_fails++;
    const char * why = exit_code == 86 ? "memory preflight refused (won't fit)"
                     : exit_code == 87 ? "another instance already holds the model lock"
                     : "crashed";
    std::fprintf(stderr, "moert-server: child '%s' EXITED code %d (%s) — consecutive failures: %d\n",
                 name ? name : "?", exit_code, why, g_consec_fails);
    llapis_phase("Model child '%s' exited with code %d (%s) — spawn gate arms a backoff",
                 name ? name : "?", exit_code, why);
}

static bool moe_spawn_gate(const char * name, char * reason, int reason_cap) {
    std::lock_guard<std::mutex> lk(g_spawn_mtx);
    if (g_consec_fails == 0) return true;
    long long backoff_s = 15;
    // code 87 = another instance holds the model lock: that will not clear in seconds, keep a long
    // floor. code 86 = the memory preflight refused. A SINGLE 86 right after a model switch/unload
    // is almost always the previous child's VRAM not yet reclaimed by the driver — a self-clearing
    // transient, not a persistent overcommit — so let the FIRST 86 retry on the short base window
    // and only treat REPEATED 86s as a real won't-fit (the *=consec_fails escalation below then
    // grows it fast: 15,30,45,60...). Fixes "switch model in the browser -> locked out ~60 s / 500s"
    // (the load then succeeds as soon as the old child's VRAM frees, typically within a couple s).
    if (g_last_exit_code == 87)
        backoff_s = std::max(backoff_s, 60ll);                 // lock held elsewhere: won't change fast
    else if (g_last_exit_code == 86 && g_consec_fails >= 2)
        backoff_s = std::max(backoff_s, 30ll);                 // repeated preflight refusals: real overcommit
    backoff_s *= std::min(g_consec_fails, 4);                  // escalate up to 4x on repeat failures
    const long long since_s = (lifecycle_now_ms() - g_last_exit_ms) / 1000;
    if (since_s < backoff_s) {
        std::snprintf(reason, (size_t) reason_cap,
                      "last child ('%s') exited code %d %llds ago (%d consecutive failure(s)) — "
                      "retry in %llds, or free the memory it complained about first",
                      g_last_exit_name.c_str(), g_last_exit_code, (long long) since_s,
                      g_consec_fails, (long long) (backoff_s - since_s));
        std::fprintf(stderr, "moert-server: spawn gate REFUSED load of '%s': %s\n",
                     name ? name : "?", reason);
        llapis_phase("Load of '%s' REFUSED by the spawn gate: %s", name ? name : "?", reason);
        return false;
    }
    return true;   // backoff expired: allow ONE retry; a failure re-arms a longer window
}

// POST /moe/profiles host callback (router): plan->profile save. Clones an existing
// profile with a different K (or auto_k) and/or a different context size (M6b1),
// persists moert-profiles.json, regenerates the preset INI (the fork reloads the
// model list right after). STRICT: unknown base profile / no change requested are
// errors. Body: {"from":"<profile>","k":36}, {"from":...,"auto_k":true}, and/or
// {"from":...,"ctx":8192} (+ optional "name"; default "<from>[-k<k>|-auto][-c<ctx>]").
static int moe_profiles_save(const char * body_json, char * err, int errcap) {
    static std::mutex mtx;
    std::lock_guard<std::mutex> lock(mtx);
    auto fail = [&](const std::string & msg) {
        std::snprintf(err, (size_t) errcap, "%s", msg.c_str());
        std::fprintf(stderr, "moert-server: /moe/profiles: %s\n", msg.c_str());
        return 0;
    };
    try {
        const json body = json::parse(body_json ? body_json : "");

        // C3: {"delete":"<name>"} removes a saved profile (the UI's ✕). The list must stay
        // non-empty (load_profiles refuses an empty "profiles" object and the router would
        // have no model list). Deleting the loaded profile only removes it from the LIST —
        // the running child is untouched until unload.
        if (body.contains("delete")) {
            const std::string name = body["delete"].get<std::string>();
            if (name.empty()) return fail("'delete' needs a profile name");
            const std::string path = profiles_path();
            std::ifstream f(path, std::ios::binary);
            if (!f) return fail("cannot open " + path);
            json root = json::parse(f);
            if (!root.contains("profiles") || !root["profiles"].contains(name))
                return fail("profile '" + name + "' not found in " + path);
            if (root["profiles"].size() <= 1)
                return fail("cannot delete '" + name + "': it is the last profile (the router needs at least one)");
            root["profiles"].erase(name);
            {
                std::ofstream o(path, std::ios::binary | std::ios::trunc);
                if (!o) return fail("cannot write " + path);
                o << root.dump(2) << "\n";
            }
            MoertProfiles profs = load_profiles();   // strict re-validation, as on save
            generate_preset_ini(profs);
            std::fprintf(stderr, "moert-server: /moe/profiles: deleted '%s'\n", name.c_str());
            std::fflush(stderr);
            return 1;
        }

        const std::string from = body.value("from", std::string());
        const bool auto_k = body.value("auto_k", false);
        const int  k      = body.value("k", 0);
        const int  ctx    = body.value("ctx", 0);
        if (from.empty()) return fail("'from' (base profile name) is required");
        if (!auto_k && k <= 0 && ctx <= 0)
            return fail("nothing to change: need 'k' > 0, 'auto_k': true, and/or 'ctx' > 0");
        if (ctx < 0 || (ctx > 0 && ctx < 512))
            return fail("'ctx' must be >= 512 (got " + std::to_string(ctx) + ")");

        const std::string path = profiles_path();
        std::ifstream f(path, std::ios::binary);
        if (!f) return fail("cannot open " + path);
        json root = json::parse(f);
        if (!root.contains("profiles") || !root["profiles"].contains(from))
            return fail("base profile '" + from + "' not found in " + path);

        json prof = root["profiles"][from];   // clone
        if (auto_k) {
            prof["auto_k"] = true;
            if (prof.contains("env")) prof["env"].erase("MOE_DUAL_K");
        } else if (k > 0) {
            prof["auto_k"] = false;
            prof["env"]["MOE_DUAL_K"] = std::to_string(k);
        }
        if (ctx > 0) {
            prof["args"]["ctx-size"] = std::to_string(ctx);
        }
        // C3 canonical nomenclature: the profile IS its user-visible config — gemma4_k=<K>_ctx=<CTX>.
        // Unchanged fields inherit the base profile's values so the name always tells the truth
        // (no more "-k36-c16384" suffix chains on an already-suffixed base).
        std::string eff_k = "auto";
        if (!prof.value("auto_k", false)) {
            if (k > 0)                                  eff_k = std::to_string(k);
            else if (prof.contains("env") && prof["env"].contains("MOE_DUAL_K"))
                                                        eff_k = prof["env"]["MOE_DUAL_K"].get<std::string>();
        }
        std::string eff_ctx;
        if (prof.contains("args") && prof["args"].contains("ctx-size"))
            eff_ctx = prof["args"]["ctx-size"].get<std::string>();
        std::string name = body.value("name", std::string());
        if (name.empty()) name = "gemma4_k=" + eff_k + "_ctx=" + (eff_ctx.empty() ? "default" : eff_ctx);
        prof["env"]["MOE_PRESET"] = name;     // strip identity follows the new profile
        root["profiles"][name] = prof;

        {
            std::ofstream o(path, std::ios::binary | std::ios::trunc);
            if (!o) return fail("cannot write " + path);
            o << root.dump(2) << "\n";
        }
        MoertProfiles profs = load_profiles();   // re-validate STRICTLY what we just wrote
        generate_preset_ini(profs);
        std::string what = auto_k ? std::string("auto_k") : (k > 0 ? "K=" + std::to_string(k) : "");
        if (ctx > 0) what += (what.empty() ? "" : ", ") + std::string("ctx=") + std::to_string(ctx);
        std::fprintf(stderr, "moert-server: /moe/profiles: saved '%s' (from '%s', %s)\n",
                     name.c_str(), from.c_str(), what.c_str());
        std::fflush(stderr);
        return 1;
    } catch (const std::exception & e) {
        return fail(e.what());
    }
}

// Child-mode env application: clear EVERY inherited MOE_* first (the router's own env
// leaks into children by design of the spawner), then apply the profile's env verbatim.
static void child_apply_profile_env(const std::string & alias, const json & prof) {
#ifdef _WIN32
    std::vector<std::string> to_clear;
    for (char ** e = _environ; e && *e; ++e)
        if (std::strncmp(*e, "MOE_", 4) == 0) {
            const char * eq = std::strchr(*e, '=');
            if (eq) to_clear.emplace_back(*e, (size_t)(eq - *e));
        }
    for (const auto & k : to_clear) _putenv_s(k.c_str(), "");
    if (prof.contains("env"))
        for (const auto & [k, v] : prof["env"].items()) {
            if (!v.is_string())
                throw std::runtime_error("moert-server: profile '" + alias + "' env." + k +
                                         " must be a string");
            _putenv_s(k.c_str(), v.get<std::string>().c_str());
        }
    // strip identity: surface the profile name unless the profile set its own
    if (!std::getenv("MOE_PRESET")) _putenv_s("MOE_PRESET", alias.c_str());
#else
    (void) alias; (void) prof;
#endif
}

// Child-mode auto-K (R8): run the planner IN-PROCESS right before load — the freshest
// possible free-VRAM probe. K = k_max_safe (fill the safe band, the R8 contract).
static void child_auto_k(const MoertProfiles & profs, const json & prof) {
    model_plan::PlanOptions po;
    if (prof.contains("args") && prof["args"].contains("ctx-size"))
        po.ctx = std::atoi(prof["args"]["ctx-size"].get<std::string>().c_str());
    model_plan::ModelPair mp = model_plan::scan_models_dir(profs.models_dir);
    model_plan::validate_pair(mp);
    model_plan::HwProbe hw = model_plan::probe_hw();
    model_plan::Plan    pl = model_plan::compute_plan(mp, hw, po);   // throws if no SAFE K
    const std::string k = std::to_string(pl.k_max_safe);
#ifdef _WIN32
    _putenv_s("MOE_DUAL_K", k.c_str());
#endif
    std::fprintf(stderr, "moert-server: auto-K planner: K=%s (k_max_safe; free VRAM %lld MiB, "
                         "per-K %.2f MiB, ctx %d) — k_default would be %d\n",
                 k.c_str(), hw.vram_free_mib, pl.per_k_step_mib, po.ctx, pl.k_default);
    std::fflush(stderr);
}

// Heartbeat watchdog (child): unload when nobody is watching (/moe polls or decode refresh the stamp;
// auto_unload_s=0 disables). The unload goes THROUGH THE ROUTER (POST /models/unload): a self-shutdown
// left a zombie holding 6.5 GiB VRAM. Standalone keeps the direct clean shutdown. (history: docs/code-comments-archive-2026-10-05.md, "watchdog")
static void (*g_hook_prompt_tokens)(int) = nullptr;
static void (*g_hook_prompt_end)()       = nullptr;
static void (*g_hook_token_end)()        = nullptr;
static void (*g_hook_turn_end)()         = nullptr;

static void set_moe_hooks_recording(void (*on_prompt_tokens)(int), void (*on_prompt_end)(),
                                    void (*on_token_end)(), void (*on_turn_end)(),
                                    void (*on_ready)()) {
    g_hook_prompt_tokens = on_prompt_tokens;
    g_hook_prompt_end    = on_prompt_end;
    g_hook_token_end     = on_token_end;
    g_hook_turn_end      = on_turn_end;
    llama_server_set_moe_hooks(on_prompt_tokens, on_prompt_end, on_token_end, on_turn_end, on_ready);
}

static void start_watchdog() {
    if (g_auto_unload_s <= 0) return;
    g_last_activity_ms.store(lifecycle_now_ms());
    std::thread([]() {
        for (;;) {
            std::this_thread::sleep_for(std::chrono::seconds(5));
            const long long idle_ms = lifecycle_now_ms() - g_last_activity_ms.load();
            if (idle_ms > g_auto_unload_s * 1000LL) {
                std::fprintf(stderr, "[moert-server] auto-unload: no observer for %lld s "
                                     "(auto_unload_s=%lld) — requesting unload\n",
                             idle_ms / 1000, g_auto_unload_s);
                std::fflush(stderr);
                bool via_router = false;
                if (g_router_port > 0 && !g_child_alias.empty()) {
                    httplib::Client cli("127.0.0.1", g_router_port);
                    cli.set_connection_timeout(5);
                    const std::string body = "{\"model\":\"" + g_child_alias + "\"}";
                    auto res = cli.Post("/models/unload", body, "application/json");
                    via_router = res && res->status >= 200 && res->status < 300;
                    if (!via_router)
                        std::fprintf(stderr, "[moert-server] auto-unload: router unload request "
                                             "failed (%s) — falling back to direct shutdown\n",
                                     res ? std::to_string(res->status).c_str() : "no connection");
                }
                if (!via_router) llama_server_request_shutdown();
                return;
            }
        }
    }).detach();
}

// --- `moert-server prepare`: derive the runtime artifacts ------------
// From the models/high|low pair, build into <models>/cache/ what the product profiles
// actually run: (a) base_neq4.gguf = byte-exact splice, experts <- LOW (IQ2 cold tier),
// every other tensor <- HIGH (the measured ~0.10-JS non-expert lever); (b) the LOW
// expert store (experts.index + experts.pack) = the owned cold-arena source. Pure
// streaming file transforms (gguf_tools), no model load, no GPU.
// A speed-only install (no model in models/high) gets (b) alone: speed runs the IQ2 file itself and
// never reads base_neq4.gguf. An existing store in the old expert-major layout (built before
// 2026-10-08, findings 197) is rebuilt without --force: it cannot be mapped.
// STRICT: pair validation failures and build errors are hard exits; existing outputs
// are SKIPPED (printed) unless --force.
static int run_prepare_mode(int argc, char ** argv) {
    try {
        std::string models = "models";
        std::string out;
        bool force = false;
        for (int i = 2; i < argc; ++i) {
            const std::string a = argv[i];
            auto next = [&]() -> const char * {
                if (i + 1 >= argc) throw std::runtime_error("prepare: flag " + a + " needs a value");
                return argv[++i];
            };
            if      (a == "--models") models = next();
            else if (a == "--out")    out    = next();
            else if (a == "--force")  force  = true;
            else throw std::runtime_error("prepare: unknown flag: " + a +
                     " (usage: moert-server prepare [--models dir] [--out dir] [--force])");
        }
        // speed-only when models/high holds no model at all (a projector alone does not count);
        // anything else in it goes through the strict pair check
        bool have_high = false;
        if (const fs::path hd = fs::path(models) / "high"; fs::exists(hd)) {
            for (const auto & e : fs::directory_iterator(hd)) {
                if (!e.is_regular_file() || e.path().extension() != ".gguf") continue;
                std::string stem = e.path().stem().string();
                std::transform(stem.begin(), stem.end(), stem.begin(), [](unsigned char c) { return (char) std::tolower(c); });
                if (stem.rfind("mmproj", 0) != 0) { have_high = true; break; }
            }
        }
        std::string low_path, high_path;
        if (have_high) {
            model_plan::ModelPair mp = model_plan::scan_models_dir(models);
            model_plan::validate_pair(mp);
            low_path = mp.low.path; high_path = mp.high.path;
        } else {
            low_path = model_plan::read_model(model_plan::find_model_gguf((fs::path(models) / "low").string())).path;
        }
        if (out.empty()) out = (fs::path(models) / "cache").string();
        fs::create_directories(out);

        const fs::path base  = fs::path(out) / "base_neq4.gguf";
        const fs::path index = fs::path(out) / "low.experts.index";
        const fs::path pack  = fs::path(out) / "low.experts.pack";

        std::fprintf(stderr, "[prepare] high = %s\n[prepare] low  = %s\n[prepare] out  = %s\n",
                     have_high ? high_path.c_str() : "(none: speed-only install)", low_path.c_str(), out.c_str());

        if (!have_high) {
            std::fprintf(stderr, "[prepare] no model in %s -- base_neq4.gguf (quality only) SKIPPED; "
                                 "put the Q4_K_XL GGUF there and re-run prepare to get quality\n",
                         (fs::path(models) / "high").string().c_str());
        } else if (!force && fs::exists(base) && fs::file_size(base) > 0) {
            std::fprintf(stderr, "[prepare] base_neq4.gguf exists (%lld MiB) — SKIP (--force rebuilds)\n",
                         (long long) (fs::file_size(base) >> 20));
        } else {
            std::fprintf(stderr, "[prepare] splicing base_neq4.gguf (experts <- low, rest <- high) ...\n");
            const fs::path tbase = fs::path(out) / "base_neq4.new.gguf";   // renamed over: see the store below
            if (gguf_tools::make_hybrid(low_path, high_path, tbase) != 0) {
                std::error_code ec; fs::remove(tbase, ec);
                throw std::runtime_error("prepare: splice FAILED (see message above)");
            }
            std::error_code ec;
            fs::rename(tbase, base, ec);
            if (ec) throw std::runtime_error("prepare: cannot replace " + base.string() + " (" + ec.message() +
                                             ") -- is a server still running? run 'llapis stop' and prepare again");
        }

        const bool store_there = fs::exists(index) && fs::exists(pack) && fs::file_size(pack) > 0;
        const int32_t layout = store_there ? moert_memmgr_store_group_major(index.string().c_str()) : -1;
        if (!force && store_there && layout == 1) {
            std::fprintf(stderr, "[prepare] low expert store exists (%lld MiB pack) — SKIP (--force rebuilds)\n",
                         (long long) (fs::file_size(pack) >> 20));
        } else {
            if (!force && store_there)
                std::fprintf(stderr, "[prepare] low expert store exists but %s -- rebuilding it\n",
                             layout == 0 ? "is in the old expert-major layout (cannot be mapped)" : "is unreadable");
            // Built under temporary names, then renamed over the old ones: writing in place would also
            // rewrite every HARD LINK to the old files (how this box shares its model store), and a
            // failed build would leave no store at all.
            const fs::path tindex = fs::path(out) / "low.experts.new.index";
            const fs::path tpack  = fs::path(out) / "low.experts.new.pack";   // the index's sibling name
            std::fprintf(stderr, "[prepare] building low expert store (index + pack) ...\n");
            if (gguf_tools::make_expert_store(low_path, tindex, tpack) != 0) {
                std::error_code ec; fs::remove(tindex, ec); fs::remove(tpack, ec);
                throw std::runtime_error("prepare: expert store build FAILED (see message above)");
            }
            std::error_code ec;
            fs::rename(tpack, pack, ec);
            if (!ec) fs::rename(tindex, index, ec);
            if (ec) throw std::runtime_error("prepare: cannot replace " + pack.string() + " (" + ec.message() +
                                             ") -- is a server still running? run 'llapis stop' and prepare again");
        }

        // Two-stage drafter head (claude dump findings 291): <drafter>.draft-head-r<r>.gguf next to the drafter, from the
        // basis shipped in assets/mtp/draft-head-r<r>.basis (fit once on bench captures) and the user's own drafter. A
        // drafter the basis was not fit on is reported and left alone: the plain drafter keeps working.
        {
            const fs::path mtp_dir = fs::path(models) / "mtp";
            const fs::path assets = fs::path(models).parent_path() / "assets" / "mtp";
            fs::path drafter, basis;
            if (fs::exists(mtp_dir))
                for (const auto & e : fs::directory_iterator(mtp_dir))
                    if (e.is_regular_file() && e.path().extension() == ".gguf" &&
                        e.path().stem().string().find(".draft-head-") == std::string::npos) { drafter = e.path(); break; }
            if (fs::exists(assets))
                for (const auto & e : fs::directory_iterator(assets))
                    if (e.is_regular_file() && e.path().extension() == ".basis") { basis = e.path(); break; }
            if (drafter.empty() || basis.empty()) {
                std::fprintf(stderr, "[prepare] two-stage drafter head SKIPPED: %s\n",
                             drafter.empty() ? "no drafter in models/mtp" : "no basis in assets/mtp");
            } else {
                const std::string tag = basis.stem().string();   // "draft-head-r192"
                const fs::path dh  = mtp_dir / (drafter.stem().string() + "." + tag + ".gguf");
                const fs::path tdh = mtp_dir / (drafter.stem().string() + "." + tag + ".new.part");
                if (!force && fs::exists(dh) && fs::file_size(dh) > 0) {
                    std::fprintf(stderr, "[prepare] %s exists -- SKIP (--force rebuilds)\n", dh.filename().string().c_str());
                } else {
                    std::fprintf(stderr, "[prepare] building %s ...\n", dh.filename().string().c_str());
                    const int rc = gguf_tools::make_draft_head(drafter, basis, tdh);
                    std::error_code ec;
                    if (rc == 0) {
                        fs::rename(tdh, dh, ec);
                        if (ec) throw std::runtime_error("prepare: cannot replace " + dh.string() + " (" + ec.message() +
                                                         ") -- is a server still running? run 'llapis stop' and prepare again");
                    } else {
                        fs::remove(tdh, ec);
                        if (rc != 2) throw std::runtime_error("prepare: two-stage drafter head FAILED (see message above)");
                    }
                }
            }
        }

        const bool have_base = fs::exists(base);
        std::printf("{\"base\":%s,\"arena_index\":\"%s\",\"arena_pack\":\"%s\","
                    "\"base_mib\":%lld,\"pack_mib\":%lld}\n",
                    have_base ? ("\"" + base.generic_string() + "\"").c_str() : "null",
                    index.generic_string().c_str(), pack.generic_string().c_str(),
                    have_base ? (long long) (fs::file_size(base) >> 20) : 0LL,
                    (long long) (fs::file_size(pack) >> 20));
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "prepare: ERROR: %s\n", e.what());
        return 1;
    }
}

// --- `moert-server plan`: pre-load planner mode -----------------------
// Scan models/high|low, validate the pair, probe free VRAM/RAM (NO model load), print
// the K plan JSON on stdout. The UI's "configure before load" data source. STRICT:
// any failure prints `plan: ERROR: <reason>` and exits 1 — never a silent fallback.
static int run_plan_mode(int argc, char ** argv) {
    try {
        std::string models = "models";      // portable default: models/ next to the exe cwd
        model_plan::PlanOptions po;
        // --terms: the launcher's quant-dependent VRAM terms for one mode, from the GGUF headers only
        // (no probe, no load): moert-server plan --terms --mode speed|quality --low F [--high F] [--drafter F]
        bool terms = false;
        std::string mode, high, low, drafter;
        for (int i = 2; i < argc; ++i) {
            const std::string a = argv[i];
            if (a == "--terms") { terms = true; break; }
        }
        if (terms) {
            for (int i = 2; i < argc; ++i) {
                const std::string a = argv[i];
                auto next = [&]() -> const char * {
                    if (i + 1 >= argc) throw std::runtime_error("plan: flag " + a + " needs a value");
                    return argv[++i];
                };
                if      (a == "--terms")   {}
                else if (a == "--mode")    mode    = next();
                else if (a == "--high")    high    = next();
                else if (a == "--low")     low     = next();
                else if (a == "--drafter") drafter = next();
                else throw std::runtime_error("plan: unknown flag with --terms: " + a +
                         " (usage: moert-server plan --terms --mode speed|quality --low F [--high F] [--drafter F])");
            }
            std::printf("%s\n", model_plan::terms_to_json(model_plan::compute_terms(mode, high, low, drafter)).c_str());
            return 0;
        }
        for (int i = 2; i < argc; ++i) {
            const std::string a = argv[i];
            auto next = [&]() -> const char * {
                if (i + 1 >= argc) throw std::runtime_error("plan: flag " + a + " needs a value");
                return argv[++i];
            };
            if      (a == "--models")  models  = next();
            else if (a == "--ctx")     po.ctx  = std::atoi(next());
            else if (a == "--k-min")   po.k_min = std::atoi(next());
            else if (a == "--k-max")   po.k_max = std::atoi(next());
            else if (a == "--reserve") po.compute_reserve_mib = std::atoll(next());
            else if (a == "--safe")    po.safe_mib  = std::atoll(next());
            else if (a == "--tight")   po.tight_mib = std::atoll(next());
            else throw std::runtime_error("plan: unknown flag: " + a +
                     " (usage: moert-server plan [--models dir] [--ctx N] [--k-min N] [--k-max N]"
                     " [--reserve MIB] [--safe MIB] [--tight MIB])");
        }
        model_plan::ModelPair mp = model_plan::scan_models_dir(models);
        model_plan::validate_pair(mp);
        model_plan::HwProbe hw = model_plan::probe_hw();
        model_plan::Plan    pl = model_plan::compute_plan(mp, hw, po);
        std::printf("%s\n", model_plan::plan_to_json(mp, hw, pl).c_str());
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "plan: ERROR: %s\n", e.what());
        return 1;
    }
}

int main(int argc, char ** argv) {
    // C0: install the log filter first so router/child std::cerr banners + backend/model logs
    // are captured to the full log and kept off a normal-mode terminal (no-op unless the wrapper
    // set LLAPIS_LOG_FILE). Re-asserted inside llama_server() after the fork's own common_init().
    llapis_log_filter_init();
#ifdef _WIN32
#endif
    if (argc >= 2 && std::strcmp(argv[1], "prepare") == 0) {
        return run_prepare_mode(argc, argv);   // pure file IO — no backend needed
    }
    if (argc >= 2 && std::strcmp(argv[1], "plan") == 0) {
        llama_backend_init();               // loads the CUDA backend => real VRAM probe
        const int rc = run_plan_mode(argc, argv);
        llama_backend_free();
        return rc;
    }

    // --- scan the command line for what the pre-load init needs (the server parses the
    //     full set itself; we only peek, never consume) ---
    std::string model_path, alias;
    bool        has_model_source = false;   // any way the server could load/list models itself
    bool        has_spec_type = false;      // caller picked a speculative drafter explicitly
    bool        has_mmproj = false;         // a multimodal projector is loaded -> the mtmd path is live
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if ((a == "-m" || a == "--model") && i + 1 < argc)      model_path = argv[i + 1];
        else if (a == "--alias" && i + 1 < argc)                alias = argv[i + 1];
        else if (a == "--models-preset" || a == "--models-dir" || a == "-hf" || a == "--hf-repo")
            has_model_source = true;
        else if (a == "--spec-type")
            has_spec_type = true;
        else if (a == "--mmproj" || a == "-mm")
            has_mmproj = true;
    }
    // the projector can also arrive via the profile env (LLAMA_ARG_MMPROJ) instead of an explicit flag
    if (std::getenv("LLAMA_ARG_MMPROJ")) has_mmproj = true;
    g_has_mmproj = has_mmproj;   // the ctx sink reads this to enable the vision-transient decode
    // Vision SWA headroom: a non-causal image is ONE ubatch and Gemma's iSWA sizes the SWA slot headroom
    // as n_swa + n_ubatch, so the loader cut's tiny n_ubatch made deep-n_past images fail find_slot. Set
    // through an exported setter (an env var does not cross the exe/DLL CRT boundary), before
    // llama_server() so every KV cache sees it; 512 covers the ~256-token chunk (1024 cost ~150 MiB).
    // The prefill decouple below may RAISE it (the larger requirement wins). (history: docs/code-comments-archive-2026-10-05.md, "vision-swa")
    uint32_t swa_ub_hint = 0;
    if (has_mmproj) {
        uint32_t vub = 512;
        llama_set_swa_ubatch_hint(vub);
        swa_ub_hint = vub;
    }
    const bool spawned_by_router = std::getenv("LLAMA_SERVER_ROUTER_PORT") != nullptr;

    // C5.2 single-instance guard: exactly ONE router and ONE model-loading process per
    // login session — a second full instance OOMs the box (16 GB RAM / 8 GB VRAM are
    // sized for one). Named-mutex handles are held for the process lifetime and released
    // by the OS on ANY exit (crash included) — no stale lockfiles possible. Exit 87 is
    // the dedicated "already running" code (the router's spawn gate knows it).
    if (!model_path.empty()) {
        if (!llapis_singleton_acquire("Local\\LLapis-model")) {
            std::fprintf(stderr,
                "moert-server: FATAL: another LLapis process already holds a model loaded "
                "(the RAM arena + VRAM budget are sized for ONE instance). Stop it first "
                "(run `llapis stop`, or close the other terminal).\n");
            return 87;
        }
    } else if (!spawned_by_router) {
        if (!llapis_singleton_acquire("Local\\LLapis-router")) {
            std::fprintf(stderr,
                "moert-server: FATAL: an LLapis router is already running in this session — "
                "refusing a second one. chat/serve reuse the running router on its port "
                "automatically, or stop it with `llapis stop`.\n");
            return 87;
        }
    }

    // --- M4 lifecycle glue ---------------------------------------
    // ROUTER mode: no model on the command line and no explicit model source. If
    // moert-profiles.json sits next to the exe, it becomes the model list (generated
    // preset INI) + /moe/plan provider. STRICT: a malformed file refuses to start.
    // Durable storage for injected argv (must outlive llama_server()).
    static std::string s_preset_flag = "--models-preset", s_preset_ini,
                       s_max_flag = "--models-max", s_max_val = "1";
    std::vector<char *> extra_args;
    if (model_path.empty() && !has_model_source && !spawned_by_router) {
        if (fs::exists(profiles_path())) {
            try {
                MoertProfiles profs = load_profiles();
                s_preset_ini = generate_preset_ini(profs);
                extra_args.push_back(s_preset_flag.data());
                extra_args.push_back(s_preset_ini.data());
                extra_args.push_back(s_max_flag.data());
                extra_args.push_back(s_max_val.data());
                // NOTE: --tools and --ui-mcp-proxy used to be injected HERE,
                // unconditionally. They are capability grants — builtin tools execute in
                // THIS router process, and the proxy makes outbound requests for the page —
                // so they now come from the launcher's -Tools / -McpProxy and are absent
                // unless the user asked. Nothing on /v1/chat/completions depends on either:
                // an agent harness sends its own tool definitions and runs them itself.
                g_plan_cmdline = "\"" + exe_path() + "\" plan --models \"" + profs.models_dir + "\"";
                llama_server_set_moe_plan_provider(&moe_plan_provider);
                llama_server_set_moe_profiles_save(&moe_profiles_save);
                // C5.2: admission gate + child-exit ledger (no blind respawns; see above)
                llama_server_set_moe_spawn_gate(&moe_spawn_gate);
                llama_server_set_moe_child_exit(&moe_child_exit_note);
                std::fprintf(stderr, "moert-server: ROUTER mode — %zu profile(s) from %s, "
                                     "starting UNLOADED (models-max 1)\n",
                             profs.profiles.size(), profiles_path().c_str());
            } catch (const std::exception & e) {
                std::fprintf(stderr, "moert-server: error: %s\n", e.what());
                return 1;
            }
        } else {
            std::fprintf(stderr, "moert-server: note: no -m and no %s — starting as a stock router "
                                 "(no moert profiles, no /moe/plan)\n", profiles_path().c_str());
        }
        if (std::getenv("MOE_DUAL_DYNAMIC")) {
            std::fprintf(stderr, "moert-server: note: MOE_DUAL_* env is set but this is the ROUTER "
                                 "process (no model) — the dual runtime initializes in the child, "
                                 "from the profile's env, not from yours\n");
        }
    }

    // Backend init BEFORE the child glue: child_auto_k probes the GPU via the ggml
    // device registry. The router process still never enumerates devices (no probe here;
    // /moe/plan runs in a subprocess), so its CUDA primary context stays uncreated.
    llama_backend_init();

    // CHILD mode (spawned by the router): resolve our profile by --alias and apply its
    // env BEFORE the dual init below reads it. STRICT: missing alias/profile = hard exit
    // (the router marks the model UNLOADED with our exit code; reason in the router log).
    if (spawned_by_router && !model_path.empty()) {
        try {
            if (alias.empty())
                throw std::runtime_error("moert-server: child has no --alias — cannot resolve its profile");
            MoertProfiles profs = load_profiles();
            if (!profs.profiles.contains(alias))
                throw std::runtime_error("moert-server: profile '" + alias + "' not found in " + profs.path);
            const json & prof = profs.profiles[alias];
            child_apply_profile_env(alias, prof);
#ifdef _WIN32
            // re-check the diag hook: the router's ps1 wrapper clears MOE_* from its own env, so a
            // child only sees MOE_DIAG_ABORT_STACK via the profile env applied just above.
#endif
            if (prof.value("auto_k", false)) child_auto_k(profs, prof);
            g_auto_unload_s = profs.auto_unload_s;
            g_child_alias   = alias;
            if (const char * rp = std::getenv("LLAMA_SERVER_ROUTER_PORT"); rp && rp[0])
                g_router_port = std::atoi(rp);
            std::fprintf(stderr, "moert-server: CHILD of router — profile '%s' applied "
                                 "(auto_k=%s, auto_unload_s=%lld)\n",
                         alias.c_str(), prof.value("auto_k", false) ? "true" : "false",
                         g_auto_unload_s);
            {
                const char * kk = std::getenv("MOE_DUAL_K");
                llapis_phase("Runtime parameters loaded: profile '%s' (K=%s%s)", alias.c_str(),
                             kk && kk[0] ? kk : "default",
                             prof.value("auto_k", false) ? ", planner auto-K" : "");
            }
        } catch (const std::exception & e) {
            std::fprintf(stderr, "moert-server: error: %s\n", e.what());
            return 1;
        }
    }

    // --- env-driven dual-precision runtime init (must run
    //     before the server loads the model so the expert hook is registered). Skipped
    //     in ROUTER mode (no -m): the dual runtime belongs to the child processes. ---
    bool dual = false;
    if (model_path.empty()) {
        // router / bare start: never initialize the dual runtime here
    } else
    if (const char * dual_q4 = std::getenv("MOE_DUAL_DYNAMIC"); dual_q4 && dual_q4[0]) {
        const char * ks = std::getenv("MOE_DUAL_K");
        // Fallback 32, not 48: a profile that loses its MOE_DUAL_K (the M6c fallback bug
        // class) must land on the gate-proven 8 GB fit, not the deprecated K=48 config.
        const int dual_k = (ks && ks[0]) ? std::atoi(ks) : 32;
        if (!moe_dual::init_dynamic(dual_q4, dual_k)) {
            // C5.2: an admission-preflight refusal is NOT a crash — exit 86 so the router's
            // spawn gate arms the long backoff instead of respawning into the same wall.
            if (moe_dual::preflight_failed()) {
                std::fprintf(stderr, "moert-server: exiting 86 (admission preflight refused — "
                                     "free the memory it named, or lower K/ctx)\n");
                llama_backend_free();
                return 86;
            }
            std::fprintf(stderr, "moert-server: warning: MOE_DUAL_DYNAMIC set but init failed; serving stock experts\n");
        } else {
            dual = true;
        }
    }

    // q4stream server prefill (C3): configured below inside the dual block (needs spec_k).

    std::vector<char *> args(argv, argv + argc);
    // M4 router glue: injected model source (--models-preset <generated ini> --models-max 1)
    args.insert(args.end(), extra_args.begin(), extra_args.end());
    std::string np_flag = "-np", np_val = "1", nowarm = "--no-warmup", ub_flag = "--ubatch-size", ub_val = "1";
    // Speculative decode K=2 drafter injection (durable storage, must outlive llama_server()).
    // ngram-simple size_n=3 == the validated mlen>=3 confidence gate (: speculate
    // only when the 3-gram ending at the sampled token recurs — structured targets GO, prose
    // protected structurally); size_m=3 is the drafter's structural minimum (it never proposes
    // fewer than size_n tokens), the fork-side draft cap below truncates the proposal to 1.
    std::string sp_type_flag = "--spec-type",                sp_type_val = "ngram-simple",
                sp_n_flag    = "--spec-ngram-simple-size-n", sp_n_val    = "3",
                sp_m_flag    = "--spec-ngram-simple-size-m", sp_m_val    = "3";
    if (dual) {
        // Routing observation on the server path is device-counter only (no cb_eval).
        if (!moe_dual::dev_counts_active()) {
            std::fprintf(stderr, "moert-server: error: the device routing counters could not be allocated -- "
                                 "refusing to run with blind routing (the cache would never adapt)\n");
            llama_backend_free();
            return 1;
        }

        // Speculative decode: the fused verifier handles n_tokens <= spec_kmax (4). Guardrails: draft cap =
        // spec_k - 1; a drafter injected unless --spec-type was given; under the loader cut --ubatch-size
        // spec_k so the verify batch stays ONE ubatch. Without MOE_DUAL_SPEC_K a --spec-type is refused.
        // (history: docs/code-comments-archive-2026-10-05.md, "spec-guards")
        const int spec_k = [] {
            const char * v = std::getenv("MOE_DUAL_SPEC_K");
            const int    k = (v && v[0]) ? std::atoi(v) : 1;
            // Ceiling 4 = the fused op's Pmax=32 / decode top_k=8 (matches moe_dual's clamp): verify up
            // to 4-wide (draft-n-max 3). k=2 stays the validated default; 3-4 need a matching n_ubatch.
            return k < 1 ? 1 : (k > 4 ? 4 : k);
        }();
        // -NoMtp: take the fused width, arm NO drafter. Without this the block at the launch args
        // below sees spec_k>=2, finds no caller --spec-type, and pushes the server's OWN ngram-simple
        // drafter -- so "speculation off" silently measured a different drafter (found live, findings 150).
        const bool spec_nodraft = [] { const char * v = std::getenv("MOE_DUAL_SPEC_NODRAFT"); return v && v[0] == '1'; }();
        g_spec_k_tel = spec_k;   // policy-telemetry phase tag (target-decode vs target-mtp)
        if (spec_k < 2 && has_spec_type) {
            std::fprintf(stderr, "moert-server: error: --spec-type with the dual runtime needs MOE_DUAL_SPEC_K>=2 "
                                 "(without it every multi-token verify batch silently runs the cold-precision "
                                 "stock expert path) — set MOE_DUAL_SPEC_K=2..4 or drop --spec-type\n");
            llama_backend_free();
            return 1;
        }
        if (spec_k >= 2) {
            llama_server_set_moe_spec_draft_max(spec_k - 1);
            ub_val = std::to_string(spec_k);   // the spec_k-wide verify batch must stay ONE ubatch under
                                               // the loader cut; consumed by the push below, inert without it
            // alpha-gated adaptive width: overrides the fixed cap, widening the draft only while
            // acceptance stays high (needs spec_k>=3 to have room to adapt; a no-op ladder at spec_k=2).
            // C5: the observer wrapper is ALWAYS registered so phase lines see live MTP acceptance;
            // without the adaptive controller it returns the fixed max width (behavior unchanged).
            g_spec_fixed_w = spec_k - 1;
            llama_server_set_moe_spec_draft_fn(&moe_spec_observe);
        }

        // Q4Stream server prefill: the TTFT lever and the prompt-KV precision lever. It coexists with vision:
        // the vision-room callback closes the phase before an image chunk (which bypasses pre_decode) and
        // re-allocates the staging after it. (history: docs/code-comments-archive-2026-10-05.md, "q4stream-vision")
        if (const char * src = std::getenv("MOE_DUAL_PREFILL_Q4SRC"); src && src[0]) {
            if (spec_k < 2) {
                std::fprintf(stderr, "moert-server: error: q4stream server prefill requires the speculative "
                                     "path (MOE_DUAL_SPEC_K>=2): backend sampling must be off or the "
                                     "layer-major merged prefill graph cannot run — enable the MTP drafter "
                                     "or unset MOE_DUAL_PREFILL_Q4SRC\n");
                llama_backend_free();
                return 1;
            }
            g_q4s_src    = src;
            g_q4s_spec_k = spec_k;
            int chunk = 64;   // prompt ubatch size (chunk only sets ubatch under layer-major)
            if (const char * v = std::getenv("MOE_DUAL_PREFILL_CHUNK"); v && v[0]) chunk = std::atoi(v);
            if (chunk < spec_k) chunk = spec_k;
            // The streamed-vs-fused threshold for a WARM server, readable. Default 128 (the
            // measured crossing); 1 restores the 2026-08-08 always-stream behaviour in both
            // regimes, a large value forces fused whenever the slab is resident. It cannot
            // force fused on a COLD server -- that costs 418 ms/token and is not a knob.
            if (const char * v = std::getenv("MOE_DUAL_PREFILL_MIN"); v && v[0]) {
                const int m = std::atoi(v);
                if (m >= 1) g_q4s_min = m;
            }
            // g_q4s_min (the streamed-vs-fused break-even) is independent of chunk: a prompt with
            // min <= n <= chunk still streams correctly through the single-ubatch build. The lean arm only keeps
            // graph_reserve on the streamed path. (history: docs/code-comments-archive-2026-10-05.md, "q4s-min-chunk")
            if (!moe_dual::q4stream_init(g_q4s_src.c_str(), "Q4STREAM", /*lean_ring=*/true)) {
                std::fprintf(stderr, "moert-server: error: q4stream init failed (see message above) — "
                                     "fix MOE_DUAL_PREFILL_Q4SRC or unset it\n");
                llama_backend_free();
                return 1;
            }
            moe_dual::q4stream_begin();   // armed through load: lean graph_reserve
            g_q4s_open = true;                                      // closed at the ctx sink below
            content_trace_init();   // bench content capture (opt-in, decode only)
            if (g_ct) llama_server_set_moe_batch_tokens(&content_trace_tokens);
            llama_server_set_moe_cb_eval(&moe_srv_q4s_eval_cb, nullptr);
            llama_server_set_moe_prompt_cap(&moe_q4s_prompt_cap);
            // P4 prefix reuse (MOE_DUAL_PREFIX_REUSE=1): streamed prompts stop spec_k tokens short so the
            // SWA checkpoint covers the prompt; the tail closes the phase and runs fused on the resident
            // slab (findings 181: neutral on the TTFT, opt-in).
            if (const char * pr = std::getenv("MOE_DUAL_PREFIX_REUSE"); pr && pr[0] == '1') {
                llama_server_set_moe_ckpt_tail(g_q4s_spec_k);
                std::fprintf(stderr, "moert-server: prefix reuse ON -- streamed prompts stop %d tokens short "
                                     "for the SWA context checkpoint (tail on the fused path)\n", g_q4s_spec_k);
            }
            g_q4s_pre_decode_wired = true;   // moe_srv_pre_decode (registered below) delegates here
            // PREFILL-UBATCH DECOUPLE: n_ubatch stays decode-sized (it sizes the PERMANENT graph reserve) while
            // `chunk` becomes the prefill-only split (llama_set_prefill_ubatch at the ctx sink; the layer-major
            // prefill runs on its own one-shot scheduler). n_ubatch >= spec_k (one verify ubatch); a wider split
            // needs SWA headroom, so the hint is raised here, before the context exists. (history: docs/code-comments-archive-2026-10-05.md, "decouple")
            bool decouple = false;
            if (const char * v = std::getenv("MOE_DUAL_PREFILL_DECOUPLE"); v && v[0] == '1') decouple = true;
            const int ub_decode = std::max(spec_k, 64);
            if (decouple && chunk > ub_decode) {
                g_q4s_prefill_ub = chunk;               // applied to the target ctx at the ctx sink
                ub_val = std::to_string(ub_decode);     // permanent reserve: decode-sized
                if ((uint32_t) chunk > swa_ub_hint) {   // trap 2: SWA headroom for the wide split
                    llama_set_swa_ubatch_hint((uint32_t) chunk);
                    swa_ub_hint = (uint32_t) chunk;
                }
                std::fprintf(stderr,
                    "moert-server: prefill-ubatch DECOUPLED: n_ubatch %d (decode reserve, >= spec_k %d) "
                    "+ prefill split %d (one-shot sched) + SWA hint %u — frees the permanent graph "
                    "reserve the prefill never used\n",
                    ub_decode, spec_k, chunk, swa_ub_hint);
            } else {
                ub_val = std::to_string(chunk);   // prompt ubatch; verify (<= spec_k) stays one ubatch
            }
            std::fprintf(stderr, "moert-server: q4stream server prefill ENABLED (chunk %d, warm min %d tokens, "
                                 "transient staging, source %s) — an uncached suffix of >= min tokens streams "
                                 "the full-layer Q4 pool once (layer-major), and so does ANY suffix while the "
                                 "hot slab is still evicted (first prompt of the session, 418 ms/token fused); "
                                 "a shorter suffix on a warm server runs the fused path with the phase closed\n",
                         chunk, g_q4s_min, g_q4s_src.c_str());
        }

        // Pre-decode gate: always registered under the dual runtime; the wrapper delegates to
        // the q4stream phase gate only when that was wired above.
        llama_server_set_moe_pre_decode(&moe_srv_pre_decode);
        // Abort path: a slot released WITHOUT generating never reaches on_prompt_end, so the q4stream phase is
        // closed here (else the staging eval-callback keeps asking for sync points). Idempotent.
        // (history: docs/code-comments-archive-2026-10-05.md, "turn-abort")
        llama_server_set_moe_turn_abort([]() {
            const bool was_open = moe_dual::q4stream_prompt_active();
            moe_q4s_close(/*log_stats=*/false);   // no stats line: this prefill produced no turn
            g_prompt_started = false;
            g_prompt_tok     = 0;
            if (was_open) {
                std::fprintf(stderr, "[moert-server] turn aborted during prefill — q4stream staging "
                                     "released (%lld MiB free)\n", moe_dual::pool_free_mib());
                std::fflush(stderr);
                llapis_phase("Turn cancelled while reading the prompt — prefill staging released");
            }
        });

        moe_dual::RuntimeInfo ri = moe_dual::runtime_info();
        set_moe_hooks_recording(
            [](int n) {
                // C5 phase: announce the prefill once per turn. The streamed q4stream case
                // announces itself in moe_q4s_pre_decode (which runs first); anything else
                // that reaches here un-announced is the fused per-token path.
                if (!g_prompt_started) {
                    g_prompt_started = true;
                    if (!g_q4s_open)
                        llapis_phase("Fused prefill ... (per-token path%s)",
                                     g_q4s_src.empty()
                                         ? " — no promotion source, so q4stream is off"
                                         : " — the streamed path was NOT taken; raise the log level to see why");
                }
                g_prompt_tok += n;
                // Decision telemetry: stamp the phase BEFORE the policy call, so a record written
                // from inside it carries the phase that produced its fold window (no-op when off).
                moe_dual::on_prompt_token(n);
                g_last_activity_ms.store(lifecycle_now_ms()); },   // M4 heartbeat: prefill = alive
            []()      {
                // everything in this lambda runs AFTER the server stops the prompt clock and BEFORE
                // the first token, so `timings.prompt_ms` excludes it while a user waits through it:
                // timed here and reported as `prompt_end.*` on /moe/info and in the PHASE line.
                using pclk = std::chrono::steady_clock;
                const pclk::time_point pe0 = pclk::now();
                auto ms_since = [](pclk::time_point t) {
                    return std::chrono::duration<double, std::milli>(pclk::now() - t).count();
                };
                moe_q4s_close(/*log_stats=*/true);   // q4stream: end phase (the staging stays: fixed reserve)
                // Re-fit K to the measured VRAM (a no-op inside the hysteresis): quiescent, q4stream off.
                const double pe_q4s_ms = ms_since(pe0);
                moe_dual::vkv_restore_or_refit();
                const double pe_slab_ms = ms_since(pe0) - pe_q4s_ms;
                const long long promos0 = moe_dual::runtime_info().refresh_promotions;
                const double pe_refresh0 = ms_since(pe0);
                moe_dual::refresh_cache(moe_dual::REFRESH_PROMPT_END);
                const double pe_refresh_ms = ms_since(pe0) - pe_refresh0;
                const moe_dual::RuntimeInfo r = moe_dual::runtime_info();
                llapis_phase("Loading hot experts into VRAM, K=%d ... (prompt-end refresh: %lld promoted, "
                             "pause %.0f ms, cache fill %.0f%%)",
                             r.hot_k, r.refresh_promotions - promos0, r.last_refresh_pause_ms,
                             100.0 * r.occupancy);
                // Publish the window for /moe/info + the gate's first-token accounting.
                g_pe_slab_ms.store(pe_slab_ms);
                g_pe_refresh_ms.store(pe_refresh_ms);
                g_pe_total_ms.store(ms_since(pe0));
                llapis_phase("Hot-expert residency materialized in %.0f ms (slab %.0f + refresh %.0f"
                             "; q4stream close %.0f) — this lands AFTER prompt_ms and BEFORE the first token",
                             g_pe_total_ms.load(), pe_slab_ms, pe_refresh_ms, pe_q4s_ms);
                llapis_phase("Decoding started (%lld prompt tokens processed)", g_prompt_tok);
                g_prompt_started = false;
                g_prompt_tok = 0;
                g_mtp_proposed.store(0);             // per-turn MTP acceptance for the refresh lines
                g_mtp_accepted.store(0);
                g_last_refresh_seen = r.refresh_count;
                g_ref_gen_tok = llama_server_moe_gen_tokens();   // emitted-rate baseline for the refresh lines
                g_ref_gen_ms  = lifecycle_now_ms();
                g_ref_line_vt     = r.visible_tokens;
                g_ref_line_promos = r.refresh_promotions;
                g_ref_line_pause  = r.refresh_pause_ms_total;
                g_dec0 = r;                          // visible-decode baseline (run_turn tel_dec0)
                g_early_done = false;                // P3 gate: the first EARLY_TOKENS of this turn
            },
            []()      {
                // Phase tag: a turn with an active drafter is target-MTP, otherwise plain
                // target-decode. Both are the VISIBLE phase — the distinction the analysis needs
                // is which cost model applies (verify width > 1 changes cold pairs per fire).
                moe_dual::on_token_end();
                g_last_activity_ms.store(lifecycle_now_ms());   // M4 heartbeat: decoding = alive
                // fire exactly once, on the first decode token this process ever
                // emits — after this, on_token_end's own throttled refresh (16 tok) is enough.
                dump_vram_breakdown_once();
                const moe_dual::RuntimeInfo r = moe_dual::runtime_info();
                // C5 phase: the live ">>> Refresh" heartbeat, one line per REFRESH_LINE_EVERY visible tokens
                // summing the per-token decisions since the last line.
                if (r.refresh_count != g_last_refresh_seen &&
                    r.visible_tokens - g_ref_line_vt >= REFRESH_LINE_EVERY) {
                    g_last_refresh_seen = r.refresh_count;
                    const long long line_promos = r.refresh_promotions - g_ref_line_promos;
                    const double    line_pause  = r.refresh_pause_ms_total - g_ref_line_pause;
                    g_ref_line_vt     = r.visible_tokens;
                    g_ref_line_promos = r.refresh_promotions;
                    g_ref_line_pause  = r.refresh_pause_ms_total;
                    const long long mp = g_mtp_proposed.load(), ma = g_mtp_accepted.load();
                    char mtp[32];
                    if (mp > 0) std::snprintf(mtp, sizeof(mtp), "%.0f%%", 100.0 * ma / mp);
                    else        std::snprintf(mtp, sizeof(mtp), "n/a");
                    // tok/s = EMITTED tokens (incl. accepted MTP drafts) since the last refresh —
                    // the policy window counts decode cycles and undercounts under speculation
                    const long long gen_now = llama_server_moe_gen_tokens();
                    const long long now_ms  = lifecycle_now_ms();
                    const double    dt_s    = (now_ms - g_ref_gen_ms) / 1000.0;
                    const double    rate    = dt_s > 0.05 ? (gen_now - g_ref_gen_tok) / dt_s : 0.0;
                    g_ref_gen_tok = gen_now;
                    g_ref_gen_ms  = now_ms;
                    // the RAM the exclusive arena is holding free (pages of VRAM-resident experts) --
                    // 0 when it is off.
                    const long long ram_freed = (r.cold_arena_mib > 0 && r.arena_committed_mib > 0)
                                              ? r.cold_arena_mib - r.arena_committed_mib : 0;
                    llapis_phase(">>> Refresh #%lld: %lld experts swapped (pause %.0f ms) | %.1f tok/s | "
                                 "hit %.0f%% | MTP accept %s | RAM freed %lld MiB",
                                 r.refresh_count, line_promos, line_pause,
                                 rate, 100.0 * r.win_hit_rate, mtp, ram_freed);
                }
                // P3 gate: realized hit over the first EARLY_TOKENS visible tokens of the turn -- the
                // tokens the boundary decision (residency chosen from the prompt) is meant to serve
                if (!g_early_done && r.visible_tokens - g_dec0.visible_tokens >= EARLY_TOKENS) {
                    g_early_done = true;
                    g_early_hot  = r.hot_routes  - g_dec0.hot_routes;
                    g_early_cold = r.cold_routes - g_dec0.cold_routes;
                }
                // M2: refresh the partition cache from the slot thread, throttled (16 tok)
                const long long vt = r.visible_tokens;
                if (vt - g_part_tok >= 16) { g_part_tok = vt; refresh_partition(); }
            },
            []()      {
                moe_dual::on_turn_end();
                refresh_partition();   // M2: idle snapshot reflects the finished turn
                const moe_dual::RuntimeInfo r = moe_dual::runtime_info();
                const long long dh = r.hot_routes - g_dec0.hot_routes;
                const long long dc = r.cold_routes - g_dec0.cold_routes;
                if (!g_early_done) { g_early_hot = dh; g_early_cold = dc; }   // a turn shorter than the window
                std::fprintf(stderr,
                    "[moert-server] turn telemetry (visible decode): hit-rate %.1f%% (hot %lld / cold %lld), "
                    "first-%d hit %.1f%% (hot %lld / cold %lld), "
                    "promo %lld demo %lld, refresh-promo %lld, occupancy %.1f%%, session tokens %lld, "
                    "vram free %lld MiB, wddm shared %lld MiB, cuda pools %lld MiB (%lld used)\n",
                    (dh + dc) > 0 ? 100.0 * dh / (dh + dc) : 0.0, dh, dc,
                    EARLY_TOKENS, (g_early_hot + g_early_cold) > 0 ? 100.0 * g_early_hot / (g_early_hot + g_early_cold) : 0.0,
                    g_early_hot, g_early_cold,
                    r.promotions - g_dec0.promotions, r.demotions - g_dec0.demotions,
                    r.refresh_promotions - g_dec0.refresh_promotions,
                    100.0 * r.occupancy, r.tokens,
                    // shared segment + scratch pools printed per turn (the pools only grow; per turn is where the deep-
                    // prompt allocations happen). (history: docs/code-comments-archive-2026-10-05.md, "shared-per-turn")
                    g_part.vram_free, g_part.vram_shared_ours,
                    g_part.vram_pool_reserved, g_part.vram_pool_used);
                std::fflush(stderr);   // stderr is a redirected FILE (fully buffered) under the server
                {
                    const long long mp = g_mtp_proposed.load(), ma = g_mtp_accepted.load();
                    char mtp[32];
                    if (mp > 0) std::snprintf(mtp, sizeof(mtp), "%.0f%%", 100.0 * ma / mp);
                    else        std::snprintf(mtp, sizeof(mtp), "n/a");
                    llapis_phase("Turn done: hit %.1f%% | cache fill %.0f%% | MTP accept %s | session tokens %lld",
                                 (dh + dc) > 0 ? 100.0 * dh / (dh + dc) : 0.0,
                                 100.0 * r.occupancy, mtp, r.tokens);
                }
                g_prompt_started = false;   // next request announces its prefill again
            },
            []()      {
                // server ready: the startup can_seq_rm probe decoded 2 dummy tokens through the
                // runtime — drop its device-counter routes so request #1's first refresh sees
                // exactly the app-path scores. Do NOT reset_stats() here: it
                // wipes the policy's seed-prior scores too (measured: prompt-end refresh then
                // evicts 22 seeded residents that the app path keeps — parity gate divergence).
                moe_dual::discard_pending_routes();
                // stage 5 — after the startup probe. The gap from stage 4 is what the probe costs.
                std::fprintf(stderr, "[moert-server] [load-peak] stage 5 startup probe done: %lld MiB free, "
                                     "shared %lld MiB\n",
                             moe_dual::pool_free_mib(), mem_partition::wddm_nonlocal_usage_mib());
                // BOOT SLAB (findings 183). The slab sat at the K floor through the load peaks; now the
                // staging is allocated as the prefill's fixed reserve and K is fitted to the measured VRAM
                // around it, with the slab left EMPTY: the first prompt streams (an empty slab is not
                // warm) and its boundary decisions fill the slab from the staging. One flow in every
                // configuration -- with variable-KV off, K is fitted here once.
                if (moe_dual::is_dynamic()) {
                    const auto b0 = std::chrono::steady_clock::now();
                    if (!g_q4s_src.empty() && !moe_dual::q4stream_available() &&
                        !moe_dual::q4stream_init(g_q4s_src.c_str(), "Q4STREAM"))
                        std::fprintf(stderr, "[moert-server] boot slab: the staging could not be allocated -- "
                                             "prompts take the fused per-token prefill\n");
                    moe_dual::vkv_refit_k(g_vkv_keep_post);
                    const moe_dual::RuntimeInfo rb = moe_dual::runtime_info();
                    std::fprintf(stderr, "[moert-server] boot slab: staging %lld MiB + EMPTY slab at K=%d allocated "
                                         "before the first prompt in %.0f ms (%lld MiB free)\n",
                                 (long long) (moe_dual::q4stream_stats().buf_bytes >> 20), rb.hot_k,
                                 std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - b0).count(),
                                 moe_dual::pool_free_mib());
                    llapis_phase("Hot slab ready, K=%d, empty -- the first prompt fills it from its own prefill",
                                 rb.hot_k);
                    arena_ws_boot();
                    refresh_partition();
                    std::fflush(stderr);
                }
                // measure device-wide VRAM now that model+context are in place (post the boot
                // slab release, so the number documents the true idle state). Measure-only: the
                // server path does not enforce the VRAM target here (only the init-time K clamp).
                moe_dual::measure_vram_post_load();
                refresh_partition();
                // C5.6: publish the WDDM shared-segment baseline: ~0.5-0.9 GiB there is normal (host memory the GPU
                // may address); only GROWTH above it is the overcommit signature. The POST-LOAD reading is the
                // reference -- the segment legitimately grows ~770 MiB during the load. (history: docs/code-comments-archive-2026-10-05.md, "shared-baseline")
                g_vram_shared_baseline_mib.store(g_part.vram_shared_ours);
                {
                    long long budget = 0, usage = 0;
                    const bool have_budget = mem_partition::wddm_local(&budget, &usage);
                    std::fprintf(stderr, "[moert-server] boot baseline: this process's WDDM shared-segment "
                                         "usage = %lld MiB (was %lld before the first device allocation — the "
                                         "delta is pinned host memory: the prefill's pinned staging slices, plus the "
                                         "input embeddings unless --no-host; not a spill); dedicated %lld of a %lld "
                                         "MiB per-process budget\n",
                                 g_part.vram_shared_ours, moe_dual::nonlocal_boot_mib(),
                                 have_budget ? usage : -1, have_budget ? budget : -1);
                    // the honest spill test is BUDGET PRESSURE, not shared-segment size.
                    // WDDM only demotes once this process's dedicated usage reaches its budget; below
                    // that, a large shared segment is just host memory the GPU may address. Warn on
                    // the condition that actually precedes a demotion, so the alarm means something.
                    if (have_budget && budget > 0 && usage >= budget - 128) {
                        std::fprintf(stderr, "[moert-server] WARNING: this process is at its WDDM dedicated "
                                             "budget (%lld / %lld MiB). Further allocations will be demoted to "
                                             "system RAM and read over PCIe. K is refit down automatically; "
                                             "close other GPU apps or lower -VramBudget / -Ctx to keep it up.\n",
                                     usage, budget);
                    }
                }
                std::fflush(stderr);
                {
                    const moe_dual::RuntimeInfo r = moe_dual::runtime_info();
                    llapis_phase("Runtime ready — waiting for input (VRAM %lld / %lld MiB in use, "
                                 "hot cache fill %.0f%%)",
                                 r.vram_used_post_load_mib, r.vram_total_mib, 100.0 * r.occupancy);
                }
                std::fprintf(stderr, "[moert-server] ready: startup-probe device counters discarded\n");
                // CPU cold sidecar startup note: mirrors /moe/info.
                if (const char * cc = std::getenv("MOE_DUAL_COLD_CPU"); cc && cc[0] && cc[0] != '0') {
                    const char * mx = std::getenv("MOE_DUAL_COLD_CPU_MAX");
                    const bool   cs = std::getenv("MOE_DUAL_COPYSTREAM") && std::getenv("MOE_DUAL_COPYSTREAM")[0] == '1';
                    std::fprintf(stderr,
                        "[moert-server] CPU cold sidecar ENABLED: max=%s copy_stream=%s "
                        "(cold pairs computed on host, weights never cross PCIe)\n",
                        mx ? mx : "3", cs ? "on" : "off");
                }
                std::fflush(stderr);
                // M4: arm the heartbeat watchdog only now — the load window (arena pin,
                // mmap warm) must not count as "nobody watching"
                start_watchdog();
                start_mem_timer();   // C5: 5 s system-level mem re-probe for the UI panel
            });
        // M1 observability: GET /moe/telemetry + /moe/info (the server registers the routes
        // only because this provider is set — a stock llama-server keeps no /moe/*)
        llama_server_set_moe_json_provider(&moe_json_provider);
        // Variable-KV S6a: POST /moe/rebuild_k schedules a hot-slab rebuild at a new uniform K;
        // moe_dual::on_turn_end applies it on the slot thread at the next turn boundary.
        llama_server_set_moe_rebuild_k(&moe_dual::request_rebuild_k);
        // Variable-KV S6b: POST /moe/rebuild_ctx schedules a context rebuild at a new n_ctx (KV
        // replayed); the fork applies it at the all-idle quiescent point of update_slots.
        llama_server_set_moe_rebuild_ctx(&moe_ctx_rebuild_schedule, &moe_ctx_rebuild_take, &moe_ctx_rebuild_done);
        // arm q4stream staging for the reserve at the point of MAXIMUM available
        // VRAM during a rebalance — right after rebalance_ctx frees the old context, before it
        // creates the new one — so the admission check inside q4stream_init never needs to evict
        // the slab to fit its own staging buffer (the remaining gap in the fix).
        llama_server_set_moe_pre_ctx_create(&vkv_prepare_ctx_swap);
        // Variable-KV Phase 3: the automatic capacity policy (MOE_DUAL_VKV=1). The fork asks at
        // task launch; vkv_policy decides the ctx; K is re-fitted to the measured VRAM after.
        vkv_init_from_env();
        llama_server_set_moe_plan_ctx(&vkv_plan_ctx);
        // M2: read-only llama handles for the memory-partition probes (KV bytes, weights);
        // first snapshot right away = the post-load idle state
        // the speculative handles arrive on their own sink, fired after the drafter
        // is built (and again after every ctx rebuild, possibly with nullptr when the rebuild
        // dropped speculative decoding — so this can never keep a pointer to a freed context).
        llama_server_set_moe_spec_ctx_sink([](llama_context * cd, const llama_model * md) {
            g_obs_ctx_dft = cd; g_obs_model_dft = md;
            std::fprintf(stderr, "[moert-server] speculative handles %s (drafter weights: %s)\n",
                         cd ? "registered" : "CLEARED (speculative off)",
                         md ? "separate model" : "shared with target");
            // stage 4 — THE LOAD PEAK. The drafter loads LAST, with the hot slab,
            // the NE weights and the target KV all already resident, so this is the high-water
            // mark the admission preflight and the launcher's K clamp both exist to protect.
            // proposed reordering to drop it; this is the number that decides whether that
            // is worth anything.
            std::fprintf(stderr, "[moert-server] [load-peak] stage 4 drafter loaded: %lld MiB free, "
                                 "shared %lld MiB\n",
                         moe_dual::pool_free_mib(), mem_partition::wddm_nonlocal_usage_mib());
            std::fflush(stderr);
        });
        llama_server_set_moe_ctx_sink([](llama_context * c, const llama_model * m) {
            g_obs_ctx = c; g_obs_model = m;
            g_obs_n_ctx = c ? (int) llama_n_ctx(c) : 0;
            llapis_phase("Model context created (ctx=%d) — NE weights + KV cache are in place", g_obs_n_ctx);
            refresh_partition();
            // q4stream: graph_reserve is done — disarm + free the lean load-time staging (on_ready
            // allocates the real one) and enable layer-major prefill on the target context
            // (it only engages when a decode splits into multiple ubatches, i.e. prompts).
            if (!g_q4s_src.empty()) {
                moe_q4s_close(/*log_stats=*/false, /*force_release=*/true);
                if (c) llama_set_prefill_layer_major(c, true);
                // prefill-ubatch decouple: split prompts into g_q4s_prefill_ub-sized chunks while the
                // context's n_ubatch (and its permanent graph reserve) stays decode-sized. 0 = off.
                if (c && g_q4s_prefill_ub > 0) llama_set_prefill_ubatch(c, (uint32_t) g_q4s_prefill_ub);
            }
            // vision-transient (C4b): non-causal image chunks decode on the one-shot scheduler
            // (transient compute buffer, CPU expert matmuls) instead of needing a permanent
            // big-ubatch reserve — see the vision block in main.
            if (g_has_mmproj && c) llama_set_vision_transient(c, true);
            // The slab stays RESIDENT through the startup probe (the probe must run the fused path: the stock
            // loader-cut path at n<=spec_k crashes) but small: shrunk to the floor before the drafter load, so
            // 2.9 GiB of never-written VRAM is not held across the two load peaks (WDDM demoted our buffers to
            // make room for the drafter). on_ready then fits K (the boot slab). (history: docs/code-comments-archive-2026-10-05.md, "boot-shrink")
            {
                const long long free_before = moe_dual::pool_free_mib();
                const long long nl_before   = mem_partition::wddm_nonlocal_usage_mib();
                moe_dual::vkv_fit_slab_for_ctx_swap(-1);
                std::fprintf(stderr, "[moert-server] [load-peak] stage 3b boot slab shrunk to the floor "
                                     "before the drafter load: %lld -> %lld MiB free, shared %lld -> %lld MiB\n",
                             free_before, moe_dual::pool_free_mib(),
                             nl_before, mem_partition::wddm_nonlocal_usage_mib());
                mem_partition::wddm_dump("stage3b-ctx-sink");
                std::fflush(stderr);
            }
            {
                const char * sk = std::getenv("MOE_DUAL_SPEC_K");
                const bool spec = sk && std::atoi(sk) >= 2;
                llapis_phase("%s ... (then the 2-token startup probe)",
                             spec ? "Loading MTP drafter (spec model)" : "Finalizing server startup");
            }
            std::fprintf(stderr, "[moert-server] ctx sink fired: ctx=%p q4s_src=%s layer_major=%s vision_transient=%s\n",
                         (void *) c, g_q4s_src.empty() ? "EMPTY" : "set",
                         (!g_q4s_src.empty() && c) ? "ON" : "off",
                         (g_has_mmproj && c) ? "ON" : "off");
            std::fflush(stderr);
        });

        // single slot + no warmup forward: appended last so they win over anything earlier
        args.push_back(np_flag.data()); args.push_back(np_val.data());
        args.push_back(nowarm.data());
        std::fprintf(stderr, "moert-server: dual runtime ACTIVE (K=%d, %s) — forcing -np 1 (per-session "
                             "hot-set needs a single slot) and --no-warmup (routing-score purity)\n",
                     ri.hot_k, ri.dynamic ? "dynamic" : "static pack");
        if (spec_k >= 2 && !spec_nodraft) {   // -NoMtp keeps the width, arms no drafter (findings 150)
            if (!has_spec_type) {
                args.push_back(sp_type_flag.data()); args.push_back(sp_type_val.data());
                args.push_back(sp_n_flag.data());    args.push_back(sp_n_val.data());
                args.push_back(sp_m_flag.data());    args.push_back(sp_m_val.data());
            }
#ifdef _WIN32
            // reflected by /moe/info like the other MOE_* facts (child mode clears MOE_* first)
            _putenv_s("MOE_DUAL_SPEC_DRAFTER", has_spec_type ? "custom" : "ngram-simple");
#endif
            std::fprintf(stderr, "moert-server: speculative decode ENABLED (K=%d, drafter=%s, draft cap %s%d) — "
                                 "verify runs up to %d-wide through the fused dual-tier op; prose is protected by "
                                 "the 3-gram confidence gate (no match -> normal decode)\n",
                         spec_k, has_spec_type ? "caller --spec-type" : "ngram-simple n=3",
                         "", spec_k - 1, spec_k);
        }
        // Vision (TRANSIENT): a non-causal image must be ONE ubatch; raising --ubatch-size reserved ~2.7 GiB
        // forever. The fork's vision-transient path runs the image on the one-shot scheduler instead (buffer
        // freed after the chunk, CPU expert matmuls), so the context keeps its lean ubatch.
        // (history: docs/code-comments-archive-2026-10-05.md, "vision-transient")
        if (has_mmproj) {
            // Vision-room: the image chunk's transient graph measured 2735 MiB on CUDA (81-256
            // tokens; dominated by the sched's split-input copies, which gallocr never reuses).
            // At full K there is no such headroom, so the room callback EVICTS the hot slab for
            // the duration of the image decode and bit-identically re-streams the residents after
            // (~1.5-2.5 s from the Q4 source). Small-K configs skip the evict via the free-VRAM
            // threshold. MOE_DUAL_VISION_FREE_MIB overrides the threshold (0 = never evict).
            llama_set_vision_room(+[](int need_room) -> int {
                static const long long min_free_mib = 3100;   // measured on the product config
                if (need_room) {
                    llapis_phase("Loading compute buffer for image processing ... (transient decode; "
                                 "q4stream phase closed, hot slab evicted if VRAM is short)");
                    // The image chunk's expert matmuls run on the CPU straight from the cold IQ2 arena,
                    // so every released (exclusive-arena) page must be rebuilt first; a no-op otherwise.
                    // Fires for EVERY image (before the evict-skip early return below).
                    moe_dual::arena_vision_full_commit();
                    // C4e phase discipline: the image chunk bypasses moe_q4s_pre_decode, so an
                    // open q4stream phase would make the image graph compute against staging with
                    // no re-arm (the C3.3 garbage class). Close it HERE — phase off + staging
                    // freed (extra room for the transient buffer); re-allocated after the image.
                    moe_q4s_close(/*log_stats=*/true, /*force_release=*/true);   // image: max room, as before
                    if (min_free_mib <= 0) return 1;   // eviction disabled — try the decode as-is
                    return moe_dual::slab_evict(min_free_mib) ? 1 : 0;
                }
                // The staging first, while the slab is still out (it is the fixed reserve K was fitted
                // around), then the slab at the same K: the VRAM both held before the image.
                if (!g_q4s_src.empty() && !moe_dual::q4stream_available() &&
                    !moe_dual::q4stream_init(g_q4s_src.c_str(), "Q4STREAM"))
                    std::fprintf(stderr, "[moert-server] vision: the prefill staging could not be re-allocated "
                                         "after the image -- the next prompt retries it\n");
                moe_dual::slab_restore();   // no-op if the evict was skipped
                return 1;
            });
            // Lazy projector (LLAPIS_MMPROJ_LAZY=1, claude dump findings 257): the projector's ~1.1 GiB of weights sit
            // in VRAM only while an image is encoded, so between images they belong to the expert cache. Before the
            // reload the room is made the same way as for the image decode just after it: nothing when enough VRAM
            // is free (a tight budget keeps it free anyway), else the q4stream phase closed and the hot slab evicted;
            // the vision-room callback above restores both after the image.
            mtmd_llapis_set_weights_room(+[](long long need_mib) -> int {
                const long long want = need_mib + 512;
                if (moe_dual::pool_free_mib() >= want) return 1;
                moe_dual::arena_vision_full_commit();
                moe_q4s_close(/*log_stats=*/true, /*force_release=*/true);
                return moe_dual::slab_evict(want) ? 1 : 0;
            });
            std::fprintf(stderr, "moert-server: vision: TRANSIENT image decode active — non-causal image "
                                 "chunks run on a one-shot scheduler (transient compute buffer, CPU expert "
                                 "matmuls); context ubatch stays %s; the hot slab is evicted+restored around "
                                 "each image when free VRAM is short\n", ub_val.c_str());
        }
        if (const char * nea = std::getenv("MOE_DUAL_NO_EXPERT_ALLOC"); nea && nea[0] && nea[0] != '0') {
            args.push_back(ub_flag.data()); args.push_back(ub_val.data());
            std::fprintf(stderr, "moert-server: loader cut active — forcing --ubatch-size %s (lean fused "
                                 "graph_reserve; avoids the ~700 MiB shell staging reserve%s)\n",
                         ub_val.c_str(), spec_k >= 2 ? "; matches spec_k so the verify batch stays one ubatch" : "");
        }
    }

    // C0: re-assert the LLapis log filter after the fork server's own common_init() (no-op unless
    // LLAPIS_LOG_FILE is set). Pointer crosses the DLL boundary; same pattern as the moe hooks.
    llama_server_set_post_common_init(&llapis_log_filter_init);

    // C5 phase: the next big step inside llama_server() is the model load — the non-expert (NE)
    // tensors go to VRAM (experts are loader-cut shells served by moe_dual). Announced here
    // because the load happens inside the fork; completion is the ctx-sink phase line.
    if (dual) llapis_phase("Loading NE into VRAM ... (non-expert weights; experts stay loader-cut)");

    return llama_server((int) args.size(), args.data());
}
