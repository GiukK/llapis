#include "moe_dual.hpp"
#include "moe_dual_state.hpp"

#include "io_util.hpp"

#include "gguf.h"
#include "llama.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <regex>
#include <string>
#include <vector>

// Q4Stream staging reads bypass the page cache on Windows (FILE_FLAG_NO_BUFFERING): the 13.4 GiB
// pool cannot fit in it next to the no-mmap IQ2 heap, so buffered freads pay eviction + memcpy for
// nothing (measured 1.2 GB/s vs the NVMe's 2-3.5).
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <psapi.h>            // arena-source residency evidence (page faults / working set)
#pragma comment(lib, "psapi.lib")
#endif

namespace {
// Page faults charged to this process so far, and the current working set in MiB. Used to tell a
// RESIDENT arena (copies cost ~0 faults) from a committed-but-trimmed one (~1 fault per 4 KiB).
struct ProcMem { long long faults = 0; long long ws_mib = 0; };
static ProcMem proc_mem() {
#if defined(_WIN32)
    PROCESS_MEMORY_COUNTERS pmc{};
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc)))
        return { (long long) pmc.PageFaultCount, (long long) (pmc.WorkingSetSize >> 20) };
#endif
    return {};
}
}  // namespace

// ===========================================================================
// Q4Stream / BF16Stream prefill: batched prompt chunks computed against ONE full
// layer of high-precision experts, staged just-in-time into a single reused VRAM
// buffer at each layer's ffn_moe_topk sync point. Type-generic over the source
// precision; loud failure on any gap (never degrades silently). Public contract
// and call sequence: moe_dual.hpp §Q4Stream. Measured:
// ===========================================================================

namespace moe_dual {

// GGUF multi-file splits name shards "<prefix>-NNNNN-of-MMMMM.gguf" (llama.cpp's own convention).
// If `path` matches, return every shard path in order 1..count; otherwise return {path} unchanged
// (the common single-file case). Needed for real-world sources: a shard boundary can land MID
// expert set, and a source that ignored that would silently miss a tensor (measured on a 2-shard
// full-precision GGUF: layer 29's down_exps.weight was the sole tensor in shard 2).
static std::string zero_pad(int v, int width) {
    std::string s = std::to_string(v);
    if ((int) s.size() < width) s = std::string(width - s.size(), '0') + s;
    return s;
}
static std::vector<std::string> gguf_split_shards(const std::string & path) {
    namespace fs = std::filesystem;
    const fs::path p(path);
    const std::string fname = p.filename().string();
    static const std::regex re(R"(^(.*)-(\d+)-of-(\d+)\.gguf$)", std::regex::icase);
    std::smatch m;
    if (!std::regex_match(fname, m, re)) return { path };
    const std::string prefix = m[1].str();
    const int width = (int) m[2].str().size();
    const int count = std::atoi(m[3].str().c_str());
    if (count <= 0) return { path };
    std::vector<std::string> out;
    const fs::path dir = p.parent_path();
    for (int i = 1; i <= count; ++i)
        out.push_back((dir / (prefix + "-" + zero_pad(i, width) + "-of-" + zero_pad(count, width) + ".gguf")).string());
    return out;
}

// VRAM admission: on Windows cudaMalloc does NOT fail when VRAM is short -- WDDM backs the buffer with
// SHARED system RAM and every staging read/write crosses PCIe. True when `staging_bytes` would not fit
// in dedicated free VRAM with a compute-graph margin; the staging is then refused (fused per-token
// prefill), never paid for by evicting the slab (findings 179, 183). (history: docs/code-comments-archive-2026-10-05.md, "q4s-admission")
static bool q4s_vram_would_overcommit(size_t staging_bytes) {
    if (!g.backend) return false;   // no device view yet -> let the alloc itself decide (first-init path)
    const long long margin_mib = 384;   // headroom for the prefill compute graph + WDDM soft-budget slack
    ggml_backend_dev_t dev = ggml_backend_get_device(g.backend);
    if (!dev) return false;
    size_t free_b = 0, total_b = 0;
    ggml_backend_dev_memory(dev, &free_b, &total_b);
    const long long free_mib = (long long) (free_b >> 20);
    const long long need_mib = (long long) (staging_bytes >> 20) + margin_mib;
    return free_mib < need_mib;
}

// R1 ring lifecycle (defined below, next to the prefetch thread they drive).
static void q4s_ring_stop();
static void q4s_ring_start();

// --- arena-sourced staging (MOE_DUAL_Q4S_ARENA, default in speed) ---------------------------------
// When the staging source is the model's own file, every byte the prefill reads is also held by the
// owned cold arena, so the staging copies can come from RAM. Each (layer, kind, weight/scale) group is
// mapped only after its arena bytes are proven identical to the file's on a sample of experts; one
// mismatch disarms the whole mechanism. Worth it only while the arena is RESIDENT -- a trimmed arena
// pays a page-file read per 4 KiB (the residency trip below; findings 179). Verified once per process
// (s_arena_proven); the pointer map is rebuilt every init. (history: docs/code-comments-archive-2026-10-05.md, "arena-source")
static int s_arena_proven = -1;   // -1 not yet verified, 0 disproven (stay on the file), 1 proven
static void q4s_arm_arena(const char * label) {
    g.q4s_arena_mode = 0;
    const int n_layers = (int) g.q4s_wt.size();
    g.q4s_ar.assign(n_layers, { nullptr, nullptr });      g.q4s_ar_s.assign(n_layers, { nullptr, nullptr });
    g.q4s_ar_str.assign(n_layers, { 0, 0 });              g.q4s_ar_str_s.assign(n_layers, { 0, 0 });
    if (s_arena_proven == 0) return;                      // already disproven this process
    const char * ev = std::getenv("MOE_DUAL_Q4S_ARENA");
    if (!(ev && ev[0] == '1')) return;                    // opt-in
    if (g.hcfg.arena_exclusive && !(g.mm && moert_memmgr_cold_arena_mapped(g.mm))) {   // released pages would be read unguarded
                                                          // (a mapped pack's trimmed pages stay readable)
        if (s_arena_proven < 0)
            std::cerr << "moe_dual: " << label << ": arena-sourced staging REFUSED under the exclusive "
                         "arena (hot experts' pages are released) — staging from the file\n";
        s_arena_proven = 0;
        return;
    }
    if (!g.mm || !g.cold_arena_on) {
        if (s_arena_proven < 0)
            std::cerr << "moe_dual: " << label << ": arena-sourced staging requested but no owned cold "
                         "arena is active — staging from the file\n";
        s_arena_proven = 0;
        return;
    }
    const int n_expert = g.q4s_expert_n;
    // Sample the same experts the hook's arena verify does (first / middle / last): a stride or
    // layout error shows up at the ends, a wrong block at the middle.
    const int samples[3] = { 0, n_expert / 2, n_expert - 1 };
    std::vector<char> buf;
    long long ok = 0, bad = 0, mapped = 0;
    bool complete = true;

    for (int il = 0; il < n_layers && complete; ++il) {
        for (int kind = 0; kind < 2 && complete; ++kind) {
            // ws 0 = weight, 1 = per-expert scale; both must map or neither does for this layer/kind.
            const struct { ggml_tensor * t; int64_t off; size_t blk; int16_t shard; int ws; } parts[2] = {
                { g.q4s_wt[il][kind], g.q4s_off[il][kind],   g.q4s_blk[il][kind],   g.q4s_shard[il][kind],   0 },
                { g.q4s_st[il][kind], g.q4s_off_s[il][kind], g.q4s_blk_s[il][kind], g.q4s_shard_s[il][kind], 1 },
            };
            for (const auto & pr : parts) {
                if (!pr.t || pr.off < 0) continue;        // absent in this source -> nothing to map
                moert_cold_arena_view a;
                if (!moert_memmgr_get_cold_arena(g.mm, il, kind, pr.ws, &a) || !a.base) { complete = false; break; }
                // geometry must line up EXACTLY with the staging tensor this would fill
                if (a.n_expert != n_expert || a.type_tag != (int32_t) pr.t->type ||
                    a.stride != (uint64_t) pr.blk || (size_t) n_expert * pr.blk != ggml_nbytes(pr.t)) {
                    complete = false; break;
                }
                if (s_arena_proven < 0) {
                    FILE * f = (pr.shard >= 0 && pr.shard < (int16_t) g.q4s_f.size()) ? g.q4s_f[pr.shard] : nullptr;
                    if (!f) { complete = false; break; }
                    buf.resize(pr.blk);
                    for (int e : samples) {
                        if (!io::seek64(f, pr.off + (int64_t) e * (int64_t) pr.blk, SEEK_SET) ||
                            fread(buf.data(), 1, pr.blk, f) != pr.blk) { complete = false; break; }
                        const char * aref = (const char *) a.base + (size_t) e * (size_t) a.arena_stride;
                        if (std::memcmp(buf.data(), aref, pr.blk) == 0) { ++ok; }
                        else { ++bad; complete = false; break; }
                    }
                    if (!complete) break;
                }
                if (pr.ws == 0) { g.q4s_ar[il][kind]   = a.base; g.q4s_ar_str[il][kind]   = a.arena_stride; }
                else            { g.q4s_ar_s[il][kind] = a.base; g.q4s_ar_str_s[il][kind] = a.arena_stride; }
                ++mapped;
            }
        }
    }
    if (!complete || bad) {
        std::cerr << "moe_dual: " << label << ": arena-sourced staging DISARMED ("
                  << (bad ? "byte mismatch vs the source file" : "arena layout does not cover the source")
                  << "; verify ok=" << ok << " bad=" << bad << ") — staging from the file\n";
        s_arena_proven = 0;
        g.q4s_ar.assign(n_layers, { nullptr, nullptr });  g.q4s_ar_s.assign(n_layers, { nullptr, nullptr });
        return;
    }
    g.q4s_arena_mode = 1;
    if (s_arena_proven < 0) {
        g.q4s_ar_ok = ok; g.q4s_ar_bad = bad;
        std::cerr << "moe_dual: " << label << ": arena-sourced staging ARMED — " << mapped
                  << " tensor groups map onto the owned cold arena, byte-verified against " << ok
                  << " sampled experts (0 bad). The per-prompt source re-read is gone.\n";
    }
    s_arena_proven = 1;
}

// P2: bytes of ONE hot slot in every layer (all kinds, weights + scales) = the VRAM price of one K.
static long long q4s_per_k_mib() {
    if (!g.mm || g.n_layers <= 0) return 0;
    uint64_t b = 0;
    for (int L = 0; L < g.n_layers; ++L)
        for (int kind = 0; kind < 2; ++kind)
            for (int ws = 0; ws < 2; ++ws) {
                moert_memmgr_page pg;
                if (moert_memmgr_page_info(g.mm, L, kind, ws, &pg)) b += pg.bytes_per_expert;
            }
    return (long long) (b >> 20);
}

bool q4stream_init(const char * src_path, const char * label, bool lean_ring) {
    if (g.q4s_enabled) return true;
    parse_hook_env();   // q4stream can run standalone (no pack/dynamic init)
    g.q4s_lean = lean_ring;
    const std::vector<std::string> shard_paths = gguf_split_shards(src_path);

    struct Shard { gguf_context * gg = nullptr; ggml_context * meta = nullptr; int64_t data_off = 0; };
    std::vector<Shard> shards;
    auto close_shards = [&]() { for (auto & s : shards) { if (s.gg) gguf_free(s.gg); if (s.meta) ggml_free(s.meta); } };
    for (const std::string & sp : shard_paths) {
        ggml_context * meta = nullptr;
        gguf_init_params ip; ip.no_alloc = true; ip.ctx = &meta;
        gguf_context * gg = gguf_init_from_file(sp.c_str(), ip);
        if (!gg) {
            std::cerr << "moe_dual: " << label << ": cannot open source shard " << sp << "\n";
            close_shards(); return false;
        }
        shards.push_back({ gg, meta, (int64_t) gguf_get_data_offset(gg) });
    }
    if (shard_paths.size() > 1)
        std::cerr << "moe_dual: " << label << ": " << shard_paths.size() << "-shard GGUF split detected\n";

    // discover n_layers / n_expert across ALL shards (same probe as init_dynamic)
    int n_layers = 0, n_expert = 0;
    for (auto & sh : shards) {
        const int64_t nt = gguf_get_n_tensors(sh.gg);
        for (int64_t i = 0; i < nt; ++i) {
            const std::string nm = gguf_get_tensor_name(sh.gg, i);
            const size_t p = nm.find("blk.");
            if (p != std::string::npos) { const int L = std::atoi(nm.c_str() + p + 4); if (L + 1 > n_layers) n_layers = L + 1; }
            if (nm.find("_exps.weight") != std::string::npos) {
                ggml_tensor * t = ggml_get_tensor(sh.meta, nm.c_str());
                if ((int) t->ne[2] > n_expert) n_expert = (int) t->ne[2];
            }
        }
    }
    if (n_layers <= 0 || n_expert <= 0) {
        std::cerr << "moe_dual: " << label << ": no experts in source " << src_path << "\n";
        close_shards(); return false;
    }
    if (g.n_layers > 0 && (n_layers != g.n_layers || n_expert != g.n_expert)) {
        std::cerr << "moe_dual: " << label << ": source geometry (" << n_layers << "L x " << n_expert
                  << "E) != active hot pool (" << g.n_layers << "L x " << g.n_expert << "E)\n";
        close_shards(); return false;
    }

    g.q4s_off.assign(n_layers, {-1, -1});     g.q4s_blk.assign(n_layers, {0, 0});
    g.q4s_off_s.assign(n_layers, {-1, -1});   g.q4s_blk_s.assign(n_layers, {0, 0});
    g.q4s_shard.assign(n_layers, {-1, -1});   g.q4s_shard_s.assign(n_layers, {-1, -1});
    g.q4s_wt.assign(n_layers, {nullptr, nullptr});
    g.q4s_st.assign(n_layers, {nullptr, nullptr});

    ggml_init_params pp;
    pp.mem_size = 96 * ggml_tensor_overhead() + (1u << 14);   // variants x ring slots + byte views
    pp.mem_buffer = nullptr; pp.no_alloc = true;
    g.q4s_ctx = ggml_init(pp);

    // One staging VARIANT tensor per distinct (kind, scale?, type, shape). Sources can mix types
    // across layers (measured: UD Q4 blk.29 down Q8_0 / gate_up Q5_K vs Q5_1/Q4_K elsewhere), so a
    // single tensor per kind cannot represent the source. Variants of the same slot alias one
    // buffer region (max variant size): only the current layer's variant is ever live.
    struct Slot { std::vector<ggml_tensor *> variants; size_t max_bytes = 0; size_t region_off = 0; };
    Slot slots[2][2];   // [kind][is_scale]
    // R1: which VARIANT (index into slots[kind][sc].variants) each layer/kind uses. The template
    // tensors below are slot 0; every other ring slot creates its own copies in the same order, so
    // the same index resolves that layer's tensor in any slot.
    std::vector<std::array<int, 2>> vidx_w(n_layers, { -1, -1 }), vidx_s(n_layers, { -1, -1 });
    for (int shard_idx = 0; shard_idx < (int) shards.size(); ++shard_idx) {
        auto & sh = shards[shard_idx];
        const int64_t nt = gguf_get_n_tensors(sh.gg);
        for (int64_t i = 0; i < nt; ++i) {
            const std::string nm = gguf_get_tensor_name(sh.gg, i);
            if (nm.find("_exps") == std::string::npos) continue;
            const size_t p = nm.find("blk."); if (p == std::string::npos) continue;
            const int L = std::atoi(nm.c_str() + p + 4); if (L < 0 || L >= n_layers) continue;
            const int kind = (nm.find("gate_up_exps") != std::string::npos) ? 0
                           : (nm.find("down_exps")    != std::string::npos) ? 1 : -1;
            if (kind < 0) continue;
            ggml_tensor * st = ggml_get_tensor(sh.meta, nm.c_str());
            int axis = -1; for (int d = GGML_MAX_DIMS - 1; d >= 0; --d) if ((int) st->ne[d] == n_expert) { axis = d; break; }
            if (axis < 0) continue;
            const bool    is_scale = ends_with(nm, ".scale");
            const int64_t soff     = sh.data_off + (int64_t) gguf_get_tensor_offset(sh.gg, i);
            Slot & slot = slots[kind][is_scale ? 1 : 0];
            ggml_tensor * var = nullptr;
            int vi = -1;
            for (size_t k = 0; k < slot.variants.size(); ++k) {
                ggml_tensor * v = slot.variants[k];
                if (v->type == st->type && std::memcmp(v->ne, st->ne, sizeof(v->ne)) == 0) { var = v; vi = (int) k; break; }
            }
            if (!var) {
                var = ggml_new_tensor(g.q4s_ctx, st->type, ggml_n_dims(st), st->ne);
                ggml_set_name(var, (std::string("moe_dual.q4s.") + (is_scale ? "s" : "w") + std::to_string(kind)
                                    + "." + ggml_type_name(st->type)).c_str());
                vi = (int) slot.variants.size();
                slot.variants.push_back(var);
                slot.max_bytes = std::max(slot.max_bytes, ggml_nbytes(var));
            }
            if (is_scale) { g.q4s_off_s[L][kind] = soff; g.q4s_blk_s[L][kind] = st->nb[axis]; g.q4s_st[L][kind] = var; g.q4s_shard_s[L][kind] = (int16_t) shard_idx; vidx_s[L][kind] = vi; }
            else          { g.q4s_off[L][kind]   = soff; g.q4s_blk[L][kind]   = st->nb[axis]; g.q4s_wt[L][kind] = var; g.q4s_shard[L][kind]   = (int16_t) shard_idx; vidx_w[L][kind] = vi; }
        }
    }
    close_shards();

    // Completeness gate: a layer that participates in the merged gate_up(0)/down(1) expert scheme
    // must have BOTH halves, and every layer with a weight must have that kind's scale whenever the
    // source emits scales for that kind at all. Catches partial layers from a bad/incomplete source
    // OR a split shard silently missing (this is exactly what a naive single-file scan of the real
    // BF16 source would miss: layer 29's down_exps.weight lives ONLY in its second shard).
    bool scale_used[2] = {};
    for (int kind = 0; kind < 2; ++kind) scale_used[kind] = !slots[kind][1].variants.empty();
    bool complete = true;
    for (int L = 0; L < n_layers && complete; ++L) {
        const bool has_gu = g.q4s_wt[L][0] != nullptr, has_dn = g.q4s_wt[L][1] != nullptr;
        if (has_gu != has_dn) {
            std::cerr << "moe_dual: " << label << ": layer " << L << " has partial expert tensors"
                      << " (gate_up=" << has_gu << " down=" << has_dn << ") — incomplete source\n";
            complete = false; break;
        }
        for (int kind = 0; kind < 2 && complete; ++kind) {
            if (g.q4s_wt[L][kind] && scale_used[kind] && !g.q4s_st[L][kind]) {
                std::cerr << "moe_dual: " << label << ": layer " << L << " kind " << kind
                          << " weight present but its scale is missing — incomplete source\n";
                complete = false;
            }
        }
    }
    if (!complete) {
        ggml_free(g.q4s_ctx); g.q4s_ctx = nullptr;
        g.q4s_wt.clear(); g.q4s_st.clear(); g.q4s_shard.clear(); g.q4s_shard_s.clear();
        return false;
    }

    // region layout: one max-sized region per (kind, scale?) slot, 1 KiB aligned
    size_t total = 0;
    int n_variants = 0;
    for (int kind = 0; kind < 2; ++kind)
        for (int sc = 0; sc < 2; ++sc) {
            Slot & slot = slots[kind][sc];
            if (slot.variants.empty()) continue;
            slot.region_off = total;
            total += (slot.max_bytes + 1023) & ~(size_t) 1023;
            n_variants += (int) slot.variants.size();
        }
    if (total == 0) {
        std::cerr << "moe_dual: " << label << ": no usable expert tensors in " << src_path << "\n";
        ggml_free(g.q4s_ctx); g.q4s_ctx = nullptr;
        g.q4s_wt.clear(); g.q4s_st.clear();
        return false;
    }

    if (!g.backend) g.backend = pick_pack_backend();
    // R1: ring depth. 1 = single-buffer synchronous staging (slot 0 IS that buffer and holds the
    // template variant tensors). >1 arms the cross-layer prefetch: the reader fills slot (il+1)%N while
    // the graph computes layer il out of slot il%N.
    int  ring_n = 1;
    if (lean_ring) {
        ring_n = 1;   // load-time arm: no prompt streams through it, and free VRAM here is inflated
    } else {
        // P2 fixed reserve: the ring is paid for in K, once. Depth 1 when the cold pages come from the RAM arena
        // (~0.6 s per prompt, depth 2 only costs K); with an SSD source depth 2 when the K left after both slots
        // stays >= 48 (findings 177, 179). (history: docs/code-comments-archive-2026-10-05.md, "ring-depth")
        long long avail_mib = -1;
        if (ggml_backend_dev_t dev = g.backend ? ggml_backend_get_device(g.backend) : nullptr) {
            size_t free_b = 0, total_b = 0;
            ggml_backend_dev_memory(dev, &free_b, &total_b);
            avail_mib = (long long) (free_b >> 20);
            if (slab_evicted()) avail_mib -= (long long) (g.slab_total_bytes >> 20);   // comes back
            else                avail_mib += (long long) (g.slab_total_bytes >> 20);   // refit re-sizes it
        }
        const long long per_k = q4s_per_k_mib();
        const long long min_k = 48;
        const long long margin_mib = 384 + 250;   // staging admission margin + the refit's keep_post
        const long long k2 = per_k > 0 ? (avail_mib - margin_mib - (long long) ((total * 2) >> 20)) / per_k : -1;
        const char * av = std::getenv("MOE_DUAL_Q4S_ARENA");
        const bool arena_src = av && av[0] == '1' && g.mm && g.cold_arena_on &&
                               !(g.hcfg.arena_exclusive && !moert_memmgr_cold_arena_mapped(g.mm)) &&
                               s_arena_proven != 0;
        ring_n = arena_src ? 1 : ((k2 >= min_k) ? 2 : 1);
        std::cerr << "moe_dual: " << label << ": staging reserve (no-evict prefill) -> depth " << ring_n
                  << (arena_src ? " (cold pages from the RAM arena: one slot hides the transfer"
                                : " (SSD source")
                  << "; K left with 2 slots ~" << k2 << ", threshold " << min_k << "; " << per_k
                  << " MiB per K, " << (total >> 20) << " MiB per slot).\n";
    }
    g.q4s_ring_n = ring_n;

    // The ring multiplies the footprint -- admit the FULL amount, not one layer. A prefetch slot buys
    // only overlap, never capacity: if depth 2 would not fit as-is, drop to depth 1.
    size_t ring_bytes = total * (size_t) ring_n;
    if (ring_n > 1 && q4s_vram_would_overcommit(ring_bytes)) {
        std::cerr << "moe_dual: " << label << ": staging ring AUTO 2 -> 1 (the second slot does not fit; "
                     "the prefetch is not worth capacity)\n";
        ring_n = 1;
        g.q4s_ring_n = 1;
        ring_bytes = total;
    }
    auto init_fail = [&]() {
        for (ggml_backend_buffer_t b : g.q4s_buf_r) if (b) ggml_backend_buffer_free(b);
        g.q4s_buf_r.clear(); g.q4s_wt_r.clear(); g.q4s_st_r.clear(); g.q4s_bytes_r.clear();
        g.q4s_buf = nullptr;
        ggml_free(g.q4s_ctx); g.q4s_ctx = nullptr;
        g.q4s_wt.clear(); g.q4s_st.clear();
        g.q4s_ring_n = 1;
        return false;
    };
    if (q4s_vram_would_overcommit(ring_bytes)) {
        std::cerr << "moe_dual: " << label << ": the " << (ring_bytes >> 20) << " MiB staging does not fit "
                     "in free VRAM -- fused per-token prefill\n";
        return init_fail();
    }
    g.q4s_buf_r.assign(ring_n, nullptr);
    g.q4s_wt_r.assign(ring_n, {});
    g.q4s_st_r.assign(ring_n, {});
    for (int s = 0; s < ring_n; ++s) {
        g.q4s_buf_r[s] = ggml_backend_alloc_buffer(g.backend, total);
        if (!g.q4s_buf_r[s]) {
            std::cerr << "moe_dual: " << label << ": staging buffer alloc FAILED (slot " << s << " of "
                      << ring_n << ", " << (total >> 20) << " MiB = one full expert layer; "
                      << "lower the hot-cache K, or free VRAM)\n";
            return init_fail();
        }
        ggml_backend_buffer_clear(g.q4s_buf_r[s], 0);   // defined bytes until the first staging
        char * base = (char *) ggml_backend_buffer_get_base(g.q4s_buf_r[s]);
        // slot 0 places the template tensors themselves (identical to the pre-R1 code); slots >0 get
        // their own copies of every variant, created in the SAME order so a variant index resolves
        // the same logical tensor in every slot.
        ggml_tensor * svar[2][2][8] = {};   // [kind][sc][variant] -- 8 covers any real source
        for (int kind = 0; kind < 2; ++kind)
            for (int sc = 0; sc < 2; ++sc)
                for (size_t k = 0; k < slots[kind][sc].variants.size(); ++k) {
                    if (k >= 8) { std::cerr << "moe_dual: " << label << ": too many staging variants\n"; return init_fail(); }
                    ggml_tensor * tmpl = slots[kind][sc].variants[k];
                    ggml_tensor * v = tmpl;
                    if (s > 0) {
                        v = ggml_new_tensor(g.q4s_ctx, tmpl->type, ggml_n_dims(tmpl), tmpl->ne);
                        ggml_set_name(v, (std::string(tmpl->name) + ".r" + std::to_string(s)).c_str());
                    }
                    if (ggml_backend_tensor_alloc(g.q4s_buf_r[s], v, base + slots[kind][sc].region_off) != GGML_STATUS_SUCCESS) {
                        std::cerr << "moe_dual: " << label << ": staging tensor placement failed (" << v->name << ")\n";
                        return init_fail();
                    }
                    svar[kind][sc][k] = v;
                }
        // P2: one I8 view over the whole slot, the target of the slab -> staging D2D copies
        {
            ggml_tensor * bv = ggml_new_tensor_1d(g.q4s_ctx, GGML_TYPE_I8, (int64_t) total);
            ggml_set_name(bv, (std::string("moe_dual.q4s.bytes.r") + std::to_string(s)).c_str());
            if (ggml_backend_tensor_alloc(g.q4s_buf_r[s], bv, base) != GGML_STATUS_SUCCESS) {
                std::cerr << "moe_dual: " << label << ": staging byte view placement failed\n";
                return init_fail();
            }
            g.q4s_bytes_r.push_back(bv);
        }
        // per-layer view of this slot: same layer->variant mapping, this slot's tensors
        g.q4s_wt_r[s].assign(n_layers, { nullptr, nullptr });
        g.q4s_st_r[s].assign(n_layers, { nullptr, nullptr });
        for (int L = 0; L < n_layers; ++L)
            for (int kind = 0; kind < 2; ++kind) {
                if (vidx_w[L][kind] >= 0) g.q4s_wt_r[s][L][kind] = svar[kind][0][vidx_w[L][kind]];
                if (vidx_s[L][kind] >= 0) g.q4s_st_r[s][L][kind] = svar[kind][1][vidx_s[L][kind]];
            }
    }
    g.q4s_buf = g.q4s_buf_r[0];   // legacy alias: telemetry buf_bytes + the single-buffer code paths
    for (const std::string & sp : shard_paths) {
        FILE * f = fopen(sp.c_str(), "rb");
        if (!f) {
            std::cerr << "moe_dual: " << label << ": cannot reopen source shard " << sp << " for streaming\n";
            for (FILE * of : g.q4s_f) if (of) fclose(of);
            g.q4s_f.clear();
            ggml_backend_buffer_free(g.q4s_buf); g.q4s_buf = nullptr;
            ggml_free(g.q4s_ctx); g.q4s_ctx = nullptr;
            g.q4s_wt.clear(); g.q4s_st.clear();
            return false;
        }
        g.q4s_f.push_back(f);
    }

    // Overlapped staging pipeline: 2 pinned slices (MOE_DUAL_Q4S_SLICE_MB each, default 64) + upload
    // fences. Pinned memory makes ggml_backend_tensor_set_async a real async H2D (and ~4x pageable
    // bandwidth), so each slice's disk read overlaps the previous slice's upload. Any piece missing
    // (host buffer type, events — e.g. CPU backend) falls back to the synchronous pageable path,
    // which is slower but byte-identical.
    g.q4s_hfile.assign(shard_paths.size(), nullptr);
    {
        const size_t slice_sz = (size_t) 64 << 20;   // measured working point for the pinned slices
        ggml_backend_dev_t         dev = ggml_backend_get_device(g.backend);
        ggml_backend_buffer_type_t hbt = dev ? ggml_backend_dev_host_buffer_type(dev) : nullptr;
        if (hbt) g.q4s_pin_buf = ggml_backend_buft_alloc_buffer(hbt, slice_sz * 2);
        if (g.q4s_pin_buf && dev) {
            g.q4s_ev[0] = ggml_backend_event_new(dev);
            g.q4s_ev[1] = ggml_backend_event_new(dev);
            g.q4s_ev_d2d = ggml_backend_event_new(dev);   // null => no slab source
        }
        if (g.q4s_pin_buf && g.q4s_ev[0] && g.q4s_ev[1]) {
            g.q4s_slice_sz = slice_sz;
            g.q4s_slice[0] = (char *) ggml_backend_buffer_get_base(g.q4s_pin_buf);
            g.q4s_slice[1] = g.q4s_slice[0] + slice_sz;
#if defined(_WIN32)
            // unbuffered handle per shard for the pipelined reads (falls back to q4s_f[shard] if it
            // cannot open — the read_at() lambda in q4stream_on_topk checks each entry individually)
            for (size_t si = 0; si < shard_paths.size(); ++si) {
                HANDLE h = CreateFileA(shard_paths[si].c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                                       FILE_ATTRIBUTE_NORMAL | FILE_FLAG_NO_BUFFERING, nullptr);
                if (h != INVALID_HANDLE_VALUE) g.q4s_hfile[si] = (void *) h;
            }
#endif
        } else {
            if (g.q4s_ev[0])   { ggml_backend_event_free(g.q4s_ev[0]); g.q4s_ev[0] = nullptr; }
            if (g.q4s_ev[1])   { ggml_backend_event_free(g.q4s_ev[1]); g.q4s_ev[1] = nullptr; }
            if (g.q4s_pin_buf) { ggml_backend_buffer_free(g.q4s_pin_buf); g.q4s_pin_buf = nullptr; }
            std::cerr << "moe_dual: " << label << ": pinned/async staging unavailable on this backend — "
                         "using the synchronous pageable path\n";
        }
    }

    g.q4s_layers_n = n_layers;
    g.q4s_expert_n = n_expert;
    g.q4s_src_path = src_path ? src_path : "";
    g.q4s_pending.assign(n_layers, 0);
    g.bnd_cnt.assign(n_layers, std::vector<int32_t>((size_t) n_expert, 0));
    g.bnd_ntok.assign(n_layers, 0);
    g.bnd_pending.assign(n_layers, 0);
    q4s_arm_arena(label);   // after the file handles open: verification reads through them
    llama_set_moe_dual_mm_id(&hook);   // idempotent with pack/dynamic init; enables q4stream standalone too
    g.q4s_enabled = true;
    const bool any_pipelined = g.q4s_slice[0] != nullptr;
    const bool any_unbuffered = std::any_of(g.q4s_hfile.begin(), g.q4s_hfile.end(), [](void * h) { return h != nullptr; });
    std::cerr << "moe_dual: " << label << " prefill ready: " << n_layers << " layers x " << n_expert
              << " experts, staging buffer " << (ggml_backend_buffer_get_size(g.q4s_buf) >> 20)
              << " MiB x " << ring_n << " ring slot(s) = " << (ring_bytes >> 20) << " MiB"
              << (ring_n > 1 ? " (R1 cross-layer prefetch ON: reader runs a layer ahead of compute)"
                             : " (single buffer, synchronous staging)")
              << " (one full expert layer; " << n_variants
              << " type variants aliased), "
              << (any_pipelined ? "pinned+overlapped staging (2x" : "SYNCHRONOUS staging (no pinned slices, 0x")
              << (g.q4s_slice_sz >> 20) << " MiB slices, "
              << (any_unbuffered ? "unbuffered" : "buffered") << " reads), source " << src_path
              << (shard_paths.size() > 1 ? (" (" + std::to_string(shard_paths.size()) + " shards)") : std::string())
              << (!lean_ring ? "; kept across prompts as the fixed reserve, hot experts staged D2D from the "
                               "slab, residency decided in the prefill" : "")
              << "\n";
    return true;
}

bool q4stream_available() { return g.q4s_enabled; }
bool q4stream_keep_after_prompt() { return g.q4s_enabled && !g.q4s_lean; }

// Undo q4stream_init(): free the staging buffer/context/file handles and reset all state. Used when a
// lean arm ends and around an image decode. Safe to call when not initialized (no-op).
void q4stream_shutdown() {
    loan_revoke("staging released");   // findings 246: the loaned slots live in the staging about to be freed
    q4s_ring_stop();          // no read may be in flight when the staging VRAM is freed
    g.q4s_on = false;
    g.q4s_error = false;
    g.q4s_enabled = false;
    if (g.q4s_ev[0]) { ggml_backend_event_free(g.q4s_ev[0]); g.q4s_ev[0] = nullptr; }
    if (g.q4s_ev[1]) { ggml_backend_event_free(g.q4s_ev[1]); g.q4s_ev[1] = nullptr; }
    g.q4s_ev_live[0] = g.q4s_ev_live[1] = false;
    if (g.q4s_ev_d2d) { ggml_backend_event_free(g.q4s_ev_d2d); g.q4s_ev_d2d = nullptr; }
    g.q4s_ev_d2d_live = false;
    // Safe to free here: q4s_ring_stop() joined the reader above, and the reader host-blocks on every
    // upload's event before returning, so no copy can still be in flight on this stream.
    if (g.q4s_pin_buf) { ggml_backend_buffer_free(g.q4s_pin_buf); g.q4s_pin_buf = nullptr; }
    g.q4s_slice[0] = g.q4s_slice[1] = nullptr;
    g.q4s_slice_sz = 0;
    for (ggml_backend_buffer_t b : g.q4s_buf_r) if (b) ggml_backend_buffer_free(b);
    g.q4s_buf_r.clear(); g.q4s_wt_r.clear(); g.q4s_st_r.clear(); g.q4s_bytes_r.clear();
    g.q4s_slot_layer.clear(); g.q4s_slot_ready.clear();
    g.q4s_consumed_upto = -1; g.q4s_pf_fail = false; g.q4s_ring_n = 1;
    g.q4s_buf = nullptr;      // was an alias into q4s_buf_r, already freed above
    if (g.q4s_ctx) { ggml_free(g.q4s_ctx); g.q4s_ctx = nullptr; }
    for (FILE * f : g.q4s_f) if (f) fclose(f);
    g.q4s_f.clear();
#if defined(_WIN32)
    for (void * h : g.q4s_hfile) if (h) CloseHandle((HANDLE) h);
#endif
    g.q4s_hfile.clear();
    g.q4s_wt.clear(); g.q4s_st.clear();
    g.q4s_off.clear(); g.q4s_off_s.clear();
    g.q4s_blk.clear(); g.q4s_blk_s.clear();
    g.q4s_shard.clear(); g.q4s_shard_s.clear();
    g.q4s_pending.clear();
    g.q4s_ar.clear(); g.q4s_ar_s.clear(); g.q4s_ar_str.clear(); g.q4s_ar_str_s.clear();
    g.q4s_arena_mode = 0;     // re-armed (and re-verified once per process) by the next init
    g.q4s_layers_n = g.q4s_expert_n = 0;
    g.q4s_lean = false;
    // telemetry counters (q4s_bytes/staged/stage_ms/read_ms/arena_bytes) intentionally NOT reset: q4stream_stats()
    // callers diff a before/after snapshot, so a mid-process reset would corrupt the next diff.
}

bool q4stream_prompt_active() { return g.q4s_enabled && g.q4s_on; }

// The quantized matmuls read past the end of their weight tensor, into the padding the CUDA buffer type adds
// (MATRIX_ROW_PADDING), and count on it being zero: ggml clears it for compute buffers, the staging is cleared once
// at init. The loaned slots (findings 246) write expert pages across the whole staging, padding included, so before
// a streamed prompt computes from the staging every variant's padding is zeroed again. Without it (findings 292,
// quality at 5 GiB, K 4 + 5 loaned) the prefill read another expert's bytes there and the turn's hidden states
// collapsed: MTP acceptance 0%, hit-rate ~95%.
static void q4s_clear_padding() {
    for (size_t s = 0; s < g.q4s_buf_r.size() && s < g.q4s_bytes_r.size(); ++s) {
        ggml_backend_buffer_t buf = g.q4s_buf_r[s];
        ggml_tensor * bytes = g.q4s_bytes_r[s];
        if (!buf || !bytes) continue;
        const char * base = (const char *) ggml_backend_buffer_get_base(buf);
        const size_t cap  = ggml_nbytes(bytes);
        std::vector<const ggml_tensor *> seen;
        for (const auto * parts : { &g.q4s_wt_r[s], &g.q4s_st_r[s] })
            for (const auto & lk : *parts)
                for (const ggml_tensor * t : lk) {
                    if (!t || std::find(seen.begin(), seen.end(), t) != seen.end()) continue;
                    seen.push_back(t);
                    const size_t off = (size_t) ((const char *) t->data - base), nb = ggml_nbytes(t);
                    const size_t end = std::min(cap, off + ggml_backend_buffer_get_alloc_size(buf, t));
                    if (end > off + nb) ggml_backend_tensor_memset(bytes, 0, off + nb, end - off - nb);
                }
    }
}

void q4stream_begin() {
    if (!g.q4s_enabled) return;
    g.q4s_on = true; g.q4s_error = false;
    g.q4s_arena_trip = false;   // every prompt tries the arena source again
    // P3: a fresh prompt -- fresh counts, and no copy in flight into a slot
    for (auto & c : g.bnd_cnt) std::fill(c.begin(), c.end(), 0);
    std::fill(g.bnd_ntok.begin(), g.bnd_ntok.end(), 0);
    std::fill(g.bnd_pending.begin(), g.bnd_pending.end(), 0);
    if (g.pol) xfer_settle();
    const bool was_lent = g.loan > 0 && g.loan_live;
    loan_revoke("streamed prompt");   // findings 246: the staging is about to be written
    if (was_lent) q4s_clear_padding();   // findings 292: the loan wrote expert pages over the staging's padding
    g.q4s_ar_p_bytes = 0; g.q4s_ar_p_ms = 0.0;
    std::fill(g.q4s_pending.begin(), g.q4s_pending.end(), 0);
}

static void q4s_boundary_decide(int il);

void q4stream_end() {
    q4s_ring_stop();          // join before the prompt-end hook restores the hot slab into this VRAM
    // P3: the last layer(s) counted in the final decode call are decided now, while the staging still
    // holds them (the graph has completed; the prefetcher was just joined)
    for (int il = 0; il < (int) g.bnd_pending.size(); ++il)
        if (g.bnd_pending[(size_t) il]) q4s_boundary_decide(il);
    g.q4s_on = false;
    std::fill(g.q4s_pending.begin(), g.q4s_pending.end(), 0);
    loan_lend();   // findings 246: the staging is idle until the next streamed prompt
}

// Arm every layer for (re-)staging regardless of whether the hook runs at build time: graph reuse
// skips the build for equal-shaped consecutive chunks, but the eval callback still fires per layer,
// so pre-armed pending flags are exactly what makes the reused graph re-stage. Layers absent from
// the graph are never consumed (their topk callback never fires) — over-arming is harmless.
void q4stream_chunk_begin() {
    if (!g.q4s_enabled || !g.q4s_on) return;
    std::fill(g.q4s_pending.begin(), g.q4s_pending.end(), 1);
    // R1: one prefetch pass per llama_decode call (each call re-streams every layer).
    if (g.q4s_ring_n > 1) q4s_ring_start();
}

// Read layer `il`'s complete expert set from the source and upload it into the GIVEN staging
// tensors. Shared by the synchronous callback path (ring_n == 1, historical behavior) and the R1
// prefetch thread (ring_n > 1), which runs it for slot (il % ring_n) while the graph computes an
// earlier layer.
// Pipelined path: slices go up via ggml_backend_tensor_set_async on g.backend's OWN stream (pinned
// source => true async), so each slice's fread overlaps the previous slice's upload; a full backend
// sync before returning keeps the contract (the graph's compute stream is a different stream, so
// only that sync orders the uploads before the matmuls — and on the prefetch thread that sync waits
// on the UPLOAD stream only, leaving the compute stream free, which is what buys the overlap).
// Fallback: host-synchronous set.
// P2 slab source. -1 = not yet proven, 0 = disproven (stage everything from the source), 1 = proven:
// a READY slot of the hot slab holds byte-for-byte the expert page the source holds (checked once per
// process on a sample, the same evidence standard as the arena source), so it can be staged D2D.
static int s_slab_proven = -1;

// Which experts of layer il can be staged from the slab, and where: hot[e] = slot, or -1. Empty when
// the slab source does not apply to this (layer, kind) weight part. Read at staging time, inside the
// llama_decode that owns the prefill: the policy's views change only on the server thread, which is
// blocked in that decode, and a READY slot is never written while READY (a promotion writes only a
// FILLING slot), so the snapshot is stable for the copies it feeds.
static bool q4s_slab_plan(int il, int kind, const ggml_tensor * t, size_t blk,
                          moert_memmgr_page & pg, std::vector<int> & hot) {
    hot.clear();
    if (s_slab_proven == 0 || !g.q4s_ev_d2d || g.q4s_slice[0] == nullptr) return false;
    if (!g.mm || !g.pol || !g.slab_buf || !g.slab_bytes_t || slab_evicted()) return false;
    if (il < 0 || il >= g.n_layers || !moert_memmgr_page_info(g.mm, il, kind, 0, &pg)) return false;
    if (pg.type_tag != (int32_t) t->type || pg.bytes_per_expert != (uint64_t) blk) return false;
    const int32_t * es = moert_policy_exp_slot_view(g.pol, il);
    const int32_t * ss = moert_policy_slot_state_view(g.pol, il);
    if (!es || !ss) return false;
    const int kL = klayer(il);
    hot.assign((size_t) g.q4s_expert_n, -1);
    int n = 0;
    for (int e = 0; e < g.q4s_expert_n && e < g.n_expert; ++e) {
        const int32_t s = es[e];
        if (s >= 0 && s < kL && s < g.k && ss[s] == MOERT_SLOT_READY) { hot[(size_t) e] = s; ++n; }   // slab slots only
    }
    if (n == 0) { hot.clear(); return false; }
    return true;
}

static bool q4s_stage_layer(int il, int rs,
                            const std::vector<std::array<ggml_tensor *, 2>> & wt_r,
                            const std::vector<std::array<ggml_tensor *, 2>> & st_r) {
    using clk = std::chrono::steady_clock;
    const clk::time_point t0 = clk::now();
    const double read_ms_before = g.q4s_read_ms;   // per-layer trace: this call's own read delta
    const bool pipelined = g.q4s_slice[0] != nullptr;
    int  cur = 0;      // pipeline slice cursor (persists across parts: overlap crosses tensor bounds)
    bool ok  = true;
    std::vector<int> hot;   // P2: expert -> READY slab slot for the current weight part (-1 = cold)
    for (int kind = 0; kind < 2 && ok; ++kind) {
        const struct { ggml_tensor * t; int64_t off; int16_t shard; const void * ar; uint64_t ar_str; size_t blk; } parts[2] = {
            { wt_r[il][kind], g.q4s_off[il][kind],   g.q4s_shard[il][kind],
              g.q4s_ar.empty()   ? nullptr : g.q4s_ar[il][kind],
              g.q4s_ar_str.empty()   ? 0 : g.q4s_ar_str[il][kind],   g.q4s_blk[il][kind]   },
            { st_r[il][kind], g.q4s_off_s[il][kind], g.q4s_shard_s[il][kind],
              g.q4s_ar_s.empty() ? nullptr : g.q4s_ar_s[il][kind],
              g.q4s_ar_str_s.empty() ? 0 : g.q4s_ar_str_s[il][kind], g.q4s_blk_s[il][kind] },
        };
        for (size_t pi = 0; pi < 2 && ok; ++pi) {
            const auto & pr = parts[pi];
            if (!pr.t || pr.off < 0) continue;
            const size_t bytes = ggml_nbytes(pr.t);   // experts are contiguous: one read covers the layer
            const int shard = pr.shard;
            const bool from_arena = pr.ar && g.q4s_arena_mode && !g.q4s_arena_trip.load();
            // ARENA SOURCE: copy from the owned cold arena through the same pinned slices + upload fences the file
            // path uses (only the producer changes). proc_mem() brackets the copy as residency evidence.
            // Stages bytes [toff, toff + len) of this part's layer data into the same offsets of pr.t.
            // (history: docs/code-comments-archive-2026-10-05.md, "arena-range")
            auto arena_range = [&](size_t toff, size_t len) -> bool {
                const ProcMem pm0  = proc_mem();
                const double  rd0  = g.q4s_read_ms;
                const char * abase = (const char *) pr.ar;
                const size_t blk   = pr.blk;
                auto arena_copy = [&](char * dst, size_t off, size_t n) {
                    if (pr.ar_str == (uint64_t) blk) { std::memcpy(dst, abase + off, n); return; }
                    for (size_t d = 0; d < n; ) {
                        const size_t o = off + d, e = o / blk, w = o % blk;
                        const size_t m = std::min(n - d, blk - w);
                        std::memcpy(dst + d, abase + e * (size_t) pr.ar_str + w, m);
                        d += m;
                    }
                };
                if (!pipelined) {
                    g.q4s_rbuf.resize(len);
                    const clk::time_point r0 = clk::now();
                    arena_copy(g.q4s_rbuf.data(), toff, len);
                    g.q4s_read_ms += std::chrono::duration<double, std::milli>(clk::now() - r0).count();
                    ggml_backend_tensor_set(pr.t, g.q4s_rbuf.data(), toff, len);
                } else {
                    for (size_t done = 0; done < len; cur ^= 1) {
                        const size_t n = std::min(g.q4s_slice_sz, len - done);
                        if (g.q4s_ev_live[cur]) { ggml_backend_event_synchronize(g.q4s_ev[cur]); g.q4s_ev_live[cur] = false; }
                        const clk::time_point r0 = clk::now();
                        arena_copy(g.q4s_slice[cur], toff + done, n);
                        g.q4s_read_ms += std::chrono::duration<double, std::milli>(clk::now() - r0).count();
                        ggml_backend_t ub = g.backend;
                        ggml_backend_tensor_set_async(ub, pr.t, g.q4s_slice[cur], toff + done, n);
                        ggml_backend_event_record(g.q4s_ev[cur], ub);
                        g.q4s_ev_live[cur] = true;
                        done += n;
                    }
                }
                const ProcMem pm1 = proc_mem();
                g.q4s_faults      += pm1.faults - pm0.faults;
                // Residency trip (findings section 179): the arena is a source only while it is faster
                // than the file. A trimmed arena pays one page-file read per 4 KiB: measured 22 s for a
                // full sweep with 5.8 GiB of RAM free (0.36 GB/s) and 0.65 GB/s at 31% faults, against
                // ~2.8 GB/s from the SSD and ~7 GB/s resident. The rate is judged over WINDOWS of
                // >= 256 MiB of arena copies (a cumulative mean hid a slow stretch for ~3 GiB): a window
                // under 1 GB/s means the rest of THIS prompt reads the file; the next prompt tries
                // again. 1 GB/s keeps a clean box's first sweep (~1.9 GB/s, which warms the arena for
                // every later prompt) on the arena.
                g.q4s_ar_p_bytes += (long long) len;
                g.q4s_ar_p_ms    += g.q4s_read_ms - rd0;
                if (g.q4s_ar_p_bytes >= (256ll << 20)) {
                    const double gbs = g.q4s_ar_p_ms > 0 ? (double) g.q4s_ar_p_bytes / (g.q4s_ar_p_ms * 1e6) : 1e9;
                    if (gbs < 1.0 && !g.q4s_arena_trip.exchange(true))
                        std::cerr << "moe_dual: q4stream: arena source TRIPPED at l" << il << " ("
                                  << (g.q4s_ar_p_bytes >> 20) << " MiB in " << (long long) g.q4s_ar_p_ms
                                  << " ms = " << gbs << " GB/s, " << (pm1.faults - pm0.faults)
                                  << " faults in the last copy: not resident) -- the rest of this prompt"
                                     " stages from the file\n";
                    g.q4s_ar_p_bytes = 0; g.q4s_ar_p_ms = 0.0;   // next window
                }
                g.q4s_ws_mib       = pm1.ws_mib;
                g.q4s_bytes       += (long long) len;
                g.q4s_arena_bytes += (long long) len;
                return true;
            };
            FILE * f = (shard >= 0 && shard < (int) g.q4s_f.size()) ? g.q4s_f[shard] : nullptr;
            void * hfile = (shard >= 0 && shard < (int) g.q4s_hfile.size()) ? g.q4s_hfile[shard] : nullptr;
            auto file_range = [&](size_t toff, size_t len) -> bool {
                if (!f) return false;   // internal invariant: every offset must own a valid shard
                if (!pipelined) {
                    io::seek64(f, pr.off + (int64_t) toff, SEEK_SET);
                    g.q4s_rbuf.resize(len);
                    const clk::time_point r0 = clk::now();
                    if (fread(g.q4s_rbuf.data(), 1, len, f) != len) return false;
                    g.q4s_read_ms += std::chrono::duration<double, std::milli>(clk::now() - r0).count();
                    ggml_backend_tensor_set(pr.t, g.q4s_rbuf.data(), toff, len);
                    g.q4s_bytes += (long long) len;
                    return true;
                }
                // positioned read into a pinned slice: Win32 unbuffered handle when open, else fread.
                // NO_BUFFERING needs 4 KiB-aligned offset/size/dest, so the window starts at the
                // aligned-down offset and the upload skips the leading pad (first slice only).
                auto read_at = [f, hfile](char * dst, int64_t off, size_t n) -> size_t {
#if defined(_WIN32)
                    if (hfile) {
                        OVERLAPPED ov = {};
                        ov.Offset     = (DWORD) ((uint64_t) off & 0xffffffffu);
                        ov.OffsetHigh = (DWORD) ((uint64_t) off >> 32);
                        DWORD got = 0;
                        if (!ReadFile((HANDLE) hfile, dst, (DWORD) n, &got, &ov)
                            && GetLastError() != ERROR_HANDLE_EOF) return 0;
                        return (size_t) got;
                    }
#endif
                    if (!io::seek64(f, off, SEEK_SET)) return 0;
                    return fread(dst, 1, n, f);
                };
                const int64_t start = pr.off + (int64_t) toff;
                const int64_t a_off = start & ~(int64_t) 4095;
                size_t  pad  = (size_t) (start - a_off);
                int64_t foff = a_off;
                for (size_t done = 0; done < len; cur ^= 1) {
                    const size_t need = pad + (len - done);                            // window bytes we must get
                    const size_t want = std::min(g.q4s_slice_sz, (need + 4095) & ~(size_t) 4095);
                    // fence: the previous async upload FROM this slice must finish before overwriting it
                    if (g.q4s_ev_live[cur]) { ggml_backend_event_synchronize(g.q4s_ev[cur]); g.q4s_ev_live[cur] = false; }
                    const clk::time_point r0 = clk::now();
                    const size_t got = read_at(g.q4s_slice[cur], foff, want);
                    g.q4s_read_ms += std::chrono::duration<double, std::milli>(clk::now() - r0).count();
                    if (got < std::min(want, need)) return false;
                    const size_t n = std::min(len - done, got - pad);
                    // upload + its completion event on g.backend's stream; correctness rests on the
                    // host fence above, not on which stream this is
                    ggml_backend_t ub = g.backend;
                    ggml_backend_tensor_set_async(ub, pr.t, g.q4s_slice[cur] + pad, toff + done, n);
                    ggml_backend_event_record(g.q4s_ev[cur], ub);
                    g.q4s_ev_live[cur] = true;
                    done += n; foff += (int64_t) want; pad = 0;
                }
                g.q4s_bytes += (long long) len;
                return true;
            };
            auto source_range = [&](size_t toff, size_t len) -> bool {
                return from_arena ? arena_range(toff, len) : file_range(toff, len);
            };

            // P2: the weight part of a layer whose hot experts sit READY in the slab. Those pages are
            // copied device-to-device from their slot (enqueued first, on the same upload stream, fenced
            // by q4s_ev_d2d below); the runs of cold experts between them come from the source.
            moert_memmgr_page pg;
            if (pi == 0 && q4s_slab_plan(il, kind, pr.t, pr.blk, pg, hot)) {
                const size_t blk = pr.blk;
                if (s_slab_proven < 0) {
                    // once per process: one READY slot against the source's bytes for the same expert
                    int e0 = -1;
                    for (int e = 0; e < (int) hot.size(); ++e) if (hot[(size_t) e] >= 0) { e0 = e; break; }
                    std::vector<char> a(blk), b(blk);
                    bool same = false;
                    if (e0 >= 0 && f && io::seek64(f, pr.off + (int64_t) e0 * (int64_t) blk, SEEK_SET) &&
                        fread(a.data(), 1, blk, f) == blk) {
                        ggml_backend_tensor_get(g.slab_bytes_t, b.data(),
                                                (size_t) (pg.region_off + (uint64_t) hot[(size_t) e0] * blk), blk);
                        same = std::memcmp(a.data(), b.data(), blk) == 0;
                    }
                    s_slab_proven = same ? 1 : 0;
                    std::cerr << "moe_dual: q4stream: slab source " << (same ? "PROVEN" : "DISPROVEN")
                              << " (l" << il << " kind " << kind << " e" << e0 << " slot "
                              << (e0 >= 0 ? hot[(size_t) e0] : -1) << ": slab page "
                              << (same ? "==" : "!=") << " source bytes)"
                              << (same ? " -- hot experts are staged D2D from now on"
                                       : " -- staging everything from the source") << "\n";
                    if (!same) hot.clear();
                }
                if (!hot.empty()) {
                    ggml_backend_t ub = g.backend;
                    const size_t dst0 = (size_t) ((const char *) pr.t->data -
                                                  (const char *) ggml_backend_buffer_get_base(g.q4s_buf_r[(size_t) rs]));
                    ggml_init_params vp;
                    vp.mem_size   = (size_t) (2 * hot.size() + 8) * ggml_tensor_overhead();
                    vp.mem_buffer = nullptr;
                    vp.no_alloc   = true;
                    ggml_context * vctx = ggml_init(vp);
                    long long n_hot = 0;
                    for (int e = 0; e < (int) hot.size(); ++e) {
                        const int sl = hot[(size_t) e];
                        if (sl < 0) continue;
                        ggml_tensor * src = ggml_view_1d(vctx, g.slab_bytes_t, (int64_t) blk,
                                                         (size_t) (pg.region_off + (uint64_t) sl * blk));
                        ggml_tensor * dst = ggml_view_1d(vctx, g.q4s_bytes_r[(size_t) rs], (int64_t) blk,
                                                         dst0 + (size_t) e * blk);
                        ggml_backend_tensor_copy_async(ub, ub, src, dst);   // D2D on the upload stream
                        ++n_hot;
                    }
                    ggml_backend_event_record(g.q4s_ev_d2d, ub);
                    g.q4s_ev_d2d_live = true;
                    ggml_free(vctx);   // views are metadata only: the copies hold raw device pointers
                    g.q4s_slab_bytes   += n_hot * (long long) blk;
                    g.q4s_slab_experts += n_hot;
                    for (int e = 0; e < (int) hot.size() && ok; ) {
                        if (hot[(size_t) e] >= 0) { ++e; continue; }
                        int e1 = e;
                        while (e1 < (int) hot.size() && hot[(size_t) e1] < 0) ++e1;
                        ok = source_range((size_t) e * blk, (size_t) (e1 - e) * blk);
                        e = e1;
                    }
                    continue;
                }
            }
            ok = source_range(0, bytes);
        }
    }
    if (pipelined) {
        // Fence ONLY our own uploads (also on error: no in-flight copy may outlive this call): waiting on the
        // per-slice events gives the guarantee without serializing the prefetch thread against the whole
        // backend stream. (history: docs/code-comments-archive-2026-10-05.md, "upload-fence")
        for (int i = 0; i < 2; ++i)
            if (g.q4s_ev_live[i]) { ggml_backend_event_synchronize(g.q4s_ev[i]); g.q4s_ev_live[i] = false; }
        if (g.q4s_ev_d2d_live) { ggml_backend_event_synchronize(g.q4s_ev_d2d); g.q4s_ev_d2d_live = false; }
    }
    if (!ok) {
        std::cerr << "moe_dual: q4stream: source read failed (l" << il << ") — aborting graph\n";
        return false;
    }
    ++g.q4s_staged;
    const double elapsed_ms = std::chrono::duration<double, std::milli>(clk::now() - t0).count();
    g.q4s_stage_ms += elapsed_ms;
    return true;
}

// --- R1 prefetch thread ------------------------------------------------------
// Streams layers 0..n-1 in graph (layer-major) order into the ring, one slot ahead of the compute.
// Before refilling slot s for layer il it waits until layer il-ring_n has been CONSUMED (published by
// the callback), so it can never overwrite bytes an in-flight expert matmul is still reading.
static void q4s_prefetch_main() {
    const int N = g.q4s_ring_n;
    const int n_layers = g.q4s_layers_n;
    for (int il = g.q4s_pf_start_il; il < n_layers; ++il) {
        const int s = il % N;
        {
            using clk = std::chrono::steady_clock;
            const clk::time_point cv0 = clk::now();
            std::unique_lock<std::mutex> lk(g.q4s_ring_mtx);
            g.q4s_ring_cv.wait(lk, [&] { return !g.q4s_pf_run || il - N <= g.q4s_consumed_upto; });
            if (!g.q4s_pf_run) return;                       // phase closed / shutdown
        }
        const bool ok = q4s_stage_layer(il, s, g.q4s_wt_r[s], g.q4s_st_r[s]);
        {
            std::lock_guard<std::mutex> lk(g.q4s_ring_mtx);
            if (!ok) { g.q4s_pf_fail = true; g.q4s_ring_cv.notify_all(); return; }
            g.q4s_slot_layer[s] = il;
            g.q4s_slot_ready[s] = 1;
        }
        g.q4s_ring_cv.notify_all();
    }
}

// Join the prefetcher (idempotent). MUST run before the staging buffers are freed or the hot slab is
// restored -- no read may be in flight when that VRAM is re-committed.
static void q4s_ring_stop() {
    if (!g.q4s_pf_thread.joinable()) return;
    { std::lock_guard<std::mutex> lk(g.q4s_ring_mtx); g.q4s_pf_run = false; }
    g.q4s_ring_cv.notify_all();
    g.q4s_pf_thread.join();
}

// Arm a fresh prefetch pass for one llama_decode call (each call re-streams every layer).
// The prefetch thread and the graph's first topk(0) call start from the same line, so layer 0's
// read is fully exposed -- nothing has been computed yet to hide it behind. A synchronous
// warm-start that primed slots 0..N-2 on the calling thread was tried behind
// MOE_DUAL_Q4S_RING_WARMSTART and never enabled: it moves that unavoidable cost earlier rather
// than removing it (some layer has to be first), and it is not what the product measured.
static void q4s_ring_start() {
    q4s_ring_stop();
    const int N = g.q4s_ring_n;
    {
        std::lock_guard<std::mutex> lk(g.q4s_ring_mtx);
        g.q4s_slot_layer.assign(N, -1);
        g.q4s_slot_ready.assign(N, 0);
        g.q4s_consumed_upto = -1;
        g.q4s_pf_fail = false;
        g.q4s_pf_start_il = 0;
    }
    g.q4s_pf_run  = true;
    g.q4s_pf_thread = std::thread(q4s_prefetch_main);
}

// --- P3: residency decided at the boundary ---------------------------------------------------------
// Which staging slot holds layer il right now, or -1 if it no longer does.
static int q4s_slot_of(int il) {
    if (g.q4s_ring_n <= 1) return 0;   // the single slot holds the last staged layer (the caller knows which)
    const int s = il % g.q4s_ring_n;
    std::lock_guard<std::mutex> lk(g.q4s_ring_mtx);
    return (s < (int) g.q4s_slot_layer.size() && g.q4s_slot_layer[(size_t) s] == il) ? s : -1;
}

// Copy expert e of layer il from the staging slot g.bnd_rs into hot slot `slot` (every page group the
// memmgr lays out for the layer, weights and scales), synchronously. Called by the policy, through
// cb_stream_expert, during q4s_boundary_decide. 1 = in place (READY), 0 = refused (slot left free).
int32_t q4stream_boundary_fill(int32_t il, int32_t e, int32_t slot) {
    const int rs = g.bnd_rs;
    if (rs < 0 || rs >= (int) g.q4s_bytes_r.size() || !g.slab_bytes_t || !g.q4s_ev_d2d) { ++g.bnd_fail; return 0; }
    ggml_backend_t ub = g.backend;
    const char * sbase = (const char *) ggml_backend_buffer_get_base(g.q4s_buf_r[(size_t) rs]);
    ggml_init_params vp;
    vp.mem_size = 16 * ggml_tensor_overhead(); vp.mem_buffer = nullptr; vp.no_alloc = true;
    ggml_context * vctx = ggml_init(vp);
    bool ok = true;
    int  n  = 0;
    for (int kind = 0; kind < 2 && ok; ++kind)
        for (int ws = 0; ws < 2 && ok; ++ws) {
            moert_memmgr_page pg;
            if (!moert_memmgr_page_info(g.mm, il, kind, ws, &pg)) continue;   // the slab has no such group
            const ggml_tensor * t = (ws ? g.q4s_st_r : g.q4s_wt_r)[(size_t) rs][(size_t) il][(size_t) kind];
            const size_t blk = ws ? g.q4s_blk_s[(size_t) il][(size_t) kind] : g.q4s_blk[(size_t) il][(size_t) kind];
            if (!t || pg.type_tag != (int32_t) t->type || pg.bytes_per_expert != (uint64_t) blk) { ok = false; break; }
            ggml_tensor * src = ggml_view_1d(vctx, g.q4s_bytes_r[(size_t) rs], (int64_t) blk,
                                             (size_t) ((const char *) t->data - sbase) + (size_t) e * blk);
            ggml_tensor * dst = ggml_view_1d(vctx, g.slab_bytes_t, (int64_t) blk,
                                             (size_t) (pg.region_off + (uint64_t) slot * blk));
            ggml_backend_tensor_copy_async(ub, ub, src, dst);
            ++n;
        }
    ggml_free(vctx);
    if (n > 0) {
        ggml_backend_event_record(g.q4s_ev_d2d, ub);
        ggml_backend_event_synchronize(g.q4s_ev_d2d);
    }
    if (!ok || n == 0) { ++g.bnd_fail; return 0; }
    return 1;
}

// The staging and the promotions read the SAME file (the `quality` regime: both are the Q4 GGUF), so a
// staged expert page is byte-for-byte what a promotion would write -- the boundary needs no other proof.
static bool q4s_same_source_file() {
    if (g.q4s_src_path.empty() || g.dyn_src_path.empty()) return false;
    std::error_code ec;
    return std::filesystem::equivalent(g.q4s_src_path, g.dyn_src_path, ec) && !ec;
}

// Re-decide layer il with its prompt counts (see moert_policy_boundary_layer). Only while the staging
// still holds the layer, the slab is live, and the slab bytes are the staging's bytes.
static void q4s_boundary_decide(int il) {
    if (il < 0 || il >= (int) g.bnd_pending.size() || !g.bnd_pending[(size_t) il]) return;
    g.bnd_pending[(size_t) il] = 0;
    // the copy-mode exclusive arena tracks slot contents itself; on a mapped pack it never needs to
    if (!g.pol || !g.mm || !g.slab_buf || slab_evicted() ||
        (g.hcfg.arena_exclusive && !moert_memmgr_cold_arena_mapped(g.mm))) return;
    if (s_slab_proven == 0 || g.bnd_ntok[(size_t) il] <= 0) return;
    // The staging holds the q4stream source's bytes; a slot must hold the PROMOTION source's. Proven
    // directly once a READY slot can be compared (s_slab_proven, staging path), or -- on an empty slab,
    // the boot-slab first prompt -- transitively: the armed arena source was byte-verified against the
    // q4stream file, and the arena IS the hot tier's bytes in the full-IQ2 regime.
    if (s_slab_proven < 0 && !(g.q4s_arena_mode && moert_memmgr_arena_matches_hot(g.mm)) && !q4s_same_source_file()) return;
    const int rs = q4s_slot_of(il);
    if (rs < 0) return;
    using clk = std::chrono::steady_clock;
    const clk::time_point t0 = clk::now();
    g.bnd_il = il; g.bnd_rs = rs;
    const int32_t np = moert_policy_boundary_layer(g.pol, il, g.bnd_cnt[(size_t) il].data(),
                                                   g.bnd_ntok[(size_t) il], g.bnd_w, g.bnd_r);   // w, r: findings 180
    g.bnd_il = -1;
    if (np >= 0) { g.bnd_promos += np; ++g.bnd_layers; }
    g.bnd_ms += std::chrono::duration<double, std::milli>(clk::now() - t0).count();
}

void q4stream_count_topk(int il, const ggml_tensor * t) {
    if (!g.q4s_on || !t || t->type != GGML_TYPE_I32) return;
    if (il < 0 || il >= (int) g.bnd_cnt.size()) return;
    const int64_t n_used = t->ne[0], n_tok = t->ne[1];
    if (n_used <= 0 || n_tok <= 0) return;
    static std::vector<char> buf;
    const size_t nb = ggml_nbytes(t);   // the strided span of a (possibly) non-contiguous view
    buf.resize(nb);
    ggml_backend_tensor_get(t, buf.data(), 0, nb);
    auto & c = g.bnd_cnt[(size_t) il];
    for (int64_t r = 0; r < n_tok; ++r)
        for (int64_t j = 0; j < n_used; ++j) {
            const int32_t e = *(const int32_t *) (buf.data() + r * t->nb[1] + j * t->nb[0]);
            if (e >= 0 && e < (int32_t) c.size()) ++c[(size_t) e];
        }
    g.bnd_ntok[(size_t) il] += n_tok;
    g.bnd_pending[(size_t) il] = 1;
}

// Called from the ffn_moe_topk eval callback, backend SYNCHRONIZED, before layer il's expert matmuls.
// ring_n == 1: stage layer il right here (the original path). ring_n > 1: the prefetcher already read
// it -- just wait for its slot, then publish that layers < il are consumed so the slot they used can
// be refilled. The wait time is telemetry (q4s_wait_ms): ~0 means the overlap fully hid the SSD read.
bool q4stream_on_topk(int il) {
    if (!g.q4s_on) return true;
    if (g.q4s_error) {
        std::cerr << "moe_dual: q4stream: aborting graph (incomplete promotion source for this model)\n";
        return false;
    }
    if (il < 0 || il >= (int) g.q4s_pending.size() || !g.q4s_pending[il]) return true;
    g.q4s_pending[il] = 0;
    // P3: layer il-1 has finished computing for every ubatch (this is layer il's first callback) and the
    // staging still holds it -- decide its residency before anything restages over it
    if (il > 0 && !g.bnd_pending.empty() && g.bnd_pending[(size_t) (il - 1)]) q4s_boundary_decide(il - 1);

    if (g.q4s_ring_n <= 1) return q4s_stage_layer(il, 0, g.q4s_wt, g.q4s_st);

    const int s = il % g.q4s_ring_n;
    using clk = std::chrono::steady_clock;
    const clk::time_point w0 = clk::now();
    {
        std::unique_lock<std::mutex> lk(g.q4s_ring_mtx);
        g.q4s_ring_cv.wait(lk, [&] {
            return g.q4s_pf_fail || (g.q4s_slot_ready[s] && g.q4s_slot_layer[s] == il);
        });
        if (g.q4s_pf_fail) {
            std::cerr << "moe_dual: q4stream: prefetch read failed (l" << il << ") — aborting graph\n";
            return false;
        }
        // the backend is synchronized here => every layer < il has finished computing
        g.q4s_consumed_upto = il - 1;
        if (il > 0) g.q4s_slot_ready[(il - 1) % g.q4s_ring_n] = 0;
    }
    g.q4s_ring_cv.notify_all();
    const double waited_ms = std::chrono::duration<double, std::milli>(clk::now() - w0).count();
    g.q4s_wait_ms += waited_ms;
    return true;
}

Q4StreamStats q4stream_stats() {
    Q4StreamStats s;
    s.bytes_streamed = g.q4s_bytes;
    s.layers_staged  = g.q4s_staged;
    s.stage_ms       = g.q4s_stage_ms;
    s.read_ms        = g.q4s_read_ms;
    s.buf_bytes      = g.q4s_buf ? (long long) ggml_backend_buffer_get_size(g.q4s_buf) : 0;
    s.wait_ms        = g.q4s_wait_ms;
    s.ring_n         = g.q4s_ring_n;
    s.arena_bytes    = g.q4s_arena_bytes;
    s.arena_mode     = g.q4s_arena_mode != 0;
    s.arena_faults   = g.q4s_faults;
    s.ws_mib         = g.q4s_ws_mib;
    s.slab_bytes     = g.q4s_slab_bytes;
    s.slab_experts   = g.q4s_slab_experts;
    s.bnd_layers     = g.bnd_layers;
    s.bnd_promos     = g.bnd_promos;
    s.bnd_fail       = g.bnd_fail;
    s.bnd_ms         = g.bnd_ms;
    return s;
}

}  // namespace moe_dual
