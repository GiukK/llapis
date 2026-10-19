#include "moe_dual.hpp"
#include "moe_dual_state.hpp"

#include "io_util.hpp"
#include "llapis_log_filter.hpp"   // C5 phase tracing (llapis_phase)
#include "moe_sidecar.hpp"
#include "moe_xfer.hpp"
#include "mtp_forecast.hpp"
#include "moert/cuda.h"   // island config struct (pushed through the ggml-cuda forwarder)
#include "moert/registry.h"   // the cheapest source of an expert's bytes (transfer engine, P1)

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "gguf.h"
#include "llama.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <iostream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

// The ggml-cuda host services (instant pin, cold registration, route counters) live in ggml-cuda.dll,
// so they are only linkable in the CUDA build; MOE_DUAL_HAVE_CUDA_MAP is defined by CMake there.
// Without it, Plan C never activates.
#ifdef MOE_DUAL_HAVE_CUDA_MAP
#include "ggml-cuda.h"
#endif

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <psapi.h>            // GetProcessMemoryInfo (RAM-dedup RSS telemetry)
#pragma comment(lib, "psapi.lib")
#endif

// ===========================================================================
// moe_dual runtime core: the dual-gather graph hook (hot pool vs cold tier: fused selective op, Plan C
// staging, stock fallback for batched calls), the ggml/file callbacks the moert policy + memmgr drive,
// asynchronous promotion and the exclusive arena, and the decode-time API (refresh / telemetry).
// Initialization: moe_dual_init.cpp; Q4Stream prefill: moe_dual_q4stream.cpp; state: moe_dual_state.hpp.
// ===========================================================================

namespace moe_dual {

// The one process-lifetime runtime state (moe_dual_state.hpp).
State g;

// Skeleton 1: place the hot pool in VRAM. Prefer a GPU device from the backend registry
// (the model itself can stay in RAM at --ngl 0; only the hot Q4 experts go to VRAM).
ggml_backend_t pick_pack_backend() {
    {
        ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
        if (dev) {
            ggml_backend_t b = ggml_backend_dev_init(dev, nullptr);
            if (b) { std::cerr << "moe_dual: hot pool -> GPU '" << ggml_backend_dev_name(dev) << "'\n"; return b; }
        }
        std::cerr << "moe_dual: no GPU backend available; hot pool on CPU\n";
    }
    return ggml_backend_cpu_init();
}

bool ends_with(const std::string & s, const char * suf) {
    const size_t n = std::strlen(suf);
    return s.size() >= n && s.compare(s.size() - n, n, suf) == 0;
}

// M3a: device memory of the pool backend's device (free/total). False when the pool is not on
// a GPU device (CPU fallback / MOE_DUAL_CPU=1) - the VRAM guardrail is then inert by design.
bool pool_dev_memory(size_t * free_b, size_t * total_b) {
    if (!g.backend) return false;
    ggml_backend_dev_t dev = ggml_backend_get_device(g.backend);
    if (!dev || ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU) return false;
    ggml_backend_dev_memory(dev, free_b, total_b);
    return true;
}

// Replicate build_lora_mm_id's per-expert scale application (no-op if w_s == null).
static ggml_tensor * apply_scale(ggml_context * ctx, ggml_tensor * res, ggml_tensor * w_s, ggml_tensor * ids) {
    if (!w_s) return res;
    const int64_t n_expert = w_s->ne[0];
    const int64_t n_tokens = ids->ne[1];
    ggml_tensor * s = ggml_reshape_3d(ctx, w_s, 1, n_expert, 1);
    s = ggml_repeat_4d(ctx, s, 1, n_expert, n_tokens, 1);
    s = ggml_get_rows(ctx, s, ids);
    return ggml_mul(ctx, res, s);
}


// Parse the MOE_DUAL_* env compat surface once (called from every init entry point) and push
// the CUDA island's slice of it through the ggml-cuda forwarder (the island itself reads no
// environment —; config-at-init like the moert policy/memmgr).
// cold_ffn mode: layer il's down cold pool in the owned arena (the gate_up pool travels in the job).
static bool cold_down_pool(int32_t il, moe_sidecar::DownPool * out) {
    if (!g.mm || !g.cold_arena_on || !out) return false;
    moert_cold_arena_view aw;
    if (!moert_memmgr_get_cold_arena(g.mm, il, 1, 0, &aw)) return false;
    out->base = aw.base; out->bytes = aw.stride; out->stride = aw.arena_stride ? aw.arena_stride : aw.stride;
    out->type = aw.type_tag; out->d_in = (int32_t) aw.ne[0]; out->d_out = (int32_t) aw.ne[1];
    return true;
}

void parse_hook_env() {
    if (g.hcfg.parsed) return;
    g.hcfg.parsed = true;
    auto flag = [](const char * n) { const char * v = std::getenv(n); return (v && v[0] == '1') ? 1 : 0; };
    g.hcfg.fused        = flag("MOE_DUAL_FUSED");
    if (const char * v = std::getenv("MOE_DUAL_PLANC"))           g.hcfg.planc     = std::atoi(v);
    if (const char * v = std::getenv("MOE_DUAL_SPEC_K")) { g.hcfg.spec_kmax = std::atoi(v);
        // Ceiling = 4: the single-warp partition packs Pmax = n_used*n_tokens routed pairs into one
        // 32-lane warp (ffn_moe.cu:78), and this model's decode top_k = n_used = 8, so n_tokens can go
        // up to 32/8 = 4 (verify 4-wide, draft-n-max 3) before the warp overflows. n_tokens=2 stays the
        // validated default; 3-4 ride the SAME gathered path with more of the pre-sized-32 buffers used.
        if (g.hcfg.spec_kmax < 1) g.hcfg.spec_kmax = 1; if (g.hcfg.spec_kmax > 4) g.hcfg.spec_kmax = 4; }
    // Exclusive arena: DEFAULT ON since claude dump findings 283 (the hot experts' pages of the mapped pack leave the RAM,
    // which the cold experts' pages need: at 5 GiB the decode paged 2-3 thousand reads/s); MOE_DUAL_ARENA_EXCLUSIVE=0
    // turns it off. A default (not explicit) request is dropped when the arena is a copy (moe_dual_init.cpp): there it
    // would turn off the boundary residency and the arena staging.
    { const char * v = std::getenv("MOE_DUAL_ARENA_EXCLUSIVE"); g.hcfg.arena_exclusive = (!v || !v[0]) ? 1 : (v[0] == '1'); }
    // Variant P (research, findings §11): capture the pure-GPU segments between the MoE ops in CUDA
    // graphs while the op itself runs eager. ggml-cuda reads GGML_CUDA_GRAPH_SEGMENTS; the child server
    // only inherits MOE_* from its profile, so this alias sets it in-process before the first compute.
    if (flag("MOE_DUAL_GRAPH_SEGMENTS")) {
#if defined(_WIN32)
        _putenv_s("GGML_CUDA_GRAPH_SEGMENTS", "1");
#else
        setenv("GGML_CUDA_GRAPH_SEGMENTS", "1", 1);
#endif
        std::cerr << "moe_dual: graph segments ON (GGML_CUDA_GRAPH_SEGMENTS=1; default since findings 198, MOE_DUAL_GRAPH_SEGMENTS=0 turns it off)\n";
    }
#ifdef MOE_DUAL_HAVE_CUDA_MAP
    moert_cuda_config c;
    moert_cuda_config_defaults(&c);
    // the guard is registered whenever the exclusive arena is requested; it is a no-op (one atomic
    // load) until moe_xfer::exclusive_start actually arms it at the first policy step
    if (g.hcfg.arena_exclusive) c.cold_guard_fn = (uint64_t) (uintptr_t) &moe_xfer_cold_guard;
    c.copy_stream   = flag("MOE_DUAL_COPYSTREAM");
    // LLapis V2 CPU cold sidecar. Only the
    // planc/IQ2 cold representation is implemented live — a mislabeled MODE must not run silently.
    if (flag("MOE_DUAL_COLD_CPU")) {
        // workers default 10 (was 6): gated 2026-07-10 slice-A
        // stats arms — gate_up's IQ2_XXS vec_dot is lookup-bound (not bandwidth-bound like
        // down's IQ4_NL) and scales past 6 cores; w=10 beat w=6 by ~3.5 ms/tok replicated at
        // MAX=8. Outputs are thread-count-invariant (per-row, no cross-thread reduction).
        int workers = 10, maxc = 3;
        if (const char * v = std::getenv("MOE_DUAL_COLD_CPU_WORKERS")) workers = std::atoi(v);
        if (const char * v = std::getenv("MOE_DUAL_COLD_CPU_MAX"))     maxc    = std::atoi(v);
        moe_sidecar::init(workers);
        c.coldcpu       = 1;
        c.coldcpu_max   = maxc;
        c.coldcpu_fn    = (uint64_t) (uintptr_t) &moe_sidecar_compute;
        // The cold FFN of a decode layer is ONE asynchronous host job (findings 206, default since 220): the gate_up
        // op submits gate_up -> GELU-GLU -> down to a coordinator thread and returns, the dense FFN runs on the GPU
        // meanwhile, the down op waits for the job. Calls it cannot take (Pmax > 32: prefill) keep the per-kind
        // synchronous sidecar behind a per-fire D2H and a host block.
        c.cold_ffn = 1;
        c.coldffn_submit_fn = (uint64_t) (uintptr_t) &moe_sidecar_ffn_submit;
        c.coldffn_wait_fn   = (uint64_t) (uintptr_t) &moe_sidecar_ffn_wait;
        moe_sidecar::set_down_resolver(&cold_down_pool);
        if (g.hcfg.arena_exclusive) moe_sidecar::set_cold_guard(&moe_xfer_cold_guard);
        std::cerr << "moe_dual: CPU cold sidecar active: fires with n_cold <= " << maxc
                  << " compute cold pairs on host (" << workers << " workers), weights never cross PCIe;"
                  << " above the threshold -> Plan C staging ring\n";
    }
    // A CUDA graph over the MoE op is only valid with a provably sync-free op, and this one is
    // not: the RAM-cold path syncs mid-op to read the cold list — "operation not permitted when
    // stream is capturing". Nothing in the product enables that graph, so the hazard is closed by
    // construction rather than by a guard.
    ggml_backend_cuda_moe_dual_configure(&c);

    if (c.coldcpu) {
        std::cerr << "moe_dual: cold executor = legacy-sync"
                  << " | MTP verify width <= " << g.hcfg.spec_kmax
                  << " | coldcpu_max = " << c.coldcpu_max
                  << "\n";
    }
#endif
}

// The hook: combine the cold (all-expert, low-precision) and hot (K-expert, high-precision)
// expert matmuls per-expert via the residency mask. Returns NULL to fall back to stock.
ggml_tensor * hook(ggml_context * ctx, int il, int kind,
                   ggml_tensor * w_cold, ggml_tensor * cur,
                   ggml_tensor * ids, ggml_tensor * w_cold_s) {
    const int64_t n_used   = ids->ne[0];
    const int64_t n_tokens = ids->ne[1];

    // kinds 2/3 (separate gate/up) were deleted in M2 — gemma4 only routes 0 (merged gate_up)
    // and 1 (down) here. Defensive stock fallback if another arch ever calls differently.
    if (kind < 0 || kind > 1) return nullptr;


    // Q4STREAM prefill: inside q4stream_begin/end, batched chunks (n_tokens > 1) compute the expert
    // matmul against the FULL-LAYER Q4 staging tensors — all n_expert experts of layer `il`, uploaded
    // just-in-time by q4stream_on_topk at this layer's ffn_moe_topk sync point. Global expert ids
    // index the staging tensor directly (it holds experts 0..n_expert-1 in order), so the stock
    // mul_mat_id + per-expert scale path applies unchanged. NEVER degrades silently: a layer/kind
    // missing from the Q4 source arms q4s_error and the next topk callback aborts the graph.
    if (n_tokens > 1 && g.q4s_on) {
        // R1: with a staging ring (q4s_ring_n > 1) layer il computes out of slot
        // il % ring_n, which the prefetch thread filled while an earlier layer was still computing.
        // ring_n == 1 resolves to slot 0 == the historical single staging buffer, unchanged.
        const bool  ring   = g.q4s_ring_n > 1 && !g.q4s_wt_r.empty();
        const int   slot   = ring ? (il % g.q4s_ring_n) : 0;
        const auto & wt_src = ring ? g.q4s_wt_r[slot] : g.q4s_wt;
        const auto & st_src = ring ? g.q4s_st_r[slot] : g.q4s_st;
        ggml_tensor * w_full = (il >= 0 && il < (int) wt_src.size()) ? wt_src[il][kind] : nullptr;
        if (!w_full) {
            std::cerr << "moe_dual: q4stream: no promotion-source weight for layer " << il << " kind " << kind
                      << " — refusing to degrade to cold precision (graph will abort)\n";
            g.q4s_error = true;
            return nullptr;
        }
        g.q4s_pending[il] = 1;
        ggml_tensor * out = ggml_mul_mat_id(ctx, w_full, cur, ids);
        return apply_scale(ctx, out, st_src[il][kind], ids);
    }

    if (il < 0 || il >= g.n_layers) return nullptr;

    // M1: register the borrowed RAM cold-tier view (base+stride into llama.cpp's --cpu-moe-placed
    // tensors) with the memmgr, once per (layer, kind) per load — the loader stays the owner, the
    // memmgr just holds a read-only view. No consumer reads it yet (that's M3's miss fast path);
    // registering from the live graph proves the borrow interface on the real tensors.
    if (g.mm) {
        auto reg_cold = [il, kind](const ggml_tensor * t, int ws) {
            if (!t || !t->buffer || !t->data || !ggml_backend_buffer_is_host(t->buffer)) return;
            moert_cold_view have;
            if (moert_memmgr_get_cold_view(g.mm, il, kind, ws, &have)) return;   // re-derive once per load
            int axis = -1;
            for (int d = GGML_MAX_DIMS - 1; d >= 0; --d) if ((int) t->ne[d] == g.n_expert) { axis = d; break; }
            if (axis < 0) return;
            moert_cold_view v;
            v.base     = t->data;
            v.stride   = t->nb[axis];
            v.type_tag = (int32_t) t->type;
            v.n_expert = g.n_expert;
            moert_memmgr_set_cold_view(g.mm, il, kind, ws, &v);
            // 3b: prove the OWNED arena == the borrowed tensor for this (layer,kind,ws), byte-for-
            // byte, on a sample of experts — so sourcing cold from the arena is bit-identical to
            // borrowing. Reads only a few borrowed expert pages (mmap-friendly).
            if (g.cold_arena_on) {
                moert_cold_arena_view a;
                if (moert_memmgr_get_cold_arena(g.mm, il, kind, ws, &a) && a.stride == v.stride) {
                    const int ne = g.n_expert;
                    for (int e : { 0, ne / 2, ne - 1 }) {
                        const void * bref = (const char *) t->data + (size_t) e * v.stride;
                        const void * aref = (const char *) a.base  + (size_t) e * a.stride;
                        if (std::memcmp(bref, aref, (size_t) v.stride) == 0) ++g.cold_arena_verify_ok;
                        else ++g.cold_arena_verify_bad;
                    }
                } else {
                    ++g.cold_arena_verify_bad;
                }
            }
        };
        reg_cold(w_cold,   0);
        reg_cold(w_cold_s, 1);
    }

    // Batched-prefill fallback under the loader cut: the expert WEIGHT tensors are shape-only, so give them
    // a read-only data view into moert's owned arena (byte-identical layout, proven in reg_cold) and the
    // stock path works as before the cut. One-time per (layer, kind); moert owns the bytes, llama borrows.
    // (history: docs/code-comments-archive-2026-10-05.md, "cut-fallback")
    if (g.cold_arena_on && g.mm) {
        auto alias_cold = [il, kind](ggml_tensor * t, int ws) {
            if (!t || t->data) return;                  // already loaded (scale) or no tensor -> leave it
            moert_cold_arena_view a;
            if (moert_memmgr_get_cold_arena(g.mm, il, kind, ws, &a) && a.base) t->data = (void *) a.base;
        };
        alias_cold(w_cold,   0);
        alias_cold(w_cold_s, 1);
    }

    // Vision evict window: while the hot slab is NOT resident nothing may reference the pool tensors
    // (their buffer is freed); no decode should reach here (C4c invariant) -- serve the stock path as
    // hardening rather than touching freed pointers.
    if (slab_evicted()) return nullptr;

    ggml_tensor * w_hot = g.w_hot[il][kind];
    if (!w_hot) return nullptr;  // no pack for this layer/kind -> stock path
    ggml_tensor * s_hot = g.s_hot[il][kind];

    // Phase 3b: the FUSED selective op (each routed pair computed once at its tier). Opt-in via
    // MOE_DUAL_FUSED=1; decode-only (the single-warp partition needs Pmax = n_used*n_tokens <= 32,
    // so n_tokens must be 1). Falls back to the compute-both blend below for prefill / when the
    // residency mirror is absent. MOE_DUAL_PLANC != 0 enables the RAM-cold
    // fused delivery (staging selection; the CPU sidecar computes the cold pairs, the staging ring is
    // the over-max fallback); 0/unset = fall back to the blend. The zero-copy (2) and Variant-F de-dup
    // deliveries were removed 2026-10-03 (tag pre-prune-cold-research-2026-10-03).
    const int fused     = g.hcfg.fused;
    const int planc     = g.hcfg.planc;
    // Batched calls fall through to the STOCK single-tier path (the compute-both blend is broken above one
    // token), except n_tokens <= spec_kmax: the fused op handles the MTP verify batch through its gathered
    // path (per-pair activations), so those go to the fused op. (history: docs/code-comments-archive-2026-10-05.md, "batched-calls")
    const bool spec_batch = fused && n_tokens > 1 && n_tokens <= g.hcfg.spec_kmax;
    if (n_tokens > 1 && !spec_batch) return nullptr;

    // The fused op is CUDA-only. w_hot lives in the GPU pool; the cold pool is GPU-resident for offloaded
    // layers (cold_on_gpu, mode 0) and RAM-resident under --cpu-moe (cold_on_ram -> Plan C).
    const bool fused_ok    = fused && n_tokens <= g.hcfg.spec_kmax && !g.es_t.empty() && g.es_t[il];
    const bool cold_on_gpu = w_cold && w_cold->buffer && !ggml_backend_buffer_is_host(w_cold->buffer);
    const bool cold_on_ram = w_cold && w_cold->buffer &&  ggml_backend_buffer_is_host(w_cold->buffer);
    if (fused_ok && cold_on_gpu) {
        static long long fused_fires = 0;
        if (fused_fires++ == 0)
            std::cerr << "moe_dual: FUSED selective op ACTIVE (GPU cold; first fire: layer " << il
                      << " kind " << kind << ")\n";
        return ggml_moe_dual(ctx, cur, ids, w_hot, w_cold,
                             g.es_t[il], g.se_t[il], g.ss_t[il],
                             s_hot, w_cold_s, kind, (int) n_used, il, 0, 0, 0, 0, 0);
    }
    // The owned arena serves cold even when the borrowed tensor is shape-only (no buffer) — the 3b
    // loader cut. w_cold is then used only for its type/ne metadata; its data is never touched.
    if (fused_ok && planc && (cold_on_ram || g.cold_arena_on)) {
        const int mode = 2;   // staging ring (the zero-copy and de-dup modes were removed 2026-10-03)
        // 3b: source the staged cold bytes from moert's OWNED arena instead of the borrowed tensor
        // (byte-identical layout, proven in reg_cold).
        const void * arena_w = nullptr, * arena_s = nullptr;
        if (g.cold_arena_on) {
            moert_cold_arena_view aw, as;
            if (moert_memmgr_get_cold_arena(g.mm, il, kind, 0, &aw)) arena_w = aw.base;
            if (w_cold_s && moert_memmgr_get_cold_arena(g.mm, il, kind, 1, &as)) arena_s = as.base;
        }
        // In staging mode pass the HOST base pointers (the op copies selected experts to VRAM per token).
        const void * wsrc = arena_w ? arena_w : (const void *) w_cold->data;
        const void * ssrc = arena_s ? arena_s : (const void *) (w_cold_s ? w_cold_s->data : nullptr);
        static long long planc_fires = 0;
        if (planc_fires++ == 0)
            std::cerr << "moe_dual: FUSED selective op ACTIVE (Plan C staging ring"
                      << " RAM cold; first fire: layer " << il << " kind " << kind << ")\n";
        return ggml_moe_dual(ctx, cur, ids, w_hot, /*w_cold=*/nullptr,
                             g.es_t[il], g.se_t[il], g.ss_t[il],
                             s_hot, /*s_cold=*/nullptr, kind, (int) n_used, il,
                             (uint64_t) (uintptr_t) wsrc, (int) w_cold->type, (int) w_cold->ne[2],
                             (uint64_t) (uintptr_t) ssrc, mode);
    }

    // Reaching here with n_tokens>1 means a speculative batch whose layer could NOT take a fused
    // path above (no GPU cold, no Plan C). The compute-both blend below is KNOWN BROKEN at
    // n_tokens>1, so fall back to STOCK exactly as the pre-spec batched path did — a spec batch
    // must never reach the blend. This keeps speculative decode safe under partial offload and
    // non-Plan-C configurations.
    if (n_tokens > 1) return nullptr;

    // cold path: all experts at low precision (base model, RAM)
    ggml_tensor * out_cold = ggml_mul_mat_id(ctx, w_cold, cur, ids);
    out_cold = apply_scale(ctx, out_cold, w_cold_s, ids);

    // remap global expert ids -> local hot-pool slots (i32 get_rows is supported)
    ggml_tensor * hm = ggml_reshape_3d(ctx, g.hotmap[il], 1, g.n_expert, 1);
    hm = ggml_repeat_4d(ctx, hm, 1, g.n_expert, n_tokens, 1);
    ggml_tensor * hids = ggml_get_rows(ctx, hm, ids);            // [1, n_used, n_tokens] i32
    hids = ggml_reshape_2d(ctx, hids, n_used, n_tokens);         // [n_used, n_tokens]

    // hot path: K experts at high precision (pack)
    ggml_tensor * out_hot = ggml_mul_mat_id(ctx, w_hot, cur, hids);
    out_hot = apply_scale(ctx, out_hot, s_hot, hids);

    // residency mask per (slot, token), broadcast over the row dim
    ggml_tensor * rm = ggml_reshape_3d(ctx, g.resid[il], 1, g.n_expert, 1);
    rm = ggml_repeat_4d(ctx, rm, 1, g.n_expert, n_tokens, 1);
    ggml_tensor * mask = ggml_get_rows(ctx, rm, ids);            // [1, n_used, n_tokens] f32

    // out = cold + mask * (hot - cold)
    ggml_tensor * diff = ggml_sub(ctx, out_hot, out_cold);
    return ggml_add(ctx, out_cold, ggml_mul(ctx, diff, mask));
}
// --- dynamic cache helpers ---------------------------------------------------

// M1 device services for the moert memory manager: memmgr decides WHICH bytes go
// WHERE in the slab (layout, page reads); these three do the ggml/device work.
// slab_alloc also creates the I8 byte-view tensor so slab_upload can address the
// slab by raw byte offset through the generic ggml_backend_tensor_set.
int32_t cb_slab_alloc(void * /*user*/, uint64_t nbytes) {
    g.slab_buf = ggml_backend_alloc_buffer(g.backend, (size_t) nbytes);
    if (!g.slab_buf) return 0;
    g.slab_total_bytes = (size_t) nbytes;   // slab_restore re-allocates at exactly this size
    ggml_backend_buffer_clear(g.slab_buf, 0);  // unpromoted slots = 0 (valid quant, no NaN; masked out anyway)
    // The byte view lives in pack_ctx (fixed size): a re-allocation at the same size (the vision
    // evict/restore) reuses it instead of adding a tensor per restore, which would exhaust the context.
    ggml_tensor * bt = ggml_get_tensor(g.pack_ctx, "moe_dual.hot_slab");
    if (bt && ggml_nbytes(bt) == (size_t) nbytes) {
        bt->buffer = nullptr;   // ggml_backend_tensor_alloc asserts an unallocated tensor
        bt->data   = nullptr;
    } else {
        bt = ggml_new_tensor_1d(g.pack_ctx, GGML_TYPE_I8, (int64_t) nbytes);
        ggml_set_name(bt, "moe_dual.hot_slab");
    }
    g.slab_bytes_t = bt;
    if (ggml_backend_tensor_alloc(g.slab_buf, g.slab_bytes_t,
                                  ggml_backend_buffer_get_base(g.slab_buf)) != GGML_STATUS_SUCCESS) {
        std::cerr << "moe_dual: slab byte-view placement failed\n";
        return 0;
    }
    return 1;
}
// Batch transport (findings 216k-216l, opt-in MOE_DUAL_RAM_DIR=1): the asynchronous promotions' copies (the transfer
// worker's slab uploads) are issued at once on the island's copy stream and the promotion is reported done when its
// event lands. Every other upload (synchronous fills, restores, rebuilds) keeps the direct path.
static bool g_dir_on = false;
// A memmgr offset past the slab is a loaned slot (findings 246): it lives in staging slot 0.
static ggml_tensor * slab_target(uint64_t & off, uint64_t nbytes) {
    if (off < (uint64_t) g.slab_total_bytes) return g.slab_bytes_t;
    if (g.loan <= 0 || g.q4s_bytes_r.empty() || !g.q4s_bytes_r[0]) return nullptr;
    off -= (uint64_t) g.slab_total_bytes;
    return off + nbytes <= (uint64_t) ggml_nbytes(g.q4s_bytes_r[0]) ? g.q4s_bytes_r[0] : nullptr;
}
int32_t cb_slab_upload(void * /*user*/, uint64_t off, const void * src, uint64_t nbytes) {
    ggml_tensor * t = slab_target(off, nbytes);
    if (!t) { std::cerr << "moe_dual: slab upload past the slab with no loan buffer (dropped)\n"; return 0; }
#ifdef MOE_DUAL_HAVE_CUDA_MAP
    if (g_dir_on && moe_xfer::on_worker() && t->data && !slab_evicted() &&
        ggml_backend_cuda_moe_dual_dir_async((char *) t->data + off, src, nbytes))
        return 1;
#endif
    ggml_backend_tensor_set(t, src, (size_t) off, (size_t) nbytes);
    return 1;
}
int32_t cb_slab_download(void * /*user*/, uint64_t off, void * dst, uint64_t nbytes) {
    if (!g.slab_bytes_t) return 0;
    ggml_tensor * t = slab_target(off, nbytes);
    if (!t) return 0;
    ggml_backend_tensor_get(t, dst, (size_t) off, (size_t) nbytes);
    return 1;
}

// --- Loaned slots (claude dump findings 246) ---------------------------------------------------------
// The prefill staging reserve (one full expert layer, ~291 MiB) is idle between prompts. With MOE_DUAL_LOAN=1 it
// carries `loan` more hot slots per layer: the memmgr gives those slots virtual offsets past the slab, the uploads
// above map them into staging slot 0, and the hot matmul reads every slot through a per-layer pointer table.
int loan_slots_for(int k_slab) {
    const char * v = std::getenv("MOE_DUAL_LOAN");
    if (!v || v[0] != '1' || !g.mm) return 0;
    if (!g.q4s_enabled || g.q4s_lean || g.q4s_bytes_r.empty() || !g.q4s_bytes_r[0]) return 0;
    const uint64_t cap = (uint64_t) ggml_nbytes(g.q4s_bytes_r[0]);
    int best = 0;
    for (int L = 1; k_slab + L < g.n_expert; ++L) {
        uint64_t need = 0;
        for (int il = 0; il < g.n_layers; ++il)
            for (int kind = 0; kind < 2; ++kind) {
                moert_memmgr_page pg;
                if (!moert_memmgr_page_info(g.mm, il, kind, 0, &pg)) continue;
                need = (need + 255) & ~(uint64_t) 255;
                need += (uint64_t) L * pg.bytes_per_expert;
            }
        if (need > cap) break;
        best = L;
    }
    return best;
}

void loan_tables_build() {
#ifdef MOE_DUAL_HAVE_CUDA_MAP
    for (int il = 0; il < g.n_layers && il < 128; ++il)
        for (int kind = 0; kind < 2; ++kind) ggml_backend_cuda_moe_dual_hot_table(il, kind, nullptr, 0, 0);
    if (g.hot_tab_buf) { ggml_backend_buffer_free(g.hot_tab_buf); g.hot_tab_buf = nullptr; }
    if (g.hot_tab_ctx) { ggml_free(g.hot_tab_ctx); g.hot_tab_ctx = nullptr; }
    if (g.loan <= 0 || !g.slab_buf || g.q4s_bytes_r.empty() || !g.q4s_bytes_r[0] || !g.q4s_bytes_r[0]->data) return;
    const int cap = g.k + g.loan;
    ggml_init_params ip; ip.mem_size = 2 * ggml_tensor_overhead() + 1024; ip.mem_buffer = nullptr; ip.no_alloc = true;
    g.hot_tab_ctx = ggml_init(ip);
    ggml_tensor * t = ggml_new_tensor_1d(g.hot_tab_ctx, GGML_TYPE_I32, (int64_t) g.n_layers * 2 * cap * 2);   // 64-bit ptrs
    g.hot_tab_buf = ggml_backend_alloc_ctx_tensors(g.hot_tab_ctx, g.backend);
    if (!g.hot_tab_buf) { std::cerr << "moe_dual: loan: pointer tables not allocated -- loan OFF\n"; g.loan = 0; return; }
    std::vector<uint64_t> h((size_t) g.n_layers * 2 * cap, 0);
    const uint64_t slab = (uint64_t) (uintptr_t) ggml_backend_buffer_get_base(g.slab_buf);
    const uint64_t stg  = (uint64_t) (uintptr_t) g.q4s_bytes_r[0]->data;
    for (int il = 0; il < g.n_layers; ++il)
        for (int kind = 0; kind < 2; ++kind)
            for (int s = 0; s < cap; ++s) {
                uint64_t off = 0;
                if (!moert_memmgr_slot_offset(g.mm, il, kind, 0, s, &off)) continue;
                h[((size_t) il * 2 + kind) * cap + s] = off < (uint64_t) g.slab_total_bytes ? slab + off
                                                                                         : stg + (off - (uint64_t) g.slab_total_bytes);
            }
    ggml_backend_tensor_set(t, h.data(), 0, h.size() * sizeof(uint64_t));
    const char * base = (const char *) t->data;
    for (int il = 0; il < g.n_layers && il < 128; ++il)
        for (int kind = 0; kind < 2; ++kind) {
            if (!g.w_hot[il][kind]) continue;
            ggml_backend_cuda_moe_dual_hot_table(il, kind, (const void * const *) (base + ((size_t) il * 2 + kind) * cap * sizeof(uint64_t)),
                                                 cap, (int) g.w_hot[il][kind]->type);
        }
#endif
}

void loan_revoke(const char * why) {
    if (g.loan <= 0 || !g.loan_live || !g.pol) return;
    xfer_settle();   // no copy may still be writing into a loaned slot
    moert_policy_set_active_k(g.pol, g.k);
    for (int il = 0; il < g.n_layers; ++il) push_residency(il);
    g.loan_live = false;
    static int n_log = 0;
    if (n_log++ < 3) std::cerr << "moe_dual: loan: " << g.loan << " slots/layer revoked (" << why << ")\n";
}

void loan_lend() {
    if (g.loan <= 0 || g.loan_live || !g.pol) return;
    if (!g.q4s_enabled || g.q4s_lean || g.q4s_bytes_r.empty() || !g.q4s_bytes_r[0] ||
        (uint64_t) ggml_nbytes(g.q4s_bytes_r[0]) < moert_memmgr_loan_bytes(g.mm)) return;
    static const void * s_stg = nullptr;   // the staging buffer the tables point into
    if (!g.hot_tab_buf || s_stg != g.q4s_bytes_r[0]->data) {   // re-created since the last lend
        loan_tables_build();
        s_stg = g.q4s_bytes_r[0]->data;
    }
    if (g.loan <= 0) return;
    moert_policy_set_active_k(g.pol, g.k + g.loan);
    for (int il = 0; il < g.n_layers; ++il) push_residency(il);
    g.loan_live = true;
    // The slots come back empty. Re-decide every layer the prompt just counted, at the boundary's low price and
    // WITHOUT folding the prompt counts again (weight 0: the prefill already did): the loaned slots take the next
    // best experts now, copied from the arena on the promotion worker, instead of 4 per token during decoding.
    if (const char * v = std::getenv("MOE_DUAL_LOAN_FILL"); !(v && v[0] == '0')) {
        long long n = 0;
        for (int il = 0; il < g.n_layers && il < (int) g.bnd_ntok.size(); ++il)
            if (g.bnd_ntok[(size_t) il] > 0)
                n += std::max(0, moert_policy_boundary_layer(g.pol, il, g.bnd_cnt[(size_t) il].data(),
                                                             g.bnd_ntok[(size_t) il], 0.0f, g.bnd_r));
        moe_xfer::flush();
        static int n_log = 0;
        if (n_log++ < 3) std::cerr << "moe_dual: loan: lent again, " << n << " promotions to fill it\n";
    }
}
static void excl_drain(const char * why);   // exclusive arena (below)
void cb_slab_free(void * /*user*/) {
    excl_drain("slab free");   // every released page is rebuilt from its slot BEFORE the slot memory goes
    moe_xfer::wait_idle();     // no asynchronous promotion may still be writing into the slab being freed
    if (g.slab_buf) { ggml_backend_buffer_free(g.slab_buf); g.slab_buf = nullptr; }
    g.slab_bytes_t = nullptr;
}

// --- Vision-transient slab evict/restore (C4c; see moe_dual.hpp) -------------------------------
// The evict drops ONLY the physical VRAM: the policy page table, the fused-op residency mirrors
// (es/se/ss, hotmap, resid) and the memmgr layout all stay live, so the restore — re-alloc at the
// finalized size, re-place the pool tensors at their region offsets (the init pass-2 replay), then
// re-stream every resident expert into its ORIGINAL slot via moert_memmgr_promote — is
// bit-identical to the pre-evict slab. Safe w.r.t. stale pointers because (a) the vision graph
// itself never references the slab (batched hook calls take the arena fallback), and (b) the
// fork resets gf_res_prev after the vision decode, so no cached graph with old w_hot pointers
// can ever run.
static bool g_slab_evicted       = false;

bool slab_evicted()      { return g_slab_evicted; }
// Variable-KV: slab_rebuild_at_k can run while the slab is EVICTED (it re-allocates at the new K,
// which IS a restore) — it clears the flag through here. Internal to the moe_dual TUs.
long long pool_free_mib() {
    size_t free_b = 0, total_b = 0;
    if (!pool_dev_memory(&free_b, &total_b)) return 0;
    return (long long) (free_b >> 20);
}

void slab_set_evicted(bool v) { g_slab_evicted = v; }

bool slab_evict(long long min_free_mib) {
    if (g_slab_evicted) return true;
    if (!g.slab_buf || !g.mm || !g.pol || !g.dynamic) {
        // nothing evictable (no dual runtime) — proceed without making room
        return true;
    }
    if (min_free_mib > 0) {
        size_t free_b = 0, total_b = 0;
        ggml_backend_dev_t dev = ggml_backend_get_device(g.backend);
        if (dev) {
            ggml_backend_dev_memory(dev, &free_b, &total_b);
            if ((long long) (free_b >> 20) >= min_free_mib) {
                std::cerr << "moe_dual: vision slab evict SKIPPED (" << (free_b >> 20)
                          << " MiB already free >= " << min_free_mib << ")\n";
                return true;   // enough room already — keep the slab hot
            }
        }
    }
    const int64_t t0 = ggml_time_us();
    ggml_backend_synchronize(g.backend);
    const long long freed_mib = (long long) (g.slab_total_bytes >> 20);
    cb_slab_free(nullptr);
    g_slab_evicted = true;
    std::cerr << "moe_dual: vision slab EVICTED (" << freed_mib << " MiB freed, "
              << (ggml_time_us() - t0) / 1000 << " ms) — restore re-streams the residents after the image\n";
    llapis_phase("Evicting hot experts from VRAM ... (%lld MiB freed, %lld ms — making room for the image compute buffer)",
                 freed_mib, (long long) ((ggml_time_us() - t0) / 1000));
    return true;
}

bool slab_restore() {
    if (!g_slab_evicted) return true;
    const int64_t t0 = ggml_time_us();
    if (!cb_slab_alloc(nullptr, g.slab_total_bytes)) {
        std::cerr << "moe_dual: CRITICAL: vision slab restore ALLOC FAILED (" << (g.slab_total_bytes >> 20)
                  << " MiB) — hot tier is GONE; decode will be cold-only garbage-free but degraded\n";
        return false;
    }
    // init pass-2 replay: re-place the typed pool tensors at their finalized region offsets
    char * base = (char *) ggml_backend_buffer_get_base(g.slab_buf);
    for (int L = 0; L < g.n_layers; ++L) {
        for (int kind = 0; kind < 2; ++kind) {
            for (int ws = 0; ws < 2; ++ws) {
                ggml_tensor * ot = (ws ? g.s_hot : g.w_hot)[L][kind];
                if (!ot) continue;
                moert_memmgr_page pg;
                if (!moert_memmgr_page_info(g.mm, L, kind, ws, &pg)) continue;
                ot->buffer = nullptr;   // ggml_backend_tensor_alloc asserts an unallocated tensor
                ot->data   = nullptr;
                if (ggml_backend_tensor_alloc(g.slab_buf, ot, base + pg.region_off) != GGML_STATUS_SUCCESS) {
                    std::cerr << "moe_dual: CRITICAL: vision slab restore: re-placement failed ("
                              << ot->name << ")\n";
                    return false;
                }
            }
        }
    }
    // re-stream every resident expert into its original slot (policy page table = ground truth)
    int n_ok = 0, n_fail = 0;
    for (int L = 0; L < g.n_layers; ++L) {
        const int32_t * se = moert_policy_slot_exp_view(g.pol, L);
        if (!se) continue;
        const int kL = klayer(L);
        for (int s = 0; s < kL; ++s) {
            const int32_t e = se[s];
            if (e < 0) continue;
            if (moert_memmgr_promote(g.mm, L, e, s)) ++n_ok; else ++n_fail;
        }
    }
    g_slab_evicted       = false;
    loan_tables_build();   // findings 246: the slab moved; the hot pointer tables follow it
    std::cerr << "moe_dual: vision slab RESTORED: " << n_ok << " experts re-streamed"
              << (n_fail ? (" (" + std::to_string(n_fail) + " FAILED — those slots are zeroed)") : std::string())
              << " in " << (ggml_time_us() - t0) / 1000 << " ms\n";
    {
        long long fmib = 0, tmib = 0;
        vram_now(&fmib, &tmib);
        llapis_phase("Hot slab restored (%lld MiB VRAM; %d experts re-streamed, %lld ms; %lld / %lld MiB free)",
                     (long long) (g.slab_total_bytes >> 20), n_ok,
                     (long long) ((ggml_time_us() - t0) / 1000), fmib, tmib);
    }
    return n_fail == 0;
}

// Stream all of expert `e`'s high-precision pages into pool `slot` of layer `il`:
// the memmgr reads them from its promotion source (expert store or legacy GGUF
// catalog) and uploads at region_off + slot*stride — the same bytes at the same
// slot addresses as the pre-M1 in-file stream_one path.
static void excl_before_sync_fill(int il, int e, int slot);   // exclusive arena (below)
bool stream_expert(int il, int e, int slot) {
    excl_before_sync_fill(il, e, slot);   // also waits for the engine: its file promotions share the memmgr read staging
    return moert_memmgr_promote(g.mm, il, e, slot) != 0;
}

static void excl_layer_restores(int il);  // exclusive arena: same duty, from the slot (below)

// Push layer `il`'s residency mirrors (read from the moert policy views) to the device tensors read
// by the graph: hotmap/resid (blend path) AND exp_slot/slot_exp/slot_state (fused op).
// Residency publication: one synchronous tensor_set per mirror, or -- with the batch transport (findings 216l) --
// DEFERRED: push_residency marks the layer, and before each decode ONE batched copy publishes every marked layer.
static bool g_pub_deferred = false;
static std::vector<char> g_pub_dirty;
static void mirror_set(ggml_tensor * t, const void * src, size_t n) { ggml_backend_tensor_set(t, src, 0, n); }
// before any decode: the published mirrors have landed
void publish_fence() {
#ifdef MOE_DUAL_HAVE_CUDA_MAP
    if (!g_pub_deferred || g_pub_dirty.empty()) return;
    static std::vector<void *> dst; static std::vector<const void *> src; static std::vector<uint64_t> sz;
    dst.clear(); src.clear(); sz.clear();
    auto add = [&](ggml_tensor * t, const void * s, size_t n) { if (t && t->data && s) { dst.push_back(t->data); src.push_back(s); sz.push_back(n); } };
    for (int il = 0; il < g.n_layers; ++il) {
        if (!g_pub_dirty[(size_t) il]) continue;
        g_pub_dirty[(size_t) il] = 0;
        add(g.hotmap[il], moert_policy_hotmap_view(g.pol, il), (size_t) g.n_expert * sizeof(int32_t));
        add(g.resid[il],  moert_policy_resid_view(g.pol, il),  (size_t) g.n_expert * sizeof(float));
        if (!g.es_t.empty() && g.es_t[il]) {
            add(g.es_t[il], moert_policy_exp_slot_view(g.pol, il),   (size_t) g.n_expert * sizeof(int32_t));
            add(g.se_t[il], moert_policy_slot_exp_view(g.pol, il),   (size_t) klayer(il) * sizeof(int32_t));
            add(g.ss_t[il], moert_policy_slot_state_view(g.pol, il), (size_t) klayer(il) * sizeof(int32_t));
        }
    }
    if (!dst.empty()) ggml_backend_cuda_moe_dual_publish_h2d((int) dst.size(), dst.data(), src.data(), sz.data());
#endif
}
void push_residency(int il) {
    if (g_pub_deferred && il >= 0 && il < (int) g_pub_dirty.size()) {   // deferred: published in one batch before the next decode
        g_pub_dirty[(size_t) il] = 1;
        excl_layer_restores(il);
        return;
    }
    mirror_set(g.hotmap[il], moert_policy_hotmap_view(g.pol, il), (size_t) g.n_expert * sizeof(int32_t));
    mirror_set(g.resid[il],  moert_policy_resid_view(g.pol, il),  (size_t) g.n_expert * sizeof(float));
    if (!g.es_t.empty() && g.es_t[il]) {
        mirror_set(g.es_t[il], moert_policy_exp_slot_view(g.pol, il),   (size_t) g.n_expert * sizeof(int32_t));
        mirror_set(g.se_t[il], moert_policy_slot_exp_view(g.pol, il),   (size_t) klayer(il) * sizeof(int32_t));
        mirror_set(g.ss_t[il], moert_policy_slot_state_view(g.pol, il), (size_t) klayer(il) * sizeof(int32_t));
    }
    // exclusive arena: every now-cold expert must have a resident arena page (restore queued here)
    excl_layer_restores(il);
}

// Native observation: pull the device routing counters into the policy's count/route accumulators
// (additive — the policy folds them at its refresh points with the right n). The D2H is synchronous
// but runs only at refresh points / stats snapshots (~1 per 64 visible tokens), after llama_decode
// returned. No-op if the device counters could not be allocated.
// MTP forecaster (findings 216k, opt-in MOE_DUAL_MTP_NN=<weights>): the policy ranks and prices on its forecast
// f = H * P(used) instead of its rate EMA. It sees every fetched count; a decode step's counts close at the per-token
// decision, a prompt's at the prompt-end refresh, and the forecast is recomputed right there (inside the policy's
// fetch callback, before it pairs).
static std::vector<float> g_nn_f;
static int g_nn_phase = 0;   // 0 none, 1 a decode step just ended, 2 a prompt just ended
static struct NnStats { long long steps = 0, prompts = 0; double ms = 0.0, ms_max = 0.0;
    ~NnStats() { if (steps + prompts) std::fprintf(stderr, "[MOE_MTP_NN] %lld steps, %lld prompts, forecast %.3f ms mean, %.2f ms max\n",
                                                  steps, prompts, ms / (double) (steps + prompts), ms_max); } } g_nn_stats;

// Routing trace (bench, findings 213, MOE_DUAL_ROUTE_TRACE=<file>): the same counts, one record per decode
// step and per prompt, so the residency simulator replays what the product really routed (verify positions
// included). File: "LRT1" i32 n_layers i32 n_expert, then records i32 phase (1 step, 2 prompt) i32 nnz,
// u16 idx[nnz] (layer * n_expert + expert), i32 count[nnz]. Flushed per record: a client can cut turns by size.
static FILE * g_trace = nullptr;
static std::vector<int32_t> g_trace_pending;
static void route_trace_init() {
    const char * path = std::getenv("MOE_DUAL_ROUTE_TRACE");
    if (!path || !*path || !g.dev_counts) return;
    g_trace = std::fopen(path, "wb");
    if (!g_trace) { std::fprintf(stderr, "moe_dual:   route trace: cannot open %s\n", path); return; }
    const int32_t hdr[2] = { g.n_layers, g.n_expert };
    std::fwrite("LRT1", 1, 4, g_trace); std::fwrite(hdr, 4, 2, g_trace); std::fflush(g_trace);
    g_trace_pending.assign((size_t) g.n_layers * g.n_expert, 0);
    std::fprintf(stderr, "moe_dual:   route trace ON: %s\n", path);
}
static void route_trace_observe(const int32_t * counts, int phase) {
    if (!g_trace) return;
    for (size_t i = 0; i < g_trace_pending.size(); ++i) g_trace_pending[i] += counts[i];
    if (!phase) return;
    static std::vector<uint16_t> idx; static std::vector<int32_t> cnt;
    idx.clear(); cnt.clear();
    for (size_t i = 0; i < g_trace_pending.size(); ++i)
        if (g_trace_pending[i]) { idx.push_back((uint16_t) i); cnt.push_back(g_trace_pending[i]); }
    const int32_t rh[2] = { phase, (int32_t) idx.size() };
    std::fwrite(rh, 4, 2, g_trace);
    std::fwrite(idx.data(), 2, idx.size(), g_trace); std::fwrite(cnt.data(), 4, cnt.size(), g_trace);
    std::fflush(g_trace);
    std::fill(g_trace_pending.begin(), g_trace_pending.end(), 0);
}

// Batch transport setup (findings 216k-216l, opt-in MOE_DUAL_RAM_DIR=1): the cold arena is registered so the
// promotion copies are pure DMA; residency publication is deferred to one batch per decode.
static struct DirExit { ~DirExit() {
#ifdef MOE_DUAL_HAVE_CUDA_MAP
    if (!g_dir_on) return;
    moert_cuda_dir_stats s = {};
    ggml_backend_cuda_moe_dual_dir_stats(&s);
    std::fprintf(stderr, "[MOE_RAM_DIR] copies=%lld (%.1f MiB) completion marks=%lld | publication: %lld calls, %lld copies\n",
                 s.async_copies, s.bytes / 1048576.0, s.async_marks, s.pub_calls, s.pub_copies);
    if (const int64_t ec = ggml_backend_cuda_moe_dual_dir_gate(6, 0))
        std::fprintf(stderr, "[MOE_RAM_DIR] engine source: %lld runs on the copy engine, %lld pacing waits (%.1f ms on the worker)\n",
                     (long long) ec, (long long) ggml_backend_cuda_moe_dual_dir_gate(6, 1), ggml_backend_cuda_moe_dual_dir_gate(6, 2) / 1000.0);
    if (const int64_t gw = ggml_backend_cuda_moe_dual_dir_gate(4, 0))
        std::fprintf(stderr, "[MOE_RAM_DIR] early-decision gate: %lld gated waits, %lld timeouts, %.1f ms waited on the worker\n",
                     (long long) gw, (long long) ggml_backend_cuda_moe_dual_dir_gate(4, 1), ggml_backend_cuda_moe_dual_dir_gate(4, 2) / 1000.0);
    if (s.sm_copies || s.sm_fallback)
        std::fprintf(stderr, "[MOE_RAM_DIR] SM copy: %lld kernel copies, %lld on the copy engine (no device alias) | GPU time per "
                             "copy %.3f ms mean, %.3f ms max (%lld timed), %.1f ms total\n",
                     s.sm_copies, s.sm_fallback, s.sm_timed ? s.sm_gpu_ns / 1e6 / (double) s.sm_timed : 0.0,
                     s.sm_gpu_ns_max / 1e6, s.sm_timed, s.sm_gpu_ns / 1e6);
    if (s.sm_copies)
        std::fprintf(stderr, "[MOE_RAM_DIR] SM ring: %lld of the copies through the ring (the rest read in place), worker copy into the ring %.1f ms total (%.3f ms per copy), %lld waits for a free span (%.1f ms)\n",
                     s.sm_ring_copies, s.sm_stage_ms, s.sm_ring_copies ? s.sm_stage_ms / (double) s.sm_ring_copies : 0.0, s.sm_ring_waits, s.sm_ring_wait_ms);
#endif
} } g_dir_exit;
// SM transport, direct source (findings 233): the M most used experts of each layer (MOE_DUAL_DIR_SM_PIN=<order file>,
// "LPOP" i32 L i32 E i32 M, u16 ids[L][M] by decreasing use; MOE_DUAL_DIR_SM_PIN_M, default 64) are registered MAPPED, so
// their copies are read by the kernel in place; the others go through the ring. Contiguous experts form one range; a
// range is widened to whole pages and merged with a neighbour it touches.
static void dir_register_popular() {
    const char * pf = std::getenv("MOE_DUAL_DIR_SM_PIN");
    if (!pf || !*pf) return;
    FILE * fp = std::fopen(pf, "rb");
    int32_t hdr[3] = {};
    char tag[4] = {};
    if (!fp || std::fread(tag, 1, 4, fp) != 4 || std::memcmp(tag, "LPOP", 4) || std::fread(hdr, 4, 3, fp) != 3 ||
        hdr[0] != g.n_layers || hdr[1] != g.n_expert) {
        std::fprintf(stderr, "moe_dual:   SM direct source OFF: cannot read %s\n", pf);
        if (fp) std::fclose(fp);
        return;
    }
    std::vector<uint16_t> ord((size_t) hdr[0] * hdr[2]);
    const bool ok_read = std::fread(ord.data(), 2, ord.size(), fp) == ord.size();
    std::fclose(fp);
    if (!ok_read) { std::fprintf(stderr, "moe_dual:   SM direct source OFF: short file %s\n", pf); return; }
    const char * mv = std::getenv("MOE_DUAL_DIR_SM_PIN_M");
    const int M = std::max(0, std::min(hdr[2], mv ? std::atoi(mv) : 64));
    const uintptr_t pg = 4096;
    const auto t0 = std::chrono::steady_clock::now();
    long long calls = 0, ok = 0; double mib = 0.0;
    for (int il = 0; il < g.n_layers; ++il) {
        std::vector<int> ids(ord.begin() + (size_t) il * hdr[2], ord.begin() + (size_t) il * hdr[2] + M);
        std::sort(ids.begin(), ids.end());
        for (int kind = 0; kind < 2; ++kind) {
            moert_cold_arena_view aw;
            if (!moert_memmgr_get_cold_arena(g.mm, il, kind, 0, &aw)) continue;
            const uint64_t eb = aw.arena_stride ? aw.arena_stride : aw.stride;
            const uintptr_t base = (uintptr_t) aw.base;
            std::vector<std::pair<uintptr_t, uintptr_t>> runs;
            for (int e : ids) {
                const uintptr_t lo = (base + (uintptr_t) e * eb) & ~(pg - 1);
                const uintptr_t hi = (base + (uintptr_t) (e + 1) * eb + pg - 1) & ~(pg - 1);
                if (!runs.empty() && lo <= runs.back().second) runs.back().second = std::max(runs.back().second, hi);
                else runs.push_back({ lo, hi });
            }
            for (const auto & r : runs) {
                ++calls;
                if (ggml_backend_cuda_host_register_mapped((void *) r.first, (size_t) (r.second - r.first))) {
                    ++ok; mib += (double) (r.second - r.first) / 1048576.0;
                }
            }
        }
    }
    std::fprintf(stderr, "moe_dual:   SM direct source: top %d experts per layer registered mapped -- %lld of %lld ranges, %.0f MiB, %.0f ms\n",
                 M, ok, calls, mib, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
}

static void dir_init() {
    const char * v = std::getenv("MOE_DUAL_RAM_DIR");
    if (!v || !v[0] || v[0] == '0') return;
    if (!g.cold_arena_on || !g.mm) { std::fprintf(stderr, "moe_dual:   batch transport OFF: no cold arena\n"); return; }
#ifdef MOE_DUAL_HAVE_CUDA_MAP
    // MOE_DUAL_DIR_SM=<blocks> (findings 231): the copies are a kernel with that many blocks, not the copy engine, from a
    // pinned ring of MOE_DUAL_DIR_SM_RING_MIB (default 64) -- the arena is NOT registered: mapped, its 7 GiB fill WDDM's
    // shared-memory budget and the boot dies.
    const char * smv = std::getenv("MOE_DUAL_DIR_SM");
    const int sm_blocks = smv ? std::max(0, std::atoi(smv)) : 0;
    const char * rmv = std::getenv("MOE_DUAL_DIR_SM_RING_MIB");
    const int sm_ring_mib = rmv ? std::max(8, std::atoi(rmv)) : 64;
    const char * tmv = std::getenv("MOE_DUAL_DIR_SM_THREADS");   // threads filling the ring (findings 233), default 1
    const int sm_stage_threads = tmv ? std::max(1, std::atoi(tmv)) : 1;
    const auto t0 = std::chrono::steady_clock::now();
    int ok = 0;
    double reg_mib = 0.0;
    // MOE_DUAL_DIR_ENGINE=<pace> (findings 240, with the SM transport): the arena is registered (not mapped) and its runs go
    // on the copy engine, at most <pace> in its FIFO; the ring and the kernel serve what the registration left out
    const char * ev = std::getenv("MOE_DUAL_DIR_ENGINE");
    const int engine_pace = (sm_blocks && ev) ? std::max(0, std::atoi(ev)) : 0;
    // the transport's own pinned buffers first (the ring): the arena registration can exhaust the pinned budget
    if (engine_pace && !ggml_backend_cuda_moe_dual_dir_init(sm_blocks, sm_ring_mib, sm_stage_threads)) return;
    const char * emv = std::getenv("MOE_DUAL_DIR_ENGINE_MIB");   // registration cap with the engine source (default 6 GiB)
    const double reg_cap = engine_pace ? (emv ? std::atof(emv) : 6144.0) : 1e12;
    for (int il = 0; il < g.n_layers && (!sm_blocks || engine_pace); ++il)
        for (int kind = 0; kind < 2; ++kind) {
            moert_cold_arena_view aw;
            if (!moert_memmgr_get_cold_arena(g.mm, il, kind, 0, &aw)) { std::fprintf(stderr, "moe_dual:   batch transport OFF: no cold arena view\n"); return; }
            const uint64_t eb = aw.arena_stride ? aw.arena_stride : aw.stride;
            if (reg_mib + (double) eb * (double) g.n_expert / 1048576.0 > reg_cap) continue;
            const int r = ggml_backend_cuda_host_register_pinned((void *) aw.base, (size_t) eb * (size_t) g.n_expert);
            ok += r;
            if (r) reg_mib += (double) eb * (double) g.n_expert / 1048576.0;
        }
    if (sm_blocks) dir_register_popular();
    if (!sm_blocks || engine_pace)
        std::fprintf(stderr, "moe_dual:   batch transport: %d of %d cold arena regions registered (%.0f MiB) in %.0f ms\n",
                     ok, g.n_layers * 2, reg_mib, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    if (!engine_pace && !ggml_backend_cuda_moe_dual_dir_init(sm_blocks, sm_ring_mib, sm_stage_threads)) return;
    if (engine_pace) {
        ggml_backend_cuda_moe_dual_dir_gate(5, engine_pace);
        std::fprintf(stderr, "moe_dual:   engine source ON: registered runs on the copy engine, at most %d in its FIFO\n", engine_pace);
    }
    g_dir_on = true;
    g_pub_deferred = true;
    g_pub_dirty.assign((size_t) g.n_layers, 0);
    // promotions are done when their copies landed, polled at each policy read of the completions
    moe_xfer::set_async_completion([]() -> int64_t { return ggml_backend_cuda_moe_dual_dir_mark(); },
                                   [](int32_t w) -> int64_t { return ggml_backend_cuda_moe_dual_dir_poll(w); });
#endif
}

// MTP forecaster input: the drafts' h_nextn arrive from the drafting loop; at the pre-decode hook the drafts this
// verify will use become the forecaster's input (zeros on a turn's first verify).
static bool g_mtp_turn_start = true;
void mtp_draft(int i, const float * h, int n) { mtp_forecast::set_draft(i, h, n); }

// Decision trace (bench, claude dump findings 232, MOE_DUAL_DECISION_TRACE=<file>): what the live policy saw and did, step
// by step, to split every cold read into "not forecast" / "forecast, not promoted" / "promoted, not ready in time".
// File "LDT1" i32 n_layers i32 n_expert, then records (u8 tag):
//   'R' before each decode verify: u8 ready[n_layers*n_expert] (1 = hot and READY), i32 promotions still in flight;
//   'C' at each decision: u8 phase (1 decode step, 2 prompt end), i32 nnz, u16 idx[nnz], i32 count[nnz] (routes since
//       the previous decision, layer * n_expert + expert);
//   'D' after each decision: u8 phase, u8 has_forecast, [f32 f[n_layers*n_expert]], i32 n, then n x (u16 il, u16 e)
//       promotions submitted. Flushed per decode. Off = one branch per hook.
static FILE * g_dt = nullptr;
static std::vector<int32_t> g_dt_pending;
static std::vector<uint16_t> g_dt_promos;
static void dtrace_init() {
    const char * p = std::getenv("MOE_DUAL_DECISION_TRACE");
    if (!p || !*p || !g.pol) return;
    g_dt = std::fopen(p, "wb");
    if (!g_dt) { std::fprintf(stderr, "moe_dual:   decision trace: cannot open %s\n", p); return; }
    const int32_t hdr[2] = { g.n_layers, g.n_expert };
    std::fwrite("LDT1", 1, 4, g_dt); std::fwrite(hdr, 4, 2, g_dt); std::fflush(g_dt);
    g_dt_pending.assign((size_t) g.n_layers * g.n_expert, 0);
    std::fprintf(stderr, "moe_dual:   decision trace ON: %s\n", p);
}
static void dtrace_counts(const int32_t * counts, int phase) {
    if (!g_dt) return;
    for (size_t i = 0; i < g_dt_pending.size(); ++i) g_dt_pending[i] += counts[i];
    if (!phase) return;
    static std::vector<uint16_t> idx; static std::vector<int32_t> cnt;
    idx.clear(); cnt.clear();
    for (size_t i = 0; i < g_dt_pending.size(); ++i) if (g_dt_pending[i]) { idx.push_back((uint16_t) i); cnt.push_back(g_dt_pending[i]); }
    const uint8_t h[2] = { 'C', (uint8_t) phase };
    const int32_t n = (int32_t) idx.size();
    std::fwrite(h, 1, 2, g_dt); std::fwrite(&n, 4, 1, g_dt);
    std::fwrite(idx.data(), 2, idx.size(), g_dt); std::fwrite(cnt.data(), 4, cnt.size(), g_dt);
    std::fill(g_dt_pending.begin(), g_dt_pending.end(), 0);
}
static void dtrace_decision(int phase) {
    if (!g_dt) return;
    const uint8_t has_f = g_nn_f.empty() ? 0 : 1;
    const uint8_t h[3] = { 'D', (uint8_t) phase, has_f };
    std::fwrite(h, 1, 3, g_dt);
    if (has_f) std::fwrite(g_nn_f.data(), 4, g_nn_f.size(), g_dt);
    const int32_t n = (int32_t) (g_dt_promos.size() / 2);
    std::fwrite(&n, 4, 1, g_dt); std::fwrite(g_dt_promos.data(), 2, g_dt_promos.size(), g_dt);
    g_dt_promos.clear();
}
static void dtrace_ready() {
    if (!g_dt) return;
    static std::vector<uint8_t> ready;
    ready.assign((size_t) g.n_layers * g.n_expert, 0);
    for (int il = 0; il < g.n_layers; ++il) {
        const int32_t * es = moert_policy_exp_slot_view(g.pol, il);
        const int32_t * se = moert_policy_slot_exp_view(g.pol, il);
        const int32_t * ss = moert_policy_slot_state_view(g.pol, il);
        if (!es || !se || !ss) continue;
        for (int e = 0; e < g.n_expert; ++e) {
            const int s = es[e];
            ready[(size_t) il * g.n_expert + e] = s >= 0 && s < klayer(il) && ss[s] == SLOT_READY && se[s] == e;
        }
    }
    const int32_t late = moe_xfer::running() ? (int32_t) moe_xfer::promotions_in_flight() : 0;
    std::fputc('R', g_dt); std::fwrite(ready.data(), 1, ready.size(), g_dt); std::fwrite(&late, 4, 1, g_dt);
    std::fflush(g_dt);
}

// Early decision (claude dump findings 234, MOE_DUAL_EARLY_DECIDE=1, needs MOE_DUAL_RAM_DIR and MOE_DUAL_EARLY_READY): the
// per-step decision runs DURING the next verify instead of after it, so its copies are ready to start the moment the GPU
// leaves the verify's last MoE layer (the gate in moert/cuda), not 3-4 ms into the window between verifies.
//   after verify t (hook thread, on_token_end): only fetch the routing counts of t into g_pref, open the gate from the host;
//   before verify t+1 (server thread, after the publication): post the decision to the early thread -- counts of t, the
//     drafts of t+1 for the forecaster -- and arm the gate: its copies wait for verify t+1's last MoE layer;
//   every other policy entry point joins the early thread first.
static bool g_early_decide = false;
static bool g_in_early = false;                 // the early thread is inside the policy call
static std::vector<int32_t> g_pref; static bool g_pref_have = false;
static struct EarlyThread {
    std::thread th; std::mutex mu; std::condition_variable cv;
    bool pending = false, running = false, quit = false;
    std::function<void()> fn;
    void post(std::function<void()> f) {
        std::unique_lock<std::mutex> lk(mu);
        cv.wait(lk, [&] { return !pending && !running; });
        if (!th.joinable()) th = std::thread([this] { loop(); });
        fn = std::move(f); pending = true;
        cv.notify_all();
    }
    void join() {
        std::unique_lock<std::mutex> lk(mu);
        cv.wait(lk, [&] { return !pending && !running; });
    }
    void loop() {
        std::unique_lock<std::mutex> lk(mu);
        for (;;) {
            cv.wait(lk, [&] { return pending || quit; });
            if (quit) return;
            std::function<void()> f = std::move(fn); pending = false; running = true;
            lk.unlock(); f(); lk.lock();
            running = false; cv.notify_all();
        }
    }
    ~EarlyThread() {
        if (!th.joinable()) return;
        { std::lock_guard<std::mutex> lk(mu); quit = true; } cv.notify_all(); th.join();
    }
} g_early_th;
void early_join() { if (g_early_decide) g_early_th.join(); }
static void gate_wait_cb(int32_t il, int64_t target) {
#ifdef MOE_DUAL_HAVE_CUDA_MAP
    ggml_backend_cuda_moe_dual_dir_gate(2, ((int64_t) il << 40) | target);
#else
    (void) il; (void) target;
#endif
}
static void early_init() {
    // 1: copies wait for the verify's last MoE layer; 2: for their own layer (the pauses between layers, findings 235)
    const char * v = std::getenv("MOE_DUAL_EARLY_DECIDE");
    if (!v || (v[0] != '1' && v[0] != '2')) return;
    const bool per_layer = v[0] == '2';
#ifdef MOE_DUAL_HAVE_CUDA_MAP
    if (!g_dir_on || !g.dev_counts) { std::fprintf(stderr, "moe_dual:   early decision OFF: needs MOE_DUAL_RAM_DIR\n"); return; }
    ggml_backend_cuda_moe_dual_dir_gate(0, (int64_t) (g.n_layers - 1) | ((int64_t) per_layer << 16));
    moe_xfer::set_gate_wait(&gate_wait_cb);
    g_pref.assign((size_t) g.n_layers * g.n_expert + (size_t) g.n_layers * 2, 0);
    g_early_decide = true;
    std::fprintf(stderr, "moe_dual:   early decision ON: decided during the verify, copies gated on %s\n",
                 per_layer ? "their own layer (they run in the pauses between layers)" : "its last MoE layer");
#endif
}

// Live ceiling oracle (claude dump findings 255, opt-in MOE_DUAL_ORACLE=<route trace>): a bench measurement, not a
// product feature. The route trace of a previous greedy run of the SAME conversations (MOE_DUAL_ROUTE_TRACE) tells the
// policy what the next decode steps of this response really use: f = uses over the next H non-empty steps of the
// response (MOE_DUAL_ORACLE_H, default 5), priced at MOE_DUAL_ORACLE_R (default 1 step use) and capped at
// MOE_DUAL_ORACLE_CAP promotions per decision (default 4, the default rule's cap). Same transport as the default: only
// the forecast changes. The cursor re-aligns on every prompt (prompt k <-> recorded prompt k); per step the live routes
// are compared with the recorded ones (Jaccard), so a run whose text drifts from the recording shows it at exit.
namespace oracle {
static bool on = false;
static int H = 5;
static std::vector<std::vector<uint32_t>> steps;   // per non-empty recorded decode step: used (il * n_expert + e)
static std::vector<long long> prompt_at;           // first step index of each recorded response
static long long cur = -1, resp_end = 0;
static size_t next_prompt = 0;
static struct Stats { long long live = 0, inrange = 0, past = 0, prompts = 0, prompts_over = 0; double jac = 0.0;
    ~Stats() { if (on) std::fprintf(stderr, "[MOE_ORACLE] %lld decode steps (%lld inside the recorded response, mean Jaccard "
                                            "live vs recorded %.3f; %lld past its end) | %lld prompts (%lld past the recording)\n",
                                    live, inrange, inrange ? jac / (double) inrange : 0.0, past, prompts, prompts_over); } } st;

static bool load(const char * path, int nl, int ne) {
    FILE * f = std::fopen(path, "rb");
    char tag[4] = {}; int32_t hdr[2] = {};
    if (!f || std::fread(tag, 1, 4, f) != 4 || std::memcmp(tag, "LRT1", 4) || std::fread(hdr, 4, 2, f) != 2 ||
        hdr[0] != nl || hdr[1] != ne) {
        std::fprintf(stderr, "moe_dual:   oracle OFF: cannot read %s\n", path);
        if (f) std::fclose(f);
        return false;
    }
    int32_t rh[2];
    std::vector<uint16_t> idx; std::vector<int32_t> cnt;
    while (std::fread(rh, 4, 2, f) == 2) {
        idx.resize((size_t) std::max(0, rh[1])); cnt.resize(idx.size());
        if (std::fread(idx.data(), 2, idx.size(), f) != idx.size() || std::fread(cnt.data(), 4, cnt.size(), f) != cnt.size()) break;
        if (rh[0] == 2) prompt_at.push_back((long long) steps.size());
        else if (rh[0] == 1 && !idx.empty()) steps.emplace_back(idx.begin(), idx.end());
    }
    std::fclose(f);
    std::fprintf(stderr, "moe_dual:   oracle ON: %zu recorded decode steps, %zu prompts from %s\n", steps.size(), prompt_at.size(), path);
    return !steps.empty();
}

static void forecast(float * f, size_t n) {
    std::fill(f, f + n, 0.0f);
    for (long long k = cur + 1; k <= cur + H && k < resp_end && k < (long long) steps.size(); ++k)
        for (uint32_t i : steps[(size_t) k]) if (i < n) f[i] += 1.0f;
}

static void observe(const int32_t * counts, int phase, float * f, size_t n) {
    if (phase == 2) {
        ++st.prompts;
        if (next_prompt < prompt_at.size()) {
            cur = prompt_at[next_prompt] - 1;
            resp_end = next_prompt + 1 < prompt_at.size() ? prompt_at[next_prompt + 1] : (long long) steps.size();
            ++next_prompt;
        } else { ++st.prompts_over; cur = resp_end = (long long) steps.size(); }
        forecast(f, n);
        return;
    }
    if (phase != 1) return;
    bool any = false;
    for (size_t i = 0; i < n && !any; ++i) any = counts[i] != 0;
    if (!any) return;   // a decision with no new routes (same verify): keep the forecast
    ++st.live; ++cur;
    if (cur < resp_end && cur < (long long) steps.size()) {
        const auto & rec = steps[(size_t) cur];
        long long inter = 0, live_n = 0;
        for (size_t i = 0; i < n; ++i) live_n += counts[i] != 0;
        for (uint32_t i : rec) inter += i < n && counts[i] != 0;
        const long long uni = live_n + (long long) rec.size() - inter;
        st.jac += uni ? (double) inter / (double) uni : 1.0; ++st.inrange;
    } else ++st.past;
    forecast(f, n);
}
}  // namespace oracle

void before_decode() {
    if (mtp_forecast::active()) { mtp_forecast::snapshot(g_mtp_turn_start); g_mtp_turn_start = false; }
    dtrace_ready();
}

void forecast_init() {
    route_trace_init();
    dtrace_init();
    dir_init();
    early_init();
    if (const char * op = std::getenv("MOE_DUAL_ORACLE"); op && *op && g.pol && g.dev_counts &&
        oracle::load(op, g.n_layers, g.n_expert)) {
        const char * h = std::getenv("MOE_DUAL_ORACLE_H");
        const char * r = std::getenv("MOE_DUAL_ORACLE_R");
        const char * cap = std::getenv("MOE_DUAL_ORACLE_CAP");
        oracle::H = h ? std::max(1, std::atoi(h)) : 5;
        const float rr = r ? (float) std::atof(r) : 1.0f;
        const int   cc = cap ? std::atoi(cap) : 4;
        g_nn_f.assign((size_t) g.n_layers * g.n_expert, 0.0f);
        moert_policy_set_external_forecast(g.pol, g_nn_f.data(), rr, cc);
        oracle::on = true;
        std::fprintf(stderr, "moe_dual:   oracle forecast ON: next %d steps, price %.2f, at most %d promotions per decision\n",
                     oracle::H, rr, cc);
        return;
    }
    const char * mp = std::getenv("MOE_DUAL_MTP_NN");
    if (!mp || !*mp || !g.pol || !g.dev_counts || !mtp_forecast::load(mp, g.n_layers, g.n_expert)) return;
    const char * r = std::getenv("MOE_DUAL_MTP_NN_R");
    const char * cap = std::getenv("MOE_DUAL_MTP_NN_CAP");
    const float rr = r ? (float) std::atof(r) : 0.25f;
    const int   cc = cap ? std::atoi(cap) : 20;
    g_nn_f.assign((size_t) g.n_layers * g.n_expert, 0.0f);
    moert_policy_set_external_forecast(g.pol, g_nn_f.data(), rr, cc);
    std::fprintf(stderr, "moe_dual:   MTP forecaster ON: price %.2f, at most %d promotions per step\n", rr, cc);
}

// A streamed prefill (q4stream) routes outside the device counters: its routes are counted on the host for P3
// (bnd_cnt). At the prompt-end decision they join the fetched counts, once (findings 213: before this the
// forecaster's prompt features and the trace saw zero for every streamed prompt).
static const int32_t * with_prompt_routes(const int32_t * counts) {
    if (g_nn_phase != 2 || g.bnd_cnt.empty() || (!g_trace && !g_dt && !mtp_forecast::active())) return counts;
    long long n = 0;
    for (long long x : g.bnd_ntok) n += x;
    if (!n) return counts;
    static std::vector<int32_t> sum;
    sum.assign(counts, counts + (size_t) g.n_layers * g.n_expert);
    for (int il = 0; il < g.n_layers && il < (int) g.bnd_cnt.size(); ++il)
        for (int e = 0; e < g.n_expert; ++e) sum[(size_t) il * g.n_expert + e] += g.bnd_cnt[(size_t) il][(size_t) e];
    for (auto & c : g.bnd_cnt) std::fill(c.begin(), c.end(), 0);
    std::fill(g.bnd_ntok.begin(), g.bnd_ntok.end(), 0);
    return sum.data();
}

static void routes_observe(const int32_t * counts_in) {
    const int32_t * counts = with_prompt_routes(counts_in);
    route_trace_observe(counts, g_nn_phase);
    dtrace_counts(counts, g_nn_phase);
    if (oracle::on) {
        oracle::observe(counts, g_nn_phase, g_nn_f.data(), g_nn_f.size());
        g_nn_phase = 0;
        return;
    }
    if (!mtp_forecast::active()) { g_nn_phase = 0; return; }
    mtp_forecast::add_counts(counts);
    if (!g_nn_phase) return;
    if (g_nn_phase == 1) { mtp_forecast::close_step(); ++g_nn_stats.steps; }
    else                 { mtp_forecast::close_prompt(); ++g_nn_stats.prompts; g_mtp_turn_start = true; }
    g_nn_phase = 0;
    const double ms = mtp_forecast::forecast(g_nn_f.data());
    g_nn_stats.ms += ms; g_nn_stats.ms_max = std::max(g_nn_stats.ms_max, ms);
}

static void fetch_dev_counts() {
#ifdef MOE_DUAL_HAVE_CUDA_MAP
    if (!g.dev_counts || !g.pol) return;
    static std::vector<int32_t> buf;
    buf.resize((size_t) g.n_layers * g.n_expert + (size_t) g.n_layers * 2);
    if (g_in_early) {   // early decision: the counts fetched after the previous verify; no device read during a verify
        if (!g_pref_have) return;
        buf = g_pref; g_pref_have = false;
    } else if (!ggml_backend_cuda_moe_dual_route_counts_fetch(buf.data(), /*reset=*/1)) return;
    const int32_t * hits = buf.data() + (size_t) g.n_layers * g.n_expert;
    for (int il = 0; il < g.n_layers; ++il)
        moert_policy_add_external_counts(g.pol, il, buf.data() + (size_t) il * g.n_expert,
                                         hits[il * 2 + 0], hits[il * 2 + 1]);
    routes_observe(buf.data());
#endif
}

// --- moert policy callbacks: the policy decides WHAT to promote/evict/publish; these do the work. ---
// --- asynchronous promotion (MOE_DUAL_ASYNC_PROMOTE, default on) -----------------------------------
// Decided once at the first promotion. The source of each copy comes from the registry: the RAM arena
// when it holds the hot bytes (speed), else the promotion file (quality, design P5). The policy marks
// the slot FILLING (cb returns 2); the job stays PENDING until the policy call that made it returns --
// the layer is then published with the slot not READY -- and is flushed to the worker (moe_xfer.hpp).
// (history: docs/code-comments-archive-2026-10-05.md, "async-promotion")
static int g_async_state = -1;   // -1 undecided, 0 off, 1 on
static bool async_promote_on() {
    if (g_async_state >= 0) return g_async_state == 1;
    g_async_state = 0;
    const char * v = std::getenv("MOE_DUAL_ASYNC_PROMOTE");
    if (!(v && v[0] == '1')) {
        std::cerr << "moe_dual: WARNING: the per-token cost-aware rule is priced for "
                     "asynchronous promotion, which is off -- expect decode pauses\n";
        return false;
    }
    // The source of each promotion comes from the registry (cb_stream_expert): the RAM arena when it holds
    // the hot bytes (full IQ2), otherwise the promotion file (the Q4 pages of `quality`, design P5).
    const bool from_arena = g.cold_arena_on && moert_memmgr_arena_matches_hot(g.mm);
    if (!g.mm) {
        std::cerr << "moe_dual: async promotion REFUSED -- no memory manager" << "\n";
    } else if (!moe_xfer::start(g.mm)) {
        std::cerr << "moe_dual: async promotion REFUSED -- the transfer worker did not start" << "\n";
    } else {
        g_async_state = 1;
        std::cerr << "moe_dual: async promotion ACTIVE -- promotions upload from the "
                  << (from_arena ? "RAM arena" : "promotion file (SSD)")
                  << " on a worker thread; slots serve cold (FILLING) until their copy lands" << "\n";
    }
    if (g_async_state != 1)
        std::cerr << "moe_dual: WARNING: the per-token cost-aware rule is priced for "
                     "asynchronous promotion, but promotions here are SYNCHRONOUS -- expect decode pauses\n";
    return g_async_state == 1;
}

// --- exclusive arena (MOE_DUAL_ARENA_EXCLUSIVE, full-IQ2 plan phase 2) ---------------------------
// Each expert lives in ONE place: a hot READY expert's arena page is released, and rebuilt from its
// slot when the expert goes cold. The engine (moe_xfer) runs both on its worker and owns the safety
// invariant (cold_guard). The host owns two maps and three duties:
//   phys[il][s] -- whose bytes slot s holds once every job submitted so far has run (-1 unknown);
//   rel[il*E+e] -- e's release was requested and no restore since;
//   (1) release every hot READY expert whose slot holds its bytes (excl_reconcile, after a policy step);
//   (2) queue the restore of a released expert BEFORE any job that overwrites its slot or reads its
//       page (excl_before_fill / excl_before_sync_fill), and as soon as it is published cold
//       (excl_layer_restores, from push_residency);
//   (3) drain -- restore everything and wait -- before the slab is freed or the image path reads the
//       whole arena (excl_drain).
// phys is rebuilt from the policy's READY slots whenever it was invalidated (drain) and the slab is
// live; while it is invalid nothing is released, so nothing can need a restore.
// SOFT (the arena is the mapped pack, findings 197): a release only trims the page from the working set
// and the data stays in the file, so duties (2)-(3) shrink to prefetches nobody waits for, phys is not
// needed, and the hot tier's format does not matter (quality frees RAM too).
static struct {
    bool on = false;
    bool soft = false;
    bool phys_valid = false;
    std::vector<std::vector<int32_t>> phys;
    std::vector<uint8_t> rel;
} gx;

static bool excl_on() {
    static int state = -1;   // -1 undecided
    if (state >= 0) return state == 1;
    if (!g.hcfg.arena_exclusive) { state = 0; return false; }
    if (!async_promote_on()) {
        state = 0;
        std::cerr << "moe_dual: exclusive arena REFUSED -- it needs asynchronous promotion "
                     "(MOE_DUAL_ASYNC_PROMOTE=1, full-IQ2) and the worker it starts\n";
        return false;
    }
    const bool soft = moert_memmgr_cold_arena_mapped(g.mm) != 0;
    if (!soft && !moert_memmgr_arena_matches_hot(g.mm)) {   // a released page would be the ONLY copy of those bytes
        state = 0;
        std::cerr << "moe_dual: exclusive arena REFUSED -- the hot tier is not the cold tier's quantization, "
                     "so an arena page is never a duplicate\n";
        return false;
    }
    if (!moe_xfer::exclusive_start(g.n_layers, g.n_expert, soft)) {
        state = 0;
        std::cerr << "moe_dual: exclusive arena REFUSED -- the transfer engine did not arm it\n";
        return false;
    }
    state = 1;
    gx.on = true;
    gx.soft = soft;
    gx.rel.assign((size_t) g.n_layers * g.n_expert, 0);
    gx.phys_valid = false;
    if (soft)
        std::cerr << "moe_dual: exclusive arena ACTIVE (mapped pack) -- hot experts' arena pages are trimmed "
                     "from RAM and faulted back in on a worker when they go cold; cold reads never wait\n";
    else
        std::cerr << "moe_dual: exclusive arena ACTIVE -- hot experts' arena pages are released and rebuilt "
                     "from their VRAM slot when they go cold (no SSD); cold reads wait on a pending rebuild\n";
    return true;
}

static bool excl_slab_live() { return g.slab_buf && g.slab_bytes_t && !slab_evicted(); }

static void excl_phys_init() {
    if (gx.phys_valid || !excl_slab_live()) return;
    gx.phys.assign((size_t) g.n_layers, {});
    for (int il = 0; il < g.n_layers; ++il) {
        const int K = klayer(il);
        const int32_t * se = moert_policy_slot_exp_view(g.pol, il);
        const int32_t * ss = moert_policy_slot_state_view(g.pol, il);
        gx.phys[(size_t) il].assign((size_t) K, -1);
        for (int s = 0; s < K; ++s)
            if (se && ss && ss[s] == MOERT_SLOT_READY) gx.phys[(size_t) il][(size_t) s] = se[s];
    }
    gx.phys_valid = true;
}

static int excl_phys_slot(int il, int e) {
    const auto & ph = gx.phys[(size_t) il];
    for (size_t s = 0; s < ph.size(); ++s) if (ph[s] == e) return (int) s;
    return -1;
}

// Queue the restore of released expert e of layer il from the slot holding its bytes.
static void excl_restore(int il, int e) {
    uint8_t & r = gx.rel[(size_t) il * g.n_expert + e];
    if (!r) return;
    r = 0;
    if (gx.soft) { moe_xfer::submit_restore(il, e, -1); return; }   // a prefetch: the page never left the file
    const int s = gx.phys_valid ? excl_phys_slot(il, e) : -1;
    if (s < 0) {   // cannot happen: a released expert's bytes stay in its slot until it is restored
        std::cerr << "moe_dual: CRITICAL: exclusive arena lost the slot of released l" << il << " e" << e << "\n";
        return;
    }
    moe_xfer::submit_restore(il, e, s);
}

// A promotion of e into (il, slot) is about to be queued: restore e's own page if released (the
// worker reads it) and the page of the expert whose bytes the slot still holds (it overwrites them).
static void excl_before_fill(int il, int e, int slot) {
    if (!excl_on() || gx.soft) return;   // soft: no page is ever lost, nothing to order
    excl_phys_init();
    if (!gx.phys_valid || slot >= (int) gx.phys[(size_t) il].size()) return;
    excl_restore(il, e);
    const int v = gx.phys[(size_t) il][(size_t) slot];
    if (v >= 0) excl_restore(il, v);
    gx.phys[(size_t) il][(size_t) slot] = e;
}

// A synchronous write (seeding) into (il, slot): the slot's previous expert gets its page back first,
// and no queued job may still touch the slot.
static void excl_before_sync_fill(int il, int e, int slot) {
    if (!gx.on || gx.soft) { moe_xfer::wait_idle(); return; }   // no-op unless the engine was started
    if (gx.phys_valid && slot < (int) gx.phys[(size_t) il].size()) {
        const int v = gx.phys[(size_t) il][(size_t) slot];
        if (v >= 0) excl_restore(il, v);
        gx.phys[(size_t) il][(size_t) slot] = e;
    }
    moe_xfer::wait_idle();
}

static void excl_layer_restores(int il) {
    if (!gx.on || (!gx.soft && !gx.phys_valid)) return;
    const int32_t * es = moert_policy_exp_slot_view(g.pol, il);
    const int32_t * ss = moert_policy_slot_state_view(g.pol, il);
    if (!es || !ss) return;
    for (int e = 0; e < g.n_expert; ++e) {
        if (!gx.rel[(size_t) il * g.n_expert + e]) continue;
        const int s = es[e];
        if (s < 0 || ss[s] != MOERT_SLOT_READY) excl_restore(il, e);
    }
}

// RAM-dedup / exclusive arena: set for an image turn (arena held whole), cleared at the turn end.
static bool g_arena_vision_hold = false;

// After a policy step, before its jobs are flushed: release every hot READY expert whose slot holds
// its bytes, and catch any released expert that went cold without a publish we saw.
static void excl_reconcile() {
    if (!excl_on() || !excl_slab_live() || g_arena_vision_hold) return;
    if (!gx.soft) excl_phys_init();
    if (!gx.soft && !gx.phys_valid) return;
    for (int il = 0; il < g.n_layers; ++il) {
        const int32_t * es = moert_policy_exp_slot_view(g.pol, il);
        const int32_t * ss = moert_policy_slot_state_view(g.pol, il);
        if (!es || !ss) continue;
        const int K = klayer(il);
        for (int e = 0; e < g.n_expert; ++e) {
            uint8_t & r = gx.rel[(size_t) il * g.n_expert + e];
            const int s = es[e];
            const bool hot_ready = s >= 0 && s < K && ss[s] == MOERT_SLOT_READY;
            // copy mode releases only a page whose bytes the slot provably holds (phys); soft needs no proof
            const bool holds = gx.soft || (s < (int) gx.phys[(size_t) il].size() && gx.phys[(size_t) il][(size_t) s] == e);
            if (!r && hot_ready && holds) { moe_xfer::submit_release(il, e); r = 1; }
            else if (r && !hot_ready) excl_restore(il, e);
        }
    }
}

// Rebuild every released page from its slot and wait for it. Called before the slab is freed (any
// reason) and before the image path, which reads the whole arena on the CPU.
static void excl_drain(const char * why) {
    if (!gx.on) return;
    const auto t0 = std::chrono::steady_clock::now();
    int n = 0;
    for (int il = 0; il < g.n_layers; ++il)
        for (int e = 0; e < g.n_expert; ++e)
            if (gx.rel[(size_t) il * g.n_expert + e]) { excl_restore(il, e); ++n; }
    if (gx.soft) {   // the pages never left the file: the restores are prefetches, nobody waits on them
        moe_xfer::flush();
        return;
    }
    moe_xfer::wait_idle();
    gx.phys_valid = false;   // the slot contents are about to change outside the engine
    if (n) std::cerr << "moe_dual: exclusive arena drain (" << why << ") -- rebuilt " << n
                     << " pages from VRAM in " << std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::steady_clock::now() - t0).count() << " ms\n";
}

static void excl_log(const char * where) {
    if (!gx.on) return;
    const moe_xfer::Stats st = moe_xfer::stats();
    uint64_t rss = 0;
#if defined(_WIN32)
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) rss = pmc.WorkingSetSize;
#endif
    std::cerr << "moe_dual: exclusive arena (" << where << ") -- released " << (st.released_bytes >> 20)
              << " MiB | arena committed ~" << (moert_memmgr_arena_committed_bytes(g.mm) >> 20) << " / "
              << (moert_memmgr_cold_arena_bytes(g.mm) >> 20) << " MiB | releases " << st.releases
              << " restores " << st.restores << " guard waits " << st.guard_waits << " | RSS "
              << (rss >> 20) << " MiB\n";
}

// Apply finished transfers to the policy and publish the layers whose slots became READY (or were
// released). Called between decodes, before every policy step.
static int xfer_apply_completions() {
    if (g_async_state != 1) return 0;
    static std::vector<moe_xfer::Done> done;
    done.clear();
    if (!moe_xfer::take_done(done)) return 0;
    int n_ok = 0;
    std::vector<char> touched((size_t) g.n_layers, 0);
    for (const moe_xfer::Done & d : done) {
        const int32_t changed = d.ok ? moert_policy_complete_fill(g.pol, d.il, d.slot, d.e)
                                     : moert_policy_abort_fill(g.pol, d.il, d.slot, d.e);
        if (changed && d.il >= 0 && d.il < g.n_layers) touched[(size_t) d.il] = 1;
        if (changed && d.ok) ++n_ok;
    }
    for (int il = 0; il < g.n_layers; ++il) if (touched[(size_t) il]) push_residency(il);
    return n_ok;
}

int32_t cb_stream_expert(void * /*user*/, int32_t il, int32_t e, int32_t slot) {
    if (g.bnd_il == il) return q4stream_boundary_fill(il, e, slot);   // P3: copy from the prefill staging
    if (g_dt) { g_dt_promos.push_back((uint16_t) il); g_dt_promos.push_back((uint16_t) e); }
    if (async_promote_on()) {
        excl_before_fill(il, e, slot);   // exclusive arena: restores queued ahead of the copy
        // where the expert's bytes are: not in a slot (it is being promoted); its arena page is valid by
        // the time the job runs (a released page has its restore queued ahead, at higher priority)
        static const int32_t same = moert_memmgr_arena_matches_hot(g.mm);
        const moert_expert_where w = { -1, g.cold_arena_on ? 1 : 0, same, 1, 0 };
        moe_xfer::submit(il, e, slot, moert_source_for(&w, MOERT_REP_HOT));
        return 2;   // FILLING until the worker's copy completes
    }
    return stream_expert(il, e, slot) ? 1 : 0;
}
void cb_publish_layer(void * /*user*/, int32_t il) { push_residency(il); }

// P3: before the boundary decisions of a prompt, no asynchronous promotion may still be writing a slot
// (a boundary eviction of a FILLING slot would race the worker's copy into the same bytes): drain the
// engine and apply its completions.
void xfer_settle() {
    early_join();
    moe_xfer::wait_idle();
    xfer_apply_completions();
}
// [llapis] findings 210: the stats readers (/moe/telemetry and /moe/seam on the HTTP thread, the
// per-step heartbeat) must not pull the device counters: since the per-step hook runs on its own thread
// (findings 207) that pull raced the policy's own fetch-and-fold (a synchronous CUDA copy plus writes to
// the policy's count accumulators and the step forecaster, from two threads). A reader now sees the
// counters as of the last decision; the counts it would have pulled are folded at the next one.
static thread_local bool t_stats_reader = false;
struct StatsReader { StatsReader() { t_stats_reader = true; } ~StatsReader() { t_stats_reader = false; } };
void cb_fetch_external(void * /*user*/)            { if (!t_stats_reader) fetch_dev_counts(); }
// Diagnostic (route-trace, cache-aware top-k gate): would a routed (il, e) pair be served from the
// hot pool RIGHT NOW? Mirrors the fused-op partition predicate exactly (exp_slot/slot_exp match +
// slot READY). Dynamic cache only:
// returns false when no dynamic policy is live.
bool expert_is_hot(int il, int e) {
    if (!g.dynamic || !g.pol || il < 0 || il >= g.n_layers || e < 0 || e >= g.n_expert) return false;
    const int32_t * es = moert_policy_exp_slot_view(g.pol, il);
    const int32_t * se = moert_policy_slot_exp_view(g.pol, il);
    const int32_t * st = moert_policy_slot_state_view(g.pol, il);
    const int s = es[e];
    return s >= 0 && s < klayer(il) && st[s] == SLOT_READY && se[s] == e;
}

// PREFILL (prompt) progress (1 token, or a whole batched chunk): fold routing into the scores; NEVER
// promote (prefill only observes); the fold is decay-corrected for chunks.
void on_prompt_token(long long n_tokens) {
    if (!g.dynamic) return;
    early_join();
    moert_policy_on_prompt_token(g.pol, n_tokens);
}

// One VISIBLE (decode) token ended: fold, then the per-token decision.
// [llapis] findings 210: the copies of the promotions decided after step t run while the server samples and
// drafts step t+1; picking up the finished ones right before step t+1's verify makes them hot one step
// earlier. Runs on the server thread after the async hook was joined (no concurrent policy call).
static bool early_ready_on() {
    static const bool on = [] { const char * e = std::getenv("MOE_DUAL_EARLY_READY"); return e && e[0] == '1'; }();
    return on;
}
// findings 231: what is still travelling when the verify starts (it lands during the verify, hot one step later),
// and what the per-step decision costs on the hook thread.
static struct EarlyStats { long long calls = 0, applied = 0, late = 0, late_steps = 0, decisions = 0; double dec_ms = 0.0, dec_ms_max = 0.0;
    ~EarlyStats() {
        if (calls) std::fprintf(stderr, "[MOE_EARLY_READY] %lld pre-decode checks, %lld promotions made hot one step early | "
                                        "still in flight at the verify: %lld promotions over %lld steps (%.2f per check)\n",
                                calls, applied, late, late_steps, (double) late / (double) calls);
        if (decisions) std::fprintf(stderr, "[MOE_STEP_DECISION] %lld per-token decisions, %.3f ms mean, %.2f ms max (hook thread)\n",
                                    decisions, dec_ms / (double) decisions, dec_ms_max);
    } } g_early;
void after_publish_before_decode() {
#ifdef MOE_DUAL_HAVE_CUDA_MAP
    if (!g.dynamic || !g_early_decide) return;
    const int64_t target = ggml_backend_cuda_moe_dual_dir_gate(1, 0);
    g_early_th.post([target] {
        const auto t0 = std::chrono::steady_clock::now();
        moe_xfer::set_submit_gate(target);
        g_in_early = true;
        g_nn_phase = 1;
        moert_policy_on_token_end(g.pol);
        g_nn_phase = 0;
        g_in_early = false;
        dtrace_decision(1);
        excl_reconcile();
        moe_xfer::set_submit_gate(0);
        moe_xfer::flush();
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        ++g_early.decisions; g_early.dec_ms += ms; g_early.dec_ms_max = std::max(g_early.dec_ms_max, ms);
    });
#endif
}

void apply_ready_before_decode() {
    early_join();
    if (!g.dynamic || !early_ready_on() || g_async_state != 1) return;
    ++g_early.calls;
    g_early.applied += xfer_apply_completions();
    const long long late = moe_xfer::promotions_in_flight();
    g_early.late += late; g_early.late_steps += late > 0;
    excl_reconcile();
    moe_xfer::flush();
}

void on_token_end() {
    if (!g.dynamic) return;
#ifdef MOE_DUAL_HAVE_CUDA_MAP
    if (g_early_decide) {   // after the verify: fetch its counts for the decision taken during the next one
        g_early_th.join();
        if (g_pref_have) {   // a step whose decision never ran (no decode followed): fold them in
            std::vector<int32_t> keep = g_pref, now(g_pref.size());
            if (ggml_backend_cuda_moe_dual_route_counts_fetch(now.data(), 1))
                for (size_t i = 0; i < now.size(); ++i) g_pref[i] = keep[i] + now[i];
        } else if (ggml_backend_cuda_moe_dual_route_counts_fetch(g_pref.data(), 1)) {
            g_pref_have = true;
        }
        ggml_backend_cuda_moe_dual_dir_gate(3, 0);   // the verify is over: a gated copy may never wait past this
        return;
    }
#endif
    const auto t0 = std::chrono::steady_clock::now();
    xfer_apply_completions();
    g_nn_phase = 1;
    moert_policy_on_token_end(g.pol);
    g_nn_phase = 0;
    dtrace_decision(1);
    excl_reconcile();
    moe_xfer::flush();   // jobs of a refresh run only after its layers were published
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    ++g_early.decisions; g_early.dec_ms += ms; g_early.dec_ms_max = std::max(g_early.dec_ms_max, ms);
}

// The decision (the cost-aware rule) lives in moert/policy. Promotion streams + residency pushes come
// back through the callbacks above.
// Vision coexistence: the transient image decode runs its expert matmuls on the CPU straight from the cold
// IQ2 arena (op_offload=false, arena-aliased), so EVERY expert it routes must be resident. Under the
// exclusive arena the hot experts' pages are released: rebuild them all from VRAM and hold until the turn
// ends. Called from the vision-room callback's need_room branch, before the image graph runs.
void arena_vision_full_commit() {
#ifdef MOE_DUAL_HAVE_CUDA_MAP
    if (gx.on) {   // exclusive arena: rebuild every page from VRAM, hold until the turn ends
        excl_drain("image turn");
        g_arena_vision_hold = true;
        return;
    }
#endif
}

void refresh_cache(int reason) {
    if (!g.dynamic) return;
    early_join();
    xfer_apply_completions();
    if (reason == REFRESH_PROMPT_END) g_nn_phase = 2;
    moert_policy_refresh(g.pol, reason);
    g_nn_phase = 0;
    dtrace_decision(reason == REFRESH_PROMPT_END ? 2 : 3);
    excl_reconcile();
    moe_xfer::flush();
    excl_log(reason == REFRESH_TURN_END ? "turn end" : "refresh");
}

// Variable-KV S6a deferred trigger: the HTTP thread (POST /moe/rebuild_k) only SCHEDULES the
// rebuild here; on_turn_end — which the fork fires on the SLOT thread at slot release, i.e. a
// quiescent point with no in-flight graph — applies it. 0 = none pending.
static std::atomic<int> g_pending_rebuild_k{0};

int request_rebuild_k(int k, char * err, int errcap) {
    auto fail = [&](const char * m) {
        if (err && errcap > 0) { std::snprintf(err, (size_t) errcap, "%s", m); }
        return 0;
    };
    if (!g.dynamic || !g.mm || !g.pol) return fail("no dynamic runtime loaded");
    if (k < 1 || k > g.n_expert)       return fail("K out of range");
    g_pending_rebuild_k.store(k);
    std::cerr << "moe_dual: rebuild K=" << k << " SCHEDULED (applies at the next turn end)\n";
    return 1;
}

void on_turn_end() {
    if (!g.dynamic) return;
    g_arena_vision_hold = false;   // release the image-turn hold: the turn-end refresh below re-shrinks safely
                                   // (between turns — no decode follows to read a just-decommitted arena page)
    refresh_cache(REFRESH_TURN_END);
    // apply a scheduled variable-KV rebuild at this quiescent point (slot thread, turn over)
    if (const int k = g_pending_rebuild_k.exchange(0); k > 0) slab_rebuild_at_k(k);
}

bool is_dynamic() { return g.dynamic; }
bool is_staged()  { return g.dynamic; }
bool island_stats(IslandStats * out) {
#ifdef MOE_DUAL_HAVE_CUDA_MAP
    moert_cuda_stats s{};
    ggml_backend_cuda_moe_dual_stats(&s);
    *out = { s.fires, s.op_host_ns, s.seam_syncs, s.seam_ns, s.coldcpu_calls, s.coldcpu_pairs,
             s.coldcpu_ns, s.coldcpu_over_max };
    return true;
#else
    (void) out;
    return false;
#endif
}
int  cost_per_token()   { return is_staged() ? g.pcfg.cost_per_token : 0; }

void reset_stats() {
    // live counters only (learned residency + scores kept) — no-op when no dynamic cache is loaded
    moert_policy_reset_stats(g.pol);
}

void discard_pending_routes() {
    // Drain the on-device routing counters WITHOUT folding them into the policy (fetch with
    // reset, result dropped). Server path: llama-server's startup can_seq_rm probe decodes 2
    // dummy tokens through the runtime before any request; discarding right before serving
    // keeps the first real refresh's scores identical to the app path (parity gate M0).
#ifdef MOE_DUAL_HAVE_CUDA_MAP
    if (!g.dev_counts) return;
    std::vector<int32_t> buf((size_t) g.n_layers * g.n_expert + (size_t) g.n_layers * 2);
    ggml_backend_cuda_moe_dual_route_counts_fetch(buf.data(), /*reset=*/1);
#endif
}

bool dev_counts_active() { return g.dev_counts; }

double hit_rate() {
    // native observation: the policy pulls the latest device counters via the fetch callback
    StatsReader sr;
    return g.pol ? moert_policy_hit_rate(g.pol) : 0.0;
}

void seam_cache_state(SeamCacheState * out) {
    if (!out) return;
    *out = SeamCacheState();
    if (!g.pol) return;
    out->n_layers = g.n_layers > 128 ? 128 : g.n_layers;
    for (int il = 0; il < out->n_layers; ++il) {
        const int32_t kl = moert_policy_layer_k(g.pol, il);
        out->k_by_layer[il] = kl < 0 ? 0 : (int) kl;
    }
    out->k_changes = g.k_changes;
    out->rebuilds  = g.slab_rebuilds;
    moert_policy_stats s = {};
    { StatsReader sr; moert_policy_get_stats(g.pol, &s); }
    out->promotions = (long long) s.promotions;
    out->demotions  = (long long) s.demotions;
}

bool vram_now(long long * free_mib, long long * total_mib) {
    size_t free_b = 0, total_b = 0;
    if (!pool_dev_memory(&free_b, &total_b)) return false;
    if (free_mib)  *free_mib  = (long long) (free_b  >> 20);
    if (total_mib) *total_mib = (long long) (total_b >> 20);
    return true;
}

// C5.6 memory accounting: pass-throughs so mem_partition can ask the OS which arena pages are
// actually resident instead of assuming all of them are (see mem_partition.hpp's RAM block).
int cold_arena_block_count() {
    if (!g.mm) return 0;
    return (int) moert_memmgr_cold_arena_block_count(g.mm);
}
bool cold_arena_block(int i, void ** base, unsigned long long * bytes) {
    if (!g.mm || !base || !bytes) return false;
    uint64_t b = 0;
    if (!moert_memmgr_cold_arena_block(g.mm, (int32_t) i, base, &b)) return false;
    *bytes = (unsigned long long) b;
    return true;
}
long long cold_arena_committed_mib() {
    if (!g.mm) return 0;
    return (long long) (moert_memmgr_arena_committed_bytes(g.mm) >> 20);
}
void measure_vram_post_load() {
    size_t free_b = 0, total_b = 0;
    if (!pool_dev_memory(&free_b, &total_b)) return;   // runtime inactive / pool not on a GPU
    g.vram_total_mib     = (long long) (total_b >> 20);
    g.vram_used_post_mib = (long long) ((total_b - free_b) >> 20);
}

RuntimeInfo runtime_info() {
    RuntimeInfo r;
    r.active     = (g.k > 0) || g.dynamic;
    r.dynamic    = g.dynamic;
    r.hot_k      = g.k;
    r.n_expert   = g.n_expert;
    r.n_layers   = g.n_layers;
    r.total_slots = g.k * g.n_layers;
    parse_hook_env();   // no-op after any init; makes a pre-init info snapshot consistent too
    r.fused = g.hcfg.fused != 0;
    r.planc = g.hcfg.planc;
    r.cache_mode = g.dynamic ? "staged" : "";
    r.k_requested             = g.k_requested;
    r.vram_total_mib          = g.vram_total_mib;
    r.vram_free_pre_slab_mib  = g.vram_free_pre_mib;
    r.slab_mib                = g_slab_evicted ? 0 : g.slab_mib;   // C5.2: evicted slab = 0 VRAM (mem panel honesty)
    r.vram_used_post_load_mib = g.vram_used_post_mib;
    if (g.pol) {
        // counters as of the last decision (StatsReader: no device pull from a reader, see cb_fetch_external)
        moert_policy_stats s;
        { StatsReader sr; moert_policy_get_stats(g.pol, &s); }
        r.promotions             = s.promotions;
        r.demotions              = s.demotions;
        r.tokens                 = s.tokens;
        r.hot_routes             = s.hot_routes;
        r.cold_routes            = s.cold_routes;
        r.visible_tokens         = s.visible_tokens;
        r.refresh_count          = s.refresh_count;
        r.refresh_promotions     = s.refresh_promotions;
        r.refresh_demotions      = s.refresh_demotions;
        r.refresh_pause_ms_total = s.refresh_pause_ms_total;
        r.refresh_pause_ms_max   = s.refresh_pause_ms_max;
        r.last_refresh_tok       = s.last_refresh_tok;
        r.last_refresh_promos    = s.last_refresh_promos;
        r.last_refresh_pause_ms  = s.last_refresh_pause_ms;
        r.occupancy              = s.occupancy;   // mean READY hot slots / K across layers
        moert_policy_window w;
        { StatsReader sr; moert_policy_get_window(g.pol, 32, &w); }   // M2 R9/R10: instantaneous rates, ~32-token window
        r.win_tokens     = w.tokens;
        r.win_tok_per_s  = w.tok_per_s;
        r.win_ms_per_tok = w.ms_per_tok;
        r.win_hit_rate   = w.hit_rate;
    }
    r.cold_arena_active     = g.cold_arena_on;
    r.cold_arena_verify_ok  = g.cold_arena_verify_ok;
    r.cold_arena_verify_bad = g.cold_arena_verify_bad;

    // --- live memory map (/moe/telemetry + bench mem-map). moe_dual-owned regions only; llama-owned KV/non-expert are
    // probed separately (mem_probe.hpp) where a llama_context is in scope.
    if (g.mm) {
        // borrowed IQ2 cold pool: sum the per-(layer,kind,is_scale) cold views (stride * n_expert).
        uint64_t cold_bytes = 0;
        for (int il = 0; il < g.n_layers; ++il)
            for (int kind = 0; kind < MOERT_MEMMGR_MAX_KIND; ++kind)
                for (int ws = 0; ws <= 1; ++ws) {
                    moert_cold_view v;
                    if (moert_memmgr_get_cold_view(g.mm, il, kind, ws, &v))
                        cold_bytes += v.stride * (uint64_t) v.n_expert;
                }
        r.cold_pool_mib  = (long long) (cold_bytes >> 20);
        r.cold_arena_mib = (long long) (moert_memmgr_cold_arena_bytes(g.mm) >> 20);
        r.arena_committed_mib = (long long) (moert_memmgr_arena_committed_bytes(g.mm) >> 20);
    }
    r.staging_mib = (long long) (q4stream_stats().buf_bytes >> 20);
    return r;
}

}  // namespace moe_dual
