#include "moe-dual.cuh"
#include "mmvq.cuh"    // ggml_cuda_mul_mat_vec_q (the one ggml kernel the island borrows)
#include "ggml-cuda.h" // export decls for the moe_dual host services (forwarders below)

#include "moert/cuda.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>

// GGML_OP_MOE_DUAL is a THIN TRAMPOLINE. The selective dual-precision MoE matmul
// (partition/matmuls/scatter, Plan C staging, cold-stage planner, device routing counters)
// lives in the moert-cuda library (moert/cuda, device-pointer ABI in moert/cuda.h) — this
// file only:
//   1. unpacks the ggml op (tensor srcs + op_params) into the plain-C args struct,
//   2. lends the island the two ggml services it borrows, via the host vtable:
//      the tuned quantized mul_mat_id (ggml_cuda_mul_mat_vec_q, with the device-counted
//      live-channel guard) and the backend's LIFO VRAM scratch pool,
//   3. re-exports the island's host services under the ggml_backend_cuda_* names the
//      host process links (single copy of island state: it lives in THIS library).
// MoE logic must not grow here — it goes in moert/cuda. This file is what keeps the fork's
// footprint inside ggml a trampoline instead of a second implementation.

namespace {

// Plain contiguous ggml_tensor descriptor over device scratch (F32/I32 activations/ids) or over
// a raw expert pool (quantized). Uses ggml_row_size for nb[1] so quantized types (blck size > 1)
// get the correct row stride; F32/I32 are unchanged (blck size 1). mul_mat_vec_q reads only
// ->type/->data/->ne/->nb of its operands (verified), never their buffer.
ggml_tensor mk_view(ggml_type type, int64_t ne0, int64_t ne1, int64_t ne2, void * data) {
    ggml_tensor t = {};
    t.type = type;
    t.ne[0] = ne0; t.ne[1] = ne1; t.ne[2] = ne2; t.ne[3] = 1;
    t.nb[0] = ggml_type_size(type);
    t.nb[1] = ggml_row_size(type, ne0);
    t.nb[2] = t.nb[1] * ne1;
    t.nb[3] = t.nb[2] * ne2;
    t.data  = data;
    t.op    = GGML_OP_NONE;
    return t;
}

// Per-fire host context behind the moert_ffn_moe_host vtable: the backend context (pool +
// stream), the weight tensors the island selects by MOERT_MM_W_*, and the LIFO bookkeeping
// the ggml pool free needs (actual allocated size per pointer).
struct HostCtx {
    ggml_backend_cuda_context * ctx;
    const ggml_tensor * w_hot;    // K-slot hot pool (also lends its buffer to raw views:
                                  // a non-compute buffer makes mul_mat_vec_q skip its
                                  // compute-padding clear)
    const ggml_tensor * w_cold;   // cold source tensor (planc mode 0) or NULL
    int64_t d_in, d_out;

    struct { void * ptr; size_t actual; } allocs[16];
    int n_allocs = 0;
};

void * host_pool_alloc(void * user, uint64_t nbytes) {
    HostCtx * hc = (HostCtx *) user;
    GGML_ASSERT(hc->n_allocs < 16);
    size_t actual = 0;
    void * p = hc->ctx->pool().alloc((size_t) nbytes, &actual);
    hc->allocs[hc->n_allocs++] = { p, actual };
    return p;
}

void host_pool_free(void * user, void * ptr) {
    HostCtx * hc = (HostCtx *) user;
    GGML_ASSERT(hc->n_allocs > 0 && hc->allocs[hc->n_allocs - 1].ptr == ptr && "pool free must be LIFO");
    --hc->n_allocs;
    hc->ctx->pool().free(ptr, hc->allocs[hc->n_allocs].actual);
}

void host_mm(void * user, const moert_mm_call * c) {
    HostCtx * hc = (HostCtx *) user;
    ggml_tensor w_view;
    const ggml_tensor * w = nullptr;
    switch (c->w_sel) {
        case MOERT_MM_W_HOT:      w = hc->w_hot;  break;
        case MOERT_MM_W_COLD_SRC: w = hc->w_cold; break;
        default:                                     // raw device view over a mapped/staged pool
            w_view = mk_view((ggml_type) c->w_type_tag, hc->d_in, hc->d_out, c->w_n, (void *) c->w_data);
            w_view.buffer = hc->w_hot->buffer;       // borrow: skip the compute-padding clear
            w = &w_view;
            break;
    }
    ggml_tensor b_t   = mk_view(GGML_TYPE_F32, hc->d_in,  c->b_ncols, 1, (void *) c->b);
    ggml_tensor ids_t = mk_view(GGML_TYPE_I32, c->n_ids,  1,          1, (void *) c->ids);
    ggml_tensor y_t   = mk_view(GGML_TYPE_F32, hc->d_out, c->n_ids,   1, c->y);
    // Variant-F de-dup: c->w_ptrs (per-expert mapped base table) is honored only on the RAW view;
    // for HOT / COLD_SRC selectors it is meaningless (w is a real contiguous tensor) and stays null.
    const void * const * w_ptrs = (c->w_sel != MOERT_MM_W_HOT && c->w_sel != MOERT_MM_W_COLD_SRC) ? c->w_ptrs : nullptr;
    ggml_cuda_mul_mat_vec_q(*hc->ctx, w, &b_t, &ids_t, &y_t, nullptr, c->ne_live, w_ptrs);
}

}  // namespace

void ggml_cuda_moe_dual(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * cur        = dst->src[0];   // [D_in, n_used|1, n_tokens] f32
    const ggml_tensor * ids        = dst->src[1];   // [n_used, n_tokens] i32
    const ggml_tensor * w_hot      = dst->src[2];   // [D_in, D_out, K] quantized
    const ggml_tensor * w_cold     = dst->src[3];   // [D_in, D_out, n_expert] quantized, or NULL (Plan C)
    const ggml_tensor * exp_slot   = dst->src[4];   // [n_expert] i32 expert->slot or -1
    const ggml_tensor * slot_exp   = dst->src[5];   // [K] i32 slot->expert or -1
    const ggml_tensor * slot_state = dst->src[6];   // [K] i32 SlotState
    const ggml_tensor * s_hot      = dst->src[7];   // [K] f32 down scale, or NULL
    const ggml_tensor * s_cold     = dst->src[8];   // [n_expert] f32 down scale, or NULL

    // op_params: {kind, top_k, il, cold_type, n_expert, w_cold_dev(2), s_cold_dev(2), planc_mode}.
    // planc_mode: 0 = w_cold is src[3]; 2 = staging ring (w_cold_dev/s_cold_dev are HOST ptrs; selected experts copied to VRAM per token).
    int32_t params[10];
    memcpy(params, dst->op_params, sizeof(params));

    moert_ffn_moe_args a = {};
    a.kind          = params[0];
    a.top_k         = params[1];
    a.il            = params[2];
    a.planc_mode    = params[9];
    a.n_used        = (int32_t) ids->ne[0];
    a.n_tokens      = (int32_t) ids->ne[1];
    a.ids_s1        = (int32_t) (ids->nb[1] / sizeof(int32_t));   // argsort_top_k VIEW row stride
    a.d_in          = (int32_t) cur->ne[0];
    a.d_out         = (int32_t) w_hot->ne[1];
    a.n_expert      = w_cold ? (int32_t) w_cold->ne[2] : params[4];
    a.k             = (int32_t) slot_exp->ne[0];   // capacity: the slab slots plus any loaned ones (findings 246)
    a.cold_type_tag = w_cold ? (int32_t) w_cold->type : params[3];
    a.expert_bytes  = (uint64_t) ggml_row_size((ggml_type) a.cold_type_tag, a.d_in) * (uint64_t) a.d_out;
    a.cur           = cur->data;
    a.ids           = ids->data;
    a.exp_slot      = exp_slot->data;
    a.slot_exp      = slot_exp->data;
    a.slot_state    = slot_state->data;
    a.s_hot         = s_hot  ? s_hot->data  : nullptr;
    a.s_cold        = s_cold ? s_cold->data : nullptr;
    memcpy(&a.w_cold_dev, &params[5], sizeof(uint64_t));
    memcpy(&a.s_cold_dev, &params[7], sizeof(uint64_t));
    a.has_w_cold    = w_cold != nullptr;
    a.dst           = dst->data;
    a.stream        = (void *) ctx.stream();
    GGML_ASSERT(a.n_used * a.n_tokens <= 32 && "moe_dual single-warp partition: Pmax must be <= 32 (decode)");

    HostCtx hc;
    hc.ctx    = &ctx;
    hc.w_hot  = w_hot;
    hc.w_cold = w_cold;
    hc.d_in   = a.d_in;
    hc.d_out  = a.d_out;

    moert_ffn_moe_host h;
    h.user       = &hc;
    h.pool_alloc = &host_pool_alloc;
    h.pool_free  = &host_pool_free;
    h.mm         = &host_mm;

    moert_ffn_moe(&a, &h);
    GGML_ASSERT(hc.n_allocs == 0 && "moert_ffn_moe must release all pool scratch");
}

// ---------------------------------------------------------------------------------
// Host-service forwarders. The island's state lives in moert-cuda linked into THIS
// library; the host process (moe_dual.cpp) reaches it only through these exports —
// it must never link moert-cuda directly (that would duplicate the state).
// ---------------------------------------------------------------------------------

extern "C" void ggml_backend_cuda_moe_dual_configure(const struct moert_cuda_config * cfg) {
    moert_cuda_configure(cfg);
}

extern "C" int ggml_backend_cuda_host_register_pinned(void * base, size_t size) {
    return (int) moert_cuda_host_register_pinned(base, (uint64_t) size);
}

extern "C" void * ggml_backend_cuda_host_map(void * base, size_t size) {
    return moert_cuda_host_map(base, (uint64_t) size);
}

extern "C" int ggml_backend_cuda_host_register_mapped(void * base, size_t size) {
    return (int) moert_cuda_host_register_mapped(base, (uint64_t) size);
}

extern "C" void ggml_backend_cuda_host_unregister(void * base) {
    moert_cuda_host_unregister(base);
}

extern "C" int ggml_backend_cuda_moe_dual_route_counts_enable(int n_layers, int n_expert) {
    return (int) moert_cuda_route_counts_enable(n_layers, n_expert);
}

extern "C" int ggml_backend_cuda_moe_dual_route_counts_fetch(int32_t * dst, int reset) {
    return (int) moert_cuda_route_counts_fetch(dst, reset);
}

extern "C" void ggml_backend_cuda_moe_dual_stats(struct moert_cuda_stats * out) {
    moert_cuda_stats_get(out);
}

extern "C" int ggml_backend_cuda_moe_dual_dir_init(int sm_blocks, int sm_ring_mib, int sm_stage_threads) {
    return (int) moert_cuda_dir_init((int32_t) sm_blocks, (int32_t) sm_ring_mib, (int32_t) sm_stage_threads);
}

extern "C" int ggml_backend_cuda_moe_dual_dir_async(void * dst_dev, const void * src, uint64_t nbytes) {
    return (int) moert_cuda_dir_async(dst_dev, src, nbytes);
}

extern "C" int64_t ggml_backend_cuda_moe_dual_dir_mark(void) {
    return moert_cuda_dir_mark();
}

extern "C" int64_t ggml_backend_cuda_moe_dual_dir_poll(int wait_all) {
    return moert_cuda_dir_poll(wait_all);
}

extern "C" int ggml_backend_cuda_moe_dual_publish_h2d(int n, void * const * dsts, const void * const * srcs, const uint64_t * sizes) {
    return (int) moert_cuda_publish_h2d(n, dsts, srcs, sizes);
}

extern "C" void ggml_backend_cuda_moe_dual_hot_table(int il, int kind, const void * const * dev_ptrs, int n, int type_tag) {
    moert_cuda_set_hot_table(il, kind, dev_ptrs, n, type_tag);
}

extern "C" int64_t ggml_backend_cuda_moe_dual_dir_gate(int op, int64_t arg) {
    return moert_cuda_dir_gate((int32_t) op, arg);
}

extern "C" void ggml_backend_cuda_moe_dual_dir_stats(struct moert_cuda_dir_stats * out) {
    moert_cuda_dir_stats_get(out);
}

