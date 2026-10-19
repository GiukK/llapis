// test-model-plan — the launcher's VRAM terms read from GGUF headers, for quants this box cannot load.
//
// WHY THIS FILE EXISTS. Which quant sits in models/low and models/high is the user's choice: IQ2_M + Q4_K_XL
// on the reference 8 GB box, Q4 + Q8 on a bigger card. The launcher used to size K from constants measured
// on the reference pair alone, so another pair was mis-sized (a Q4 low read as 61 MiB per slot instead of
// ~108: the boot K came out ~2x too high and died at the drafter load). `moert-server plan --terms` now
// reads the terms from the files; this test writes header-only GGUFs (metadata + tensor table, no weights:
// a few KiB each) for several quant pairs and checks every term against an independent computation.
// No GPU, no model. Takes a scratch directory as its one argument. Run it: ctest -R model-plan

#include "model_plan.hpp"

#include "ggml.h"
#include "gguf.h"

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;

static int g_fail = 0;

static void check(bool ok, const char * what) {
    std::printf("%s  %s\n", ok ? "  ok  " : "  FAIL", what);
    if (!ok) ++g_fail;
}

static void check_eq(long long got, long long want, const char * what) {
    const bool ok = got == want;
    std::printf("%s  %s (got %lld, want %lld)\n", ok ? "  ok  " : "  FAIL", what, got, want);
    if (!ok) ++g_fail;
}

static void check_near(double got, double want, const char * what) {
    const bool ok = std::fabs(got - want) <= 1e-6 * std::fmax(1.0, std::fabs(want));
    std::printf("%s  %s (got %.6f, want %.6f)\n", ok ? "  ok  " : "  FAIL", what, got, want);
    if (!ok) ++g_fail;
}

// The shapes of Gemma-4-26B-A4B (30 layers, period 6: every 6th layer is full attention, the other 25
// slide over a 1024-token window; 128 experts; 262144-token vocabulary). Header-only, so the size is free.
constexpr int N_LAYER = 30, N_EXPERT = 128, N_EMBD = 2816, N_FF = 768, N_VOCAB = 262144, WINDOW = 1024;
constexpr int N_SWA = N_LAYER - N_LAYER / 6, N_FULL = N_LAYER / 6;
constexpr int KV_HEADS_SWA = 8, KV_HEADS_FULL = 2, KEY_LEN = 512, KEY_LEN_SWA = 256;

static long long bytes(ggml_type t, long long ne0, long long ne1, long long ne2 = 1) {
    return (long long) ggml_row_size(t, ne0) * ne1 * ne2;
}
static long long expert_layer_bytes(ggml_type t) {   // gate_up [embd, 2*ff, n_expert] + down [ff, embd, n_expert]
    return bytes(t, N_EMBD, 2 * N_FF, N_EXPERT) + bytes(t, N_FF, N_EMBD, N_EXPERT);
}
static long long nonexpert_bytes(ggml_type ne_type) {  // per layer: q, k, v, o at the non-expert type
    return (long long) N_LAYER * 4 * bytes(ne_type, N_EMBD, N_EMBD);
}

// Header-only GGUF: the tensor table declares full-size tensors, the file carries no data.
static std::string write_model(const fs::path & dir, const char * name, ggml_type expert_type, ggml_type ne_type,
                               bool shared_kv) {
    ggml_init_params ip{};
    ip.mem_size   = 256 * ggml_tensor_overhead();
    ip.no_alloc   = true;
    ggml_context * ctx = ggml_init(ip);
    gguf_context * g   = gguf_init_empty();

    gguf_set_val_str(g, "general.architecture", "gemma4");
    gguf_set_val_u32(g, "general.file_type", (uint32_t) expert_type);
    gguf_set_val_u32(g, "gemma4.block_count", N_LAYER);
    gguf_set_val_u32(g, "gemma4.expert_count", N_EXPERT);
    gguf_set_val_u32(g, "gemma4.expert_used_count", 2);
    gguf_set_val_u32(g, "gemma4.attention.key_length", KEY_LEN);
    gguf_set_val_u32(g, "gemma4.attention.value_length", KEY_LEN);
    gguf_set_val_u32(g, "gemma4.attention.key_length_swa", KEY_LEN_SWA);
    gguf_set_val_u32(g, "gemma4.attention.value_length_swa", KEY_LEN_SWA);
    gguf_set_val_u32(g, "gemma4.attention.sliding_window", WINDOW);
    if (shared_kv) gguf_set_val_u32(g, "gemma4.attention.shared_kv_layers", 1);
    uint32_t heads[N_LAYER];
    for (int l = 0; l < N_LAYER; ++l) heads[l] = (l + 1) % 6 == 0 ? KV_HEADS_FULL : KV_HEADS_SWA;
    gguf_set_arr_data(g, "gemma4.attention.head_count_kv", GGUF_TYPE_UINT32, heads, N_LAYER);

    auto add = [&](const std::string & tname, ggml_type t, int64_t ne0, int64_t ne1, int64_t ne2) {
        ggml_tensor * x = ne2 > 1 ? ggml_new_tensor_3d(ctx, t, ne0, ne1, ne2) : ggml_new_tensor_2d(ctx, t, ne0, ne1);
        ggml_set_name(x, tname.c_str());
        gguf_add_tensor(g, x);
    };
    add("token_embd.weight", GGML_TYPE_Q8_0, N_EMBD, N_VOCAB, 1);
    for (int l = 0; l < N_LAYER; ++l) {
        const std::string b = "blk." + std::to_string(l) + ".";
        for (const char * w : { "attn_q", "attn_k", "attn_v", "attn_output" }) add(b + w + ".weight", ne_type, N_EMBD, N_EMBD, 1);
        add(b + "ffn_gate_up_exps.weight", expert_type, N_EMBD, 2 * N_FF, N_EXPERT);
        add(b + "ffn_down_exps.weight",    expert_type, N_FF, N_EMBD, N_EXPERT);
    }
    const std::string path = (dir / name).string();
    if (!gguf_write_to_file(g, path.c_str(), /*only_meta =*/ true)) throw std::runtime_error("cannot write " + path);
    gguf_free(g);
    ggml_free(ctx);
    return path;
}

static std::string write_drafter(const fs::path & dir) {
    ggml_init_params ip{};
    ip.mem_size = 8 * ggml_tensor_overhead();
    ip.no_alloc = true;
    ggml_context * ctx = ggml_init(ip);
    gguf_context * g   = gguf_init_empty();
    gguf_set_val_str(g, "general.architecture", "gemma4");
    ggml_tensor * e = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, 1024, N_VOCAB);
    ggml_set_name(e, "token_embd.weight");
    gguf_add_tensor(g, e);
    ggml_tensor * q = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 1024, 1024);
    ggml_set_name(q, "blk.0.attn_q.weight");
    gguf_add_tensor(g, q);
    const std::string path = (dir / "drafter.gguf").string();
    if (!gguf_write_to_file(g, path.c_str(), true)) throw std::runtime_error("cannot write " + path);
    gguf_free(g);
    ggml_free(ctx);
    return path;
}

// Every term of one mode against the arithmetic of the tensor shapes.
static void check_terms(const model_plan::Terms & t, ggml_type hot_type, ggml_type hot_ne_type, ggml_type cold_type,
                        long long drafter_bytes, const char * label) {
    std::printf("-- %s\n", label);
    const double MiB = 1024.0 * 1024.0;
    check_near(t.slab_mib_per_k, (double) N_LAYER * expert_layer_bytes(hot_type) / N_EXPERT / MiB,
               "slab MiB per K = the hot source's expert bytes / n_expert");
    check_eq(t.ne_vram_mib, (nonexpert_bytes(hot_ne_type) + bytes(GGML_TYPE_Q8_0, N_EMBD, N_VOCAB)) >> 20,
             "non-experts in VRAM = the hot model's non-expert tensors + token_embd (tied output head)");
    check_eq(t.staging_mib, (expert_layer_bytes(hot_type) + (1 << 20) - 1) >> 20,
             "staging = one full expert layer of the hot source, rounded up");
    check_eq(t.cold_arena_mib, ((long long) N_LAYER * expert_layer_bytes(cold_type)) >> 20,
             "cold arena = the LOW model's expert bytes");
    check_near(t.kv_mib_per_tok, (double) N_FULL * KV_HEADS_FULL * (KEY_LEN + KEY_LEN) * 2 / MiB,
               "KV per token = the full-attention layers only (f16 K+V)");
    check_eq(t.kv_swa_mib, ((long long) N_SWA * WINDOW * KV_HEADS_SWA * (KEY_LEN_SWA + KEY_LEN_SWA) * 2) >> 20,
             "SWA KV = the window-capped layers");
    check_eq(t.drafter_mib, drafter_bytes >> 20, "drafter = every tensor of the drafter file");
    check_eq(t.n_expert, N_EXPERT, "n_expert");
}

int main(int argc, char ** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: test-model-plan <scratch dir>\n"); return 2; }
    const fs::path dir(argv[1]);
    fs::create_directories(dir);

    const std::string drafter = write_drafter(dir);
    const long long drafter_bytes = bytes(GGML_TYPE_Q8_0, 1024, N_VOCAB) + bytes(GGML_TYPE_F16, 1024, 1024);

    // 1. the reference shape: an IQ2 low and a Q4 high, Q4/IQ2 non-experts
    const std::string iq2 = write_model(dir, "iq2.gguf", GGML_TYPE_IQ2_XXS, GGML_TYPE_Q4_K, true);
    const std::string q4  = write_model(dir, "q4.gguf",  GGML_TYPE_Q4_K,    GGML_TYPE_Q4_K, true);
    // 2. the bigger card: a Q4 low (the fast model) and a Q8 high (the quality model)
    const std::string q8  = write_model(dir, "q8.gguf",  GGML_TYPE_Q8_0,    GGML_TYPE_Q8_0, true);

    try {
        check_terms(model_plan::compute_terms("speed", "", iq2, drafter), GGML_TYPE_IQ2_XXS, GGML_TYPE_Q4_K,
                    GGML_TYPE_IQ2_XXS, drafter_bytes, "speed, IQ2 low: both tiers are the low file");
        check_terms(model_plan::compute_terms("quality", q4, iq2, drafter), GGML_TYPE_Q4_K, GGML_TYPE_Q4_K,
                    GGML_TYPE_IQ2_XXS, drafter_bytes, "quality, Q4 high over IQ2 low");
        check_terms(model_plan::compute_terms("speed", "", q4, drafter), GGML_TYPE_Q4_K, GGML_TYPE_Q4_K,
                    GGML_TYPE_Q4_K, drafter_bytes, "speed, Q4 low: a Q4 slot, not an IQ2 one");
        check_terms(model_plan::compute_terms("quality", q8, q4, drafter), GGML_TYPE_Q8_0, GGML_TYPE_Q8_0,
                    GGML_TYPE_Q4_K, drafter_bytes, "quality, Q8 high over Q4 low");
    } catch (const std::exception & e) {
        check(false, (std::string("compute_terms threw on a valid pair: ") + e.what()).c_str());
    }

    std::printf("-- refusals\n");
    {
        const model_plan::Terms a = model_plan::compute_terms("quality", q8, q4, drafter);
        const model_plan::Terms b = model_plan::compute_terms("quality", q4, iq2, drafter);
        check(a.staging_mib > b.staging_mib && a.slab_mib_per_k > b.slab_mib_per_k,
              "a Q8 hot source costs more per slot and per staging layer than a Q4 one");
        check(model_plan::compute_terms("speed", "", iq2, "").drafter_mib == 0, "no drafter: drafter_mib 0");
    }
    auto throws = [](auto f) { try { f(); return false; } catch (const std::exception &) { return true; } };
    check(throws([&] { model_plan::compute_terms("quality", iq2, q4, drafter); }),
          "quality refuses a high model with FEWER expert bytes than the low one");
    check(throws([&] { model_plan::compute_terms("quality", "", q4, drafter); }), "quality refuses a missing high model");
    check(throws([&] { model_plan::compute_terms("speed", "", (dir / "nope.gguf").string(), drafter); }),
          "a missing low file throws");
    check(throws([&] { model_plan::compute_terms("turbo", "", q4, drafter); }), "an unknown mode throws");
    check(!throws([&] { model_plan::read_model(q4); }),
          "shared_kv_layers no longer refuses the model (gemma4 has 4; KV is an upper bound)");

    std::printf("%s\n", g_fail ? "FAILED" : "all ok");
    return g_fail ? 1 : 0;
}
