#include "model_plan.hpp"

#include "gguf.h"
#include "ggml-backend.h"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

#include <algorithm>
#include <cctype>
#include <cinttypes>
#include <cstring>
#include <filesystem>
#include <sstream>
#include <stdexcept>

namespace fs = std::filesystem;

namespace model_plan {

// --- gguf helpers (STRICT: every getter throws with the key name) -----------------------

namespace {

struct Gguf {
    gguf_context * g = nullptr;
    std::string path;
    explicit Gguf(const std::string & p) : path(p) {
        gguf_init_params ip{};
        ip.no_alloc = true;
        ip.ctx      = nullptr;
        g = gguf_init_from_file(p.c_str(), ip);
        if (!g) throw std::runtime_error("model_plan: cannot open GGUF (missing or corrupt): " + p);
    }
    ~Gguf() { if (g) gguf_free(g); }
    Gguf(const Gguf &) = delete;
    Gguf & operator=(const Gguf &) = delete;

    int64_t find(const std::string & key) const { return gguf_find_key(g, key.c_str()); }

    uint32_t u32_req(const std::string & key) const {
        const int64_t i = find(key);
        if (i < 0) throw std::runtime_error("model_plan: required GGUF key missing: '" + key + "' in " + path
                                            + " (the GGUF does not carry it; check that this is a Gemma 4 MoE model)");
        const gguf_type t = gguf_get_kv_type(g, i);
        if (t == GGUF_TYPE_UINT32) return gguf_get_val_u32(g, i);
        if (t == GGUF_TYPE_INT32)  return (uint32_t) gguf_get_val_i32(g, i);
        throw std::runtime_error("model_plan: key '" + key + "' has unexpected type (want scalar u32/i32) in " + path);
    }
    // per-layer u32: scalar broadcasts to all n layers; an array must have exactly n
    // entries (gemma4: head_count_kv is a genuine non-uniform per-layer array).
    std::vector<uint32_t> u32_layers_req(const std::string & key, size_t n) const {
        const int64_t i = find(key);
        if (i < 0) throw std::runtime_error("model_plan: required GGUF key missing: '" + key + "' in " + path
                                            + " (the GGUF does not carry it; check that this is a Gemma 4 MoE model)");
        const gguf_type t = gguf_get_kv_type(g, i);
        if (t == GGUF_TYPE_UINT32 || t == GGUF_TYPE_INT32)
            return std::vector<uint32_t>(n, u32_req(key));
        if (t == GGUF_TYPE_ARRAY) {
            const gguf_type at = gguf_get_arr_type(g, i);
            const size_t    an = gguf_get_arr_n(g, i);
            if (at != GGUF_TYPE_UINT32 && at != GGUF_TYPE_INT32)
                throw std::runtime_error("model_plan: key '" + key + "' is an array of unsupported type in " + path);
            if (an != n)
                throw std::runtime_error("model_plan: key '" + key + "' has " + std::to_string(an) +
                                         " entries, expected " + std::to_string(n) + " (block_count) in " + path);
            const uint32_t * d = (const uint32_t *) gguf_get_arr_data(g, i);
            return std::vector<uint32_t>(d, d + n);
        }
        throw std::runtime_error("model_plan: key '" + key + "' has unexpected type in " + path);
    }
    uint32_t u32_opt(const std::string & key, uint32_t fallback, bool * found = nullptr) const {
        const int64_t i = find(key);
        if (i < 0) { if (found) *found = false; return fallback; }
        if (found) *found = true;
        return u32_req(key);
    }
    std::string str_req(const std::string & key) const {
        const int64_t i = find(key);
        if (i < 0) throw std::runtime_error("model_plan: required GGUF key missing: '" + key + "' in " + path);
        return gguf_get_val_str(g, i);
    }
};

bool is_expert_tensor(const char * name) {
    // ffn_{gate,up,down}_exps.weight / merged gate_up variants — the "_exps" marker
    return std::strstr(name, "_exps") != nullptr;
}

}  // namespace

double ModelInfo::per_k_step_mib() const {
    return n_expert > 0 ? (double) expert_bytes / n_expert / (1024.0 * 1024.0) : 0.0;
}

ModelInfo read_model(const std::string & gguf_path) {
    Gguf f(gguf_path);
    ModelInfo m;
    m.path = gguf_path;
    std::error_code ec;
    m.file_bytes = (long long) fs::file_size(gguf_path, ec);
    if (ec) throw std::runtime_error("model_plan: cannot stat file: " + gguf_path);

    m.arch = f.str_req("general.architecture");
    {
        const int64_t i = f.find("general.file_type");
        m.file_type = i >= 0 ? (int32_t) gguf_get_val_u32(f.g, i) : -1;
    }
    const std::string a = m.arch;
    m.n_layer       = (int32_t) f.u32_req(a + ".block_count");
    m.n_expert      = (int32_t) f.u32_req(a + ".expert_count");
    m.n_expert_used = (int32_t) f.u32_req(a + ".expert_used_count");
    // key/value length: explicit keys only; STRICT (no embedding/head_count derivation).
    m.key_length    = (int32_t) f.u32_req(a + ".attention.key_length");
    m.value_length  = (int32_t) f.u32_req(a + ".attention.value_length");
    m.sliding_window = (int32_t) f.u32_opt(a + ".attention.sliding_window", 0);
    m.key_length_swa   = (int32_t) f.u32_opt(a + ".attention.key_length_swa",   (uint32_t) m.key_length);
    m.value_length_swa = (int32_t) f.u32_opt(a + ".attention.value_length_swa", (uint32_t) m.value_length);

    // kv heads: per layer (gemma4: 8 on SWA layers, 2 on full layers — a real array)
    {
        const auto v = f.u32_layers_req(a + ".attention.head_count_kv", (size_t) m.n_layer);
        m.kv_heads.assign(v.begin(), v.end());
    }
    // cross-layer KV sharing removes whole layers from the allocation; it is recorded but not subtracted,
    // so every KV figure below is an upper bound (gemma4 shares 4 layers: the reference box's measured
    // 0.0195 MiB/token counts all 5 full-attention layers).
    m.shared_kv_layers = (int32_t) f.u32_opt(a + ".attention.shared_kv_layers", 0);

    // which layers are SWA: explicit per-layer bool pattern > integer period > arch rule
    m.is_swa.assign((size_t) m.n_layer, 0);
    if (m.sliding_window > 0) {
        const std::string pk = a + ".attention.sliding_window_pattern";
        const int64_t pi = f.find(pk);
        if (pi >= 0 && gguf_get_kv_type(f.g, pi) == GGUF_TYPE_ARRAY) {
            if (gguf_get_arr_type(f.g, pi) != GGUF_TYPE_BOOL ||
                gguf_get_arr_n(f.g, pi) != (size_t) m.n_layer)
                throw std::runtime_error("model_plan: '" + pk + "' array is not bool[block_count] in " + gguf_path);
            const int8_t * d = (const int8_t *) gguf_get_arr_data(f.g, pi);
            for (int32_t l = 0; l < m.n_layer; ++l) m.is_swa[(size_t) l] = d[l] ? 1 : 0;
        } else {
            uint32_t period = 0;
            if (pi >= 0)                     period = f.u32_req(pk);
            else if (a.rfind("gemma3", 0) == 0 || a.rfind("gemma4", 0) == 0)
                period = 6;   // gemma3/4 family: every 6th layer is full-attention (llama.cpp hparams)
            else
                throw std::runtime_error(
                    "model_plan: arch '" + a + "' uses sliding-window attention but neither the "
                    "'" + pk + "' key nor a known arch rule exists — refusing to guess the kv "
                    "layout (add the arch rule in model_plan.cpp)");
            if (period == 0)
                throw std::runtime_error("model_plan: '" + pk + "' period is 0 in " + gguf_path);
            for (int32_t l = 0; l < m.n_layer; ++l)
                m.is_swa[(size_t) l] = ((uint32_t)(l + 1) % period != 0) ? 1 : 0;
        }
    }

    // tensor table sums
    const int64_t nt = gguf_get_n_tensors(f.g);
    if (nt <= 0) throw std::runtime_error("model_plan: GGUF has no tensors: " + gguf_path);
    int32_t max_moe_block = -1;
    std::vector<long long> layer_expert_bytes((size_t) m.n_layer, 0);
    for (int64_t i = 0; i < nt; ++i) {
        const char *  name = gguf_get_tensor_name(f.g, i);
        const long long sz = (long long) gguf_get_tensor_size(f.g, i);
        if (is_expert_tensor(name)) {
            m.expert_bytes += sz;
            int blk = -1;
            if (std::sscanf(name, "blk.%d.", &blk) == 1) max_moe_block = std::max(max_moe_block, blk);
            if (blk >= 0 && blk < m.n_layer) {
                layer_expert_bytes[(size_t) blk] += sz;
                m.max_layer_expert_bytes = std::max(m.max_layer_expert_bytes, layer_expert_bytes[(size_t) blk]);
            }
        } else if (std::strncmp(name, "token_embd.", 11) == 0) {
            m.token_embd_bytes += sz;
        } else {
            m.nonexpert_bytes += sz;
        }
    }
    // count distinct MoE layers exactly
    {
        std::vector<char> seen((size_t) m.n_layer, 0);
        for (int64_t i = 0; i < nt; ++i) {
            const char * name = gguf_get_tensor_name(f.g, i);
            int blk = -1;
            if (is_expert_tensor(name) && std::sscanf(name, "blk.%d.", &blk) == 1 &&
                blk >= 0 && blk < m.n_layer)
                seen[(size_t) blk] = 1;
        }
        for (char c : seen) m.n_moe_layer += c ? 1 : 0;
    }
    if (m.expert_bytes <= 0)
        throw std::runtime_error("model_plan: no *_exps expert tensors found in " + gguf_path +
                                 " — not a MoE model, or a split shard (pass the first shard)");
    return m;
}

std::vector<std::string> dump_keys(const std::string & gguf_path) {
    Gguf f(gguf_path);
    std::vector<std::string> out;
    const int64_t n = gguf_get_n_kv(f.g);
    for (int64_t i = 0; i < n; ++i) {
        std::ostringstream os;
        os << gguf_get_key(f.g, i) << " = ";
        const gguf_type t = gguf_get_kv_type(f.g, i);
        auto scalar = [&](gguf_type st, int64_t j, bool arr, size_t ai) {
            switch (st) {
                case GGUF_TYPE_UINT8:   os << (arr ? (uint64_t)((const uint8_t  *) gguf_get_arr_data(f.g,j))[ai] : (uint64_t)gguf_get_val_u8 (f.g,j)); break;
                case GGUF_TYPE_INT8:    os << (arr ? (int64_t )((const int8_t   *) gguf_get_arr_data(f.g,j))[ai] : (int64_t )gguf_get_val_i8 (f.g,j)); break;
                case GGUF_TYPE_UINT16:  os << (arr ? (uint64_t)((const uint16_t *) gguf_get_arr_data(f.g,j))[ai] : (uint64_t)gguf_get_val_u16(f.g,j)); break;
                case GGUF_TYPE_INT16:   os << (arr ? (int64_t )((const int16_t  *) gguf_get_arr_data(f.g,j))[ai] : (int64_t )gguf_get_val_i16(f.g,j)); break;
                case GGUF_TYPE_UINT32:  os << (arr ? (uint64_t)((const uint32_t *) gguf_get_arr_data(f.g,j))[ai] : (uint64_t)gguf_get_val_u32(f.g,j)); break;
                case GGUF_TYPE_INT32:   os << (arr ? (int64_t )((const int32_t  *) gguf_get_arr_data(f.g,j))[ai] : (int64_t )gguf_get_val_i32(f.g,j)); break;
                case GGUF_TYPE_UINT64:  os << (arr ? ((const uint64_t *) gguf_get_arr_data(f.g,j))[ai] : gguf_get_val_u64(f.g,j)); break;
                case GGUF_TYPE_INT64:   os << (arr ? ((const int64_t  *) gguf_get_arr_data(f.g,j))[ai] : gguf_get_val_i64(f.g,j)); break;
                case GGUF_TYPE_FLOAT32: os << (arr ? ((const float    *) gguf_get_arr_data(f.g,j))[ai] : gguf_get_val_f32(f.g,j)); break;
                case GGUF_TYPE_FLOAT64: os << (arr ? ((const double   *) gguf_get_arr_data(f.g,j))[ai] : gguf_get_val_f64(f.g,j)); break;
                case GGUF_TYPE_BOOL:    os << ((arr ? (bool)((const int8_t *) gguf_get_arr_data(f.g,j))[ai] : gguf_get_val_bool(f.g,j)) ? "true" : "false"); break;
                default: os << "?";
            }
        };
        if (t == GGUF_TYPE_STRING) {
            os << "\"" << gguf_get_val_str(f.g, i) << "\"";
        } else if (t == GGUF_TYPE_ARRAY) {
            const gguf_type at = gguf_get_arr_type(f.g, i);
            const size_t    an = gguf_get_arr_n(f.g, i);
            os << "[" << an << " x type" << (int) at << "] ";
            if (at == GGUF_TYPE_STRING) {
                os << "(strings)";
            } else {
                const size_t show = std::min<size_t>(an, 64);
                for (size_t ai = 0; ai < show; ++ai) { if (ai) os << ","; scalar(at, i, true, ai); }
                if (show < an) os << ",...";
            }
        } else {
            scalar(t, i, false, 0);
        }
        out.push_back(os.str());
    }
    return out;
}

std::string find_model_gguf(const std::string & dir_s) {
    const fs::path dir(dir_s);
    if (!fs::exists(dir))
        throw std::runtime_error("model_plan: models folder missing: " + dir.string());
    std::vector<std::string> found;
    for (const auto & e : fs::directory_iterator(dir)) {
        if (!e.is_regular_file() || e.path().extension() != ".gguf")
            continue;
        // multimodal projectors (mmproj-*.gguf, e.g. the C4d-quantized one that lives
        // next to the Q4 model so the -Vision auto-find ranks it first) are not model
        // candidates — the pair is the text model only
        std::string stem = e.path().stem().string();
        std::transform(stem.begin(), stem.end(), stem.begin(),
                       [](unsigned char c) { return (char) std::tolower(c); });
        if (stem.rfind("mmproj", 0) == 0)
            continue;
        found.push_back(e.path().string());
    }
    if (found.size() != 1) {
        std::ostringstream os;
        os << "model_plan: " << dir.string() << " must contain EXACTLY one model .gguf "
              "(mmproj-* projectors are ignored), found " << found.size();
        for (const auto & p : found) os << "\n  - " << p;
        throw std::runtime_error(os.str());
    }
    return found[0];
}

ModelPair scan_models_dir(const std::string & models_dir) {
    const fs::path root(models_dir);
    ModelPair p{ read_model(find_model_gguf((root / "high").string())), read_model(find_model_gguf((root / "low").string())) };
    return p;
}

void validate_pair(const ModelPair & p) {
    std::vector<std::string> errs;
    auto chk = [&](bool ok, const std::string & msg) { if (!ok) errs.push_back(msg); };
    auto eq  = [&](const char * what, long long h, long long l) {
        if (h != l) errs.push_back(std::string(what) + " differs: high=" + std::to_string(h) +
                                   " low=" + std::to_string(l));
    };
    chk(p.high.arch == p.low.arch, "arch differs: high=" + p.high.arch + " low=" + p.low.arch);
    eq("n_layer",        p.high.n_layer,        p.low.n_layer);
    eq("n_moe_layer",    p.high.n_moe_layer,    p.low.n_moe_layer);
    eq("n_expert",       p.high.n_expert,       p.low.n_expert);
    eq("n_expert_used",  p.high.n_expert_used,  p.low.n_expert_used);
    eq("key_length",     p.high.key_length,     p.low.key_length);
    eq("value_length",   p.high.value_length,   p.low.value_length);
    eq("key_length_swa",   p.high.key_length_swa,   p.low.key_length_swa);
    eq("value_length_swa", p.high.value_length_swa, p.low.value_length_swa);
    eq("sliding_window", p.high.sliding_window, p.low.sliding_window);
    chk(p.high.kv_heads == p.low.kv_heads, "per-layer head_count_kv differs");
    chk(p.high.is_swa   == p.low.is_swa,   "per-layer SWA pattern differs");
    chk(p.high.expert_bytes > p.low.expert_bytes,
        "high must be the higher-precision expert source (expert bytes high=" +
        std::to_string(p.high.expert_bytes) + " <= low=" + std::to_string(p.low.expert_bytes) + ")");
    if (!errs.empty()) {
        std::ostringstream os;
        os << "model_plan: high/low pair INVALID (" << errs.size() << " problem(s)):";
        for (const auto & e : errs) os << "\n  - " << e;
        throw std::runtime_error(os.str());
    }
}

HwProbe probe_hw() {
    HwProbe hw;
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (!dev)
        throw std::runtime_error("model_plan: no GPU backend device found (CUDA build required, "
                                 "or the driver is unavailable) — refusing to plan without a probe");
    size_t free_b = 0, total_b = 0;
    ggml_backend_dev_memory(dev, &free_b, &total_b);
    if (total_b == 0)
        throw std::runtime_error("model_plan: GPU device reports zero total memory — probe failed");
    hw.vram_free_mib  = (long long) (free_b  >> 20);
    hw.vram_total_mib = (long long) (total_b >> 20);
    hw.gpu_desc = ggml_backend_dev_description(dev);
#ifdef _WIN32
    MEMORYSTATUSEX ms{};
    ms.dwLength = sizeof(ms);
    if (!GlobalMemoryStatusEx(&ms))
        throw std::runtime_error("model_plan: GlobalMemoryStatusEx failed");
    hw.ram_total_mib = (long long) (ms.ullTotalPhys >> 20);
    hw.ram_avail_mib = (long long) (ms.ullAvailPhys >> 20);
#endif
    return hw;
}

const char * verdict_name(Verdict v) {
    switch (v) {
        case Verdict::SAFE:   return "SAFE";
        case Verdict::WARN:   return "WARN";
        case Verdict::REFUSE: return "REFUSE";
    }
    return "?";
}

static long long kv_alloc_mib(const ModelInfo & m, int32_t ctx) {
    // exact per-layer sum, f16 cells: tokens(l) * kv_heads[l] * (K_len(l)+V_len(l)) * 2 bytes.
    // SWA layers hold min(ctx, window) tokens and use the *_swa lengths (gemma4: 256 vs 512).
    long long total = 0;
    for (int32_t l = 0; l < m.n_layer; ++l) {
        const bool swa = m.is_swa[(size_t) l] != 0;
        const long long toks  = swa ? std::min<int32_t>(ctx, m.sliding_window) : ctx;
        const long long kvlen = swa ? (m.key_length_swa + m.value_length_swa)
                                    : (m.key_length     + m.value_length);
        total += toks * m.kv_heads[(size_t) l] * kvlen * 2 /*f16*/;
    }
    return total >> 20;
}

Plan compute_plan(const ModelPair & p, const HwProbe & hw, const PlanOptions & opt) {
    if (opt.ctx <= 0 || opt.k_min <= 0 || opt.k_max < opt.k_min)
        throw std::runtime_error("model_plan: invalid plan options (ctx/k range)");
    Plan pl;
    pl.opt            = opt;
    pl.per_k_step_mib = p.high.per_k_step_mib();
    // token_embd goes to VRAM too (gemma: tied output head — M2 partition measured
    // nonexpert_weights = 2459 MiB = non-expert 1711 + token_embd 748; the "host-side
    // embeddings" assumption was WRONG on the reference box).
    pl.ne_vram_mib    = (p.high.nonexpert_bytes + p.high.token_embd_bytes) >> 20;
    pl.kv_alloc_mib   = kv_alloc_mib(p.high, opt.ctx);
    pl.ram_low_arena_mib = p.low.expert_bytes >> 20;

    const long long fixed = pl.ne_vram_mib + pl.kv_alloc_mib + opt.compute_reserve_mib;
    for (int32_t k = opt.k_min; k <= opt.k_max; ++k) {
        PlanRow r;
        r.k        = k;
        r.slab_mib = (long long) (pl.per_k_step_mib * k + 0.5);
        r.need_mib = fixed + r.slab_mib;
        r.headroom_mib = hw.vram_free_mib - r.need_mib;
        r.verdict = r.headroom_mib >= opt.safe_mib  ? Verdict::SAFE
                  : r.headroom_mib >= opt.tight_mib ? Verdict::WARN
                                                    : Verdict::REFUSE;
        if (r.verdict == Verdict::SAFE) pl.k_max_safe = k;
        pl.rows.push_back(r);
    }
    if (pl.k_max_safe == 0)
        throw std::runtime_error("model_plan: no K in [" + std::to_string(opt.k_min) + "," +
                                 std::to_string(opt.k_max) + "] is SAFE on this machine (free " +
                                 std::to_string(hw.vram_free_mib) + " MiB) — refusing to suggest a config");
    pl.k_default = std::min<int32_t>(32, pl.k_max_safe);
    // RAM check against TOTAL (capacity): cold arena + ~1 GiB working set. Available-now
    // is transient (other apps) — the shell warns on it separately; measured reality on
    // the reference box: peak RSS ≈ 5.8 GiB with a 7857 MiB arena (not all pages resident).
    pl.ram_fits = pl.ram_low_arena_mib + 1024 <= hw.ram_total_mib;
    return pl;
}

static void json_model(std::ostringstream & os, const char * tag, const ModelInfo & m) {
    std::string esc;
    for (char c : m.path) { if (c == '\\' || c == '"') esc += '\\'; esc += c; }
    os << "\"" << tag << "\":{"
       << "\"path\":\"" << esc << "\",\"arch\":\"" << m.arch << "\",\"file_type\":" << m.file_type
       << ",\"file_mib\":" << (m.file_bytes >> 20)
       << ",\"n_layer\":" << m.n_layer << ",\"n_moe_layer\":" << m.n_moe_layer
       << ",\"n_expert\":" << m.n_expert << ",\"n_expert_used\":" << m.n_expert_used
       << ",\"key_length\":" << m.key_length << ",\"value_length\":" << m.value_length
       << ",\"key_length_swa\":" << m.key_length_swa << ",\"value_length_swa\":" << m.value_length_swa
       << ",\"sliding_window\":" << m.sliding_window
       << ",\"n_swa_layers\":" << (int) std::count(m.is_swa.begin(), m.is_swa.end(), (uint8_t) 1)
       << ",\"expert_mib\":" << (m.expert_bytes >> 20)
       << ",\"nonexpert_mib\":" << (m.nonexpert_bytes >> 20)
       << ",\"token_embd_mib\":" << (m.token_embd_bytes >> 20)
       << ",\"per_k_step_mib\":" << m.per_k_step_mib() << "}";
}

std::string plan_to_json(const ModelPair & mp, const HwProbe & hw, const Plan & p) {
    std::ostringstream os;
    os << "{";
    json_model(os, "high", mp.high); os << ",";
    json_model(os, "low",  mp.low);  os << ",";
    os << "\"hw\":{\"gpu\":\"" << hw.gpu_desc << "\",\"vram_total_mib\":" << hw.vram_total_mib
       << ",\"vram_free_mib\":" << hw.vram_free_mib
       << ",\"ram_total_mib\":" << hw.ram_total_mib << ",\"ram_avail_mib\":" << hw.ram_avail_mib << "},";
    os << "\"constants\":{\"per_k_step_mib\":" << p.per_k_step_mib
       << ",\"ne_vram_mib\":" << p.ne_vram_mib
       << ",\"kv_alloc_mib\":" << p.kv_alloc_mib
       << ",\"compute_reserve_mib\":" << p.opt.compute_reserve_mib
       << ",\"compute_reserve_source\":\"CALIBRATED residual (M2 partition + product-gate anchors K32/K36/K38, reference box); override with --reserve\""
       << ",\"safe_mib\":" << p.opt.safe_mib << ",\"tight_mib\":" << p.opt.tight_mib
       << ",\"ctx\":" << p.opt.ctx << "},";
    os << "\"rows\":[";
    for (size_t i = 0; i < p.rows.size(); ++i) {
        const auto & r = p.rows[i];
        os << (i ? "," : "") << "{\"k\":" << r.k << ",\"slab_mib\":" << r.slab_mib
           << ",\"need_mib\":" << r.need_mib << ",\"headroom_mib\":" << r.headroom_mib
           << ",\"verdict\":\"" << verdict_name(r.verdict) << "\"}";
    }
    os << "],";
    os << "\"k_max_safe\":" << p.k_max_safe << ",\"k_default\":" << p.k_default
       << ",\"ram_low_arena_mib\":" << p.ram_low_arena_mib
       << ",\"ram_fits\":" << (p.ram_fits ? "true" : "false") << "}";
    return os.str();
}

// --- launcher terms -------------------------------------------------------

static long long all_tensor_bytes(const std::string & path) {
    Gguf f(path);
    long long total = 0;
    const int64_t nt = gguf_get_n_tensors(f.g);
    for (int64_t i = 0; i < nt; ++i) total += (long long) gguf_get_tensor_size(f.g, i);
    return total;
}

static std::string json_str(const std::string & s) {
    std::string out = "\"";
    for (char c : s) { if (c == '\\' || c == '"') out += '\\'; out += c; }
    return out + "\"";
}

Terms compute_terms(const std::string & mode, const std::string & high_path, const std::string & low_path,
                    const std::string & drafter_path) {
    if (mode != "speed" && mode != "quality")
        throw std::runtime_error("model_plan: --mode must be speed or quality, got '" + mode + "'");
    if (low_path.empty()) throw std::runtime_error("model_plan: the low model is required in both modes");
    const ModelInfo low = read_model(low_path);
    ModelInfo high;
    if (mode == "quality") {
        if (high_path.empty()) throw std::runtime_error("model_plan: quality needs the high model");
        high = read_model(high_path);
        validate_pair({ high, low });
    }
    const ModelInfo & hot = mode == "quality" ? high : low;

    Terms t;
    t.mode           = mode;
    t.hot_path       = hot.path;
    t.cold_path      = low.path;
    t.drafter_path   = drafter_path;
    t.slab_mib_per_k = hot.per_k_step_mib();
    t.ne_vram_mib    = (hot.nonexpert_bytes + hot.token_embd_bytes) >> 20;
    t.drafter_mib    = drafter_path.empty() ? 0 : all_tensor_bytes(drafter_path) >> 20;
    // per-token growth = the full-attention layers' cells; the SWA layers are a fixed, window-capped block
    long long full_bytes_per_tok = 0;
    for (int32_t l = 0; l < hot.n_layer; ++l)
        if (!hot.is_swa[(size_t) l])
            full_bytes_per_tok += (long long) hot.kv_heads[(size_t) l] * (hot.key_length + hot.value_length) * 2;
    t.kv_mib_per_tok = (double) full_bytes_per_tok / (1024.0 * 1024.0);
    t.kv_swa_mib     = kv_alloc_mib(hot, std::max<int32_t>(hot.sliding_window, 1)) -
                       (long long) (t.kv_mib_per_tok * std::max<int32_t>(hot.sliding_window, 1));
    if (t.kv_swa_mib < 0) t.kv_swa_mib = 0;
    t.cold_arena_mib = low.expert_bytes >> 20;
    t.staging_mib    = (hot.max_layer_expert_bytes + (1 << 20) - 1) >> 20;
    t.n_expert       = hot.n_expert;
    t.hot_file_type  = hot.file_type;
    t.cold_file_type = low.file_type;
    return t;
}

std::string terms_to_json(const Terms & t) {
    std::ostringstream os;
    os << "{\"mode\":" << json_str(t.mode)
       << ",\"hot_path\":" << json_str(t.hot_path) << ",\"cold_path\":" << json_str(t.cold_path)
       << ",\"drafter_path\":" << json_str(t.drafter_path)
       << ",\"slab_mib_per_k\":" << t.slab_mib_per_k
       << ",\"ne_vram_mib\":" << t.ne_vram_mib
       << ",\"drafter_mib\":" << t.drafter_mib
       << ",\"kv_mib_per_tok\":" << t.kv_mib_per_tok
       << ",\"kv_swa_mib\":" << t.kv_swa_mib
       << ",\"cold_arena_mib\":" << t.cold_arena_mib
       << ",\"staging_mib\":" << t.staging_mib
       << ",\"n_expert\":" << t.n_expert
       << ",\"hot_file_type\":" << t.hot_file_type << ",\"cold_file_type\":" << t.cold_file_type << "}";
    return os.str();
}

}  // namespace model_plan
