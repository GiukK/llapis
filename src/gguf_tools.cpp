#include "gguf_tools.hpp"

#include "io_util.hpp"

#include "ggml.h"
#include "gguf.h"

#include "moert/store.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <regex>
#include <set>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

namespace gguf_tools {

// Open a GGUF metadata-only (no_alloc): tensor shapes/types/offsets are read but
// no tensor data is loaded. The ggml_context receiving the tensor descriptors is
// returned through `gctx`.
static gguf_context * open_meta(const fs::path & p, ggml_context ** gctx) {
    gguf_init_params ip; ip.no_alloc = true; ip.ctx = gctx;
    gguf_context * g = gguf_init_from_file(p.string().c_str(), ip);
    if (!g) std::cerr << "failed to read gguf: " << p << "\n";
    return g;
}

static int blk_index(const std::string & n) {
    const auto p = n.find("blk.");
    if (p == std::string::npos) return -1;
    return std::atoi(n.c_str() + p + 4);
}

static bool ends_with(const std::string & s, const std::string & suf) {
    return s.size() >= suf.size() && s.compare(s.size() - suf.size(), suf.size(), suf) == 0;
}

// GGUF multi-file splits name shards "<prefix>-NNNNN-of-MMMMM.gguf" (llama.cpp's convention). If
// `path` matches, return every shard path in order 1..count; otherwise return {path} unchanged (the
// common single-file case). Mirrors moe_dual.cpp's gguf_split_shards — needed here so make_hybrid can
// take a split rest source (e.g. the unsloth BF16 GGUF ships as 2 shards, and the boundary lands MID
// tensor set: ~22 non-expert tensors of the last layers live only in the second shard).
static std::string zero_pad(int v, int width) {
    std::string s = std::to_string(v);
    if ((int) s.size() < width) s = std::string(width - s.size(), '0') + s;
    return s;
}
static std::vector<std::string> split_shards(const std::string & path) {
    const fs::path p(path);
    const std::string fname = p.filename().string();
    static const std::regex re(R"(^(.*)-(\d+)-of-(\d+)\.gguf$)", std::regex::icase);
    std::smatch m;
    if (!std::regex_match(fname, m, re)) return { path };
    const std::string prefix = m[1].str();
    const int width = (int) m[2].str().size();
    const int count = std::atoi(m[3].str().c_str());
    if (count <= 1) return { path };
    std::vector<std::string> out;
    const fs::path dir = p.parent_path();
    for (int i = 1; i <= count; ++i)
        out.push_back((dir / (prefix + "-" + zero_pad(i, width) + "-of-" + zero_pad(count, width) + ".gguf")).string());
    return out;
}

int make_hybrid(const fs::path & experts_from, const fs::path & rest_from, const fs::path & out_path,
                const std::string & take_low) {
    ggml_context * gc_exp = nullptr;
    gguf_context * g_exp = open_meta(experts_from, &gc_exp);
    if (!g_exp) return 1;

    // --take-low: exact non-expert tensor names ALSO sourced from experts_from.
    std::set<std::string> low_names;
    for (size_t p = 0; p < take_low.size();) {
        size_t c = take_low.find(',', p);
        if (c == std::string::npos) c = take_low.size();
        if (c > p) low_names.insert(take_low.substr(p, c - p));
        p = c + 1;
    }
    std::set<std::string> low_unmatched = low_names;

    // The "rest" (non-expert) source may be a multi-file GGUF split; open every shard. Each shard
    // owns a disjoint subset of the model's tensors, so the union over shards == the full tensor set.
    struct Shard { std::string path; gguf_context * g = nullptr; ggml_context * gc = nullptr; FILE * f = nullptr; };
    const std::vector<std::string> rest_paths = split_shards(rest_from.string());
    std::vector<Shard> shards(rest_paths.size());
    for (size_t si = 0; si < shards.size(); ++si) {
        shards[si].path = rest_paths[si];
        shards[si].g = open_meta(rest_paths[si], &shards[si].gc);
        if (!shards[si].g) return 1;
    }

    auto is_expert = [](const std::string & n) { return n.find("_exps") != std::string::npos; };

    // Build the output gguf: copy KV from the FIRST rest shard (holds arch + hyperparams), drop the
    // split.* keys so the single-file output is not mistaken for a shard, then add every tensor
    // (metadata only) with the type/shape from its chosen source: experts <- experts_from, every
    // non-expert tensor <- whichever rest shard owns it.
    ggml_init_params op; op.mem_size = 64u * 1024 * 1024; op.mem_buffer = nullptr; op.no_alloc = true;
    ggml_context * gc_out = ggml_init(op);
    gguf_context * g_out = gguf_init_empty();
    gguf_set_kv(g_out, shards[0].g);
    if (shards.size() > 1) {
        gguf_remove_key(g_out, "split.no");
        gguf_remove_key(g_out, "split.count");
        gguf_remove_key(g_out, "split.tensors.count");
    }

    struct Ent { std::string name; bool exp; int shard; };  // shard = owning rest shard (ignored if exp)
    std::vector<Ent> order;  // output add-order == iteration order
    int n_exp = 0, n_rest = 0;
    for (size_t si = 0; si < shards.size(); ++si) {
        const int64_t n = gguf_get_n_tensors(shards[si].g);
        for (int64_t i = 0; i < n; ++i) {
            const std::string name = gguf_get_tensor_name(shards[si].g, i);
            const bool take_exp = is_expert(name) || low_names.count(name) > 0;
            if (low_names.count(name)) low_unmatched.erase(name);
            ggml_context * src_gc = take_exp ? gc_exp : shards[si].gc;
            ggml_tensor * st = ggml_get_tensor(src_gc, name.c_str());
            if (!st) { std::cerr << "tensor missing in source: " << name << "\n"; return 1; }
            ggml_tensor * ot = ggml_new_tensor(gc_out, st->type, ggml_n_dims(st), st->ne);
            ggml_set_name(ot, name.c_str());
            gguf_add_tensor(g_out, ot);
            order.push_back({ name, take_exp, (int) si });
            if (take_exp) ++n_exp; else ++n_rest;
        }
    }
    if (!low_unmatched.empty()) {
        for (const auto & nm : low_unmatched) std::cerr << "splice: --take-low tensor not found: " << nm << "\n";
        return 1;
    }
    std::cerr << "hybrid: experts<-" << experts_from.filename().string() << " (" << n_exp
              << " tensors" << (low_names.empty() ? "" : ", incl. " + std::to_string(low_names.size()) + " --take-low")
              << "), rest<-" << rest_from.filename().string()
              << (shards.size() > 1 ? " [" + std::to_string(shards.size()) + " shards]" : "")
              << " (" << n_rest << " tensors)\n";

    // Write metadata, then append each tensor's bytes from its chosen source file.
    if (!gguf_write_to_file(g_out, out_path.string().c_str(), /*only_meta*/ true)) {
        std::cerr << "failed to write meta: " << out_path << "\n"; return 1;
    }
    FILE * fout = fopen(out_path.string().c_str(), "r+b");
    FILE * f_exp = fopen(experts_from.string().c_str(), "rb");
    if (!fout || !f_exp) { std::cerr << "fopen failed\n"; return 1; }
    for (auto & sh : shards) {
        sh.f = fopen(sh.path.c_str(), "rb");
        if (!sh.f) { std::cerr << "fopen failed: " << sh.path << "\n"; return 1; }
    }
    io::seek64(fout, 0, SEEK_END);
    const int64_t base = io::tell64(fout);

    std::vector<char> buf(8u * 1024 * 1024);
    int64_t written = 0;
    for (int64_t i = 0; i < (int64_t) order.size(); ++i) {
        const Ent & e = order[i];
        gguf_context * sg = e.exp ? g_exp : shards[e.shard].g;
        FILE * sf         = e.exp ? f_exp : shards[e.shard].f;
        const int64_t sid = gguf_find_tensor(sg, e.name.c_str());
        const int64_t src_pos = (int64_t) gguf_get_data_offset(sg) + (int64_t) gguf_get_tensor_offset(sg, sid);
        const int64_t dst_pos = base + (int64_t) gguf_get_tensor_offset(g_out, i);
        const size_t  sz_out  = gguf_get_tensor_size(g_out, i);
        const size_t  sz_src  = gguf_get_tensor_size(sg, sid);
        if (sz_out != sz_src) { std::cerr << "size mismatch for " << e.name << ": " << sz_out << " vs " << sz_src << "\n"; return 1; }
        io::seek64(fout, dst_pos, SEEK_SET);
        io::seek64(sf, src_pos, SEEK_SET);
        size_t remaining = sz_src;
        while (remaining > 0) {
            const size_t chunk = std::min(remaining, buf.size());
            if (fread(buf.data(), 1, chunk, sf) != chunk) { std::cerr << "read short on " << e.name << "\n"; return 1; }
            if (fwrite(buf.data(), 1, chunk, fout) != chunk) { std::cerr << "write short on " << e.name << "\n"; return 1; }
            remaining -= chunk;
        }
        written += (int64_t) sz_src;
    }
    // Finalize the output file with an explicit error check: a silent short flush
    // here would corrupt a multi-GB model that otherwise "looks" written.
    bool ok = (ferror(fout) == 0);
    if (fclose(fout) != 0) ok = false;
    fclose(f_exp);
    for (auto & sh : shards) { if (sh.f) fclose(sh.f); gguf_free(sh.g); ggml_free(sh.gc); }
    gguf_free(g_out); gguf_free(g_exp);
    ggml_free(gc_out); ggml_free(gc_exp);
    if (!ok) { std::cerr << "error finalizing " << out_path << "\n"; return 1; }
    std::cerr << "hybrid written: " << out_path << " (" << written / (1024 * 1024) << " MiB of tensor data)\n";
    return 0;
}

// --- expert store v0 --------------------------------------------------

uint64_t expert_source_uuid(const fs::path & src_path) {
    ggml_context * gc = nullptr;
    gguf_context * g = open_meta(src_path, &gc);
    if (!g) return 0;
    uint64_t h = MOERT_STORE_FNV_SEED;
    const int64_t akey = gguf_find_key(g, "general.architecture");
    if (akey >= 0) {
        const char * a = gguf_get_val_str(g, akey);
        h = moert_store_fnv1a64(a, std::strlen(a), h);
    }
    const int64_t n = gguf_get_n_tensors(g);
    for (int64_t i = 0; i < n; ++i) {
        const std::string name = gguf_get_tensor_name(g, i);
        if (name.find("_exps") == std::string::npos) continue;
        const ggml_tensor * t = ggml_get_tensor(gc, name.c_str());
        h = moert_store_fnv1a64(name.data(), name.size(), h);
        const int32_t ty = (int32_t) t->type;
        h = moert_store_fnv1a64(&ty, sizeof(ty), h);
        h = moert_store_fnv1a64(t->ne, sizeof(t->ne), h);
        const uint64_t off = gguf_get_tensor_offset(g, i);
        h = moert_store_fnv1a64(&off, sizeof(off), h);
    }
    gguf_free(g);
    ggml_free(gc);
    return h ? h : 1;   // 0 means "no check" downstream — never return it for a valid file
}

int make_expert_store(const fs::path & src_path, const fs::path & out_index, const fs::path & out_pack) {
    // v0: single-file sources only (the Q4 promotion source is single-file; the split-shard
    // merge logic lives in q4stream and can move here if a split source ever needs a store).
    {
        const std::string s = src_path.string();
        const size_t p = s.rfind("-of-");
        if (p != std::string::npos && p >= 6 && s[p - 6] == '-') {
            std::cerr << "make_expert_store: split GGUF sources are not supported in v0: " << src_path << "\n";
            return 1;
        }
    }
    ggml_context * gc = nullptr;
    gguf_context * g = open_meta(src_path, &gc);
    if (!g) return 1;

    // geometry + arch metadata (the same probes the runtime uses)
    int n_layers = 0, n_expert = 0;
    const int64_t n = gguf_get_n_tensors(g);
    for (int64_t i = 0; i < n; ++i) {
        const std::string name = gguf_get_tensor_name(g, i);
        const int L = blk_index(name);
        if (L + 1 > n_layers) n_layers = L + 1;
        if (name.find("_exps.weight") != std::string::npos) {
            ggml_tensor * t = ggml_get_tensor(gc, name.c_str());
            if (t && (int) t->ne[2] > n_expert) n_expert = (int) t->ne[2];
        }
    }
    if (n_layers <= 0 || n_expert <= 0) {
        std::cerr << "make_expert_store: no expert tensors in " << src_path << "\n";
        gguf_free(g); ggml_free(gc);
        return 1;
    }
    std::string arch = "unknown";
    if (const int64_t akey = gguf_find_key(g, "general.architecture"); akey >= 0) {
        arch = gguf_get_val_str(g, akey);
    }
    auto kv_u64 = [&](const std::string & key) -> uint64_t {
        const int64_t id = gguf_find_key(g, key.c_str());
        if (id < 0) return 0;
        switch (gguf_get_kv_type(g, id)) {
            case GGUF_TYPE_UINT32: return gguf_get_val_u32(g, id);
            case GGUF_TYPE_INT32:  return (uint64_t) gguf_get_val_i32(g, id);
            case GGUF_TYPE_UINT64: return gguf_get_val_u64(g, id);
            case GGUF_TYPE_INT64:  return (uint64_t) gguf_get_val_i64(g, id);
            default:               return 0;
        }
    };

    // collect the page groups; enforce the shape convention the store encodes
    // (weight: 3-D, expert axis 2; per-expert scale: 1-D, expert axis 0)
    struct Grp { int L, kind, ws; ggml_type type; uint64_t bpe; int64_t s0, s1; int64_t src_off; std::string name; };
    std::vector<Grp> grps;
    bool shape_ok = true;
    for (int64_t i = 0; i < n && shape_ok; ++i) {
        const std::string name = gguf_get_tensor_name(g, i);
        if (name.find("_exps") == std::string::npos) continue;
        const int L = blk_index(name);
        if (L < 0 || L >= n_layers) continue;
        const int kind = (name.find("gate_up_exps") != std::string::npos) ? 0
                       : (name.find("down_exps")    != std::string::npos) ? 1
                       : (name.find("up_exps")      != std::string::npos) ? 2
                       : (name.find("gate_exps")    != std::string::npos) ? 3 : -1;
        if (kind < 0) continue;
        ggml_tensor * st = ggml_get_tensor(gc, name.c_str());
        const bool ws = ends_with(name, ".scale");
        Grp gr;
        gr.L = L; gr.kind = kind; gr.ws = ws ? 1 : 0;
        gr.type = st->type;
        gr.name = name;
        gr.src_off = (int64_t) gguf_get_data_offset(g) + (int64_t) gguf_get_tensor_offset(g, i);
        if (ws) {
            if (st->ne[0] != n_expert || st->ne[1] != 1 || st->ne[2] != 1 || st->ne[3] != 1) {
                std::cerr << "make_expert_store: unsupported scale shape for " << name << "\n";
                shape_ok = false; break;
            }
            gr.bpe = st->nb[0];
            gr.s0 = gr.s1 = 1;
        } else {
            if (st->ne[2] != n_expert || st->ne[3] != 1) {
                std::cerr << "make_expert_store: unsupported weight shape for " << name << "\n";
                shape_ok = false; break;
            }
            gr.bpe = st->nb[2];
            gr.s0 = st->ne[0]; gr.s1 = st->ne[1];
        }
        grps.push_back(gr);
    }
    if (!shape_ok || grps.empty()) {
        if (grps.empty()) std::cerr << "make_expert_store: nothing to pack\n";
        gguf_free(g); ggml_free(gc);
        return 1;
    }

    // completeness gate (q4stream_init's): merged gate_up(0)/down(1) pairing per layer,
    // and scale presence uniform per kind across layers.
    {
        std::vector<std::array<bool, 8>> have(n_layers);
        for (auto & a : have) a.fill(false);
        bool scale_used[4] = {};
        for (const Grp & gr : grps) {
            have[gr.L][gr.kind * 2 + gr.ws] = true;
            if (gr.ws) scale_used[gr.kind] = true;
        }
        for (int L = 0; L < n_layers; ++L) {
            if (have[L][0] != have[L][2 * 1]) {   // gate_up weight vs down weight
                std::cerr << "make_expert_store: layer " << L << " has partial expert tensors (gate_up="
                          << have[L][0] << " down=" << have[L][2] << ") — incomplete source\n";
                gguf_free(g); ggml_free(gc);
                return 1;
            }
            for (int kind = 0; kind < 4; ++kind) {
                if (have[L][kind * 2] && scale_used[kind] && !have[L][kind * 2 + 1]) {
                    std::cerr << "make_expert_store: layer " << L << " kind " << kind
                              << " weight present but its scale is missing — incomplete source\n";
                    gguf_free(g); ggml_free(gc);
                    return 1;
                }
            }
        }
    }

    // group by layer, deterministic (kind, ws) order within a layer; page order below is
    // (layer, kind, ws, expert): every (layer, kind) weight group is one DENSE run of experts, so
    // the runtime can map the pack read-only and address expert e at base + e*bytes, exactly like
    // the stacked GGUF tensor (findings 196-197). Every per-expert page size of this model is a
    // multiple of the pack's 256-byte page alignment, so no padding breaks the run. Packs written
    // in the old expert-major order still load (copied into RAM instead of mapped).
    std::sort(grps.begin(), grps.end(), [](const Grp & a, const Grp & b) {
        return a.L != b.L ? a.L < b.L : (a.kind != b.kind ? a.kind < b.kind : a.ws < b.ws);
    });

    moert_store_info info{};
    info.source_uuid = expert_source_uuid(src_path);
    std::snprintf(info.arch, sizeof(info.arch), "%s", arch.c_str());
    info.n_layers = n_layers;
    info.n_expert = n_expert;
    info.n_embd   = kv_u64(arch + ".embedding_length");
    info.n_ff_exp = kv_u64(arch + ".expert_feed_forward_length");
    moert_store_writer * w = moert_store_writer_open(out_index.string().c_str(),
                                                     out_pack.string().c_str(), &info);
    if (!w) { gguf_free(g); ggml_free(gc); return 1; }

    FILE * fin = fopen(src_path.string().c_str(), "rb");
    if (!fin) {
        std::cerr << "make_expert_store: cannot open " << src_path << "\n";
        moert_store_writer_close(w);
        gguf_free(g); ggml_free(gc);
        return 1;
    }
    std::vector<char> buf;
    uint64_t written = 0;
    long long pages = 0;
    bool ok = true;
    size_t gi0 = 0;
    for (int L = 0; L < n_layers && ok; ++L) {
        size_t gi1 = gi0;
        while (gi1 < grps.size() && grps[gi1].L == L) ++gi1;
        for (size_t gi = gi0; gi < gi1 && ok; ++gi) {
            for (int e = 0; e < n_expert && ok; ++e) {
                const Grp & gr = grps[gi];
                buf.resize(gr.bpe);
                io::seek64(fin, gr.src_off + (int64_t) e * (int64_t) gr.bpe, SEEK_SET);
                if (fread(buf.data(), 1, gr.bpe, fin) != gr.bpe) {
                    std::cerr << "make_expert_store: read failed (" << gr.name << " e" << e << ")\n";
                    ok = false; break;
                }
                const int64_t shape[2] = { gr.s0, gr.s1 };
                if (!moert_store_writer_add_page(w, L, e, gr.kind, gr.ws,
                                                 (int32_t) gr.type, (int32_t) gr.type, 0,
                                                 shape, 2, buf.data(), gr.bpe)) {
                    std::cerr << "make_expert_store: write failed (" << gr.name << " e" << e << ")\n";
                    ok = false; break;
                }
                written += gr.bpe;
                ++pages;
            }
        }
        gi0 = gi1;
        if ((L + 1) % 5 == 0 || L + 1 == n_layers) {
            std::cerr << "make_expert_store: layer " << (L + 1) << "/" << n_layers << " ("
                      << (written >> 20) << " MiB)\n";
        }
    }
    fclose(fin);
    gguf_free(g); ggml_free(gc);
    if (!moert_store_writer_close(w)) ok = false;
    if (!ok) {
        std::cerr << "make_expert_store: FAILED — " << out_index << " / " << out_pack << " are not usable\n";
        return 1;
    }
    std::cerr << "expert store: " << pages << " pages, " << (written >> 20) << " MiB, "
              << n_layers << "L x " << n_expert << "E (" << arch << ", source_uuid "
              << std::hex << info.source_uuid << std::dec << ")\n  index: " << out_index
              << "\n  pack:  " << out_pack << "\n";
    return 0;
}

// Two-stage drafter head (claude dump findings 243/250/291). The basis Q is fit once on bench captures and shipped as a
// small file; E.Q (the approximate head) follows from the user's drafter. Same tensors and layout as the lab's
// tools/draft_head_export.py: draft_head_proj.weight F32 ne=[n_embd, r] (= Q), draft_head_low.weight Q8_0 ne=[r, vocab]
// (= E.Q), uint32 llapis.draft_head.candidates. The drafter's head must match the basis's fingerprint.
static uint64_t head_fingerprint(const uint8_t * p, size_t n) {   // sum_i w_i * (2 i + 1) over little-endian uint64 words
    uint64_t h = 0;
    const size_t nw = n / 8;
    for (size_t i = 0; i < nw; ++i) {
        uint64_t w; std::memcpy(&w, p + i * 8, 8);
        h += w * (2 * (uint64_t) i + 1);
    }
    return h;
}

int make_draft_head(const fs::path & drafter, const fs::path & basis, const fs::path & out_path) {
    // --- the basis
    FILE * fb = std::fopen(basis.string().c_str(), "rb");
    if (!fb) { std::cerr << "draft head: cannot open " << basis << "\n"; return 1; }
    char magic[4] = {}; int32_t hdr[4] = {}; uint64_t want[2] = {};
    bool ok = std::fread(magic, 1, 4, fb) == 4 && std::memcmp(magic, "LDHB", 4) == 0 &&
              std::fread(hdr, 4, 4, fb) == 4 && hdr[0] == 1 && std::fread(want, 8, 2, fb) == 2;
    const int64_t n_embd = hdr[1], r = hdr[2]; const uint32_t cand = (uint32_t) hdr[3];
    std::vector<float> Q;
    if (ok) { Q.resize((size_t) (r * n_embd)); ok = std::fread(Q.data(), 4, Q.size(), fb) == Q.size(); }
    std::fclose(fb);
    if (!ok || r <= 0 || n_embd <= 0 || r % 32 != 0) { std::cerr << "draft head: bad basis file " << basis << "\n"; return 1; }

    // --- the drafter, data loaded (~0.5 GiB)
    ggml_context * src_ctx = nullptr;
    gguf_init_params ip = { /*no_alloc*/ false, &src_ctx };
    gguf_context * g = gguf_init_from_file(drafter.string().c_str(), ip);
    if (!g || !src_ctx) { std::cerr << "draft head: cannot read " << drafter << "\n"; return 1; }
    ggml_tensor * E = ggml_get_tensor(src_ctx, "token_embd.weight");
    if (!E || E->ne[0] != n_embd || ggml_nbytes(E) != want[0] ||
        head_fingerprint((const uint8_t *) E->data, ggml_nbytes(E)) != want[1]) {
        std::cerr << "draft head: " << drafter << " is not the drafter the basis was fit on (head shape or bytes differ)"
                  << " -- the two-stage head is not built; the plain drafter keeps working\n";
        gguf_free(g); ggml_free(src_ctx); return 2;
    }
    if (gguf_find_key(g, "llapis.draft_head.candidates") >= 0 || ggml_get_tensor(src_ctx, "draft_head_low.weight")) {
        std::cerr << "draft head: " << drafter << " already is a derived drafter\n";
        gguf_free(g); ggml_free(src_ctx); return 1;
    }
    const int64_t vocab = E->ne[1];
    const ggml_type_traits * tr = ggml_get_type_traits(E->type);
    if (!tr || !tr->to_float) { std::cerr << "draft head: cannot dequantize the head (" << ggml_type_name(E->type) << ")\n"; gguf_free(g); ggml_free(src_ctx); return 1; }

    // --- E.Q in float, rows quantized to Q8_0 as they come (threads over row ranges)
    const size_t low_row = ggml_row_size(GGML_TYPE_Q8_0, r);
    std::vector<uint8_t> low((size_t) vocab * low_row);
    const int nt = (int) std::max(1u, std::min(16u, std::thread::hardware_concurrency()));
    std::vector<std::thread> th;
    for (int t = 0; t < nt; ++t) th.emplace_back([&, t]() {
        const int64_t lo = vocab * t / nt, hi = vocab * (t + 1) / nt;
        std::vector<float> e((size_t) n_embd), y((size_t) r);
        for (int64_t v = lo; v < hi; ++v) {
            tr->to_float((const char *) E->data + v * E->nb[1], e.data(), n_embd);
            for (int64_t k = 0; k < r; ++k) {
                const float * q = Q.data() + k * n_embd;
                float s = 0.0f;
                for (int64_t j = 0; j < n_embd; ++j) s += e[(size_t) j] * q[j];
                y[(size_t) k] = s;
            }
            ggml_quantize_chunk(GGML_TYPE_Q8_0, y.data(), low.data() + (size_t) v * low_row, 0, 1, r, nullptr);
        }
    });
    for (auto & x : th) x.join();

    // --- the new tensors, in their own context
    ggml_init_params cp = { /*mem_size*/ 2 * ggml_tensor_overhead() + 1024, /*mem_buffer*/ nullptr, /*no_alloc*/ true };
    ggml_context * nctx = ggml_init(cp);
    ggml_tensor * proj = ggml_new_tensor_2d(nctx, GGML_TYPE_F32, n_embd, r);
    ggml_set_name(proj, "draft_head_proj.weight");
    proj->data = Q.data();
    ggml_tensor * lowt = ggml_new_tensor_2d(nctx, GGML_TYPE_Q8_0, r, vocab);
    ggml_set_name(lowt, "draft_head_low.weight");
    lowt->data = low.data();

    gguf_context * go = gguf_init_empty();
    gguf_set_kv(go, g);
    for (int64_t i = 0; i < gguf_get_n_tensors(g); ++i)
        gguf_add_tensor(go, ggml_get_tensor(src_ctx, gguf_get_tensor_name(g, i)));
    gguf_add_tensor(go, proj);
    gguf_add_tensor(go, lowt);
    gguf_set_val_u32(go, "llapis.draft_head.candidates", cand);
    const bool wrote = gguf_write_to_file(go, out_path.string().c_str(), /*only_meta*/ false);
    gguf_free(go); ggml_free(nctx); gguf_free(g); ggml_free(src_ctx);
    if (!wrote) { std::cerr << "draft head: cannot write " << out_path << "\n"; return 1; }
    std::cerr << "draft head: " << out_path.filename().string() << " written (rank " << r << ", " << cand
              << " exact candidates per draft)\n";
    return 0;
}

}  // namespace gguf_tools
