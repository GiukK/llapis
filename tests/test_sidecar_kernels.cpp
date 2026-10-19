// The sidecar's decode-once kernels (src/sidecar_kernels.hpp) against ggml's own vec_dot:
//  (1) BIT-EXACT for N = 1..4 activations on random rows of the two expert types the product runs
//      (IQ2_XXS gate_up, d_in 2816; IQ4_NL down, d_in 704) -- fails the test on any difference;
//  (2) a timing of one realistic fire per type, single thread, weights far larger than the caches:
//      25 cold pairs from 4 tokens, about a third of them sharing an expert with another token
//      (findings 201), per-pair vec_dot (the old path) against grouped decode-once (the new one).
// No GPU, no model. `test-sidecar-kernels bench` prints the timing; without it only (1) runs.
#include "sidecar_kernels.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {

struct Shape { const char * name; ggml_type wt; int d_in, d_out; };

int check_exact(const Shape & sh, std::mt19937 & rng) {
    const auto * trw = ggml_get_type_traits_cpu(sh.wt);
    const ggml_type vt = trw->vec_dot_type;
    const auto * trq = ggml_get_type_traits_cpu(vt);
    const size_t wrow = ggml_row_size(sh.wt, sh.d_in), qrow = ggml_row_size(vt, sh.d_in);
    const int rows = 64;
    std::vector<uint8_t> w(wrow * rows);
    for (auto & b : w) b = (uint8_t) rng();
    // a sane fp16 scale at the head of every block (random bytes there could be inf/nan)
    const size_t bsz = ggml_type_size(sh.wt);
    std::uniform_int_distribution<int> sc(0x1C00, 0x2C00);
    for (size_t off = 0; off < w.size(); off += bsz) { const uint16_t d = (uint16_t) sc(rng); std::memcpy(&w[off], &d, 2); }
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> xf((size_t) 4 * sh.d_in);
    for (auto & v : xf) v = nd(rng);
    std::vector<char> xq(qrow * 4);
    for (int k = 0; k < 4; ++k) trq->from_float(xf.data() + (size_t) k * sh.d_in, xq.data() + k * qrow, sh.d_in);
    int bad = 0;
    for (int n = 1; n <= 4; ++n) {
        for (int r = 0; r < rows; ++r) {
            const char * wr = (const char *) w.data() + r * wrow;
            float ref[4], got[4];
            for (int k = 0; k < n; ++k) trw->vec_dot(sh.d_in, &ref[k], 0, wr, 0, xq.data() + k * qrow, 0, 1);
            if (sh.wt == GGML_TYPE_IQ2_XXS) {
                const sck::blk_q8_K * ys[4];
                for (int k = 0; k < 4; ++k) ys[k] = (const sck::blk_q8_K *) (xq.data() + k * qrow);
                const int nsb = sh.d_in / 256;
                if (n == 1) sck::iq2xxs_row_decodeN<1>(wr, ys, got, nsb);
                if (n == 2) sck::iq2xxs_row_decodeN<2>(wr, ys, got, nsb);
                if (n == 3) sck::iq2xxs_row_decodeN<3>(wr, ys, got, nsb);
                if (n == 4) sck::iq2xxs_row_decodeN<4>(wr, ys, got, nsb);
            } else {
                const sck::blk_q8_0 * ys[4];
                for (int k = 0; k < 4; ++k) ys[k] = (const sck::blk_q8_0 *) (xq.data() + k * qrow);
                const int nb = sh.d_in / 32;
                if (n == 1) sck::iq4nl_row_decodeN<1>(wr, ys, got, nb);
                if (n == 2) sck::iq4nl_row_decodeN<2>(wr, ys, got, nb);
                if (n == 3) sck::iq4nl_row_decodeN<3>(wr, ys, got, nb);
                if (n == 4) sck::iq4nl_row_decodeN<4>(wr, ys, got, nb);
            }
            for (int k = 0; k < n; ++k)
                if (std::memcmp(&ref[k], &got[k], sizeof(float)) != 0) {
                    if (bad < 5) std::fprintf(stderr, "  %s N=%d row %d act %d: ggml %.9g kernel %.9g\n", sh.name, n, r, k, ref[k], got[k]);
                    ++bad;
                }
        }
    }
    std::printf("%-8s bit-exact vs ggml vec_dot, N=1..4 x %d rows: %s\n", sh.name, rows, bad ? "FAIL" : "ok");
    return bad;
}

// One fire: pairs (expert, token). Old = every pair alone through vec_dot; new = grouped by expert.
void bench(const Shape & sh, std::mt19937 & rng) {
    const auto * trw = ggml_get_type_traits_cpu(sh.wt);
    const ggml_type vt = trw->vec_dot_type;
    const auto * trq = ggml_get_type_traits_cpu(vt);
    const size_t wrow = ggml_row_size(sh.wt, sh.d_in), qrow = ggml_row_size(vt, sh.d_in);
    const size_t eb = wrow * sh.d_out;
    const int n_exp = 128;
    std::vector<uint8_t> w(eb * n_exp);
    for (size_t i = 0; i < w.size(); i += 4) { const uint32_t v = (uint32_t) rng(); std::memcpy(&w[i], &v, 4); }
    const size_t bsz = ggml_type_size(sh.wt);
    for (size_t off = 0; off < w.size(); off += bsz) { const uint16_t d = 0x2400; std::memcpy(&w[off], &d, 2); }
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> xf((size_t) 4 * sh.d_in);
    for (auto & v : xf) v = nd(rng);
    std::vector<char> xq(qrow * 4);
    for (int k = 0; k < 4; ++k) trq->from_float(xf.data() + (size_t) k * sh.d_in, xq.data() + k * qrow, sh.d_in);
    std::vector<float> y((size_t) 32 * sh.d_out);

    const int fires = 60;
    double t_old = 0, t_new = 0;
    long long pairs_total = 0, groups_total = 0;
    std::uniform_int_distribution<int> pick(0, n_exp - 1), coin(0, 99);
    for (int f = 0; f < fires; ++f) {
        // 25 pairs over 4 tokens; ~1/3 reuse an expert another token already picked
        std::vector<int> pe, pt;
        while ((int) pe.size() < 25) {
            const int t = (int) pe.size() % 4;
            int e = pick(rng);
            if (!pe.empty() && coin(rng) < 34) e = pe[(size_t) pick(rng) % pe.size()];
            bool clash = false;   // top-k is distinct within one token
            for (size_t i = 0; i < pe.size(); ++i) if (pt[i] == t && pe[i] == e) clash = true;
            if (clash) continue;
            pe.push_back(e); pt.push_back(t);
        }
        std::vector<std::vector<int>> groups;
        for (size_t i = 0; i < pe.size(); ++i) {
            bool put = false;
            for (auto & g : groups) if (pe[(size_t) g[0]] == pe[i] && g.size() < 4) { g.push_back((int) i); put = true; break; }
            if (!put) groups.push_back({ (int) i });
        }
        pairs_total += (long long) pe.size(); groups_total += (long long) groups.size();
        // flush the caches between the two arms so both read from RAM: touch a 128 MiB buffer
        static std::vector<char> flush((size_t) 128 << 20, 1);
        auto do_flush = [&] { volatile char s = 0; for (size_t i = 0; i < flush.size(); i += 64) { flush[i]++; s ^= flush[i]; } };
        do_flush();
        auto t0 = std::chrono::steady_clock::now();
        for (size_t i = 0; i < pe.size(); ++i) {
            const char * we = (const char *) w.data() + (size_t) pe[i] * eb;
            for (int r = 0; r < sh.d_out; ++r)
                trw->vec_dot(sh.d_in, &y[i * sh.d_out + r], 0, we + r * wrow, 0, xq.data() + pt[i] * qrow, 0, 1);
        }
        auto t1 = std::chrono::steady_clock::now();
        do_flush();
        auto t2 = std::chrono::steady_clock::now();
        for (const auto & g : groups) {
            const char * we = (const char *) w.data() + (size_t) pe[(size_t) g[0]] * eb;
            const int n = (int) g.size();
            for (int r = 0; r < sh.d_out; ++r) {
                const char * wr = we + r * wrow;
                float outs[4];
                if (sh.wt == GGML_TYPE_IQ2_XXS) {
                    const sck::blk_q8_K * ys[4];
                    for (int k = 0; k < n; ++k) ys[k] = (const sck::blk_q8_K *) (xq.data() + pt[(size_t) g[(size_t) k]] * qrow);
                    const int nsb = sh.d_in / 256;
                    if (n == 1) sck::iq2xxs_row_decodeN<1>(wr, ys, outs, nsb);
                    else if (n == 2) sck::iq2xxs_row_decodeN<2>(wr, ys, outs, nsb);
                    else if (n == 3) sck::iq2xxs_row_decodeN<3>(wr, ys, outs, nsb);
                    else sck::iq2xxs_row_decodeN<4>(wr, ys, outs, nsb);
                } else {
                    const sck::blk_q8_0 * ys[4];
                    for (int k = 0; k < n; ++k) ys[k] = (const sck::blk_q8_0 *) (xq.data() + pt[(size_t) g[(size_t) k]] * qrow);
                    const int nb = sh.d_in / 32;
                    if (n == 1) sck::iq4nl_row_decodeN<1>(wr, ys, outs, nb);
                    else if (n == 2) sck::iq4nl_row_decodeN<2>(wr, ys, outs, nb);
                    else if (n == 3) sck::iq4nl_row_decodeN<3>(wr, ys, outs, nb);
                    else sck::iq4nl_row_decodeN<4>(wr, ys, outs, nb);
                }
                for (int k = 0; k < n; ++k) y[(size_t) g[(size_t) k] * sh.d_out + r] = outs[k];
            }
        }
        auto t3 = std::chrono::steady_clock::now();
        t_old += std::chrono::duration<double, std::micro>(t1 - t0).count();
        t_new += std::chrono::duration<double, std::micro>(t3 - t2).count();
    }
    const double mb_distinct = (double) groups_total * eb / 1e6;
    std::printf("%-8s 1 thread, %d fires, %.1f pairs/fire, %.1f%% pairs share an expert: "
                "old %.1f us/pair (%.1f GB/s distinct), new %.1f us/pair (%.1f GB/s) -> %.2fx\n",
                sh.name, fires, (double) pairs_total / fires, 100.0 * (pairs_total - groups_total) / pairs_total,
                t_old / pairs_total, mb_distinct / t_old * 1e3, t_new / pairs_total, mb_distinct / t_new * 1e3,
                t_old / t_new);
}

}  // namespace

int main(int argc, char ** argv) {
    ggml_cpu_init();
    sck::build_tables();
    if (!sck::g_tables_ok) { std::printf("no AVX2: skipped\n"); return 0; }
    std::mt19937 rng(1234);
    const Shape gate_up = { "gate_up", GGML_TYPE_IQ2_XXS, 2816, 1408 };
    const Shape down    = { "down",    GGML_TYPE_IQ4_NL,  704,  2816 };
    int bad = check_exact(gate_up, rng) + check_exact(down, rng);
    if (argc > 1 && std::strcmp(argv[1], "bench") == 0) { bench(gate_up, rng); bench(down, rng); }
    return bad ? 1 : 0;
}
