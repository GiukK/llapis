// mtp_forecast.cpp -- see mtp_forecast.hpp.
//
// File format (tools/mtp_forecast_export.py, little endian): "LMTP" | int32 n_in, hid, n_out, H, n_test |
// float mu[n_in], sd[n_in], W1[n_in][hid], b1[hid], W2[hid][n_out], b2[n_out], a, c[n_out] |
// test vectors: n_test x (x[n_in], prior_logit[n_out], p[n_out]) checked at load.
#include "mtp_forecast.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <vector>

namespace mtp_forecast {
namespace {

constexpr int N_DRAFT = 3;

struct Net {
    bool on = false;
    int nl = 0, ne = 0, nin = 0, hid = 0, nout = 0, H = 0, wdraft = 0;
    std::vector<float> mu, sd, W1, b1, W2, b2, c;
    float a = 0.0f;
    std::mutex mu_draft;
    std::vector<float> work, snap;       // [N_DRAFT * wdraft]: drafts being made / drafts of the last verify
    std::vector<int32_t> pending;        // counts since the last close
    std::vector<float> ema;              // per-step usage EMA, half-life 4 steps
    float decay = 0.0f;
    std::vector<float> x, h, prior;
} g;

static void axpy(float * __restrict y, const float * __restrict w, float a, int n) {
    for (int j = 0; j < n; ++j) y[j] += a * w[j];
}

void run(const float * x_raw, const float * prior, float * p) {
    float * __restrict x = g.x.data(); float * __restrict h = g.h.data();
    const float * __restrict mu = g.mu.data(); const float * __restrict sd = g.sd.data();
    for (int i = 0; i < g.nin; ++i) x[i] = (x_raw[i] - mu[i]) / sd[i];
    std::copy(g.b1.begin(), g.b1.end(), g.h.begin());
    for (int i = 0; i < g.nin; ++i) {
        if (x[i] == 0.0f) continue;
        axpy(h, g.W1.data() + (size_t) i * g.hid, x[i], g.hid);
    }
    const float * __restrict b2 = g.b2.data(); const float * __restrict c = g.c.data();
    for (int j = 0; j < g.nout; ++j) p[j] = b2[j] + c[j] + g.a * prior[j];
    for (int k = 0; k < g.hid; ++k) {
        const float hk = h[k] > 0.0f ? h[k] : 0.0f;
        if (hk == 0.0f) continue;
        axpy(p, g.W2.data() + (size_t) k * g.nout, hk, g.nout);
    }
    for (int j = 0; j < g.nout; ++j) p[j] = 1.0f / (1.0f + std::exp(-p[j]));
}

template <class T> bool rd(FILE * f, T * dst, size_t n) { return std::fread(dst, sizeof(T), n, f) == n; }

}  // namespace

bool load(const char * path, int n_layers, int n_expert) {
    g.on = false;
    FILE * f = path ? std::fopen(path, "rb") : nullptr;
    if (!f) { std::fprintf(stderr, "mtp_forecast: cannot open %s\n", path ? path : "(null)"); return false; }
    char magic[4]; int32_t hdr[5];
    bool ok = rd(f, magic, 4) && std::memcmp(magic, "LMTP", 4) == 0 && rd(f, hdr, 5);
    if (ok) { g.nin = hdr[0]; g.hid = hdr[1]; g.nout = hdr[2]; g.H = hdr[3]; }
    ok = ok && g.nout == n_layers * n_expert && g.nin % N_DRAFT == 0 && g.hid > 0;
    if (ok) {
        g.nl = n_layers; g.ne = n_expert; g.wdraft = g.nin / N_DRAFT;
        g.mu.resize((size_t) g.nin); g.sd.resize((size_t) g.nin); g.W1.resize((size_t) g.nin * g.hid); g.b1.resize((size_t) g.hid);
        g.W2.resize((size_t) g.hid * g.nout); g.b2.resize((size_t) g.nout); g.c.resize((size_t) g.nout);
        ok = rd(f, g.mu.data(), g.mu.size()) && rd(f, g.sd.data(), g.sd.size()) && rd(f, g.W1.data(), g.W1.size()) &&
             rd(f, g.b1.data(), g.b1.size()) && rd(f, g.W2.data(), g.W2.size()) && rd(f, g.b2.data(), g.b2.size()) &&
             rd(f, &g.a, 1) && rd(f, g.c.data(), g.c.size());
    }
    g.x.assign((size_t) g.nin, 0.0f); g.h.assign((size_t) g.hid, 0.0f);
    double max_err = 0.0;
    if (ok) {   // the exporter's test vectors: C++ must match the Python reference
        std::vector<float> x((size_t) g.nin), pr((size_t) g.nout), want((size_t) g.nout), got((size_t) g.nout);
        for (int t = 0; t < hdr[4] && ok; ++t) {
            ok = rd(f, x.data(), x.size()) && rd(f, pr.data(), pr.size()) && rd(f, want.data(), want.size());
            if (!ok) break;
            run(x.data(), pr.data(), got.data());
            for (int j = 0; j < g.nout; ++j) max_err = std::max(max_err, (double) std::fabs(got[(size_t) j] - want[(size_t) j]));
        }
    }
    std::fclose(f);
    if (!ok || max_err > 1e-3) {
        std::fprintf(stderr, "mtp_forecast: %s rejected (%s, max |C++ - Python| = %.2e)\n", path,
                     ok ? "test vectors differ" : "bad format or geometry", max_err);
        return false;
    }
    g.work.assign((size_t) g.nin, 0.0f); g.snap.assign((size_t) g.nin, 0.0f);
    g.pending.assign((size_t) g.nout, 0); g.ema.assign((size_t) g.nout, 0.0f); g.prior.assign((size_t) g.nout, 0.0f);
    g.decay = std::pow(0.5f, 0.25f);
    g.on = true;
    std::fprintf(stderr, "mtp_forecast: %s loaded: %d inputs (%d drafts x %d), %d hidden, H = %d, C++ = Python to %.1e\n",
                 path, g.nin, N_DRAFT, g.wdraft, g.hid, g.H, max_err);
    return true;
}

bool active() { return g.on; }
int  horizon() { return g.H; }

void set_draft(int i, const float * h, int n) {
    if (!g.on || !h || n != g.wdraft || i < 0 || i >= N_DRAFT) return;
    std::lock_guard<std::mutex> lk(g.mu_draft);
    if (i == 0) std::fill(g.work.begin(), g.work.end(), 0.0f);   // a new drafting round
    std::memcpy(g.work.data() + (size_t) i * g.wdraft, h, (size_t) n * sizeof(float));
}

void snapshot(bool turn_start) {
    if (!g.on) return;
    std::lock_guard<std::mutex> lk(g.mu_draft);
    if (turn_start) std::fill(g.snap.begin(), g.snap.end(), 0.0f);
    else            g.snap = g.work;
}

void add_counts(const int32_t * counts) {
    if (!g.on || !counts) return;
    for (int j = 0; j < g.nout; ++j) g.pending[(size_t) j] += counts[j];
}

void close_step() {
    if (!g.on) return;
    for (int j = 0; j < g.nout; ++j) {
        g.ema[(size_t) j] = g.ema[(size_t) j] * g.decay + (g.pending[(size_t) j] > 0 ? 1.0f - g.decay : 0.0f);
        g.pending[(size_t) j] = 0;
    }
}

void close_prompt() {
    if (!g.on) return;
    std::fill(g.pending.begin(), g.pending.end(), 0);
}

double forecast(float * out) {
    if (!g.on || !out) return 0.0;
    const auto t0 = std::chrono::steady_clock::now();
    for (int j = 0; j < g.nout; ++j) {
        const float q = std::min(std::max(g.ema[(size_t) j], 1e-3f), 1.0f - 1e-3f);
        g.prior[(size_t) j] = std::log(q / (1.0f - q));
    }
    std::vector<float> snap;
    { std::lock_guard<std::mutex> lk(g.mu_draft); snap = g.snap; }
    run(snap.data(), g.prior.data(), out);
    for (int j = 0; j < g.nout; ++j) out[j] *= (float) g.H;
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

}  // namespace mtp_forecast
