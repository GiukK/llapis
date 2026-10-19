// bench-sidecar: the CPU cold executor end to end (src/moe_sidecar.cpp: pool, tiles, kernels), on
// fires shaped like an MTP decode step -- 30 layers x (gate_up, down), ~25 cold pairs from 4 tokens per
// fire, about a third of them sharing an expert -- with weights far larger than the caches and a short
// busy gap between fires standing in for the GPU work (workers go back to sleep, as in the product).
// Not a test: a measuring tool for the sidecar alone (findings 202).
//   bench-sidecar [workers=10] [steps=40] [gap_us=150] [cold|repeat]      BENCH_NCOLD=<pairs per fire> (25)
// repeat (findings 214, gate G0): every fire runs twice and only the second is timed -- the same weights, now in
// the caches. cold vs repeat bounds how much of the cold path waits on RAM (the value of a cache prefetch).
// pfdown (gate G1): while the gate_up fire computes, two helper threads (one per CCD, on the last logical CPU of
// each) load the layer's DOWN weights -- the down cold set is the gate_up cold set -- so the down fire finds them
// in the L3 of both CCDs. The gate_up time includes waiting for the helpers to finish.
// dmacon / dmagap (findings 216, the "RAM conveyor"): DMA copies of whole experts (gate_up + down, 2.14 MB) from a
// separate registered host arena to VRAM, BENCH_DMA_N per layer (default 4). dmacon enqueues them right before the
// gate_up fire, so they run while the CPU computes (the staging prototype of 213g); dmagap runs them inside the gap
// before the gate_up fire and waits for them there (the idle window), then finishes the gap. Do copies cost the CPU
// cold path because they overlap it (RAM contention: dmagap ~ free) or regardless of timing (power: dmagap ~ dmacon)?
#include "moe_sidecar.hpp"

#include "ggml.h"
#include "ggml-cpu.h"

#if defined(BENCH_DMA)
#include <cuda_runtime.h>
#endif
#include <atomic>
#include <immintrin.h>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <mutex>
#include <thread>
#if defined(_WIN32)
#include <windows.h>
#endif
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {
using clk = std::chrono::steady_clock;

// helpers that load a list of expert pages (one 8-byte load per cache line) on request
struct Helper {
    std::thread th;
    std::mutex mu; std::condition_variable cv;
    long long gen = 0, done = 0;
    bool quit = false;
    const char * base = nullptr; size_t eb = 0; const int32_t * ids = nullptr; int n = 0;
    volatile uint64_t sink = 0;
    int part = 0, parts = 1;   // this helper loads experts part, part + parts, ...
    void start(int cpu) {
        th = std::thread([this, cpu] {
#if defined(_WIN32)
            SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR) 1 << cpu);
#endif
            long long seen = 0;
            for (;;) {
                std::unique_lock<std::mutex> lk(mu);
                cv.wait(lk, [&] { return quit || gen != seen; });
                if (quit) return;
                seen = gen;
                lk.unlock();
                uint64_t a0 = 0, a1 = 0, a2 = 0, a3 = 0, a4 = 0, a5 = 0, a6 = 0, a7 = 0;
                for (int i = part; i < n; i += parts) {
                    const char * p = base + (size_t) ids[i] * eb;
                    size_t off = 0;
                    for (; off + 512 <= eb; off += 512) {   // 8 independent cache-line loads in flight
                        a0 += *(const uint64_t *) (p + off);       a1 += *(const uint64_t *) (p + off + 64);
                        a2 += *(const uint64_t *) (p + off + 128); a3 += *(const uint64_t *) (p + off + 192);
                        a4 += *(const uint64_t *) (p + off + 256); a5 += *(const uint64_t *) (p + off + 320);
                        a6 += *(const uint64_t *) (p + off + 384); a7 += *(const uint64_t *) (p + off + 448);
                    }
                    for (; off < eb; off += 64) a0 += *(const uint64_t *) (p + off);
                }
                sink = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
                lk.lock(); done = seen; lk.unlock(); cv.notify_all();
            }
        });
    }
    void go(const char * b, size_t e, const int32_t * i, int k) {
        { std::lock_guard<std::mutex> lk(mu); base = b; eb = e; ids = i; n = k; ++gen; }
        cv.notify_all();
    }
    void wait() { std::unique_lock<std::mutex> lk(mu); cv.wait(lk, [&] { return done == gen; }); }
    void stop() { { std::lock_guard<std::mutex> lk(mu); quit = true; } cv.notify_all(); th.join(); }
};
void spin_us(double us) { const auto t0 = clk::now(); while (std::chrono::duration<double, std::micro>(clk::now() - t0).count() < us) {} }
}

int main(int argc, char ** argv) {
    const int workers = argc > 1 ? std::atoi(argv[1]) : 10;
    const int steps   = argc > 2 ? std::atoi(argv[2]) : 40;
    const double gap  = argc > 3 ? std::atof(argv[3]) : 150.0;
    const bool repeat = argc > 4 && std::strcmp(argv[4], "repeat") == 0;
    // pfdown: every CCD loads ALL the down experts (2 helpers per CCD split them); pfsplit: each CCD loads half
    const bool pfded = argc > 4 && std::strcmp(argv[4], "pfded") == 0;   // 2 helpers on dedicated cores, half each
    // pfw1 (findings 214): the down weights are loaded INSIDE the gap before the gate_up fire (the window in which
    // the GPU runs attention and the CPU is idle), by 4 helpers (a quarter each); the gap is still waited in full
    const bool pfw1 = argc > 4 && std::strcmp(argv[4], "pfw1") == 0;
    const bool pfsplit = pfded || pfw1 || (argc > 4 && std::strcmp(argv[4], "pfsplit") == 0);
    const bool pfdown = pfsplit || pfw1 || (argc > 4 && std::strcmp(argv[4], "pfdown") == 0);
    const bool dmacon = argc > 4 && std::strcmp(argv[4], "dmacon") == 0;
    const bool dmagap = argc > 4 && std::strcmp(argv[4], "dmagap") == 0;
    // subst (claude dump findings 224, promotion by substitution): the first BENCH_SUB_M distinct experts of each
    // layer's cold list leave the CPU fires (all their pairs) and are copied whole (gate_up + down, from the same weight
    // arrays, registered) to VRAM, issued at the start of the gate_up fire and waited after the down fire: is the copy
    // hidden under the CPU work it removes? subcpu: the same reduced CPU fires, no copy (the ideal); subdup: copies of
    // those experts WHILE the CPU still computes all of them (the paid copy of today's promotions).
    const bool subdma = argc > 4 && std::strcmp(argv[4], "subdma") == 0;
    const bool subcpu = argc > 4 && std::strcmp(argv[4], "subcpu") == 0;
    const bool subdup = argc > 4 && std::strcmp(argv[4], "subdup") == 0;
    const bool subany = subdma || subcpu || subdup;
    const int sub_m = std::getenv("BENCH_SUB_M") ? std::atoi(std::getenv("BENCH_SUB_M")) : 2;
    const double gap2 = std::getenv("BENCH_GAP2") ? std::atof(std::getenv("BENCH_GAP2")) : gap;   // gap before down
    double sub_wait_ms = 0; long long sub_bytes = 0, sub_removed = 0;
    const bool dma = dmacon || dmagap || subdma || subdup;
    const int ncold = std::getenv("BENCH_NCOLD") ? std::atoi(std::getenv("BENCH_NCOLD")) : 25;   // cold pairs per fire (findings 218b)
    const int dma_n = std::getenv("BENCH_DMA_N") ? std::atoi(std::getenv("BENCH_DMA_N")) : 4;
    double dma_ms = 0; long long dma_bytes = 0;
#if defined(BENCH_DMA)
    const size_t ex_bytes = (size_t) 2140000, arena_n = 256;   // 256 experts = ~0.55 GB source, beyond any cache
    std::vector<uint8_t> arena;
    void * dbuf = nullptr; cudaStream_t cs = nullptr; cudaEvent_t ce = nullptr;
    if (dma) {
        arena.resize(ex_bytes * arena_n);
        for (size_t i = 0; i < arena.size(); i += 4096) arena[i] = (uint8_t) i;
        if (cudaHostRegister(arena.data(), arena.size(), cudaHostRegisterDefault) != cudaSuccess) { std::printf("register failed\n"); return 1; }
        if (cudaMalloc(&dbuf, ex_bytes * 16) != cudaSuccess) { std::printf("cudaMalloc failed\n"); return 1; }
        cudaStreamCreateWithFlags(&cs, cudaStreamNonBlocking);
        cudaEventCreateWithFlags(&ce, cudaEventDisableTiming);
    }
    std::uniform_int_distribution<int> pick_src(0, (int) arena_n - 1);
    std::mt19937 rng_dma(11);
    auto dma_issue = [&](int il) {   // dma_n whole experts from random arena slots into a 16-slot device ring
        for (int i = 0; i < dma_n; ++i) {
            const size_t src = (size_t) pick_src(rng_dma) * ex_bytes;
            cudaMemcpyAsync((char *) dbuf + (size_t) ((il * dma_n + i) % 16) * ex_bytes, arena.data() + src, ex_bytes, cudaMemcpyHostToDevice, cs);
            dma_bytes += (long long) ex_bytes;
        }
        cudaEventRecord(ce, cs);
    };
    auto dma_wait = [&]() { cudaEventSynchronize(ce); };
#else
    if (dma) { std::printf("built without BENCH_DMA (no CUDA)\n"); return 1; }
    auto dma_issue = [&](int) {};
    auto dma_wait = [&]() {};
#endif
    Helper helpers[4];
    const int hcpu[4] = { 9, 11, 21, 23 };   // two SMT siblings on each CCD
    if (pfded) {   // one helper per CCD on the first logical CPU of its last core (cores 5 and 11)
        helpers[0].start(10); helpers[0].part = 0; helpers[0].parts = 2;
        helpers[1].start(22); helpers[1].part = 1; helpers[1].parts = 2;
    } else if (pfdown)
        for (int h = 0; h < 4; ++h) {
            helpers[h].start(hcpu[h]);
            if (pfsplit) { helpers[h].part = h; helpers[h].parts = 4; }           // the 4 quarters, 2 per CCD
            else         { helpers[h].part = h % 2; helpers[h].parts = 2; }       // each CCD: both halves
        }
    ggml_cpu_init();
    moe_sidecar::init(workers);

    struct Kind { ggml_type t; int d_in, d_out; size_t eb; std::vector<uint8_t> w[4]; };
    Kind kinds[2] = { { GGML_TYPE_IQ2_XXS, 2816, 1408, 0, {} }, { GGML_TYPE_IQ4_NL, 704, 2816, 0, {} } };
    std::mt19937 rng(7);
    for (auto & k : kinds) {
        k.eb = ggml_row_size(k.t, k.d_in) * k.d_out;
        const size_t bsz = ggml_type_size(k.t);
        for (auto & w : k.w) {   // 4 distinct "layers" of 128 experts each: ~0.5 GiB per kind, far beyond L3
            w.resize(k.eb * 128);
            for (size_t i = 0; i + 4 <= w.size(); i += 4) { const uint32_t v = (uint32_t) rng(); std::memcpy(&w[i], &v, 4); }
            for (size_t off = 0; off < w.size(); off += bsz) { const uint16_t d = 0x2400; std::memcpy(&w[off], &d, 2); }
        }
    }
#if defined(BENCH_DMA)
    if (subdma || subdup)
        for (auto & k : kinds)
            for (auto & w : k.w)
                if (cudaHostRegister(w.data(), w.size(), cudaHostRegisterDefault) != cudaSuccess) { std::printf("register failed\n"); return 1; }
    auto sub_issue = [&](int il, const int32_t * ex, int m) {   // whole experts: gate_up then down, one call each
        for (int i = 0; i < m; ++i)
            for (int kk = 0; kk < 2; ++kk) {
                const Kind & k = kinds[kk];
                cudaMemcpyAsync((char *) dbuf + (size_t) ((i * 2 + kk) % 16) * ex_bytes, k.w[il % 4].data() + (size_t) ex[i] * k.eb,
                                k.eb, cudaMemcpyHostToDevice, cs);
                sub_bytes += (long long) k.eb;
            }
        cudaEventRecord(ce, cs);
    };
#else
    auto sub_issue = [&](int, const int32_t *, int) {};
#endif
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> x((size_t) 32 * 2816), y((size_t) 32 * 2816);
    for (auto & v : x) v = nd(rng);

    std::uniform_int_distribution<int> pick(0, 127), coin(0, 99);
    double ms[2] = { 0, 0 }, ms_wait = 0; long long pairs[2] = { 0, 0 }, distinct[2] = { 0, 0 };
    const auto t_all = clk::now();
    for (int s = 0; s < steps; ++s) {
        for (int il = 0; il < 30; ++il) {
            // one cold list per layer, shared by its two fires (same routing, as in the product)
            int32_t ids[32]; int n = 0;
            int tok[32];
            while (n < ncold) {
                const int t = n % 4;
                int e = pick(rng);
                if (n && coin(rng) < 34) e = ids[pick(rng) % n];
                bool clash = false;
                for (int i = 0; i < n; ++i) if (tok[i] == t && ids[i] == e) clash = true;
                if (clash) continue;
                ids[n] = e; tok[n] = t; ++n;
            }
            int nd_ = 0;
            int32_t dist[32];
            for (int i = 0; i < n; ++i) { bool seen = false; for (int j = 0; j < i; ++j) if (ids[j] == ids[i]) seen = true; if (!seen) dist[nd_++] = ids[i]; }
            // substitution: the CPU list without the pairs of the first m distinct experts
            int32_t cids[32]; int cn = n;
            const int m = subany ? (sub_m < nd_ - 1 ? sub_m : nd_ - 1) : 0;
            if (subcpu || subdma) {
                cn = 0;
                for (int i = 0; i < n; ++i) { bool out = false; for (int j = 0; j < m; ++j) if (ids[i] == dist[j]) out = true; if (!out) cids[cn++] = ids[i]; }
                sub_removed += n - cn;
            } else std::memcpy(cids, ids, sizeof(ids));
            for (int kk = 0; kk < 2; ++kk) {
                Kind & k = kinds[kk];
                moert_coldcpu_call c = {};
                c.il = il; c.kind = kk; c.n_cold = cn; c.cold_ids = cids;
                c.x = x.data(); c.x_per_pair = 1;
                c.w_host = k.w[il % 4].data(); c.expert_bytes = k.eb; c.stride = k.eb;
                c.w_type_tag = (int32_t) k.t; c.d_in = k.d_in; c.d_out = k.d_out; c.y = y.data();
                if (dmagap && kk == 0) {   // the copies inside the idle gap, waited there; then finish the gap
                    const auto tg = clk::now();
                    dma_issue(il); dma_wait();
                    const double used = std::chrono::duration<double, std::micro>(clk::now() - tg).count();
                    dma_ms += used / 1000.0;
                    if (used < gap) spin_us(gap - used);
                } else if (pfw1 && kk == 0) {   // load the layer's down weights during the idle gap, then finish the gap
                    const auto tg = clk::now();
                    for (int h = 0; h < 4; ++h) helpers[h].go((const char *) kinds[1].w[il % 4].data(), kinds[1].eb, dist, nd_);
                    for (int h = 0; h < 4; ++h) helpers[h].wait();
                    const double used = std::chrono::duration<double, std::micro>(clk::now() - tg).count();
                    ms_wait += used / 1000.0;
                    if (used < gap) spin_us(gap - used);
                } else {
                    spin_us(kk == 1 ? gap2 : gap);
                }
                if (repeat) moe_sidecar_compute(&c);   // warm the caches with this fire's weights (untimed)
                const auto t0 = clk::now();
                if (dmacon && kk == 0) dma_issue(il);   // copies run while the CPU computes (not waited in the fire)
                if ((subdma || subdup) && kk == 0 && m > 0) sub_issue(il, dist, m);
                if (pfdown && !pfw1 && kk == 0)   // load this layer's down weights while gate_up computes
                    for (int h = 0; h < (pfded ? 2 : 4); ++h) helpers[h].go((const char *) kinds[1].w[il % 4].data(), kinds[1].eb, dist, nd_);
                moe_sidecar_compute(&c);
                if (pfdown && !pfw1 && kk == 0) { const auto tw = clk::now(); for (int h = 0; h < (pfded ? 2 : 4); ++h) helpers[h].wait();
                                         ms_wait += std::chrono::duration<double, std::milli>(clk::now() - tw).count(); }
                ms[kk] += std::chrono::duration<double, std::milli>(clk::now() - t0).count();
                if (dmacon && kk == 1) { const auto tw = clk::now(); dma_wait(); dma_ms += std::chrono::duration<double, std::milli>(clk::now() - tw).count(); }
                if ((subdma || subdup) && kk == 1 && m > 0) { const auto tw = clk::now(); dma_wait(); sub_wait_ms += std::chrono::duration<double, std::milli>(clk::now() - tw).count(); }
                pairs[kk] += cn; distinct[kk] += nd_ - ((subcpu || subdma) ? m : 0);
            }
        }
    }
    const double wall = std::chrono::duration<double>(clk::now() - t_all).count();
    const char * kn[2] = { "gate_up", "down" };
    if (dma) std::printf("DMA %s: %d experts/layer, %.1f MB/layer; %s %.1f us/layer, %.2f GB/s over the run\n",
                         dmacon ? "CONCURRENT with the cold fires" : "in the GAP before gate_up", dma_n,
                         (double) dma_bytes / (steps * 30) / 1e6, dmacon ? "residual wait after down" : "copy time in the gap",
                         1000.0 * dma_ms / (steps * 30), (double) dma_bytes / 1e9 / std::chrono::duration<double>(clk::now() - t_all).count());
    std::printf("mode: %s\n", repeat ? "REPEAT (each fire's weights already in the caches)" :
                              pfsplit ? "PFSPLIT (down weights: half per CCD, 2 helpers each, during gate_up)" :
                              pfdown ? "PFDOWN (down weights: all in both CCDs, 2 helpers each, during gate_up)" :
                                       "cold (weights in RAM)");
    if (pfdown) { std::printf(pfw1 ? "  helper load time inside the gap: %.1f us/layer (not timed)\n" : "  helper wait after gate_up: %.1f us/call (included in gate_up)\n", 1000.0 * ms_wait / (steps * 30));
                  for (int h = 0; h < (pfded ? 2 : 4); ++h) helpers[h].stop(); }
    std::printf("workers %d, %d steps x 30 layers, gap %.0f us, %d cold pairs per fire\n", workers, steps, gap, ncold);
    for (int kk = 0; kk < 2; ++kk)
        std::printf("  %-7s %.1f us/call  %.1f us/pair  %.1f GB/s distinct  (%.1f%% shared)\n", kn[kk],
                    1000.0 * ms[kk] / (steps * 30), 1000.0 * ms[kk] / pairs[kk],
                    (double) distinct[kk] * kinds[kk].eb / (ms[kk] * 1e6),
                    100.0 * (pairs[kk] - distinct[kk]) / pairs[kk]);
    std::printf("  cold CPU per step %.2f ms (wall per step incl. gaps %.2f ms)\n", (ms[0] + ms[1]) / steps, 1000.0 * wall / steps);
    if (subany)
        std::printf("SUBST %s m=%d: %.2f distinct cold experts/layer before, %.2f pairs/layer moved off the CPU, %.2f MB/layer copied; "
                    "residual copy wait after down %.1f us/layer; CRITICAL per step (gate_up + down + residual wait) %.2f ms\n",
                    argv[4], sub_m, (double) (distinct[0] + ((subcpu || subdma) ? 0 : 0)) / (steps * 30), (double) sub_removed / (steps * 30),
                    (double) sub_bytes / (steps * 30) / 1e6, 1000.0 * sub_wait_ms / (steps * 30), (ms[0] + ms[1] + sub_wait_ms) / steps);
    moe_sidecar::shutdown();
    return 0;
}
