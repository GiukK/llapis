// test-kfit-policy — how many experts fit in VRAM, on the CPU, in milliseconds.
//
// WHY THIS FILE EXISTS. This arithmetic sets K, and K is the product: the difference between
// K=22 and K=4 was measured at 33.5 vs 17.4 tok/s of decode (findings 131). A shipped version
// of it was ~5x too pessimistic and it took a campaign to notice, because the only way to see
// K was to boot a server and read a log line.
//
// Each block below is a rule that was paid for in GPU time. No GPU, no model, no server.
// Run it: ctest -R kfit-policy

#include "kfit_policy.hpp"

#include <cstdio>

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

// The reference box: RTX 4070 8 GB, ~107 MiB of VRAM per expert slot across 30 layers.
static kfit::inputs box() {
    kfit::inputs in;
    in.per_k_mib     = 107;
    in.keep_free_mib = 500;
    in.k_min         = 4;
    in.n_expert      = 128;
    in.headroom_mib  = -1;      // DXGI unavailable unless a test says otherwise
    return in;
}

int main() {
    std::printf("rule 1: ONE formula for both regimes\n");
    // The bug this replaced: the resident branch truncated twice and answered 30 where the
    // evicted branch answered 31 on the IDENTICAL 3828 MiB pool. Same physics, same K.
    {
        kfit::inputs res = box();                 // resident: slab bytes are inside `free`
        res.slab_mib_now = 3103; res.free_mib = 725; res.k_cur = 29; res.evicted = false;
        kfit::inputs evi = box();                 // evicted: the pool is all free VRAM
        evi.slab_mib_now = 0;    evi.free_mib = 3828; evi.k_cur = 29; evi.evicted = true;
        const auto a = kfit::fit(res);
        const auto b = kfit::fit(evi);
        check_eq(a.pool_mib, 3828, "resident pool = slab + free");
        check_eq(b.pool_mib, 3828, "evicted pool = free alone");
        check_eq(a.k, b.k, "SAME physical VRAM -> SAME K, whichever regime");
        check_eq(a.k, 31, "and it is the ceiling (3828-500)/107 = 31, not the old 30");
    }

    std::printf("rule 2: WDDM headroom is the other half of the constraint\n");
    {
        kfit::inputs in = box();
        in.slab_mib_now = 0; in.free_mib = 3828; in.k_cur = 4; in.evicted = true;
        in.headroom_mib = 2821;   // measured: the two disagree by ~1007 MiB when another process holds VRAM
        const auto d = kfit::fit(in);
        check(d.wddm_bound, "the tighter WDDM budget binds, and says so");
        check_eq(d.pool_mib, 2821, "pool is the WDDM budget, not device-free");
        check_eq(d.k, (2821 - 500) / 107, "K fits the budget: overcommit avoided");
    }
    {
        kfit::inputs in = box();
        in.slab_mib_now = 0; in.free_mib = 2000; in.k_cur = 4; in.evicted = true;
        in.headroom_mib = 3500;   // headroom looser than device-free: device-free binds
        const auto d = kfit::fit(in);
        check(!d.wddm_bound, "a looser budget does not bind");
        check_eq(d.pool_mib, 2000, "device-free stays the limit");
    }
    {
        kfit::inputs in = box();
        in.slab_mib_now = 0; in.free_mib = 2000; in.k_cur = 4; in.evicted = true;
        in.headroom_mib = -1;     // DXGI unavailable
        const auto d = kfit::fit(in);
        check(!d.wddm_bound && d.pool_mib == 2000,
              "headroom < 0 falls back to device-free alone, the pre-WDDM behaviour");
    }

    std::printf("rule 3: the quality floor is a preference; physics is not\n");
    {
        // The measured overcommit: k_min = 4 forced a 429 MiB allocation with 376 MiB free.
        kfit::inputs in = box();
        in.slab_mib_now = 0; in.free_mib = 376; in.k_cur = 8; in.evicted = true;
        const auto d = kfit::fit(in);
        check(d.floor_capped, "the floor did not fit, and the decision says so");
        check_eq(d.k, 376 / 107, "capped to what physically fits (3), not the floor (4)");
        check(d.k < in.k_min, "a degraded cache is the answer, never a slab on PCIe");
    }
    {
        kfit::inputs in = box();
        in.slab_mib_now = 0; in.free_mib = 900; in.k_cur = 8; in.evicted = true;
        const auto d = kfit::fit(in);   // (900-500)/107 = 3 -> floor lifts to 4, and 4 fits (900/107=8)
        check(!d.floor_capped, "when the floor fits, it applies");
        check_eq(d.k, 4, "the safety margin yields to the quality floor when physics allows");
    }
    {
        kfit::inputs in = box();
        in.slab_mib_now = 0; in.free_mib = 999999; in.k_cur = 4; in.evicted = true;
        check_eq(kfit::fit(in).k, 128, "cannot cache more experts than exist");
    }

    std::printf("rule 4: hysteresis -- never rebuild for one slot\n");
    {
        kfit::inputs in = box();
        in.slab_mib_now = 2247; in.free_mib = 500; in.k_cur = 21; in.evicted = false;
        for (int delta = -1; delta <= 1; ++delta) {
            kfit::inputs t = in;
            t.k_cur = (int) ((t.slab_mib_now + t.free_mib - t.keep_free_mib) / t.per_k_mib) - delta;
            const auto d = kfit::fit(t);
            check(!d.rebuild && d.k == t.k_cur, "within +/-1 slot: no rebuild, K unchanged");
        }
        kfit::inputs far = in;
        far.k_cur = 10;
        check(kfit::fit(far).rebuild, "two slots or more: rebuild");
    }
    {
        // The exception that matters: when the slab is EVICTED the rebuild IS the restore, so
        // it must happen even if K is unchanged -- otherwise the slab never comes back.
        kfit::inputs in = box();
        in.slab_mib_now = 0; in.free_mib = 3828; in.evicted = true;
        in.k_cur = (int) ((3828 - 500) / 107);          // exactly the K it would compute
        const auto d = kfit::fit(in);
        check(d.rebuild, "EVICTED always rebuilds, even at an unchanged K (the rebuild is the restore)");
    }

    std::printf("the regression this file exists to prevent\n");
    // findings 131: --swa-full pushed the CUDA reserve 560 -> 1910 MiB and K fell 22 -> 4,
    // taking decode from 33.5 to 17.4 tok/s. The arithmetic was right; the VRAM was gone. This
    // asserts the collapse is a faithful reading of the pool, not a bug in the fit.
    {
        kfit::inputs in = box();
        in.slab_mib_now = 0; in.free_mib = 2850; in.k_cur = 22; in.evicted = true;
        check(kfit::fit(in).k >= 20, "with 2850 MiB of pool, K stays in the twenties");
        in.free_mib = 950;                                   // what swa-full left
        const auto d = kfit::fit(in);
        check(d.k <= 5, "with 950 MiB it collapses to <= 5 -- the measured 22 -> 4");
        check(d.rebuild, "and it does rebuild rather than pretend");
    }

    std::printf(g_fail == 0 ? "\nPASS\n" : "\nFAIL: %d check(s)\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
