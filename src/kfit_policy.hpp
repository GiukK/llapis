#pragma once

// ===========================================================================
// kfit_policy — THE single home of "how many experts fit in VRAM right now?"
// ===========================================================================
// Header-only, no deps, no CUDA: eight numbers in, a K and three reasons out. Extracted
// 2026-08-11 under findings 133's criterion (a decision is extracted once it has already cost
// GPU time to verify). This one has cost more than any other: it is the arithmetic behind
// "K auto-clamped 72 -> 21", and a shipped version of it was ~5x too pessimistic, which took
// a whole campaign to notice because the only way to see K was to boot a server and ask.
//
// FOUR RULES, each one paid for. None of them is obvious, and all four are testable:
//
// 1. ONE FORMULA FOR BOTH REGIMES. The slab is either RESIDENT (its bytes are already inside
//    `free`, so the answer is a delta) or EVICTED by q4stream staging (it occupies nothing, so
//    free VRAM must cover the whole slab — an absolute target). The old code had two formulas
//    and the resident one truncated TWICE: once on the integer division, once by modelling the
//    current slab as `k * per_k` instead of its real byte count. The SAME physical situation
//    gave a different answer depending on whether the slab happened to be resident, and it
//    cost the boot K one to two slots every session. Here: pool = slab_now + free, divided
//    once.
//
// 2. DEVICE-FREE IS HALF THE CONSTRAINT. WDDM also caps what this process may keep on the
//    dedicated segment, and passing that cap does NOT fail an allocation — it silently demotes
//    pages to system RAM over PCIe. So the pool is also bounded by slab_now + headroom, and the
//    binding limit is whichever is smaller. Measured: the two disagree by over a gigabyte
//    whenever another process holds VRAM (1007 MiB on the reference box), which is exactly when
//    a device-free-only fit overcommits. headroom < 0 means DXGI was unavailable: fall back to
//    device-free alone.
//
// 3. THE QUALITY FLOOR IS A PREFERENCE; PHYSICS IS NOT. k_max_affordable is computed BEFORE the
//    floor is applied and caps it afterwards, because a real overcommit was measured here:
//    k_min = 4 forced a 429 MiB allocation when only 376 MiB was free. A floor that cannot be
//    afforded is a degraded cache, never a slab living on PCIe.
//
// 4. HYSTERESIS. Never rebuild for a single slot (~100 MiB) — except when the slab is evicted,
//    where the rebuild IS the restore and must happen at the right size.

namespace kfit {

struct inputs {
    long long free_mib      = 0;   // device-free VRAM, measured now
    long long slab_mib_now  = 0;   // what the slab holds right now; 0 when evicted
    long long headroom_mib  = -1;  // WDDM per-process headroom; < 0 = DXGI unavailable
    long long per_k_mib     = 1;   // VRAM cost of one expert slot across all layers
    long long keep_free_mib = 0;   // safety margin to leave unallocated
    int       k_cur         = 0;   // K the slab is built at
    int       k_min         = 4;   // quality floor (a preference, see rule 3)
    int       n_expert      = 128; // hard ceiling: cannot cache more experts than exist
    bool      evicted       = false;
};

struct decision {
    int       k          = 0;      // the K to fit to
    bool      rebuild    = false;  // false = hysteresis says leave the slab alone
    bool      wddm_bound = false;  // the WDDM budget was the binding limit (host logs this)
    bool      floor_capped = false;// the quality floor did not physically fit (host logs this)
    long long pool_mib   = 0;      // the VRAM the slab is allowed to occupy (host logs this)
};

inline decision fit(const inputs & in) {
    decision d;

    // Rule 1 + 2: the pool, one formula, bounded by whichever limit binds.
    d.pool_mib = in.slab_mib_now + in.free_mib;
    if (in.headroom_mib >= 0) {
        const long long budget_pool = in.slab_mib_now + in.headroom_mib;
        if (budget_pool < d.pool_mib) {
            d.pool_mib   = budget_pool;
            d.wddm_bound = true;
        }
    }

    const long long per_k = in.per_k_mib > 0 ? in.per_k_mib : 1;
    long long k_new = (d.pool_mib - in.keep_free_mib) / per_k;

    // Rule 3: physics is computed before the preference, and wins after it.
    const long long k_max_affordable = d.pool_mib / per_k;
    if (k_new < in.k_min)     k_new = in.k_min;
    if (k_new > in.n_expert)  k_new = in.n_expert;
    if (k_new > k_max_affordable) {
        k_new = k_max_affordable < 1 ? 1 : k_max_affordable;
        d.floor_capped = true;
    }
    d.k = (int) k_new;

    // Rule 4. (The original also had a separate `k_new == k_cur` early return; it is subsumed
    // by this window, which already covers k_cur-1, k_cur and k_cur+1. Same behaviour.)
    d.rebuild = in.evicted || d.k <= in.k_cur - 2 || d.k >= in.k_cur + 2;
    if (!d.rebuild) {
        d.k = in.k_cur;
    }
    return d;
}

} // namespace kfit
