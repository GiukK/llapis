// moert policy — implementation; see policy.h for the contract.
// NO ggml / CUDA / llama includes here — standard library only.
//
// One rule since elision E3 (2026-10-05): the cost-aware pairing decided at every visible token
// (findings §173). The V1 planner it replaced (score ratio + margin, min-residency, cooldown, layer
// cap, refresh budgets), the V2 provisional-seed / no-blind-decay slices and the decision-telemetry
// sink live at the tag pre-e3-2026-10-05.
#include "moert/policy.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <vector>

namespace {
constexpr int32_t SLOT_FREE    = MOERT_SLOT_FREE;
constexpr int32_t SLOT_READY   = MOERT_SLOT_READY;
constexpr int32_t SLOT_FILLING = MOERT_SLOT_FILLING;
}

struct moert_policy {
    moert_policy_config    cfg{};
    moert_policy_callbacks cbs{};
    std::vector<int32_t>   kl;   // per-layer hot capacity (all == cfg.k for the uniform path)

    bool external  = false;   // device/external observation: per-token folds skipped
    long long dev_since_fold = 0;

    // scores
    std::vector<std::vector<float>> cnt;            // [il][e] routes seen since the last fold
    std::vector<std::vector<float>> session_score;  // [il][e] slow EMA  } the resize ranking
    std::vector<std::vector<float>> recent_score;   // [il][e] fast EMA  }
    std::vector<std::vector<float>> rate_score;     // [il][e] per-visible-token use rate (the forecast)
    long long visible_since_fold = 0;               // external observation: visible steps the pending counts cover

    // page table + device-mirror sources
    std::vector<std::vector<int32_t>> slot_exp;     // [il][slot] expert | -1
    std::vector<std::vector<int32_t>> exp_slot;     // [il][e] slot | -1
    std::vector<std::vector<int32_t>> ss;           // [il][slot] slot state
    std::vector<std::vector<int32_t>> hm;           // [il][e] hotmap (slot, 0 if cold)
    std::vector<std::vector<float>>   rs;           // [il][e] residency mask 1.0/0.0

    // P3 boundary pass (moert_policy_boundary_layer): restrict the pairing to one layer and price it at
    // r_override. -1 = off (every other refresh).
    int   only_layer = -1;
    float r_override = -1.0f;

    // External forecast (moert_policy_set_external_forecast): f[il * n_expert + e] replaces the rate EMA's
    // forecast, priced at ext_r and capped at ext_cap per per-token decision. Not used by the boundary pass.
    const float * ext_f = nullptr;
    float ext_r = 0.0f;
    int32_t ext_cap = 0;

    // telemetry
    long long promotions = 0, demotions = 0, tokens = 0;
    long long hot_routes = 0, cold_routes = 0;
    long long visible_tokens = 0;
    long long refresh_count = 0, refresh_promotions = 0, refresh_demotions = 0;
    double refresh_pause_ms_total = 0.0, refresh_pause_ms_max = 0.0;
    long long last_refresh_tok = 0, last_refresh_promos = 0;
    double last_refresh_pause_ms = 0.0;
    long long fills_async = 0, fills_completed = 0, fills_stale = 0, fills_aborted = 0;   // async promotion ledger

    // windowed-rate ring (M2 R9/R10): one sample per VISIBLE token end — steady-clock stamp
    // + cumulative route counters at that moment. Fixed capacity; position = win_n % WIN_CAP.
    static constexpr int WIN_CAP = 128;
    struct WinSample { double t_ms; long long hot, cold; };
    WinSample  win[WIN_CAP] = {};
    long long  win_n = 0;   // samples ever recorded
};

namespace {

// Fold the accumulated per-expert routing counts into the session/recent EMAs, then clear the counters.
// n_tokens > 1: the counts were accumulated over a whole prefill CHUNK (batched prefill). Folding a
// chunk as one raw step (s = d*s + C) would give prompt routes ~n× the weight of per-token folding and
// drown every later signal. The exact EMA for counts spread uniformly over the chunk is
//   s = d^n * s + C * (1 - d^n) / ((1 - d) * n)
// which degenerates to the per-token formula at n == 1.
void fold_scores(moert_policy & g, long long n_tokens = 1) {
    const double n = (double) std::max<long long>(1, n_tokens);
    auto coef = [&](float d, float & dn, float & w) {
        dn = (n_tokens > 1) ? (float) std::pow((double) d, n) : d;
        w  = (n_tokens > 1) ? (float) ((1.0 - dn) / ((1.0 - (double) d) * n)) : 1.0f;
    };
    float dn_ss, w_ss, dn_rc, w_rc;
    coef(g.cfg.session_decay, dn_ss, w_ss);
    coef(g.cfg.recent_decay,  dn_rc, w_rc);
    for (int il = 0; il < g.cfg.n_layers; ++il) {
        auto & c  = g.cnt[il];
        auto & ss = g.session_score[il];
        auto & rc = g.recent_score[il];
        for (int e = 0; e < g.cfg.n_expert; ++e) {
            ss[e] = dn_ss * ss[e] + w_ss * c[e];
            rc[e] = dn_rc * rc[e] + w_rc * c[e];
            c[e]  = 0.0f;
        }
    }
}

// Fold the pending counts into the use-rate EMA as n_visible visible steps (the same uniform-spread
// correction as fold_scores). Must run BEFORE fold_scores, which clears cnt. A fold that covers no
// visible step (prompt chunk) leaves the rate untouched: the rate is a forecast of DECODE demand, and
// prompt routing is a poor proxy for it (findings §159) -- the boundary pass is the one place the
// prompt enters it, deliberately and at a weight.
void fold_rate(moert_policy & g, long long n_visible) {
    if (n_visible <= 0) return;
    const double n  = (double) n_visible;
    const double d  = (double) g.cfg.cost_rate_decay;
    const float  dn = (float) std::pow(d, n);
    const float  w  = (n_visible > 1) ? (float) ((1.0 - dn) / ((1.0 - d) * n)) : 1.0f;
    for (int il = 0; il < g.cfg.n_layers; ++il) {
        auto & c = g.cnt[il];
        auto & r = g.rate_score[il];
        for (int e = 0; e < g.cfg.n_expert; ++e) r[e] = dn * r[e] + w * c[e];
    }
}

// Forecast uses of expert e in layer il over the next cost_horizon visible tokens: the steady-state
// per-token rate of an EMA fed one count per use is rate * (1 - decay); an expert is used at most
// once per token, so the forecast is capped at H.
inline float cost_forecast(const moert_policy & g, int il, int e) {
    if (g.ext_f && g.only_layer < 0) return g.ext_f[(size_t) il * g.cfg.n_expert + e];
    const float H = (float) g.cfg.cost_horizon;
    return H * std::min(1.0f, g.rate_score[il][e] * (1.0f - g.cfg.cost_rate_decay));
}

// Blended score (recency-weighted, plus session locality, plus an optional incumbency bonus): the
// ranking moert_policy_resize_k keeps survivors by.
inline float blended(const moert_policy & g, int il, int e) {
    float v = g.cfg.blend_a * g.recent_score[il][e] + g.cfg.blend_b * g.session_score[il][e];
    if (g.cfg.blend_c != 0.0f && g.rs[il][e] > 0.0f) v += g.cfg.blend_c * g.cfg.seed_prior;
    return v;
}

// Evict the expert resident in (il, slot): clear its residency mirrors. Host mirrors only — caller
// publishes the layer once all its streaming is done.
void evict_slot(moert_policy & g, int il, int slot) {
    const int ev = g.slot_exp[il][slot];
    if (ev < 0) return;
    g.exp_slot[il][ev] = -1; g.slot_exp[il][slot] = -1;
    g.ss[il][slot] = SLOT_FREE;                 // not READY until re-streamed (fused-op safety)
    g.hm[il][ev] = 0; g.rs[il][ev] = 0.0f;
    ++g.demotions;
}

// Stream expert e's weights into (il, slot) via the host callback and mark it READY. Host mirrors
// only — the device sees READY only after publish_layer, which the caller defers until ALL streaming
// for the layer is done, so the fused op never reads a half-written slot as hot.
// An asynchronous accept (callback returned 2) occupies the slot as FILLING: exp_slot/slot_exp point at
// e (so no other decision takes the slot), but hm/rs and READY wait for moert_policy_complete_fill.
bool fill_slot(moert_policy & g, int il, int e, int slot) {
    const int32_t r = g.cbs.stream_expert ? g.cbs.stream_expert(g.cbs.user, il, e, slot) : 0;
    if (!r) return false;
    g.slot_exp[il][slot] = e; g.exp_slot[il][e] = slot;
    if (r == 2) {
        g.ss[il][slot] = SLOT_FILLING;
        ++g.fills_async;
    } else {
        g.ss[il][slot] = SLOT_READY;
        g.hm[il][e] = slot; g.rs[il][e] = 1.0f;
    }
    ++g.promotions;
    return true;
}

}  // namespace

void moert_policy_config_defaults(moert_policy_config * cfg) {
    if (!cfg) return;
    std::memset(cfg, 0, sizeof(*cfg));
    cfg->seed_prior              = 5.0f;
    cfg->session_decay           = 0.995f;
    cfg->recent_decay            = 0.95f;
    cfg->blend_a                 = 1.0f;
    cfg->blend_b                 = 0.25f;
    cfg->blend_c                 = 0.0f;
    cfg->cost_horizon            = 16;             // findings §173 (the product rule)
    cfg->cost_reuse              = 2.0f;           // an asynchronous promotion, in cold pairs
    cfg->cost_rate_decay         = 0.97857206f;    // 0.5^(1/32)
    cfg->cost_per_token          = 4;              // the copy engine's capacity per visible token
}

moert_policy * moert_policy_create(const moert_policy_config * cfg,
                                   const moert_policy_callbacks * cbs) {
    if (!cfg || cfg->n_layers <= 0 || cfg->n_expert <= 0 || cfg->k <= 0 || cfg->k > cfg->n_expert)
        return nullptr;
    // per-layer capacity: the optional table (validated in [1,E]) or uniform k everywhere.
    std::vector<int32_t> kl(cfg->n_layers, cfg->k);
    if (cfg->k_per_layer) {
        for (int il = 0; il < cfg->n_layers; ++il) {
            const int32_t kv = cfg->k_per_layer[il];
            if (kv < 1 || kv > cfg->n_expert) return nullptr;
            kl[il] = kv;
        }
    }
    moert_policy * p = new moert_policy();
    p->cfg = *cfg;
    p->cfg.k_per_layer = nullptr;   // do not retain the caller's pointer; kl is the owned copy
    if (p->cfg.cost_per_token < 1) p->cfg.cost_per_token = 1;
    p->kl = kl;
    if (cbs) p->cbs = *cbs;
    const int L = cfg->n_layers, E = cfg->n_expert;
    p->cnt.assign(L, std::vector<float>(E, 0.0f));
    p->session_score.assign(L, std::vector<float>(E, 0.0f));
    p->recent_score.assign(L, std::vector<float>(E, 0.0f));
    p->rate_score.assign(L, std::vector<float>(E, 0.0f));
    p->exp_slot.assign(L, std::vector<int32_t>(E, -1));
    p->hm.assign(L, std::vector<int32_t>(E, 0));
    p->rs.assign(L, std::vector<float>(E, 0.0f));
    p->slot_exp.resize(L); p->ss.resize(L);
    for (int il = 0; il < L; ++il) {
        p->slot_exp[il].assign(kl[il], -1);
        p->ss[il].assign(kl[il], SLOT_FREE);
    }
    return p;
}

void moert_policy_free(moert_policy * p) { delete p; }

void moert_policy_set_external_observation(moert_policy * p, int32_t on) {
    if (p) p->external = on != 0;
}

void moert_policy_observe(moert_policy * p, int32_t il, const int32_t * ids,
                          int32_t n_used, int64_t n_tok) {
    if (!p || il < 0 || il >= p->cfg.n_layers || !ids) return;
    auto & g = *p;
    auto & c = g.cnt[il];
    for (long long t = 0; t < n_tok; ++t)
        for (int u = 0; u < n_used; ++u) {
            const int e = ids[(size_t) t * n_used + u];
            if (e < 0 || e >= g.cfg.n_expert) continue;
            c[e] += 1.0f;
            // realized hit/miss: exactly the predicate the fused partition uses (resident + READY)
            const int s = g.exp_slot[il][e];
            const bool hot = (s >= 0 && s < g.kl[il] && g.ss[il][s] == SLOT_READY);
            hot ? ++g.hot_routes : ++g.cold_routes;
        }
}

void moert_policy_set_external_forecast(moert_policy * p, const float * f, float r, int32_t cap) {
    if (!p) return;
    p->ext_f = f; p->ext_r = r; p->ext_cap = cap;
}

void moert_policy_on_prompt_token(moert_policy * p, int64_t n_tokens) {
    if (!p) return;
    auto & g = *p;
    g.tokens += std::max<int64_t>(1, n_tokens);
    if (g.external) { g.dev_since_fold += std::max<int64_t>(1, n_tokens); return; }  // fold at refresh
    fold_scores(g, n_tokens);
}

void moert_policy_on_token_end(moert_policy * p) {
    if (!p) return;
    auto & g = *p;
    ++g.tokens; ++g.visible_tokens;
    {   // window sample: per-token stamp + cumulative routes (external counts may lag by up
        // to one decision -- see the moert_policy_get_window contract in policy.h)
        auto & s = g.win[g.win_n % moert_policy::WIN_CAP];
        s.t_ms = std::chrono::duration<double, std::milli>(
                     std::chrono::steady_clock::now().time_since_epoch()).count();
        s.hot = g.hot_routes; s.cold = g.cold_routes;
        ++g.win_n;
    }
    if (g.external) { ++g.dev_since_fold; ++g.visible_since_fold; }   // counts stay external; folded at the refresh
    else { fold_rate(g, 1); fold_scores(g); }
    moert_policy_refresh(p, MOERT_REFRESH_VISIBLE_INTERVAL);   // the per-token decision
}

// The cost-aware pairing. Per layer, the non-residents ranked by forecast use over the next H visible
// tokens (desc) are paired with the free slots (pay r, save f_in) and then with the residents ranked
// by forecast (asc) while f_in - f_out > r; the pairs are applied highest net saving first. The price
// is the restraint; the only budget is the per-token cap of a per-token decision (the copy engine's
// capacity). Streams are batched and residency is published once per touched layer.
void moert_policy_refresh(moert_policy * p, int32_t reason) {
    if (!p) return;
    auto & g = *p;
    // External observation: refresh points are the ONLY places scores must be current — pull the
    // pending counts and fold once for the whole window (the same uniform-spread EMA correction the
    // batched-prefill fold uses).
    if (g.external) {
        if (g.cbs.fetch_external) g.cbs.fetch_external(g.cbs.user);
        fold_rate(g, g.visible_since_fold);
        fold_scores(g, std::max<long long>(1, g.dev_since_fold));
        g.dev_since_fold = 0;
        g.visible_since_fold = 0;
    }
    using clk = std::chrono::steady_clock;
    const clk::time_point t0 = clk::now();

    struct Cand { int il, e, slot, victim_e; bool evict; float gain; };
    std::vector<Cand> cands;
    const bool ext = g.ext_f && g.only_layer < 0;
    const long long eff_budget = reason == MOERT_REFRESH_VISIBLE_INTERVAL
                                     ? (long long) (ext && g.ext_cap > 0 ? g.ext_cap : g.cfg.cost_per_token)
                                     : (long long) 1 << 40;
    for (int il = 0; il < g.cfg.n_layers; ++il) {
        if (g.only_layer >= 0 && il != g.only_layer) continue;
        const int   K = g.kl[il];
        const float r = g.r_override >= 0.0f ? g.r_override : (ext ? g.ext_r : g.cfg.cost_reuse);
        std::vector<int> cand, free_slots, resid;      // resid = occupied slots
        for (int e = 0; e < g.cfg.n_expert; ++e) if (g.exp_slot[il][e] < 0) cand.push_back(e);
        for (int s = 0; s < K; ++s) (g.slot_exp[il][s] < 0 ? free_slots : resid).push_back(s);
        // forecast desc / asc, ties by id (stable on an id-ordered list) -- same order as the replay
        std::stable_sort(cand.begin(), cand.end(),
                         [&](int a, int b) { return cost_forecast(g, il, a) > cost_forecast(g, il, b); });
        std::sort(resid.begin(), resid.end(), [&](int a, int b) {
            const int ea = g.slot_exp[il][a], eb = g.slot_exp[il][b];
            const float fa = cost_forecast(g, il, ea), fb = cost_forecast(g, il, eb);
            return fa != fb ? fa < fb : ea < eb; });
        size_t ci = 0;
        for (size_t fs = 0; fs < free_slots.size() && ci < cand.size(); ++fs) {   // free slot: pay r, save f_in
            const float fin = cost_forecast(g, il, cand[ci]);
            if (fin <= r) break;
            cands.push_back({il, cand[ci], free_slots[fs], -1, false, fin - r});
            ++ci;
        }
        for (size_t q = 0; q < resid.size() && ci < cand.size(); ++q, ++ci) {      // swap: save f_in - f_out
            const int   s    = resid[q];
            const int   xe   = g.slot_exp[il][s];
            const float fin  = cost_forecast(g, il, cand[ci]);
            const float fout = cost_forecast(g, il, xe);
            if (fin - fout <= r) break;
            cands.push_back({il, cand[ci], s, xe, true, fin - fout - r});
        }
    }

    // apply the highest-gain pairs first, up to the per-token cap
    std::sort(cands.begin(), cands.end(), [](const Cand & a, const Cand & b) { return a.gain > b.gain; });
    std::vector<char> touched(g.cfg.n_layers, 0);
    long long did_promote = 0, did_demote = 0;
    for (const Cand & c : cands) {
        if (did_promote >= eff_budget) break;
        if (g.exp_slot[c.il][c.e] >= 0) continue;                       // stale: already resident
        if (c.evict) {
            if (g.slot_exp[c.il][c.slot] != c.victim_e) continue;       // victim moved since planning
            evict_slot(g, c.il, c.slot);
            ++did_demote;
        } else if (g.slot_exp[c.il][c.slot] >= 0) {
            continue;                                                    // slot no longer free (defensive)
        }
        if (!fill_slot(g, c.il, c.e, c.slot)) continue;                 // stream failed -> slot left free
        touched[c.il] = 1; ++did_promote;
    }
    if (g.cbs.publish_layer)
        for (int il = 0; il < g.cfg.n_layers; ++il)
            if (touched[il]) g.cbs.publish_layer(g.cbs.user, il);

    const double ms = std::chrono::duration<double, std::milli>(clk::now() - t0).count();
    ++g.refresh_count;
    g.refresh_promotions += did_promote;
    g.refresh_demotions  += did_demote;
    g.refresh_pause_ms_total += ms;
    g.refresh_pause_ms_max = std::max(g.refresh_pause_ms_max, ms);
    g.last_refresh_tok = g.visible_tokens; g.last_refresh_promos = did_promote; g.last_refresh_pause_ms = ms;
}

int32_t moert_policy_resize_k(moert_policy * p, int32_t k_new) {
    if (!p || k_new < 1 || k_new > p->cfg.n_expert) return 0;
    auto & g = *p;
    for (int il = 0; il < g.cfg.n_layers; ++il) {
        // rank this layer's residents by the blended score; ties broken by expert id for determinism
        std::vector<int> res;
        for (int s = 0; s < g.kl[il]; ++s)
            if (g.slot_exp[il][s] >= 0) res.push_back(s);
        std::sort(res.begin(), res.end(), [&](int a, int b) {
            const int ea = g.slot_exp[il][a], eb = g.slot_exp[il][b];
            const float sa = blended(g, il, ea), sb = blended(g, il, eb);
            return sa != sb ? sa > sb : ea < eb;
        });
        const int keep = std::min((int) res.size(), (int) k_new);
        // evict-account the residents that no longer fit
        for (size_t r = keep; r < res.size(); ++r) {
            const int ev = g.slot_exp[il][res[r]];
            g.exp_slot[il][ev] = -1;
            g.hm[il][ev] = 0; g.rs[il][ev] = 0.0f;
            ++g.demotions;
        }
        // rebuild the slot arrays at k_new; survivors compact into slots 0..keep-1 in rank order. They
        // stay READY: the host MUST re-stream every survivor into its new slot before the next decode
        // (slab_restore contract).
        std::vector<int32_t> se_new(k_new, -1), ss_new(k_new, SLOT_FREE);
        for (int r = 0; r < keep; ++r) {
            const int e = g.slot_exp[il][res[r]];
            se_new[r] = e;
            ss_new[r] = SLOT_READY;
            g.exp_slot[il][e] = r;
            g.hm[il][e] = r; g.rs[il][e] = 1.0f;
        }
        g.slot_exp[il].swap(se_new);
        g.ss[il].swap(ss_new);
        g.kl[il] = k_new;
    }
    g.cfg.k = k_new;
    return 1;
}

int32_t moert_policy_set_active_k(moert_policy * p, int32_t k) {
    if (!p || k < 1) return 0;
    auto & g = *p;
    for (int il = 0; il < g.cfg.n_layers; ++il) {
        const int cap = (int) g.slot_exp[il].size();
        const int ka  = std::min((int) k, cap);
        for (int s = ka; s < cap; ++s) {
            const int e = g.slot_exp[il][s];
            if (e < 0) continue;
            if (g.ss[il][s] == SLOT_READY) { g.hm[il][e] = 0; g.rs[il][e] = 0.0f; ++g.demotions; }
            g.exp_slot[il][e] = -1; g.slot_exp[il][s] = -1;
            g.ss[il][s] = SLOT_FREE;
        }
        g.kl[il] = ka;
    }
    return 1;
}

int32_t moert_policy_complete_fill(moert_policy * p, int32_t il, int32_t slot, int32_t e) {
    if (!p || il < 0 || il >= p->cfg.n_layers || slot < 0 || slot >= p->kl[il]) return 0;
    auto & g = *p;
    if (g.slot_exp[il][slot] != e || g.ss[il][slot] != SLOT_FILLING) { ++g.fills_stale; return 0; }
    g.ss[il][slot] = SLOT_READY;
    g.hm[il][e] = slot; g.rs[il][e] = 1.0f;
    ++g.fills_completed;
    return 1;
}

int32_t moert_policy_abort_fill(moert_policy * p, int32_t il, int32_t slot, int32_t e) {
    if (!p || il < 0 || il >= p->cfg.n_layers || slot < 0 || slot >= p->kl[il]) return 0;
    auto & g = *p;
    if (g.slot_exp[il][slot] != e || g.ss[il][slot] != SLOT_FILLING) { ++g.fills_stale; return 0; }
    g.exp_slot[il][e] = -1; g.slot_exp[il][slot] = -1;
    g.ss[il][slot] = SLOT_FREE;
    ++g.fills_aborted;
    return 1;
}

int32_t moert_policy_boundary_layer(moert_policy * p, int32_t il, const int32_t * counts, int64_t n_tok,
                                    float weight, float r) {
    if (!p || !counts || n_tok <= 0 || il < 0 || il >= p->cfg.n_layers) return -1;
    auto & g = *p;
    const float w = std::min(1.0f, std::max(0.0f, weight));
    const float d = g.cfg.cost_rate_decay;
    auto & rate = g.rate_score[il];
    for (int e = 0; e < g.cfg.n_expert; ++e) {
        const float q = std::min(1.0f, (float) counts[e] / (float) n_tok);   // routes per prompt token
        rate[e] = (1.0f - w) * rate[e] + w * q / (1.0f - d);                 // EMA units: rate*(1-d) = q
    }
    g.only_layer = il;
    g.r_override = std::max(0.0f, r);
    const long long before = g.refresh_promotions;
    moert_policy_refresh(p, MOERT_REFRESH_BOUNDARY);
    g.only_layer = -1;
    g.r_override = -1.0f;
    return (int32_t) (g.refresh_promotions - before);
}

int32_t moert_policy_can_seed(const moert_policy * p, int32_t il, int32_t e) {
    if (!p || il < 0 || il >= p->cfg.n_layers) return 0;
    return (e >= 0 && e < p->cfg.n_expert && p->exp_slot[il][e] < 0) ? 1 : 0;
}

int32_t moert_policy_commit_seed(moert_policy * p, int32_t il, int32_t e, int32_t slot) {
    if (!moert_policy_can_seed(p, il, e) || slot < 0 || slot >= p->kl[il]) return 0;
    auto & g = *p;
    g.slot_exp[il][slot] = e; g.exp_slot[il][e] = slot;
    g.ss[il][slot] = SLOT_READY;
    g.hm[il][e] = slot; g.rs[il][e] = 1.0f;
    g.session_score[il][e] = g.recent_score[il][e] = g.cfg.seed_prior;    // the resize ranking's prior
    return 1;
}

void moert_policy_add_external_counts(moert_policy * p, int32_t il,
                                      const int32_t * counts, int64_t hot, int64_t cold) {
    if (!p || il < 0 || il >= p->cfg.n_layers || !counts) return;
    auto & g = *p;
    auto & c = g.cnt[il];
    for (int e = 0; e < g.cfg.n_expert; ++e) c[e] += (float) counts[e];
    g.hot_routes += hot; g.cold_routes += cold;
}

void moert_policy_reset_stats(moert_policy * p) {
    if (!p) return;
    auto & g = *p;
    g.hot_routes = g.cold_routes = 0;
    g.promotions = g.demotions = g.tokens = 0;
    // keep learned residency + scores; only the live counters reset
    g.visible_tokens = 0;
    g.refresh_count = g.refresh_promotions = g.refresh_demotions = 0;
    g.refresh_pause_ms_total = g.refresh_pause_ms_max = 0.0;
    g.last_refresh_tok = g.last_refresh_promos = 0; g.last_refresh_pause_ms = 0.0;
    g.fills_async = g.fills_completed = g.fills_stale = g.fills_aborted = 0;
    g.win_n = 0;   // windowed-rate ring restarts with the counters it snapshots
}

double moert_policy_hit_rate(moert_policy * p) {
    if (!p) return 0.0;
    if (p->external && p->cbs.fetch_external) p->cbs.fetch_external(p->cbs.user);
    const long long tot = p->hot_routes + p->cold_routes;
    return tot > 0 ? (double) p->hot_routes / (double) tot : 0.0;
}

void moert_policy_get_stats(moert_policy * p, moert_policy_stats * out) {
    if (!p || !out) return;
    if (p->external && p->cbs.fetch_external) p->cbs.fetch_external(p->cbs.user);
    const auto & g = *p;
    std::memset(out, 0, sizeof(*out));
    out->promotions            = g.promotions;
    out->demotions             = g.demotions;
    out->tokens                = g.tokens;
    out->visible_tokens        = g.visible_tokens;
    out->hot_routes            = g.hot_routes;
    out->cold_routes           = g.cold_routes;
    out->refresh_count         = g.refresh_count;
    out->refresh_promotions    = g.refresh_promotions;
    out->refresh_demotions     = g.refresh_demotions;
    out->refresh_pause_ms_total= g.refresh_pause_ms_total;
    out->refresh_pause_ms_max  = g.refresh_pause_ms_max;
    out->last_refresh_tok      = g.last_refresh_tok;
    out->last_refresh_promos   = g.last_refresh_promos;
    out->last_refresh_pause_ms = g.last_refresh_pause_ms;
    out->fills_async           = g.fills_async;
    out->fills_completed       = g.fills_completed;
    out->fills_stale           = g.fills_stale;
    out->fills_aborted         = g.fills_aborted;
    // occupancy: mean READY hot slots / K across layers (per-layer capacity aware)
    if (g.cfg.n_layers > 0) {
        long long filled = 0, slots = 0;
        for (int L = 0; L < g.cfg.n_layers; ++L) {
            slots += g.kl[L];
            for (int s = 0; s < g.kl[L]; ++s) if (g.ss[L][s] == SLOT_READY) ++filled;
        }
        if (slots > 0) out->occupancy = (double) filled / (double) slots;
    }
}

int32_t moert_policy_layer_k(const moert_policy * p, int32_t il) {
    if (!p || il < 0 || il >= p->cfg.n_layers) return -1;
    return p->kl[il];
}

void moert_policy_get_window(moert_policy * p, int32_t n_tokens, moert_policy_window * out) {
    if (!out) return;
    std::memset(out, 0, sizeof(*out));
    if (!p || n_tokens < 2) return;
    if (p->external && p->cbs.fetch_external) p->cbs.fetch_external(p->cbs.user);
    const auto & g = *p;
    const long long have = std::min<long long>(g.win_n, moert_policy::WIN_CAP);
    const long long n    = std::min<long long>(n_tokens, have);
    if (n < 2) return;
    const auto & first = g.win[(g.win_n - n) % moert_policy::WIN_CAP];
    const auto & last  = g.win[(g.win_n - 1) % moert_policy::WIN_CAP];
    out->tokens  = (int32_t) n;
    out->span_ms = last.t_ms - first.t_ms;
    if (out->span_ms > 0.0) {
        out->tok_per_s  = 1000.0 * (double) (n - 1) / out->span_ms;
        out->ms_per_tok = out->span_ms / (double) (n - 1);
    }
    // routes attributed to the window = cumulative delta first->NOW (the current counters
    // include any external pour that happened after the last sample; attributing it here
    // keeps the window rate fresher than the last refresh point)
    out->hot_routes  = g.hot_routes  - first.hot;
    out->cold_routes = g.cold_routes - first.cold;
    const long long tot = out->hot_routes + out->cold_routes;
    if (tot > 0) out->hit_rate = (double) out->hot_routes / (double) tot;
}

const int32_t * moert_policy_exp_slot_view(const moert_policy * p, int32_t il) {
    return (p && il >= 0 && il < p->cfg.n_layers) ? p->exp_slot[il].data() : nullptr;
}
const int32_t * moert_policy_slot_exp_view(const moert_policy * p, int32_t il) {
    return (p && il >= 0 && il < p->cfg.n_layers) ? p->slot_exp[il].data() : nullptr;
}
const int32_t * moert_policy_slot_state_view(const moert_policy * p, int32_t il) {
    return (p && il >= 0 && il < p->cfg.n_layers) ? p->ss[il].data() : nullptr;
}
const int32_t * moert_policy_hotmap_view(const moert_policy * p, int32_t il) {
    return (p && il >= 0 && il < p->cfg.n_layers) ? p->hm[il].data() : nullptr;
}
const float * moert_policy_resid_view(const moert_policy * p, int32_t il) {
    return (p && il >= 0 && il < p->cfg.n_layers) ? p->rs[il].data() : nullptr;
}
