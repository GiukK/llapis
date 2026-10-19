// moe_xfer -- asynchronous expert transfer engine. Contract and rationale in moe_xfer.hpp.

#include "moe_xfer.hpp"

#include "moert/memmgr.h"
#include "moert/registry.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>

namespace moe_xfer {
namespace {

enum JobKind : int32_t { JOB_PROMOTE = 0, JOB_RELEASE = 1, JOB_RESTORE = 2 };
struct Job { JobKind kind; int32_t il, e, slot, src; int64_t gate = 0; };

enum Prio { PRIO_STAGING = 0, PRIO_RESTORE = 1, PRIO_OTHER = 2, N_PRIO = 3 };
static Prio prio_of(const Job & j) { return j.kind == JOB_RESTORE ? PRIO_RESTORE : PRIO_OTHER; }

using clk = std::chrono::steady_clock;
static double ms_since(clk::time_point t0) {
    return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
}

struct Engine {
    moert_memmgr *          mm = nullptr;
    std::thread             th;
    std::mutex              mu;
    std::condition_variable cv_work;   // worker: a job is queued / quit
    std::condition_variable cv_idle;   // waiters: queue drained and nothing running
    std::condition_variable cv_res;    // cold_guard: a page became resident
    std::vector<Job>        pending;   // submitted during a policy call, not yet visible
    // Visible to the worker, one FIFO per priority. RESTOREs jump ahead of promotions and releases:
    // a cold read may be waiting on one, and running it early is always safe -- (a) no job queued
    // ahead of RESTORE(e, s) writes slot s (the host queues the restore of the bytes a slot holds
    // before any job that overwrites it, so an earlier writer of s would have made s hold that
    // writer's expert, not e); (b) an earlier RELEASE(e) was cancelled by the restore request, and
    // the restore is a no-op if that release never ran; (c) an earlier PROMOTE reading e's page only
    // gains. Restores keep their own FIFO order.
    std::deque<Job>         q[N_PRIO];
    bool queues_empty() const { for (const auto & d : q) if (!d.empty()) return false; return true; }
    size_t queued() const { size_t n = 0; for (const auto & d : q) n += d.size(); return n; }
    std::vector<Done>       done;
    // batch transport: promotions whose copies are in flight, with the completion id of their mark
    int64_t (*mark_fn)() = nullptr;
    int64_t (*poll_fn)(int32_t) = nullptr;
    std::deque<std::pair<int64_t, Done>> inflight;
    std::vector<Done> unmarked;   // issued, waiting for the batch's mark (one event per batch, not per promotion)
    void collect(bool wait_all) {   // under mu
        if (!poll_fn || inflight.empty()) return;
        const int64_t upto = poll_fn(wait_all ? 1 : 0);
        while (!inflight.empty() && inflight.front().first <= upto) { done.push_back(inflight.front().second); inflight.pop_front(); }
    }
    bool                    busy = false;
    bool                    quit = false;
    Stats                   st;

    // exclusive arena. resident[] is written only under `mu` (so cold_guard's wait cannot miss a
    // wakeup) and read lock-free on the fast path; want_release[] and freed[] only under `mu`.
    std::atomic<bool>                     excl{false};
    // soft = the arena is the mapped pack: a released page stays readable (it faults back from the file),
    // so cold_guard never waits and a restore is only a prefetch, queued behind the promotions
    std::atomic<bool>                     soft{false};
    int32_t                               n_layers = 0, n_expert = 0;
    std::unique_ptr<std::atomic<uint8_t>[]> resident;
    std::vector<uint8_t>                  want_release;
    std::vector<uint64_t>                 freed;

    int32_t idx(int32_t il, int32_t e) const { return il * n_expert + e; }
    bool in_range(int32_t il, int32_t e) const { return il >= 0 && il < n_layers && e >= 0 && e < n_expert; }

    void (*gate_wait)(int32_t, int64_t) = nullptr;
    int64_t submit_gate = 0;
    void do_promote(const Job & j, std::unique_lock<std::mutex> & lk) {
        if (j.gate > 0 && gate_wait) { lk.unlock(); gate_wait(j.il, j.gate); lk.lock(); }
        const bool arena = j.src == MOERT_SRC_ARENA;
        if (!arena && j.src != MOERT_SRC_FILE) {
            std::fprintf(stderr, "[MOE_XFER] promotion of l%d e%d refused: no source (%s)\n",
                         j.il, j.e, moert_source_name(j.src));
            done.push_back({ j.il, j.e, j.slot, false });
            ++st.failed;
            return;
        }
        if (arena && excl.load() && in_range(j.il, j.e) && !resident[idx(j.il, j.e)].load()) {
            // host contract broken (no RESTORE queued ahead): fail the fill instead of reading a
            // released page; the policy's abort_fill frees the slot
            std::fprintf(stderr, "[MOE_XFER] promotion of l%d e%d refused: its arena page is released\n", j.il, j.e);
            done.push_back({ j.il, j.e, j.slot, false });
            ++st.failed;
            return;
        }
        lk.unlock();
        const auto t0 = clk::now();
        const bool ok = (arena ? moert_memmgr_promote_from_arena(mm, j.il, j.e, j.slot)
                               : moert_memmgr_promote(mm, j.il, j.e, j.slot)) != 0;
        const double ms = ms_since(t0);
        lk.lock();
        if (ok && mark_fn) {   // batch transport: done when the batch's mark lands
            unmarked.push_back(Done{ j.il, j.e, j.slot, ok });
            if (queues_empty() || unmarked.size() >= 16) {
                lk.unlock();
                const int64_t mk = mark_fn();
                lk.lock();
                for (const Done & d : unmarked) inflight.push_back({ mk, d });
                unmarked.clear();
            }
        } else {
            done.push_back({ j.il, j.e, j.slot, ok });
        }
        st.copy_ms_total += ms;
        ++(ok ? st.completed : st.failed);
        if (ok) ++(arena ? st.from_arena : st.from_file);
    }

    void do_release(const Job & j, std::unique_lock<std::mutex> & lk) {
        const int32_t i = idx(j.il, j.e);
        // a RESTORE requested after this job was queued wins: the host already decided e is cold
        if (!want_release[i] || !resident[i].load()) { ++st.releases_skipped; return; }
        want_release[i] = 0;
        resident[i].store(0);   // from here on cold_guard blocks on e until its RESTORE runs
        lk.unlock();
        const auto t0 = clk::now();
        const uint64_t b = moert_memmgr_arena_release(mm, j.il, j.e);
        const double ms = ms_since(t0);
        lk.lock();
        freed[i] = b;
        st.released_bytes += (long long) b;
        if (st.released_bytes > st.released_bytes_max) st.released_bytes_max = st.released_bytes;
        st.release_ms_total += ms;
        ++st.releases;
    }

    void do_restore(const Job & j, std::unique_lock<std::mutex> & lk) {
        const int32_t i = idx(j.il, j.e);
        if (resident[i].load()) { ++st.restores_skipped; return; }   // its release never ran
        lk.unlock();
        const auto t0 = clk::now();
        bool ok = moert_memmgr_arena_restore(mm, j.il, j.e, j.slot) != 0;
        if (!ok) {
            // last resort: the promotion source on disk (the pre-phase-2 path). Never leave the page
            // unreadable -- cold_guard would wait on it forever.
            ok = moert_memmgr_arena_reload_expert(mm, j.il, j.e) != 0;
            std::fprintf(stderr, "[MOE_XFER] CRITICAL: restore of l%d e%d from slot %d failed; %s\n",
                         j.il, j.e, j.slot, ok ? "reloaded from the store file" : "store reload ALSO failed");
        }
        const double ms = ms_since(t0);
        lk.lock();
        st.released_bytes -= (long long) freed[i];
        freed[i] = 0;
        st.restore_ms_total += ms;
        ++st.restores;
        if (!ok) ++st.restore_failed;
        resident[i].store(1);
        cv_res.notify_all();
    }

    void run() {
        std::unique_lock<std::mutex> lk(mu);
        for (;;) {
            cv_work.wait(lk, [&] { return quit || !queues_empty(); });
            if (queues_empty()) return;   // quit with nothing left
            std::deque<Job> * from = nullptr;
            for (auto & d : q) if (!d.empty()) { from = &d; break; }
            const Job j = from->front();
            from->pop_front();
            busy = true;
            switch (j.kind) {
                case JOB_PROMOTE: do_promote(j, lk); break;
                case JOB_RELEASE: do_release(j, lk); break;
                case JOB_RESTORE: do_restore(j, lk); break;
            }
            if (queues_empty() && !unmarked.empty() && mark_fn) {   // a batch ended on a non-promotion job
                lk.unlock();
                const int64_t mk = mark_fn();
                lk.lock();
                for (const Done & d : unmarked) inflight.push_back({ mk, d });
                unmarked.clear();
            }
            busy = false;
            if (queues_empty()) cv_idle.notify_all();
        }
    }

    ~Engine() {
        if (th.joinable()) {
            { std::lock_guard<std::mutex> lk(mu); quit = true; }
            cv_work.notify_all();
            th.join();
        }
        if (st.submitted)
            std::fprintf(stderr, "[MOE_XFER] async promotions submitted=%lld completed=%lld (arena %lld, file %lld)"
                                 " failed=%lld | max queue=%lld | copy %.1f ms total (%.2f ms/promotion)"
                                 " | decode-thread wait_idle %.1f ms over %lld calls\n",
                         st.submitted, st.completed, st.from_arena, st.from_file, st.failed, st.max_queue, st.copy_ms_total,
                         st.completed ? st.copy_ms_total / (double) st.completed : 0.0,
                         st.wait_idle_ms, st.wait_idle_calls);
        if (excl.load())
            std::fprintf(stderr, "[MOE_XFER] exclusive arena releases=%lld (skipped %lld, %.1f ms) restores=%lld"
                                 " (skipped %lld, failed %lld, %.1f ms, %.2f ms/restore) | released now %lld MiB,"
                                 " max %lld MiB | cold-guard waits=%lld (%.1f ms)\n",
                         st.releases, st.releases_skipped, st.release_ms_total, st.restores,
                         st.restores_skipped, st.restore_failed, st.restore_ms_total,
                         st.restores ? st.restore_ms_total / (double) st.restores : 0.0,
                         st.released_bytes >> 20, st.released_bytes_max >> 20,
                         st.guard_waits, st.guard_wait_ms);
    }
};

Engine g_eng;
thread_local bool t_worker = false;

void submit_job(const Job & j) {
    std::lock_guard<std::mutex> lk(g_eng.mu);
    g_eng.pending.push_back(j);
    if (j.kind == JOB_PROMOTE) ++g_eng.st.submitted;
}

}  // namespace

bool start(moert_memmgr * mm) {
    std::lock_guard<std::mutex> lk(g_eng.mu);
    if (g_eng.th.joinable()) return true;
    if (!mm) return false;
    g_eng.mm = mm;
    g_eng.th = std::thread([] { t_worker = true; g_eng.run(); });
    return true;
}

bool on_worker() { return t_worker; }

bool running() {
    std::lock_guard<std::mutex> lk(g_eng.mu);
    return g_eng.th.joinable();
}

void submit(int32_t il, int32_t e, int32_t slot, int32_t src) {
    Job j{ JOB_PROMOTE, il, e, slot, src };
    { std::lock_guard<std::mutex> lk(g_eng.mu); j.gate = g_eng.submit_gate; }
    submit_job(j);
}

void set_submit_gate(int64_t target) { std::lock_guard<std::mutex> lk(g_eng.mu); g_eng.submit_gate = target; }
void set_gate_wait(void (*wait)(int32_t, int64_t)) { std::lock_guard<std::mutex> lk(g_eng.mu); g_eng.gate_wait = wait; }

void flush() {
    {
        std::lock_guard<std::mutex> lk(g_eng.mu);
        if (g_eng.pending.empty()) return;
        if (g_eng.pending.front().gate > 0)
            std::stable_sort(g_eng.pending.begin(), g_eng.pending.end(), [](const Job & a, const Job & b) { return a.il < b.il; });
        for (const Job & j : g_eng.pending) g_eng.q[(g_eng.soft.load() && j.kind == JOB_RESTORE) ? PRIO_OTHER : prio_of(j)].push_back(j);
        g_eng.pending.clear();
        const long long depth = (long long) g_eng.queued();
        if (depth > g_eng.st.max_queue) g_eng.st.max_queue = depth;
    }
    g_eng.cv_work.notify_one();
}

void set_async_completion(int64_t (*mark)(), int64_t (*poll)(int32_t wait_all)) {
    std::lock_guard<std::mutex> lk(g_eng.mu);
    g_eng.mark_fn = mark; g_eng.poll_fn = poll;
}

size_t take_done(std::vector<Done> & out) {
    std::lock_guard<std::mutex> lk(g_eng.mu);
    g_eng.collect(false);
    const size_t n = g_eng.done.size();
    out.insert(out.end(), g_eng.done.begin(), g_eng.done.end());
    g_eng.done.clear();
    return n;
}

long long promotions_in_flight() {
    std::lock_guard<std::mutex> lk(g_eng.mu);
    g_eng.collect(false);
    long long n = (long long) (g_eng.inflight.size() + g_eng.unmarked.size()) + (g_eng.busy ? 1 : 0);
    for (const auto & d : g_eng.q) for (const Job & j : d) n += j.kind == JOB_PROMOTE;
    for (const Job & j : g_eng.pending) n += j.kind == JOB_PROMOTE;
    return n;
}

void wait_idle() {
    if (!running()) return;   // never started: nothing can be queued or in flight
    flush();
    const auto t0 = clk::now();
    std::unique_lock<std::mutex> lk(g_eng.mu);
    g_eng.cv_idle.wait(lk, [] { return g_eng.queues_empty() && !g_eng.busy; });
    g_eng.collect(true);   // batch transport: every issued copy has landed
    g_eng.st.wait_idle_ms += ms_since(t0);
    ++g_eng.st.wait_idle_calls;
}

bool exclusive_start(int32_t n_layers, int32_t n_expert, bool soft) {
    std::lock_guard<std::mutex> lk(g_eng.mu);
    if (g_eng.excl.load()) return true;
    g_eng.soft.store(soft);
    if (!g_eng.th.joinable() || n_layers <= 0 || n_expert <= 0) return false;
    const size_t n = (size_t) n_layers * (size_t) n_expert;
    g_eng.n_layers = n_layers;
    g_eng.n_expert = n_expert;
    g_eng.resident.reset(new std::atomic<uint8_t>[n]);
    for (size_t i = 0; i < n; ++i) g_eng.resident[i].store(1);
    g_eng.want_release.assign(n, 0);
    g_eng.freed.assign(n, 0);
    g_eng.excl.store(true);
    return true;
}

bool exclusive_on() { return g_eng.excl.load(); }

void submit_release(int32_t il, int32_t e) {
    std::lock_guard<std::mutex> lk(g_eng.mu);
    if (!g_eng.excl.load() || !g_eng.in_range(il, e)) return;
    g_eng.want_release[g_eng.idx(il, e)] = 1;
    g_eng.pending.push_back({ JOB_RELEASE, il, e, -1, MOERT_SRC_NONE });
}

void submit_restore(int32_t il, int32_t e, int32_t slot) {
    std::lock_guard<std::mutex> lk(g_eng.mu);
    if (!g_eng.excl.load() || !g_eng.in_range(il, e)) return;
    g_eng.want_release[g_eng.idx(il, e)] = 0;   // cancels a release that has not run yet
    g_eng.pending.push_back({ JOB_RESTORE, il, e, slot, MOERT_SRC_SLOT });
}

bool resident(int32_t il, int32_t e) {
    if (!g_eng.excl.load() || !g_eng.in_range(il, e)) return true;
    return g_eng.resident[g_eng.idx(il, e)].load() != 0;
}

void cold_guard(int32_t il, const int32_t * ids, int32_t n) {
    if (!g_eng.excl.load() || g_eng.soft.load() || !ids || il < 0 || il >= g_eng.n_layers) return;
    auto all_resident = [&] {
        for (int32_t i = 0; i < n; ++i) {
            const int32_t e = ids[i];
            if (e >= 0 && e < g_eng.n_expert && !g_eng.resident[g_eng.idx(il, e)].load()) return false;
        }
        return true;
    };
    if (all_resident()) return;   // the fast path: one load per cold expert
    flush();                      // defensive: a restore still PENDING could otherwise never run
    const auto t0 = clk::now();
    std::unique_lock<std::mutex> lk(g_eng.mu);
    g_eng.cv_res.wait(lk, all_resident);
    g_eng.st.guard_wait_ms += ms_since(t0);
    ++g_eng.st.guard_waits;
}

Stats stats() {
    std::lock_guard<std::mutex> lk(g_eng.mu);
    return g_eng.st;
}

}  // namespace moe_xfer

extern "C" void moe_xfer_cold_guard(int32_t il, const int32_t * ids, int32_t n) {
    moe_xfer::cold_guard(il, ids, n);
}
