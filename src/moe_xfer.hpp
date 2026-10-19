// moe_xfer -- the expert transfer engine (full-IQ2 plan phases 1-2, claude dump
// docs/speculative-runtime-design.md section 8; generalized as the single transfer engine of
// docs/design-prefill-transfer-2026-10-04.md section 2.2, phase P1).
//
// MECHANISM ONLY: prioritized queues, one worker thread and a completion list. It never decides
// anything: the policy picks (layer, expert, slot) and marks the slot FILLING (moert_policy,
// stream_expert returning 2); the host picks the SOURCE of the bytes with moert/registry.h; the host
// applies completions back into the policy and publishes the layer.
//
// SOURCES of a promotion: the owned cold arena in RAM (moert_memmgr_promote_from_arena: valid only
// when the hot and cold tiers are the same bytes, the full-IQ2 regime) or the promotion file
// (moert_memmgr_promote: the SSD, e.g. the Q4 pages of the quality regime). Uploads go through the
// slab_upload callback, which runs on the calling thread's own CUDA stream (cudaStreamPerThread):
// the decode thread never waits for a copy. While the engine is running, the decode thread must not
// call moert_memmgr_promote itself without wait_idle() first (the memmgr's read staging is shared).
//
// PRIORITIES (three queues, the worker always takes the highest non-empty one; FIFO inside each):
//   0 STAGING  -- prefill staging (design P2; no producer yet);
//   1 RESTORE  -- exclusive-arena page rebuilds, which a cold read may be waiting on;
//   2 OTHER    -- promotions and releases.
//
// ORDERING CONTRACT (the reason submit and flush are separate): a job may only start once the
// device mirror that marks its slot not-READY has been published. The policy publishes at the end
// of a refresh, so the host submits during the policy call (jobs stay PENDING, invisible to the
// worker) and flushes after the call returns. Before anything reallocates or frees the hot slab
// the host must call wait_idle().
//
// EXCLUSIVE ARENA (phase 2, opt-in through exclusive_start): each expert lives in ONE place. A hot
// expert's arena page is RELEASED (decommitted); when the expert goes cold again its page is
// RESTORED by downloading its slot (moert_memmgr_arena_restore). Both run on the worker, in FIFO
// order with the promotions, so the host's only duty is to queue a RESTORE before any job that
// overwrites the slot holding the bytes (and before any promotion that reads the page). The engine
// owns the one safety invariant: a page is readable iff resident(il, e). Every reader of the arena
// on the cold path calls cold_guard first, which returns at once when all its experts are resident
// and otherwise blocks until the queued restore lands. A RELEASE re-checks, under the engine lock,
// that no RESTORE was requested since it was queued, so a release can never overtake the host's
// decision that the expert is cold. RESTOREs run ahead of every other queued job (the reasons it is
// always safe are next to the queue in moe_xfer.cpp): a cold read may be waiting on one.
#pragma once

#include <cstdint>
#include <vector>

struct moert_memmgr;

namespace moe_xfer {

struct Done {
    int32_t il, e, slot;
    bool    ok;
};

struct Stats {
    long long submitted = 0, completed = 0, failed = 0;
    long long from_arena = 0, from_file = 0;   // completed promotions by source
    long long max_queue = 0;          // deepest queue seen at a flush
    double    copy_ms_total = 0.0;    // worker-side time inside the uploads
    double    wait_idle_ms = 0.0;     // decode-thread time spent in wait_idle
    long long wait_idle_calls = 0;
    // exclusive arena
    long long releases = 0, releases_skipped = 0, restores = 0, restores_skipped = 0, restore_failed = 0;
    double    release_ms_total = 0.0, restore_ms_total = 0.0;
    long long released_bytes = 0;     // bytes currently released (gauge)
    long long released_bytes_max = 0;
    long long guard_waits = 0;        // cold reads that had to wait for a restore
    double    guard_wait_ms = 0.0;
};

// Start the worker for this memmgr (idempotent). false = could not start.
bool start(moert_memmgr * mm);
bool running();
// true on the transfer worker thread (its promotions may go through the RAM director, findings 216)
bool on_worker();
// Batch transport (findings 216k): with a completion source set, a promotion whose copies were issued
// asynchronously (by the slab_upload callback on the worker) is reported done only when mark()'s id <= poll().
// sync_poll(1) waits for every issued copy (slab free, turn end).
void set_async_completion(int64_t (*mark)(), int64_t (*poll)(int32_t wait_all));

// Queue a promotion of expert e into hot slot `slot`, copied from `src` (moert_source: MOERT_SRC_ARENA
// or MOERT_SRC_FILE), as PENDING (not visible to the worker until flush()).
void submit(int32_t il, int32_t e, int32_t slot, int32_t src);
// Hand every PENDING job to the worker.
void flush();
// Move every finished promotion into `out` (appended); non-blocking. Returns how many were moved.
size_t take_done(std::vector<Done> & out);
// Promotions not yet reported done: queued, running, or with their copy still in flight (findings 231: how many
// copies are still travelling when the next verify starts).
long long promotions_in_flight();
// Early decision (findings 234): promotions submitted while a gate target is set carry it; the worker calls wait(target)
// before copying them (the copy may only start once the verify that was running at the decision left the slab).
void set_submit_gate(int64_t target);
void set_gate_wait(void (*wait)(int32_t il, int64_t target));
// Block until no job is pending, queued or running (pending jobs are flushed first).
void wait_idle();

// --- exclusive arena --------------------------------------------------------------------------
// Arm it (after start): every page starts resident. Idempotent. false = not started / bad geometry.
// soft = the arena is the mapped pack (moert_memmgr_cold_arena_mapped): a release only trims the page from
// the working set, so it stays readable -- cold_guard never waits and restores are prefetches in FIFO
// order with the promotions instead of jumping ahead of them.
bool exclusive_start(int32_t n_layers, int32_t n_expert, bool soft);
bool exclusive_on();
// Queue (PENDING) the release of expert e's arena page; the host has checked that e is hot and its
// slot holds its bytes.
void submit_release(int32_t il, int32_t e);
// Queue (PENDING) the rebuild of expert e's page from `slot`, which must still hold e's bytes when
// the job runs (FIFO: queue it before anything that overwrites the slot). Cancels a release of e
// that has not run yet.
void submit_restore(int32_t il, int32_t e, int32_t slot);
bool resident(int32_t il, int32_t e);
// Called by every cold-path reader of the arena before it reads experts ids[0..n) of layer il.
void cold_guard(int32_t il, const int32_t * ids, int32_t n);

Stats stats();

}  // namespace moe_xfer

// C entry point of cold_guard, registered into moert_cuda_config::cold_guard_fn (island -> host).
extern "C" void moe_xfer_cold_guard(int32_t il, const int32_t * ids, int32_t n);
