#pragma once
// ===========================================================================
// llapis_log_filter — C0 CLI cleanup: normal-mode terminal quiet, full raw log.
// C5: PHASE tracing — `-Debug` shows the runtime's logical flow, `-Trace` the firehose.
// ===========================================================================
// Driven by two env vars the PowerShell launcher sets:
//   LLAPIS_LOG       = normal | verbose | debug | trace   (default normal)
//   LLAPIS_LOG_FILE  = path to an append-only full log (shared by router/child)
//
// Levels (terminal mirroring; the full raw log always gets everything):
//   normal   quiet: curated wrapper output only; errors still show.
//   verbose  + llama/ggml WARN lines.
//   debug    PHASE mode (C5): the terminal shows ONE clean line per logical-flow
//            step (llapis_phase below) + WARN/ERROR. The raw stderr firehose
//            (fprintf + std::cerr, incl. the CUDA island telemetry) is captured
//            into the log file instead of the terminal; phase lines are written
//            to a saved handle of the ORIGINAL stderr so they still reach the
//            console (or the router pipe, for a child).
//   trace    the pre-C5 `-Debug`: full raw firehose on the terminal.
//
// When LLAPIS_LOG_FILE is set, init() takes over llama/ggml logging AFTER
// common_init() (which installs its own callback): every raw llama/ggml line is
// appended to the log file, and only the level-appropriate subset is mirrored to
// the terminal. In quiet (normal + interactive TTY) mode it also redirects
// std::cerr (the moe_dual init banners) into the same log file. stdout is never
// touched — generated tokens stay byte-identical.
//
// No-op when LLAPIS_LOG_FILE is unset: dev/gate runs keep the exact common_init
// logging path (parity/C0g preserved). llapis_phase still prints to stderr when
// LLAPIS_LOG=debug|trace is set without a log file (bare-exe debugging).

#include <cstdio>

// Install the filter (llama_log_set + ggml_log_set + optional std::cerr redirect).
// Call ONCE, after common_init() and before the model loads. Idempotent.
void llapis_log_filter_init();

// True in "quiet" mode: LLAPIS_LOG=normal AND stderr is an interactive TTY.
// Callers route their own startup banners to llapis_log_file() when this is true.
bool llapis_quiet();

// The raw full-log FILE* (append), or nullptr if the filter is inactive.
std::FILE * llapis_log_file();

// True when the live in-place status line should render: filter active, stderr is an interactive
// TTY, level is normal/verbose, and LLAPIS_NO_STATUS is unset. (stdout is never touched.)
bool llapis_status_enabled();

// C5 PHASE tracing: one line per logical-flow step ("Pinning IQ2 experts ...",
// "Q4 batched prefill ...", ">>> Refresh ..."). Always appended to the raw log
// (prefixed "[PHASE]"); mirrored to the terminal only at level debug/trace, with
// a process-relative "[+  12.3s]" timestamp. printf-style. Thread-safe.
void llapis_phase(const char * fmt, ...);

// True when phase lines are mirrored to the terminal (level debug or trace) —
// lets callers skip building expensive stats strings otherwise.
bool llapis_phase_enabled();

// C5.2 single-instance guard (Windows named mutex, Local\ = per-login-session).
// Returns true if THIS process now owns `name` (handle held for the process
// lifetime, released by the OS at exit — crash-safe, no stale lockfiles).
// Returns false if another live process already holds it. Always true off-Windows.
// Two names are in use:
//   "Local\\LLapis-router" — the HTTP router on this box (one per session)
//   "Local\\LLapis-model"  — any process that loads a full model into RAM/VRAM
bool llapis_singleton_acquire(const char * name);
