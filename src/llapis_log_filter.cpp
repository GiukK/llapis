// ===========================================================================
// llapis_log_filter — see llapis_log_filter.hpp.
// ===========================================================================
#include "llapis_log_filter.hpp"

#include "llama.h"
#include "ggml.h"

#include <chrono>
#include <cstdarg>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <mutex>
#include <set>
#include <streambuf>
#include <string>

#ifdef _WIN32
  #include <io.h>
  #include <share.h>
  #define LLAPIS_ISATTY(fd)  _isatty(fd)
  #define LLAPIS_FILENO(f)   _fileno(f)
  #define LLAPIS_DUP(fd)     _dup(fd)
  #define LLAPIS_FDOPEN(fd, m) _fdopen(fd, m)
#else
  #include <unistd.h>
  #define LLAPIS_ISATTY(fd)  isatty(fd)
  #define LLAPIS_FILENO(f)   fileno(f)
  #define LLAPIS_DUP(fd)     dup(fd)
  #define LLAPIS_FDOPEN(fd, m) fdopen(fd, m)
#endif

namespace {

enum class Level { NORMAL, VERBOSE, DEBUG, TRACE };

Level       g_level       = Level::NORMAL;
std::FILE * g_logf        = nullptr;   // append-only full log (raw)
std::FILE * g_console     = nullptr;   // DEBUG (phase) mode: saved handle of the ORIGINAL stderr
bool        g_stderr_tty  = false;
bool        g_active      = false;
bool        g_child_silent = false;    // router child in normal/verbose: ALL stderr -> log, none to pipe
std::mutex  g_mtx;
std::string g_accum;                   // partial-line accumulator
int         g_last_level  = GGML_LOG_LEVEL_INFO;

const std::chrono::steady_clock::time_point g_t0 = std::chrono::steady_clock::now();

// A std::streambuf that forwards to a C FILE* — lets us redirect std::cerr (the
// moe_dual init banners) into the same raw log file, single handle, line-flushed.
struct FileStreambuf : std::streambuf {
    std::FILE * f = nullptr;
    int overflow(int c) override {
        if (c != EOF && f) { std::fputc(c, f); if (c == '\n') std::fflush(f); }
        return c;
    }
    std::streamsize xsputn(const char * s, std::streamsize n) override {
        if (!f) return n;
        const std::streamsize w = (std::streamsize) std::fwrite(s, 1, (size_t) n, f);
        if (n && s[n - 1] == '\n') std::fflush(f);
        return w;
    }
};
FileStreambuf g_cerrbuf;

// Benign errors that are NOT fatal in our config — suppressed from the terminal in
// normal/verbose mode (still written raw to the log). Everything else at ERROR shows.
bool is_benign_error(const std::string & line) {
    return line.find("failed to fit params to free device memory") != std::string::npos;
}

// Decide whether a completed line (at `level`) should be MIRRORED to the terminal.
bool should_mirror(int level, const std::string & line) {
    if (!g_stderr_tty && !g_console)  return true;  // redirected (bench/pipe): mirror all, filter is pointless on a file
    if (g_level == Level::TRACE)      return true;  // trace: full firehose
    if (level == GGML_LOG_LEVEL_ERROR) return !is_benign_error(line);   // real errors always show
    if (g_level == Level::VERBOSE || g_level == Level::DEBUG)
        return level >= GGML_LOG_LEVEL_WARN;        // verbose/debug: warnings + errors, no INFO spam
    return false;                                   // normal: drop all INFO/WARN loader/CUDA noise
}

void emit_line(int level, const std::string & line) {
    if (g_logf) {                               // raw, always, to the full log
        std::fwrite(line.data(), 1, line.size(), g_logf);
        std::fputc('\n', g_logf);
        std::fflush(g_logf);
    }
    if (should_mirror(level, line)) {
        // DEBUG (phase) mode redirected the real stderr into the log file; the console
        // channel lives on the saved handle. Everywhere else stderr IS the terminal.
        std::FILE * out = g_console ? g_console : stderr;
        std::fputs(line.c_str(), out);
        std::fputc('\n', out);
        std::fflush(out);
    }
}

void log_cb(enum ggml_log_level level, const char * text, void * /*user*/) {
    if (!text) return;
    std::lock_guard<std::mutex> lk(g_mtx);
    int lvl = (int) level;
    if (lvl == GGML_LOG_LEVEL_CONT) lvl = g_last_level; else g_last_level = lvl;
    g_accum += text;
    size_t pos;
    while ((pos = g_accum.find('\n')) != std::string::npos) {
        emit_line(lvl, g_accum.substr(0, pos));
        g_accum.erase(0, pos + 1);
    }
}

std::FILE * open_append(const char * path) {
#ifdef _WIN32
    // shared append so the router + its child + plan/prepare subprocesses can all
    // target the same LLAPIS_LOG_FILE without denying each other.
    return _fsopen(path, "a", _SH_DENYNO);
#else
    return std::fopen(path, "a");   // POSIX O_APPEND writes are atomic per call
#endif
}

}  // namespace

void llapis_log_filter_init() {
    // One-time setup (open log, resolve level/TTY, redirect std::cerr).
    if (!g_active) {
        const char * lf = std::getenv("LLAPIS_LOG_FILE");
        if (!lf || !lf[0]) return;              // inactive: keep the stock common_init logger (parity)

        const char * lvl = std::getenv("LLAPIS_LOG");
        if      (lvl && std::strcmp(lvl, "trace")   == 0) g_level = Level::TRACE;
        else if (lvl && std::strcmp(lvl, "debug")   == 0) g_level = Level::DEBUG;
        else if (lvl && std::strcmp(lvl, "verbose") == 0) g_level = Level::VERBOSE;
        else g_level = Level::NORMAL;

        g_logf = open_append(lf);
        if (!g_logf) {                          // can't open the log: leave logging untouched
            std::fprintf(stderr, "[llapis] warning: cannot open LLAPIS_LOG_FILE '%s' — logging unchanged\n", lf);
            return;
        }
        g_stderr_tty = LLAPIS_ISATTY(LLAPIS_FILENO(stderr)) != 0;
        g_active = true;

        // Router child in normal/verbose serve: the child's stdout+stderr are a single pipe the
        // router forwards to the console. Redirect ALL of the child's stderr (the llama/ggml
        // firehose + std::cerr banners) into the log file so NOTHING firehose-y reaches that pipe;
        // stdout keeps the CMD_CHILD_TO_ROUTER_STATE readiness protocol intact. -Trace keeps the
        // firehose (falls through to the normal mirroring path); -Debug uses the PHASE capture below.
        const bool is_router_child = std::getenv("LLAMA_SERVER_ROUTER_PORT") != nullptr;
        if (is_router_child && (g_level == Level::NORMAL || g_level == Level::VERBOSE)) {
            g_child_silent = true;
            std::freopen(lf, "a", stderr);   // C stderr -> log file (common_init's logger follows it)
            g_cerrbuf.f = stderr;            // std::cerr -> the same (freopen'd) stderr == log file
            std::cerr.rdbuf(&g_cerrbuf);
            return;                          // no mirroring callback: everything is log-only
        }

        // C5 PHASE mode (-Debug): the terminal shows phase lines + WARN/ERROR only. Save a
        // duplicate of the ORIGINAL stderr (the console, or the router pipe for a child) as the
        // phase/mirror channel, then point the real stderr — every fprintf(stderr) firehose line,
        // moert-server telemetry, CUDA-island prints — at the log file, and std::cerr with it.
        // Only when stderr is a console or the router pipe: a bench-style file redirect must keep
        // receiving the full mirror (the ps1 parses the loader lines out of it).
        if (g_level == Level::DEBUG && (g_stderr_tty || is_router_child)) {
            const int fd = LLAPIS_DUP(LLAPIS_FILENO(stderr));
            if (fd >= 0) g_console = LLAPIS_FDOPEN(fd, "w");
            if (g_console) {
                std::freopen(lf, "a", stderr);
                g_cerrbuf.f = stderr;
                std::cerr.rdbuf(&g_cerrbuf);
            }
            // (if the dup failed we keep stderr untouched — phases + firehose both show; better loud)
        }

        // quiet mode: send the C++-stream init banners (moe_dual std::cerr) to the log too
        if (llapis_quiet()) {
            g_cerrbuf.f = g_logf;
            std::cerr.rdbuf(&g_cerrbuf);
        }
    }
    // (Re)assert the llama+ggml callback on EVERY call: the fork server's own common_init()
    // runs after our first call and would otherwise reinstall its default logger. Cheap + safe.
    // (Skipped for a silenced router child — its stderr already points at the log file.)
    if (g_active && !g_child_silent) {
        llama_log_set(log_cb, nullptr);
        ggml_log_set(log_cb, nullptr);
    }
}

bool llapis_quiet() {
    return g_active && g_level == Level::NORMAL && g_stderr_tty;
}

std::FILE * llapis_log_file() {
    return g_active ? g_logf : nullptr;
}

bool llapis_status_enabled() {
    if (!g_active || !g_stderr_tty) return false;
    if (g_level == Level::DEBUG || g_level == Level::TRACE) return false;   // phases/firehose own the terminal
    const char * ns = std::getenv("LLAPIS_NO_STATUS");
    return !(ns && ns[0] && ns[0] != '0');
}

bool llapis_phase_enabled() {
    if (g_active) return g_level == Level::DEBUG || g_level == Level::TRACE;
    // filter inactive (no LLAPIS_LOG_FILE): honor the level env alone (bare-exe debugging)
    const char * lvl = std::getenv("LLAPIS_LOG");
    return lvl && (std::strcmp(lvl, "debug") == 0 || std::strcmp(lvl, "trace") == 0);
}

void llapis_phase(const char * fmt, ...) {
    char msg[1024];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(msg, sizeof(msg), fmt, ap);
    va_end(ap);

    std::lock_guard<std::mutex> lk(g_mtx);
    if (g_active && g_logf) {
        std::fprintf(g_logf, "[PHASE] %s\n", msg);
        std::fflush(g_logf);
    }
    if (!llapis_phase_enabled()) return;
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - g_t0).count();
    std::FILE * out = g_console ? g_console : stderr;
    std::fprintf(out, "[+%7.1fs] %s\n", s, msg);
    std::fflush(out);
}

// --- C5.2 single-instance guard (see llapis_log_filter.hpp) ---------------------------
#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
bool llapis_singleton_acquire(const char * name) {
    // Idempotent per process: a name this process already owns re-acquires as true
    // (CreateMutex reports ERROR_ALREADY_EXISTS even to the owning process, so both
    // the exe entry point and init_dynamic can guard without tripping each other).
    static std::mutex        s_mtx;
    static std::set<std::string> s_owned;
    std::lock_guard<std::mutex> lk(s_mtx);
    if (s_owned.count(name)) return true;
    // The handle is deliberately kept open for the whole process lifetime: the OS
    // releases the mutex when the process dies (clean exit OR crash), so a stale
    // guard can never outlive its owner the way a lockfile would.
    HANDLE h = CreateMutexA(nullptr, TRUE, name);
    if (h == nullptr) return true;   // ACL/namespace failure: fail OPEN, never block startup
    if (GetLastError() == ERROR_ALREADY_EXISTS) {
        CloseHandle(h);
        return false;
    }
    s_owned.insert(name);
    return true;
}
#else
bool llapis_singleton_acquire(const char *) { return true; }
#endif
