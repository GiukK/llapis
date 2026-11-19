#pragma once

// LLapis reachability probe. OFF unless the build sets -DLLAPIS_CUDA_PROBE=ON.
//
// Every pruning decision in this backend rests on the same claim: "this code cannot run
// here". Reading the dispatch is how the claim is FORMED; this probe is how it is TESTED.
// It exists because the claim was wrong twice on flash-attention head sizes -- the GGUF's
// metadata said 512, the first probe said 256, and the truth was both. A prune that is not
// preceded by an empty probe line is a guess.
//
// Each distinct key is printed once, to stderr, prefixed LLAPIS-PROBE. Grep the server's
// stderr after a full boot + the golden prompt set; anything that never appears is a
// candidate, anything that appears is not.

// OUTPUT PATH, and it is not a detail (2026-11-08). This used to write with fprintf(stderr) from
// inside ggml-cuda.dll. That is visible when the server runs in a console and SILENTLY LOST under
// the launcher, which redirects the child's streams -- a probe run then prints nothing at all,
// which is indistinguishable from "none of this code is reached". It cost three runs of the F1-bis
// re-measurement and was caught only by a control key in ggml_cuda_init, a function whose own log
// line starts every serve log. The lesson is §132's, one level up: the probe's own channel has to
// be proven before its silence means anything. GGML_LOG_INFO goes through ggml's log callback,
// which is the channel those first lines already prove works.
//
// UNFINISHED, AND DO NOT TRUST THIS AS A FIX YET (2026-11-08, same session). With the probe ON and
// this channel, `llapis serve` dies with an access violation at `initializing slots` -- on a tree
// built clean in one piece, so it is not a mixed-build artifact (that was the first, wrong guess).
// The control key, which fires during device init, prints fine; the crash arrives later, when the
// §147 keys in mmf.cu / fattn.cu fire at high frequency during the graph reserve and every one of
// them now goes through OUR log callback (src/llapis_log_filter.cpp). That is the prime suspect and
// it is NOT yet proven. Nothing shipped is affected: all of this compiles to nothing with the
// option off, which is how the release is built.
// Next step, when someone picks this up: give the cuBLAS attribution keys their OWN build option,
// so the three sites that answer the F1-bis question can run without the high-frequency §147 keys,
// and isolate the crash instead of carrying both changes at once.
#ifdef LLAPIS_PROBE

#include "ggml-impl.h"

#include <cstdarg>
#include <cstdio>
#include <mutex>
#include <set>
#include <string>

static void llapis_probe_emit(const char * fmt, ...) {
    char buf[512];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);

    // One `seen` per translation unit, so a key can repeat across .cu files. That is
    // harmless: the question asked of this output is "did it EVER appear", not how often.
    static std::mutex             mtx;
    static std::set<std::string>  seen;
    std::lock_guard<std::mutex> lock(mtx);
    if (seen.insert(std::string(buf)).second) {
        GGML_LOG_INFO("LLAPIS-PROBE %s\n", buf);
    }
}

#define LLAPIS_PROBE_KEY(...) llapis_probe_emit(__VA_ARGS__)

#else // LLAPIS_PROBE

#define LLAPIS_PROBE_KEY(...) ((void) 0)

#endif // LLAPIS_PROBE
