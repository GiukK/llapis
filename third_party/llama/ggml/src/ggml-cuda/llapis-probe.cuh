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

#ifdef LLAPIS_PROBE

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
        fprintf(stderr, "LLAPIS-PROBE %s\n", buf);
        fflush(stderr);
    }
}

#define LLAPIS_PROBE_KEY(...) llapis_probe_emit(__VA_ARGS__)

#else // LLAPIS_PROBE

#define LLAPIS_PROBE_KEY(...) ((void) 0)

#endif // LLAPIS_PROBE
