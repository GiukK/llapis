#pragma once

#include <cstdint>
#include <cstdio>
#include <vector>

struct ggml_tensor;

// ---------------------------------------------------------------------------
// Low-level I/O helpers shared across translation units.
// ---------------------------------------------------------------------------
namespace io {

// 64-bit FILE* seek/tell that work on both Win32 (_fseeki64/_ftelli64) and
// POSIX (fseeko/ftello). Centralized so the platform #ifdef lives in exactly
// one place instead of being repeated at every seek site. `origin` is one of
// SEEK_SET / SEEK_CUR / SEEK_END. seek64 returns true on success.
bool    seek64(FILE * f, int64_t off, int origin);
int64_t tell64(FILE * f);

// Obtain a host-readable pointer to an I32 tensor's data. If the tensor lives
// on a host buffer this is just t->data; otherwise the bytes are copied out
// into `scratch` (resized as needed) and a pointer into it is returned. The
// returned pointer is valid until the next call that reuses the same scratch.
const int32_t * host_ids(ggml_tensor * t, std::vector<uint8_t> & scratch);

}  // namespace io
