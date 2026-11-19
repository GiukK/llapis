// moert/seqio.h -- sequential file I/O for the multi-gigabyte transforms of `prepare`, past the OS cache.
//
// WHY. `prepare` copies 10+ GB between files on a 16 GB machine. Through stdio every byte goes
// through the Windows file cache twice (read, then written dirty), and with files this size the
// cache has to evict as fast as it fills: measured on the reference laptop, splicing the 10.3 GB
// quality base took 203 s (51 MB/s) on an SSD that reads 2.4 GB/s. Same disease as the lazy
// projector's buffered reload (findings 257), same cure: unbuffered, sector-aligned, large
// sequential transfers -- ~30 s for the same splice, byte-identical output.
//
// NOT used for the 7.9 GB expert store (gguf_tools.cpp says why): there the cache wins, because the
// pack is what `serve` reads next and buffered writes leave it in RAM.
//
// Writer: append-only from a start offset (which need not be aligned: the partial first sector is
// read back), with zero padding to an absolute offset. The file is extended in whole sectors and
// trimmed to its true size on close.
// Reader: read(offset, n) through one large aligned window; meant for mostly-forward access.
//
// MOERT_SEQIO_BUFFERED=1 in the environment keeps the same code on cached handles (for an A/B, and
// the automatic fallback when a volume refuses unbuffered access). Non-Windows builds use stdio.
// Header-only, no dependency beyond the platform: moert-memmgr must stay free of ggml/llama.
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace moert_seqio {

#ifdef _WIN32

constexpr uint64_t kSector = 4096;                   // a multiple of every sector size in use (512, 4096)
constexpr size_t   kBuf    = 16u * 1024 * 1024;      // one transfer

inline bool want_buffered() {
    const char * e = std::getenv("MOERT_SEQIO_BUFFERED");
    return e && e[0] && std::strcmp(e, "0") != 0;
}

inline bool io_at(HANDLE h, uint64_t off, void * p, DWORD n, bool write, DWORD * done) {
    OVERLAPPED ov{};
    ov.Offset     = (DWORD) (off & 0xffffffffu);
    ov.OffsetHigh = (DWORD) (off >> 32);
    return write ? WriteFile(h, p, n, done, &ov) != 0 : ReadFile(h, p, n, done, &ov) != 0;
}

// Writer: kRing buffers of kBuf, each written with an OVERLAPPED WriteFile while the caller fills the
// next one. A synchronous unbuffered writer waited for every 16 MB to reach the drive before it
// copied the next byte: measured, the 7.9 GB expert store took 28 s that way against 17-21 s through
// the file cache (which only looks faster: it returns with gigabytes still dirty in RAM).
class Writer {
public:
    static constexpr int kRing = 4;

    Writer() = default;
    Writer(const Writer &) = delete;
    Writer & operator=(const Writer &) = delete;
    ~Writer() { abandon(); }

    // start == 0 creates/truncates; start > 0 continues an existing file at that offset.
    bool open(const char * path, uint64_t start) {
        abandon();
        path_ = path;
        for (int i = 0; i < kRing; ++i) {
            bufs_[i] = (uint8_t *) VirtualAlloc(nullptr, kBuf, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
            ev_[i]   = CreateEventA(nullptr, TRUE, FALSE, nullptr);
            if (!bufs_[i] || !ev_[i]) { abandon(); return false; }
        }
        const DWORD disp = start ? OPEN_EXISTING : CREATE_ALWAYS;
        if (!want_buffered()) {
            h_ = CreateFileA(path, GENERIC_READ | GENERIC_WRITE, 0, nullptr, disp,
                             FILE_FLAG_NO_BUFFERING | FILE_FLAG_SEQUENTIAL_SCAN | FILE_FLAG_OVERLAPPED, nullptr);
        }
        if (h_ == INVALID_HANDLE_VALUE) {
            h_ = CreateFileA(path, GENERIC_READ | GENERIC_WRITE, 0, nullptr, disp,
                             FILE_FLAG_SEQUENTIAL_SCAN | FILE_FLAG_OVERLAPPED, nullptr);
        }
        if (h_ == INVALID_HANDLE_VALUE) { abandon(); return false; }
        cur_      = 0;
        file_off_ = start & ~(kSector - 1);
        fill_     = (size_t) (start - file_off_);
        if (fill_) {   // the partial sector already in the file comes back into the buffer
            if (!issue(0, file_off_, (DWORD) kSector, false) || !wait(0, fill_)) { abandon(); return false; }
        }
        return true;
    }

    uint64_t pos() const { return file_off_ + fill_; }

    bool write(const void * p, size_t n) {
        const uint8_t * s = (const uint8_t *) p;
        while (n) {
            const size_t k = n < kBuf - fill_ ? n : kBuf - fill_;
            std::memcpy(bufs_[cur_] + fill_, s, k);
            fill_ += k; s += k; n -= k;
            if (fill_ == kBuf && !flush_full()) return false;
        }
        return true;
    }

    bool pad_to(uint64_t off) {
        if (off < pos()) return false;
        uint64_t n = off - pos();
        while (n) {
            const size_t k = n < kBuf - fill_ ? (size_t) n : kBuf - fill_;
            std::memset(bufs_[cur_] + fill_, 0, k);
            fill_ += k; n -= k;
            if (fill_ == kBuf && !flush_full()) return false;
        }
        return true;
    }

    // Writes the tail, waits for every write in flight, trims the file to its true size.
    // false on any failure (the file is then suspect).
    bool close() {
        if (h_ == INVALID_HANDLE_VALUE) return false;
        const uint64_t size = pos();
        bool ok = !failed_;
        if (ok && fill_) {
            const size_t padded = (size_t) ((fill_ + kSector - 1) & ~(kSector - 1));
            std::memset(bufs_[cur_] + fill_, 0, padded - fill_);
            ok = issue(cur_, file_off_, (DWORD) padded, true);
        }
        for (int i = 0; i < kRing; ++i) {
            if (pending_[i] && !wait(i, want_[i])) ok = false;
        }
        CloseHandle(h_); h_ = INVALID_HANDLE_VALUE;
        if (ok) {   // a cached handle to set the exact length: an unbuffered one cannot end mid-sector
            HANDLE t = CreateFileA(path_.c_str(), GENERIC_WRITE, 0, nullptr, OPEN_EXISTING, 0, nullptr);
            LARGE_INTEGER li; li.QuadPart = (LONGLONG) size;
            ok = t != INVALID_HANDLE_VALUE && SetFilePointerEx(t, li, nullptr, FILE_BEGIN) && SetEndOfFile(t);
            if (t != INVALID_HANDLE_VALUE) CloseHandle(t);
        }
        abandon();
        return ok;
    }

private:
    bool issue(int i, uint64_t off, DWORD n, bool write) {
        OVERLAPPED & ov = ov_[i];
        std::memset(&ov, 0, sizeof(ov));
        ov.Offset     = (DWORD) (off & 0xffffffffu);
        ov.OffsetHigh = (DWORD) (off >> 32);
        ov.hEvent     = ev_[i];
        ResetEvent(ev_[i]);
        const BOOL r = write ? WriteFile(h_, bufs_[i], n, nullptr, &ov) : ReadFile(h_, bufs_[i], n, nullptr, &ov);
        if (!r && GetLastError() != ERROR_IO_PENDING) { failed_ = true; return false; }
        pending_[i] = true;
        want_[i]    = write ? n : 1;
        return true;
    }
    bool wait(int i, size_t at_least) {
        DWORD done = 0;
        const bool r = GetOverlappedResult(h_, &ov_[i], &done, TRUE) != 0;
        pending_[i] = false;
        if (!r || done < at_least) { failed_ = true; return false; }
        return true;
    }
    bool flush_full() {
        if (!issue(cur_, file_off_, (DWORD) kBuf, true)) return false;
        file_off_ += kBuf; fill_ = 0;
        cur_ = (cur_ + 1) % kRing;
        // the next buffer may still be on its way to the drive from kRing writes ago
        return !pending_[cur_] || wait(cur_, want_[cur_]);
    }
    void abandon() {
        if (h_ != INVALID_HANDLE_VALUE) {
            for (int i = 0; i < kRing; ++i) {
                if (pending_[i]) { DWORD d; CancelIoEx(h_, &ov_[i]); GetOverlappedResult(h_, &ov_[i], &d, TRUE); pending_[i] = false; }
            }
            CloseHandle(h_); h_ = INVALID_HANDLE_VALUE;
        }
        for (int i = 0; i < kRing; ++i) {
            if (bufs_[i]) { VirtualFree(bufs_[i], 0, MEM_RELEASE); bufs_[i] = nullptr; }
            if (ev_[i])   { CloseHandle(ev_[i]); ev_[i] = nullptr; }
            pending_[i] = false;
        }
        fill_ = 0; file_off_ = 0; cur_ = 0;
    }
    HANDLE      h_ = INVALID_HANDLE_VALUE;
    uint8_t *   bufs_[kRing] = {};
    HANDLE      ev_[kRing] = {};
    OVERLAPPED  ov_[kRing] = {};
    bool        pending_[kRing] = {};
    size_t      want_[kRing] = {};
    int         cur_ = 0;
    uint64_t    file_off_ = 0;   // where bufs_[cur_][0] lands; always sector-aligned
    size_t      fill_ = 0;
    bool        failed_ = false;
    std::string path_;
};

class Reader {
public:
    Reader() = default;
    Reader(const Reader &) = delete;
    Reader & operator=(const Reader &) = delete;
    ~Reader() { close(); }

    bool open(const char * path) {
        close();
        buf_ = (uint8_t *) VirtualAlloc(nullptr, kBuf, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
        if (!buf_) return false;
        if (!want_buffered()) {
            h_ = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                             FILE_FLAG_NO_BUFFERING | FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
        }
        if (h_ == INVALID_HANDLE_VALUE) {
            h_ = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
        }
        if (h_ == INVALID_HANDLE_VALUE) { close(); return false; }
        return true;
    }

    bool read(uint64_t off, void * dst, size_t n) {
        uint8_t * d = (uint8_t *) dst;
        while (n) {
            if (off < win_off_ || off >= win_off_ + win_len_) {
                win_off_ = off & ~(kSector - 1);
                DWORD got = 0;
                if (!io_at(h_, win_off_, buf_, (DWORD) kBuf, false, &got) || got == 0) { win_len_ = 0; return false; }
                win_len_ = got;
                if (off >= win_off_ + win_len_) return false;   // past the end of the file
            }
            const size_t avail = (size_t) (win_off_ + win_len_ - off);
            const size_t k = n < avail ? n : avail;
            std::memcpy(d, buf_ + (off - win_off_), k);
            d += k; off += k; n -= k;
        }
        return true;
    }

    void close() {
        if (h_ != INVALID_HANDLE_VALUE) { CloseHandle(h_); h_ = INVALID_HANDLE_VALUE; }
        if (buf_) { VirtualFree(buf_, 0, MEM_RELEASE); buf_ = nullptr; }
        win_off_ = 0; win_len_ = 0;
    }

private:
    HANDLE    h_ = INVALID_HANDLE_VALUE;
    uint8_t * buf_ = nullptr;
    uint64_t  win_off_ = 0;
    uint64_t  win_len_ = 0;
};

#else  // ---- not Windows: stdio, the behaviour this replaced ------------------------------------

class Writer {
public:
    ~Writer() { if (f_) std::fclose(f_); }
    bool open(const char * path, uint64_t start) {
        f_ = std::fopen(path, start ? "r+b" : "wb");
        pos_ = start;
        return f_ && (start == 0 || fseeko(f_, (off_t) start, SEEK_SET) == 0);
    }
    uint64_t pos() const { return pos_; }
    bool write(const void * p, size_t n) { pos_ += n; return std::fwrite(p, 1, n, f_) == n; }
    bool pad_to(uint64_t off) {
        static const char z[4096] = {};
        if (off < pos_) return false;
        while (pos_ < off) { const size_t k = (size_t) (off - pos_ < sizeof(z) ? off - pos_ : sizeof(z)); if (!write(z, k)) return false; }
        return true;
    }
    bool close() {
        if (!f_) return false;
        bool ok = std::fflush(f_) == 0 && !std::ferror(f_);
        if (std::fclose(f_) != 0) ok = false;
        f_ = nullptr;
        return ok;
    }
private:
    FILE * f_ = nullptr;
    uint64_t pos_ = 0;
};

class Reader {
public:
    ~Reader() { close(); }
    bool open(const char * path) { close(); f_ = std::fopen(path, "rb"); return f_ != nullptr; }
    bool read(uint64_t off, void * dst, size_t n) { return fseeko(f_, (off_t) off, SEEK_SET) == 0 && std::fread(dst, 1, n, f_) == n; }
    void close() { if (f_) { std::fclose(f_); f_ = nullptr; } }
private:
    FILE * f_ = nullptr;
};

#endif

}  // namespace moert_seqio
