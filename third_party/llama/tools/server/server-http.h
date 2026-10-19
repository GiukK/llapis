#pragma once

#include <atomic>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <vector>
#include <cstdint>
#include <cctype>
#include <unordered_map>

struct common_params;
struct stream_pipe_producer; // defined in server-stream.h

// generator-like API for HTTP response generation
// this object response with one of the 2 modes:
// 1) normal response: `data` contains the full response body
// 2) streaming response: each call to next(output) generates the next chunk
//    when next(output) returns false, no more data after the current chunk
//    note: some chunks can be empty, in which case no data is sent for that chunk
struct server_http_res {
    std::string content_type = "application/json; charset=utf-8";
    int status = 200;
    std::string data;
    std::map<std::string, std::string> headers;

    // if set, the stream survives a client disconnect: the producer pipe keeps draining into the
    // ring buffer and finalizes the session on destruction, so no explicit on_stream_end is needed.
    // shared_ptr (not unique_ptr) so the forward-declared type is safe to delete here.
    std::shared_ptr<stream_pipe_producer> spipe;

    std::function<bool(std::string &)> next = nullptr;
    bool is_stream() const {
        return next != nullptr;
    }

    // called when the session is cancelled (e.g. DELETE /v1/stream/<conv_id>).
    // server_res_generator overrides this to stop its reader; the default is a no-op.
    virtual void stop() {}

    virtual ~server_http_res() = default;
};

// unique pointer, used by set_chunked_content_provider
// httplib requires the stream provider to be stored in heap
using server_http_res_ptr = std::unique_ptr<server_http_res>;
using raw_buffer = std::vector<uint8_t>;

struct uploaded_file {
    raw_buffer data;
    std::string filename;
    std::string content_type;
};

struct server_http_req {
    std::map<std::string, std::string> params; // path_params + query_params
    std::map<std::string, std::string> headers; // used by MCP proxy
    std::string path;
    std::string query_string; // query parameters string (e.g. "action=save")
    std::string body;
    std::map<std::string, uploaded_file> files; // used for file uploads (form data)
    const std::function<bool()> & should_stop;

    std::string get_param(const std::string & key, const std::string & def = "") const {
        auto it = params.find(key);
        if (it != params.end()) {
            return it->second;
        }
        return def;
    }

    // Case-insensitive header lookup. `headers` above is a case-SENSITIVE std::map filled
    // from httplib, which preserves the sender's casing — so headers.find("Origin") misses
    // a client that wrote "origin". Anything that makes a decision on a header must come
    // through here.
    std::string get_header(const std::string & key, const std::string & def = "") const {
        for (const auto & [k, v] : headers) {
            if (k.size() != key.size()) {
                continue;
            }
            bool eq = true;
            for (size_t i = 0; i < k.size(); ++i) {
                if (std::tolower((unsigned char) k[i]) != std::tolower((unsigned char) key[i])) {
                    eq = false;
                    break;
                }
            }
            if (eq) {
                return v;
            }
        }
        return def;
    }
};

// ===========================================================================
// [llapis] Authorization for the one destructive route: POST /moe/exit.
// ===========================================================================
// The web UI has a power button, and it stops the runtime — so this is the single route
// where a cross-site request would do real damage. Three independent conditions, none of
// which a page on another origin can satisfy:
//
//   0. The route is REGISTERED ONLY on a loopback bind. Serve on a LAN address and
//      /moe/exit and /moe/exit-token simply do not exist (404), and the UI hides the
//      button because the token fetch fails. There is no environment variable, header or
//      flag that turns them back on.
//   1. A per-PROCESS nonce, handed out by GET /moe/exit-token. It is generated at startup,
//      never written to disk, never baked into the UI bundle, and dies with the process.
//      The Same-Origin Policy is what protects it: a page on evil.com may *send* a request
//      to our port, but it may not *read* the response, so it cannot learn the nonce.
//   2. An Origin header exactly equal to the origin we are serving — scheme, loopback
//      host and the real listening port. Browsers always send Origin on POST and page
//      script cannot forge it.
//
// (1) alone would fall to a nonce that leaked into a log; (2) alone would fall to any
// non-browser client that just writes the header. Together they mean: only a document
// this process served, in a browser, can stop it.
//
// `llapis stop` deliberately does NOT use this route — it identifies the process that owns
// the port and verifies its image path and start time. A local shell has local authority
// already; this guard exists for the browser, not for the shell.
inline constexpr const char * LLAPIS_EXIT_TOKEN_HEADER = "X-LLapis-Exit-Token";

enum class exit_auth {
    ok = 0,
    not_local,   // bind is not loopback — the route should not even be registered
    bad_host,    // Host absent, not loopback, or not our port (DNS rebinding)
    bad_token,   // nonce absent or wrong
    bad_origin,  // Origin absent, or not exactly our own origin
};

inline const char * exit_auth_reason(exit_auth a) {
    switch (a) {
        case exit_auth::ok:         return "ok";
        case exit_auth::not_local:  return "server is not bound to loopback";
        case exit_auth::bad_host:   return "Host is not this loopback server";
        case exit_auth::bad_token:  return "missing or invalid exit token";
        case exit_auth::bad_origin: return "missing or foreign Origin";
    }
    return "unknown";
}

// True when `authority` — a Host header, or the authority part of an Origin — names a
// loopback address. Accepts an optional :port and bracketed IPv6.
inline bool server_http_is_loopback_authority(const std::string & authority) {
    std::string h = authority;
    // strip an optional :port ("[::1]:8080" / "127.0.0.1:8080" / "localhost")
    if (!h.empty() && h[0] == '[') {                  // bracketed IPv6
        const size_t close = h.find(']');
        if (close == std::string::npos) {
            return false;
        }
        h = h.substr(1, close - 1);
    } else {
        const size_t colon = h.find(':');
        if (colon != std::string::npos && h.find(':', colon + 1) == std::string::npos) {
            h = h.substr(0, colon);                   // exactly one ':' = v4/host + port
        }
    }
    if (h == "localhost" || h == "::1") {
        return true;
    }
    // 127.0.0.0/8, and only as a real dotted quad. The obvious `h.rfind("127.", 0) == 0`
    // also accepts "127.evil.com" — which is precisely the name a DNS-rebinding attack
    // registers, so it is a hole rather than a shortcut.
    int first = -1;
    int idx = 0, digits = 0, val = 0;
    for (size_t i = 0; i <= h.size(); ++i) {
        if (i == h.size() || h[i] == '.') {
            if (digits == 0 || idx > 3) {
                return false;
            }
            if (idx == 0) {
                first = val;
            }
            ++idx;
            val    = 0;
            digits = 0;
        } else if (h[i] >= '0' && h[i] <= '9') {
            val = val * 10 + (h[i] - '0');
            if (++digits > 3 || val > 255) {
                return false;
            }
        } else {
            return false;
        }
    }
    return idx == 4 && first == 127;
}

// The port in `authority`, or -1 when it carries none.
inline int server_http_authority_port(const std::string & authority) {
    size_t colon = std::string::npos;
    if (!authority.empty() && authority[0] == '[') {  // bracketed IPv6: the port follows ']'
        const size_t close = authority.find(']');
        if (close == std::string::npos) {
            return -1;
        }
        colon = authority.find(':', close);
    } else {
        colon = authority.find(':');
        // a bare IPv6 literal has several colons and no port
        if (colon != std::string::npos && authority.find(':', colon + 1) != std::string::npos) {
            return -1;
        }
    }
    if (colon == std::string::npos || colon + 1 >= authority.size()) {
        return -1;
    }
    int p = 0;
    for (size_t i = colon + 1; i < authority.size(); ++i) {
        if (authority[i] < '0' || authority[i] > '9') {
            return -1;
        }
        p = p * 10 + (authority[i] - '0');
        if (p > 65535) {
            return -1;
        }
    }
    return p;
}

// The whole decision, as a pure function so it can be tested without a socket.
// See tests/test_exit_guard.cpp.
inline exit_auth server_http_check_exit_auth(
        bool                loopback_bind,
        const std::string & expected_token,
        int                 expected_port,
        bool                is_ssl,
        const std::string & host,
        const std::string & origin,
        const std::string & token) {
    if (!loopback_bind || expected_token.empty()) {
        return exit_auth::not_local;
    }
    if (host.empty() || !server_http_is_loopback_authority(host)) {
        return exit_auth::bad_host;
    }
    // The port must be the one we are actually listening on. An absent port means the
    // scheme default, which is never ours in practice but is checked rather than assumed.
    const int host_port = server_http_authority_port(host);
    if ((host_port < 0 ? (is_ssl ? 443 : 80) : host_port) != expected_port) {
        return exit_auth::bad_host;
    }
    // Constant-time: the nonce is not a password (the process dies on success and a new one
    // is generated on the next run) but there is no reason to leak its prefix either.
    if (token.size() != expected_token.size()) {
        return exit_auth::bad_token;
    }
    unsigned char diff = 0;
    for (size_t i = 0; i < token.size(); ++i) {
        diff |= (unsigned char) (token[i] ^ expected_token[i]);
    }
    if (diff != 0 || token.empty()) {
        return exit_auth::bad_token;
    }
    // Byte-exact, not "starts with" and not "is also loopback": our own origin only.
    if (origin.empty() || origin != std::string(is_ssl ? "https://" : "http://") + host) {
        return exit_auth::bad_origin;
    }
    return exit_auth::ok;
}

struct server_http_context {
    class Impl;
    std::unique_ptr<Impl> pimpl;

    std::thread thread; // server thread
    std::atomic<bool> is_ready = false;

    // note: the handler should never throw exceptions
    using handler_t = std::function<server_http_res_ptr(const server_http_req & req)>;
    mutable std::unordered_map<std::string, handler_t> handlers;

    std::string path_prefix;
    std::string hostname;
    int port    = 8080;
    bool is_ssl = false;

    // [llapis] Set by init(). `loopback_bind` decides whether the shutdown routes exist at
    // all; `exit_token` is the per-process nonce described above. Both are read by
    // server.cpp at route-registration time, which runs after init().
    bool        loopback_bind = false;
    std::string exit_token;

    server_http_context();
    ~server_http_context();

    bool init(const common_params & params);
    bool start();
    void stop() const;

    void get(const std::string & path, const handler_t & handler) const;
    void post(const std::string & path, const handler_t & handler) const;
    void del(const std::string & path, const handler_t & handler) const;

    // Register the Google Cloud Platform (Vertex AI) compat (AIP_PREDICT_ROUTE env var, or /predict)
    // Must be called AFTER all other API routes are registered
    void register_gcp_compat() const;

    // for debugging
    std::string listening_address;
};
