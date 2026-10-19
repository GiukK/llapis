// CPU unit test for the POST /moe/exit authorization decision. Milliseconds, no deps, no
// socket: the decision is a pure function in tools/server/server-http.h precisely so the
// matrix below can be checked without standing a server up.
//
// The live half of the contract -- that GET /moe/exit is a 404, that the routes are absent
// on a LAN bind, and that a real browser POST is accepted -- is asserted over HTTP by the
// smoke harness. This file covers the decision itself, exhaustively.
#include "server-http.h"

#include <cstdio>
#include <string>

static int g_fail = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++g_fail; } } while (0)

// The shape of a real request from the UI we serve on 127.0.0.1:8080.
static const std::string TOK  = "0123456789abcdef0123456789abcdef";
static const std::string HOST = "127.0.0.1:8080";
static const std::string ORIG = "http://127.0.0.1:8080";

static exit_auth check(const std::string & host, const std::string & origin, const std::string & token,
                       bool loopback = true, int port = 8080, bool ssl = false) {
    return server_http_check_exit_auth(loopback, TOK, port, ssl, host, origin, token);
}

int main() {
    // --- the authority parser, which everything else rests on ---------------------------
    CHECK(server_http_is_loopback_authority("127.0.0.1"));
    CHECK(server_http_is_loopback_authority("127.0.0.1:8080"));
    CHECK(server_http_is_loopback_authority("localhost"));
    CHECK(server_http_is_loopback_authority("localhost:8080"));
    CHECK(server_http_is_loopback_authority("127.5.5.5"));      // all of 127/8 is loopback
    CHECK(server_http_is_loopback_authority("[::1]:8080"));
    CHECK(server_http_is_loopback_authority("::1"));
    CHECK(!server_http_is_loopback_authority(""));
    CHECK(!server_http_is_loopback_authority("evil.com"));
    CHECK(!server_http_is_loopback_authority("evil.com:8080"));
    CHECK(!server_http_is_loopback_authority("192.168.1.4:8080"));
    // the near-misses a rebinding attempt actually uses
    CHECK(!server_http_is_loopback_authority("127.0.0.1.evil.com"));
    CHECK(!server_http_is_loopback_authority("localhost.evil.com"));
    CHECK(!server_http_is_loopback_authority("127.evil.com"));  // the "127." prefix hole
    CHECK(!server_http_is_loopback_authority("127.0.0"));       // three octets is not a quad
    CHECK(!server_http_is_loopback_authority("127.0.0.1.1"));   // five is not a quad either
    CHECK(!server_http_is_loopback_authority("127.0.0.256"));   // octet out of range
    CHECK(!server_http_is_loopback_authority("128.0.0.1"));     // adjacent /8, not loopback
    CHECK(!server_http_is_loopback_authority("[::1"));          // unterminated bracket

    CHECK(server_http_authority_port("127.0.0.1:8080") == 8080);
    CHECK(server_http_authority_port("[::1]:8080")     == 8080);
    CHECK(server_http_authority_port("localhost")      == -1);
    CHECK(server_http_authority_port("::1")            == -1);  // bare v6, not a port
    CHECK(server_http_authority_port("127.0.0.1:")     == -1);
    CHECK(server_http_authority_port("127.0.0.1:80x")  == -1);
    CHECK(server_http_authority_port("127.0.0.1:99999")== -1);  // out of range, not truncated

    // --- the happy path, in each spelling a browser can produce -------------------------
    CHECK(check(HOST, ORIG, TOK) == exit_auth::ok);
    CHECK(check("localhost:8080", "http://localhost:8080", TOK) == exit_auth::ok);
    CHECK(check("[::1]:8080",     "http://[::1]:8080",     TOK) == exit_auth::ok);

    // --- LAN mode: the routes are not registered, and the decision refuses anyway -------
    CHECK(check(HOST, ORIG, TOK, /*loopback=*/false) == exit_auth::not_local);
    // no nonce generated == same answer, whatever the headers say
    CHECK(server_http_check_exit_auth(true, "", 8080, false, HOST, ORIG, TOK) == exit_auth::not_local);

    // --- token: missing, wrong, truncated, over-long ------------------------------------
    CHECK(check(HOST, ORIG, "")                                   == exit_auth::bad_token);
    CHECK(check(HOST, ORIG, "ffffffffffffffffffffffffffffffff")    == exit_auth::bad_token);
    CHECK(check(HOST, ORIG, TOK.substr(0, 31))                     == exit_auth::bad_token);
    CHECK(check(HOST, ORIG, TOK + "0")                             == exit_auth::bad_token);
    // a correct PREFIX is not a correct token (the compare is length-first, then all bytes)
    CHECK(check(HOST, ORIG, "0123456789abcdef0123456789abcdee")    == exit_auth::bad_token);

    // --- Origin: absent, foreign, near-miss, wrong scheme, wrong port -------------------
    CHECK(check(HOST, "", TOK)                          == exit_auth::bad_origin);
    CHECK(check(HOST, "http://evil.com", TOK)           == exit_auth::bad_origin);
    CHECK(check(HOST, "https://127.0.0.1:8080", TOK)    == exit_auth::bad_origin);  // scheme
    CHECK(check(HOST, "http://127.0.0.1:8081", TOK)     == exit_auth::bad_origin);  // port
    CHECK(check(HOST, "http://localhost:8080", TOK)     == exit_auth::bad_origin);  // != Host
    CHECK(check(HOST, "null", TOK)                      == exit_auth::bad_origin);  // sandboxed frame
    CHECK(check(HOST, "http://127.0.0.1:8080/", TOK)    == exit_auth::bad_origin);  // trailing slash
    CHECK(check(HOST, "http://127.0.0.1:8080.evil.com", TOK) == exit_auth::bad_origin);

    // --- Host: absent, foreign, rebinding, and the port we are NOT listening on ---------
    CHECK(check("", ORIG, TOK)                                  == exit_auth::bad_host);
    CHECK(check("evil.com:8080", "http://evil.com:8080", TOK)    == exit_auth::bad_host);
    // DNS rebinding: the socket is ours, the name in the Host header is not
    CHECK(check("127.0.0.1.evil.com:8080", ORIG, TOK)            == exit_auth::bad_host);
    // right host, wrong port -> a second LLapis on 8081 cannot be stopped through this one
    CHECK(check("127.0.0.1:8081", "http://127.0.0.1:8081", TOK)  == exit_auth::bad_host);
    // no port in Host means the scheme default, which is never the port we bind
    CHECK(check("127.0.0.1", "http://127.0.0.1", TOK)            == exit_auth::bad_host);
    // ...unless we really are on 80, which is a legal (if unusual) configuration
    CHECK(check("127.0.0.1", "http://127.0.0.1", TOK, true, 80)  == exit_auth::ok);

    // --- precedence: Host is judged before the token, the token before the Origin -------
    // A refusal must not tell a caller which of several things it got right.
    CHECK(check("evil.com", "http://evil.com", "wrong") == exit_auth::bad_host);
    CHECK(check(HOST, "http://evil.com", "wrong")       == exit_auth::bad_token);

    if (g_fail) {
        std::fprintf(stderr, "test-exit-guard: %d FAILED\n", g_fail);
        return 1;
    }
    std::printf("test-exit-guard: all checks passed\n");
    return 0;
}
