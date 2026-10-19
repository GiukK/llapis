#include "server-context.h"
#include "server-http.h"
#include "server-models.h"
#include "server-cors-proxy.h"
#include "server-stream.h"
#include "server-tools.h"
#include "server-moe-hooks.h"

#include "arg.h"
#include "build-info.h"
#include "common.h"
#include "fit.h"
#include "llama.h"
#include "log.h"

#include <atomic>
#include <clocale>
#include <exception>
#include <signal.h>
#include <thread> // for std::thread::hardware_concurrency

#if defined(_WIN32)
#include <windows.h>
#endif

// moe_dual trampolines (see server-moe-hooks.h) — defined here so they live in the
// llama-server-impl target and the setter is exported from its DLL.
void (*g_srv_moe_on_prompt_tokens)(int) = nullptr;
void (*g_srv_moe_on_prompt_end)()       = nullptr;
void (*g_srv_moe_on_token_end)()        = nullptr;
void (*g_srv_moe_on_turn_end)()         = nullptr;
void (*g_srv_moe_on_ready)()            = nullptr;
void (*g_srv_moe_on_ctx)(struct llama_context *, const struct llama_model *) = nullptr;
void (*g_srv_moe_on_spec_ctx)(struct llama_context *, const struct llama_model *) = nullptr;
int  (*g_srv_moe_json)(const char *, char *, int) = nullptr;

void llama_server_set_moe_hooks(void (*on_prompt_tokens)(int),
                                void (*on_prompt_end)(),
                                void (*on_token_end)(),
                                void (*on_turn_end)(),
                                void (*on_ready)()) {
    g_srv_moe_on_prompt_tokens = on_prompt_tokens;
    g_srv_moe_on_prompt_end    = on_prompt_end;
    g_srv_moe_on_token_end     = on_token_end;
    g_srv_moe_on_turn_end      = on_turn_end;
    g_srv_moe_on_ready         = on_ready;
}

// [moe-hooks] C5.6: slot released WITHOUT having generated (see server-moe-hooks.h).
void (*g_srv_moe_on_turn_abort)() = nullptr;
void llama_server_set_moe_turn_abort(void (*fn)()) { g_srv_moe_on_turn_abort = fn; }

// C0 (moert): run a callback right after common_init() reinstalls its logger, so the LLapis log
// filter can re-assert itself. Set by server_main (exe); the pointer crosses the DLL boundary,
// no symbol import needed. Same pattern as the moe hooks above.
static void (*g_srv_post_common_init)() = nullptr;
void llama_server_set_post_common_init(void (*f)()) { g_srv_post_common_init = f; }

void llama_server_set_moe_json_provider(int (*moe_json)(const char *, char *, int)) {
    g_srv_moe_json = moe_json;
}

// Variable-KV S6a rebuild scheduler (see server-moe-hooks.h)
int (*g_srv_moe_rebuild_k)(int, char *, int) = nullptr;
void llama_server_set_moe_rebuild_k(int (*fn)(int, char *, int)) {
    g_srv_moe_rebuild_k = fn;
}

// Variable-KV Phase 3 automatic rebalance trigger (see server-moe-hooks.h)
int (*g_srv_moe_plan_ctx)(int, int, int) = nullptr;
void llama_server_set_moe_plan_ctx(int (*fn)(int, int, int)) { g_srv_moe_plan_ctx = fn; }

// Variable-KV pre-ctx-create hook (see server-moe-hooks.h)
void (*g_srv_moe_pre_ctx_create)(int) = nullptr;
void llama_server_set_moe_pre_ctx_create(void (*fn)(int)) { g_srv_moe_pre_ctx_create = fn; }

// Variable-KV Phase 2 ctx-rebuild protocol (see server-moe-hooks.h)
int  (*g_srv_moe_rebuild_ctx)(int, char *, int) = nullptr;
int  (*g_srv_moe_ctx_rebuild_take)()            = nullptr;
void (*g_srv_moe_ctx_rebuild_done)(int, int)    = nullptr;
void llama_server_set_moe_rebuild_ctx(int (*schedule)(int, char *, int),
                                      int (*take)(),
                                      void (*done)(int, int)) {
    g_srv_moe_rebuild_ctx     = schedule;
    g_srv_moe_ctx_rebuild_take = take;
    g_srv_moe_ctx_rebuild_done = done;
}

int g_srv_moe_spec_draft_max = 0;

void llama_server_set_moe_spec_draft_max(int n) {
    g_srv_moe_spec_draft_max = n;
}

int (*g_srv_moe_spec_draft_fn)(int, int) = nullptr;

void llama_server_set_moe_spec_draft_fn(int (*fn)(int, int)) {
    g_srv_moe_spec_draft_fn = fn;
}

// [moe-hooks] C5: emitted-generation-token counter (see server-moe-hooks.h). Single writer
// (the slot/update thread under -np 1); plain long long is fine for a display-rate source.
long long g_srv_moe_gen_tokens = 0;

long long llama_server_moe_gen_tokens() {
    return g_srv_moe_gen_tokens;
}

void llama_server_set_moe_ctx_sink(void (*on_ctx)(struct llama_context *, const struct llama_model *)) {
    g_srv_moe_on_ctx = on_ctx;
}

void llama_server_set_moe_spec_ctx_sink(void (*on_spec_ctx)(struct llama_context *, const struct llama_model *)) {
    g_srv_moe_on_spec_ctx = on_spec_ctx;
}

// [moe-hooks] C5.2: router admission gate + child-exit observation (see server-moe-hooks.h)
bool (*g_srv_moe_spawn_gate)(const char *, char *, int) = nullptr;
void llama_server_set_moe_spawn_gate(bool (*fn)(const char *, char *, int)) {
    g_srv_moe_spawn_gate = fn;
}
void (*g_srv_moe_child_exit)(const char *, int) = nullptr;
void llama_server_set_moe_child_exit(void (*fn)(const char *, int)) {
    g_srv_moe_child_exit = fn;
}

int (*g_srv_moe_plan)(int, char *, int) = nullptr;

void llama_server_set_moe_plan_provider(int (*plan_json)(int, char *, int)) {
    g_srv_moe_plan = plan_json;
}

int (*g_srv_moe_profiles_save)(const char *, char *, int) = nullptr;

void llama_server_set_moe_profiles_save(int (*save)(const char *, char *, int)) {
    g_srv_moe_profiles_save = save;
}

// streamed (q4stream) prefill trampolines (C3) — see server-moe-hooks.h
srv_moe_cb_eval_t g_srv_moe_cb_eval    = nullptr;
void *            g_srv_moe_cb_eval_ud = nullptr;

void llama_server_set_moe_cb_eval(srv_moe_cb_eval_t cb, void * user_data) {
    g_srv_moe_cb_eval    = cb;
    g_srv_moe_cb_eval_ud = user_data;
}

int (*g_srv_moe_prompt_cap_fn)(int) = nullptr;

void llama_server_set_moe_prompt_cap(int (*fn)(int)) {
    g_srv_moe_prompt_cap_fn = fn;
}

int g_srv_moe_ckpt_tail = 0;

void llama_server_set_moe_ckpt_tail(int n_tail) {
    g_srv_moe_ckpt_tail = n_tail > 0 ? n_tail : 0;
}

bool (*g_srv_moe_pre_decode)(bool, int) = nullptr;

void llama_server_set_moe_pre_decode(bool (*fn)(bool, int)) {
    g_srv_moe_pre_decode = fn;
}

void (*g_srv_moe_batch_tokens)(const int32_t *, int) = nullptr;

void llama_server_set_moe_batch_tokens(void (*fn)(const int32_t *, int)) {
    g_srv_moe_batch_tokens = fn;
}

static std::function<void(int)> shutdown_handler;
static std::atomic_flag is_terminating = ATOMIC_FLAG_INIT;

static inline void signal_handler(int signal) {
    if (is_terminating.test_and_set()) {
        // in case it hangs, we can force terminate the server by hitting Ctrl+C twice
        // this is for better developer experience, we can remove when the server is stable enough
        fprintf(stderr, "Received second interrupt, terminating immediately.\n");
        exit(1);
    }

    shutdown_handler(signal);
}

void llama_server_request_shutdown() {
    // same discipline as the signal path; shutdown_handler is assigned before the HTTP
    // loop starts accepting, but guard anyway (fork addition, see server-moe-hooks.h)
    if (shutdown_handler) {
        signal_handler(SIGINT);
    } else {
        exit(0);
    }
}

// wrapper function that handles exceptions and logs errors
// this is to make sure handler_t never throws exceptions; instead, it returns an error response
static server_http_context::handler_t ex_wrapper(server_http_context::handler_t func) {
    return [func = std::move(func)](const server_http_req & req) -> server_http_res_ptr {
        std::string message;
        error_type error;
        try {
            return func(req);
        } catch (const std::invalid_argument & e) {
            // treat invalid_argument as invalid request (400)
            error = ERROR_TYPE_INVALID_REQUEST;
            message = e.what();
        } catch (const std::exception & e) {
            // treat other exceptions as server error (500)
            error = ERROR_TYPE_SERVER;
            message = e.what();
        } catch (...) {
            error = ERROR_TYPE_SERVER;
            message = "unknown error";
        }

        auto res = std::make_unique<server_http_res>();
        res->status = 500;
        try {
            json error_data = format_error_response(message, error);
            res->status = json_value(error_data, "code", 500);
            res->data = safe_json_to_str({{ "error", error_data }});
            SRV_WRN("got exception: %s\n", res->data.c_str());
        } catch (const std::exception & e) {
            SRV_ERR("got another exception: %s | while handling exception: %s\n", e.what(), message.c_str());
            res->data = "Internal Server Error";
        }
        return res;
    };
}

// [llapis] The shutdown pair — GET /moe/exit-token and POST /moe/exit. Registered from two
// places (the router, and a single-model moert host) and identical in both, so it lives
// here once. The threat model and the reason for each check are in server-http.h.
//
// On a non-loopback bind NOTHING is registered: both routes 404, and the UI's power button
// hides itself because the token fetch fails. That is the whole of "LAN mode".
static void llapis_register_exit_routes(server_http_context & ctx_http) {
    if (!ctx_http.loopback_bind) {
        SRV_INF("%s", "LLapis: non-loopback bind - /moe/exit and /moe/exit-token are not registered\n");
        return;
    }

    // Hands the per-process nonce to a same-origin document. Safe to expose: the
    // Same-Origin Policy is what stops a foreign page from reading the body, and a leaked
    // token on its own still cannot pass the Origin check on the POST below.
    ctx_http.get("/moe/exit-token", ex_wrapper([&ctx_http](const server_http_req & req) -> server_http_res_ptr {
        auto res = std::make_unique<server_http_res>();
        const std::string host = req.get_header("Host");
        if (host.empty() || !server_http_is_loopback_authority(host) ||
            server_http_authority_port(host) != ctx_http.port) {
            res->status = 403;
            res->data   = "{\"error\":\"refused: Host is not this loopback server\"}";
            return res;
        }
        res->headers["Cache-Control"] = "no-store";
        res->data = "{\"token\":\"" + ctx_http.exit_token + "\"}";
        return res;
    }));

    // The destructive one. POST only — a GET stays a 404 because the route is not
    // registered for that method, which also means no <img>/<iframe> can ever reach it.
    ctx_http.post("/moe/exit", ex_wrapper([&ctx_http](const server_http_req & req) -> server_http_res_ptr {
        auto res = std::make_unique<server_http_res>();
        const exit_auth a = server_http_check_exit_auth(
            ctx_http.loopback_bind, ctx_http.exit_token, ctx_http.port, ctx_http.is_ssl,
            req.get_header("Host"), req.get_header("Origin"),
            req.get_header(LLAPIS_EXIT_TOKEN_HEADER));
        if (a != exit_auth::ok) {
            res->status = 403;
            res->data   = std::string("{\"error\":\"refused: ") + exit_auth_reason(a) + "\"}";
            SRV_WRN("POST /moe/exit refused: %s\n", exit_auth_reason(a));
            return res;
        }
        res->data = "{\"ok\":true,\"msg\":\"moert runtime shutting down\"}";
        std::thread([]() { llama_server_request_shutdown(); }).detach();
        return res;
    }));
}

// satisfies -Wmissing-declarations
int llama_server(int argc, char ** argv);

int llama_server(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    // own arguments required by this example
    common_params params;

    common_init();

    // C0 (moert): re-assert the LLapis log filter after common_init reinstalls its own logger,
    // so normal-mode serve keeps a quiet terminal + full raw log (no-op unless server_main set it).
    if (g_srv_post_common_init) g_srv_post_common_init();

    // start the stream session manager GC right after common init, before any HTTP route can
    // touch it. lifecycle is symmetric, stop_gc() runs in clean_up() before backend free
    g_stream_sessions.start_gc();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SERVER)) {
        return 1;
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    common_models_handler models_handler;
    try {
        models_handler = common_models_handler_init(params, LLAMA_EXAMPLE_SERVER);
        if (common_models_handler_is_preset_repo(models_handler)) {
            // apply the preset and start the server in router mode
            common_models_handler_apply(models_handler, params);
        }
    } catch (const std::exception & e) {
        SRV_ERR("failed to fetch model metadata: %s\n", e.what());
        return 1;
    }

    // router server never loads a model and must not touch the GPU
    const bool is_router_server = params.model.path.empty()
                               && params.model.hf_repo.empty();

    // skip device enumeration so the CUDA primary context stays uncreated
    common_params_print_info(params, !is_router_server);

    if (!is_router_server) {
        // validate batch size for embeddings
        // embeddings require all tokens to be processed in a single ubatch
        // see https://github.com/ggml-org/llama.cpp/issues/12836
        if (params.embedding && params.n_batch > params.n_ubatch) {
            SRV_WRN("embeddings enabled with n_batch (%d) > n_ubatch (%d)\n", params.n_batch, params.n_ubatch);
            SRV_WRN("setting n_batch = n_ubatch = %d to avoid assertion failure\n", params.n_ubatch);
            params.n_batch = params.n_ubatch;
        }

        if (params.n_parallel < 0) {
            SRV_INF("%s", "n_parallel is set to auto, using n_parallel = 4 and kv_unified = true\n");

            params.n_parallel = 4;
            params.kv_unified = true;
        }
    }

    // for consistency between server router mode and single-model mode, we set the same model name as alias
    auto model_name = params.model.get_name();
    if (params.model_alias.empty() && !model_name.empty()) {
        params.model_alias.insert(model_name);
    }

    // struct that contains llama context and inference
    server_context ctx_server;

    server_http_context ctx_http;
    if (!ctx_http.init(params)) {
        SRV_ERR("%s", "failed to initialize HTTP server\n");
        return 1;
    }

    //
    // Router
    //

    // register API routes
    server_child child; // only used in non-router mode
    server_routes routes(params, ctx_server);
    server_tools tools;

    std::optional<server_models_routes> models_routes{};
    if (is_router_server) {
        // setup server instances manager
        try {
            models_routes.emplace(params, argc, argv);
        } catch (const std::exception & e) {
            SRV_ERR("failed to initialize router models: %s\n", e.what());
            return 1;
        }

        // proxy handlers
        // note: routes.get_health stays the same
        routes.get_metrics                 = models_routes->proxy_get;
        routes.post_props                  = models_routes->proxy_post;
        routes.post_completions            = models_routes->proxy_post;
        routes.post_completions_oai        = models_routes->proxy_post;
        routes.post_chat_completions       = models_routes->proxy_post;
        routes.post_control                = models_routes->proxy_post;
        routes.post_responses_oai          = models_routes->proxy_post;
        routes.post_transcriptions_oai     = models_routes->proxy_post;
        routes.post_anthropic_messages     = models_routes->proxy_post;
        routes.post_anthropic_count_tokens = models_routes->proxy_post;
        routes.post_infill                 = models_routes->proxy_post;
        routes.post_embeddings             = models_routes->proxy_post;
        routes.post_embeddings_oai         = models_routes->proxy_post;
        routes.post_rerank                 = models_routes->proxy_post;
        routes.post_tokenize               = models_routes->proxy_post;
        routes.post_detokenize             = models_routes->proxy_post;
        routes.post_apply_template         = models_routes->proxy_post;
        routes.post_chat_completions_tok   = models_routes->proxy_post;
        routes.post_responses_tok_oai      = models_routes->proxy_post;
        routes.get_lora_adapters           = models_routes->proxy_get;
        routes.post_lora_adapters          = models_routes->proxy_post;
        routes.get_slots                   = models_routes->proxy_get;
        routes.post_slots                  = models_routes->proxy_post;

        // custom routes for router
        routes.get_props                   = models_routes->get_router_props;
        routes.get_models                  = models_routes->get_router_models;

        ctx_http.post("/models",               ex_wrapper(models_routes->post_router_models));
        ctx_http.post("/models/load",          ex_wrapper(models_routes->post_router_models_load));
        ctx_http.post("/models/unload",        ex_wrapper(models_routes->post_router_models_unload));
        ctx_http.get ("/models/sse",           ex_wrapper(models_routes->get_router_models_sse));
        ctx_http.del ("/models",               ex_wrapper(models_routes->del_router_models));
    }

    ctx_http.get ("/health",                   ex_wrapper(routes.get_health)); // public endpoint (no API key check)
    ctx_http.get ("/v1/health",                ex_wrapper(routes.get_health)); // public endpoint (no API key check)
    ctx_http.get ("/metrics",                  ex_wrapper(routes.get_metrics));
    ctx_http.get ("/props",                    ex_wrapper(routes.get_props));
    ctx_http.post("/props",                    ex_wrapper(routes.post_props));
    ctx_http.get ("/models",                   ex_wrapper(routes.get_models)); // public endpoint (no API key check)
    ctx_http.get ("/v1/models",                ex_wrapper(routes.get_models)); // public endpoint (no API key check)
    ctx_http.post("/completion",               ex_wrapper(routes.post_completions)); // legacy
    ctx_http.post("/completions",              ex_wrapper(routes.post_completions));
    ctx_http.post("/v1/completions",           ex_wrapper(routes.post_completions_oai));
    ctx_http.post("/chat/completions",         ex_wrapper(routes.post_chat_completions));
    ctx_http.post("/v1/chat/completions",      ex_wrapper(routes.post_chat_completions));
    ctx_http.post("/v1/chat/completions/control", ex_wrapper(routes.post_control));
    ctx_http.post("/v1/responses",             ex_wrapper(routes.post_responses_oai));
    ctx_http.post("/responses",                ex_wrapper(routes.post_responses_oai));
    ctx_http.post("/v1/audio/transcriptions",  ex_wrapper(routes.post_transcriptions_oai));
    ctx_http.post("/audio/transcriptions",     ex_wrapper(routes.post_transcriptions_oai));
    ctx_http.post("/v1/messages",              ex_wrapper(routes.post_anthropic_messages)); // anthropic messages API
    ctx_http.post("/infill",                   ex_wrapper(routes.post_infill));
    ctx_http.post("/embedding",                ex_wrapper(routes.post_embeddings)); // legacy
    ctx_http.post("/embeddings",               ex_wrapper(routes.post_embeddings));
    ctx_http.post("/v1/embeddings",            ex_wrapper(routes.post_embeddings_oai));
    ctx_http.post("/rerank",                   ex_wrapper(routes.post_rerank));
    ctx_http.post("/reranking",                ex_wrapper(routes.post_rerank));
    ctx_http.post("/v1/rerank",                ex_wrapper(routes.post_rerank));
    ctx_http.post("/v1/reranking",             ex_wrapper(routes.post_rerank));
    ctx_http.post("/tokenize",                 ex_wrapper(routes.post_tokenize));
    ctx_http.post("/detokenize",               ex_wrapper(routes.post_detokenize));
    ctx_http.post("/apply-template",           ex_wrapper(routes.post_apply_template));
    // token counting
    ctx_http.post("/chat/completions/input_tokens",    ex_wrapper(routes.post_chat_completions_tok));
    ctx_http.post("/v1/chat/completions/input_tokens", ex_wrapper(routes.post_chat_completions_tok));
    ctx_http.post("/responses/input_tokens",           ex_wrapper(routes.post_responses_tok_oai));
    ctx_http.post("/v1/responses/input_tokens",        ex_wrapper(routes.post_responses_tok_oai));
    ctx_http.post("/v1/messages/count_tokens",         ex_wrapper(routes.post_anthropic_count_tokens)); // anthropic token counting
    // LoRA adapters hotswap
    ctx_http.get ("/lora-adapters",            ex_wrapper(routes.get_lora_adapters));
    ctx_http.post("/lora-adapters",            ex_wrapper(routes.post_lora_adapters));
    // Save & load slots
    ctx_http.get ("/slots",                    ex_wrapper(routes.get_slots));
    ctx_http.post("/slots/:id_slot",           ex_wrapper(routes.post_slots));

    // moe_dual observability (fork addition, see server-moe-hooks.h): routes exist only when
    // the host executable registered a JSON provider — a stock llama-server has no /moe/*.
    if (g_srv_moe_json) {
        auto moe_json_route = [](const char * what) -> server_http_context::handler_t {
            return [what](const server_http_req &) -> server_http_res_ptr {
                auto res = std::make_unique<server_http_res>();
                std::vector<char> buf(65536);
                const int n = g_srv_moe_json(what, buf.data(), (int) buf.size());
                if (n > 0) {
                    res->data.assign(buf.data(), (size_t) n);
                } else {
                    res->status = 503;
                    res->data   = "{\"error\":\"moe runtime telemetry unavailable\"}";
                }
                return res;
            };
        };
        ctx_http.get("/moe/telemetry", ex_wrapper(moe_json_route("telemetry")));
        ctx_http.get("/moe/info",      ex_wrapper(moe_json_route("info")));

        // Variable-KV S6a: POST /moe/rebuild_k?k=N (or body {"k":N}) — SCHEDULES a hot-slab
        // rebuild at a new uniform K; the host applies it at the next turn boundary on the
        // slot thread. Trampoline-only: validation + the rebuild itself live in the host.
        if (g_srv_moe_rebuild_k) {
            ctx_http.post("/moe/rebuild_k", ex_wrapper([](const server_http_req & req) -> server_http_res_ptr {
                auto res = std::make_unique<server_http_res>();
                int k = std::atoi(req.get_param("k").c_str());
                if (k <= 0 && !req.body.empty()) {
                    const char * p = std::strstr(req.body.c_str(), "\"k\"");
                    if (p) { p = std::strchr(p, ':'); if (p) k = std::atoi(p + 1); }
                }
                char err[256] = {0};
                if (g_srv_moe_rebuild_k(k, err, (int) sizeof(err)) > 0) {
                    res->data = "{\"ok\":true,\"scheduled_k\":" + std::to_string(k) + "}";
                } else {
                    res->status = 400;
                    std::string esc;
                    for (const char * c = err[0] ? err : "rebuild rejected"; *c; ++c) {
                        if (*c == '\\' || *c == '"') esc += '\\';
                        esc += *c;
                    }
                    res->data = "{\"error\":\"" + esc + "\"}";
                }
                return res;
            }));
        }

        // Variable-KV Phase 2: POST /moe/rebuild_ctx?n=N (or body {"n":N}) — SCHEDULES a context
        // rebuild at a new n_ctx with KV replay; applied at the all-idle quiescent point of
        // update_slots. Trampoline-only, same shape as /moe/rebuild_k above.
        if (g_srv_moe_rebuild_ctx) {
            ctx_http.post("/moe/rebuild_ctx", ex_wrapper([&ctx_server](const server_http_req & req) -> server_http_res_ptr {
                auto res = std::make_unique<server_http_res>();
                int n = std::atoi(req.get_param("n").c_str());
                if (n <= 0 && !req.body.empty()) {
                    const char * p = std::strstr(req.body.c_str(), "\"n\"");
                    if (p) { p = std::strchr(p, ':'); if (p) n = std::atoi(p + 1); }
                }
                char err[256] = {0};
                if (g_srv_moe_rebuild_ctx(n, err, (int) sizeof(err)) > 0) {
                    res->data = "{\"ok\":true,\"scheduled_n_ctx\":" + std::to_string(n) + "}";
                    // wake the task loop: the rebuild applies at the next all-idle pass of
                    // update_slots, which does NOT otherwise run while the server sits idle
                    ctx_server.kick_update_slots();
                } else {
                    res->status = 400;
                    std::string esc;
                    for (const char * c = err[0] ? err : "ctx rebuild rejected"; *c; ++c) {
                        if (*c == '\\' || *c == '"') esc += '\\';
                        esc += *c;
                    }
                    res->data = "{\"error\":\"" + esc + "\"}";
                }
                return res;
            }));
        }
    }

    // M4 lifecycle routes, ROUTER side (fork addition, see server-moe-hooks.h). Active only
    // when the host registered the plan provider — a stock llama-server router has none.
    if (is_router_server && g_srv_moe_plan && !g_srv_moe_json) {
        // /moe/telemetry + /moe/info: NO-AUTOLOAD proxy to the single running child.
        // Never ensure_model_ready here: a telemetry poll must not (re)load the model,
        // it would fight the child's heartbeat auto-unload.
        auto moe_router_proxy = [&models_routes](const server_http_req & req) -> server_http_res_ptr {
            std::string name = req.get_param("model");
            bool loading = false;
            if (name.empty()) {
                for (const auto & m : models_routes->models.get_all_meta()) {
                    if (m.is_ready()) { name = m.name; break; }
                    if (m.is_running()) loading = true;
                }
            } else {
                auto meta = models_routes->models.get_meta(name);
                if (!meta.has_value() || !meta->is_ready()) {
                    loading = meta.has_value() && meta->is_running();
                    name.clear();
                }
            }
            if (name.empty()) {
                auto res = std::make_unique<server_http_res>();
                res->data = loading ? "{\"state\":\"loading\"}" : "{\"state\":\"unloaded\"}";
                return res;
            }
            return models_routes->models.proxy_request(req, "GET", name, false);
        };
        ctx_http.get("/moe/telemetry", ex_wrapper(moe_router_proxy));
        ctx_http.get("/moe/info",      ex_wrapper(moe_router_proxy));

        // Variable-KV: NO-AUTOLOAD POST proxies to the running child (same child-resolution as
        // the telemetry proxy above; a rebuild request must never (re)load the model).
        auto moe_rebuild_proxy = [&models_routes](const server_http_req & req) -> server_http_res_ptr {
            std::string name;
            for (const auto & m : models_routes->models.get_all_meta()) {
                if (m.is_ready()) { name = m.name; break; }
            }
            if (name.empty()) {
                auto res = std::make_unique<server_http_res>();
                res->status = 503;
                res->data   = "{\"error\":\"no model loaded\"}";
                return res;
            }
            return models_routes->models.proxy_request(req, "POST", name, false);
        };
        ctx_http.post("/moe/rebuild_k",   ex_wrapper(moe_rebuild_proxy));
        ctx_http.post("/moe/rebuild_ctx", ex_wrapper(moe_rebuild_proxy));

        // /moe/plan: pre-load planner (host spawns its own `plan` subprocess for the probe).
        // ?ctx=N (M6b1) sizes the planner's KV budget; 0/absent = the host's default.
        ctx_http.get("/moe/plan", ex_wrapper([](const server_http_req & req) -> server_http_res_ptr {
            auto res = std::make_unique<server_http_res>();
            int ctx = 0;
            const std::string ctx_s = req.get_param("ctx");
            if (!ctx_s.empty()) {
                ctx = std::atoi(ctx_s.c_str());
                if (ctx < 0) ctx = 0;
            }
            std::vector<char> buf(131072);
            const int n = g_srv_moe_plan(ctx, buf.data(), (int) buf.size());
            if (n > 0) {
                res->data.assign(buf.data(), (size_t) n);
            } else {
                res->status = 500;
                res->data   = "{\"error\":\"planner failed - see the server log for the reason\"}";
            }
            return res;
        }));

        // /moe/exit: graceful full shutdown. clean_up() unloads all children on the way out.
        llapis_register_exit_routes(ctx_http);

        // /moe/profiles: plan->profile save. The host rewrites moert-profiles.json +
        // the generated preset INI; on success the model list is reloaded from disk so
        // the new profile shows up in /models (and the UI selector) immediately.
        if (g_srv_moe_profiles_save) {
            ctx_http.post("/moe/profiles", ex_wrapper([&models_routes](const server_http_req & req) -> server_http_res_ptr {
                auto res = std::make_unique<server_http_res>();
                char err[1024] = {0};
                const int rc = g_srv_moe_profiles_save(req.body.c_str(), err, (int) sizeof(err));
                if (rc <= 0) {
                    res->status = 400;
                    std::string e(err[0] ? err : "profile save failed");
                    std::string esc; for (char c : e) { if (c == '\\' || c == '"') esc += '\\'; esc += c; }
                    res->data = "{\"error\":\"" + esc + "\"}";
                    return res;
                }
                models_routes->models.load_models();   // pick up the regenerated preset INI
                res->data = "{\"ok\":true}";
                return res;
            }));
        }
    }

    // /moe/exit in SINGLE-MODEL mode too (moert host, not router-spawned): the UI's
    // "Exit Runtime" must work when the server was launched directly with -m (legacy
    // `serve -Direct` / dev runs). NOT registered on a router CHILD: a child must be
    // stopped by its router's stop sequence (a self-shutdown deadlocks on the stdin
    // monitor join — see the M4 lifecycle gate), and the router handles /moe/exit itself.
    if (!is_router_server && g_srv_moe_json && !child.is_child()) {
        llapis_register_exit_routes(ctx_http);
    }

    // resumable streaming, the conversation_id is the session identity end to end. router and
    // child wire different handlers under the same paths: a child binds the local g_stream_sessions
    // backed factories, the router binds proxies that resolve the owning child through the
    // conv_id -> model map
    server_http_context::handler_t stream_get_h;
    server_http_context::handler_t streams_lookup_h;
    server_http_context::handler_t stream_delete_h;
    if (is_router_server) {
        stream_get_h     = models_routes->router_stream_get;
        streams_lookup_h = models_routes->router_streams_lookup;
        stream_delete_h  = models_routes->router_stream_delete;
    } else {
        stream_get_h     = make_stream_get_handler();
        streams_lookup_h = make_streams_lookup_handler();
        stream_delete_h  = make_stream_delete_handler();
    }
    ctx_http.get ("/v1/stream/:conv_id",       ex_wrapper(stream_get_h));
    // POST /v1/streams/lookup with body {"conversation_ids": [...]}. you can only ask for ids
    // you already own (the WebUI passes the convs visible in its sidebar). the server never
    // lists ids it has not been asked about, so a random caller cannot enumerate live sessions
    ctx_http.post("/v1/streams/lookup",        ex_wrapper(streams_lookup_h));
    ctx_http.del ("/v1/stream/:conv_id",       ex_wrapper(stream_delete_h));

    // Google Cloud Platform (Vertex AI) compat
    ctx_http.register_gcp_compat();

    // return 403 for disabled features
    server_http_context::handler_t res_403 = [](const server_http_req &) {
        auto res = std::make_unique<server_http_res>();
        res->status = 403;
        res->data = safe_json_to_str({
            {"error", {
                {"message", "this feature is disabled"},
                {"type", "feature_disabled"},
            }}
        });
        return res;
    };

    // CORS proxy (EXPERIMENTAL, only used by the Web UI for MCP)
    if (params.ui_mcp_proxy) {
        SRV_WRN("%s", "-----------------\n");
        SRV_WRN("%s", "CORS proxy is enabled, do not expose server to untrusted environments\n");
        SRV_WRN("%s", "This feature is EXPERIMENTAL and may be removed or changed in future versions\n");
        SRV_WRN("%s", "-----------------\n");
        ctx_http.get ("/cors-proxy",      ex_wrapper(proxy_handler_get));
        ctx_http.post("/cors-proxy",      ex_wrapper(proxy_handler_post));
    } else {
        ctx_http.get ("/cors-proxy",      ex_wrapper(res_403));
        ctx_http.post("/cors-proxy",      ex_wrapper(res_403));
    }

    // EXPERIMENTAL built-in tools
    if (!params.server_tools.empty()) {
        try {
            tools.setup(params.server_tools);
        } catch (const std::exception & e) {
            SRV_ERR("tools setup failed: %s\n", e.what());
            return 1;
        }
        SRV_WRN("%s", "-----------------\n");
        SRV_WRN("%s", "Built-in tools are enabled, do not expose server to untrusted environments\n");
        SRV_WRN("%s", "This feature is EXPERIMENTAL and may be changed in the future\n");
        SRV_WRN("%s", "-----------------\n");
        ctx_http.get ("/tools",           ex_wrapper(tools.handle_get));
        ctx_http.post("/tools",           ex_wrapper(tools.handle_post));
    } else {
        ctx_http.get ("/tools",           ex_wrapper(res_403));
        ctx_http.post("/tools",           ex_wrapper(res_403));
    }

    //
    // Handle downloading model
    //

    if (child.is_child() && child.get_mode() == SERVER_CHILD_MODE_DOWNLOAD) {
        return child.run_download(params);
    } else if (!is_router_server) {
        // single-model mode (NOT spawned by router)
        try {
            common_models_handler_apply(models_handler, params);
        } catch (const std::exception & e) {
            SRV_ERR("failed to download model: %s\n", e.what());
            return 1;
        }
    }

    //
    // Start the server
    //

    std::function<void()> clean_up;

    if (is_router_server) {
        SRV_INF("%s", "starting router server, no model will be loaded in this process\n");

        clean_up = [&models_routes]() {
            SRV_INF("%s: cleaning up before exit...\n", __func__);
            // stop the session GC first, it finalizes live sessions and wakes pending readers
            g_stream_sessions.stop_gc();
            if (models_routes.has_value()) {
                models_routes->stopping.store(true); // maybe redundant, but just to be safe
                models_routes->models.unload_all();
            }
            llama_backend_free();
        };

        if (!ctx_http.start()) {
            clean_up();
            SRV_ERR("%s", "exiting due to HTTP server error\n");
            return 1;
        }
        ctx_http.is_ready.store(true);

        shutdown_handler = [&](int) {
            if (models_routes.has_value()) {
                // important to disconnect any SSE clients
                models_routes->stopping.store(true);
            }
            ctx_http.stop();
        };

    } else {
        // setup clean up function, to be called before exit
        clean_up = [&ctx_http, &ctx_server]() {
            SRV_INF("%s: cleaning up before exit...\n", __func__);
            // stop the session GC first, it finalizes live sessions and wakes pending readers
            g_stream_sessions.stop_gc();
            ctx_http.stop();
            ctx_server.terminate();
            llama_backend_free();
        };

        // start the HTTP server before loading the model to be able to serve /health requests
        if (!ctx_http.start()) {
            clean_up();
            SRV_ERR("%s", "exiting due to HTTP server error\n");
            return 1;
        }

        // setup communication child --> router if necessary
        if (child.is_child()) {
            ctx_server.set_state_callback([&](server_state state, json payload) {
                child.notify_to_router(server_state_to_str(state), payload);
            });
        }

        // load the model
        SRV_INF("%s", "loading model\n");

        if (!ctx_server.load_model(params)) {
            clean_up();
            if (ctx_http.thread.joinable()) {
                ctx_http.thread.join();
            }
            SRV_ERR("%s", "exiting due to model loading error\n");
            return 1;
        }

        routes.update_meta(ctx_server);
        ctx_http.is_ready.store(true);

        SRV_INF("%s", "model loaded\n");

        shutdown_handler = [&](int) {
            // this will unblock start_loop()
            ctx_server.terminate();
        };
    }

    // TODO: refactor in common/console
#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
    struct sigaction sigint_action;
    sigint_action.sa_handler = signal_handler;
    sigemptyset (&sigint_action.sa_mask);
    sigint_action.sa_flags = 0;
    sigaction(SIGINT, &sigint_action, NULL);
    sigaction(SIGTERM, &sigint_action, NULL);
#elif defined (_WIN32)
    auto console_ctrl_handler = +[](DWORD ctrl_type) -> BOOL {
        return (ctrl_type == CTRL_C_EVENT) ? (signal_handler(SIGINT), true) : false;
    };
    SetConsoleCtrlHandler(reinterpret_cast<PHANDLER_ROUTINE>(console_ctrl_handler), true);
#endif

    if (is_router_server) {
        SRV_INF("router server is listening on %s\n", ctx_http.listening_address.c_str());
        SRV_WRN("%s", "NOTE: router mode is experimental\n");
        SRV_WRN("%s", "      it is not recommended to use this mode in untrusted environments\n");

        if (!params.models_preset_hf.empty()) {
            SRV_WRN(      "NOTE: using preset.ini from HF repo '%s'\n", params.models_preset_hf.c_str());
            SRV_WRN("%s", "      please only use presets that you can trust! Unknown presets may be unsafe\n");
        }

        if (ctx_http.thread.joinable()) {
            ctx_http.thread.join(); // keep the main thread alive
        }

        // when the HTTP server stops, clean up and exit
        clean_up();
    } else {
        SRV_INF("server is listening on %s\n", ctx_http.listening_address.c_str());

        // [moe-hooks] model + startup probe done, about to serve: let the host runtime
        // neutralize any probe-polluted observation state before the first task.
        if (g_srv_moe_on_ready) {
            g_srv_moe_on_ready();
        }

        // optionally, notify router server that this instance is ready
        std::thread monitor_thread;
        if (child.is_child()) {
            monitor_thread = child.setup(shutdown_handler);
            child.notify_to_router(server_state_to_str(SERVER_STATE_READY), routes.get_model_info());
        }

        // this call blocks the main thread until queue_tasks.terminate() is called
        ctx_server.start_loop();

        clean_up();
        if (ctx_http.thread.joinable()) {
            ctx_http.thread.join();
        }
        if (monitor_thread.joinable()) {
            monitor_thread.join();
        }

        auto * ll_ctx = ctx_server.get_llama_context();
        if (ll_ctx != nullptr) {
            common_memory_breakdown_print(ll_ctx);
        }
    }

    return 0;
}
