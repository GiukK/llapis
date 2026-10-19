cmake_minimum_required(VERSION 3.16)
project(llapis LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# ===========================================================================
# Upstream llama.cpp: what we build, and what we refuse to fetch.
# ===========================================================================
# third_party/llama IS in this repository, with the LLapis changes already applied as
# ordinary source. There is no clone, no patch to apply, no checksum manifest and no
# pinned SHA to re-fetch: `git clone` then `cmake` is the whole build, and it needs no
# network. Where the code came from, and what we changed in it, is recorded in
# third_party/llama/PROVENANCE.md.
set(LLAMA_BUILD_COMMON   ON  CACHE BOOL "" FORCE)   # moert-server links `common`
set(LLAMA_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
# moert-server embeds the server implementation (llama-server-impl); targeting it
# pulls in only server-context / llama-server-impl / llama-ui / cpp-httplib / mtmd.
set(LLAMA_BUILD_TOOLS    ON  CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER   ON  CACHE BOOL "" FORCE)
# ...but ONLY the three tool directories moert-server actually consumes (ui, server,
# mtmd). Upstream's tools/CMakeLists.txt otherwise configures ~20 standalone CLI
# executables -- llama-cli, llama-bench, llama-quantize and friends -- that LLapis
# neither ships nor uses. See LLAPIS_TOOLS_MINIMAL in third_party/llama/tools/CMakeLists.txt.
set(LLAPIS_TOOLS_MINIMAL ON  CACHE BOOL "" FORCE)

# --- the build must not touch the network ----------------------------------
# Each of these was traced to its actual consumer in the pinned upstream, not
# assumed from its name.
#
# The web UI ships pre-built in third_party/llama/tools/ui/dist. ui-assets.cmake gives
# that directory priority 1 and then `return()`s, so the npm and Hugging Face paths are
# unreachable rather than merely disabled. Both flags are still set to OFF so that a
# build without the bundle fails visibly instead of silently downloading a different UI.
set(LLAMA_BUILD_UI        OFF CACHE BOOL "" FORCE)  # no npm build (no node required)
set(LLAMA_USE_PREBUILT_UI OFF CACHE BOOL "" FORCE)  # no dist.tar.gz download from HF
#
# TLS: vendor/cpp-httplib tries BoringSSL, then LibreSSL, then OpenSSL, IN THAT ORDER.
# The first two are FetchContent downloads (googlesource / cdn.openbsd.org) that run at
# CONFIGURE time. Setting only LLAMA_OPENSSL=OFF does nothing when either is on, because
# they are earlier branches of the same if/elseif chain — all three must be OFF.
# LLapis serves plain HTTP on loopback, so no TLS is needed: every use of it upstream is
# behind #ifdef CPPHTTPLIB_OPENSSL_SUPPORT and the HTTPS path fails with an explicit
# error rather than silently misbehaving.
set(LLAMA_BUILD_BORINGSSL OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_LIBRESSL  OFF CACHE BOOL "" FORCE)
set(LLAMA_OPENSSL         OFF CACHE BOOL "" FORCE)
#
# NOTE: there is deliberately no LLAMA_CURL here. At the pinned commit it is
# `llama_option_depr(WARNING LLAMA_CURL)` with no replacement and no remaining CURL
# consumer anywhere in the CMake tree — setting it would be an inert line that reads
# like a protection.

# CUDA architecture: without -DCMAKE_CUDA_ARCHITECTURES, build for the GPU in this machine ("native",
# CMake >= 3.24), whatever its generation. It must be decided HERE, before moert enables CUDA: ggml picks
# "native" only when the variable is still undefined, and moert-cuda (configured first) left it empty, so
# its kernels fell back to nvcc's default (sm_75 + PTX, JIT-compiled at the first launch on any other GPU)
# while ggml's were native. GGML_NATIVE=OFF (a portable build) keeps ggml's own list unless an arch is given.
if (GGML_CUDA AND NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND (NOT DEFINED GGML_NATIVE OR GGML_NATIVE)
    AND CMAKE_VERSION VERSION_GREATER_EQUAL "3.24")
  set(CMAKE_CUDA_ARCHITECTURES native)
  message(STATUS "llapis: no CMAKE_CUDA_ARCHITECTURES given -- building for this machine's GPU (native)")
endif()

# moert: the vertical runtime library. Added BEFORE llama.cpp because the fork's
# ggml-cuda links moert-cuda (the GGML_OP_MOE_DUAL trampoline calls moert_ffn_moe),
# so the target must exist when that directory configures. moert-server itself does
# NOT link moert-cuda: the island's single copy of state lives inside ggml-cuda and
# the host reaches it through ggml-cuda's exported forwarders.
add_subdirectory(moert)

add_subdirectory(third_party/llama)

# ===========================================================================
# moert-server — the product. The fork's llama-server implementation (HTTP routes,
# OpenAI-compatible API, slots) plus the dual-precision runtime, wired in
# server_main.cpp: env-driven init before model load, single slot enforced,
# per-phase trampolines set on server-context.
# ===========================================================================
add_executable(moert-server
  src/server_main.cpp
  src/model_plan.cpp
  src/mem_probe.cpp
  src/mem_partition.cpp
  src/llapis_log_filter.cpp
  src/moe_dual.cpp
  src/moe_dual_init.cpp
  src/moe_dual_q4stream.cpp
  src/moe_sidecar.cpp
  src/moe_xfer.cpp
  src/mtp_forecast.cpp
  src/gguf_tools.cpp
  src/io_util.cpp)
target_compile_features(moert-server PRIVATE cxx_std_17)
target_include_directories(moert-server PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama/common
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama/include
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama/ggml/include
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama/tools/server
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama/tools/mtmd  # mtmd.h: the lazy projector's VRAM callback
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama/vendor      # nlohmann/json.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/moert/include)
target_link_libraries(moert-server PRIVATE
  llama-server-impl llama-common llama ggml mtmd moert-policy moert-memmgr)
if(WIN32)
  # Psapi: process RSS. dxgi: per-process VRAM. ws2_32: the watchdog's httplib client.
  target_link_libraries(moert-server PRIVATE Psapi dxgi ws2_32)
endif()
if(TARGET ggml-cuda)
  # Zero-copy cold reads map a host pointer via ggml_backend_cuda_host_register_mapped,
  # exported from ggml-cuda. Enable the path only in a CUDA configuration.
  target_link_libraries(moert-server PRIVATE ggml-cuda)
  target_compile_definitions(moert-server PRIVATE MOE_DUAL_HAVE_CUDA_MAP)
endif()

# ===========================================================================
# CPU unit tests. No GPU, no model. Run them, do not just build them.
# ===========================================================================
# moert-memmgr + expert store v0. Takes a scratch directory as its one argument.
add_executable(test-moert-memmgr tests/test_moert_memmgr.cpp)
target_link_libraries(test-moert-memmgr PRIVATE moert-memmgr)

# moert_policy_resize_k — the variable-KV resizing foundation. No args.
add_executable(test-moert-policy-resize tests/test_moert_policy_resize.cpp)
target_link_libraries(test-moert-policy-resize PRIVATE moert-policy)

# Policy V2 COST_AWARE: the price binds, prompt routing never forecasts, warmup keeps V1. No args.
add_executable(test-moert-policy-cost tests/test_moert_policy_cost.cpp)
target_link_libraries(test-moert-policy-cost PRIVATE moert-policy)

# Asynchronous promotion: FILLING slots, complete/abort, stale completions, sync path unchanged. No args.
add_executable(test-moert-policy-async tests/test_moert_policy_async.cpp)
target_link_libraries(test-moert-policy-async PRIVATE moert-policy)

# moert/registry.h: the cheapest valid source of an expert's bytes, both regimes. Header-only. No args.
add_executable(test-moert-registry tests/test_moert_registry.cpp)
target_include_directories(test-moert-registry PRIVATE moert/include)

# The transfer engine (src/moe_xfer): pending until flush, file-source promotions byte-identical to the
# synchronous path, clean failures without a source. Takes a scratch directory as its one argument.
add_executable(test-moe-xfer tests/test_moe_xfer.cpp src/moe_xfer.cpp)
target_include_directories(test-moe-xfer PRIVATE src moert/include)
target_link_libraries(test-moe-xfer PRIVATE moert-memmgr)


# The sidecar's decode-once kernels against ggml's vec_dot, bit for bit (`bench` arg: single-thread timing).
add_executable(test-sidecar-kernels tests/test_sidecar_kernels.cpp)
target_compile_features(test-sidecar-kernels PRIVATE cxx_std_17)
target_include_directories(test-sidecar-kernels PRIVATE src
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama/ggml/include)
target_link_libraries(test-sidecar-kernels PRIVATE ggml ggml-cpu)
# The CPU cold executor end to end on decode-shaped fires (a measuring tool, not a test).
add_executable(bench-sidecar tests/bench_sidecar.cpp src/moe_sidecar.cpp)
target_compile_features(bench-sidecar PRIVATE cxx_std_17)
target_include_directories(bench-sidecar PRIVATE src moert/include
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama/ggml/include)
target_link_libraries(bench-sidecar PRIVATE ggml ggml-cpu)
# DMA modes (findings 216): only when the CUDA toolkit is there
find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
  target_link_libraries(bench-sidecar PRIVATE CUDA::cudart)
  target_compile_definitions(bench-sidecar PRIVATE BENCH_DMA)
endif()
# The sidecar kernels are AVX2 intrinsics: compile their translation units for AVX2 (VEX encoding
# throughout, F16C) like ggml-cpu itself, instead of the SSE2 baseline around the intrinsics.
if(MSVC)
  set_source_files_properties(src/moe_sidecar.cpp tests/test_sidecar_kernels.cpp tests/bench_sidecar.cpp src/mtp_forecast.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
else()
  set_source_files_properties(src/moe_sidecar.cpp tests/test_sidecar_kernels.cpp tests/bench_sidecar.cpp src/mtp_forecast.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c")
endif()

# Exclusive per-expert page ownership: state machine, shadow accounting, owned IQ2
# pool, recreate-before-free. No args.

# The centralized variable-KV resizing policy (src/vkv_policy.hpp). No args.
add_executable(test-vkv-policy tests/test_vkv_policy.cpp)
target_compile_features(test-vkv-policy PRIVATE cxx_std_17)
target_include_directories(test-vkv-policy PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

# The streamed-vs-fused prefill gate (src/prefill_policy.hpp). Three A/Bs on 2026-08-09 went
# into reading this decision off a GPU; it is three inputs and a path (findings 125, 134). No args.
add_executable(test-prefill-policy tests/test_prefill_policy.cpp)
target_compile_features(test-prefill-policy PRIVATE cxx_std_17)
target_include_directories(test-prefill-policy PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

# How many experts fit in VRAM right now (src/kfit_policy.hpp) — the arithmetic that sets K,
# and K is the product (K=22 vs K=4 was 33.5 vs 17.4 tok/s of decode). A shipped version of it
# was ~5x too pessimistic and took a campaign to notice. Four rules, all paid for. No args.
add_executable(test-kfit-policy tests/test_kfit_policy.cpp)
target_compile_features(test-kfit-policy PRIVATE cxx_std_17)
target_include_directories(test-kfit-policy PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

# When a decode batch may run with the q4stream phase ARMED (src/q4s_phase_policy.hpp). The
# C3.3 corruption class: getting this wrong does not crash, it produces confident wrong text,
# which is also why the golden transcript is the weakest gate here. The invariant is checked
# EXHAUSTIVELY (findings 136). No args.
add_executable(test-q4s-phase-policy tests/test_q4s_phase_policy.cpp)
target_compile_features(test-q4s-phase-policy PRIVATE cxx_std_17)
target_include_directories(test-q4s-phase-policy PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

# The POST /moe/exit authorization decision (tools/server/server-http.h). Header-only, so
# this links nothing: it includes the SAME header the server decides with. No args.
add_executable(test-exit-guard tests/test_exit_guard.cpp)
target_compile_features(test-exit-guard PRIVATE cxx_std_17)
target_include_directories(test-exit-guard PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama/tools/server)

# The context-checkpoint restore decision (tools/server/server-ckpt-policy.h). Same shape as
# the exit guard: header-only, links nothing, includes the SAME header the server decides
# with. Five integers that decide whether a returning turn reuses its prefix or re-prefills
# it -- diagnosed wrong in writing three times because reading it off a GPU run was the only
# way to ask (findings 124, 128, 129, 133). No args.
add_executable(test-ckpt-policy tests/test_ckpt_policy.cpp)
target_compile_features(test-ckpt-policy PRIVATE cxx_std_17)
target_include_directories(test-ckpt-policy PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama/tools/server)

# The launcher's VRAM terms (`moert-server plan --terms`, src/model_plan.cpp) on header-only GGUFs of quant
# pairs this box cannot load (IQ2/Q4, Q4/Q8): a different pair must be sized from its own files, never from
# the reference pair's constants. Takes a scratch directory as its one argument.
add_executable(test-model-plan tests/test_model_plan.cpp src/model_plan.cpp)
target_compile_features(test-model-plan PRIVATE cxx_std_17)
target_include_directories(test-model-plan PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama/ggml/include)
target_link_libraries(test-model-plan PRIVATE ggml)

enable_testing()
add_test(NAME moert-policy-resize COMMAND test-moert-policy-resize)
add_test(NAME moert-policy-cost   COMMAND test-moert-policy-cost)
add_test(NAME moert-policy-async  COMMAND test-moert-policy-async)
add_test(NAME vkv-policy          COMMAND test-vkv-policy)
add_test(NAME exit-guard          COMMAND test-exit-guard)
add_test(NAME ckpt-policy         COMMAND test-ckpt-policy)
add_test(NAME prefill-policy      COMMAND test-prefill-policy)
add_test(NAME kfit-policy         COMMAND test-kfit-policy)
add_test(NAME q4s-phase-policy    COMMAND test-q4s-phase-policy)
add_test(NAME moert-memmgr        COMMAND test-moert-memmgr "${CMAKE_CURRENT_BINARY_DIR}/test-memmgr-scratch")
add_test(NAME moert-registry      COMMAND test-moert-registry)
add_test(NAME moe-xfer            COMMAND test-moe-xfer "${CMAKE_CURRENT_BINARY_DIR}/test-xfer-scratch")
add_test(NAME model-plan          COMMAND test-model-plan "${CMAKE_CURRENT_BINARY_DIR}/test-model-plan-scratch")
add_test(NAME sidecar-kernels     COMMAND test-sidecar-kernels)
# These two link the ggml DLLs, which the build writes to bin/<config>, not next to the tests: on a fresh
# build tree they failed with 0xc0000135 (DLL not found). They only passed where the launcher had already
# staged the DLLs next to moert-server.exe.
if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.22")
  set_tests_properties(model-plan sidecar-kernels PROPERTIES
    ENVIRONMENT_MODIFICATION "PATH=path_list_prepend:$<TARGET_FILE_DIR:ggml>")
endif()
