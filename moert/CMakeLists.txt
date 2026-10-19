# moert — the vertical Gemma-MoE runtime library, extracted phase by phase
# M0: policy/. M1: memmgr/. M2: cuda/.
#
# HARD RULE: moert-policy and moert-memmgr must not depend on ggml, CUDA, or llama
# targets/headers — keep those targets dependency-free so the policy/memmgr stay
# CPU-unit-testable and linkable from any host. moert-cuda may use the CUDA runtime
# (it IS the CUDA island) but still no ggml headers: ggml services (quantized
# mul_mat_id, VRAM scratch pool) are reached through the moert_ffn_moe_host vtable.
add_library(moert-policy STATIC
  policy/policy.cpp)

target_include_directories(moert-policy PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(moert-policy PUBLIC cxx_std_17)

# M1: expert memory manager (hot-slab layout/lifetime via device callbacks, page
# catalog + promotion reads, expert-store v0 read/write, cold-tier borrow views).
add_library(moert-memmgr STATIC
  memmgr/memmgr.cpp
  memmgr/store.cpp)

target_include_directories(moert-memmgr PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(moert-memmgr PUBLIC cxx_std_17)

# M2: the MoE CUDA island (partition/gather/scatter orchestration + the RAM-cold path +
# device routing counters) behind the device-pointer ABI in
# include/moert/cuda.h. Linked INTO ggml-cuda (the GGML_OP_MOE_DUAL trampoline is the
# only caller of moert_ffn_moe; host-process code reaches the island's services through
# ggml-cuda's exported forwarders so there is exactly ONE copy of its state). Built only
# in CUDA configurations; this subdirectory is added BEFORE external/llama.cpp so the
# target exists when ggml-cuda's CMakeLists links it.
if (GGML_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if (NOT CMAKE_CUDA_COMPILER)
    message(FATAL_ERROR "moert-cuda: GGML_CUDA is ON but no CUDA compiler was found")
  endif()
  enable_language(CUDA)
  add_library(moert-cuda STATIC
    cuda/ffn_moe.cu)
  target_include_directories(moert-cuda PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_compile_features(moert-cuda PUBLIC cxx_std_17)
endif()
