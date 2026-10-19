# dependencies

find_package(Threads REQUIRED)

# third-party

# ...

# flags

llama_add_compile_flags()

# tools

# [llapis] LLAPIS_TOOLS_MINIMAL: configure ONLY what moert-server actually needs.
#
# moert-server embeds the server implementation (tools/server -> llama-server-impl),
# which needs the embedded web UI (tools/ui -> llama-ui) and multimodal support
# (tools/mtmd -> mtmd). Everything else below is a standalone llama.cpp CLI tool that
# LLapis neither ships nor uses; configuring them leaves ~20 extra executables in the
# generated build system, which cannot be reconciled with a positive target allowlist.
if (LLAPIS_TOOLS_MINIMAL)
    add_subdirectory(ui)
    add_subdirectory(server)
    add_subdirectory(mtmd)
elseif (EMSCRIPTEN)
else()
    add_subdirectory(batched-bench)
    add_subdirectory(gguf-split)
    add_subdirectory(imatrix)
    add_subdirectory(llama-bench)
    add_subdirectory(completion)
    add_subdirectory(perplexity)
    add_subdirectory(quantize)
    if (LLAMA_BUILD_SERVER)
        add_subdirectory(ui)
        add_subdirectory(cli)
        add_subdirectory(server)
    endif()
    add_subdirectory(tokenize)
    add_subdirectory(parser)
    add_subdirectory(tts)
    add_subdirectory(mtmd)
    if (GGML_RPC)
        add_subdirectory(rpc)
    endif()
    if (NOT GGML_BACKEND_DL AND GGML_CPU)
        # these tools use backends directly (no dynamic loading) and depend on CPU backend symbols
        add_subdirectory(cvector-generator)
        add_subdirectory(export-lora)
    endif()
    add_subdirectory(fit-params)
    add_subdirectory(results)
endif()
