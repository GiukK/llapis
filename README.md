# LLapis

Run **Gemma 4 26B-A4B** on a machine that should not be able to run it.

LLapis is a local inference runtime for one model family. It patches
[llama.cpp](https://github.com/ggml-org/llama.cpp) with a dual-precision
mixture-of-experts runtime: the experts your prompts actually route to are kept
in VRAM at Q4, the rest stay in system RAM at IQ2, and the split is re-fitted to
the VRAM you really have — at every prompt, not once at startup. The result is a
26B model that answers on an 8 GB laptop GPU, over an OpenAI-compatible API and a
browser UI, with nothing leaving the machine.

## Quick start

Windows 10/11, an NVIDIA RTX 20-series card or newer with a current driver, 16 GB of
RAM. **Nothing else to install**: no compiler, no CUDA Toolkit, no Git.

1. Download `llapis-<version>-win-x64.zip` from the
   [Releases](https://github.com/GiukK/llapis/releases) page and extract it to a short
   path, such as `C:\llapis`.
2. In PowerShell, from that folder:

```powershell
cd C:\llapis
.\build-check.cmd                   # checks the GPU, the driver, RAM and disk
# download the model files into models\ (commands in "Models" below), then:
.\bin\moert-server.exe prepare
.\llapis.cmd setup
.\llapis.cmd serve -Open
```

Windows may warn the first time that it "protected your PC" from an unrecognised app:
the binaries are not code-signed. **More info → Run anyway** lets them run.

Building from source instead is described under [Build from source](#build-from-source);
the commands are the same except that the server lives in `build-cuda\Release\`
instead of `bin\`.

In the rest of this file, `llapis <verb>` is short for `.\llapis.cmd <verb>`, run
from the LLapis folder.

### Which GPUs

The release is compiled for every consumer generation CUDA 13 supports, and
`build-check` (and the server, at launch) tells you whether your card is covered:

| GPU | Code in the release | Tested |
|---|---|---|
| RTX 40 series (Ada, sm_89) | machine code | **yes** — the reference machine, RTX 4070 Laptop 8 GB |
| RTX 30 series (Ampere, sm_86) | machine code | not yet |
| RTX 20 / GTX 16 series (Turing, sm_75) | machine code | not yet |
| RTX 50 series (Blackwell, sm_120) | machine code + PTX | not yet |
| a later NVIDIA GPU | PTX, compiled by the driver at the first start (minutes, once) | no |
| GTX 10 series and older | none: CUDA 13 does not support them | — |

"Not yet" means compiled for and expected to work, but not run by us. If you run it
on one of those, the most useful report you can send is an issue saying whether it
worked, with the output of `.\build-check.cmd` attached.

---

## What this is, and what it is not

**It is:**

- a local HTTP server on `127.0.0.1:8077` — OpenAI-compatible API at `/v1`, plus a browser UI;
- a runtime specialised for Gemma 4 26B-A4B (a 26B mixture-of-experts model with 4B active parameters);
- offline at runtime: once the model files are on disk, LLapis opens no outgoing connections of its own;
- self-contained: llama.cpp is *in* this repository, at `third_party/llama`, with the LLapis changes already applied as ordinary source. `git clone` then `cmake` is the whole build, and it touches the network exactly zero times. Where that code came from is recorded in [`third_party/llama/PROVENANCE.md`](third_party/llama/PROVENANCE.md).

**It is not:**

- a general model runner. Other architectures are not supported, not tested, and not a goal.
- a distributor of model weights. You download the GGUF files yourself (see [Models](#models)).
- multi-user or multi-tenant. One slot, one conversation, no authentication.
- a Linux or macOS project. Windows + NVIDIA only, for now — the memory work it does is written against WDDM and CUDA.
- production infrastructure. There is no password, no TLS, no rate limiting.

---

## Requirements

Verified on the reference machine. Anything materially different is untested, not
unsupported — but you are the first to try it. The release zip needs only the first
five rows; the rest are for [building from source](#build-from-source).

| | Reference machine | Notes |
|---|---|---|
| OS | Windows 11, Windows PowerShell 5.1 | PowerShell 7 also works |
| GPU | NVIDIA RTX 4070 Laptop, 8 GB | **compute capability 7.5 or newer** — see below |
| Driver | 592.00 | one that supports CUDA 13: the R580 family or newer |
| RAM | 16 GB (15.2 GiB usable) | this is the binding constraint, see below |
| Disk | ~21 GB for `speed`, ~48 GB for `quality` | see the breakdown below |
| CUDA | Toolkit 13.3 (`nvcc` 13.3.33) | source build only; `CUDA_PATH` must be set. The release ships the runtime DLLs it needs |
| Compiler | Visual Studio 2022 (v17), C++ toolset | source build only; Build Tools alone are enough |
| CMake | 3.24+ | source build only; Visual Studio 2022's bundled CMake qualifies |
| Git | Git for Windows | source build only, to clone this repository |
| Node / npm | **not required** | the web UI is prebuilt, in the release and in `third_party/llama/tools/ui/dist` |

**On RAM.** The IQ2 cold tier is a ~9.3 GiB file and the runtime keeps the routed
working set resident in RAM. On a 16 GB machine that fits with the arena partly
paged — which is by design and measured, not an accident — but 16 GB is the floor,
not a comfortable margin. Close what you can before a long session. The server
refuses to load (with a message saying why) on a machine with less installed RAM
than the expert pack plus 4 GiB, or when RAM plus page file cannot hold what it
would commit; free RAM at launch is not the test, because Windows makes room by
paging other applications out. `LLAPIS_PREFLIGHT_RAM=0` skips the check.

**On which GPUs work.** CUDA 13 dropped Maxwell, Pascal and Volta, so the pinned
toolkit cannot generate code for anything below compute capability **7.5**
(Turing, the RTX 20 series). A GTX 10-series card will not build against it at
any architecture setting — this is the toolkit's limit, not LLapis's. Check yours
first, before downloading anything:

```bash
nvidia-smi --query-gpu=name,compute_cap --format=csv
```

**The architecture of your card is detected.** Without `-DCMAKE_CUDA_ARCHITECTURES`
the build compiles for the GPU in the machine it runs on (CMake's `native`), whatever
its generation, the CUDA kernels of the runtime and of llama.cpp alike. Pass the
architecture only to build here for a *different* machine's GPU — the number is its
compute capability without the dot:

```bash
cmake -S . -B build-cuda -G "Visual Studio 17 2022" -A x64 -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86-real
```

75 = Turing (RTX 20), 86 = Ampere (RTX 30), 89 = Ada (RTX 40), 120 = Blackwell
(RTX 50). Getting the number wrong does not fail the build; it produces a binary
that refuses to run on your device. A build meant to run on other machines also wants
`-DGGML_NATIVE=OFF -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON`: the CPU code is
otherwise compiled for the build machine's processor, and one with AVX-512 makes a
binary that crashes on a CPU without it.

**When you pass an architecture, the `-real` suffix is deliberate.** Without it, nvcc embeds PTX alongside the
machine code — an intermediate form kept so that a *future* GPU, one whose
machine code is not in the binary, can JIT-compile the kernels at load time. You
are compiling for the card in front of you, so that never happens here, and it is
not free: measured, the PTX is **34.7% of `ggml-cuda.dll`** — 29.2 MB down to
19.1 MB — for **no change in build time** (4.2 → 4.1 min, within noise) and **no
change in output** (the golden transcript reproduces character for character).

Drop the suffix if you intend to build here and run the binaries on a machine
with a different GPU without rebuilding. That is the only thing it buys.

**On VRAM.** 8 GB is what the expert-cache sizing was designed and measured
against. More VRAM means more resident experts and a higher hit rate; the runtime
fits itself to whatever it measures at runtime, so there is nothing to configure.
The smallest launch — non-expert weights, drafter, KV at the minimum context and a
minimal expert cache — is computed from the models you installed (`llapis setup`
shows it: ~5.1 GB for `quality` and ~4.1 GB for `speed` with the files below). Any
VRAM limit is accepted: under that floor LLapis starts at the floor anyway, over your
limit, and says so; it refuses only a card that cannot hold the floor at all. Below
8 GB the cache shrinks and the cold path dominates — but VRAM is not what stops a
small machine. RAM is: see above.

**On disk.** The model files you download are 10.5 GB for `speed` (IQ2 + drafter) or 27.5 GB
for `quality` (those plus the Q4 source); the vision projector adds 1.2 GB. On top of that,
`prepare` derives 8.8 GB of artifacts for `speed` or 19.6 GB for `quality` (see the table in
[Models](#models)). Add the build tree (~1 GB) and the repository itself (~0.1 GB) and you get
the figures above. All sizes here are measured, in GB of 10⁹ bytes.

**Install path.** Keep it short. Source paths inside `third_party/llama` reach 161
characters, so a very deep install directory is asking for trouble on a Windows
without long paths enabled. `C:\llapis` or `%USERPROFILE%\llapis` is fine. Paths
containing spaces are fine — every command in this file was run from one.

---

## Build from source

Only if you want to build LLapis yourself: the release zip (see [Quick start](#quick-start))
is the same program, already built. Building needs the CUDA Toolkit, Visual Studio 2022
and CMake (see [Requirements](#requirements)); `.\build-check.cmd -Build` checks for them
and does the steps below in one go.

### 1. Clone

```bash
git clone https://github.com/GiukK/llapis
```

### 2. Configure

```bash
cmake -S . -B build-cuda -G "Visual Studio 17 2022" -A x64 -DGGML_CUDA=ON
```

The GPU's architecture is detected (see [Requirements](#requirements)); pass
`-DCMAKE_CUDA_ARCHITECTURES` only to build for a different card than this machine's.

Not sure the machine has everything? Run `.\build-check.cmd` first: it checks the GPU
and its driver, the CUDA toolkit, Visual Studio's C++ tools and CUDA integration,
CMake, the CPU, RAM, disk and long paths, and says what is missing and where to get
it. `.\build-check.cmd -Build` then configures, builds and runs the tests, and checks
that the CUDA code was compiled for this GPU; once the models are in place and
`llapis setup` has run, `.\build-check.cmd -Smoke` starts the server and asks it one
question.

If CMake cannot find the CUDA toolkit, add
`-DCUDAToolkit_ROOT="$env:CUDA_PATH"` — with **forward slashes**: backslashes in a
`-D` value are escape sequences to CMake, and `...\v13.3` in particular is not a
valid one.

There is no fetch step and no patch step. llama.cpp is already here, in
`third_party/llama`, with the LLapis changes applied as ordinary source — see
[`third_party/llama/PROVENANCE.md`](third_party/llama/PROVENANCE.md) for the
upstream commit it came from and the list of what we changed.

### 3. Build — offline, no npm, no node

```bash
cmake --build build-cuda --config Release
```

If that says `cmake` is not recognised, you are not missing anything: Visual
Studio installs its own CMake with "Desktop development with C++" and does not
put it on the PATH. Look under
`C:\Program Files\Microsoft Visual Studio\2022\<edition>\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin`,
or run from "x64 Native Tools Command Prompt for VS 2022".

No `--target`: the build produces `moert-server.exe` and the unit tests (14, run
with `ctest -C Release` from `build-cuda`) together, and restricting it to one target
has been observed not to rebuild the backend DLLs after their sources changed. The
DLLs the server needs (llama.cpp's and the CUDA runtime's) are copied next to
`build-cuda\Release\moert-server.exe` at the end of the build, so it runs on its own.

Nothing in this step reaches the network. The web UI is not compiled: it is a
prebuilt bundle in `third_party/llama/tools/ui/dist`, which upstream's
`ui-assets.cmake` takes with priority over its npm and Hugging Face download
paths, so those paths are unreachable rather than merely switched off. Because
the UI source is never compiled here, the diff that produced the bundle is kept
separately in `provenance/`, which no script applies. See
[`provenance/README.md`](provenance/README.md).

### 4. Models

See [Models](#models) below — download two files (three for `quality`), put each in its folder.

### 5. Choose the runtime

```bash
.\llapis.cmd setup
```

Asks two things: `quality` or `speed`, and a VRAM budget. Writes one file,
`llapis-user.json`. That is the whole configuration.

`.\llapis.cmd`, not `llapis.ps1`, for every command in this file. The `.cmd` is
not decoration: a fresh Windows account runs PowerShell under the `Restricted`
policy, where invoking a `.ps1` directly fails with "running scripts is disabled
on this system" — and a `.ps1` extracted from a downloaded zip is blocked even
under `RemoteSigned`. The wrapper runs the same script with
`-ExecutionPolicy Bypass` for that one call, so installing LLapis never asks you
to change a system-wide setting.

---

## Models

**LLapis does not distribute model weights and does not download them for you.**
The files below are published by third parties under their own terms. Verify
those terms at the source before you rely on them.

Everything comes from one repository:

> **`unsloth/gemma-4-26B-A4B-it-GGUF`**
> <https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF>
> Revision `c099eb48e663fd284577b04978a94ffccb261841`, licence Apache-2.0, not gated
> (checked 2026-08-06). Quantized from `google/gemma-4-26b-a4b-it`, also Apache-2.0.

Download by revision, not from the branch tip — a branch moves and these SHA-256
values are for this revision.

| Put it in | File | Size | SHA-256 |
|---|---|---|---|
| `models/low/` **(required)** | `gemma-4-26B-A4B-it-UD-IQ2_M.gguf` | 9.33 GiB | `3fe245f4436ab44c103291222e81c514e53ca382871b555df24fea11e36b3c3a` |
| `models/mtp/` **(required)** | `MTP/mtp-gemma-4-26B-A4B-it-Q8_0.gguf` | 440 MiB | `6326fb9f5e487aa8dcdd313a091e3c67724cb2a666ec3b7d2895b5b26d93ed1b` |
| `models/high/` (quality mode only) | `gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf` | 15.8 GiB | `ef728c8e0c337fd1067b947af006e38a9ef2419e56feced4fd29b4bf0636e30c` |
| `models/mmproj/` (only for `-Vision`) | `mmproj-F16.gguf` | 1.11 GiB | `418a6d8723067cd712235facbbc5cba6c8fbbd413fc1292d2aace5a027d5a42f` |

To download them, from the LLapis folder (`curl.exe` ships with Windows 10 and 11;
`-C -` resumes an interrupted download when you run the same line again):

```powershell
$hf = 'https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF/resolve/c099eb48e663fd284577b04978a94ffccb261841'
curl.exe -L -C - -o models\low\gemma-4-26B-A4B-it-UD-IQ2_M.gguf      "$hf/gemma-4-26B-A4B-it-UD-IQ2_M.gguf"
curl.exe -L -C - -o models\mtp\mtp-gemma-4-26B-A4B-it-Q8_0.gguf      "$hf/MTP/mtp-gemma-4-26B-A4B-it-Q8_0.gguf"
# quality mode only:
curl.exe -L -C - -o models\high\gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf   "$hf/gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf"
# -Vision only:
curl.exe -L -C - -o models\mmproj\mmproj-F16.gguf                    "$hf/mmproj-F16.gguf"
```

The drafter sits in the repository's `MTP/` subfolder, but it goes directly in `models\mtp\`,
not in `models\mtp\MTP\`: each folder is read one level deep.

**Other quants.** These files are the reference pair, and the only one measured. Any
other pair of the same model works the same way — one `.gguf` in `models/low/`, a
higher-precision one in `models/high/` (Q4 + Q8 on a 24 GB card, say): the launcher
reads the sizes it plans with from the files' own headers (`moert-server plan --terms`),
so nothing is tuned to these two.

Check a download before trusting it:

```bash
Get-FileHash .\models\low\gemma-4-26B-A4B-it-UD-IQ2_M.gguf -Algorithm SHA256
```

A folder resolves whatever single `.gguf` it contains, so the filename does not
have to match — only the folder does. (The drafter head `prepare` writes next to the
drafter in `models/mtp/` is not counted.) Two or more `.gguf` in one folder without a
recognised name is reported rather than guessed at.

`models/cache/` holds artifacts LLapis derives from the files above. Deleting any
of them is safe — each one rebuilds itself:

| artifact | size | needed by | rebuilt by |
|---|---|---|---|
| `base_neq4.gguf` — experts from the IQ2 file, everything else from the Q4 one | 10.1 GiB | quality (**required**) | `prepare` |
| `low.experts.{index,pack}` — the IQ2 experts repacked into the runtime's own cold store, one dense run per layer and matrix so the runtime can map it read-only | 7.7 GiB | both (optional, see below) | `prepare` (a pack built before 2026-10-08 still works but is copied into RAM, and the startup says so: run `prepare` once and it rebuilds it) |
| `models/mtp/<drafter>.draft-head-r192.gguf` — the drafter with a two-stage vocabulary head (an approximate head from the basis shipped in `assets/mtp/`, then the exact logits of 2048 candidates): ~2.6 ms less per decode step, the text is unchanged (the target verifies every draft) | 492 MiB | both (optional: without it the plain drafter runs) | `prepare` (a few seconds; skipped with a message if your drafter is not the one in the table above) |

**You build them yourself, once**, from the directory holding `llapis.ps1`:

```bash
.\bin\moert-server.exe prepare                  # the release zip
.\build-cuda\Release\moert-server.exe prepare    # a source build
```

The launcher does not run this for you. It is a pure offline file transform — no
GPU, no model load — and it takes a few minutes, mostly disk.

Quality mode cannot start without `base_neq4.gguf`: that splice *is* the quality
runtime's base model. Speed mode does not use it at all.

The expert store is optional in both modes. With it, the runtime owns its cold
tier and the loader cut saves ~584 MiB of VRAM; without it, you get a yellow
warning at startup and the older borrowed path, which works but costs that VRAM
back. A speed-only install (nothing in `models/high`) gets the store alone:
`prepare` says it skipped `base_neq4.gguf` and builds that too once the Q4 file
is in `models/high`. Rebuilt files are written under temporary names and renamed
over the old ones, so a hard-linked model store is never rewritten in place;
stop the server first (`llapis stop`), it holds the pack open.

Nothing is remembered between sessions. Only a fraction of the experts fit in
VRAM, and which fraction is right depends on what you ask: every session starts
with an empty expert cache and fills it from the first prompt; by the end of that
prompt the cache is full, so the first answer runs at about the speed of the rest.

---

## Using it

### The commands

`llapis` is the only command. Seven verbs, and none of them takes a tuning knob.

| | |
|---|---|
| `llapis serve` | start the server + web UI on `http://127.0.0.1:8077` |
| `llapis chat` | terminal chat against that same server (starts it if needed) |
| `llapis setup` | choose the runtime (`quality` \| `speed`) and the VRAM budget |
| `llapis agents` | print the provider JSON for an agent harness |
| `llapis status` | what is the running server actually running? |
| `llapis reset` | clear the cached conversation on the server |
| `llapis stop` | shut the running server down |

`llapis help` prints the long version of this; `llapis` alone prints the short one.

Two things people expect to find here and will not:

- **`prepare` is not an `llapis` verb.** It is a `moert-server` subcommand that
  derives `models/cache/`, and you run it yourself — once, after downloading the
  weights. Nothing invokes it for you.
- **`quality` and `speed` are not verbs either.** They are the two runtimes, and
  `llapis setup` is where you pick one.

### The flags

Fourteen parameters, and that is deliberately all of them. Everything else the
runtime needs, it measures: the expert cache size is fitted to VRAM and re-fitted
at every context change, the context grows on demand, the prefill mode follows
from the drafter, the cache policy is the measured one.

| Flag | Default | What it does |
|---|---|---|
| `serve` \| `chat` \| `setup` \| `agents` \| `status` \| `reset` \| `stop` \| `help` | — | the verb (positional) |
| `-Host <addr>` | `127.0.0.1` | bind address. Anything else is LAN mode — read the warning below |
| `-Port <n>` | `8077` | HTTP port |
| `-Ctx <n>` | grows on demand | cap the context ceiling (model maximum 262144) |
| `-VramBudget <MiB>` | `0` | cap total VRAM for this session; 0 = use what is free |
| `-Vision` | off | load the vision projector (needs `models/mmproj/`) |
| `-Open` | off | open the browser once the server is ready |
| `-DryRun` | off | resolve everything, print the plan, start nothing |
| `-Trace` | off | raw runtime logs |
| `-Tools <list>` | **off** | let the *web UI* run tools on this machine — see below |
| `-McpProxy` | **off** | let the *web UI* reach MCP servers through this server — see below |
| `-NoVariableKv` | off | pin the context at `-Ctx` instead of growing it |
| `-NoLoaderCut` | off | load expert tensors the stock way instead of moert owning them |
| `-NoMtp` | off | decode without the MTP drafter (a measurement lever; see `llapis help`) |

Plus `-Verbose` and `-Debug` from PowerShell itself. `-Debug` is the useful one:
it shows the runtime's phases (pinning, prefill, cache refreshes) as they happen.

`-NoVariableKv` and `-NoLoaderCut` are escape hatches, not tuning. Both defaults were measured on one
GPU, so both need a way back to the conservative path that does not involve
editing the source:

- **`-NoVariableKv`** stops the context from growing. Normally LLapis boots small
  and rebuilds the KV cache larger on demand, re-fitting the expert cache to the
  measured free VRAM each time — which is what lets a short chat keep a big cache
  instead of paying up front for a context it never reaches. That rebuild is a
  live reallocation. If a session misbehaves around a context change, pin it.
- **`-NoLoaderCut`** goes back to letting llama.cpp own the expert tensors. The
  default has moert own the cold tier out of its own store, which is worth about
  584 MiB of VRAM but needs `models/cache/low.experts.index` — without that file
  the cut disables itself anyway, and says so.

### quality vs speed

Same runtime, same code path. The only difference is which file each precision
tier reads from.

**`quality`** — the base is a spliced non-expert-Q4 model, and hot experts are
promoted from the Q4_K_XL source. Hot experts are Q4 in VRAM over an IQ2 cold tier
in RAM: the experts you use most are at the higher precision. Needs
`models/high/`, so it costs an extra ~16 GB of disk.

**`speed`** — the base is the IQ2 file and promotions come from that same file, so
hot and cold hold identical bytes: the same weights, at the same precision, as stock IQ2.
Answers are not bit-identical from one run to the next — an expert is computed on the GPU
when it is cached and on the CPU when it is cold, which sums in a different order, and
which experts are cached depends on timing. The dual
machinery still runs, which is the point: it is what keeps the streamed prefill
instead of the token-by-token one, and that prefill is the whole speed edge.

Pick `speed` if you have not downloaded the Q4 file. Pick `quality` if you have.

### The web UI and the API

`llapis serve` gives you both on the same port:

- **UI**: <http://127.0.0.1:8077>
- **API**: `http://127.0.0.1:8077/v1` — OpenAI-compatible (`/v1/chat/completions`, `/v1/models`, streaming)

```bash
curl.exe http://127.0.0.1:8077/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\":\"llapis\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}"
```

### Agent harnesses and other clients

The model id is **`llapis`**. It is deliberately flat and constant: it does not
encode the runtime, the cache size or the context, so a harness configured once
never has to be reconfigured when any of those change.

Do not invent a model name. Either read it from `/v1/models`:

```bash
curl.exe http://127.0.0.1:8077/v1/models
```

`curl.exe`, with the extension, in both examples above. Windows PowerShell aliases
the bare word `curl` to `Invoke-WebRequest`, which does not take `-H` or `-d` and
fails with a parameter-binding error that names neither the cause nor the fix.

or take the generated provider block:

```bash
.\llapis.cmd agents
```

which emits, verbatim:

```json
{
  "providers": {
    "llapis": {
      "baseUrl": "http://127.0.0.1:8077/v1",
      "api": "openai-completions",
      "apiKey": "llapis-local",
      "compat": {
        "supportsDeveloperRole": false,
        "supportsReasoningEffort": false,
        "supportsUsageInStreaming": false,
        "maxTokensField": "max_tokens"
      },
      "models": [
        {
          "id": "llapis",
          "name": "LLapis - Gemma 4 26B-A4B (local)",
          "reasoning": true,
          "input": ["text"],
          "contextWindow": 65536,
          "maxTokens": 65536,
          "cost": {
            "input": 0,
            "output": 0,
            "cacheRead": 0,
            "cacheWrite": 0
          }
        }
      ]
    }
  }
}
```

The API key is a placeholder — the server does not check it, because it only
listens on loopback. Clients that insist on a key will accept this one.

A harness on the API brings its **own** tools and runs them in its **own**
process. It does not need `-Tools` and it does not need `-McpProxy`; those two
flags only ever grant capability to the browser UI.

---

## What LLapis can reach, and how to change it

The defaults are the conservative ones. All three of the following are **off**
until you turn them on, and each one is off because turning it on gives something
access it did not have.

### Network exposure

LLapis binds `127.0.0.1`. That is a local server: anything running on this machine
can reach it, the network cannot.

`-Host <addr>` binds somewhere else — and then anything that can reach this
machine on that port can use the model and read the conversation. **There is no
password and no TLS.** In that mode the shutdown route is not registered at all,
so the UI's power button disappears; stop the server from this machine with
`llapis stop`, which identifies the process holding the port rather than asking
the server to exit.

Only use it on a network you control, and prefer an SSH tunnel if you have one.

### Tools

`-Tools` lets the **web UI** run tools on this machine: `read_file`,
`write_file`, `exec_shell_command`, `web_fetch`, `web_search`.

```bash
.\llapis.cmd serve -Tools all                    # all five
.\llapis.cmd serve -Tools read_file,web_fetch    # just these two
```

`write_file` and `exec_shell_command` mean the model can change files and run
commands with your user's permissions. `web_fetch` and `web_search` mean the
server makes outbound requests.

The UI asks before each new tool and remembers an "always". **That prompt lives in
the browser.** The endpoint itself is not gated by it, so any program on this
machine can call it directly and never see a prompt. Granting a tool grants it to
everything local, not just to the page you are looking at.

### MCP proxy

`-McpProxy` lets the UI reach MCP servers by fetching them **through** the LLapis
server, which makes those outbound requests on the page's behalf. Off by default;
without it the MCP panel in the UI can only reach servers the browser itself can
reach directly.

### Outbound connections

With none of the above, a running LLapis starts no outgoing connections. It does
open a local listening socket — that is what a server is — but nothing leaves the
machine. The web UI is served from the bundle embedded at build time, with no CDN
and no external fonts: the KaTeX fonts, for instance, are embedded in the bundle's
CSS as data URIs precisely so the page never fetches anything.

The build reaches the network zero times. There is nothing to fetch: llama.cpp is
in the repository.

---

## Where things live

```
models/high/      Q4_K_XL source        (you download; quality mode only)
models/low/       IQ2 base              (you download; required)
models/mtp/       MTP drafter           (you download; required)
models/mmproj/    vision projector      (you download; only for -Vision)
models/cache/     derived artifacts     (`prepare` builds them; `setup` caches the models' sizes here)
logs/             one log per run       <timestamp>_<mode>.log
llapis-user.json  your setup choice     (regenerate with `llapis setup`)
bin/              server, DLLs, CUDA runtime   (the release zip)
build-cuda/       build tree                   (a source build)
third_party/llama llama.cpp, absorbed, LLapis changes applied (source only; see its PROVENANCE.md)
provenance/       how the web UI bundle was built — nothing applies or reads it (source only)
```

Everything is inside the LLapis folder. LLapis writes nothing to `%APPDATA%`, the
registry, or anywhere else on the system.

---

## Troubleshooting

**CMake cannot find CUDA** — install the CUDA Toolkit, make sure `CUDA_PATH` is
set, and pass `-DCUDAToolkit_ROOT="$env:CUDA_PATH"` with forward slashes. Do not
configure without CUDA: it produces a CPU-only build that looks fine and runs at a
tenth of the speed.

**`git clone failed` / `Filename too long`** — your install path is too deep.
Move the checkout somewhere shorter, or enable long paths
(`git config --global core.longpaths true`).

**`FATAL: the low model (models\low, the fast one) is required and was not found`** —
`models/low/` has no `.gguf` in it, or more than one and none with the expected name.
The lines under it say which.

**`moert-server.exe` exits at once, printing nothing** (exit code `-1073741515`,
`0xC0000135`: a DLL was not found) — the DLLs next to it are missing. From the
release zip: extract the whole zip again, into an empty folder. From a source build: a
build copies them there; rebuild (`cmake --build build-cuda --config Release`), or start
it once through `llapis serve`, which copies them too.

**`FATAL: ... this package has no GPU code it can run`** — your GPU is older than the
RTX 20 series (or not one the release was built for); `build-check` names it. CUDA 13
cannot run on GTX 10 series and older at all.

**`FATAL: the NVIDIA driver supports CUDA 12.x; this package needs a driver for CUDA 13`**
— update the driver from <https://www.nvidia.com/Download/index.aspx> (the R580 family
or newer) and start again.

**`quality needs models/high`** — you chose quality without the Q4 file. Download
it, or run `llapis setup` again and choose speed.

**`FATAL: the MTP drafter is required and was not found`** — `models/mtp/` is
empty. This one refuses to start rather than degrade: without the drafter the
prompt is prefilled token by token instead of streamed, measured at 74.5 against
1275 tok/s on a 4752-token prompt. That is roughly a minute of extra TTFT, and
applying it silently would look like the product is simply slow.

**Port 8077 already in use** — `llapis stop`, or `llapis serve -Port 8078`.

**Closing the window** that runs `llapis serve` stops the server with it: the
server processes are tied to that window, so nothing is left holding RAM or VRAM.
`llapis stop` does the same from another terminal.

**`Port N is held by PID X, which is not an LLapis server from this checkout`** —
exactly what it says. `llapis stop` verifies the process identity before it kills
anything, and refuses when it does not recognise the owner.

**It is slow, or slower than it was** — check `llapis status` for the live hit
rate and expert-cache size. The number that matters is the hit rate: only a
fraction of the experts fit in VRAM, and everything else is computed on the CPU,
so a run at 96% and a run at 67% are not the same product. If you are comparing two builds, compare them alternately in one session —
on a machine this close to its memory limit, the same binaries can span a wide
range of tok/s over an hour, and a measurement taken now against one taken
earlier says nothing.

**Where are the logs** — `logs/<timestamp>_<mode>.log`, one per run. `-Debug`
adds the runtime's phases; `-Trace` adds everything.

---

## Uninstall

There is no installer, so there is nothing to uninstall. Delete the folder.

To free space without losing the setup:

```bash
.\llapis.cmd stop
Remove-Item -Recurse -Force .\models\cache\*      # derived artifacts, ~8 GB (speed) or ~19 GB (quality); `prepare` rebuilds them
Remove-Item -Recurse -Force .\build-cuda          # build tree (a source build only)
Remove-Item -Recurse -Force .\logs\*
```

`build-cuda/` comes back with a configure and a rebuild. Do not delete
`third_party/llama` — that is source, not a fetched artifact. The model files in
`models/high|low|mtp|mmproj` are yours; LLapis never deletes them.

---

## Known limits of this first release

- **Windows + NVIDIA only.** The memory management is written against WDDM and
  CUDA. There is no CPU-only, ROCm, Metal or Linux path.
- **One model family.** Gemma 4 26B-A4B. Other architectures fall back to stock
  llama.cpp behaviour at best, and are not tested.
- **One slot.** One conversation at a time; concurrent requests queue.
- **No authentication, no TLS.** Safe on loopback, and only on loopback. LAN mode
  exposes the model and the conversation to anyone who can reach the port.
- **The UI's tool prompt is a UI feature.** Once `-Tools` is on, the endpoint is
  open to every local program, prompt or no prompt.
- **HEIC images are refused** in the UI. Decoding them upstream requires fetching
  a decoder from a CDN at runtime, which would break the offline guarantee, so it
  was removed rather than kept as a quiet exception.
- **A first turn is slow.** The model streams from disk; steady-state speed
  arrives once the expert cache is warm.
- **16 GB of RAM is the floor, and LLapis keeps its 7.7 GiB expert arena in RAM
  while it runs.** The arena is the expert store mapped read-only, so it lives in
  Windows' file cache, not in LLapis' own committed memory. It asks Windows for a minimum working set of arena + 0.5 GiB,
  because an arena that Windows trims away costs millions of page faults and
  roughly halves the first answer's speed. The price is paid by everything else:
  while the server is up, other applications are the ones pushed to the page
  file. Set `LLAPIS_ARENA_RESIDENT=0` before launching to give that RAM back.
  The exclusive arena (on by default; `MOE_DUAL_ARENA_EXCLUSIVE=0` turns it off) gives part of it back without that cost: the
  pages of the experts currently held in VRAM are dropped from RAM (~1.1 GiB in
  `speed` at a 5 GiB budget, ~0.25 GiB in `quality`) and faulted back in on a
  worker thread when those experts go cold again.

---

## Licence

LLapis is MIT — see [LICENSE](LICENSE). That licence covers the original LLapis
code: `moert/`, `src/`, `cli/`, `tests/`, the build files, and the LLapis changes
inside `third_party/llama` (listed in its
[`PROVENANCE.md`](third_party/llama/PROVENANCE.md)). It does not relicense anything
else:

- llama.cpp, in `third_party/llama`, stays under its own MIT licence
  (`third_party/llama/LICENSE`, Copyright (c) 2023-2026 The ggml authors);
- the libraries vendored in `third_party/llama/vendor` keep their own licences;
- the prebuilt web UI in `third_party/llama/tools/ui/dist` contains third-party
  packages under MIT, ISC, BSD-3-Clause, Apache-2.0 and MPL-2.0.
[THIRD_PARTY_LICENSES.txt](THIRD_PARTY_LICENSES.txt) lists every component with
its version, licence and source.

The release zip also carries, in `bin\`, the NVIDIA CUDA runtime (`cudart64_13`,
`cublas64_13`, `cublasLt64_13`), redistributed under the
[NVIDIA CUDA Toolkit EULA](https://docs.nvidia.com/cuda/eula/), and the Microsoft
Visual C++ runtime (`msvcp140`, `vcruntime140`, `vcruntime140_1`, `vcomp140`),
redistributed under the Visual Studio redistribution terms. Those files are not
covered by the MIT licence; `PACKAGE.txt` in the zip says the same.

Model weights are not part of this repository and are not distributed by it.
