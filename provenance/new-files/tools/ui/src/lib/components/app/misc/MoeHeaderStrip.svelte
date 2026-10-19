<script lang="ts">
	// LLapis header strip -- product identity, live rate, memory picture.
	//
	// C5.6 (2026-07-28) scope change: the strip is now READ-ONLY except for the power button.
	// The model load/unload buttons and the "configure" popover (VRAM planner + saved-profile
	// list + save/delete) are GONE. They were built when `serve` started unloaded and the browser
	// was how you brought a model up; the product has since become API-first and loaded-by-default
	// with exactly one runtime called `llapis`, whose K and context are fitted to measured VRAM at
	// runtime. That left the whole surface offering the user choices the runtime no longer takes
	// from them -- a "saved configurations" list with one entry, and a K planner whose answer the
	// runtime overrides on the first prompt. Removed rather than maintained.
	//
	// What the strip shows: WHAT is loaded (model, live K, live context), HOW FAST it is going
	// right now, and WHERE the memory went. The rate is shown only while streaming -- and only
	// when it is real; see genRate below. Internal runtime flags (cache mode, refresh budget,
	// loader cut, arena, seeding) are deliberately not shown: the shipped config is the validated
	// standard, not a dashboard.
	//
	// Observer-only: polls GET ./moe/telemetry at 1 Hz (slot-thread caches -- polling never blocks
	// decode). On a stock llama-server /moe/* does not exist and the strip renders nothing.
	// The power button POSTs ./moe/exit (full runtime shutdown), carrying the per-process
	// nonce from ./moe/exit-token. Both routes exist only on a loopback bind, so on a LAN
	// bind the token fetch fails and the button is not rendered at all.
	import { onMount } from 'svelte';
	import { isChatStreaming } from '$lib/stores/chat.svelte';

	interface MoeInfo {
		preset: string;
		ctx: number;
		k: number;
		k_requested: number;
		n_expert: number;
		model_high: string;
	}

	interface MoeWindow {
		tokens: number;
		tok_per_s: number;
		// Emitted-token rate (includes accepted MTP draft tokens), measured server-side over a
		// fixed ~1.5 s window. tok_per_s counts decode CYCLES (the policy clock) and is 3-4x below
		// the user-visible rate under speculation -- never show it to a person.
		gen_tok_per_s?: number;
		hit_rate: number;
	}

	interface MoeVram {
		total: number;
		free: number;
		ours: number;
		shared_ours: number;
		shared_baseline: number;
		nonexpert_weights: number;
		hot_slab: number;
		l2: number;
		staging: number;
		kv: number;
		compute_tgt: number;
		compute_dft: number;
		drafter_weights: number;
		spec_active: boolean;
		pool_reserved: number;
		pool_used: number;
		unaccounted: number;
		other_procs: number;
	}

	interface MoeRam {
		total: number;
		avail: number;
		rss: number;
		cold_arena: number;
		arena_committed: number;
		arena_pinned: number;
		arena_resident: number;
		nonarena_rss: number;
		other_procs: number;
	}

	interface MoeTelemetry {
		occupancy: number;
		hit_rate: number;
		partition: { vram: MoeVram; ram: MoeRam };
	}

	// present = /moe/* exists (moert server, router or child)
	let present = $state(false);
	// routerState: '' = a model is loaded (info valid), else 'unloaded' | 'loading' | 'exited'
	let routerState = $state('');
	let info = $state<MoeInfo | null>(null);
	let win = $state<MoeWindow | null>(null);
	let tele = $state<MoeTelemetry | null>(null);
	let memOpen = $state(false);
	// The per-process nonce that authorizes POST ./moe/exit, fetched once at mount from
	// ./moe/exit-token. The server registers that route ONLY on a loopback bind, so a null
	// token also means "this server is reachable from the network" -- and the power button
	// stays hidden, because there is nothing it could successfully do.
	let exitToken = $state<string | null>(null);

	let streaming = $derived(isChatStreaming());

	// THE user-facing rate. The server computes it over a fixed ~1.5 s window and independently of
	// who polls (server_main.cpp moe_gen_rate_recent), so every observer sees the same number.
	// 0 means literally "no tokens emitted in that window" -- prefill, the prompt-end hot-expert
	// materialization, or idle. It is deliberately NOT backfilled from win.tok_per_s: that counts
	// decode CYCLES on the policy clock, which under speculative decoding is 3-4x below the rate a
	// person actually sees (measured 13-21 vs 29-59 within one turn). The old fallback is exactly
	// how the strip came to display a frozen "1.7 tok/s" for the ~5 s of prefill at the start of a
	// turn that then decoded at 49.
	let genRate = $derived(win?.gen_tok_per_s ?? 0);

	// "gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf" -> "gemma-4-26B-A4B-it" (quant/UD suffixes are runtime
	// detail; precision is explained in the mem panel instead)
	function modelDisplayName(p: string): string {
		if (!p) return '';
		let b = p.split(/[\\/]/).pop() ?? '';
		b = b.replace(/\.gguf$/i, '');
		b = b.replace(/-(UD-)?(I?Q\d[\w.]*|BF16|F16|F32)$/i, '');
		return b;
	}

	function gib(mib: number): string {
		return (mib / 1024).toFixed(1);
	}

	async function fetchInfo() {
		const r = await fetch('./moe/info');
		if (!r.ok) return false;
		const j = await r.json();
		if (j && j.state) {
			routerState = j.state as string;
			info = null;
		} else {
			routerState = '';
			info = j as MoeInfo;
		}
		return true;
	}

	async function pollTelemetry() {
		if (routerState === 'exited') return;
		try {
			const r = await fetch('./moe/telemetry');
			if (!r.ok) return;
			const j = await r.json();
			if (j && j.state) {
				// router with no ready child: track the state, drop stale identity/rates
				routerState = j.state as string;
				info = null;
				win = null;
				tele = null;
				memOpen = false;
			} else if (j && j.window) {
				if (routerState !== '' || !info) {
					// child just became ready -- (re)fetch its identity
					routerState = '';
					try {
						await fetchInfo();
					} catch {
						/* next poll retries */
					}
				}
				win = j.window as MoeWindow;
				tele = j as MoeTelemetry;
			}
		} catch {
			// server briefly unreachable -- keep the last values
		}
	}

	// --- memory panel ---------------------------------------------------------------------------
	// One segment per thing a person can act on or reason about, and EVERY measured row present.
	// Before C5.6 the telemetry payload simply did not carry compute_tgt / compute_dft /
	// drafter_weights / pool_reserved, so the panel drew four named, measured blocks of VRAM as if
	// they were free. The rule now: the segments plus `free` must sum to `total`, and the code
	// asserts it rather than trusting a flex filler to swallow the difference.
	interface MemSeg {
		label: string;
		mib: number;
		color: string;
		hint: string;
	}

	function segs(rows: [string, number, string, string][]): MemSeg[] {
		return rows
			.filter(([, mib]) => mib > 0)
			.map(([label, mib, color, hint]) => ({ label, mib, color, hint }));
	}

	let vp = $derived(tele?.partition.vram ?? null);
	let rp = $derived(tele?.partition.ram ?? null);

	let vramSegs = $derived(
		vp
			? segs([
					[
						'model weights',
						vp.nonexpert_weights,
						'#6366f1',
						'attention + non-expert tensors, resident on the GPU'
					],
					[
						'hot experts',
						vp.hot_slab + vp.l2,
						'#f59e0b',
						'the K hottest experts per layer, kept in VRAM at high precision'
					],
					[
						'prefill staging',
						vp.staging,
						'#fbbf24',
						'transient buffer used to stream experts while reading a prompt; freed after'
					],
					['chat memory (KV)', vp.kv, '#ec4899', 'the conversation itself; grows with context'],
					[
						'compute buffers',
						vp.compute_tgt + vp.compute_dft,
						'#64748b',
						'graph scratch for the model and the speculative drafter'
					],
					[
						'drafter weights',
						vp.drafter_weights,
						'#0ea5e9',
						'the MTP speculative drafter, when it has weights of its own'
					],
					[
						'CUDA scratch pools',
						vp.pool_reserved,
						'#94a3b8',
						'held from the driver by ggml-cuda and never returned; grows, then plateaus'
					],
					[
						'runtime overhead',
						vp.unaccounted,
						'#475569',
						'CUDA primary context, cuBLAS workspace, allocator slack'
					],
					['other apps', vp.other_procs, '#9ca3af', 'everything on this GPU that is not LLapis']
				])
			: []
	);

	// Ledger check: what we attribute to OUR process vs what the OS says our process holds.
	// Shown when it disagrees by more than rounding -- a silent mismatch here is what the old
	// panel had permanently, and it is the single most useful signal that the model is wrong.
	let vramOursSum = $derived(
		vp
			? vp.nonexpert_weights +
					vp.hot_slab +
					vp.l2 +
					vp.staging +
					vp.kv +
					vp.compute_tgt +
					vp.compute_dft +
					vp.drafter_weights +
					vp.pool_reserved +
					vp.unaccounted
			: 0
	);
	let vramLedgerDelta = $derived(vp ? vramOursSum - vp.ours : 0);

	let ramSegs = $derived(
		rp
			? segs([
					[
						'cold experts (resident)',
						rp.arena_resident,
						'#0ea5e9',
						'the part of the IQ2 expert arena physically in RAM right now'
					],
					[
						'runtime',
						rp.nonarena_rss,
						'#6366f1',
						'everything else this process holds: weights in RAM, buffers, code'
					],
					['other apps', rp.other_procs, '#9ca3af', 'everything on this machine that is not LLapis']
				])
			: []
	);

	// The arena is committed address space backed on demand: allocated is NOT resident, and the
	// gap is usually most of it. Surfaced as its own line because "why does the bar not show my
	// 7.9 GiB arena" is exactly the question the old panel answered wrongly.
	let arenaPagedOut = $derived(rp ? Math.max(0, rp.arena_committed - rp.arena_resident) : 0);

	// Growth of the WDDM shared segment above the boot baseline. THIS is the over-commit signal;
	// the absolute value is normal-and-large. -1 baseline = the server has not published one.
	let sharedGrowth = $derived(
		vp && vp.shared_baseline >= 0 ? Math.max(0, vp.shared_ours - vp.shared_baseline) : 0
	);

	async function fetchExitToken() {
		try {
			const r = await fetch('./moe/exit-token', { cache: 'no-store' });
			if (!r.ok) return; // LAN bind (route not registered) or stock server: stay hidden
			const j = await r.json();
			exitToken = typeof j?.token === 'string' && j.token ? j.token : null;
		} catch {
			exitToken = null;
		}
	}

	async function exitRuntime() {
		if (!exitToken) return;
		if (!confirm('Exit LLapis? The model unloads and the server stops.')) return;
		try {
			await fetch('./moe/exit', {
				method: 'POST',
				headers: { 'X-LLapis-Exit-Token': exitToken }
			});
			routerState = 'exited';
			info = null;
			win = null;
		} catch {
			/* server already gone counts as success */
		}
	}

	onMount(() => {
		let interval: ReturnType<typeof setInterval> | undefined;

		(async () => {
			try {
				if (!(await fetchInfo())) return; // stock llama-server: no /moe/*, stay hidden
				present = true;
				await fetchExitToken();
				await pollTelemetry();
				interval = setInterval(pollTelemetry, 1000);
			} catch {
				// unreachable at mount: stay hidden (page reload recovers)
			}
		})();

		return () => {
			if (interval !== undefined) clearInterval(interval);
		};
	});
</script>

{#if present}
	<div
		class="pointer-events-none fixed top-2 left-1/2 z-40 -translate-x-1/2"
		data-testid="moe-header-strip"
	>
		<div
			class="pointer-events-auto flex max-w-[92vw] flex-wrap items-center justify-center gap-x-2 gap-y-0.5 rounded-full border border-border/60 bg-background/80 px-3 py-1 text-xs leading-tight backdrop-blur"
		>
			<span class="font-semibold text-primary">LLapis</span>

			{#if routerState === 'exited'}
				<span class="text-destructive">runtime exited — close this tab</span>
			{:else if routerState === 'loading'}
				<span class="text-muted-foreground">loading model…</span>
			{:else if routerState === 'unloaded'}
				<span class="text-muted-foreground">no model loaded</span>
			{:else if info}
				<span class="max-w-[16rem] truncate font-medium" title={info.model_high}>
					{modelDisplayName(info.model_high) || 'model'}
				</span>

				<span
					class="text-muted-foreground"
					title="K = hot experts kept in fast GPU memory per layer (of {info.n_expert} per layer), re-fitted to measured VRAM at every prompt and every context change · context window, which grows on demand"
				>
					K{info.k} · ctx {info.ctx > 0 ? info.ctx : '?'}
				</span>

				{#if streaming}
					<span class="text-border">|</span>
					{#if genRate > 0}
						<span
							class="font-medium tabular-nums"
							title="generation rate: emitted tokens per second including accepted speculative drafts, over the last ~1.5 s"
						>
							{genRate.toFixed(1)} tok/s
						</span>
						{#if win}
							<span
								class="font-medium tabular-nums"
								title="share of expert computation served at high precision over the last {win.tokens} tokens"
							>
								hot {(win.hit_rate * 100).toFixed(0)}%
							</span>
						{/if}
					{:else}
						<!-- No tokens emitted in the last window: the turn is still reading the prompt (or
						     materializing hot experts right after it). Say that. A wrong number is worse
						     than no number, and the number this used to fall back to was wrong. -->
						<span class="text-muted-foreground">reading prompt…</span>
					{/if}
				{/if}
			{/if}

			{#if routerState === '' && info && tele}
				<button
					class="ml-1 transition-colors {memOpen
						? 'text-foreground'
						: 'text-muted-foreground hover:text-foreground'}"
					title="Memory: where the model lives (VRAM / system RAM)"
					aria-label="Memory"
					onclick={() => (memOpen = !memOpen)}
				>
					▤ mem
				</button>
			{/if}
			{#if routerState !== 'exited' && exitToken}
				<button
					class="ml-1 text-muted-foreground transition-colors hover:text-destructive"
					title="Exit: unload the model and stop the LLapis server"
					aria-label="Exit Runtime"
					onclick={exitRuntime}
				>
					⏻
				</button>
			{/if}
		</div>

		{#if memOpen && routerState === '' && vp && rp && info && tele}
			<div
				class="pointer-events-auto mx-auto mt-1 w-[30rem] max-w-[92vw] rounded-lg border border-border/60 bg-background/95 p-2 text-xs shadow-lg backdrop-blur"
				data-testid="moe-mem-panel"
			>
				<!-- expert cache: the dual-precision design in one line -->
				<div class="mb-2 border-b border-border/40 pb-1.5">
					<div class="flex justify-between">
						<span class="font-medium">Expert cache</span>
						<span class="tabular-nums text-muted-foreground">
							K{info.k} × {info.n_expert < 0 ? '?' : info.n_expert} experts/layer
						</span>
					</div>
					<div class="mt-0.5 flex justify-between text-muted-foreground">
						<span>{info.k} hottest experts per layer stay in VRAM</span>
						<span class="tabular-nums">{(100 * tele.occupancy).toFixed(0)}% filled</span>
					</div>
					<div class="flex justify-between text-muted-foreground">
						<span>rest served from system RAM at IQ2</span>
						<span class="tabular-nums">served hot {(100 * tele.hit_rate).toFixed(0)}%</span>
					</div>
				</div>

				<!-- ================= VRAM ================= -->
				<div class="mb-0.5 flex justify-between text-muted-foreground">
					<span>GPU memory (VRAM)</span>
					<span class="tabular-nums">
						LLapis {gib(vp.ours)} · free {gib(vp.free)} / {gib(vp.total)} GiB
					</span>
				</div>
				<div class="flex h-3 w-full overflow-hidden rounded border border-border/40">
					{#each vramSegs as s (s.label)}
						<div
							class="h-full"
							style="width:{(100 * s.mib) / vp.total}%;background:{s.color}"
							title="{s.label}: {gib(s.mib)} GiB — {s.hint}"
						></div>
					{/each}
					<div class="h-full flex-1 bg-muted/30" title="free: {gib(vp.free)} GiB"></div>
				</div>
				<div class="mt-1 mb-1 grid grid-cols-2 gap-x-3 gap-y-0.5 tabular-nums">
					{#each vramSegs as s (s.label)}
						<div class="flex items-center justify-between gap-2" title={s.hint}>
							<span class="flex items-center gap-1 text-muted-foreground">
								<span class="inline-block h-2 w-2 rounded-sm" style="background:{s.color}"></span>
								{s.label}
							</span>
							<span>{gib(s.mib)} GiB</span>
						</div>
					{/each}
					<div class="flex items-center justify-between gap-2">
						<span class="flex items-center gap-1 text-muted-foreground">
							<span class="inline-block h-2 w-2 rounded-sm bg-muted/60"></span>
							free
						</span>
						<span>{gib(vp.free)} GiB</span>
					</div>
				</div>
				{#if Math.abs(vramLedgerDelta) > 64}
					<!-- The partition model and the OS disagree about what this process holds. Not fatal,
					     but it means one of the rows above is wrong, and saying so beats drawing a
					     confident picture built on it. -->
					<div class="mb-1 text-amber-600 dark:text-amber-400">
						ledger off by {vramLedgerDelta > 0 ? '+' : ''}{vramLedgerDelta} MiB vs the OS view of
						this process
					</div>
				{/if}
				{#if vp.shared_ours > 0}
					<!-- WDDM shared (system-memory) segment. The ABSOLUTE number is not a problem and
					     must not be presented as one: a steady 0.5-0.9 GiB here is host memory the GPU
					     is allowed to address, dominated by the pinned/registered cold arena, and
					     it was measured bit-identical across a full 3-tier sweep with nothing
					     spilling. Only GROWTH above the boot baseline is the overcommit signature
					     (Windows silently backing over-committed VRAM with system RAM instead of
					     failing the allocation). So: report the baseline, flag the delta. -->
					<div
						class="mb-1 {sharedGrowth > 256 ? 'text-amber-600 dark:text-amber-400' : 'text-muted-foreground'}"
						title={sharedGrowth > 256
							? 'This process is using more shared system memory than it did at boot — VRAM allocations are being backed by system RAM, which is slow.'
							: 'Host memory the GPU can address (mostly the pinned cold-expert arena). Normal, and not part of the VRAM budget — only growth above the boot figure would indicate over-commit.'}
					>
						shared system memory: {gib(vp.shared_ours)} GiB
						{#if vp.shared_baseline >= 0}
							(boot baseline {gib(vp.shared_baseline)} GiB{sharedGrowth > 256
								? `, +${gib(sharedGrowth)} GiB — over-committing`
								: ', steady'})
						{/if}
					</div>
				{/if}

				<!-- ================= RAM ================= -->
				<div class="mt-2 mb-0.5 flex justify-between text-muted-foreground">
					<span>System RAM</span>
					<span class="tabular-nums">
						LLapis {gib(rp.rss)} · free {gib(rp.avail)} / {gib(rp.total)} GiB
					</span>
				</div>
				<div class="flex h-3 w-full overflow-hidden rounded border border-border/40">
					{#each ramSegs as s (s.label)}
						<div
							class="h-full"
							style="width:{(100 * s.mib) / rp.total}%;background:{s.color}"
							title="{s.label}: {gib(s.mib)} GiB — {s.hint}"
						></div>
					{/each}
					<div class="h-full flex-1 bg-muted/30" title="available: {gib(rp.avail)} GiB"></div>
				</div>
				<div class="mt-1 grid grid-cols-2 gap-x-3 gap-y-0.5 tabular-nums">
					{#each ramSegs as s (s.label)}
						<div class="flex items-center justify-between gap-2" title={s.hint}>
							<span class="flex items-center gap-1 text-muted-foreground">
								<span class="inline-block h-2 w-2 rounded-sm" style="background:{s.color}"></span>
								{s.label}
							</span>
							<span>{gib(s.mib)} GiB</span>
						</div>
					{/each}
					<div class="flex items-center justify-between gap-2">
						<span class="flex items-center gap-1 text-muted-foreground">
							<span class="inline-block h-2 w-2 rounded-sm bg-muted/60"></span>
							available
						</span>
						<span>{gib(rp.avail)} GiB</span>
					</div>
				</div>
				{#if rp.arena_committed > 0}
					<div
						class="mt-1 text-muted-foreground"
						title="The cold expert arena is committed address space that Windows backs with physical pages on demand. Only the part the model actually routes to stays resident; the rest costs address space, not RAM."
					>
						cold expert arena: {gib(rp.arena_committed)} GiB reserved,
						{gib(rp.arena_resident)} GiB resident
						{#if arenaPagedOut > 0}
							({gib(arenaPagedOut)} GiB paged out)
						{/if}
						{#if rp.arena_pinned > 0}
							· {gib(rp.arena_pinned)} GiB page-locked
						{/if}
					</div>
				{/if}
			</div>
		{/if}
	</div>
{/if}
