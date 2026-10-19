<script lang="ts">
	import { onMount } from 'svelte';
	import mascotEopen from '$lib/assets/mascot-eopen.png';
	import mascotEclosed from '$lib/assets/mascot-eclosed.png';

	interface Props {
		class?: string;
	}

	let { class: className = '' }: Props = $props();

	// Blink cycle: idle open (2s) -> closed -> open -> closed -> back to idle open.
	// Same hold duration for every blink frame so it reads as a deliberate blink,
	// not a glitch.
	const IDLE_MS = 2000;
	const BLINK_FRAME_MS = 130;

	let eyesOpen = $state(true);

	onMount(() => {
		if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;

		let timer: ReturnType<typeof setTimeout>;
		const sequence = [IDLE_MS, BLINK_FRAME_MS, BLINK_FRAME_MS, BLINK_FRAME_MS];
		let step = 0;

		function tick() {
			eyesOpen = step % 2 === 0;
			timer = setTimeout(tick, sequence[step]);
			step = (step + 1) % sequence.length;
		}

		timer = setTimeout(tick, sequence[step]);
		step = (step + 1) % sequence.length;

		return () => clearTimeout(timer);
	});
</script>

<img class={className} src={eyesOpen ? mascotEopen : mascotEclosed} alt="" aria-hidden="true" />
