const Alarm = (() => {
  let audioCtx = null;
  let oscillator = null;
  let gainNode = null;
  let isPlaying = false;

  // Generate alarm using Web Audio API (no external file needed)
  function ensureContext() {
    if (!audioCtx) {
      audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    }
    if (audioCtx.state === 'suspended') {
      audioCtx.resume();
    }
  }

  function play() {
    if (isPlaying) return;
    ensureContext();

    // Create an aggressive alarm tone
    oscillator = audioCtx.createOscillator();
    gainNode = audioCtx.createGain();

    oscillator.type = 'square';
    oscillator.frequency.value = 440;
    gainNode.gain.value = 0.3;

    // Modulate frequency for siren effect
    const lfo = audioCtx.createOscillator();
    const lfoGain = audioCtx.createGain();
    lfo.frequency.value = 3; // 3 Hz wobble
    lfoGain.gain.value = 200; // frequency deviation

    lfo.connect(lfoGain);
    lfoGain.connect(oscillator.frequency);
    oscillator.connect(gainNode);
    gainNode.connect(audioCtx.destination);

    lfo.start();
    oscillator.start();

    oscillator._lfo = lfo;
    oscillator._lfoGain = lfoGain;

    isPlaying = true;
  }

  function stop() {
    if (!isPlaying) return;
    try {
      if (oscillator) {
        oscillator._lfo.stop();
        oscillator.stop();
        oscillator.disconnect();
        oscillator._lfo.disconnect();
        oscillator._lfoGain.disconnect();
        oscillator = null;
      }
      if (gainNode) {
        gainNode.disconnect();
        gainNode = null;
      }
    } catch (e) {
      // Ignore errors from already-stopped nodes
    }
    isPlaying = false;
  }

  function getIsPlaying() {
    return isPlaying;
  }

  return { play, stop, isPlaying: getIsPlaying };
})();
