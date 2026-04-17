const Camera = (() => {
  let stream = null;

  async function start(videoElement) {
    if (stream) stop(videoElement);
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480, facingMode: 'user' },
      audio: false
    });
    videoElement.srcObject = stream;
    await videoElement.play();
  }

  function stop(videoElement) {
    if (stream) {
      stream.getTracks().forEach(t => t.stop());
      stream = null;
    }
    if (videoElement) {
      videoElement.srcObject = null;
    }
  }

  function isRunning() {
    return stream !== null && stream.active;
  }

  return { start, stop, isRunning };
})();
