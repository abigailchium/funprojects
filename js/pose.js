const PoseDetector = (() => {
  let pose = null;
  let latestLandmarks = null;
  let onResultsCallback = null;
  let animFrameId = null;

  function init() {
    pose = new Pose({
      locateFile: (file) =>
        `https://cdn.jsdelivr.net/npm/@mediapipe/pose@0.5.1675469404/${file}`
    });
    pose.setOptions({
      modelComplexity: 1,
      smoothLandmarks: true,
      enableSegmentation: false,
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5
    });
    pose.onResults(handleResults);
  }

  function handleResults(results) {
    if (results.poseLandmarks && results.poseLandmarks.length === 33) {
      latestLandmarks = results.poseLandmarks;
    } else {
      latestLandmarks = null;
    }
    if (onResultsCallback) {
      onResultsCallback(results);
    }
  }

  function drawLandmarks(canvasEl, results) {
    const ctx = canvasEl.getContext('2d');
    ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);

    if (!results.poseLandmarks) return;

    drawConnectors(ctx, results.poseLandmarks, POSE_CONNECTIONS, {
      color: '#00FF00',
      lineWidth: 2
    });

    drawLandmarks_(ctx, results.poseLandmarks, {
      color: '#FF0000',
      lineWidth: 1,
      radius: 3
    });
  }

  function drawLandmarks_(ctx, landmarks, style) {
    window.drawLandmarks(ctx, landmarks, style);
  }

  async function startProcessing(videoElement, canvasElement, callback) {
    onResultsCallback = (results) => {
      canvasElement.width = videoElement.videoWidth;
      canvasElement.height = videoElement.videoHeight;
      drawLandmarks(canvasElement, results);
      if (callback) callback(results);
    };

    const sendFrame = async () => {
      if (videoElement.readyState >= 2 && pose) {
        await pose.send({ image: videoElement });
      }
      animFrameId = requestAnimationFrame(sendFrame);
    };
    sendFrame();
  }

  function stopProcessing() {
    if (animFrameId) {
      cancelAnimationFrame(animFrameId);
      animFrameId = null;
    }
    onResultsCallback = null;
  }

  function getLandmarks() {
    return latestLandmarks;
  }

  // Store raw 132 values (33 landmarks x 4: x, y, z, visibility)
  // so we always keep the full data in IndexedDB for future reprocessing.
  function extractRawLandmarks(landmarks) {
    if (!landmarks || landmarks.length !== 33) return null;
    const raw = new Float32Array(132);
    for (let i = 0; i < 33; i++) {
      raw[i * 4]     = landmarks[i].x;
      raw[i * 4 + 1] = landmarks[i].y;
      raw[i * 4 + 2] = landmarks[i].z;
      raw[i * 4 + 3] = landmarks[i].visibility;
    }
    return raw;
  }

  // Focused biomechanical features (13 features) derived from raw landmarks.
  // Accepts either:
  //   - MediaPipe landmark array [{x,y,z,visibility}, ...] (33 elements)
  //   - Stored raw Float32Array/Array of 132 values [x,y,z,vis, x,y,z,vis, ...]
  //
  // Key signals:
  //   * ear-to-shoulder vertical/depth distance → slouch detection
  //   * face proximity (inter-ear, inter-eye, nose depth) → screen closeness
  //   * head-forward offset (ear Z vs shoulder Z)
  const FEATURE_COUNT = 13;

  function extractPostureFeatures(input) {
    if (!input) return null;

    // Resolve to a uniform accessor
    let get;
    if (input.length === 33 && typeof input[0] === 'object' && input[0].x !== undefined) {
      // MediaPipe landmark objects
      get = (i) => input[i];
    } else if (input.length === 132 || input.length >= 132) {
      // Stored flat array [x,y,z,vis, ...]
      get = (i) => ({
        x: input[i * 4],
        y: input[i * 4 + 1],
        z: input[i * 4 + 2],
        visibility: input[i * 4 + 3]
      });
    } else {
      return null;
    }

    const nose          = get(0);
    const leftEyeInner = get(1);
    const rightEyeInner = get(4);
    const leftEar       = get(7);
    const rightEar      = get(8);
    const leftShoulder  = get(11);
    const rightShoulder = get(12);

    // Shoulder width (Euclidean in x,y) — used to normalise distances so
    // the features are scale-invariant (works at different camera distances).
    const shoulderWidth = Math.sqrt(
      (rightShoulder.x - leftShoulder.x) ** 2 +
      (rightShoulder.y - leftShoulder.y) ** 2
    ) || 0.001;

    const shoulderMidY = (leftShoulder.y + rightShoulder.y) / 2;
    const shoulderMidZ = (leftShoulder.z + rightShoulder.z) / 2;

    return new Float32Array([
      // 0-1  Ear-shoulder VERTICAL distance (normalised).
      //      More negative = ear well above shoulder = good posture.
      //      Closer to 0 or positive = slouching.
      (leftEar.y  - leftShoulder.y)  / shoulderWidth,
      (rightEar.y - rightShoulder.y) / shoulderWidth,

      // 2-3  Ear-shoulder DEPTH distance.
      //      More negative = head forward of shoulders = bad.
      (leftEar.z  - leftShoulder.z),
      (rightEar.z - rightShoulder.z),

      // 4    Nose-to-shoulder-midpoint vertical (normalised).
      (nose.y - shoulderMidY) / shoulderWidth,

      // 5    Nose depth relative to shoulders.
      (nose.z - shoulderMidZ),

      // 6    Inter-ear distance / shoulder width — face proximity ratio.
      //      Larger = face closer to screen.
      Math.sqrt(
        (rightEar.x - leftEar.x) ** 2 +
        (rightEar.y - leftEar.y) ** 2
      ) / shoulderWidth,

      // 7    Inter-eye distance / shoulder width — another proximity ratio.
      Math.sqrt(
        (rightEyeInner.x - leftEyeInner.x) ** 2 +
        (rightEyeInner.y - leftEyeInner.y) ** 2
      ) / shoulderWidth,

      // 8    Nose raw Z — direct face-to-camera depth.
      nose.z,

      // 9-10 Ear-shoulder LATERAL offset (normalised).
      (leftEar.x  - leftShoulder.x)  / shoulderWidth,
      (rightEar.x - rightShoulder.x) / shoulderWidth,

      // 11   Head tilt (ear height difference, normalised).
      (leftEar.y - rightEar.y) / shoulderWidth,

      // 12   Raw shoulder width — absolute scale reference.
      shoulderWidth
    ]);
  }

  return {
    init,
    startProcessing,
    stopProcessing,
    getLandmarks,
    extractRawLandmarks,
    extractPostureFeatures,
    FEATURE_COUNT
  };
})();
