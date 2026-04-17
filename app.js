const App = (() => {
  // ---- DOM refs ----
  const navButtons = document.querySelectorAll('.nav-btn');
  const views      = document.querySelectorAll('.view');

  // Collect view
  const videoCollect  = document.getElementById('video-collect');
  const canvasCollect = document.getElementById('canvas-collect');
  const noPoseCollect = document.getElementById('no-pose-collect');
  const btnGood       = document.getElementById('btn-good');
  const btnBad        = document.getElementById('btn-bad');
  const btnClear      = document.getElementById('btn-clear');
  const btnToTrain    = document.getElementById('btn-to-train');
  const goodCountEl   = document.getElementById('good-count');
  const badCountEl    = document.getElementById('bad-count');

  // Train view
  const trainSampleCount = document.getElementById('train-sample-count');
  const btnTrain         = document.getElementById('btn-train');
  const trainingProgress = document.getElementById('training-progress');
  const progressBar      = document.getElementById('progress-bar');
  const progressText     = document.getElementById('progress-text');
  const metricLoss       = document.getElementById('metric-loss');
  const metricAccuracy   = document.getElementById('metric-accuracy');
  const trainResults     = document.getElementById('train-results');
  const valAccuracy      = document.getElementById('val-accuracy');
  const valLoss          = document.getElementById('val-loss');
  const btnToMonitor     = document.getElementById('btn-to-monitor');

  // Monitor view
  const videoMonitor   = document.getElementById('video-monitor');
  const canvasMonitor  = document.getElementById('canvas-monitor');
  const noPoseMonitor  = document.getElementById('no-pose-monitor');
  const postureStatus  = document.getElementById('posture-status');
  const statusText     = document.getElementById('status-text');
  const warningBanner  = document.getElementById('warning-banner');
  const thresholdSlider = document.getElementById('threshold-slider');
  const thresholdValue  = document.getElementById('threshold-value');
  const btnStopMonitor  = document.getElementById('btn-stop-monitor');

  // Calibration overlay
  const calOverlay     = document.getElementById('calibration-overlay');
  const videoCal       = document.getElementById('video-calibrate');
  const canvasCal      = document.getElementById('canvas-calibrate');
  const noPoseCal      = document.getElementById('no-pose-calibrate');
  const calGoodCount   = document.getElementById('cal-good-count');
  const calBadCount    = document.getElementById('cal-bad-count');
  const calBtnGood     = document.getElementById('cal-btn-good');
  const calBtnBad      = document.getElementById('cal-btn-bad');
  const calStatus      = document.getElementById('cal-status');
  const calBtnDone     = document.getElementById('cal-btn-done');
  const calBtnSkip     = document.getElementById('cal-btn-skip');

  let currentView     = 'collect';
  let monitorInterval = null;
  let poseHasLandmarks = false;

  // Calibration state
  const CAL_TARGET = 5; // samples per class per session
  let calGood = 0;
  let calBad  = 0;

  // =======================================================================
  //  SESSION CALIBRATION
  // =======================================================================

  async function showCalibration() {
    calGood = 0;
    calBad  = 0;
    updateCalCounts();
    calStatus.hidden = true;
    calBtnDone.disabled = true;
    calOverlay.hidden = false;

    try {
      await Camera.start(videoCal);
      PoseDetector.startProcessing(videoCal, canvasCal, (results) => {
        const hasPose = !!results.poseLandmarks;
        noPoseCal.hidden = hasPose;
        calBtnGood.disabled = !hasPose;
        calBtnBad.disabled  = !hasPose;
      });
    } catch (err) {
      console.error('Camera error during calibration:', err);
      noPoseCal.textContent = 'Camera access denied or unavailable';
      noPoseCal.hidden = false;
    }
  }

  function updateCalCounts() {
    calGoodCount.textContent = `Good: ${calGood} / ${CAL_TARGET}`;
    calBadCount.textContent  = `Bad: ${calBad} / ${CAL_TARGET}`;
    calBtnDone.disabled = calGood < CAL_TARGET || calBad < CAL_TARGET;
  }

  async function calCapture(label) {
    const landmarks = PoseDetector.getLandmarks();
    if (!landmarks) return;
    const raw = PoseDetector.extractRawLandmarks(landmarks);
    if (!raw) return;

    await DataStore.addSample(raw, label);

    if (label === 1) calGood++;
    else calBad++;
    updateCalCounts();

    // Flash
    const container = videoCal.closest('.camera-container');
    const flash = document.createElement('div');
    flash.className = 'capture-flash';
    container.appendChild(flash);
    setTimeout(() => flash.remove(), 300);
  }

  async function calFinish() {
    // Show training progress in the overlay
    calBtnDone.disabled = true;
    calBtnGood.disabled = true;
    calBtnBad.disabled  = true;
    calStatus.hidden = false;
    calStatus.textContent = 'Retraining on all data...';

    PoseDetector.stopProcessing();
    Camera.stop(videoCal);

    const samples = await DataStore.getAllSamples();
    calStatus.textContent = `Training on ${samples.length} total samples...`;

    await PostureModel.trainModel(samples, (epoch, total, logs) => {
      const accVal = logs.acc !== undefined ? logs.acc : logs.accuracy;
      calStatus.textContent =
        `Epoch ${epoch}/${total} — acc ${(accVal * 100).toFixed(0)}%`;
    });
    await PostureModel.saveModel();

    calStatus.textContent = 'Model updated! Starting monitoring...';
    await new Promise(r => setTimeout(r, 600));

    calOverlay.hidden = true;
    switchView('monitor');
  }

  function calSkip() {
    PoseDetector.stopProcessing();
    Camera.stop(videoCal);
    calOverlay.hidden = true;
    // Go to monitor if model exists, otherwise collect
    if (PostureModel.isLoaded()) {
      switchView('monitor');
    } else {
      switchView('collect');
    }
  }

  // =======================================================================
  //  VIEW SWITCHING
  // =======================================================================

  function switchView(viewName) {
    if (currentView === 'collect') {
      PoseDetector.stopProcessing();
      Camera.stop(videoCollect);
    }
    if (currentView === 'monitor') {
      stopMonitoring();
    }

    currentView = viewName;
    navButtons.forEach(b => b.classList.toggle('active', b.dataset.view === viewName));
    views.forEach(v => v.classList.toggle('active', v.id === `view-${viewName}`));

    if (viewName === 'collect') initCollectView();
    if (viewName === 'train')   initTrainView();
    if (viewName === 'monitor') initMonitorView();
  }

  // =======================================================================
  //  COLLECT VIEW
  // =======================================================================

  async function initCollectView() {
    await updateCounts();
    try {
      await Camera.start(videoCollect);
      PoseDetector.startProcessing(videoCollect, canvasCollect, (results) => {
        poseHasLandmarks = !!results.poseLandmarks;
        noPoseCollect.hidden = poseHasLandmarks;
        btnGood.disabled = !poseHasLandmarks;
        btnBad.disabled  = !poseHasLandmarks;
      });
    } catch (err) {
      console.error('Camera error:', err);
      noPoseCollect.textContent = 'Camera access denied or unavailable';
      noPoseCollect.hidden = false;
    }
  }

  async function updateCounts() {
    const counts = await DataStore.getCounts();
    goodCountEl.textContent = `Good: ${counts.good}`;
    badCountEl.textContent  = `Bad: ${counts.bad}`;
    btnToTrain.disabled = counts.good < 20 || counts.bad < 20;
  }

  async function captureSample(label) {
    const landmarks = PoseDetector.getLandmarks();
    if (!landmarks) return;
    const raw = PoseDetector.extractRawLandmarks(landmarks);
    if (!raw) return;

    await DataStore.addSample(raw, label);
    await updateCounts();

    const container = videoCollect.closest('.camera-container');
    const flash = document.createElement('div');
    flash.className = 'capture-flash';
    container.appendChild(flash);
    setTimeout(() => flash.remove(), 300);
  }

  // =======================================================================
  //  TRAIN VIEW
  // =======================================================================

  async function initTrainView() {
    trainingProgress.hidden = true;
    trainResults.hidden = true;
    btnTrain.disabled = false;

    const counts = await DataStore.getCounts();
    trainSampleCount.textContent =
      `${counts.good + counts.bad} (Good: ${counts.good}, Bad: ${counts.bad})`;
  }

  async function trainModelHandler() {
    btnTrain.disabled = true;
    trainingProgress.hidden = false;
    trainResults.hidden = true;

    const samples = await DataStore.getAllSamples();

    const results = await PostureModel.trainModel(samples, (epoch, total, logs) => {
      const pct = Math.round((epoch / total) * 100);
      progressBar.style.width = `${pct}%`;
      progressText.textContent = `Epoch ${epoch} / ${total}`;
      metricLoss.textContent = logs.loss.toFixed(4);
      const accVal = logs.acc !== undefined ? logs.acc : logs.accuracy;
      metricAccuracy.textContent = (accVal * 100).toFixed(1) + '%';
    });

    await PostureModel.saveModel();

    valAccuracy.textContent = (results.valAccuracy * 100).toFixed(1) + '%';
    valLoss.textContent     = results.valLoss.toFixed(4);
    trainResults.hidden = false;
  }

  // =======================================================================
  //  MONITOR VIEW
  // =======================================================================

  async function initMonitorView() {
    if (!PostureModel.isLoaded()) {
      const loaded = await PostureModel.loadModel();
      if (!loaded) {
        statusText.textContent = 'No model — train first';
        postureStatus.className = 'posture-status bad';
        return;
      }
    }

    try {
      await Camera.start(videoMonitor);
      PoseDetector.startProcessing(videoMonitor, canvasMonitor, (results) => {
        poseHasLandmarks = !!results.poseLandmarks;
        noPoseMonitor.hidden = poseHasLandmarks;
      });
      startMonitoring();
    } catch (err) {
      console.error('Camera error:', err);
      noPoseMonitor.textContent = 'Camera access denied or unavailable';
      noPoseMonitor.hidden = false;
    }
  }

  function startMonitoring() {
    btnStopMonitor.textContent = 'Stop Monitoring';
    monitorInterval = setInterval(() => {
      const landmarks = PoseDetector.getLandmarks();
      if (!landmarks) return;

      const prob = PostureModel.predict(landmarks);
      if (prob === null) return;

      const threshold = parseFloat(thresholdSlider.value);
      const isGood = prob >= threshold;

      if (isGood) {
        postureStatus.className = 'posture-status good';
        statusText.innerHTML = 'Good Posture &#x2713;';
        warningBanner.hidden = true;
        document.body.classList.remove('alarm-active');
        Alarm.stop();
      } else {
        postureStatus.className = 'posture-status bad';
        statusText.innerHTML = 'Bad Posture &#x2717;';
        warningBanner.hidden = false;
        document.body.classList.add('alarm-active');
        Alarm.play();
      }
    }, 500);
  }

  function stopMonitoring() {
    if (monitorInterval) {
      clearInterval(monitorInterval);
      monitorInterval = null;
    }
    Alarm.stop();
    document.body.classList.remove('alarm-active');
    warningBanner.hidden = true;
    postureStatus.className = 'posture-status good';
    statusText.innerHTML = 'Good Posture &#x2713;';
    PoseDetector.stopProcessing();
    Camera.stop(videoMonitor);
  }

  // =======================================================================
  //  EVENT BINDING
  // =======================================================================

  function bindEvents() {
    // Nav
    navButtons.forEach(btn => {
      btn.addEventListener('click', () => switchView(btn.dataset.view));
    });

    // Collect
    btnGood.addEventListener('click', () => captureSample(1));
    btnBad.addEventListener('click',  () => captureSample(0));
    btnClear.addEventListener('click', async () => {
      if (confirm('Clear all collected posture data?')) {
        await DataStore.clearAll();
        await updateCounts();
      }
    });
    btnToTrain.addEventListener('click', () => switchView('train'));

    // Train
    btnTrain.addEventListener('click', trainModelHandler);
    btnToMonitor.addEventListener('click', () => switchView('monitor'));

    // Monitor
    thresholdSlider.addEventListener('input', () => {
      thresholdValue.textContent = parseFloat(thresholdSlider.value).toFixed(2);
    });
    btnStopMonitor.addEventListener('click', () => {
      if (monitorInterval) {
        stopMonitoring();
        btnStopMonitor.textContent = 'Resume Monitoring';
      } else {
        initMonitorView();
      }
    });

    // Calibration
    calBtnGood.addEventListener('click', () => calCapture(1));
    calBtnBad.addEventListener('click',  () => calCapture(0));
    calBtnDone.addEventListener('click', calFinish);
    calBtnSkip.addEventListener('click', calSkip);
  }

  // =======================================================================
  //  INIT — decide whether to show calibration or first-run collect
  // =======================================================================

  async function init() {
    PoseDetector.init();
    await DataStore.open();
    bindEvents();

    const counts = await DataStore.getCounts();
    const hasData = counts.good >= 20 && counts.bad >= 20;

    if (hasData) {
      // Returning user — try loading model, then show calibration
      await PostureModel.loadModel();
      showCalibration();
    } else {
      // First-timer — normal collect flow
      initCollectView();
    }
  }

  document.addEventListener('DOMContentLoaded', init);

  return { switchView };
})();
