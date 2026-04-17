const PostureModel = (() => {
  const MODEL_KEY = 'indexeddb://posture-model';
  const MODEL_VERSION = 2; // bump when feature extraction changes
  let model = null;
  let normStats = null; // { mean: Float32Array(N), std: Float32Array(N) }

  const N = PoseDetector.FEATURE_COUNT; // 13 focused biomechanical features

  function createModel() {
    const m = tf.sequential();
    m.add(tf.layers.dense({ inputShape: [N], units: 32, activation: 'relu' }));
    m.add(tf.layers.dense({ units: 16, activation: 'relu' }));
    m.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));
    m.compile({
      optimizer: tf.train.adam(0.001),
      loss: 'binaryCrossentropy',
      metrics: ['accuracy']
    });
    return m;
  }

  function computeNormStats(data) {
    const count = data.length;
    const mean = new Float32Array(N);
    const std = new Float32Array(N);

    for (const sample of data) {
      for (let i = 0; i < N; i++) mean[i] += sample[i];
    }
    for (let i = 0; i < N; i++) mean[i] /= count;

    for (const sample of data) {
      for (let i = 0; i < N; i++) {
        const diff = sample[i] - mean[i];
        std[i] += diff * diff;
      }
    }
    for (let i = 0; i < N; i++) {
      std[i] = Math.sqrt(std[i] / count) || 1;
    }
    return { mean, std };
  }

  function normalize(features, stats) {
    const result = new Float32Array(N);
    for (let i = 0; i < N; i++) {
      result[i] = (features[i] - stats.mean[i]) / stats.std[i];
    }
    return result;
  }

  function shuffle(arr) {
    for (let i = arr.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [arr[i], arr[j]] = [arr[j], arr[i]];
    }
    return arr;
  }

  // Train from raw IndexedDB samples.
  // Each sample has .landmarks (raw 132-float array) and .label (0|1).
  // We derive the focused posture features from the raw landmarks.
  async function trainModel(samples, onEpochEnd) {
    // Derive focused features from stored raw landmarks
    const features = samples.map(s =>
      PoseDetector.extractPostureFeatures(s.landmarks)
    );
    const labels = samples.map(s => s.label);

    // Drop any samples where extraction failed
    const valid = [];
    for (let i = 0; i < features.length; i++) {
      if (features[i]) valid.push(i);
    }

    const cleanFeatures = valid.map(i => features[i]);
    const cleanLabels   = valid.map(i => labels[i]);

    normStats = computeNormStats(cleanFeatures);
    const normalized = cleanFeatures.map(f => normalize(f, normStats));

    // Shuffle + 80/20 split
    const indices  = shuffle([...Array(valid.length).keys()]);
    const splitIdx = Math.floor(indices.length * 0.8);
    const trainIdx = indices.slice(0, splitIdx);
    const valIdx   = indices.slice(splitIdx);

    const xTrain = tf.tensor2d(trainIdx.map(i => Array.from(normalized[i])));
    const yTrain = tf.tensor1d(trainIdx.map(i => cleanLabels[i]), 'float32');
    const xVal   = tf.tensor2d(valIdx.map(i => Array.from(normalized[i])));
    const yVal   = tf.tensor1d(valIdx.map(i => cleanLabels[i]), 'float32');

    model = createModel();

    const totalEpochs = 50;
    const history = await model.fit(xTrain, yTrain, {
      epochs: totalEpochs,
      batchSize: 16,
      validationData: [xVal, yVal],
      shuffle: true,
      callbacks: {
        onEpochEnd: (epoch, logs) => {
          if (onEpochEnd) onEpochEnd(epoch + 1, totalEpochs, logs);
        }
      }
    });

    xTrain.dispose();
    yTrain.dispose();
    xVal.dispose();
    yVal.dispose();

    const h = history.history;
    const accKey = h.val_acc ? 'val_acc' : 'val_accuracy';
    const valAcc  = h[accKey][h[accKey].length - 1];
    const valLoss = h.val_loss[h.val_loss.length - 1];

    return { valAccuracy: valAcc, valLoss: valLoss };
  }

  async function saveModel() {
    if (!model) throw new Error('No model to save');
    await model.save(MODEL_KEY);
    localStorage.setItem('posture-norm-stats', JSON.stringify({
      mean: Array.from(normStats.mean),
      std: Array.from(normStats.std)
    }));
    localStorage.setItem('posture-model-version', String(MODEL_VERSION));
  }

  async function loadModel() {
    try {
      // Version gate — reject models trained with a different feature set
      const savedVersion = parseInt(localStorage.getItem('posture-model-version') || '0', 10);
      if (savedVersion !== MODEL_VERSION) return false;

      model = await tf.loadLayersModel(MODEL_KEY);
      model.compile({
        optimizer: tf.train.adam(0.001),
        loss: 'binaryCrossentropy',
        metrics: ['accuracy']
      });
      const statsStr = localStorage.getItem('posture-norm-stats');
      if (statsStr) {
        const parsed = JSON.parse(statsStr);
        normStats = {
          mean: new Float32Array(parsed.mean),
          std: new Float32Array(parsed.std)
        };
      }
      return true;
    } catch {
      model = null;
      normStats = null;
      return false;
    }
  }

  // Predict from live MediaPipe landmarks (33-element array of objects).
  function predict(landmarks) {
    if (!model || !normStats) return null;
    const features = PoseDetector.extractPostureFeatures(landmarks);
    if (!features) return null;
    const normalized = normalize(features, normStats);
    const input  = tf.tensor2d([Array.from(normalized)]);
    const output = model.predict(input);
    const prob   = output.dataSync()[0];
    input.dispose();
    output.dispose();
    return prob; // probability of good posture
  }

  function isLoaded() {
    return model !== null && normStats !== null;
  }

  return { trainModel, saveModel, loadModel, predict, isLoaded };
})();
