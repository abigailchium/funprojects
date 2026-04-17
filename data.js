const DataStore = (() => {
  const DB_NAME = 'PostureGuardDB';
  const DB_VERSION = 1;
  const STORE_NAME = 'samples';
  let db = null;

  function open() {
    return new Promise((resolve, reject) => {
      if (db) { resolve(db); return; }
      const request = indexedDB.open(DB_NAME, DB_VERSION);
      request.onupgradeneeded = (e) => {
        const database = e.target.result;
        if (!database.objectStoreNames.contains(STORE_NAME)) {
          const store = database.createObjectStore(STORE_NAME, {
            keyPath: 'id',
            autoIncrement: true
          });
          store.createIndex('label', 'label', { unique: false });
        }
      };
      request.onsuccess = (e) => {
        db = e.target.result;
        resolve(db);
      };
      request.onerror = (e) => reject(e.target.error);
    });
  }

  async function addSample(landmarks, label) {
    const database = await open();
    return new Promise((resolve, reject) => {
      const tx = database.transaction(STORE_NAME, 'readwrite');
      const store = tx.objectStore(STORE_NAME);
      const sample = {
        landmarks: Array.from(landmarks),
        label: label, // 1 = good, 0 = bad
        timestamp: Date.now()
      };
      const request = store.add(sample);
      request.onsuccess = () => resolve(request.result);
      request.onerror = (e) => reject(e.target.error);
    });
  }

  async function getAllSamples() {
    const database = await open();
    return new Promise((resolve, reject) => {
      const tx = database.transaction(STORE_NAME, 'readonly');
      const store = tx.objectStore(STORE_NAME);
      const request = store.getAll();
      request.onsuccess = () => resolve(request.result);
      request.onerror = (e) => reject(e.target.error);
    });
  }

  async function getCounts() {
    const samples = await getAllSamples();
    let good = 0, bad = 0;
    for (const s of samples) {
      if (s.label === 1) good++;
      else bad++;
    }
    return { good, bad };
  }

  async function clearAll() {
    const database = await open();
    return new Promise((resolve, reject) => {
      const tx = database.transaction(STORE_NAME, 'readwrite');
      const store = tx.objectStore(STORE_NAME);
      const request = store.clear();
      request.onsuccess = () => resolve();
      request.onerror = (e) => reject(e.target.error);
    });
  }

  return { open, addSample, getAllSamples, getCounts, clearAll };
})();
