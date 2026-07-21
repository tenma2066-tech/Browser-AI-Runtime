// 汎用 keyed 永続化。名前付き ArrayBuffer を OPFS 優先で保存し、失敗したら
// IndexedDB にフォールバックする。Phase 0 で OPFS の実機動作を確認済みなので、
// 通常経路は OPFS になる想定。
//
// Encoder / MemoryFabric はそれぞれ serialize/deserialize を持ち、本モジュールに
// 別キー（'encoder', 'memory' など）で保存する。

const IDB_NAME = 'bai-store';
const IDB_STORE = 'blobs';

function fileNameFor(key) {
  return `bai-${key.replace(/[^a-zA-Z0-9_-]/g, '_')}.bin`;
}

// --- OPFS ------------------------------------------------------------------
async function saveOPFS(key, buf) {
  if (!navigator.storage || !navigator.storage.getDirectory) throw new Error('OPFS なし');
  const root = await navigator.storage.getDirectory();
  const fh = await root.getFileHandle(fileNameFor(key), { create: true });
  if (!fh.createWritable) throw new Error('createWritable 未対応');
  const w = await fh.createWritable();
  await w.write(buf);
  await w.close();
}

async function loadOPFS(key) {
  if (!navigator.storage || !navigator.storage.getDirectory) throw new Error('OPFS なし');
  const root = await navigator.storage.getDirectory();
  const fh = await root.getFileHandle(fileNameFor(key), { create: false });
  const file = await fh.getFile();
  return await file.arrayBuffer();
}

// --- IndexedDB -------------------------------------------------------------
function idbOpen() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(IDB_NAME, 1);
    req.onupgradeneeded = () => req.result.createObjectStore(IDB_STORE);
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

async function saveIDB(key, buf) {
  const db = await idbOpen();
  await new Promise((resolve, reject) => {
    const tx = db.transaction(IDB_STORE, 'readwrite');
    tx.objectStore(IDB_STORE).put(buf, key);
    tx.oncomplete = resolve;
    tx.onerror = () => reject(tx.error);
  });
  db.close();
}

async function loadIDB(key) {
  const db = await idbOpen();
  const buf = await new Promise((resolve, reject) => {
    const tx = db.transaction(IDB_STORE, 'readonly');
    const r = tx.objectStore(IDB_STORE).get(key);
    r.onsuccess = () => resolve(r.result);
    r.onerror = () => reject(r.error);
  });
  db.close();
  if (!buf) throw new Error(`キー '${key}' の保存データなし`);
  return buf;
}

// --- 公開 API --------------------------------------------------------------
export async function save(key, buf) {
  try {
    await saveOPFS(key, buf);
    return { store: 'opfs', bytes: buf.byteLength };
  } catch (e) {
    await saveIDB(key, buf);
    return { store: 'indexeddb', bytes: buf.byteLength, opfsError: e.message };
  }
}

export async function load(key) {
  try {
    return { arrayBuffer: await loadOPFS(key), store: 'opfs' };
  } catch (eo) {
    return { arrayBuffer: await loadIDB(key), store: 'indexeddb', opfsError: eo.message };
  }
}
