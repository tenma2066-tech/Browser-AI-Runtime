// パラメータの永続化と復元。受け入れ条件#3（リロード跨ぎで bit 一致復元）を
// 満たすため、パラメータ群を単一 ArrayBuffer に直列化してバイト列として保存する。
//
// iOS Safari の未知数を握りつぶさないため、OPFS を第一候補で試し、失敗したら
// IndexedDB にフォールバックし、「どちらを使ったか」を必ず返す。
//   - OPFS の createWritable は Safari で未対応/不安定なことがある（既知の罠）。
//     その場合は例外を捕捉して IndexedDB に落ちる。

import { H } from './trainStep.js';

const MAGIC = 0x42414931; // "BAI1"
const FILE_NAME = 'phase0-params.bin';
const IDB_NAME = 'bai-phase0';
const IDB_STORE = 'params';
const IDB_KEY = 'current';

// --- 直列化 ---------------------------------------------------------------
// レイアウト: [MAGIC u32][H u32][step u32] を先頭に置き、続けて
//   W1(2H) b1(H) W2(H) b2(1) の f32 を並べる。長さは H から一意に決まる。
export function serialize(p) {
  const floatCount = 2 * p.H + p.H + p.H + 1;
  const buf = new ArrayBuffer(12 + floatCount * 4);
  const head = new Uint32Array(buf, 0, 3);
  head[0] = MAGIC; head[1] = p.H; head[2] = p.step;
  const f = new Float32Array(buf, 12, floatCount);
  let o = 0;
  f.set(p.W1, o); o += p.W1.length;
  f.set(p.b1, o); o += p.b1.length;
  f.set(p.W2, o); o += p.W2.length;
  f.set(p.b2, o); o += p.b2.length;
  return buf;
}

export function deserialize(buf) {
  const head = new Uint32Array(buf, 0, 3);
  if (head[0] !== MAGIC) throw new Error('MAGIC 不一致（壊れた/別形式のデータ）');
  const hh = head[1];
  const step = head[2];
  const f = new Float32Array(buf, 12, 2 * hh + hh + hh + 1);
  let o = 0;
  const W1 = f.slice(o, o += 2 * hh);
  const b1 = f.slice(o, o += hh);
  const W2 = f.slice(o, o += hh);
  const b2 = f.slice(o, o += 1);
  return { H: hh, step, W1, b1, W2, b2 };
}

// --- OPFS ------------------------------------------------------------------
async function saveOPFS(buf) {
  if (!navigator.storage || !navigator.storage.getDirectory) throw new Error('OPFS なし');
  const root = await navigator.storage.getDirectory();
  const fh = await root.getFileHandle(FILE_NAME, { create: true });
  if (!fh.createWritable) throw new Error('createWritable 未対応（Safari の既知制約）');
  const w = await fh.createWritable();
  await w.write(buf);
  await w.close();
}

async function loadOPFS() {
  if (!navigator.storage || !navigator.storage.getDirectory) throw new Error('OPFS なし');
  const root = await navigator.storage.getDirectory();
  const fh = await root.getFileHandle(FILE_NAME, { create: false });
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

async function saveIDB(buf) {
  const db = await idbOpen();
  await new Promise((resolve, reject) => {
    const tx = db.transaction(IDB_STORE, 'readwrite');
    tx.objectStore(IDB_STORE).put(buf, IDB_KEY);
    tx.oncomplete = resolve;
    tx.onerror = () => reject(tx.error);
  });
  db.close();
}

async function loadIDB() {
  const db = await idbOpen();
  const buf = await new Promise((resolve, reject) => {
    const tx = db.transaction(IDB_STORE, 'readonly');
    const r = tx.objectStore(IDB_STORE).get(IDB_KEY);
    r.onsuccess = () => resolve(r.result);
    r.onerror = () => reject(r.error);
  });
  db.close();
  if (!buf) throw new Error('IndexedDB に保存データなし');
  return buf;
}

// --- 公開 API --------------------------------------------------------------
// 保存: OPFS を試し、ダメなら IndexedDB。使ったストアを返す。
export async function saveParams(p) {
  const buf = serialize(p);
  try {
    await saveOPFS(buf);
    return { store: 'opfs', bytes: buf.byteLength };
  } catch (e) {
    await saveIDB(buf);
    return { store: 'indexeddb', bytes: buf.byteLength, opfsError: e.message };
  }
}

// 復元: OPFS を試し、ダメなら IndexedDB。使ったストアと復元パラメータを返す。
export async function loadParams() {
  try {
    const buf = await loadOPFS();
    return { store: 'opfs', params: deserialize(buf) };
  } catch (eo) {
    const buf = await loadIDB();
    return { store: 'indexeddb', params: deserialize(buf), opfsError: eo.message };
  }
}

// bit 一致の自己テスト: 保存 → 実ストアから読込 → 元バイト列と完全一致か検証。
// 受け入れ条件#3 を直接チェックする。
export async function selfTestRoundTrip(p) {
  const before = new Uint8Array(serialize(p));
  const saved = await saveParams(p);
  const loaded = await loadParams();
  const after = new Uint8Array(serialize(loaded.params));
  if (before.length !== after.length) {
    return { pass: false, reason: 'サイズ不一致', store: loaded.store };
  }
  for (let i = 0; i < before.length; i++) {
    if (before[i] !== after[i]) {
      return { pass: false, reason: `byte[${i}] 不一致`, store: loaded.store };
    }
  }
  return { pass: true, store: loaded.store, saveStore: saved.store, bytes: before.length };
}
