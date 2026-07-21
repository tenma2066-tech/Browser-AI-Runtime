// Phase 1 検証 UI の配線。Encoder（自作エンコーダ）と MemoryFabric を組み合わせ、
// オンライン学習 → 記銘 → 意味的想起 → 忘却 → 永続化 を、ボタン操作で観測する。

import { selectBackend } from '../runtime/backend.js';
import { Encoder } from '../represent/encoder.js';
import { MemoryFabric } from '../memory/fabric.js';
import * as store from '../store/blobStore.js';

// 想起と忘却を見せるための小さなコーパス。
// 「雨」クラスタは高可塑(plasticity 0.9=忘れやすい)、「動物」クラスタは
// 低可塑(0.1=知識として残る) に設定し、忘却の差を観測できるようにする。
const CORPUS = [
  { text: '今日は雨が降っている', plasticity: 0.9 },
  { text: '傘を持って出かける', plasticity: 0.9 },
  { text: '天気予報では雨のち曇り', plasticity: 0.9 },
  { text: '雨雲が空を覆っている', plasticity: 0.9 },
  { text: '犬が公園を走っている', plasticity: 0.1 },
  { text: '猫が窓辺で眠っている', plasticity: 0.1 },
  { text: '動物園で象を見た', plasticity: 0.1 },
  { text: '犬と猫はどちらも動物だ', plasticity: 0.1 },
];

const $ = (id) => document.getElementById(id);
const logEl = $('log');
function log(msg) {
  logEl.textContent += `[${new Date().toLocaleTimeString()}] ${msg}\n`;
  logEl.scrollTop = logEl.scrollHeight;
}

const D = 64;
let state = { backend: null, enc: null, fab: null, lastRecall: [] };

function probeEnv() {
  const standalone = window.matchMedia('(display-mode: standalone)').matches || navigator.standalone === true;
  const rows = [
    ['standalone PWA', standalone ? 'はい' : 'いいえ'],
    ['navigator.gpu', navigator.gpu ? 'あり' : 'なし'],
    ['OPFS', navigator.storage && navigator.storage.getDirectory ? 'あり' : 'なし'],
  ];
  $('env').innerHTML = rows.map(([k, v]) => `<div><span class="k">${k}</span><span class="v">${v}</span></div>`).join('');
}

function freshModel() {
  state.enc = new Encoder({ D, window: 2, lr: 0.1 });
  state.fab = new MemoryFabric(state.backend, { D, lambda: 0.05, reinforceGain: 0.25 });
  state.lastRecall = [];
}

// エンコーダをコーパスで数エポック学習。
// 注: tiny corpus では char-CBOW を回しすぎると表現が崩壊し想起が劣化するため、
// 1回あたり控えめ（40ep）にする。実測での最適点（docs/phase-1-results.md 参照）。
async function train(epochs = 40) {
  if (!state.enc) freshModel();
  let first = null, last = null;
  const t0 = performance.now();
  for (let e = 0; e < epochs; e++) {
    let loss = 0, c = 0;
    for (const s of CORPUS) { const r = state.enc.learn(s.text); loss += r.loss * r.centers; c += r.centers; }
    const avg = c ? loss / c : 0;
    if (first === null) first = avg;
    last = avg;
  }
  log(`エンコーダ学習 ${epochs}ep: loss ${first.toFixed(4)} -> ${last.toFixed(4)}  ` +
      `(${(performance.now() - t0).toFixed(0)}ms, 語彙 ${state.enc.size})`);
}

// コーパスを記銘（encode → write）。可塑性はクラスタごとに設定。
function remember() {
  if (!state.enc) freshModel();
  // 二重記銘を避けるため一旦クリア
  state.fab = new MemoryFabric(state.backend, { D, lambda: 0.05, reinforceGain: 0.25 });
  for (const s of CORPUS) {
    const v = state.enc.encode(s.text);
    state.fab.write(v, { text: s.text, plasticity: s.plasticity, provenance: s.plasticity < 0.5 ? 'derived' : 'perception' });
  }
  const st = state.fab.stats();
  log(`記銘 ${st.count} 件（高可塑=雨クラスタ / 低可塑=動物クラスタ）`);
}

async function recall() {
  if (!state.fab || state.fab.items.length === 0) { log('先に「記銘」してください'); return; }
  const q = $('query').value.trim();
  if (!q) { log('クエリを入力してください'); return; }
  const v = state.enc.encode(q);
  const top = await state.fab.recall(v, 5);
  state.lastRecall = top;
  log(`想起: "${q}"`);
  for (const s of top) {
    log(`  score=${s.score.toFixed(3)} (sim=${s.sim.toFixed(3)} × ret=${s.item.retention.toFixed(3)})  ${s.item.text}`);
  }
}

function decay() {
  if (!state.fab) return;
  state.fab.decay(10);
  const st = state.fab.stats();
  log(`忘却 decay(10): ret 平均=${st.avgRetention.toFixed(3)} 最小=${st.minRetention.toFixed(3)} 最大=${st.maxRetention.toFixed(3)}`);
  // クラスタ別の retention を見せる
  const rain = state.fab.items.filter((x) => x.plasticity >= 0.5);
  const animal = state.fab.items.filter((x) => x.plasticity < 0.5);
  const avg = (a) => a.reduce((s, x) => s + x.retention, 0) / (a.length || 1);
  log(`  高可塑(雨)ret平均=${avg(rain).toFixed(3)}  低可塑(動物)ret平均=${avg(animal).toFixed(3)}`);
}

function reinforceTop() {
  if (!state.lastRecall.length) { log('先に「想起」してください'); return; }
  const it = state.lastRecall[0].item;
  state.fab.reinforce(it.id);
  log(`強化: "${it.text}" ret=${it.retention.toFixed(3)}`);
}

async function save() {
  if (!state.enc || !state.fab) { log('保存対象なし'); return; }
  const r1 = await store.save('encoder', state.enc.serialize());
  const r2 = await store.save('memory', state.fab.serialize());
  log(`保存: encoder(${r1.store}, ${r1.bytes}B) / memory(${r2.store}, ${r2.bytes}B)`);
}

async function restore() {
  try {
    const e = await store.load('encoder');
    const m = await store.load('memory');
    state.enc = Encoder.deserialize(e.arrayBuffer);
    state.fab = MemoryFabric.deserialize(m.arrayBuffer, state.backend);
    log(`復元: encoder(${e.store}, 語彙${state.enc.size}) / memory(${m.store}, ${state.fab.items.length}件)`);
  } catch (err) { log('復元失敗: ' + err.message); }
}

// bit 一致の自己テスト（encoder と memory の両方）
async function selfTest() {
  if (!state.enc) { await train(50); }
  if (!state.fab || !state.fab.items.length) remember();
  const check = (buf) => new Uint8Array(buf);
  // encoder
  const eB = check(state.enc.serialize());
  const eR = check(Encoder.deserialize(state.enc.serialize()).serialize());
  const ePass = eB.length === eR.length && eB.every((v, i) => v === eR[i]);
  // memory
  const mB = check(state.fab.serialize());
  const mR = check(MemoryFabric.deserialize(state.fab.serialize(), state.backend).serialize());
  const mPass = mB.length === mR.length && mB.every((v, i) => v === mR[i]);
  // 実ストア往復も検証
  await save();
  const e2 = await store.load('encoder');
  const m2 = await store.load('memory');
  const eStore = check(Encoder.deserialize(e2.arrayBuffer).serialize());
  const mStore = check(MemoryFabric.deserialize(m2.arrayBuffer, state.backend).serialize());
  const eStorePass = eB.length === eStore.length && eB.every((v, i) => v === eStore[i]);
  const mStorePass = mB.length === mStore.length && mB.every((v, i) => v === mStore[i]);
  log(`自己テスト(bit一致): encoder=${ePass && eStorePass ? 'PASS ✅' : 'FAIL ❌'}  memory=${mPass && mStorePass ? 'PASS ✅' : 'FAIL ❌'}  (store=${e2.store}/${m2.store})`);
}

async function boot() {
  probeEnv();
  const { backend, reason } = await selectBackend({});
  state.backend = backend;
  log(`backend = ${backend.kind} (${reason})`);
  freshModel();
  $('query').value = '雨の日は傘が必要だ';
  log('手順: 学習 → 記銘 → 想起 → 忘却 → （強化/保存/復元/自己テスト）');

  $('btnTrain').onclick = () => train(40).catch((e) => log('エラー: ' + e.message));
  $('btnRemember').onclick = () => { remember(); };
  $('btnRecall').onclick = () => recall().catch((e) => log('エラー: ' + e.message));
  $('btnDecay').onclick = () => decay();
  $('btnReinforce').onclick = () => reinforceTop();
  $('btnSave').onclick = () => save().catch((e) => log('エラー: ' + e.message));
  $('btnRestore').onclick = () => restore().catch((e) => log('エラー: ' + e.message));
  $('btnSelfTest').onclick = () => selfTest().catch((e) => log('エラー: ' + e.message));
  $('btnReset').onclick = () => { logEl.textContent = ''; freshModel(); probeEnv(); log('リセット'); };

  if ('serviceWorker' in navigator) {
    try { await navigator.serviceWorker.register('./sw.js'); } catch {}
  }
}

boot().catch((e) => log('起動エラー: ' + e.message));
