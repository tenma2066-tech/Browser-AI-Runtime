// Phase 2c 検証 UI。SubwordCore（substrate を凍結入力＋凍結出力ヘッドに使う
// 次 subword 予測器）を学習し、予測が substrate の意味空間で行われることを見る。

import { loadSubstrate } from '../represent/substrate.js';
import { SubwordCore } from '../cognition/subwordcore.js';
import * as store from '../store/blobStore.js';

const CORPUS = [
  '今日は雨が降っている', '傘を持って出かける', '天気予報では雨のち曇り', '雨雲が空を覆っている',
  '犬が公園を走っている', '猫が窓辺で眠っている', '動物園で象を見た', '犬と猫はどちらも動物だ',
];

const $ = (id) => document.getElementById(id);
const logEl = $('log');
const log = (m) => { logEl.textContent += `[${new Date().toLocaleTimeString()}] ${m}\n`; logEl.scrollTop = logEl.scrollHeight; };

let S = { sub: null, core: null, seqs: null };

async function loadModel() {
  $('btnLoad').disabled = true;
  log('substrate（静的 deberta, 32k語彙）を読込中…');
  S.sub = await loadSubstrate('./models/substrate-deberta-tiny-ja/');
  S.core = new SubwordCore({ inDim: S.sub.dim, H: 128, lr: 0.05 });
  S.seqs = CORPUS.map((t) => S.sub.tokenize(t));
  log(`読込完了: dim ${S.sub.dim} / 語彙 ${S.sub.pieces.length}。例「${CORPUS[4]}」→ ${S.seqs[4].map((id) => S.sub.pieces[id]).join('|')}`);
  ['btnTrain', 'btnPredict', 'btnSelfTest'].forEach((id) => { $(id).disabled = false; });
}

async function train() {
  const EP = 8;
  const t0 = performance.now();
  let last = 0;
  for (let e = 0; e < EP; e++) {
    let L = 0, n = 0;
    for (const s of S.seqs) { S.core.reset(); for (let i = 0; i < s.length - 1; i++) { L += S.core.observe(S.sub, s[i], s[i + 1]); n++; } }
    last = L / n;
  }
  log(`学習 +${EP}ep（累計 step=${S.core.step}）: surprise=${last.toFixed(3)}  ${((performance.now() - t0) / EP).toFixed(0)}ms/ep（何度か押して下げる）`);
}

function predict() {
  const custom = $('ctx').value.trim();
  const ctxs = custom ? [custom] : ['犬が', '雨が', '猫が窓'];
  for (const ctx of ctxs) {
    const ids = S.sub.tokenize(ctx);
    const top = S.core.predictTopK(S.sub, ids, 6);
    log(`文脈「${ctx}」次subword: ${top.map(([p, v]) => `${p}(${v.toFixed(2)})`).join(' ')}`);
  }
}

async function selfTest() {
  const eq = (a, b) => { a = new Uint8Array(a); b = new Uint8Array(b); return a.length === b.length && a.every((v, i) => v === b[i]); };
  const c = eq(S.core.serialize(), SubwordCore.deserialize(S.core.serialize()).serialize());
  await store.save('p2c-core', S.core.serialize());
  const c2 = eq(S.core.serialize(), SubwordCore.deserialize((await store.load('p2c-core')).arrayBuffer).serialize());
  log(`自己テスト(bit一致): core=${c && c2 ? 'PASS ✅' : 'FAIL ❌'}`);
}

function boot() {
  ['btnTrain', 'btnPredict', 'btnSelfTest'].forEach((id) => { $(id).disabled = true; });
  $('ctx').value = '犬が';
  log('「substrate読込」→「学習」を数回→「次予測」で、意味空間での予測を確認。');
  $('btnLoad').onclick = () => loadModel().catch((e) => log('エラー: ' + e.message));
  $('btnTrain').onclick = () => train().catch((e) => log('エラー: ' + e.message));
  $('btnPredict').onclick = () => { try { predict(); } catch (e) { log('エラー: ' + e.message); } };
  $('btnSelfTest').onclick = () => selfTest().catch((e) => log('エラー: ' + e.message));
  if ('serviceWorker' in navigator) navigator.serviceWorker.register('./sw.js').catch(() => {});
}
boot();
