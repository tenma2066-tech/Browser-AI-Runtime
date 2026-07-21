// Phase 2b 検証 UI。凍結 substrate（事前学習済みサブワード埋め込み）を
// /models から読み込み、Representation として Memory Fabric に差し込んで、
// 意味的想起が安定して効くこと（Phase 1 の char エンコーダの崩壊が起きないこと）
// を実証する。

import { selectBackend } from '../runtime/backend.js';
import { MemoryFabric } from '../memory/fabric.js';
import { loadSubstrate } from '../represent/substrate.js';

const DOCS = [
  '猫が窓辺で眠っている', '犬が公園を走っている', '動物園で象やライオンを見た',
  '東京は日本の首都だ', '大阪でたこ焼きを食べた', '新幹線で名古屋を通過した',
  '今日は雨が降っている', '傘を持って出かけた', '天気予報は晴れのち曇り',
  'カレーライスを作って食べた', '寿司屋で寿司を注文した', '朝はパンとコーヒーを飲む',
];
const PAIRS = [['猫', '犬'], ['東京', '大阪'], ['車', '電車'], ['雨', '傘'], ['猫', '電車'], ['雨', '東京']];

const $ = (id) => document.getElementById(id);
const logEl = $('log');
const log = (m) => { logEl.textContent += `[${new Date().toLocaleTimeString()}] ${m}\n`; logEl.scrollTop = logEl.scrollHeight; };

let S = { backend: null, sub: null, fab: null };

function probeEnv() {
  const standalone = window.matchMedia('(display-mode: standalone)').matches || navigator.standalone === true;
  $('env').innerHTML = [
    ['standalone PWA', standalone ? 'はい' : 'いいえ'],
    ['navigator.gpu', navigator.gpu ? 'あり' : 'なし'],
  ].map(([k, v]) => `<div><span class="k">${k}</span><span class="v">${v}</span></div>`).join('');
}

async function loadModel() {
  log('substrate を読み込み中…（約7MB）');
  const t0 = performance.now();
  S.sub = await loadSubstrate('./models/substrate-deberta-tiny-ja/');
  const dt = performance.now() - t0;
  log(`substrate 読込完了: 語彙 ${S.sub.pieces.length} / dim ${S.sub.dim} / ${dt.toFixed(0)}ms（凍結）`);
  // 記憶を substrate で符号化して記銘
  S.fab = new MemoryFabric(S.backend, { D: S.sub.dim, lambda: 0.05 });
  for (const d of DOCS) S.fab.write(S.sub.encode(d), { text: d, provenance: 'perception' });
  log(`${DOCS.length} 文を substrate 空間で記銘（dim ${S.sub.dim}）`);
  $('btnRecall').disabled = false;
  $('btnPairs').disabled = false;
}

async function recall() {
  const q = $('query').value.trim();
  if (!q || !S.fab) return;
  const top = await S.fab.recall(S.sub.encode(q), 4);
  const toks = S.sub.tokenize(q).map((id) => S.sub.pieces[id]).join(' | ');
  log(`想起: "${q}"  [${toks}]`);
  top.forEach((s) => log(`  sim=${s.sim.toFixed(3)}  ${s.item.text}`));
}

function showPairs() {
  log('語彙内 cosine（関連ペアが高いか）:');
  for (const [a, b] of PAIRS) {
    const va = S.sub.encode(a), vb = S.sub.encode(b);
    let d = 0; for (let i = 0; i < va.length; i++) d += va[i] * vb[i];
    log(`  ${a}-${b}: ${d.toFixed(3)}`);
  }
}

async function boot() {
  probeEnv();
  $('btnRecall').disabled = true;
  $('btnPairs').disabled = true;
  const { backend, reason } = await selectBackend({});
  S.backend = backend;
  log(`backend = ${backend.kind} (${reason})`);
  $('query').value = '美味しい食事';
  log('「substrate 読込」を押すと、事前学習済み埋め込みで意味的想起を試せます。');

  $('btnLoad').onclick = () => loadModel().catch((e) => log('エラー: ' + e.message));
  $('btnRecall').onclick = () => recall().catch((e) => log('エラー: ' + e.message));
  $('btnPairs').onclick = () => showPairs();
  $('btnReset').onclick = () => { logEl.textContent = ''; probeEnv(); log('リセット'); };

  if ('serviceWorker' in navigator) { try { await navigator.serviceWorker.register('./sw.js'); } catch {} }
}

boot().catch((e) => log('起動エラー: ' + e.message));
