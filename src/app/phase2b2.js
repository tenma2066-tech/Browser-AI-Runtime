// Phase 2b-2 検証 UI。自前実装した e5-small forward（12層 BERT）を substrate に
// 使い、文脈化された文ベクトルで抽象クエリの意味的想起が効くことを実証する。

import { selectBackend } from '../runtime/backend.js';
import { MemoryFabric } from '../memory/fabric.js';
import { loadE5 } from '../represent/e5.js';

const DOCS = [
  '猫が窓辺で眠っている', '犬が公園を走っている', '動物園で象やライオンを見た',
  '東京は日本の首都だ', '大阪でたこ焼きを食べた', '新幹線で名古屋を通過した',
  '今日は雨が降っている', '傘を持って出かけた', '天気予報は晴れのち曇り',
  'カレーライスを作って食べた', '寿司屋で寿司を注文した', '朝はパンとコーヒーを飲む',
];

const $ = (id) => document.getElementById(id);
const logEl = $('log');
const log = (m) => { logEl.textContent += `[${new Date().toLocaleTimeString()}] ${m}\n`; logEl.scrollTop = logEl.scrollHeight; };

let S = { backend: null, e5: null, fab: null };

function probeEnv() {
  const standalone = window.matchMedia('(display-mode: standalone)').matches || navigator.standalone === true;
  $('env').innerHTML = [
    ['standalone PWA', standalone ? 'はい' : 'いいえ'],
    ['navigator.gpu', navigator.gpu ? 'あり' : 'なし'],
  ].map(([k, v]) => `<div><span class="k">${k}</span><span class="v">${v}</span></div>`).join('');
}

async function loadModel() {
  $('btnLoad').disabled = true;
  log('e5-small substrate を読み込み中…（約31MB・自前forward）');
  const t0 = performance.now();
  S.e5 = await loadE5('./models/substrate-e5-small-ja/');
  log(`読込完了: 語彙 ${S.e5.tk.pieces.length} / dim ${S.e5.H} / ${S.e5.L}層 / ${(performance.now() - t0).toFixed(0)}ms`);
  log(`${DOCS.length} 文を forward で符号化中…（1文あたり時間に注目）`);
  S.fab = new MemoryFabric(S.backend, { D: S.e5.H, lambda: 0.05 });
  const t1 = performance.now();
  for (const d of DOCS) S.fab.write(S.e5.encode(d, 'passage'), { text: d });
  log(`符号化完了: ${((performance.now() - t1) / DOCS.length).toFixed(0)}ms/文`);
  $('btnRecall').disabled = false;
}

async function recall() {
  const q = $('query').value.trim();
  if (!q || !S.fab) return;
  const t0 = performance.now();
  const qv = S.e5.encode(q, 'query');
  const top = await S.fab.recall(qv, 4);
  log(`想起: "${q}"  (encode ${(performance.now() - t0).toFixed(0)}ms)`);
  top.forEach((s) => log(`  sim=${s.sim.toFixed(3)}  ${s.item.text}`));
}

async function boot() {
  probeEnv();
  $('btnRecall').disabled = true;
  const { backend, reason } = await selectBackend({});
  S.backend = backend;
  log(`backend = ${backend.kind} (${reason})`);
  $('query').value = 'おなかが空いた';
  log('「e5 読込」→ 自前 forward で文を符号化 → 抽象クエリで意味的想起を試せます。');
  $('btnLoad').onclick = () => loadModel().catch((e) => log('エラー: ' + e.message));
  $('btnRecall').onclick = () => recall().catch((e) => log('エラー: ' + e.message));
  $('btnReset').onclick = () => { logEl.textContent = ''; probeEnv(); log('リセット'); };
  if ('serviceWorker' in navigator) { try { await navigator.serviceWorker.register('./sw.js'); } catch {} }
}

boot().catch((e) => log('起動エラー: ' + e.message));
