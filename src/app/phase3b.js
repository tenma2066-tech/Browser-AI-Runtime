// Phase 3b 検証 UI。substrate で文を符号化して記銘 → consolidate で概念を作り、
// 強い忘却をかけても概念が残る（経験は忘れ概念は残る）を実証する。

import { selectBackend } from '../runtime/backend.js';
import { MemoryFabric } from '../memory/fabric.js';
import { loadSubstrate } from '../represent/substrate.js';
import * as store from '../store/blobStore.js';

// 話題ごとに語彙が反復する現実的なコーパス（実ユーザーデータもこうなる）
const DOCS = [
  '今日は雨が降った', '明日も雨らしい', '雨で試合が中止',
  'カレーを食べた', '寿司を食べた', 'ラーメンを食べた',
  '電車で会社へ行く', 'バスで学校へ行く', '車で駅へ行く',
];

const $ = (id) => document.getElementById(id);
const logEl = $('log');
const log = (m) => { logEl.textContent += `[${new Date().toLocaleTimeString()}] ${m}\n`; logEl.scrollTop = logEl.scrollHeight; };

let S = { backend: null, sub: null, fab: null };

async function loadModel() {
  $('btnLoad').disabled = true;
  log('substrate（静的 deberta 7MB）を読込中…');
  S.sub = await loadSubstrate('./models/substrate-deberta-tiny-ja/');
  S.fab = new MemoryFabric(S.backend, { D: S.sub.dim, lambda: 0.1 });
  for (const t of DOCS) S.fab.write(S.sub.encode(t), { text: t });
  log(`${DOCS.length} 文を記銘（エピソード）。天気/食事/移動 の3話題。`);
  ['btnConsolidate', 'btnRecall', 'btnForget', 'btnSelfTest'].forEach((id) => { $(id).disabled = false; });
}

function consolidate() {
  const r = S.fab.consolidate({ sim: 0.15 });
  log(`概念化 consolidate: 概念 ${r.concepts} 個（${r.clustered}件を統合）`);
  for (const c of S.fab.items.filter((x) => x.provenance === 'concept')) {
    log(`  ★概念[${c.count}件] ${c.text.replace('概念: ', '')}`);
  }
}

async function recall() {
  const q = $('query').value.trim(); if (!q) return;
  const top = await S.fab.recall(S.sub.encode(q), 4);
  log(`想起「${q}」:`);
  top.forEach((s) => log(`  ${s.item.provenance === 'concept' ? '★概念 ' : 'episode'} score=${s.score.toFixed(2)} (ret=${s.item.retention.toFixed(2)})  ${s.item.text.slice(0, 22)}`));
}

async function forget() {
  for (let i = 0; i < 8; i++) S.fab.decay(5); // 強い忘却
  const ep = S.fab.items.filter((x) => x.provenance !== 'concept');
  const co = S.fab.items.filter((x) => x.provenance === 'concept');
  const alive = (a) => a.filter((x) => x.retention > 0.1).length;
  log(`強い忘却 decay: エピソード生存 ${alive(ep)}/${ep.length}（retention>0.1）／概念生存 ${alive(co)}/${co.length}`);
  log('→ 経験（エピソード）は薄れ、概念は残る。もう一度「想起」で概念だけが返るのを確認。');
}

async function selfTest() {
  const eq = (a, b) => { a = new Uint8Array(a); b = new Uint8Array(b); return a.length === b.length && a.every((v, i) => v === b[i]); };
  const m = eq(S.fab.serialize(), MemoryFabric.deserialize(S.fab.serialize(), S.backend).serialize());
  await store.save('p3b-memory', S.fab.serialize());
  const m2 = eq(S.fab.serialize(), MemoryFabric.deserialize((await store.load('p3b-memory')).arrayBuffer, S.backend).serialize());
  log(`自己テスト(bit一致・概念含む): memory=${m && m2 ? 'PASS ✅' : 'FAIL ❌'}`);
}

async function boot() {
  ['btnConsolidate', 'btnRecall', 'btnForget', 'btnSelfTest'].forEach((id) => { $(id).disabled = true; });
  const { backend } = await selectBackend({});
  S.backend = backend;
  $('query').value = 'ご飯を食べたい';
  log('「substrate読込」→「概念化」→「想起」→「忘却」→ 再「想起」で、経験を忘れても概念が残るのを確認。');
  $('btnLoad').onclick = () => loadModel().catch((e) => log('エラー: ' + e.message));
  $('btnConsolidate').onclick = () => { try { consolidate(); } catch (e) { log('エラー: ' + e.message); } };
  $('btnRecall').onclick = () => recall().catch((e) => log('エラー: ' + e.message));
  $('btnForget').onclick = () => forget().catch((e) => log('エラー: ' + e.message));
  $('btnSelfTest').onclick = () => selfTest().catch((e) => log('エラー: ' + e.message));
  if ('serviceWorker' in navigator) navigator.serviceWorker.register('./sw.js').catch(() => {});
}
boot();
