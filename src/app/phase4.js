// Phase 4: AI Reporter の UI。e5 substrate で入力を符号化し、記憶→概念化→
// レポート→忘却→永続化 を統合。使うほど各ユーザーで育つ（ローカル保存）。

import { selectBackend } from '../runtime/backend.js';
import { loadE5 } from '../represent/e5.js';
import { AIReporter } from '../reporter/reporter.js';
import * as store from '../store/blobStore.js';

const SAMPLES = [
  '今日は会議が長かった', '資料作成に追われた', '上司に報告した',
  '朝ジョギングをした', 'ジムで筋トレした', '夜にウォーキングした',
  'ラーメンを食べた', 'カレーを作った', '寿司屋に行った',
];

const $ = (id) => document.getElementById(id);
const logEl = $('log');
const log = (m) => { logEl.textContent += m + '\n'; logEl.scrollTop = logEl.scrollHeight; };

let S = { backend: null, e5: null, rep: null, questions: [] };

async function loadModel() {
  $('btnLoad').disabled = true;
  log('AI Reporter を起動中…（e5 substrate 約31MB）');
  S.e5 = await loadE5('./models/substrate-e5-small-ja/');
  const enc = { dim: S.e5.H, encode: (t) => S.e5.encode(t, 'passage') };
  S.rep = new AIReporter(S.backend, enc, {});
  // 前回の記憶があれば復元（ユーザーごとに育つ）
  try { const r = await store.load('reporter-memory'); S.rep.loadFabric(r.arrayBuffer); log(`前回の記憶を復元（${S.rep.fab.items.length}件）。`); }
  catch { log('新規スタート。'); }
  log('準備完了。エントリを記録するか「サンプル投入」を押してください。');
  ['btnAdd', 'btnSample', 'btnConsolidate', 'btnForget', 'btnReport', 'btnSave'].forEach((id) => { $(id).disabled = false; });
}

function ingest(text) {
  const r = S.rep.ingest(text);
  let line = `📝 記録: 「${text}」 (新規性 ${r.novelty.toFixed(2)})`;
  if (r.question) { S.questions.push(r.question.prompt); line += '  ❓'; }
  log(line);
}

function addEntry() {
  const t = $('entry').value.trim(); if (!t) return;
  ingest(t); $('entry').value = '';
}

function sample() {
  log('— サンプル投入 —');
  for (const s of SAMPLES) ingest(s);
}

function consolidate() {
  const r = S.rep.consolidate();
  log(`🧠 概念化: テーマを ${r.concepts} 個 抽出（${r.clustered}件を統合）`);
}

function forget() {
  for (let i = 0; i < 8; i++) S.rep.passTime(5);
  log('⏳ 時間が経過（平凡な出来事は薄れ、テーマは残る）');
}

function report() {
  const R = S.rep.report();
  log('\n════════ 📊 レポート ════════');
  log(`▼ あなたのテーマ（${R.themes.length}）`);
  if (R.themes.length === 0) log('  （まだ概念化していません。「概念化」を押してください）');
  for (const t of R.themes) log(`  ● ${t.text}  ［${t.count}件・定着度${t.retention.toFixed(2)}］`);
  log(`▼ 注目の出来事（${R.notable.length}）`);
  for (const n of R.notable) log(`  ★ ${n.text}  (注目度${n.salience.toFixed(2)})`);
  log(`▼ AI からの質問（${S.questions.length}）`);
  for (const q of S.questions.slice(-3)) log(`  ❓ ${q}`);
  log(`（記録 ${R.stats.total}件 / テーマ ${R.stats.concepts} / 生存エントリ ${R.stats.entries}）`);
  log('═══════════════════════════\n');
}

async function save() {
  const r = await store.save('reporter-memory', S.rep.serialize());
  log(`💾 保存（${r.store}, ${S.rep.fab.items.length}件）。次回起動時に自動で続きから。`);
}

async function boot() {
  ['btnAdd', 'btnSample', 'btnConsolidate', 'btnForget', 'btnReport', 'btnSave'].forEach((id) => { $(id).disabled = true; });
  const { backend } = await selectBackend({});
  S.backend = backend;
  log('AI Reporter — あなたの記録からテーマ・注目・気づきを返す（完全ローカル）。');
  $('btnLoad').onclick = () => loadModel().catch((e) => log('エラー: ' + e.message));
  $('btnAdd').onclick = () => { try { addEntry(); } catch (e) { log('エラー: ' + e.message); } };
  $('btnSample').onclick = () => { try { sample(); } catch (e) { log('エラー: ' + e.message); } };
  $('btnConsolidate').onclick = () => { try { consolidate(); } catch (e) { log('エラー: ' + e.message); } };
  $('btnForget').onclick = () => { try { forget(); } catch (e) { log('エラー: ' + e.message); } };
  $('btnReport').onclick = () => { try { report(); } catch (e) { log('エラー: ' + e.message); } };
  $('btnSave').onclick = () => save().catch((e) => log('エラー: ' + e.message));
  if ('serviceWorker' in navigator) navigator.serviceWorker.register('./sw.js').catch(() => {});
}
boot();
