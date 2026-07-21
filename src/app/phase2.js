// Phase 2a 検証 UI の配線。凍結エンコーダ → Cognitive Core（次文字予測）→
// 予測誤差(surprise)でオンライン学習し、驚いた瞬間だけ Memory に書き戻す。

import { selectBackend } from '../runtime/backend.js';
import { Encoder } from '../represent/encoder.js';
import { MemoryFabric } from '../memory/fabric.js';
import { CognitiveCore } from '../cognition/core.js';
import * as store from '../store/blobStore.js';

// 学習ストリーム（Phase 1 と同じコーパス）
const CORPUS = [
  '今日は雨が降っている', '傘を持って出かける', '天気予報では雨のち曇り', '雨雲が空を覆っている',
  '犬が公園を走っている', '猫が窓辺で眠っている', '動物園で象を見た', '犬と猫はどちらも動物だ',
];
const FAMILIAR = '今日は雨が降っている';      // 学習済み
const NOVEL = '台風が接近し警報が出た';        // 未知語を多く含む
const NOVEL2 = '猫は傘を持って公園で眠る';     // 既知文字の新しい組み合わせ（双方向ループ用）

const $ = (id) => document.getElementById(id);
const logEl = $('log');
function log(m) { logEl.textContent += `[${new Date().toLocaleTimeString()}] ${m}\n`; logEl.scrollTop = logEl.scrollHeight; }

const D = 64, H = 64;
let S = { backend: null, enc: null, core: null, fab: null, ema: null, seen: new Set(), lastRecall: [] };

function probeEnv() {
  const standalone = window.matchMedia('(display-mode: standalone)').matches || navigator.standalone === true;
  $('env').innerHTML = [
    ['standalone PWA', standalone ? 'はい' : 'いいえ'],
    ['navigator.gpu', navigator.gpu ? 'あり' : 'なし'],
    ['OPFS', navigator.storage && navigator.storage.getDirectory ? 'あり' : 'なし'],
  ].map(([k, v]) => `<div><span class="k">${k}</span><span class="v">${v}</span></div>`).join('');
}

function freshModel() {
  S.enc = new Encoder({ D, window: 2, lr: 0.1 });
  S.core = new CognitiveCore({ D, H, lr: 0.05 });
  S.fab = new MemoryFabric(S.backend, { D, lambda: 0.05 });
  S.ema = null; S.seen = new Set(); S.lastRecall = [];
}

// エンコーダを学習して凍結（Representation を Core の入力特徴にする）
function prepEncoder() {
  for (let e = 0; e < 40; e++) for (const t of CORPUS) S.enc.learn(t);
  log(`エンコーダ準備完了（凍結）: 語彙 ${S.enc.size}`);
}

// ストリームを1パス学習し、驚いた瞬間を Memory に書き戻す
function streamPass(learn = true) {
  let total = 0, steps = 0, writes = 0;
  const factor = 1.3, alpha = 0.05;
  for (const text of CORPUS) {
    S.core.reset();
    const toks = S.enc.tokenizeRead(text);
    const chars = [...text];
    for (let i = 0; i < toks.length - 1; i++) {
      const s = S.core.observe(S.enc, toks[i], toks[i + 1], learn);
      total += s; steps++;
      // EMA 閾値による書き戻し
      S.ema = S.ema === null ? s : (1 - alpha) * S.ema + alpha * s;
      if (learn && S.ema !== null && s > S.ema * factor) {
        const win = chars.slice(Math.max(0, i - 3), i + 2).join('');
        if (win.length >= 2 && !S.seen.has(win)) {
          S.seen.add(win);
          S.fab.write(S.enc.encode(win), { text: win, salience: s, plasticity: 0.5, provenance: 'derived' });
          writes++;
        }
      }
    }
  }
  return { avg: steps ? total / steps : 0, writes };
}

async function streamLearn() {
  if (!S.enc.size || S.enc.size <= 1) prepEncoder();
  let first = null, last = null, totalWrites = 0;
  for (let p = 0; p < 5; p++) {
    const r = streamPass(true);
    if (first === null) first = r.avg;
    last = r.avg; totalWrites += r.writes;
    log(`  pass ${p + 1}: 平均surprise=${r.avg.toFixed(4)}  記銘=${r.writes}`);
  }
  log(`ストリーム学習 5pass: surprise ${first.toFixed(4)} -> ${last.toFixed(4)}  累計記銘=${totalWrites}件`);
}

// 自己評価: 馴染みの文 vs 新規の文（学習なしで平均 surprise を測る）
function selfEval() {
  const measure = (text) => {
    S.core.reset();
    const toks = S.enc.tokenizeRead(text);
    let t = 0, n = 0;
    for (let i = 0; i < toks.length - 1; i++) { t += S.core.observe(S.enc, toks[i], toks[i + 1], false); n++; }
    return n ? t / n : 0;
  };
  const f = measure(FAMILIAR), nv = measure(NOVEL);
  log(`自己評価: 馴染み「${FAMILIAR}」surprise=${f.toFixed(3)} / 新規「${NOVEL}」surprise=${nv.toFixed(3)}`);
  log(`  → 新規の方が ${nv > f ? '高い ✅（驚いている）' : '高くない ❌'}`);
}

// --- Phase 2a 後半: 双方向ループ（記憶が予測を助ける）---------------------
const ctxOf = (chars, i) => chars.slice(Math.max(0, i - 3), i + 1).join('');

// 系列の平均 surprise。useMem=true なら想起した連想(次文字)で予測をブレンド。
async function evalSeq(text, useMem, lamMax = 0.9, gate = 0.6) {
  S.core.reset();
  const chars = [...text];
  const toks = S.enc.tokenizeRead(text);
  let tot = 0, n = 0;
  for (let i = 0; i < toks.length - 1; i++) {
    const probs = S.core.predict(S.enc, toks[i]); // パラメトリック（学習しない）
    let p = probs;
    if (useMem) {
      const top = await S.fab.recall(S.enc.encode(ctxOf(chars, i)), 5);
      const pmem = new Float32Array(probs.length);
      let topSim = 0, wsum = 0;
      for (const it of top) {
        const a = it.item.assoc;
        if (a >= 0 && a < pmem.length) { const w = Math.max(0, it.sim) * it.item.retention; pmem[a] += w; wsum += w; if (it.sim > topSim) topSim = it.sim; }
      }
      if (wsum > 0) {
        for (let k = 0; k < pmem.length; k++) pmem[k] /= wsum;
        const lam = topSim > gate ? lamMax * topSim : 0;
        p = new Float32Array(probs.length);
        for (let k = 0; k < p.length; k++) p[k] = (1 - lam) * probs[k] + lam * pmem[k];
      }
    }
    tot += -Math.log((p[toks[i + 1]] || 1e-12) + 1e-12); n++;
  }
  return n ? tot / n : 0;
}

// 一度「経験」して 文脈→次文字 の連想を記憶に書く（重みは変えない）。
function experience(text) {
  const chars = [...text];
  const toks = S.enc.tokenizeRead(text);
  for (let i = 0; i < toks.length - 1; i++) {
    const c = ctxOf(chars, i);
    S.fab.write(S.enc.encode(c), { text: c, assoc: toks[i + 1], salience: 1, plasticity: 0.3, provenance: 'derived' });
  }
}

async function bidirectional() {
  if (S.core.step === 0) { prepEncoder(); await streamLearn(); }
  const stepBefore = S.core.step;
  // 対照を常にクリーンにするため、この実験で書く記憶は実験後にロールバックする
  // （④は非破壊のプローブ。①で貯まる本物のエピソード記憶には触れない）。
  const base = S.fab.items.length;
  const before = await evalSeq(NOVEL2, false);
  const ctrl = await evalSeq(NOVEL2, true);   // 経験前: memory ON でも助からないはず
  experience(NOVEL2);                          // 一度だけ経験（重み不変）
  const after = await evalSeq(NOVEL2, true);   // 経験後: 想起が予測を助ける
  S.fab.items.length = base;                    // 実験で書いた連想を巻き戻す（再現性のため）
  log(`双方向ループ実験（非破壊プローブ）: "${NOVEL2}"`);
  log(`  memory OFF          : surprise=${before.toFixed(3)}`);
  log(`  memory ON（経験前）  : surprise=${ctrl.toFixed(3)}（未経験なので下がらない）`);
  log(`  memory ON（経験後）  : surprise=${after.toFixed(3)}`);
  log(`  → 経験後に ${(before - after).toFixed(3)} 低下（${((1 - after / before) * 100).toFixed(0)}%）` +
      `。重みは不変（core.step ${stepBefore}→${S.core.step}）= 記憶が認知を助けた`);
}

function showMemory() {
  const items = S.fab.items;
  log(`記憶（驚きで記銘された断片）: ${items.length}件`);
  [...items].sort((a, b) => b.salience - a.salience).slice(0, 8)
    .forEach((it) => log(`  salience=${it.salience.toFixed(2)} ret=${it.retention.toFixed(2)}  "${it.text}"`));
}

async function recall() {
  if (!S.fab.items.length) { log('先にストリーム学習で記憶を作ってください'); return; }
  const q = $('query').value.trim(); if (!q) return;
  const top = await S.fab.recall(S.enc.encode(q), 5);
  log(`想起: "${q}"`);
  top.forEach((s) => log(`  score=${s.score.toFixed(3)} (sim=${s.sim.toFixed(3)}×ret=${s.item.retention.toFixed(3)})  "${s.item.text}"`));
}

async function save() {
  const a = await store.save('p2-encoder', S.enc.serialize());
  const b = await store.save('p2-core', S.core.serialize());
  const c = await store.save('p2-memory', S.fab.serialize());
  log(`保存: encoder(${a.store}) / core(${b.store}, ${b.bytes}B) / memory(${c.store}, ${S.fab.items.length}件)`);
}

async function restore() {
  try {
    S.enc = Encoder.deserialize((await store.load('p2-encoder')).arrayBuffer);
    S.core = CognitiveCore.deserialize((await store.load('p2-core')).arrayBuffer);
    S.fab = MemoryFabric.deserialize((await store.load('p2-memory')).arrayBuffer, S.backend);
    log(`復元: 語彙${S.enc.size} / core.step=${S.core.step} / memory ${S.fab.items.length}件`);
  } catch (e) { log('復元失敗: ' + e.message); }
}

async function selfTest() {
  if (S.core.step === 0) { prepEncoder(); await streamLearn(); }
  const eq = (x, y) => { x = new Uint8Array(x); y = new Uint8Array(y); return x.length === y.length && x.every((v, i) => v === y[i]); };
  const c = eq(S.core.serialize(), CognitiveCore.deserialize(S.core.serialize()).serialize());
  const e = eq(S.enc.serialize(), Encoder.deserialize(S.enc.serialize()).serialize());
  const m = eq(S.fab.serialize(), MemoryFabric.deserialize(S.fab.serialize(), S.backend).serialize());
  await save();
  const c2 = eq(S.core.serialize(), CognitiveCore.deserialize((await store.load('p2-core')).arrayBuffer).serialize());
  log(`自己テスト(bit一致): core=${c && c2 ? 'PASS ✅' : 'FAIL ❌'} encoder=${e ? 'PASS ✅' : 'FAIL ❌'} memory=${m ? 'PASS ✅' : 'FAIL ❌'}`);
}

async function boot() {
  probeEnv();
  const { backend, reason } = await selectBackend({});
  S.backend = backend;
  log(`backend = ${backend.kind} (${reason})`);
  freshModel();
  $('query').value = '雨';
  log('手順: ① ストリーム学習 → ② 自己評価 → ③ 記憶を見る →（想起/保存/復元/自己テスト）');

  $('btnStream').onclick = () => streamLearn().catch((e) => log('エラー: ' + e.message));
  $('btnEval').onclick = () => selfEval();
  $('btnLoop').onclick = () => bidirectional().catch((e) => log('エラー: ' + e.message));
  $('btnMem').onclick = () => showMemory();
  $('btnRecall').onclick = () => recall().catch((e) => log('エラー: ' + e.message));
  $('btnSave').onclick = () => save().catch((e) => log('エラー: ' + e.message));
  $('btnRestore').onclick = () => restore().catch((e) => log('エラー: ' + e.message));
  $('btnSelfTest').onclick = () => selfTest().catch((e) => log('エラー: ' + e.message));
  $('btnReset').onclick = () => { logEl.textContent = ''; freshModel(); probeEnv(); log('リセット'); };

  if ('serviceWorker' in navigator) { try { await navigator.serviceWorker.register('./sw.js'); } catch {} }
}

boot().catch((e) => log('起動エラー: ' + e.message));
