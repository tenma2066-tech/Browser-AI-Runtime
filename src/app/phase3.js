// Phase 3a 検証 UI。Cognitive Core（Phase 2a）の上に Meta Cognition を載せ、
// 「surprise を見て学習率を自分で変え、混乱が持続したら質問する」を実証する。

import { Encoder } from '../represent/encoder.js';
import { CognitiveCore } from '../cognition/core.js';
import { MetaCognition } from '../cognition/meta.js';
import * as store from '../store/blobStore.js';

const RAIN = ['今日は雨が降っている', '傘を持って出かける', '天気予報では雨のち曇り', '雨雲が空を覆っている'];
const ANIMAL = ['犬が公園を走っている', '猫が窓辺で眠っている', '動物園で象を見た', '犬と猫はどちらも動物だ'];
const ALL = [...RAIN, ...ANIMAL];

const $ = (id) => document.getElementById(id);
const logEl = $('log');
const log = (m) => { logEl.textContent += `[${new Date().toLocaleTimeString()}] ${m}\n`; logEl.scrollTop = logEl.scrollHeight; };

const D = 64, H = 64;
let S = { enc: null, core: null, meta: null };

function freshModel() {
  S.enc = new Encoder({ D, window: 2, lr: 0.1 });
  for (let e = 0; e < 40; e++) for (const t of ALL) S.enc.learn(t); // 全語彙を知る（凍結）
  S.core = new CognitiveCore({ D, H, lr: 0.05 });
  S.meta = new MetaCognition({ baseLr: 0.05 });
}

function avgSurprise(core, texts) {
  let t = 0, n = 0;
  for (const s of texts) { core.reset(); const tk = S.enc.tokenizeRead(s); for (let i = 0; i < tk.length - 1; i++) { t += core.forward(S.enc, tk[i], tk[i + 1]).surprise; n++; } }
  return t / n;
}

// ① 馴染み(雨)で Core を学習しつつ Meta が基準を作る
function familiar() {
  if (!S.enc) freshModel();
  let factors = [];
  for (let p = 0; p < 6; p++) {
    for (const t of RAIN) {
      S.core.reset(); const tk = S.enc.tokenizeRead(t);
      for (let i = 0; i < tk.length - 1; i++) {
        const { surprise, entropy } = S.core.forward(S.enc, tk[i], tk[i + 1]);
        const c = S.meta.control(surprise, entropy, S.enc.size);
        S.core.applyGradient(c.lr); factors.push(c.lrFactor);
      }
    }
  }
  const fam = factors.reduce((a, b) => a + b, 0) / factors.length;
  log(`① 馴染み(雨)学習: 雨surprise=${avgSurprise(S.core, RAIN).toFixed(3)}  基準emaSurprise=${S.meta.emaSurprise.toFixed(3)}  lrFactor平均=${fam.toFixed(2)}（≈1で安定）`);
}

// ② 新規(動物)注入: 固定lr と Meta制御 を並走。適応速度と質問を観測。
function inject() {
  if (!S.meta || S.meta.emaSurprise === null) { log('先に ① 馴染み学習を'); return; }
  const fixed = CognitiveCore.deserialize(S.core.serialize());
  const metaCore = CognitiveCore.deserialize(S.core.serialize());
  log(`② 新規(動物)注入前 surprise: ${avgSurprise(metaCore, ANIMAL).toFixed(3)}（未学習で高い）`);
  let asks = 0, novelFactors = [];
  const passFixed = (c) => { for (const t of ANIMAL) { c.reset(); const tk = S.enc.tokenizeRead(t); for (let i = 0; i < tk.length - 1; i++) { c.forward(S.enc, tk[i], tk[i + 1]); c.applyGradient(0.05); } } };
  const passMeta = (c) => {
    for (const t of ANIMAL) {
      const chars = [...t]; c.reset(); const tk = S.enc.tokenizeRead(t);
      for (let i = 0; i < tk.length - 1; i++) {
        const { surprise, entropy } = c.forward(S.enc, tk[i], tk[i + 1]);
        const ctl = S.meta.control(surprise, entropy, S.enc.size);
        c.applyGradient(ctl.lr); novelFactors.push(ctl.lrFactor);
        if (ctl.ask) { asks++; const frag = chars.slice(Math.max(0, i - 2), i + 2).join(''); log(`   ❓ 質問: 「${frag}」あたりが分からない（混乱が持続）`); }
      }
    }
  };
  for (let p = 1; p <= 5; p++) {
    passFixed(fixed); passMeta(metaCore);
    log(`  pass${p}: 固定lr=${avgSurprise(fixed, ANIMAL).toFixed(3)}  Meta制御=${avgSurprise(metaCore, ANIMAL).toFixed(3)}`);
  }
  const nf = novelFactors.slice(0, 20).reduce((a, b) => a + b, 0) / 20;
  log(`→ 新規のlrFactor(序盤)=${nf.toFixed(2)}（≫1で加速）／質問発火=${asks}回`);
  log(`→ Meta制御が固定lrより速く surprise を下げていれば「学び方の自己制御」が成立`);
  S.core = metaCore; // Meta 側を採用
}

async function selfTest() {
  if (!S.core) freshModel();
  const eq = (a, b) => { a = new Uint8Array(a); b = new Uint8Array(b); return a.length === b.length && a.every((v, i) => v === b[i]); };
  const c = eq(S.core.serialize(), CognitiveCore.deserialize(S.core.serialize()).serialize());
  const m = eq(S.meta.serialize(), MetaCognition.deserialize(S.meta.serialize()).serialize());
  await store.save('p3-core', S.core.serialize());
  await store.save('p3-meta', S.meta.serialize());
  const c2 = eq(S.core.serialize(), CognitiveCore.deserialize((await store.load('p3-core')).arrayBuffer).serialize());
  const m2 = eq(S.meta.serialize(), MetaCognition.deserialize((await store.load('p3-meta')).arrayBuffer).serialize());
  log(`自己テスト(bit一致): core=${c && c2 ? 'PASS ✅' : 'FAIL ❌'}  meta=${m && m2 ? 'PASS ✅' : 'FAIL ❌'}`);
}

function boot() {
  freshModel();
  log('① 馴染み学習 → ② 新規注入（固定lr vs Meta制御の対照）→ 自己テスト');
  $('btnFam').onclick = () => { try { familiar(); } catch (e) { log('エラー: ' + e.message); } };
  $('btnInject').onclick = () => { try { inject(); } catch (e) { log('エラー: ' + e.message); } };
  $('btnSelfTest').onclick = () => selfTest().catch((e) => log('エラー: ' + e.message));
  $('btnReset').onclick = () => { logEl.textContent = ''; freshModel(); log('リセット'); };
  if ('serviceWorker' in navigator) { navigator.serviceWorker.register('./sw.js').catch(() => {}); }
}
boot();
