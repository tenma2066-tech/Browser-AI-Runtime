// Phase 0 検証 UI の配線。環境プローブ → backend 選択 → 学習ループ →
// 永続化/復元/自己テストを、ボタン操作で実行しログに出す。

import { selectBackend } from '../runtime/backend.js';
import { initParams, trainStep, X, T, N, H } from './trainStep.js';
import { saveParams, loadParams, selfTestRoundTrip } from './persist.js';

const $ = (id) => document.getElementById(id);
const logEl = $('log');
function log(msg) {
  const line = `[${new Date().toLocaleTimeString()}] ${msg}`;
  logEl.textContent += line + '\n';
  logEl.scrollTop = logEl.scrollHeight;
  // eslint-disable-next-line no-console
  console.log(line);
}

let state = {
  backend: null,
  reason: '',
  params: null,
};

// --- 環境プローブ（受け入れ条件の観測項目）--------------------------------
function probeEnv() {
  const nav = navigator;
  const standalone =
    window.matchMedia('(display-mode: standalone)').matches ||
    nav.standalone === true;
  const rows = [
    ['userAgent', nav.userAgent],
    ['standalone PWA として起動', standalone ? 'はい' : 'いいえ（通常タブ）'],
    ['navigator.gpu (WebGPU)', nav.gpu ? 'あり' : 'なし'],
    ['OPFS (storage.getDirectory)', nav.storage && nav.storage.getDirectory ? 'あり' : 'なし'],
    ['crossOriginIsolated', String(window.crossOriginIsolated)],
    ['SharedArrayBuffer', typeof SharedArrayBuffer !== 'undefined' ? 'あり' : 'なし'],
  ];
  $('env').innerHTML = rows
    .map(([k, v]) => `<div><span class="k">${k}</span><span class="v">${v}</span></div>`)
    .join('');
}

// --- backend 選択 ----------------------------------------------------------
async function setupBackend(forceCpu) {
  if (state.backend) { try { state.backend.dispose(); } catch {} }
  const t0 = performance.now();
  const { backend, reason } = await selectBackend({ forceCpu });
  state.backend = backend;
  state.reason = reason;
  const info = backend.adapterInfo
    ? ` / adapter: ${JSON.stringify(backend.adapterInfo)}`
    : '';
  log(`backend = ${backend.kind}  (${reason})  ${((performance.now() - t0)).toFixed(1)}ms${info}`);
  $('backendKind').textContent = backend.kind + (forceCpu ? '（強制CPU）' : '');
}

// --- 学習 ------------------------------------------------------------------
async function train(steps) {
  if (!state.backend) { log('backend 未初期化'); return; }
  if (!state.params) { state.params = initParams(); log(`新規パラメータ初期化 (H=${H}, N=${N})`); }
  const p = state.params;
  const t0 = performance.now();
  let firstLoss = null, lastLoss = null;
  for (let i = 0; i < steps; i++) {
    const { loss } = await trainStep(state.backend, p);
    if (firstLoss === null) firstLoss = loss;
    lastLoss = loss;
    if (i === 0 || (i + 1) % Math.max(1, Math.floor(steps / 5)) === 0) {
      log(`  step ${p.step}: loss=${loss.toFixed(6)}`);
    }
  }
  const dt = performance.now() - t0;
  log(`学習 ${steps}step 完了: loss ${firstLoss.toFixed(6)} -> ${lastLoss.toFixed(6)}  ` +
      `(${dt.toFixed(1)}ms, ${(dt / steps).toFixed(2)}ms/step)`);
  await showPredictions();
}

async function showPredictions() {
  // 現在のパラメータで XOR 4 パターンを予測して表示（学習できている証拠）
  const { params, backend } = state;
  if (!params) return;
  const rows = [];
  for (let i = 0; i < N; i++) {
    const x = X.slice(i * 2, i * 2 + 2);
    // forward だけ手計算（trainStep と同一式）
    const z1 = await backend.matmul(x, 1, 2, params.W1, 2, params.H);
    let y = 0;
    const a1 = new Float32Array(params.H);
    for (let j = 0; j < params.H; j++) a1[j] = Math.tanh(z1[j] + params.b1[j]);
    const z2 = await backend.matmul(a1, 1, params.H, params.W2, params.H, 1);
    y = 1 / (1 + Math.exp(-(z2[0] + params.b2[0])));
    rows.push(`  (${x[0]},${x[1]}) -> ${y.toFixed(3)}  (正解 ${T[i]})`);
  }
  log('予測:\n' + rows.join('\n'));
}

// --- 永続化 ----------------------------------------------------------------
async function doSave() {
  if (!state.params) { log('保存対象なし'); return; }
  const r = await saveParams(state.params);
  log(`保存: store=${r.store}, ${r.bytes} bytes, step=${state.params.step}` +
      (r.opfsError ? ` (OPFS不可: ${r.opfsError})` : ''));
}

async function doRestore() {
  try {
    const r = await loadParams();
    state.params = r.params;
    log(`復元: store=${r.store}, step=${r.params.step}` +
        (r.opfsError ? ` (OPFS不可: ${r.opfsError})` : ''));
    await showPredictions();
  } catch (e) {
    log('復元失敗: ' + e.message);
  }
}

async function doSelfTest() {
  if (!state.params) { state.params = initParams(); }
  // 学習を少し進めてから round-trip 検証（自明な初期値でなく実データで検証）
  for (let i = 0; i < 50; i++) await trainStep(state.backend, state.params);
  const r = await selfTestRoundTrip(state.params);
  log(`自己テスト(保存→読込→bit一致): ${r.pass ? 'PASS ✅' : 'FAIL ❌ ' + r.reason}` +
      `  store=${r.store}${r.saveStore ? '/save=' + r.saveStore : ''}, ${r.bytes || 0} bytes`);
}

// --- 起動 ------------------------------------------------------------------
async function boot() {
  probeEnv();
  await setupBackend(false);
  log('準備完了。「学習」ボタンで loss が下がるのを確認してください。');

  $('btnTrain').onclick = () => train(500).catch((e) => log('エラー: ' + e.message));
  $('btnSave').onclick = () => doSave().catch((e) => log('エラー: ' + e.message));
  $('btnRestore').onclick = () => doRestore().catch((e) => log('エラー: ' + e.message));
  $('btnSelfTest').onclick = () => doSelfTest().catch((e) => log('エラー: ' + e.message));
  $('btnForceCpu').onclick = () =>
    setupBackend(true).then(() => { state.params = null; log('パラメータをリセットしました'); })
      .catch((e) => log('エラー: ' + e.message));
  $('btnReset').onclick = () => { state.params = null; logEl.textContent = ''; probeEnv(); log('リセット'); };

  // Service Worker 登録（オフライン/PWA 用）
  if ('serviceWorker' in navigator) {
    try {
      await navigator.serviceWorker.register('./sw.js');
      log('Service Worker 登録済み');
    } catch (e) {
      log('Service Worker 登録失敗: ' + e.message);
    }
  }
}

boot().catch((e) => log('起動エラー: ' + e.message));
