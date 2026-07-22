// Phase 0/1 の Service Worker。
//
// 方針: **ネットワーク優先（network-first）**。開発中は「PWA 化すると古い版が
// 出る」問題を避けるため、オンラインなら常に最新を取得し、失敗（オフライン）時
// のみキャッシュへフォールバックする。取得できたものはキャッシュも更新する。
//
// 注: 以前は cache-first だったため、ホーム画面 PWA が古いキャッシュを返して
//     いた。network-first に変更し、CACHE 名も上げて旧キャッシュを破棄する。
const CACHE = 'bai-v12';

const ASSETS = [
  './',
  './index.html',
  './phase0.html',
  './phase1.html',
  './phase2.html',
  './phase2b.html',
  './phase2b2.html',
  './phase3.html',
  './phase2c.html',
  './phase3b.html',
  './manifest.webmanifest',
  './icon.svg',
  './src/probe/main.js',
  './src/probe/trainStep.js',
  './src/probe/persist.js',
  './src/runtime/backend.js',
  './src/runtime/webgpu.js',
  './src/runtime/cpu.js',
  './src/represent/encoder.js',
  './src/memory/fabric.js',
  './src/store/blobStore.js',
  './src/app/phase1.js',
  './src/cognition/core.js',
  './src/app/phase2.js',
  './src/represent/substrate.js',
  './src/app/phase2b.js',
  './src/represent/e5.js',
  './src/app/phase2b2.js',
  './src/cognition/meta.js',
  './src/app/phase3.js',
  './src/cognition/subwordcore.js',
  './src/app/phase2c.js',
  './src/app/phase3b.js',
  // 注: models/substrate-*（約7MB）はプリキャッシュしない。オンデマンド fetch で
  // 取得し、network-first の fetch ハンドラが機会的にキャッシュする。
];

self.addEventListener('install', (e) => {
  e.waitUntil(
    caches.open(CACHE).then((c) => c.addAll(ASSETS)).then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', (e) => {
  e.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', (e) => {
  const req = e.request;
  if (req.method !== 'GET') return;

  // network-first: まずネットワーク、失敗時にキャッシュ。
  e.respondWith(
    fetch(req)
      .then((res) => {
        if (res && res.ok && new URL(req.url).origin === self.location.origin) {
          const copy = res.clone();
          caches.open(CACHE).then((c) => c.put(req, copy));
        }
        return res;
      })
      .catch(() => caches.match(req).then((c) => c || caches.match('./index.html')))
  );
});
