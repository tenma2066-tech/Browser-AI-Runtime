// Phase 0 の Service Worker。目的は 2 つ:
//   1) PWA としてオフライン起動できるよう、アプリシェルをキャッシュする。
//   2) （将来）COOP/COEP 注入の実験口を残す。ただし Phase 0 では既定で無効。
//
// 注: SharedArrayBuffer / crossOriginIsolated を得るための COEP 注入（coi 方式）
//     は WebGPU や外部リソース読み込みを壊すことがあり、iOS Safari では効かない
//     ケースもある。Phase 0 では計測のみ行い、注入は行わない。必要になったら
//     ENABLE_COI を true にして挙動を実機比較する。
const ENABLE_COI = false;

const CACHE = 'bai-phase0-v1';
// SW の場所を基準にした相対パス。GitHub Pages のサブパス配信でも動くようにする。
const ASSETS = [
  './',
  './index.html',
  './manifest.webmanifest',
  './icon.svg',
  './src/probe/main.js',
  './src/probe/trainStep.js',
  './src/probe/persist.js',
  './src/runtime/backend.js',
  './src/runtime/webgpu.js',
  './src/runtime/cpu.js',
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

  e.respondWith(
    caches.match(req).then((cached) => {
      const network = fetch(req)
        .then((res) => {
          // 同一オリジンの成功レスポンスは更新キャッシュ
          if (res && res.ok && new URL(req.url).origin === self.location.origin) {
            const copy = res.clone();
            caches.open(CACHE).then((c) => c.put(req, copy));
          }
          return maybeCOI(res);
        })
        .catch(() => cached);
      return cached ? maybeCOI(cached.clone()) : network;
    })
  );
});

// COEP/COOP を注入して cross-origin isolation を得る実験口（既定オフ）。
function maybeCOI(res) {
  if (!ENABLE_COI || !res) return res;
  const headers = new Headers(res.headers);
  headers.set('Cross-Origin-Embedder-Policy', 'require-corp');
  headers.set('Cross-Origin-Opener-Policy', 'same-origin');
  return new Response(res.body, { status: res.status, statusText: res.statusText, headers });
}
