# Phase 0 結果記録

- 実施日: 2026-07-21
- 実機: iPhone / iOS Safari / ホーム画面 standalone PWA
- 対象コミット: `2b03e86`（feat(phase0)）
- 前提: `docs/phase-0-spec.md`

## 判定: 全条件クリア（完全合格）

| 受け入れ条件 | 結果 | 実測 |
|---|---|---|
| #1 iOS standalone PWA で WebGPU 取得 | ✅ | `backend = webgpu (webgpu-ok)` |
| #2 GPU 上で loss が下がる | ✅ | loss 0.249121 → 0.001108（XOR 予測が正解へ収束） |
| #3 リロード跨ぎで bit 一致復元 | ✅ | 自己テスト `PASS ✅ store=opfs`、復元で step 正しく復元 |
| #4 CPU フォールバックでも #1〜#3 成立 | ✅ | `forced-cpu` で loss 低下・自己テスト PASS |

## 主要な発見（設計に効くもの）

### 1. iOS standalone PWA で WebGPU が動く ✅

最大の未知数がポジティブに解決。GPU 前提の設計に進んでよい。ホーム画面追加後の
standalone モードでも WebGPU compute（WGSL matmul）が正しく実行された。

### 2. OPFS の createWritable が iOS Safari で通る ✅

Safari の既知制約（OPFS の書き込みが worker 内の SyncAccessHandle に限られ、
main thread の `createWritable` が使えない）に**引っかからなかった**。保存・復元
とも `store=opfs` で成功。OPFS を本命ストレージにできる。IndexedDB は退避先として
残すが、通常経路にはならない見込み。

### 3. 両バックエンドが数値的に一致 ✅

GPU 経路と CPU 経路で、同一初期値・同一 step の loss が完全一致
（step 500: いずれも `0.015113`）。matmul 契約が両実装で正しく、backend を
差し替えても結果が変わらないことを実証。Runtime 抽象の妥当性が裏取りされた。

### 4. 【最重要】小さい演算では CPU が GPU より約150倍速い

| 経路 | ms/step（2層MLP, H=4, N=4） |
|---|---|
| WebGPU | 約 3.1〜3.6 ms/step |
| CPU（純JS） | 約 0.01〜0.02 ms/step |

原因は演算そのものではなく、**GPU の往復コスト**（matmul ごとにバッファ確保 →
dispatch → 結果を CPU に mapAsync で読み戻す）。この極小ネットでは往復レイテンシが
支配的で、GPU が大きく負ける。

**設計インプット（Phase 1 で必ず反映）:**

- GPU は「大きな計算を、読み戻さずに GPU 上へ留めたまま」実行するときだけ勝つ。
- よって Runtime 抽象を「matmul が結果を CPU に返す」形から、**テンソルを GPU 上に
  常駐させ、複数演算を読み戻さずに連鎖できる**形へ一般化する必要がある。
- 小さい／レイテンシ重視の演算は CPU 経路が優位。backend 選択は「WebGPU か CPU か」
  の二者択一ではなく、**規模に応じた使い分け**の余地がある（Phase 1 で検討）。

### 5. adapter.info は iOS で空 `{}`

端末情報はマスクされ取得できない。GPU の能力判定を adapter.info に依存できない。
機能検出は「実際に小さな処理を試して成否を見る」方式にする。

## 積み残し（合否には影響しない観測項目）

- `crossOriginIsolated` / `SharedArrayBuffer` の実機値は今回のログに未記録。Phase 0
  では不要だったため未確定。将来 WASM マルチスレッド等が必要になった時点で、
  `sw.js` の COEP 注入（既定オフ）を有効化して実機比較する。
- コールドスタート（アプリ完全終了 → 再起動）跨ぎの復元は、OPFS 実書き込みを通る
  自己テストが PASS しているため実質確認済みだが、明示的な kill→再起動テストは
  次回ついでに取る。

## 結論と次アクション

Phase 0 は完全合格。プロジェクトは物理的に成立する。GPU 前提でよく、OPFS を本命
ストレージにできる。Phase 1（Runtime 抽象の一般化 + Representation + Memory Fabric
最小）へ進む。Phase 1 の Runtime 設計は、上記「発見4」を最優先の制約とする
（テンソル常駐・読み戻し最小化）。
