# Phase 0 仕様書 ― 技術検証スパイク

- 前提: `docs/ADR-0001-foundations.md`
- 目的: プロジェクトが物理的に成立するかを、最も細い垂直スライスで実証する。
- 方針: **検証のための最小コードのみ**。7層アーキテクチャ・AI Reporter・
  substrate 読み込み・UI は Phase 0 では作らない。

---

## 1. Phase 0 が通す「細い線」（1本）

```
iPhone 実機 Safari で ホーム画面PWA として起動
  → WebGPU アダプタ取得（失敗したら WASM 経路に自動フォールバック）
  → 極小テンソル（2層MLP）を GPU 上で forward
  → 勾配を1ステップ backward + パラメータ更新
  → 更新後の重みを OPFS（不可なら IndexedDB）へ保存
  → PWA を完全終了 → 再起動 → 重みを復元し、続きから学習
```

学習対象は言語である必要はない。**学習ループが回り、状態が永続化・復元される**
ことだけを証明する。例として XOR や単純な回帰など、正解が自明で loss の低下が
一目で分かる極小タスクを用いる。

---

## 2. 受け入れ条件（Definition of Done）

以下がすべて満たされたら Phase 0 合格とする。実機（iPhone Safari, standalone
PWA）での確認を必須とする。

1. **[GPU取得]** iOS standalone PWA で `navigator.gpu` からアダプタ／デバイスが
   取得できることを実機で確認する。取得できない場合は WASM フォールバックが
   自動発火することを確認する。
2. **[学習]** GPU 上で loss が実際に下がる（学習が回っている証拠がログに出る）。
3. **[永続化]** リロード／PWA 再起動を跨いで、重みが復元される。復元後の重みは
   保存前と bit 一致する（丸め誤差のない完全一致）。
4. **[フォールバック]** WASM/CPU 経路でも 1〜3 が成立する（WebGPU を意図的に
   無効化したときに、同じ検証がフォールバックで通る）。

補助的に記録すること（合否には含めないが、以降の設計判断に使う）:

- WebGPU が iOS PWA で使えたか／フラグや制約はあったか。
- OPFS が iOS PWA で使えたか（不可なら IndexedDB にフォールバックしたか）。
- COOP/COEP・SharedArrayBuffer が GitHub Pages + iOS で成立するか。
- 1 学習ステップの概算所要時間（GPU 経路 / WASM 経路）。

---

## 3. ファイル構成（Phase 0 でのみ作るもの）

```
/index.html                 PWAシェル + 検証UI（ログ表示のみ）
/manifest.webmanifest       standalone 起動用マニフェスト
/sw.js                      Service Worker（オフライン + 必要なら COEP 注入）
/src/runtime/
   backend.ts               Runtime 抽象インターフェース（2実装の契約）
   webgpu.ts                WebGPU 実装（最小）
   wasm.ts                  CPU フォールバック実装（最小）
/src/probe/
   trainStep.ts             2層MLP の forward / backward / update
   persist.ts               OPFS / IndexedDB 保存・復元
/docs/
   ADR-0001-foundations.md  決定記録（本 Phase の前提）
   phase-0-spec.md          本ファイル
```

補足:

- ビルドツールに安易に依存しない方針のため、Phase 0 は原則ブラウザネイティブ
  な ES Modules で動く形を目指す。TypeScript を使う場合も、重い依存を増やさない
  最小構成にとどめる（Phase 0 の間は素の JS で書き、型は後付けする選択肢も残す）。
- GitHub Pages はルート or `/docs` 配信のため、公開パス構成は Phase 0 実装時に
  最終確定する。

---

## 4. Runtime 抽象の契約（Phase 0 版・最小）

WebGPU と WASM を差し替え可能にするための最小インターフェースの方向性。厳密な
シグネチャは実装時に確定する。

```
interface Backend {
  readonly kind: 'webgpu' | 'wasm'
  init(): Promise<void>              // デバイス取得・失敗時は例外
  // 極小 MLP を回すのに必要な最小演算のみ
  forward(weights, input): output    // 予測
  backward(weights, input, target): grads
  update(weights, grads, lr): weights // パラメータ更新
  dispose(): void
}

// 選択ロジック: WebGPU を試し、失敗したら WASM へ自動フォールバック
selectBackend(): Promise<Backend>
```

Phase 0 の目的は「この契約で 2 実装が差し替え可能である」ことを実証すること。
演算セットは MLP 1 個を回す最小限に絞り、汎用テンソルライブラリは作らない。

---

## 5. 明示的な非目標（Phase 0 では作らない）

- 7層アーキテクチャ（Representation / Memory Fabric / Cognitive Core /
  Meta Cognition 等）の実装
- substrate（凍結言語器官）の読み込み
- AI Reporter のUX
- メモリの分類（短期／長期／知識／人格）
- 汎用テンソル／自動微分ライブラリ

これらはすべて Phase 1 以降。Phase 0 は「地盤が持つか」だけを見る。

---

## 6. Phase 0 完了後の判断

- **合格** → Phase 1（Runtime 抽象の一般化 + Representation + Memory Fabric
  最小）へ進む。
- **不合格（iOS で GPU も永続化も安定しない）** → 設計を根本から見直す。CPU
  中心設計への転換、対象環境の再検討、substrate 方針の再検討などを ADR-0002
  として起票する。
