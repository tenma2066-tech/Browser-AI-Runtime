# Browser AI Runtime

HTML / CSS / JavaScript(TypeScript) のみで動く、**完全クライアントサイドの
自己成長型 AI 実行基盤**を研究・実装するプロジェクト。

- 対象は **iPhone / Safari / PWA を最優先**。
- **サーバー禁止・AI API 禁止・クラウド推論禁止**。完全ローカル実行を目指す。
- ChatGPT のコピーでも、既存 LLM のラッパーでもない。Transformer から学びつつ、
  ブラウザ・WebGPU・単一ユーザー・継続学習という条件に最適化した新しい AI
  実行基盤を設計する。
- 最終目標は **Runtime そのもの**の完成。最初のアプリとして **AI Reporter** を予定。

開発環境は iPhone / Claude Code / GitHub / GitHub Pages のみ。

---

## いまのステータス

| Phase | 内容 | 状態 |
|---|---|---|
| **Phase 0** | 技術検証: iOS PWA で GPU 学習ループ + 永続化/復元 | ✅ **実機合格**（2026-07-21） |
| **Phase 1** | Representation（自作エンコーダ）+ Memory Fabric 最小 | 🟡 実装済み・Node検証済み／**iOS実機確認 待ち** |
| Phase 2 | Cognitive Core + substrate 差し込み + 学習ループを閉じる | ⬜ 設計待ち |
| Phase 3 | Meta Cognition（予測誤差ベースの自己制御） | ⬜ |
| Phase 4 | AI Reporter を最初のアプリとして載せる | ⬜ |

**検証ページ**（GitHub Pages）:
- Phase 0: <https://tenma2066-tech.github.io/Browser-AI-Runtime/>
- Phase 1: <https://tenma2066-tech.github.io/Browser-AI-Runtime/phase1.html>

---

## アーキテクチャの考え方

層は「名前」ではなく **入力・出力・保持する状態のデータ契約** で定義する
（詳細は [`docs/ADR-0001-foundations.md`](docs/ADR-0001-foundations.md)）。

```
入力 → [Representation] → ベクトル → [Memory Fabric] 記銘/想起/忘却
                                          ↓↑
                        [Cognitive Core] state_t + input → state_{t+1} + action
                                          ↓↑
                        [Meta Cognition] 予測誤差を監視し 学習率/質問/凍結 を制御
        ────────────── すべて [Runtime]（CPU/WebGPU 差し替え可能）の上で動く ──────────────
```

- **Runtime**: テンソル/カーネル抽象。WebGPU 実装と CPU 実装を同一契約で差し替え。
- **Representation**: 世界を固定次元ベクトルに符号化（全記憶・全推論の共通通貨）。
- **Memory Fabric**: ベクトル＋属性の記憶。短期/長期/知識/人格を別テーブルにせず、
  **可塑性・保持・出所の属性で連続的に区別**。忘却を eviction でなく設計された
  連続プロセスとして扱う。
- **Cognitive Core**: 状態遷移エンジン（予測と行動）。
- **Meta Cognition**: Core の予測誤差を監視するコントローラ。

### substrate（言語器官）の位置づけ

言語の骨格は、**差し替え可能な凍結言語器官**として小型事前学習モデルで与える。
これは「賢い部分」ではなく、トークン⇄ベクトルの低レベル変換をするダム部品。
思考・推論・記憶・自己評価・概念化は、その周囲の**完全可塑の自己成長層**が担う。
制御の主権は Cognitive Core にあり、substrate は Core に駆動される末端の器官。
これにより「LLM のラッパーではない」ことを担保する。

---

## これからの詳細計画

### Phase 1（実装済み・iOS実機確認待ち）

- **Representation**: 自作の軽量エンコーダ（char 単位 CBOW オンライン学習、
  IDF 重み付けで内容語を強調）。
- **Memory Fabric**: `write / recall(類似度×retention) / reinforce(Hebbian) /
  decay(忘却)`。想起の類似度計算は Runtime の matmul を行使。
- **永続化**: OPFS 優先の汎用 keyed ストア。encoder / memory を bit 一致で復元。
- 受け入れ条件・結果は [`docs/phase-1-spec.md`](docs/phase-1-spec.md) /
  [`docs/phase-1-results.md`](docs/phase-1-results.md)。
- **残タスク**: iOS 実機で 学習→記銘→想起→忘却→保存/復元 の再現確認。

### Phase 2 — Cognitive Core と「学習ループを閉じる」（次の大きな山）

目的: 記憶する機械を、**予測し・学び・その経験を記憶に書き戻す**主体にする。

- **substrate 差し込み**: 自作エンコーダを凍結言語器官（小型事前学習）に差し替え
  可能にする。Phase 1 で判明した「少データのゼロ学習は過学習・崩壊する」問題を
  substrate で緩和する（[`docs/phase-1-results.md`](docs/phase-1-results.md) 参照）。
- **Cognitive Core**: `state_t + input → state_{t+1} + 予測`。1ステップ予測から着手。
- **予測誤差 → salience の書き戻し**: 予測が外れた経験ほど強い salience で
  Memory Fabric に記銘する。ここで「経験→記憶→想起→予測→誤差→記憶」のループが
  初めて閉じる。これが「自己成長」の最小核。
- **オンライン更新**: 破滅的忘却を避けるため、可塑性の高い層のみを更新し、
  substrate と低可塑の知識は凍結する設計を検証する。
- 想定リスク: 破滅的忘却、iOS の GPU メモリ上限、substrate の量子化/読み込み。

### Phase 3 — Meta Cognition（自己制御）

目的: AI が自分の状態を監視し、学び方を自分で調整する。

- **予測誤差の監視**: 誤差が高い領域では学習率を上げ、安定領域では凍結。
- **「質問する」能力**: 不確実性が高いときにユーザーへ問い返す（能動学習）。
- **概念化 / consolidate**: 似た記憶を統合し抽象概念を形成（記憶の圧縮と一般化）。
- **人格・目標の保持**: 低可塑メモリとして人格/目標を安定に保つ。

### Phase 4 — AI Reporter（最初のアプリ）

目的: Runtime の上で動く最初の実用アプリを載せ、基盤を検証する。

- ユーザーの入力・出来事を記憶し、要約・報告・気づきを返す。
- Runtime / Representation / Memory / Core / Meta の統合デモとして機能させる。
- AI Reporter は**最終目標ではなく**、Runtime を検証・駆動するための最初の応用。

### 継続的なテーマ（全 Phase を貫く）

- **GitHub を知識基盤に**: Personality Pack / Goal Pack / Knowledge Pack /
  Reasoning Rules を GitHub でバージョン管理し、Pack として読み込む（`provenance:pack`）。
  一方、ユーザーの学習結果・重み・経験・記憶は IndexedDB / OPFS にローカル保存。
- **Runtime の GPU 一般化**: Phase 0 の実測どおり、小演算は CPU が速い。テンソルを
  GPU に常駐させ読み戻しを最小化する resident backend は、モデルが大きくなった
  Phase 2/3 で投資する。
- **忘却を機能として設計する**: 破滅的忘却を「バグ」ではなく、可塑性で制御された
  設計要素として扱い続ける。

---

## 開発の進め方（ルール）

1. いきなり大量のコードを書かない。**要件 → 設計 → データ構造 → クラス設計 →
   ファイル構成 → 実装** の順で進める。設計が固まるまで実装は最小限。
2. 各 Phase は**受け入れ条件（DoD）**を定義し、前 Phase の DoD を満たしてから次へ。
3. 実装した数値コアは **Node で検証**してから iOS 実機で確認する。
4. 重要な決定は **ADR** に、各 Phase の仕様と結果は `docs/phase-N-*.md` に記録する。
5. 安易にライブラリ/ビルドツールへ依存しない。根本から考える。
6. 「既存 AI っぽい」より「新しい」を優先しつつ、実現可能性・保守性を犠牲にしない。

---

## ファイル構成

```
/index.html                Phase 0 検証ページ（GPU学習ループ）
/phase1.html               Phase 1 検証ページ（Representation + Memory）
/manifest.webmanifest      PWA マニフェスト
/sw.js                     Service Worker（オフライン + COEP実験口）
/icon.svg                  アイコン

/src/runtime/              Runtime 抽象（CPU/WebGPU 差し替え）
  backend.js  webgpu.js  cpu.js
/src/probe/                Phase 0 検証コード（MLP学習・永続化）
  trainStep.js  persist.js  main.js
/src/represent/            Representation
  encoder.js               自作エンコーダ（char-CBOW + IDF）
/src/memory/               Memory Fabric
  fabric.js                記銘/想起/強化/忘却
/src/store/                永続化
  blobStore.js             汎用 keyed（OPFS→IndexedDB）
/src/app/                  アプリ配線
  phase1.js

/docs/                     設計決定・仕様・結果
  ADR-0001-foundations.md
  phase-0-spec.md   phase-0-results.md
  phase-1-spec.md   phase-1-results.md
```

---

## ドキュメント索引

- [`docs/ADR-0001-foundations.md`](docs/ADR-0001-foundations.md) — 基盤決定
  （3本の柱・substrate・レイヤーのデータ契約・既知リスク）
- [`docs/phase-0-spec.md`](docs/phase-0-spec.md) / [`docs/phase-0-results.md`](docs/phase-0-results.md)
- [`docs/phase-1-spec.md`](docs/phase-1-spec.md) / [`docs/phase-1-results.md`](docs/phase-1-results.md)
