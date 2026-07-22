# Browser AI Runtime

HTML / CSS / JavaScript(TypeScript) のみで動く、**完全クライアントサイドの
自己成長型 AI 実行基盤**と、その上で動く **ユーザーごとに成長する AI の PWA
アプリ**を、研究・実装するプロジェクト。

- 対象は **iPhone / Safari / PWA を最優先**。
- **サーバー禁止・AI API 禁止・クラウド推論禁止**。推論はすべてユーザー端末で。
- 既存 LLM のラッパーではない。Transformer から学びつつ、ブラウザ・単一ユーザー・
  継続学習という条件に最適化した新しい認知アーキテクチャを、ゼロから自前で作る。
- 開発環境は iPhone / Claude Code / GitHub / GitHub Pages のみ。

## 最終目的（ADR-0002）

**ユーザーごとに成長する AI を提供する PWA アプリ**を作る。役割分担:

- **GitHub 側（全ユーザー共通・凍結）**: 大型ベースモデル（substrate＝言語器官）と
  Pack 群（Personality / Goal / Knowledge / Reasoning Rules）。
- **ローカル側（ユーザーごと・可塑）**: 適応重み（Cognitive Core / Meta）・
  Memory Fabric（エピソード / 概念 / 人格）・経験。**ここが成長する**。IndexedDB /
  OPFS に保存。

最初のアプリは **AI Reporter**。詳細と制約は
[`docs/ADR-0002-product-direction.md`](docs/ADR-0002-product-direction.md)。

---

## いまのステータス

| Phase | 内容 | 状態 |
|---|---|---|
| **Phase 0** | 技術検証: iOS PWA で GPU 学習ループ + 永続化/復元 | ✅ 実機合格 |
| **Phase 1** | Representation（自作エンコーダ）+ Memory Fabric（記銘/想起/忘却） | ✅ 実機合格 |
| **Phase 2a** | Cognitive Core（次文字予測）+ 学習ループ + 双方向ループ | ✅ 実機合格 |
| **Phase 2b** | 静的 substrate（実モデル埋め込み deberta-tiny）を Representation に | ✅ 実機合格 |
| **Phase 2b-2** | e5-small forward を自前実装（12層BERT 文モデル） | ✅ 実機動作 |
| **Phase 2c** | substrate を Core の予測に統合（subword 粒度・意味空間予測） | ✅ 実機動作 |
| **Phase 3a** | Meta Cognition（適応学習率 + 質問する） | ✅ 実機動作 |
| **Phase 3b** | 概念化 / consolidate（経験を忘れても概念は残る） | ✅ 実機合格 |
| **Phase 4** | AI Reporter（統合アプリ・現状 substrate のまま製品の形に） | ⬜ 次 |
| **Phase 5** | 大型 substrate を GitHub に + WebGPU forward | ⬜ 将来 |

### 検証ページ（GitHub Pages）

メニュー: <https://tenma2066-tech.github.io/Browser-AI-Runtime/>
（各 Phase の実機検証ページはメニューから開ける）

---

## アーキテクチャ

層は「名前」ではなく **入力・出力・保持する状態のデータ契約**で定義する
（ADR-0001）。

```
              ┌─────────────── GitHub（共通・凍結）────────────────┐
              │  大型 substrate（言語器官）  +  Pack 群            │
              └───────────────────────┬─────────────────────────┘
                                       │ fetch → OPFS キャッシュ
 入力 → [Representation] → ベクトル → [Memory Fabric] 記銘/想起/忘却/★概念化
                                         ↓↑
                       [Cognitive Core]  予測・学習・双方向ループ
                                         ↓↑
                       [Meta Cognition]  予測誤差で 学習率/質問/凍結 を制御
        ── すべて [Runtime]（CPU / WebGPU 差し替え可能）の上で動く ──
              ┌──────────── ローカル（ユーザーごと・可塑）─────────┐
              │  適応重み(Core/Meta) + Memory(エピソード/概念/人格) │
              └───────────────────────────────────────────────────┘
```

- **Runtime**: テンソル/カーネル抽象。WebGPU 実装と CPU 実装を同一契約で差し替え。
- **Representation**: 世界を固定次元ベクトルへ符号化。実装は3種を差し替え可能:
  ①自作 char エンコーダ ②静的 substrate（deberta 埋め込み）③自前 e5 forward。
- **Memory Fabric**: ベクトル+属性の記憶。短期/長期/知識/人格/概念を別テーブルに
  せず、**可塑性・保持・出所・count の属性で連続区別**。忘却は eviction でなく
  retention の減衰（設計された機能）。概念も同じ空間に同居。
- **Cognitive Core**: 状態遷移＋次予測（char 版と subword 版）。予測誤差 surprise を
  出す。学習ループを閉じ（驚きで記銘）、記憶が予測を助ける双方向ループを持つ。
- **Meta Cognition**: surprise 統計を監視し、適応学習率・質問・凍結を制御。
- **substrate（言語器官）**: 差し替え可能な凍結モデル。Core に駆動される末端の器官
  であり、制御の主権は Core にある（＝ LLM のラッパーではない）。

---

## これまでの成果（Phase 0〜3b・各1行）

- **Phase 0**: iOS standalone PWA で WebGPU 学習ループ＋OPFS 永続化が成立。小演算は
  CPU が GPU より速い（往復コスト）と実測 → Runtime を CPU 主軸で設計。
- **Phase 1**: 自作エンコーダ＋Memory Fabric。忘却が想起をゲートする創発を確認。
- **Phase 2a**: Cognitive Core が予測・学習し、驚きだけを記憶に書き戻す。さらに
  **記憶が予測を助ける双方向ループ**（重み不変で surprise 70%低下＝ワンショット記憶）。
- **Phase 2b**: 実モデル(deberta-tiny)の埋め込みをローカル搭載。具体語の意味検索。
- **Phase 2b-2**: **12層 BERT(e5-small) の forward を自前実装**し iPhone で実行
  （31MB, 263ms/文）。抽象クエリが静的の 1/4→3/4 に改善。
- **Phase 2c**: substrate を Core の**凍結出力ヘッド**にし、予測を意味空間で行う。
  粒度を subword に統一。
- **Phase 3a**: Meta が驚きに応じて学習率を上げ下げ（自己制御）、混乱が持続すると質問。
- **Phase 3b**: 似た記憶を概念に統合。**経験は忘れても概念は残る**（忘却と一般化の両立）。

各 Phase の仕様と結果は `docs/phase-*-spec.md` / `docs/phase-*-results.md`。

---

## これからの計画

### Phase 4 — AI Reporter（次）

これまでの層を統合した**最初の実アプリ**。**現状の小型 substrate のまま**、製品の
体験を先に固める（大型化は待たない）。

- ユーザーの入力・出来事を Memory に記銘 → 概念化で要点抽出 → 要約・報告・気づきを返す。
- Cognitive Core の surprise で「注目すべき出来事」を検出、Meta で「分からないこと」
  を質問（能動学習）。
- 成長物（重み・記憶・概念・人格）はローカル永続化。使うほど各ユーザーで育つ。

### Phase 5 — 大型 substrate を GitHub に + WebGPU forward（将来）

言語品質を本格的に上げる infra 投資（ADR-0002 の制約に従う）。

1. **iOS 最大モデルサイズの実機スパイク**（どこまで載る/回るか確定）。
2. **GitHub Releases で大型モデル配信 + OPFS キャッシュ**（初回 fetch → 以後ローカル）。
3. **WebGPU forward**（自前 JS では大型は遅い → Runtime の GPU 経路に投資）。

### 継続テーマ

- **GitHub を知識基盤に**: Personality / Goal / Knowledge / Reasoning Pack を
  バージョン管理し `provenance:'pack'` で読み込む。
- **ユーザー成長物のローカル永続化**: blobStore（OPFS→IndexedDB）で bit 一致保存。
- **忘却を機能として設計し続ける**: 破滅的忘却を可塑性で制御された設計要素として扱う。
- **Runtime の GPU 一般化**: 大型モデルが要求する段階で resident backend に投資。

---

## 開発の進め方（ルール）

1. **要件 → 設計 → データ構造 → クラス設計 → ファイル構成 → 実装** の順。設計が
   固まるまで実装は最小限。
2. 各 Phase は**受け入れ条件（DoD）**を定義し、満たしてから次へ。
3. 数値コアは **Node で検証**してから **iOS 実機で確認**する。
4. 重要な決定は **ADR**、各 Phase の仕様と結果は `docs/phase-*.md` に記録する。
5. 安易にライブラリ/ビルドツールへ依存しない。根本から作る。
6. 「既存 AI っぽい」より「新しい」を優先しつつ、実現可能性・保守性を犠牲にしない。

---

## ファイル構成

```
*.html                      各 Phase の検証ページ（index.html はランチャー）
manifest.webmanifest        PWA マニフェスト
sw.js                       Service Worker（network-first・オフライン）
icon.svg

src/runtime/                Runtime 抽象（CPU / WebGPU 差し替え）
  backend.js  cpu.js  webgpu.js
src/represent/              Representation（3実装）
  encoder.js                自作 char エンコーダ（CBOW + IDF）
  substrate.js              静的 substrate（SentencePiece + int8 埋め込み lookup）
  e5.js                     e5-small forward 自前実装（12層 BERT）
src/memory/
  fabric.js                 Memory Fabric（記銘/想起/強化/忘却/概念化）
src/cognition/
  core.js                   Cognitive Core（char・forward/applyGradient/双方向）
  subwordcore.js            Subword Core（substrate を凍結入出力に）
  meta.js                   Meta Cognition（適応学習率・質問）
src/store/
  blobStore.js              永続化（OPFS → IndexedDB フォールバック）
src/app/                    各 Phase の UI 配線
src/probe/                  Phase 0 検証コード

models/                     取り込んだ凍結 substrate（自前 compact 形式）
  substrate-deberta-tiny-ja/   静的埋め込み 7MB
  substrate-e5-small-ja/       e5-small 重み 31MB
tools/                      オフライン抽出パイプライン（Node）
  extract-substrate.mjs  extract-e5.mjs
docs/                       ADR・各 Phase の仕様/結果
```

---

## ドキュメント索引

- [`docs/ADR-0001-foundations.md`](docs/ADR-0001-foundations.md) — 基盤決定
  （3本の柱・substrate・レイヤーのデータ契約・既知リスク）
- [`docs/ADR-0002-product-direction.md`](docs/ADR-0002-product-direction.md) —
  最終目的の明確化・大型 substrate を GitHub に置く方針と制約
- Phase 仕様/結果: `docs/phase-0-*` … `docs/phase-3b-*`（各 spec と results）
