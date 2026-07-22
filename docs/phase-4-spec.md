# Phase 4 設計書 ― AI Reporter（統合アプリ）

- 前提: Phase 1（Memory）, 2b（substrate）, 3a（質問）, 3b（概念化）
- 目的: これまでの層を統合した**最初の実アプリ**。現状の小型 substrate のまま、
  製品の体験を固める（大型化は Phase 5）。

## 1. コンセプト

AI Reporter は**ユーザーの入力（日々の出来事・メモ・思考）を記憶し、そこから
テーマ・注目の出来事・気づき（質問）を浮かび上がらせる**パーソナル AI。

**文章生成はしない。** 我々のスタックの強みは生成でなく「記憶・概念・想起」。
「レポート」＝ユーザー自身の入力を、概念（テーマ）・高 salience（注目）・
未知（質問）として整理して返す。使うほど各ユーザーで育つ（ローカル永続化）。

## 2. 統合する層

- **Representation**: substrate.encode（静的 deberta, 7MB）でエントリを符号化。
- **Memory Fabric**: salience＝新規性でエントリを記銘。
- **概念化（Phase 3b）**: consolidate でテーマを抽出。
- **質問（Phase 3a の精神）**: 既存概念に当てはまらない新規エントリで問い返す。
- **忘却（Phase 3b）**: 平凡なエントリは decay で薄れ、概念は残る。
- **永続化**: blobStore（OPFS）で全状態を保存。使うほど育つ。
- **Runtime**: 想起の類似度計算は backend.matmul。

（Cognitive Core の次予測は本 Phase では使わない。レポートに不要。将来、文体適応
などで接続可能。）

## 3. データ構造・クラス

### AIReporter（src/reporter/reporter.js）

```
constructor(backend, substrate, {D})
  fab = new MemoryFabric(backend, {D, lambda})
  noveltyEMA

ingest(text) -> { novelty, question|null }
  v = substrate.encode(text)
  novelty = エピソードが無ければ 1、あれば 1 - max cos(v, 既存エピソード)
  salience = novelty
  fab.write(v, {text, salience, plasticity:0.7})
  最寄り概念との cos が低く novelty が高い → question（新しい話題）
consolidate() -> {concepts, clustered}      // fab.consolidate
passTime(dt) -> void                          // fab.decay（平凡な記憶を薄れさせる）
report() -> { themes, notable, questions, stats }
  themes   = 概念（count 降順、代表テキスト）
  notable  = 生存エピソードを salience×retention 降順で上位
  stats    = エントリ数・概念数
serialize()/save()/load()                     // blobStore で永続化
```

## 4. UI（phase4.html）

- テキスト入力＋「記録」。サンプル投入ボタン（複数テーマの例エントリ）。
- 「概念化」「時間経過（decay）」「レポート」ボタン。
- レポート表示: **テーマ**（概念）/ **注目の出来事**（高 salience）/ **AI からの質問**。
- 保存/復元。

## 5. 受け入れ条件（DoD）

1. **[記録・成長]** エントリを記録するほど Memory が育ち、概念が増える。
2. **[レポート]** テーマ・注目の出来事・質問を含むレポートが出る。
3. **[新規で質問]** 既存テーマに無い新規エントリで AI が質問する。
4. **[忘却と概念]** 時間経過で平凡なエントリは薄れ、テーマ（概念）は残る。
5. **[永続化・成長]** 全状態がローカル保存され、再訪で継続（使うほど各ユーザーで育つ）。

## 6. 非目標

- 文章生成（要約文の自動生成）。整理・提示で代替。
- Cognitive Core の統合（本 Phase では不要）。
- 大型 substrate（Phase 5）。
