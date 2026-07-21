# Phase 2b 結果記録（Representation への substrate 差し込み）

- 実施日: 2026-07-21
- 実機: iPhone / iOS Safari / standalone PWA（`backend = webgpu`）
- 前提: `docs/phase-2b-spec.md`
- 状態: **iOS 実機で確認。substrate を Representation として差し込み成功。**
  （Core の subword 予測統合＝2c、文脈化フル forward＝2b-2 は将来）

## 判定

| 受け入れ条件 | 結果 | 実測 |
|---|---|---|
| #1 読み込み（iOS で 7MB を保持・使用） | ✅ | 語彙32000 / dim192 / 読込 5.0秒 |
| #2 凍結・無依存 | ✅ | 学習で変化せず。外部ライブラリなし（自前 unigram Viterbi） |
| #3 非崩壊・被覆・安定想起 | ✅（具体語） | 語彙cosine が Node と完全一致、具体語クエリは的確 |
| #4 Core 統合 | ⏸ 2c へ延期 | 粒度差（char↔subword）のため分離 |
| #5 永続化 | ✅ | substrate は読み取り専用。端末学習物(Core/記憶)の永続化は 2a で実証済 |

## 実機の数字

- 語彙内 cosine（int8 逆量子化）が Node と **完全一致**: 猫-犬 0.716 / 東京-大阪
  0.729 / 車-電車 0.616 / 雨-傘 0.349 / 猫-電車 0.146 / 雨-東京 0.053。
  → int8 量子化・逆量子化が端末間で決定的に再現されることを確認。
- 意味的想起「美味しい食事」→ カレーライス / 寿司 / たこ焼き（食事クラスタ）。

## 正直な限界（記録）

- **静的埋め込みの平均プーリングは、具体語では効くが抽象的な言い換えに弱い。**
  「高い山」「楽しい旅行」等のクエリは無関係文を上位に返した。これは静的
  サブワード埋め込み＋mean-pooling の既知の限界で、文脈化（フル forward）を
  しないことの帰結。
- 改善には Phase 2b-2（文脈化 forward）が要るが、DeBERTa の disentangled
  attention 実装は iOS 向けに大きな工事。費用対効果は要検討。

## 意味

- **実在の事前学習モデル（ku-nlp/deberta-v2-tiny-japanese）を、サーバーなし・
  完全ローカルで iPhone に載せた。** 抽出 → int8量子化(6MB) → GitHub 配信 →
  ブラウザで SentencePiece + lookup + pooling。全て自前・無依存。
- Representation が「47文字・8文で崩壊する自作器官（Phase 1）」から「3.2万
  サブワード・巨大データ由来の凍結器官」に置き換わった。ADR-0001 の substrate
  構想が実物になった。

## 次アクションの候補

- **Phase 3（Meta Cognition）**: surprise 統計の監視で学習率制御・質問。既存部品の
  上に載る高価値・低リスク。
- **Phase 2c**: substrate を Core の subword 予測に統合（粒度を揃える）。
- **Phase 2b-2**: 文脈化フル forward（抽象クエリ精度↑、ただし大工事）。
