# Phase 2b-2 設計書 ― sentence-embedding substrate（自前 forward）

- 前提: `docs/phase-2b-results.md`（静的 substrate の限界＝抽象クエリに弱い）
- 方針決定: 抽象的な文類似を良くするため、文用に学習された sentence-embedding
  モデル `intfloat/multilingual-e5-small` を substrate に採用し、**その Transformer
  forward をゼロから自前実装**する（ライブラリ非依存を貫く）。

## 1. なぜこのモデル・この方針か

- 日本語専用の小型文モデル（ruri 等）は **MeCab 依存**でブラウザ実行不可。
- multilingual-e5-small は **vanilla BERT**（標準 self-attention・LN・GELU、
  相対位置なし）＋ **Unigram SentencePiece**（MeCab 不要）で、実装・検証しやすい。
- 静的埋め込み（Phase 2b）は具体語に強いが抽象文類似に弱い。文用学習モデルの
  文脈化出力が必要（Sentence-BERT 系の知見）。

## 2. モデル構造（確認済み）

- vocab 250002（Unigram SP、特殊: `<s>`0 `<pad>`1 `</s>`2 `<unk>`3）
- hidden 384, layers 12, heads 12, intermediate 1536, gelu, max_pos 512
- embeddings: word[250037,384] + position[512,384] + token_type[2,384] + LayerNorm
- layer: attention(query/key/value/output.dense + LayerNorm), intermediate.dense,
  output.dense + LayerNorm（標準 BERT）
- 文ベクトル: 最終層 hidden の **mean pooling（attention mask 加重）→ L2 正規化**
- E5 規約: クエリに `query: `、文書に `passage: ` の接頭辞

## 3. 語彙剪定（必須）

word_embeddings は fp32 で 384MB。日本語アプリに不要な多言語トークンを落とす。

- 残す条件: ピースが「ひらがな/カタカナ/漢字/長音符」を含む、または ASCII・
  数字・基本記号のみ、または特殊トークン、または `▁` 単体。
- 単一日本語文字ピースは必ず残す（任意の日本語文を最悪 1 文字ずつ分割可能に）。
- 残ったピースに新 id を連番付与し、対応する埋め込み行だけを取り出す。
- 見込み: 25万 → 概ね 3〜4万。int8 で約 12〜16MB。

## 4. 成果物（`/models/substrate-e5-small-ja/`）

```
emb.i8 / emb.scale     剪定・int8量子化した word 埋め込み [K,384]
layers.bin             position/token_type/LayerNorm と 12層の重み（int8 or fp16）
layers.meta.json       各テンソルの名前・形状・オフセット・量子化方式
tokenizer.json         剪定後の unigram pieces/scores/unk・特殊トークン id
meta.json              hidden/layers/heads/vocab/pooling 規約
```

## 5. 実装（段階・de-risk）

1. **抽出＋剪定**（`tools/extract-e5.mjs`, Node）: safetensors から必要テンソルを
   range/stream で取得、語彙剪定、量子化、`/models` へ書き出し。剪定後サイズ確定。
2. **JS forward**（Node で先に）: `src/represent/e5.js` に BERT forward を実装
   （埋め込み→12層→mean pool→L2）。SentencePiece は既存 Viterbi を流用。
3. **検証**（Node）: 抽象クエリ（かわいいペット/電車で移動/おなかが空いた 等）が
   正しいクラスタを引く（Phase 2b の静的では 1/4 だったものが改善するか）。
4. **ブラウザ移植 + iOS**: forward を移植（CPU/JS、必要なら WGSL）、実機で
   読み込み・速度・メモリを確認。

## 6. 受け入れ条件（DoD）

1. **[サイズ]** 剪定後の substrate が iOS に載る現実的サイズ（目標 <50MB）。
2. **[forward 正しさ]** JS forward が機能的に妥当（mean-pool 文ベクトルで
   語彙・文の意味が整合。MLM 参照は無いが、既知の関連/無関係で検証）。
3. **[抽象クエリ改善]** 抽象クエリの想起が静的 substrate（1/4）より明確に改善。
4. **[iOS]** 実機で読み込み・encode・想起が動く（速度・メモリ許容内）。
5. **[非依存]** 外部ライブラリに依存せず自前実装（トークナイザ・forward とも）。

## 7. リスク

- **サイズ/メモリ**: 剪定＋量子化で ~35MB 目標。超過なら層 int8・語彙さらに剪定。
- **量子化誤差**: 12層で誤差蓄積の恐れ。まず埋め込み int8・層 fp16 で検証、
  必要なら層も int8 化して差を測る。
- **速度**: 12層×短文は CPU/JS でも軽い見込み（Phase 0 の実測より）。重ければ WGSL。
- **トークナイザ差異**: NFKC・`▁`・特殊トークンの扱いで本家と完全一致しない可能性。
  検証は「意味的想起の妥当性」で行い、ビット一致は求めない。
