# Phase 2b 設計書 ― substrate（凍結言語器官）差し込み

- 前提: `docs/ADR-0001-foundations.md`, `docs/phase-2a-results.md`
- 方針決定: **既存オープン小型モデルを取り込む**（ユーザー選択）。ただしフルの
  Transformer forward は iOS に載せず、**事前学習済み埋め込み ＋ SentencePiece
  語彙**を凍結 substrate として取り込む（lookup+pooling のみ、attention 不要）。

## 1. なぜこの形か（制約との折り合い）

- 実モデルのフル forward（DeBERTa の disentangled attention）を iOS で回すのは
  重く、自前推論エンジンの大工事になる。→ Phase 2b では回さない。
- substrate の目的は「巨大データ由来の言語知識で、Phase 1 の少データ過学習・崩壊
  を防ぐ」こと。**事前学習済みサブワード埋め込み**はそれを最小コストで満たす。
- 文脈化フル forward は Phase 2b-2（将来）に分離。

## 2. 取り込むモデル

- `ku-nlp/deberta-v2-tiny-japanese`（Apache/研究用途、CC-100 等で事前学習）
  - hidden=192, layers=3, vocab=32000, tokenizer=SentencePiece(unigram)
  - 使うのは `deberta.embeddings.word_embeddings.weight` = [32000, 192] のみ
  - fp32 24.6MB → **int8（行ごとスケール）で約6MB**

## 3. データ構造・成果物（`/models/substrate-deberta-tiny-ja/`）

```
emb.i8        int8 量子化した埋め込み 32000×192（=6.14MB）
emb.scale     行ごとスケール float32 ×32000（=128KB）
meta.json     { vocab:32000, dim:192, quant:'int8-perrow', model, license }
tokenizer.json（compact） unigram の pieces[] と scores[]、unk_id、正規化フラグ
```

逆量子化: `emb[v][d] = emb_i8[v][d] * emb_scale[v]`。

## 4. クラス設計

### オフライン（`tools/extract-substrate.mjs`, Node）

- safetensors ヘッダを取得し `word_embeddings.weight` のバイト範囲を特定
- その範囲のみ range-download（24MB）→ Float32 として読む
- 行ごと int8 量子化（scale = max(abs(row))/127）
- `tokenizer.json` から unigram の pieces/scores/unk を抽出し compact 化
- `/models/...` に書き出す

### ブラウザ（`src/represent/substrate.js`）

- `loadSubstrate(baseUrl) -> Substrate`：emb.i8 / emb.scale / meta / tokenizer を
  fetch。int8 を保持し、参照時に逆量子化（メモリ節約）。
- `SentencePieceTokenizer`：`String.normalize('NFKC')` で正規化 → `▁` 変換 →
  unigram スコアの Viterbi で最良分割 → token id 列。
- `Substrate.encode(text) -> Float32Array(dim)`：tokenize → 埋め込み lookup →
  平均（または IDF）pooling → L2 正規化。**凍結**（学習しない）。
- Encoder と同じ `encode(text)->vector` 契約に合わせ、Representation にドロップイン。

## 5. 統合

- `phase2.html`/`phase2.js` に substrate 切替を追加（自作エンコーダ ↔ substrate）。
- substrate の dim（192）に合わせて Core の D を設定（Core は端末で学習し直す）。
- 想起・Cognitive Core・双方向ループが substrate 上でも動くことを確認。

## 6. 受け入れ条件（DoD）

1. **[読み込み]** `/models` の compact substrate を iOS PWA で fetch・保持できる
   （約6MB、メモリ内で逆量子化して使える）。
2. **[凍結・無依存]** substrate は学習で変化しない。外部ライブラリに依存しない。
3. **[非崩壊・被覆]** SentencePiece により未知語被覆が広く、Phase 1 の char
   エンコーダのような崩壊が起きない。関連文の想起が安定する。
4. **[統合]** substrate 埋め込みを入力に Cognitive Core が学習でき、双方向ループ
   も動く。
5. **[永続化]** 端末側の学習物（Core/記憶）は従来どおり bit 一致で保存・復元。
   substrate 自体は読み取り専用（ユーザー重みではない）。

## 7. 非目標（Phase 2b では作らない）

- DeBERTa のフル forward（文脈化埋め込み）。Phase 2b-2。
- substrate の再学習/微調整（凍結が原則）。
- GPU 常駐推論。

## 8. リスクと対策

- **iOS メモリ**: 6MB int8 は許容範囲。fp32 展開は行わず int8 保持＋都度逆量子化。
- **トークナイザ差異**: NFKC 正規化・`▁`・unk 処理の実装差で本家と完全一致しない
  可能性。Phase 2b では「実用上妥当な分割」を DoD とし、完全一致は求めない。
- **リポジトリ肥大**: 6MB 成果物を Git に置く（GitHub を model store とする方針に
  沿う）。将来サイズが問題化すれば分割や別配信を検討。
