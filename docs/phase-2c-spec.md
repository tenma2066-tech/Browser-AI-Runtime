# Phase 2c 設計書 ― substrate を Cognitive Core の予測に統合（粒度設計）

- 前提: `docs/phase-2a-results.md`（char Core）, `docs/phase-2b-results.md`（静的substrate）
- 目的: Phase 2a の char Core（語彙47・意味なし）を、**substrate の意味空間で
  予測する subword Core** に発展させ、粒度（char↔subword）を統一する。

## 1. 設計の核（なぜこうするか）

- **粒度統一**: 入力・予測を全て subword にする。substrate のトークナイザ／埋め込みを
  そのまま使う。
- **凍結 substrate を「入力埋め込み」かつ「出力ヘッド」に使う**:
  - 入力 x_t = substrate.row(subword_t)（192次元・凍結）。
  - 出力: Core の隠れ状態 h を substrate 空間へ射影 `z = Wproj·h`（192次元）し、
    **全 subword を `logits[v] = z · substrate.row(v)` でスコア**→ softmax。
  - つまり分類器＝**凍結 substrate 埋め込み行列**。巨大な学習対象 Why を新設しない。
- **効果**: Core が substrate の意味空間で予測する。予測確率が**意味的に近い
  subword に集まる**（事前学習知識の活用）。char Core にはなかった「意味的予測」。
- **使う substrate**: 静的 deberta-tiny（192次元・32k語彙）。per-token lookup が軽く、
  毎ステップの出力スコアリングに向く（e5 は per-token forward が重いので不適）。

## 2. データ構造

### SubwordCore（src/cognition/subwordcore.js）

```
inDim = 192（substrate 次元）, H = 128（隠れ）
Wxh: Float32Array(H*inDim)   // 入力→隠れ
Whh: Float32Array(H*H)       // 再帰
bh:  Float32Array(H)
Wproj: Float32Array(inDim*H) // 隠れ→substrate空間（出力射影）
h:   Float32Array(H)         // 状態
step
```

出力の「分類器」は substrate の凍結埋め込み（int8+scale）を都度使用。学習対象は
Wxh/Whh/bh/Wproj のみ（≈ H·inDim·2 + H² ≈ 6.6万 params）。

## 3. forward / 学習

- **forward(substrate, tokId, targetId)** → { surprise, entropy }:
  1. x = substrate.row(tokId)（凍結）
  2. h = tanh(Wxh·x + Whh·h_prev + bh)
  3. z = Wproj·h（inDim）
  4. logits[v] = scale[v]·(z · i8_row_v)（全 v）→ softmax
  5. surprise = -log p[target], entropy = -Σ p ln p
  6. 活性をキャッシュ、h 前進
- **applyGradient(lr)**: truncated BPTT(1)。
  dlogits = softmax - onehot(target)、
  dz[d] = Σ_v dlogits[v]·emb_v[d]、
  dz → Wproj（dWproj, dh）→ tanh → Wxh/Whh/bh。**凍結埋め込みは更新しない**。
- **観測**: 出力スコアリングと勾配は 32k×192 の演算/ステップ。短文なら CPU/JS で
  現実的か Node で計測する（重ければ WGSL 化は将来）。

## 4. 検証デモ（phase2c.html）

- コーパス（雨/動物）を subword 学習 → surprise 低下。
- **意味的予測**: 文脈（例「犬が」）を与え、Core の**次 subword 上位k**を表示。
  意味的に一貫した subword（動物・動作系）に確率が集まることを見る。
  → char Core（意味なし）との質的な違いを示す。
- 保存/復元/自己テスト。

## 5. 受け入れ条件（DoD）

1. **[学習]** subword ストリームで surprise が下がる。
2. **[意味的予測]** 文脈に対する次 subword 上位が意味的に一貫（凍結出力ヘッドの効果）。
3. **[粒度統一]** 入力・予測・想起が全て subword/substrate 空間で一貫。
4. **[永続化]** Core 重みが bit 一致で保存・復元。
5. **[速度]** iOS で 1文の forward が現実的な時間（要計測、重ければ将来最適化）。

## 6. 非目標

- e5 forward を Core に使う（重い。static deberta を使う）。
- Meta/双方向ループの再実装（既存を後段で接続可能な形にはするが本 Phase では最小）。
- 概念化（Phase 3b）。
