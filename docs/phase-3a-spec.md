# Phase 3a 設計書 ― Meta Cognition（適応学習率 ＋ 質問する）

- 前提: `docs/phase-2a-results.md`（Cognitive Core が surprise を出す）
- スコープ確定: **適応学習率 ＋ 質問する**を、**Phase 2a の char Core の上**に載せる。
  substrate 統合（Phase 2c）は持ち込まない。

## 1. 要件（Phase 3a が証明するもの）

システムが自分の学び方を自分で制御する。surprise（予測誤差）とエントロピー
（予測の不確実性）を監視し:

1. **適応学習率** — 驚く領域で lr を上げて速く適応、安定領域で lr を下げて凍結
   （過学習・破滅的忘却の緩和）。
2. **質問する** — 不確実性が高く、かつ驚きが持続するときだけユーザーに問い返す
   （能動学習）。馴染んだ入力では黙る。

## 2. 設計（なぜこうするか）

- **Core を「思考(forward)」と「学習(applyGradient)」に分割**する。現状の
  `observe(learn=true)` は forward と更新が一体で、lr を「surprise を見てから」
  決められない。分割すれば `forward → surprise/entropy → Meta が lr 決定 →
  applyGradient(lr)` の二段にでき、メタ制御が自然に入る。
- **entropy を Core が出す** — softmax から `H = -Σ p ln p` を計算して返す。
  不確実性の指標（質問トリガに使う）。
- **適応 lr は相対値（自己較正）** — `lr = baseLr × clamp(surprise / EMA(surprise),
  lo, hi)`。「最近の平常より驚いたら速く学ぶ」。学習が進み baseline が下がっても
  相対で機能。安定入力（surprise≪baseline）では factor が下限に張り付き凍結気味。
- **質問は持続する混乱に対してのみ** — 1発の高 surprise はノイズかもしれない。
  直近窓の平均 surprise が高い AND entropy が高い ときだけ ask。クールダウンで
  連発を防ぐ。char モデルなので「質問文」は自然言語生成せず、混乱している文脈
  断片を提示する（能動学習の信号として十分）。

## 3. データ構造

### CognitiveCore の拡張（後方互換を保つ）

```
forward(encoder, idx, targetIdx) -> { surprise, entropy }
   hNew/logits/softmax を計算、surprise と entropy を返し、内部に更新用の
   活性（x, hPrev, hNew, probs, target）をキャッシュ。h を前進。
applyGradient(lr)
   直前 forward のキャッシュから truncated BPTT(1) で重み更新。
observe(encoder, idx, targetIdx, learn=true) -> surprise   // 従来互換
   forward → learn なら applyGradient(this.lr) → surprise を返す
predict(...)   // 既存（双方向ループ用）そのまま
```

### MetaCognition（src/cognition/meta.js）

```
MetaState = {
  baseLr, emaSurprise, alpha,           // 平常 surprise の EMA
  window:number[], windowSize,          // 直近 surprise
  lrLo, lrHi, entropyThresh, askSurpriseMult, cooldown, sinceAsk,
}
control(surprise, entropy, V) -> { lr, ask, lrFactor, normEntropy }
   EMA/窓を更新。lrFactor=clamp(surprise/(emaSurprise+eps), lrLo, lrHi)、
   lr=baseLr×lrFactor。normEntropy=entropy/ln(V)。
   ask = normEntropy>entropyThresh AND mean(window)>emaSurprise×askSurpriseMult
         AND sinceAsk>=cooldown。
serialize()/deserialize()   // 統計の永続化（bit 一致）
```

## 4. クラス設計・責務

- `CognitiveCore`: forward/applyGradient/observe/predict（上記）。
- `MetaCognition`: 統計監視、lr 決定、質問判定。Core は触らない（観測と制御のみ）。
- `src/app/phase3.js`: ループ制御。各ステップ `{s,H}=core.forward(...)` →
  `{lr,ask}=meta.control(s,H,V)` → `core.applyGradient(lr)`。ask なら質問を提示。

## 5. ファイル構成（追加）

```
/phase3.html                Phase 3a 検証ページ
/src/cognition/meta.js      MetaCognition
/src/app/phase3.js          Phase 3a UI 配線
/docs/phase-3a-spec.md      本設計書
```
`src/cognition/core.js` を forward/applyGradient に分割（observe 互換維持）。

## 6. 受け入れ条件（DoD）

1. **[適応 lr]** 馴染みコーパス学習後に新規パターンを注入すると、lr が自動で上がり、
   固定 lr より速く surprise が下がる（適応が速い）。
2. **[凍結]** 馴染み入力では lr factor が下限側に下がる（過学習・上書きの抑制）。
3. **[質問]** 本当に新規/混乱した入力にだけ ask フラグが立ち、馴染み入力では
   立たない。連発しない（クールダウン）。
4. **[永続化]** Core ＋ Meta 統計が bit 一致で保存・復元。
5. **[回帰なし]** 既存 Phase 2a/2b（observe/predict 利用）が壊れない。

## 7. 検証デモ（phase3.html）

- フェーズA: Phase 2a の馴染みコーパス（雨/動物）で Core+Meta を数パス学習
  → surprise 低下、lr が自動で下がる。
- フェーズB: 新規文（未知語を含む）を注入 → Meta が lr を上げ、質問フラグが立ち、
  適応する様子を観測。
- 対照: 同じ注入ストリームで「固定 lr の Core」と「Meta 制御 Core」を並走させ、
  Meta 側が速く surprise を下げることを示す。

## 8. 非目標（Phase 3a では作らない）

- 概念化 / consolidate（記憶の統合・抽象化）。Phase 3b。
- substrate を Core に統合（Phase 2c）。
- 自然言語での質問文生成（char モデルのため断片提示で代替）。
