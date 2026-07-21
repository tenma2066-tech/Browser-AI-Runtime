# Phase 2a 設計書 ― Cognitive Core と「学習ループを閉じる」

- 前提: `docs/ADR-0001-foundations.md`, `docs/phase-1-results.md`
- スコープ確定（Phase 2 を細く割った前半）:
  - **次文字予測 ＋ 書き戻しのみ**（記憶→予測の条件づけは 2a 後半/2b へ）
  - substrate は差し込まない（Phase 2b）。自作エンコーダ（Phase 1）を凍結して使う。
  - Runtime は CPU 主軸（Phase 0 実測）。

## 1. 要件（Phase 2a が証明するもの）

記憶する機械を、**予測し・自己評価し・学び、驚いた経験を記憶に書き戻す主体**に
する。最小のループを閉じる:

```
入力文字列 → encode(凍結) → [Cognitive Core] 次文字を予測 → 実際を観測
                                 ↓ surprise = 予測誤差(交差エントロピー)
                    ┌────────────┴─────────────┐
              Core をオンライン更新       surprise > 閾値 の瞬間だけ
              （学ぶ）                    salience=surprise で Memory に記銘
                                          （驚いた経験ほど強く記憶）
```

**独自性の芯**: 記憶を「全部」ではなく、**AI 自身の予測が外れた驚きの瞬間だけ**
書き込む。何を覚える価値があるかを、予測誤差が決める。全文保存とは根本的に違う。

## 2. 設計（なぜこうするか）

- **Core は次文字を予測する**。surprise が交差エントロピーで明確に定義でき、
  「何を予測して外したか」が解釈可能。encoder の埋め込みと語彙をそのまま使える。
- **入力は凍結エンコーダ埋め込み**。Representation が Core に入力特徴を供給する
  という層構成が実際に接続される。Core の学習は encoder を書き換えない。
- **truncated BPTT(1) のオンライン更新**。時間方向は1ステップだけ誤差を流す
  （h_{t-1} は定数扱い）。iOS/CPU で軽く、loss 低下の実証には十分。完全 BPTT は
  後の最適化。
- **書き戻しは encoder 空間のベクトル**（驚いた位置周辺の文字窓を encode）で行う。
  これにより Phase 1 の Memory Fabric（recall は encode(query) で照合）とそのまま
  接続でき、記憶が一貫した空間に貯まる。
- **閾値は走査平均の EMA**。学習が進むほど surprise が下がり、書き込みが減る
  （馴染んだものは記憶しなくなる）挙動が自然に出る。

## 3. データ構造

### CognitiveCore

```
CoreState = {
  D,            // 入力次元（= encoder.D）
  H,            // 隠れ状態次元
  vocabCap,     // 出力語彙の最大数
  V,            // 現在の有効語彙数（= encoder.size）
  Wxh: Float32Array(H*D),      // 入力→隠れ
  Whh: Float32Array(H*H),      // 隠れ→隠れ（再帰）
  bh:  Float32Array(H),
  Why: Float32Array(vocabCap*H), // 隠れ→出力（語彙）。有効行のみ保存
  by:  Float32Array(vocabCap),
  h:   Float32Array(H),        // 現在の状態 state_t
  step, lr,
}
```

Core は encoder への参照を各メソッドで受け取り、入力埋め込み `encoder.E[idx]`
（凍結）と語彙 `encoder.size` を使う。Core の直列化は自分の重みのみ（encoder は別）。

### 書き戻しに使う Memory（Phase 1 の MemoryFabric をそのまま利用）

```
記銘 attrs = { text: 文字窓, salience: surprise, plasticity: 0.5, provenance: 'derived' }
```

## 4. クラス設計

### `CognitiveCore`（src/cognition/core.js）

- `reset()` — h を 0 に（系列/文の境界でリセット）
- `observe(encoder, idx, targetIdx, learn) -> surprise`
  1 ステップ: forward（h 更新, 語彙 logits）→ surprise 計算 →
  learn=true なら truncated BPTT(1) で Wxh/Whh/bh/Why/by を更新 → h を前進
- `serialize()/deserialize(buf)` — 有効行のみ bit 一致直列化

### `src/app/phase2.js`（配線・ループ制御）

- encoder を Phase 1 同様に学習して凍結
- ストリーム学習: 文ごとに `core.reset()` → 各位置で `observe(...,learn=true)`。
  surprise の EMA を更新し、閾値超えかつ未記銘の文字窓を Memory に write
- 自己評価: `observe(...,learn=false)` で文の平均 surprise を測る（馴染み vs 新規）
- 想起 / 記憶ビュー / 保存 / 復元 / 自己テスト

## 5. ファイル構成（Phase 2a で追加）

```
/phase2.html               Phase 2a 検証ページ
/src/cognition/core.js     Cognitive Core（次文字予測 + オンライン学習）
/src/app/phase2.js         Phase 2a UI 配線
/docs/phase-2a-spec.md     本設計書
```

既存の `src/represent/encoder.js`, `src/memory/fabric.js`, `src/store/blobStore.js`,
`src/runtime/*` を再利用。`index.html`(ランチャー) に Phase 2 カードを追加、`sw.js`
のキャッシュを更新。

## 6. 受け入れ条件（DoD）

実機（iOS PWA）確認を必須とする。

1. **[学習]** テキスト列を反復して与えると、平均 surprise（予測 loss）が下がる。
2. **[自己評価]** 学習後、馴染みの文より新規（未知語を含む）文で平均 surprise が
   高い。
3. **[驚きの記銘]** surprise が閾値を超えた瞬間だけ高 salience で Memory に記銘
   され、学習が進むと記銘が減る。全文ではなく notable な断片が貯まる。
4. **[永続化]** Core 重み ＋ encoder ＋ Memory が bit 一致で保存・復元される。

## 7. 非目標（Phase 2a では作らない）

- 記憶 → 予測の条件づけ（想起で次予測を変える双方向ループ）。2a 後半/2b。
- substrate（凍結言語器官）の差し込み。Phase 2b。
- 完全 BPTT / GPU 常駐。必要になってから。
- Meta Cognition（学習率の自己制御・質問する）。Phase 3。

## 8. Phase 2b / Phase 3 への接続

- 2b: encoder を substrate に差し替え、少データ過学習（Phase 1 で観測）を緩和。
- 2a 後半: 想起した過去の驚きで次予測を条件づけ、再出現時に surprise が下がる
  ことを示す（双方向ループ）。
- 3: surprise の統計を Meta Cognition が監視し、学習率調整・質問・凍結を制御。
