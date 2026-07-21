# Phase 1 設計書 ― Representation + Memory Fabric（最小）

- 前提: `docs/ADR-0001-foundations.md`, `docs/phase-0-results.md`
- スコープ確定:
  - Representation = **自作の軽量エンコーダ**（substrate は Phase 2 で差し込む）
  - 想起インデックス = **総当たりコサイン類似度**（ANN は Phase 2+）
  - Runtime = **CPU 主軸**（Phase 0 発見4に従う）。Backend 抽象は維持。

## 1. 要件（Phase 1 が証明するもの）

このAIの背骨の下半分:「入力をベクトルに変え、可塑性を持った記憶として蓄え、
想起し、忘れる」機械が動くこと。

```
入力テキスト → [Encoder] → 固定次元ベクトル → [Memory Fabric] 記銘/想起/強化/忘却
```

## 2. 設計の要点（なぜこうするか）

- **エンコーダは char 単位・CBOW 的なオンライン自己教師あり学習**。ユーザー入力
  列だけから「似た文脈に出る文字は似たベクトルになる」分布意味を育てる。ラベル
  不要で継続学習思想に一致。日本語のため語分割に依存せず Unicode コードポイント
  単位にする（外部トークナイザに依存しない）。
- **encode = 埋め込みの平均プーリングを L2 正規化**したもの。Phase 1 では投影
  行列を置かず、部品数を最小化する（投影は Phase 2 の選択肢）。
- **Runtime 抽象は recall で使う**。「クエリ(1×D) × 全記憶(D×N)」の類似度計算は
  matmul であり、まさに将来 GPU 常駐に載せたい大きめの計算。エンコーダ学習側は
  小さいので CPU の JS で十分。役割分担が Phase 0 発見4と噛み合う。
- **忘却を第一級の連続プロセスにする**。eviction ではなく retention の減衰。
  可塑性の低い記憶ほど忘れにくい。これが本プロジェクトの独自性の芯。

## 3. データ構造

### Encoder パラメータ

```
EncoderState = {
  D: number,                    // 埋め込み次元（例 64）
  window: number,               // CBOW 文脈窓（例 2）
  vocabCap: number,             // 埋め込み表の最大行数（例 4096）
  vocab: Map<codepoint, index>, // 出現順に採番。溢れたら予約バケットへハッシュ
  E: Float32Array,              // vocabCap × D の埋め込み表（入出力タイド）
  size: number,                 // 現在採番済みの語彙数
  step: number,
}
```

- 未知文字は出現時に採番（vocab 成長 = 小さな "成長" 挙動）。vocabCap 超過分は
  予約バケットにハッシュして衝突を許容（Phase 1 の割り切り）。
- 入出力タイド重み（word2vec と同様、入力埋め込みと出力分類器で E を共有）。

### MemoryItem

```
MemoryItem = {
  id: number,
  vector: Float32Array(D),   // L2 正規化済み
  plasticity: number,        // ∈[0,1] 可塑性。人格/知識は低く、短期記憶は高い
  retention: number,         // ∈[0,1] 生き生きさ。時間で減衰
  salience: number,          // 重要度（予測誤差が大きい記銘ほど高い; Phase1は既定値）
  provenance: string,        // 'perception' | 'derived' | 'pack'
  createdAt, lastAccess, accessCount,
  text?: string,             // デバッグ表示用（本質ではない）
}
```

### MemoryFabric パラメータ

```
FabricState = {
  D, items: MemoryItem[], nextId, tick,
  lambda: number,            // 減衰率
  reinforceGain: number,     // 想起強化の増分
}
```

## 4. クラス設計（責務と主メソッド）

### `Encoder`（src/represent/encoder.js）

- `tokenize(text) -> number[]`     コードポイント列 → 語彙 index 列（未知は採番）
- `encode(text) -> Float32Array`   埋め込み平均 → L2 正規化。副作用なしを保証
- `learn(text) -> {loss}`          CBOW を text 全体に適用しE を更新（オンライン）
- `serialize()/deserialize(buf)`   E・vocab・step の bit 一致直列化

### `MemoryFabric`（src/memory/fabric.js）

- `write(vector, attrs) -> id`     記銘
- `recall(query, k) -> Item[]`     backend.matmul で全記憶との類似度、
                                    **score = 類似度 × retention** で上位k件
- `reinforce(id)`                  retention を reinforceGain 分引き上げ（Hebbian的）
- `decay(dt)`                      retention *= exp(-lambda·dt·(1-plasticity))
- `stats()`                        件数・平均 retention 等（観測用）
- `serialize()/deserialize(buf)`   items 全体の直列化

### `blobStore`（src/store/blobStore.js）

Phase 0 の persist.js を一般化した **名前付き ArrayBuffer の永続化**。
OPFS 優先 → IndexedDB フォールバック（Phase 0 で OPFS 実機動作を確認済み）。

- `save(key, arrayBuffer) -> {store}`
- `load(key) -> {arrayBuffer, store}`

Encoder と MemoryFabric はそれぞれ serialize/deserialize を持ち、blobStore に
別キー（例 `encoder`, `memory`）で保存する。Phase 0 の `src/probe/persist.js` は
検証用として残し、本モジュールは新規に置く。

## 5. ファイル構成（Phase 1 で追加）

```
/phase1.html                  Phase 1 検証ページ（index.html=Phase0 は残す）
/src/represent/encoder.js     自作エンコーダ（char-level CBOW, 学習可能）
/src/memory/fabric.js         Memory Fabric（write/recall/reinforce/decay）
/src/store/blobStore.js       汎用 keyed 永続化（OPFS→IDB）
/src/app/phase1.js            Phase 1 検証 UI 配線
/docs/phase-1-spec.md         本設計書
```

Runtime（`src/runtime/*`）は Phase 0 のものを再利用する。

## 6. 受け入れ条件（Definition of Done）

実機（iOS PWA）での確認を必須とする。

1. **[符号化の安定性]** 同一入力に対し encode が同一ベクトルを返す（副作用なし）。
2. **[オンライン学習]** ユーザーが投入したテキスト列で CBOW loss が下がる。
3. **[意味的想起]** 学習後、意味的/文脈的に関連するテキストが recall 上位に来る
   （無関係テキストより上位）。
4. **[観測できる忘却]** decay を進めると、reinforce されない記憶の retention が
   下がり想起順位から脱落する。plasticity の低い記憶は残る。両者の差が観測できる。
5. **[永続化]** エンコーダ重みと記憶が OPFS に保存され、リロード跨ぎで復元される
   （bit 一致）。

## 7. 明示的な非目標（Phase 1 では作らない）

- substrate（凍結言語器官）の読み込み（Phase 2）
- ANN インデックス（Phase 2+）
- GPU 常駐テンソル Runtime（モデルが大きくなってから）
- Cognitive Core / Reasoning / Meta Cognition（Phase 2 以降）
- 概念化（記憶の統合・抽象化。Phase 2+ の consolidate）

## 8. 完了後の接続（Phase 2 への布石）

Phase 2 は Cognitive Core（state_t + input → state_{t+1} + action）を載せ、
その予測誤差を salience として Memory Fabric に書き戻す。ここで初めて「予測が
外れた経験ほど強く記憶される」ループが閉じる。Phase 1 の Encoder/Fabric は、
その土台として設計する。
