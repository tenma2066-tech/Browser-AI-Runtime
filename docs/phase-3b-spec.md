# Phase 3b 設計書 ― 概念化 / consolidate（似た記憶を統合し抽象概念）

- 前提: `docs/phase-1-results.md`（Memory Fabric）, `docs/phase-2b-results.md`（substrate）
- 目的: 似た記憶を統合して**抽象概念**を作る。個別エピソードは薄れても概念は残る
  （忘却と一般化の両立）。ビジョンの「概念化する」。

## 1. 設計の核（なぜこうするか）

- **概念も「属性で区別される memory」** として Memory Fabric に同居させる
  （ADR-0001「同一ベクトル空間・属性で連続区別」に整合）。別テーブルを作らない。
- consolidate: 似たエピソード群をクラスタ化し、**centroid ベクトル**を概念 memory
  として書く。概念は **低可塑（忘れにくい）・高 retention**。元エピソードは
  **高可塑化＋retention 低下**で薄れさせる。
- これにより「個別経験は忘れるが概念は残る」= 忘却と一般化が両立する。
- 概念候補は「馴染んだ記憶」（Phase 3a の surprise 低下と接続可能）だが、Phase 3b
  では類似度クラスタリングを主機構とし、手動/自動トリガで consolidate する。

## 2. データ構造（Memory Fabric の拡張）

MemoryItem に追加:
- `provenance` に `'concept'` を追加（概念マーカー）。
- `count`（int）: 概念が統合したエピソード数（エピソードは 1）。

直列化に `count`(int32) を追加、MAGIC を更新（BAFN→BAFO）。bit 一致を再検証。

## 3. consolidate（src/memory/fabric.js に追加）

```
consolidate({ sim = 0.55, minCluster = 2 }) -> { concepts, clustered }
  1. 非概念エピソードを leader clustering（cos >= sim で先頭に吸着）でクラスタ化。
  2. size >= minCluster のクラスタごとに:
     - centroid = メンバ vector の平均 → L2 正規化
     - 概念 memory を write(centroid, {provenance:'concept', plasticity:0.05,
       retention:1.0, salience:1, text:'概念: <代表テキスト>'})、count=メンバ数
     - メンバ（エピソード）は plasticity=0.95, retention*=0.6 で薄れさせる
  3. 概念数とクラスタ化数を返す
```

- 概念は通常の recall に参加する（centroid は意味的に中心なのでそのテーマで
  よく想起される）。
- leader clustering は先頭要素との類似のみで単純。デモには十分（厳密な linkage は
  将来）。

## 4. 検証デモ（phase3b.html）

substrate（静的 deberta）で文を符号化 → 記銘 → consolidate。

1. 複数トピック（動物/天気/食事）の文を記銘（エピソード）。
2. **consolidate** → トピックごとに概念が立つ（メンバを表示）。
3. **想起**: 関連クエリで概念が返る（一般化した想起）。
4. **忘却（decay）を強くかける** → エピソードは retention が落ちて消えるが、
   概念（低可塑）は残る。再想起で概念が返る＝「経験は忘れ、概念は残る」。
5. 保存/復元/自己テスト（概念含め bit 一致）。

## 5. 受け入れ条件（DoD）

1. **[概念化]** 似た記憶群が正しくクラスタ化され、トピック概念が立つ。
2. **[一般化想起]** 関連クエリで概念が上位に返る。
3. **[忘却と両立]** 強い decay 後、エピソードは想起から脱落するが概念は残る。
4. **[永続化]** 概念含め Memory Fabric が bit 一致で保存・復元。
5. **[同居]** 概念は別構造でなく属性（provenance='concept', count）で区別される。

## 6. 非目標

- 概念の階層化（概念の概念）。将来。
- 概念に自然言語ラベルを付ける生成。テキストは代表エピソードの列挙で代替。
- Phase 3a との自動連携（surprise 低下→自動 consolidate）。手動トリガで実証し、
  自動化は後段。
