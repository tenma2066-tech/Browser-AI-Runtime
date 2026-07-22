# Phase 2c 結果記録（substrate を Core の予測に統合）

- 実施日: 2026-07-22
- 実機: iPhone / iOS Safari / standalone PWA
- 前提: `docs/phase-2c-spec.md`
- 状態: **iOS 実機で動作。粒度を subword に統一、Core が substrate 意味空間で予測。**

## 判定

| 受け入れ条件 | 結果 | 実測 |
|---|---|---|
| #1 学習（surprise 低下） | ✅ | 24ep で surprise 1.693 → 0.233（ln V=10.37 から） |
| #2 意味的予測 | ✅ | 予測が substrate 空間で行われる（下記）。Node で「雨が→降/空/雲」等を確認 |
| #3 粒度統一 | ✅ | 入力・予測とも subword、出力ヘッド＝凍結 substrate 埋め込み |
| #4 永続化 bit 一致 | ✅ | core=PASS（実機） |
| #5 速度 | ✅ | iOS 約 7ms/step（450ms/ep, 8文×~8step）。Node 20ms/step より速い |

## 設計の到達点

- Core の出力層を新設せず、**凍結 substrate 埋め込み行列をそのまま分類器**として
  使う（`logits[v] = z·substrate.row(v)`）。学習対象は Wxh/Whh/bh/Wproj のみ
  （≈6.6万 params）。予測が substrate の意味空間で行われる。
- これで **入力・予測・想起がすべて subword/substrate 空間で一貫**。ADR-0001 の
  「substrate は Core に駆動される器官」が、予測ループの内側で実現した。

## 正直な限界

- tiny corpus のため、コーパス外の文脈（例「山が」）では学習済み継続
  （窓/公園/降/空/雲）にフォールバックする。意味空間の出力ヘッドは意味的な
  漏れ出し（関連 subword への確率）を与えるが、少データでは限定的。
- 実用的な意味的汎化には、より多い経験（連続学習の積み重ね）か、より大きい
  substrate が要る。メカニズム自体は正しく機能している。

## 次アクション

- **Phase 3b（概念化 / consolidate）**: 似た記憶を統合して抽象概念を作る。
  馴染んだ（surprise が下がった）記憶群を概念化候補にし、代表ベクトルへ統合。
  個別エピソードは薄れても概念は残る（忘却と両立した一般化）。
