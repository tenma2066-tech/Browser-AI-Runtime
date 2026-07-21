# Phase 2b-2 結果記録（自前 e5-small forward）

- 実施日: 2026-07-21
- 実機: iPhone / iOS Safari / standalone PWA（`backend = webgpu`）
- 前提: `docs/phase-2b2-spec.md`
- 状態: **iOS 実機で動作。12層 sentence-transformer を自前実装で完全ローカル実行。**

## 判定

| 受け入れ条件 | 結果 | 実測 |
|---|---|---|
| #1 サイズ（iOS 搭載） | ✅ | 剪定＋int8 で **31MB**、iOS で読込 6.9秒 |
| #2 forward 正しさ | ✅ | 意味的想起が機能。int8/fp32 で結果一致（実装の妥当性を裏付け） |
| #3 抽象クエリ改善 | ✅（部分） | 静的 1/4 → 文脈化で改善（下記）。ただし e5-small の天井あり |
| #4 iOS 動作 | ✅ | 符号化 **263ms/文**、query encode ~230ms（Node 600ms より速い） |
| #5 非依存 | ✅ | トークナイザ・forward とも自前。外部ライブラリなし |

## iOS の数字

- 31MB を 6.9 秒で読込、語彙 20763 / dim 384 / 12層。
- 自前 forward の符号化 **263ms/文**（iOS の JIT が効き Node より速い）。
- メモリ問題なし（層重みは int8 保持＋forward 内で行ごと逆量子化、~30MB）。

## 品質: 静的からの改善と、e5-small の天井

改善した例（静的 substrate では全滅していたもの）:
- 「かわいいペット」→ 犬/猫、「電車で移動」→ 新幹線、「天気がいい」→ 天気予報。

外す例:
- 「おなかが空いた」→ 傘（カレーは2位）、「楽しい旅行」→ 傘。

### 決定的な切り分け: 量子化は原因ではない

層重みを **int8 と fp32 で比較**したところ、想起結果・cosine 幅ともほぼ同一
（2/4、平均cosine幅 0.034 vs 0.035）。→ **圧縮も取りこぼしも int8 量子化ではなく
e5-small（小型モデル）＋ mean-pooling ＋ 少数コーパスの本質的天井**。よって
fp16/fp32（86MB）を配信しても改善せず、**31MB int8 のままが最適**。

- cosine が 0.86〜0.92 に圧縮（e5 の異方性）。特定の文（「傘を持って出かけた」）が
  ハブのように無関係クエリ上位に出る hub 現象。SIF/共通成分除去（Phase 2b-2 前段で
  実験）でも解消しなかった。

## 意味

- **12層 BERT sentence-transformer を、ライブラリ一切なし・ゼロから JS 実装し、
  iPhone で完全ローカル実行**（トークナイザ SentencePiece Viterbi・attention・
  LayerNorm・GELU すべて自前）。速度も実用域（263ms/文）。
- 制約（サーバー/API/クラウド禁止・無依存）を全て守ったまま、実在の事前学習
  Transformer を動かした。substrate 構想（ADR-0001）の最終形が実物になった。

## 正直な結論

- Phase 2b-2 の目的「自前 forward が動く＋抽象意味が静的より改善」は達成。
- ただし e5-small は小型で、難しい抽象クエリは外す。**さらなる品質は「より大きい/
  文特化の sentence モデル」を substrate にするしかない**（サイズと速度のトレードオフ）。
- 実用の観点では、substrate は「具体・語彙意味＋一部の抽象意味」に強い凍結器官として
  十分。Cognitive Core / Memory / Meta 認知（研究の本体）に注力するのが妥当。

## 次アクションの候補

- **Phase 3（Meta Cognition）**: surprise 統計の監視で学習率制御・質問。研究の本体。
- **Phase 2c**: substrate を Core の予測に統合（粒度設計）。
- substrate の品質向上（より大きい文モデル）は将来のオプション。
