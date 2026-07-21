# Browser AI Runtime

HTML / CSS / JavaScript(TypeScript) のみで動く、**完全クライアントサイドの
AI 実行基盤**を研究・実装するプロジェクト。

- 対象は **iPhone / Safari / PWA を最優先**。
- **サーバー禁止・AI API 禁止・クラウド推論禁止**。完全ローカル実行を目指す。
- ChatGPT のコピーでも、既存 LLM のラッパーでもない。Transformer から学びつつ、
  ブラウザ・WebGPU・単一ユーザー・継続学習という条件に最適化した新しい AI
  実行基盤を設計する。
- 最終目標は Runtime そのものの完成。最初のアプリとして **AI Reporter** を予定。

## ステータス

Phase 0（技術検証）に着手する段階。まだ実装コードは書いていない。まず設計を
固める。

## ドキュメント

- [`docs/ADR-0001-foundations.md`](docs/ADR-0001-foundations.md) —
  基盤決定の記録（3本の柱・substrate の位置づけ・レイヤーのデータ契約・
  既知リスク・ロードマップ）
- [`docs/phase-0-spec.md`](docs/phase-0-spec.md) —
  Phase 0（技術検証スパイク）の仕様・受け入れ条件・ファイル構成

## ロードマップ

```
Phase 0  技術検証: iOS PWA で GPU 学習ループ + 永続化/復元を1本通す  ← いまここ
Phase 1  Runtime 抽象 + Representation(encode) + Memory Fabric 最小
Phase 2  Cognitive Core（1ステップ予測 + オンライン更新）
Phase 3  Meta Cognition（予測誤差ベースの制御）
Phase 4  AI Reporter を最初のアプリとして載せる
```
