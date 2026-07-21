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

**Phase 0（技術検証）完全合格**（2026-07-21、iPhone 実機で確認）。iOS standalone
PWA で WebGPU 学習ループが回り、OPFS で bit 一致の永続化・復元ができることを実証。
結果は [`docs/phase-0-results.md`](docs/phase-0-results.md)。次は Phase 1。

## ドキュメント

- [`docs/ADR-0001-foundations.md`](docs/ADR-0001-foundations.md) —
  基盤決定の記録（3本の柱・substrate の位置づけ・レイヤーのデータ契約・
  既知リスク・ロードマップ）
- [`docs/phase-0-spec.md`](docs/phase-0-spec.md) —
  Phase 0（技術検証スパイク）の仕様・受け入れ条件・ファイル構成

## ロードマップ

```
Phase 0  技術検証: iOS PWA で GPU 学習ループ + 永続化/復元を1本通す  ✅ 合格
Phase 1  Runtime 抽象 + Representation(encode) + Memory Fabric 最小  ← いまここ
Phase 2  Cognitive Core（1ステップ予測 + オンライン更新）
Phase 3  Meta Cognition（予測誤差ベースの制御）
Phase 4  AI Reporter を最初のアプリとして載せる
```
