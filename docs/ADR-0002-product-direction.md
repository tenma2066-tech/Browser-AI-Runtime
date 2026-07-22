# ADR-0002: 最終目的の明確化と「大型 substrate を GitHub に置く」方針

- ステータス: 採択
- 日付: 2026-07-22
- 関連: ADR-0001（基盤決定）

## 背景

Phase 0〜3b で、認知の骨格（Runtime / Representation / Memory Fabric /
Cognitive Core / Meta Cognition / 概念化）が iPhone 実機で一通り動いた。ここで
最終目的を再確認し、substrate の規模方針を更新する。

## 最終目的（再確認・明確化）

- 作るのは **ユーザーごとに成長する AI を提供する PWA アプリ**（実ユーザー向けの
  製品。単なる研究ランタイムに留めない）。開発は AI（Claude Code）で行う。
- **役割分担を明確化する:**
  - **GitHub 側（全ユーザー共通・凍結）**: 大型のベースモデル（substrate＝言語
    器官）と Pack 群（Personality / Goal / Knowledge / Reasoning Rules）。
  - **ローカル側（ユーザーごと・可塑）**: 適応重み（Cognitive Core / Meta）、
    Memory Fabric（エピソード / 概念 / 人格）、経験ログ。**ここが成長する部分**で、
    IndexedDB / OPFS に保存する。
- 最初のアプリは **AI Reporter**。

## 方針更新: substrate は「大型」でよい（GitHub に置く）

ADR-0001 では iOS 制約から substrate を小型（deberta-tiny 7MB / e5-small 31MB）に
抑えていた。本 ADR で **大型モデルを GitHub に置いて配信する**方針に更新する。
ただし以下の制約を正直に踏まえる。

### 位置づけ: ローカル優先 + 大型は任意の上位層（Apple Intelligence 的）

- **基本はオンデバイス実行**（Apple Intelligence 的）。端末で普通に動く現実的
  サイズのモデルを既定とし、これだけで製品は成立する。
- **GitHub の大型モデルは任意の上位層**。載る端末では fetch → ローカル実行し、
  載らない端末では使わない（無くても既定のオンデバイスで動く）。将来はオンデバイス
  版の「蒸留元」としても使う。
- 正直な注意: Apple Intelligence の ~3B オンデバイスモデルは **OS が特権的に
  メモリを扱える**ために動く。**PWA（Safari タブ）は同じ土俵に立てず**、我々の
  「オンデバイス既定」は Apple のものより小さくなる。過大な期待は置かない。

### 制約1: iOS Safari のメモリ天井

- iOS の standalone PWA はタブ単位のメモリ上限がある（近年の iPhone で概ね 1〜2GB
  程度、機種依存・不確実）。**「大型」の現実的上限は量子化後で数百MB級**と見込む。
  数GB級は Safari で保持・実行が難しい可能性が高い。
- → **対象機種で「読み込み・保持・実行できる最大モデルサイズ」を実機スパイクで
  確定する**（Phase 0 と同じやり方）。設計はスパイク結果に従う。

### 制約2: GitHub でのホスティング

- GitHub Pages: リポジトリ soft 上限 ~1GB、単一ファイル 100MB（Git）、Pages 帯域
  soft ~100GB/月。
- Git LFS: 無料枠は帯域 1GB/月と小さく、多数ユーザーへの配信には不向き。
- **推奨: 大型モデル重みは GitHub Releases のアセットとして置く**（単一ファイル
  2GB まで、CDN 配信、LFS 帯域を消費しない）。PWA は初回に fetch し、**OPFS に
  キャッシュ**して 2 回目以降はローカルから読む。これで「サーバー禁止」を守る。

### 制約3: 大型モデルの forward は WebGPU が要る

- 自前 JS forward は e5-small（12層）で 263ms/文（iOS 実測）。**より大型のモデルを
  JS で回すと遅すぎる**。大型 substrate を実用速度で動かすには、Phase 0 で用意した
  Runtime 抽象の **WebGPU 経路（テンソル常駐 forward）** への投資が必要になる。
- → 「大型 substrate」は「WebGPU forward」とセットの大きめの投資。ロードマップに
  独立フェーズとして置く。

## ロードマップへの含意

- **Phase 4（AI Reporter）**: まず**現状の小型 substrate のまま**、これまでの層を
  統合した実アプリの形を作る（製品の体験を先に固める）。大型化は待たない。
- **Phase 5（大型 substrate + WebGPU forward）**: 言語品質を本格的に上げる infra
  投資。サブステップ:
  1. iOS 最大モデルサイズの実機スパイク
  2. GitHub Releases でのモデル配信 + OPFS キャッシュ
  3. WebGPU による大型モデル forward
- **継続テーマ**: Pack（GitHub）の読み込み、ユーザー成長物のローカル永続化（既に
  blobStore / OPFS で基盤あり）。

## 変わらないもの（ADR-0001 の芯）

- 完全ローカル実行（推論はユーザー端末）・サーバー禁止・AI API 禁止。
- substrate は「差し替え可能な凍結言語器官」、認知の主権は Cognitive Core。
- 忘却は設計された機能。可塑性で短期/長期/知識/人格/概念を連続区別。
