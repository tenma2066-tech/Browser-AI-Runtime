// Phase 4: AI Reporter。これまでの層（Representation / Memory / 概念化 / 質問 /
// 忘却 / 永続化）を統合し、ユーザーの入力からテーマ・注目の出来事・質問を返す。
// 文章生成はしない（強みは記憶・概念・想起）。
//
// 注: 文モデル埋め込みは異方性が強い（全ペアが高 cos）。novelty も概念化と同じく
// 「平均除去した空間」で測る（そうしないと全エントリが同じ novelty になる）。

import { MemoryFabric } from '../memory/fabric.js';

const cos = (a, b) => { let d = 0; for (let i = 0; i < a.length; i++) d += a[i] * b[i]; return d; };

export class AIReporter {
  // enc: { dim, encode(text)->Float32Array }
  constructor(backend, enc, { lambda = 0.08, consolidateSim = 0.06, questionNovelty = 0.65 } = {}) {
    this.enc = enc;
    this.D = enc.dim;
    this.backend = backend;
    this.fab = new MemoryFabric(backend, { D: this.D, lambda });
    this.consolidateSim = consolidateSim;
    this.questionNovelty = questionNovelty;
  }

  // エピソード集合の平均で中心化したベクトル群を返す（novelty 測定用）。
  _centered() {
    const D = this.D;
    const eps = this.fab.items.filter((x) => x.provenance !== 'concept');
    const mean = new Float32Array(D);
    for (const e of eps) for (let d = 0; d < D; d++) mean[d] += e.vector[d] / (eps.length || 1);
    const cen = eps.map((e) => this._center(e.vector, mean));
    return { eps, mean, cen };
  }

  _center(v, mean) {
    const D = this.D, o = new Float32Array(D); let n = 0;
    for (let d = 0; d < D; d++) { o[d] = v[d] - mean[d]; n += o[d] * o[d]; }
    n = Math.sqrt(n) || 1; for (let d = 0; d < D; d++) o[d] /= n; return o;
  }

  // エントリを記録。中心化空間での新規性を salience にし、新規なら質問を返す。
  ingest(text) {
    const v = this.enc.encode(text);
    const { eps, mean, cen } = this._centered();
    let novelty = 1;
    if (eps.length >= 2) {
      const cv = this._center(v, mean);
      let maxc = -1; for (const ce of cen) { const c = cos(cv, ce); if (c > maxc) maxc = c; }
      novelty = Math.max(0, Math.min(1, (1 - maxc) / 1.2)); // 中心化cos≈[-0.2,0.5]を[0,1]へ寄せる
    }
    // 注目度（novelty）が高い出来事ほど可塑性を下げる＝忘れにくくする。
    // 感情的に重要な記憶が残るのと同じ。平凡な出来事ほど速く薄れる。
    const plasticity = Math.max(0.3, Math.min(0.9, 0.9 - 0.5 * novelty));
    this.fab.write(v, { text, salience: novelty, plasticity });
    let question = null;
    if (eps.length >= 3 && novelty >= this.questionNovelty) {
      question = { text, novelty, prompt: `「${text}」——これまでに無い新しい話題ですね。もう少し教えてください。` };
    }
    return { novelty, question };
  }

  consolidate() { return this.fab.consolidate({ sim: this.consolidateSim }); }

  passTime(dt = 5) { this.fab.decay(dt); }

  report() {
    const concepts = this.fab.items.filter((x) => x.provenance === 'concept' && x.retention > 0.1)
      .sort((a, b) => b.count - a.count);
    const eps = this.fab.items.filter((x) => x.provenance !== 'concept' && x.retention > 0.15)
      .sort((a, b) => b.salience * b.retention - a.salience * a.retention);
    return {
      themes: concepts.map((c) => ({ text: c.text.replace('概念: ', ''), count: c.count, retention: c.retention })),
      notable: eps.slice(0, 5).map((e) => ({ text: e.text, salience: e.salience })),
      stats: { entries: eps.length, concepts: concepts.length, total: this.fab.items.length },
    };
  }

  serialize() { return this.fab.serialize(); }
  loadFabric(buf) { this.fab = MemoryFabric.deserialize(buf, this.backend); }
}
