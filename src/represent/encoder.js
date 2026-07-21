// 自作の軽量エンコーダ（char単位・CBOW的オンライン自己教師あり学習）。
//
// 設計（docs/phase-1-spec.md）:
//   - トークン = Unicode コードポイント。外部トークナイザに依存しない。
//   - 埋め込み表 E は入出力タイド（word2vec 同様、入力埋め込みと出力分類器で共有）。
//   - encode(text) = 埋め込みの IDF 重み付け平均を L2 正規化。副作用なし。
//     （char 平均は助詞など高頻度文字に支配されるため、文書頻度 df から求めた
//      IDF で内容語を効かせる。SIF 系の考え方を char 単位に適用したもの。）
//   - learn(text) = CBOW を text 全体に適用し E をオンライン更新。語彙と df を成長。
//
// 注: Phase 1 の狙いは記憶機構の実証。深い意味理解は substrate（Phase 2）に委ねる。

const MAGIC = 0x42414532; // "BAE2"
const UNK = 0; // 予約: 未知文字インデックス

export class Encoder {
  constructor({ D = 64, window = 2, vocabCap = 4096, lr = 0.1, seed = 7 } = {}) {
    this.D = D;
    this.window = window;
    this.vocabCap = vocabCap;
    this.lr = lr;
    this.step = 0;
    this.vocab = new Map();          // codepoint -> index（index>=1）
    this.size = 1;                   // index0 = UNK を予約済み
    this.E = new Float32Array(vocabCap * D);
    this.df = new Int32Array(vocabCap); // 文書頻度（何件のテキストに出たか）
    this.docCount = 0;               // 学習で見たテキスト件数
    let s = seed >>> 0;
    const rnd = () => {
      s ^= s << 13; s >>>= 0; s ^= s >> 17; s ^= s << 5; s >>>= 0;
      return (s / 0xffffffff - 0.5) * 0.2;
    };
    for (let i = 0; i < this.E.length; i++) this.E[i] = rnd();
  }

  // IDF: 全件に出る文字ほど 0 に近づき、稀な内容語ほど大きい。
  idf(idx) {
    if (idx === UNK) return 0;
    return Math.log((1 + this.docCount) / (1 + this.df[idx]));
  }

  tokenizeRead(text) {
    const out = [];
    for (const ch of text) out.push(this.vocab.get(ch.codePointAt(0)) ?? UNK);
    return out;
  }

  tokenizeAssign(text) {
    const out = [];
    for (const ch of text) {
      const cp = ch.codePointAt(0);
      let idx = this.vocab.get(cp);
      if (idx === undefined) {
        if (this.size < this.vocabCap) { idx = this.size++; this.vocab.set(cp, idx); }
        else idx = (cp % (this.vocabCap - 1)) + 1;
      }
      out.push(idx);
    }
    return out;
  }

  // encode: IDF 重み付け平均 → L2 正規化。同一入力は同一出力（副作用なし）。
  encode(text) {
    const D = this.D;
    const toks = this.tokenizeRead(text);
    const v = new Float32Array(D);
    if (toks.length === 0) return v;
    let wsum = 0;
    for (const idx of toks) {
      const w = this.idf(idx);
      if (w <= 0) continue;
      const base = idx * D;
      for (let d = 0; d < D; d++) v[d] += w * this.E[base + d];
      wsum += w;
    }
    if (wsum > 0) for (let d = 0; d < D; d++) v[d] /= wsum;
    let norm = 0;
    for (let d = 0; d < D; d++) norm += v[d] * v[d];
    norm = Math.sqrt(norm) || 1;
    for (let d = 0; d < D; d++) v[d] /= norm;
    return v;
  }

  // CBOW: 各中心文字を文脈平均から予測。タイド重みで E を更新。df/docCount も更新。
  learn(text) {
    const D = this.D, w = this.window, lr = this.lr;
    const toks = this.tokenizeAssign(text);
    const n = toks.length;
    if (n < 2) return { loss: 0, centers: 0 };

    // 文書頻度更新（このテキストに出た一意な文字）
    const uniq = new Set(toks);
    for (const idx of uniq) this.df[idx]++;
    this.docCount++;

    let totalLoss = 0, centers = 0;
    for (let i = 0; i < n; i++) {
      const ctxIdx = [];
      for (let j = i - w; j <= i + w; j++) {
        if (j === i || j < 0 || j >= n) continue;
        ctxIdx.push(toks[j]);
      }
      if (ctxIdx.length === 0) continue;

      const ctx = new Float32Array(D);
      for (const c of ctxIdx) { const b = c * D; for (let d = 0; d < D; d++) ctx[d] += this.E[b + d]; }
      const invC = 1 / ctxIdx.length;
      for (let d = 0; d < D; d++) ctx[d] *= invC;

      const V = this.size;
      const logits = new Float32Array(V);
      let maxL = -Infinity;
      for (let v2 = 0; v2 < V; v2++) {
        const b = v2 * D; let dot = 0;
        for (let d = 0; d < D; d++) dot += ctx[d] * this.E[b + d];
        logits[v2] = dot; if (dot > maxL) maxL = dot;
      }
      let sum = 0;
      for (let v2 = 0; v2 < V; v2++) { logits[v2] = Math.exp(logits[v2] - maxL); sum += logits[v2]; }
      const target = toks[i];
      totalLoss += -Math.log(logits[target] / sum + 1e-12);
      centers++;

      const dlogits = logits;
      for (let v2 = 0; v2 < V; v2++) dlogits[v2] /= sum;
      dlogits[target] -= 1;

      const gCtx = new Float32Array(D);
      for (let v2 = 0; v2 < V; v2++) {
        const dv = dlogits[v2]; if (dv === 0) continue;
        const b = v2 * D; for (let d = 0; d < D; d++) gCtx[d] += dv * this.E[b + d];
      }
      for (let v2 = 0; v2 < V; v2++) {
        const dv = dlogits[v2]; if (dv === 0) continue;
        const b = v2 * D; const g = lr * dv;
        for (let d = 0; d < D; d++) this.E[b + d] -= g * ctx[d];
      }
      const gin = lr * invC;
      for (const c of ctxIdx) { const b = c * D; for (let d = 0; d < D; d++) this.E[b + d] -= gin * gCtx[d]; }
      this.step++;
    }
    return { loss: centers ? totalLoss / centers : 0, centers };
  }

  // --- 直列化（使用中の行のみ。bit 一致復元）--------------------------------
  serialize() {
    const D = this.D, size = this.size;
    const buf = new ArrayBuffer(28 + size * 4 + size * 4 + size * D * 4);
    const dv = new DataView(buf);
    dv.setInt32(0, MAGIC, true);
    dv.setInt32(4, D, true);
    dv.setInt32(8, this.window, true);
    dv.setInt32(12, this.vocabCap, true);
    dv.setInt32(16, size, true);
    dv.setInt32(20, this.step, true);
    dv.setInt32(24, this.docCount, true);
    const cpByIndex = new Int32Array(size);
    for (const [cp, idx] of this.vocab) cpByIndex[idx] = cp;
    let off = 28;
    for (let i = 0; i < size; i++) { dv.setInt32(off, cpByIndex[i], true); off += 4; }
    for (let i = 0; i < size; i++) { dv.setInt32(off, this.df[i], true); off += 4; }
    new Float32Array(buf, off, size * D).set(new Float32Array(this.E.buffer, 0, size * D));
    return buf;
  }

  static deserialize(buf) {
    const dv = new DataView(buf);
    if (dv.getInt32(0, true) !== MAGIC) throw new Error('Encoder MAGIC 不一致');
    const D = dv.getInt32(4, true);
    const window = dv.getInt32(8, true);
    const vocabCap = dv.getInt32(12, true);
    const size = dv.getInt32(16, true);
    const step = dv.getInt32(20, true);
    const docCount = dv.getInt32(24, true);
    const enc = new Encoder({ D, window, vocabCap });
    enc.size = size; enc.step = step; enc.docCount = docCount; enc.vocab = new Map();
    let off = 28;
    for (let i = 0; i < size; i++) { const cp = dv.getInt32(off, true); off += 4; if (i > 0) enc.vocab.set(cp, i); }
    enc.df = new Int32Array(vocabCap);
    for (let i = 0; i < size; i++) { enc.df[i] = dv.getInt32(off, true); off += 4; }
    enc.E = new Float32Array(vocabCap * D);
    new Float32Array(enc.E.buffer, 0, size * D).set(new Float32Array(buf, off, size * D));
    return enc;
  }
}
