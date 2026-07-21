// Phase 2b-2: multilingual-e5-small の forward を自前実装した sentence-embedding
// substrate。vanilla BERT（標準 self-attention・LN・GELU）を JS で計算し、
// 最終層 hidden の mean pooling → L2 正規化で文ベクトルを返す。外部ライブラリ非依存。
//
// トークナイズは Unigram SentencePiece の Viterbi（substrate.js と同方式）。
// E5 規約: クエリに "query: "、文書に "passage: " の接頭辞。

const erf = (x) => {
  const t = 1 / (1 + 0.3275911 * Math.abs(x));
  const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
  return x >= 0 ? y : -y;
};
const gelu = (x) => 0.5 * x * (1 + erf(x / Math.SQRT2));

export class E5 {
  // cfg: { meta, tk, embI8:Int8Array, embScale:Float32Array, tensors:Map<name,{data:Float32Array,shape}> }
  constructor(cfg) {
    this.meta = cfg.meta;
    this.H = cfg.meta.hidden; this.L = cfg.meta.layers; this.heads = cfg.meta.heads;
    this.headDim = this.H / this.heads; this.inter = cfg.meta.intermediate;
    this.embI8 = cfg.embI8; this.embScale = cfg.embScale;
    this.T = cfg.tensors;
    this.tk = cfg.tk;
    this.eps = 1e-12;
    // トークナイザ索引
    this.index = new Map(); this.maxLen = 1;
    for (let i = 0; i < cfg.tk.pieces.length; i++) {
      this.index.set(cfg.tk.pieces[i], i);
      const l = [...cfg.tk.pieces[i]].length; if (l > this.maxLen && l <= 20) this.maxLen = l;
    }
  }

  t(name) { return this.T.get(name).data; } // f32 テンソル（bias/LN/pos/type）用

  tokenize(text) {
    const norm = ('▁' + text.normalize('NFKC').replace(/\s+/g, ' ').replace(/ /g, '▁'));
    const chars = [...norm]; const n = chars.length;
    const NEG = -1e12, UNK_PEN = -1e4;
    const dp = new Float64Array(n + 1).fill(NEG); dp[0] = 0;
    const back = new Array(n + 1).fill(null);
    for (let i = 0; i < n; i++) {
      if (dp[i] === NEG) continue;
      const lim = Math.min(this.maxLen, n - i);
      for (let L = lim; L >= 1; L--) {
        const id = this.index.get(chars.slice(i, i + L).join(''));
        if (id !== undefined) { const sc = dp[i] + this.tk.scores[id]; if (sc > dp[i + L]) { dp[i + L] = sc; back[i + L] = { from: i, id }; } }
      }
      const u = dp[i] + UNK_PEN; if (u > dp[i + 1]) { dp[i + 1] = u; back[i + 1] = { from: i, id: this.tk.unk }; }
    }
    const ids = []; let i = n; while (i > 0 && back[i]) { ids.push(back[i].id); i = back[i].from; }
    ids.reverse();
    return [this.tk.bos, ...ids, this.tk.eos];
  }

  // 線形: Y[L,out] = X[L,in]·W[out,in]^T + b
  // W は f32 または int8 per-row（iOS メモリ節約のため int8 のまま行内で逆量子化）。
  linear(X, Ln, inDim, wName, bName, outDim) {
    const w = this.T.get(wName), b = bName ? this.t(bName) : null;
    const Y = new Float32Array(Ln * outDim);
    if (w.i8) {
      const W = w.i8, SC = w.scale;
      for (let l = 0; l < Ln; l++) {
        const xb = l * inDim, yb = l * outDim;
        for (let o = 0; o < outDim; o++) {
          let s = 0; const wb = o * inDim;
          for (let i = 0; i < inDim; i++) s += X[xb + i] * W[wb + i];
          Y[yb + o] = s * SC[o] + (b ? b[o] : 0);
        }
      }
    } else {
      const W = w.data;
      for (let l = 0; l < Ln; l++) {
        const xb = l * inDim, yb = l * outDim;
        for (let o = 0; o < outDim; o++) {
          let s = b ? b[o] : 0; const wb = o * inDim;
          for (let i = 0; i < inDim; i++) s += X[xb + i] * W[wb + i];
          Y[yb + o] = s;
        }
      }
    }
    return Y;
  }

  layernorm(X, Ln, dim, gName, bName) {
    const g = this.t(gName), b = this.t(bName); const Y = new Float32Array(Ln * dim);
    for (let l = 0; l < Ln; l++) {
      const base = l * dim; let m = 0; for (let d = 0; d < dim; d++) m += X[base + d]; m /= dim;
      let v = 0; for (let d = 0; d < dim; d++) { const z = X[base + d] - m; v += z * z; } v /= dim;
      const inv = 1 / Math.sqrt(v + this.eps);
      for (let d = 0; d < dim; d++) Y[base + d] = (X[base + d] - m) * inv * g[d] + b[d];
    }
    return Y;
  }

  encode(text, kind = 'passage') {
    const H = this.H, heads = this.heads, hd = this.headDim;
    const prefix = kind === 'query' ? this.meta.e5_prefix.query : this.meta.e5_prefix.passage;
    const ids = this.tokenize(prefix + text);
    const Ln = ids.length;

    // 埋め込み: word + position + token_type(0) → LayerNorm
    const posE = this.t('embeddings.position_embeddings.weight');
    const typE = this.t('embeddings.token_type_embeddings.weight');
    let X = new Float32Array(Ln * H);
    for (let l = 0; l < Ln; l++) {
      const id = ids[l], sc = this.embScale[id], eb = id * H, xb = l * H, pb = l * H;
      for (let d = 0; d < H; d++) X[xb + d] = this.embI8[eb + d] * sc + posE[pb + d] + typE[d];
    }
    X = this.layernorm(X, Ln, H, 'embeddings.LayerNorm.weight', 'embeddings.LayerNorm.bias');

    for (let layer = 0; layer < this.L; layer++) {
      const p = `encoder.layer.${layer}.`;
      const Q = this.linear(X, Ln, H, p + 'attention.self.query.weight', p + 'attention.self.query.bias', H);
      const K = this.linear(X, Ln, H, p + 'attention.self.key.weight', p + 'attention.self.key.bias', H);
      const V = this.linear(X, Ln, H, p + 'attention.self.value.weight', p + 'attention.self.value.bias', H);
      const ctx = new Float32Array(Ln * H);
      const scale = 1 / Math.sqrt(hd);
      for (let h = 0; h < heads; h++) {
        const off = h * hd;
        for (let i = 0; i < Ln; i++) {
          const scores = new Float64Array(Ln); let mx = -Infinity;
          for (let j = 0; j < Ln; j++) {
            let s = 0; const qi = i * H + off, kj = j * H + off;
            for (let d = 0; d < hd; d++) s += Q[qi + d] * K[kj + d];
            s *= scale; scores[j] = s; if (s > mx) mx = s;
          }
          let Z = 0; for (let j = 0; j < Ln; j++) { scores[j] = Math.exp(scores[j] - mx); Z += scores[j]; }
          const cb = i * H + off;
          for (let j = 0; j < Ln; j++) { const w = scores[j] / Z; const vj = j * H + off; for (let d = 0; d < hd; d++) ctx[cb + d] += w * V[vj + d]; }
        }
      }
      let att = this.linear(ctx, Ln, H, p + 'attention.output.dense.weight', p + 'attention.output.dense.bias', H);
      for (let k = 0; k < Ln * H; k++) att[k] += X[k]; // 残差
      att = this.layernorm(att, Ln, H, p + 'attention.output.LayerNorm.weight', p + 'attention.output.LayerNorm.bias');
      // FFN
      let inter = this.linear(att, Ln, H, p + 'intermediate.dense.weight', p + 'intermediate.dense.bias', this.inter);
      for (let k = 0; k < Ln * this.inter; k++) inter[k] = gelu(inter[k]);
      let out = this.linear(inter, Ln, this.inter, p + 'output.dense.weight', p + 'output.dense.bias', H);
      for (let k = 0; k < Ln * H; k++) out[k] += att[k]; // 残差
      X = this.layernorm(out, Ln, H, p + 'output.LayerNorm.weight', p + 'output.LayerNorm.bias');
    }

    // mean pooling → L2
    const v = new Float32Array(H);
    for (let l = 0; l < Ln; l++) { const b = l * H; for (let d = 0; d < H; d++) v[d] += X[b + d]; }
    for (let d = 0; d < H; d++) v[d] /= Ln;
    let nrm = 0; for (let d = 0; d < H; d++) nrm += v[d] * v[d]; nrm = Math.sqrt(nrm) || 1;
    for (let d = 0; d < H; d++) v[d] /= nrm;
    return v;
  }
}

// レイヤーテンソルを層メタから展開。int8 は f32 に展開せず int8+scale のまま保持し、
// forward 内（linear）で行ごとに逆量子化する（iOS メモリ節約のため）。
export function buildTensors(layersBin, layersMeta) {
  const T = new Map();
  for (const e of layersMeta) {
    if (e.dtype === 'f32') {
      T.set(e.name, { data: new Float32Array(layersBin.slice(e.dataOffset, e.dataOffset + e.dataLen)), shape: e.shape });
    } else { // i8 per-row: そのまま保持
      const i8 = new Int8Array(layersBin, e.dataOffset, e.dataLen);
      const scale = new Float32Array(layersBin.slice(e.scaleOffset, e.scaleOffset + e.scaleLen));
      T.set(e.name, { i8, scale, shape: e.shape });
    }
  }
  return T;
}

export async function loadE5(baseUrl) {
  const j = async (p) => (await fetch(baseUrl + p)).json();
  const ab = async (p) => (await fetch(baseUrl + p)).arrayBuffer();
  const meta = await j('meta.json'); const tk = await j('tokenizer.json');
  const embI8 = new Int8Array(await ab('emb.i8'));
  const embScale = new Float32Array(await ab('emb.scale'));
  const layersBin = await ab('layers.bin'); const layersMeta = await j('layers.meta.json');
  return new E5({ meta, tk, embI8, embScale, tensors: buildTensors(layersBin, layersMeta) });
}
