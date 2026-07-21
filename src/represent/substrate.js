// Phase 2b: 凍結 substrate（事前学習済みサブワード埋め込み言語器官）。
//
// 実モデル ku-nlp/deberta-v2-tiny-japanese から抽出した int8 埋め込みと
// SentencePiece(unigram) 語彙を読み込み、encode(text)->vector を提供する。
// フルの Transformer forward は載せない（lookup + pooling のみ）。凍結（学習しない）。
//
// Encoder と同じ encode(text)->Float32Array 契約に合わせ、Representation に
// ドロップインできる。トークナイズは自前の unigram Viterbi（外部ライブラリ非依存）。

export class Substrate {
  // data: { dim, i8:Int8Array, scale:Float32Array, pieces:string[], scores:number[],
  //         unkId:number, metaspace:string }
  constructor(data) {
    this.dim = data.dim;
    this.i8 = data.i8;
    this.scale = data.scale;
    this.pieces = data.pieces;
    this.scores = data.scores;
    this.unkId = data.unkId;
    this.metaspace = data.metaspace || '▁';
    this.frozen = true;
    // piece -> id の索引と最大ピース長
    this.index = new Map();
    let maxLen = 1;
    for (let i = 0; i < data.pieces.length; i++) {
      this.index.set(data.pieces[i], i);
      const L = [...data.pieces[i]].length;
      if (L > maxLen && L <= 20) maxLen = L;
    }
    this.maxLen = maxLen;
  }

  // 逆量子化した埋め込み行（dim 次元）
  row(id) {
    const dim = this.dim, s = this.scale[id], out = new Float32Array(dim);
    const base = id * dim;
    for (let d = 0; d < dim; d++) out[d] = this.i8[base + d] * s;
    return out;
  }

  // SentencePiece unigram の Viterbi 分割。戻り値 = token id 列。
  tokenize(text) {
    const norm = text.normalize('NFKC').replace(/\s+/g, ' ');
    const withMeta = this.metaspace + norm.replace(/ /g, this.metaspace);
    const chars = [...withMeta];
    const n = chars.length;
    const NEG = -1e12;
    // unk は「どのピースにも当たらない場合の最終手段」。実ピースのスコア範囲
    // （概ね -22〜0）より十分低くし、語彙にある文字を必ず優先させる。
    const UNK_PENALTY = -1e4;
    const dp = new Float64Array(n + 1).fill(NEG);
    dp[0] = 0;
    const back = new Array(n + 1).fill(null);
    for (let i = 0; i < n; i++) {
      if (dp[i] === NEG) continue;
      const lim = Math.min(this.maxLen, n - i);
      let matched = false;
      for (let L = lim; L >= 1; L--) {
        const sub = chars.slice(i, i + L).join('');
        const id = this.index.get(sub);
        if (id !== undefined) {
          const sc = dp[i] + this.scores[id];
          if (sc > dp[i + L]) { dp[i + L] = sc; back[i + L] = { from: i, id }; }
          matched = true;
        }
      }
      // どのピースにも当たらない1文字は unk として進める
      const scU = dp[i] + UNK_PENALTY;
      if (scU > dp[i + 1]) { dp[i + 1] = scU; back[i + 1] = { from: i, id: this.unkId }; }
      void matched;
    }
    // 復元
    const ids = [];
    let i = n;
    while (i > 0 && back[i]) { ids.push(back[i].id); i = back[i].from; }
    ids.reverse();
    return ids;
  }

  // encode: サブワード埋め込みの平均プーリング → L2 正規化（凍結）。
  encode(text) {
    const dim = this.dim, v = new Float32Array(dim);
    const ids = this.tokenize(text);
    let cnt = 0;
    for (const id of ids) {
      if (id === this.unkId) continue; // unk は情報が薄いので除外
      const s = this.scale[id], base = id * dim;
      for (let d = 0; d < dim; d++) v[d] += this.i8[base + d] * s;
      cnt++;
    }
    if (cnt > 0) for (let d = 0; d < dim; d++) v[d] /= cnt;
    let norm = 0; for (let d = 0; d < dim; d++) norm += v[d] * v[d];
    norm = Math.sqrt(norm) || 1;
    for (let d = 0; d < dim; d++) v[d] /= norm;
    return v;
  }

  // 語彙内の最近傍（デバッグ・可視化用）。text の代表トークンに近いピースを返す。
  nearestPieces(text, k = 5) {
    const q = this.encode(text);
    const scored = [];
    for (let id = 0; id < this.pieces.length; id++) {
      const r = this.row(id);
      let dot = 0, nr = 0; for (let d = 0; d < this.dim; d++) { dot += q[d] * r[d]; nr += r[d] * r[d]; }
      scored.push([this.pieces[id], dot / (Math.sqrt(nr) || 1)]);
    }
    scored.sort((a, b) => b[1] - a[1]);
    return scored.slice(0, k);
  }
}

// ブラウザ用ローダ: /models/... から fetch して Substrate を構築。
export async function loadSubstrate(baseUrl) {
  const j = async (p) => (await fetch(baseUrl + p)).json();
  const b = async (p) => (await fetch(baseUrl + p)).arrayBuffer();
  const meta = await j('meta.json');
  const tk = await j('tokenizer.json');
  const i8 = new Int8Array(await b('emb.i8'));
  const scale = new Float32Array(await b('emb.scale'));
  return new Substrate({
    dim: meta.dim, i8, scale,
    pieces: tk.pieces, scores: tk.scores, unkId: tk.unk_id, metaspace: tk.metaspace,
  });
}
