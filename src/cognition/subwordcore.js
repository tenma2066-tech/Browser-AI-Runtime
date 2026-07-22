// Phase 2c: Subword Cognitive Core。substrate（静的 deberta, 192次元・32k語彙）を
// 「凍結入力埋め込み」かつ「凍結出力ヘッド」として使う次 subword 予測器。
//
//   x_t = substrate.row(subword_t)                      … 凍結入力
//   h_t = tanh(Wxh·x + Whh·h_{t-1} + bh)                … 再帰状態
//   z   = Wproj·h_t                                     … substrate 空間への射影
//   logits[v] = z · substrate.row(v)  （全 subword）    … 凍結埋め込みで分類
//   p = softmax(logits), surprise = -log p[target]
//
// 学習対象は Wxh/Whh/bh/Wproj のみ。巨大な出力層を新設しない。予測が substrate の
// 意味空間で行われるため、確率が意味的に近い subword に集まる（char Core にない性質）。

const MAGIC = 0x42415343; // "BASC"

export class SubwordCore {
  constructor({ inDim = 192, H = 128, lr = 0.05, seed = 99 } = {}) {
    this.inDim = inDim; this.H = H; this.lr = lr; this.step = 0;
    this.Wxh = new Float32Array(H * inDim);
    this.Whh = new Float32Array(H * H);
    this.bh = new Float32Array(H);
    this.Wproj = new Float32Array(inDim * H);
    this.h = new Float32Array(H);
    let s = seed >>> 0;
    const rnd = (sc) => { s ^= s << 13; s >>>= 0; s ^= s >> 17; s ^= s << 5; s >>>= 0; return (s / 0xffffffff - 0.5) * sc; };
    for (let i = 0; i < this.Wxh.length; i++) this.Wxh[i] = rnd(0.1);
    for (let i = 0; i < this.Whh.length; i++) this.Whh[i] = rnd(0.1);
    for (let i = 0; i < this.Wproj.length; i++) this.Wproj[i] = rnd(0.1);
  }

  reset() { this.h.fill(0); }

  // substrate: { i8:Int8Array, scale:Float32Array, dim, pieces }
  forward(sub, tokId, targetId) {
    const inDim = this.inDim, H = this.H;
    const x = sub.row(tokId);           // 凍結入力（逆量子化済 192）
    const hPrev = this.h;
    const hNew = new Float32Array(H);
    for (let i = 0; i < H; i++) {
      let s = this.bh[i];
      const wr = i * inDim; for (let d = 0; d < inDim; d++) s += this.Wxh[wr + d] * x[d];
      const hr = i * H; for (let j = 0; j < H; j++) s += this.Whh[hr + j] * hPrev[j];
      hNew[i] = Math.tanh(s);
    }
    // z = Wproj·h  (Wproj: inDim×H)
    const z = new Float32Array(inDim);
    for (let d = 0; d < inDim; d++) { let s = 0; const b = d * H; for (let i = 0; i < H; i++) s += this.Wproj[b + i] * hNew[i]; z[d] = s; }
    // logits[v] = scale[v] * (z · i8_row_v)
    const V = sub.pieces.length, i8 = sub.i8, scale = sub.scale;
    const logits = new Float32Array(V); let mx = -Infinity;
    for (let v = 0; v < V; v++) {
      let dot = 0; const b = v * inDim;
      for (let d = 0; d < inDim; d++) dot += z[d] * i8[b + d];
      logits[v] = dot * scale[v]; if (logits[v] > mx) mx = logits[v];
    }
    let Z = 0; for (let v = 0; v < V; v++) { logits[v] = Math.exp(logits[v] - mx); Z += logits[v]; }
    for (let v = 0; v < V; v++) logits[v] /= Z;
    const surprise = -Math.log(logits[targetId] + 1e-12);
    let entropy = 0; for (let v = 0; v < V; v++) { const p = logits[v]; if (p > 1e-12) entropy -= p * Math.log(p); }
    this._c = { x, hPrev, hNew, z, probs: logits, target: targetId, V };
    this.h = hNew;
    return { surprise, entropy };
  }

  applyGradient(lr, sub) {
    const c = this._c; if (!c) return;
    const inDim = this.inDim, H = this.H;
    const { x, hPrev, hNew, probs, target, V } = c;
    const i8 = sub.i8, scale = sub.scale;
    // dlogits
    const dlog = new Float32Array(V);
    for (let v = 0; v < V; v++) dlog[v] = probs[v]; dlog[target] -= 1;
    // dz[d] = Σ_v dlog[v]·scale[v]·i8[v,d]
    const dz = new Float32Array(inDim);
    for (let v = 0; v < V; v++) {
      const dv = dlog[v] * scale[v]; if (dv === 0) continue;
      const b = v * inDim; for (let d = 0; d < inDim; d++) dz[d] += dv * i8[b + d];
    }
    // dh from z=Wproj·h（更新前の Wproj を使う）→ そのあと Wproj 更新
    const dh = new Float32Array(H);
    for (let d = 0; d < inDim; d++) { const dzd = dz[d]; const b = d * H; for (let i = 0; i < H; i++) dh[i] += dzd * this.Wproj[b + i]; }
    for (let d = 0; d < inDim; d++) { const g = lr * dz[d]; const b = d * H; for (let i = 0; i < H; i++) this.Wproj[b + i] -= g * hNew[i]; }
    // tanh → Wxh/Whh/bh
    const dpre = new Float32Array(H);
    for (let i = 0; i < H; i++) dpre[i] = dh[i] * (1 - hNew[i] * hNew[i]);
    for (let i = 0; i < H; i++) {
      const gp = lr * dpre[i];
      const wr = i * inDim; for (let d = 0; d < inDim; d++) this.Wxh[wr + d] -= gp * x[d];
      const hr = i * H; for (let j = 0; j < H; j++) this.Whh[hr + j] -= gp * hPrev[j];
      this.bh[i] -= gp;
    }
    this.step++;
  }

  observe(sub, tokId, targetId, lr = this.lr) {
    const r = this.forward(sub, tokId, targetId);
    this.applyGradient(lr, sub);
    return r.surprise;
  }

  // 文脈を処理した後の「次 subword」上位k（学習しない）。意味的予測の可視化用。
  predictTopK(sub, contextIds, k = 6) {
    this.reset();
    for (const id of contextIds) this.forward(sub, id, 0);
    const probs = this._c.probs;
    const idx = Array.from(probs.keys());
    idx.sort((a, b) => probs[b] - probs[a]);
    return idx.slice(0, k).map((v) => [sub.pieces[v], probs[v]]);
  }

  serialize() {
    const { inDim, H } = this;
    const floats = H * inDim + H * H + H + inDim * H;
    const buf = new ArrayBuffer(16 + floats * 4);
    const dv = new DataView(buf);
    dv.setInt32(0, MAGIC, true); dv.setInt32(4, inDim, true); dv.setInt32(8, H, true); dv.setInt32(12, this.step, true);
    const f = new Float32Array(buf, 16); let o = 0;
    f.set(this.Wxh, o); o += this.Wxh.length;
    f.set(this.Whh, o); o += this.Whh.length;
    f.set(this.bh, o); o += this.bh.length;
    f.set(this.Wproj, o); o += this.Wproj.length;
    return buf;
  }

  static deserialize(buf) {
    const dv = new DataView(buf);
    if (dv.getInt32(0, true) !== MAGIC) throw new Error('SubwordCore MAGIC 不一致');
    const inDim = dv.getInt32(4, true), H = dv.getInt32(8, true), step = dv.getInt32(12, true);
    const core = new SubwordCore({ inDim, H }); core.step = step;
    const f = new Float32Array(buf, 16); let o = 0;
    core.Wxh = f.slice(o, o += H * inDim);
    core.Whh = f.slice(o, o += H * H);
    core.bh = f.slice(o, o += H);
    core.Wproj = f.slice(o, o += inDim * H);
    return core;
  }
}
