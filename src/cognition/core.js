// Cognitive Core（最小）: リカレントな次文字予測器。
//
// 設計（docs/phase-2a-spec.md）:
//   h_t = tanh(Wxh·x_t + Whh·h_{t-1} + bh)      … 状態遷移（state_t）
//   logits = Why·h_t + by,  p = softmax(logits)  … 次文字の予測
//   surprise = -log p[実際の次文字]               … 自己評価（予測誤差）
//
// 入力 x_t は凍結エンコーダ埋め込み E[idx]（Representation が Core に供給）。
// 学習は truncated BPTT(1)（h_{t-1} は定数扱い）。Core の学習は encoder を触らない。

const MAGIC = 0x42414332; // "BAC2"

export class CognitiveCore {
  constructor({ D = 64, H = 64, vocabCap = 4096, lr = 0.05, seed = 99 } = {}) {
    this.D = D; this.H = H; this.vocabCap = vocabCap; this.lr = lr;
    this.V = 0; this.step = 0;
    this.Wxh = new Float32Array(H * D);
    this.Whh = new Float32Array(H * H);
    this.bh = new Float32Array(H);
    this.Why = new Float32Array(vocabCap * H);
    this.by = new Float32Array(vocabCap);
    this.h = new Float32Array(H);
    let s = seed >>> 0;
    const rnd = () => { s ^= s << 13; s >>>= 0; s ^= s >> 17; s ^= s << 5; s >>>= 0; return (s / 0xffffffff - 0.5) * 0.2; };
    for (let i = 0; i < this.Wxh.length; i++) this.Wxh[i] = rnd();
    for (let i = 0; i < this.Whh.length; i++) this.Whh[i] = rnd();
    for (let i = 0; i < this.Why.length; i++) this.Why[i] = rnd();
  }

  reset() { this.h.fill(0); }

  // 1 ステップ。idx = 現在の文字、targetIdx = 実際の次の文字。
  // learn=true なら truncated BPTT(1) で更新。戻り値 = surprise。
  observe(encoder, idx, targetIdx, learn = true) {
    const { D, H, lr } = this;
    const V = encoder.size;
    this.V = V;
    const x = encoder.E.subarray(idx * D, idx * D + D); // 凍結入力特徴
    const hPrev = this.h;

    // forward: 隠れ状態
    const pre = new Float32Array(H);
    for (let i = 0; i < H; i++) {
      let sum = this.bh[i];
      const wr = i * D; for (let d = 0; d < D; d++) sum += this.Wxh[wr + d] * x[d];
      const hr = i * H; for (let j = 0; j < H; j++) sum += this.Whh[hr + j] * hPrev[j];
      pre[i] = sum;
    }
    const hNew = new Float32Array(H);
    for (let i = 0; i < H; i++) hNew[i] = Math.tanh(pre[i]);

    // forward: 語彙 logits と softmax
    const logits = new Float32Array(V);
    let mx = -Infinity;
    for (let v = 0; v < V; v++) {
      let sum = this.by[v]; const r = v * H;
      for (let i = 0; i < H; i++) sum += this.Why[r + i] * hNew[i];
      logits[v] = sum; if (sum > mx) mx = sum;
    }
    let Z = 0;
    for (let v = 0; v < V; v++) { logits[v] = Math.exp(logits[v] - mx); Z += logits[v]; }
    const surprise = -Math.log(logits[targetIdx] / Z + 1e-12);

    if (learn) {
      // dlogits = softmax - onehot(target)
      const dlog = logits;
      for (let v = 0; v < V; v++) dlog[v] /= Z;
      dlog[targetIdx] -= 1;

      // dh = Why^T · dlog（更新前の Why を使う）、同時に by を更新
      const dh = new Float32Array(H);
      for (let v = 0; v < V; v++) {
        const dv = dlog[v]; const r = v * H;
        for (let i = 0; i < H; i++) dh[i] += dv * this.Why[r + i];
        this.by[v] -= lr * dv;
      }
      // Why 更新
      for (let v = 0; v < V; v++) {
        const dv = dlog[v]; if (dv === 0) continue;
        const r = v * H; const g = lr * dv;
        for (let i = 0; i < H; i++) this.Why[r + i] -= g * hNew[i];
      }
      // tanh を通す
      const dpre = new Float32Array(H);
      for (let i = 0; i < H; i++) dpre[i] = dh[i] * (1 - hNew[i] * hNew[i]);
      // Wxh / Whh / bh 更新（h_{t-1} は定数扱い = truncated BPTT(1)）
      for (let i = 0; i < H; i++) {
        const gp = lr * dpre[i];
        const wr = i * D; for (let d = 0; d < D; d++) this.Wxh[wr + d] -= gp * x[d];
        const hr = i * H; for (let j = 0; j < H; j++) this.Whh[hr + j] -= gp * hPrev[j];
        this.bh[i] -= gp;
      }
      this.step++;
    }

    this.h = hNew; // 状態を前進
    return surprise;
  }

  // パラメトリック予測分布を返す（学習しない）。h は前進する。
  // 双方向ループで memory の事前分布とブレンドするために使う。
  predict(encoder, idx) {
    const { D, H } = this;
    const V = encoder.size; this.V = V;
    const x = encoder.E.subarray(idx * D, idx * D + D);
    const hPrev = this.h;
    const hNew = new Float32Array(H);
    for (let i = 0; i < H; i++) {
      let sum = this.bh[i];
      const wr = i * D; for (let d = 0; d < D; d++) sum += this.Wxh[wr + d] * x[d];
      const hr = i * H; for (let j = 0; j < H; j++) sum += this.Whh[hr + j] * hPrev[j];
      hNew[i] = Math.tanh(sum);
    }
    const probs = new Float32Array(V);
    let mx = -Infinity;
    for (let v = 0; v < V; v++) {
      let sum = this.by[v]; const r = v * H;
      for (let i = 0; i < H; i++) sum += this.Why[r + i] * hNew[i];
      probs[v] = sum; if (sum > mx) mx = sum;
    }
    let Z = 0;
    for (let v = 0; v < V; v++) { probs[v] = Math.exp(probs[v] - mx); Z += probs[v]; }
    for (let v = 0; v < V; v++) probs[v] /= Z;
    this.h = hNew;
    return probs;
  }

  // --- 直列化（有効語彙行のみ。bit 一致復元）--------------------------------
  serialize() {
    const { D, H, vocabCap, V } = this;
    const floats = H * D + H * H + H + V * H + V + H;
    const buf = new ArrayBuffer(24 + floats * 4);
    const dv = new DataView(buf);
    dv.setInt32(0, MAGIC, true);
    dv.setInt32(4, D, true); dv.setInt32(8, H, true);
    dv.setInt32(12, vocabCap, true); dv.setInt32(16, V, true); dv.setInt32(20, this.step, true);
    const f = new Float32Array(buf, 24);
    let o = 0;
    f.set(this.Wxh, o); o += this.Wxh.length;
    f.set(this.Whh, o); o += this.Whh.length;
    f.set(this.bh, o); o += this.bh.length;
    f.set(new Float32Array(this.Why.buffer, 0, V * H), o); o += V * H;
    f.set(new Float32Array(this.by.buffer, 0, V), o); o += V;
    f.set(this.h, o); o += this.h.length;
    return buf;
  }

  static deserialize(buf) {
    const dv = new DataView(buf);
    if (dv.getInt32(0, true) !== MAGIC) throw new Error('Core MAGIC 不一致');
    const D = dv.getInt32(4, true), H = dv.getInt32(8, true);
    const vocabCap = dv.getInt32(12, true), V = dv.getInt32(16, true), step = dv.getInt32(20, true);
    const core = new CognitiveCore({ D, H, vocabCap });
    core.V = V; core.step = step;
    const f = new Float32Array(buf, 24);
    let o = 0;
    core.Wxh = f.slice(o, o += H * D);
    core.Whh = f.slice(o, o += H * H);
    core.bh = f.slice(o, o += H);
    core.Why = new Float32Array(vocabCap * H);
    core.Why.set(f.slice(o, o += V * H));
    core.by = new Float32Array(vocabCap);
    core.by.set(f.slice(o, o += V));
    core.h = f.slice(o, o += H);
    return core;
  }
}
