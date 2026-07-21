// Memory Fabric（最小）。ベクトル + 可塑性/保持/出所などの属性を持つ記憶を、
// 同一ベクトル空間に蓄える。短期/長期/知識/人格は別テーブルではなく、
// plasticity/decay パラメータの違いで連続的に区別する（docs/ADR-0001）。
//
// 独自性の芯:
//   - 想起 = 類似度 × retention（純類似度ではなく「生きている」記憶が優先される）
//   - 忘却 = retention の減衰（eviction ではない）。可塑性の低い記憶ほど忘れにくい。
//
// 想起の類似度計算（クエリ × 全記憶）は Runtime の matmul を使う。これは将来
// GPU 常駐に載せたい"大きめの計算"であり、Runtime 抽象をここで行使する。

const MAGIC = 0x4241464e; // "BAFN"（assoc 追加で形式更新）
const PROV = ['perception', 'derived', 'pack'];

const now = () => (typeof performance !== 'undefined' && performance.now
  ? performance.now() : Date.now());

export class MemoryFabric {
  constructor(backend, { D = 64, lambda = 0.05, reinforceGain = 0.25 } = {}) {
    this.backend = backend;
    this.D = D;
    this.lambda = lambda;
    this.reinforceGain = reinforceGain;
    this.items = [];
    this.nextId = 1;
    this.tick = 0;
  }

  // 記銘。vector は L2 正規化済みを前提（Encoder.encode が保証）。
  write(vector, attrs = {}) {
    const t = now();
    const item = {
      id: this.nextId++,
      vector: Float32Array.from(vector),
      plasticity: attrs.plasticity ?? 0.8,
      retention: attrs.retention ?? 1.0,
      salience: attrs.salience ?? 0.5,
      provenance: attrs.provenance ?? 'perception',
      createdAt: t,
      lastAccess: t,
      accessCount: 0,
      assoc: attrs.assoc ?? -1, // 連想値（例: 次文字のトークンID）。-1 = なし
      text: attrs.text ?? '',
    };
    this.items.push(item);
    return item.id;
  }

  // 想起。score = cos類似度 × retention の上位 k 件。
  async recall(query, k = 5) {
    const N = this.items.length;
    if (N === 0) return [];
    const D = this.D;
    // B: D×N（各記憶ベクトルを列に並べる）
    const B = new Float32Array(D * N);
    for (let n = 0; n < N; n++) {
      const v = this.items[n].vector;
      for (let d = 0; d < D; d++) B[d * N + n] = v[d];
    }
    // sims(1×N) = query(1×D) · B(D×N)。query/vector とも正規化済み → cos。
    const sims = await this.backend.matmul(query, 1, D, B, D, N);
    const scored = this.items.map((item, n) => ({
      item, sim: sims[n], score: sims[n] * item.retention,
    }));
    scored.sort((a, b) => b.score - a.score);
    const top = scored.slice(0, k);
    const t = now();
    for (const s of top) { s.item.lastAccess = t; s.item.accessCount++; }
    return top;
  }

  // 強化（Hebbian 的）。想起された記憶の retention を引き上げる。
  reinforce(id) {
    const it = this.items.find((x) => x.id === id);
    if (!it) return false;
    it.retention = Math.min(1, it.retention + this.reinforceGain);
    return true;
  }

  // 忘却。retention *= exp(-lambda·dt·plasticity)。
  // 可塑性が高い（短期記憶）ほど速く忘れ、低い（知識/人格＝凍結寄り）ほど残る。
  decay(dt = 1) {
    this.tick += dt;
    for (const it of this.items) {
      const rate = this.lambda * dt * it.plasticity;
      it.retention *= Math.exp(-rate);
    }
  }

  stats() {
    const N = this.items.length;
    let sum = 0, min = 1, max = 0;
    for (const it of this.items) { sum += it.retention; min = Math.min(min, it.retention); max = Math.max(max, it.retention); }
    return { count: N, avgRetention: N ? sum / N : 0, minRetention: N ? min : 0, maxRetention: max, tick: this.tick };
  }

  // --- 直列化（bit 一致復元）------------------------------------------------
  serialize() {
    const D = this.D;
    const enc = new TextEncoder();
    const textBytes = this.items.map((it) => enc.encode(it.text || ''));
    const perItem = (i) => 4 * 5 + 4 * 3 + 8 * 2 + D * 4 + textBytes[i].length;
    //             id/prov/acc/textLen/assoc  plast/ret/sal  created/last  vector  text
    let total = 32; // header
    for (let i = 0; i < this.items.length; i++) total += perItem(i);
    const buf = new ArrayBuffer(total);
    const dv = new DataView(buf);
    dv.setInt32(0, MAGIC, true);
    dv.setInt32(4, D, true);
    dv.setInt32(8, this.items.length, true);
    dv.setInt32(12, this.nextId, true);
    dv.setFloat32(16, this.lambda, true);
    dv.setFloat32(20, this.reinforceGain, true);
    dv.setFloat64(24, this.tick, true);
    let off = 32;
    for (let i = 0; i < this.items.length; i++) {
      const it = this.items[i];
      const tb = textBytes[i];
      dv.setInt32(off, it.id, true); off += 4;
      dv.setInt32(off, PROV.indexOf(it.provenance) < 0 ? 0 : PROV.indexOf(it.provenance), true); off += 4;
      dv.setInt32(off, it.accessCount, true); off += 4;
      dv.setInt32(off, tb.length, true); off += 4;
      dv.setInt32(off, it.assoc ?? -1, true); off += 4;
      dv.setFloat32(off, it.plasticity, true); off += 4;
      dv.setFloat32(off, it.retention, true); off += 4;
      dv.setFloat32(off, it.salience, true); off += 4;
      dv.setFloat64(off, it.createdAt, true); off += 8;
      dv.setFloat64(off, it.lastAccess, true); off += 8;
      for (let d = 0; d < D; d++) { dv.setFloat32(off, it.vector[d], true); off += 4; }
      new Uint8Array(buf, off, tb.length).set(tb); off += tb.length;
    }
    return buf;
  }

  static deserialize(buf, backend) {
    const dv = new DataView(buf);
    if (dv.getInt32(0, true) !== MAGIC) throw new Error('Memory MAGIC 不一致');
    const D = dv.getInt32(4, true);
    const count = dv.getInt32(8, true);
    const nextId = dv.getInt32(12, true);
    const lambda = dv.getFloat32(16, true);
    const reinforceGain = dv.getFloat32(20, true);
    const tick = dv.getFloat64(24, true);
    const fabric = new MemoryFabric(backend, { D, lambda, reinforceGain });
    fabric.nextId = nextId;
    fabric.tick = tick;
    const dec = new TextDecoder();
    let off = 32;
    for (let i = 0; i < count; i++) {
      const id = dv.getInt32(off, true); off += 4;
      const prov = dv.getInt32(off, true); off += 4;
      const accessCount = dv.getInt32(off, true); off += 4;
      const textLen = dv.getInt32(off, true); off += 4;
      const assoc = dv.getInt32(off, true); off += 4;
      const plasticity = dv.getFloat32(off, true); off += 4;
      const retention = dv.getFloat32(off, true); off += 4;
      const salience = dv.getFloat32(off, true); off += 4;
      const createdAt = dv.getFloat64(off, true); off += 8;
      const lastAccess = dv.getFloat64(off, true); off += 8;
      const vector = new Float32Array(D);
      for (let d = 0; d < D; d++) { vector[d] = dv.getFloat32(off, true); off += 4; }
      const text = textLen ? dec.decode(new Uint8Array(buf, off, textLen)) : ''; off += textLen;
      fabric.items.push({
        id, provenance: PROV[prov] || 'perception', accessCount, assoc,
        plasticity, retention, salience, createdAt, lastAccess, vector, text,
      });
    }
    return fabric;
  }
}
