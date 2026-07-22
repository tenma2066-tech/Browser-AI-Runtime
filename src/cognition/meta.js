// Phase 3a: Meta Cognition。Cognitive Core の surprise（予測誤差）と entropy
// （予測の不確実性）を監視し、学習の仕方を制御する。
//
//   - 適応学習率: lr = baseLr × clamp(surprise / EMA(surprise), lo, hi)
//     「最近の平常より驚いたら速く学ぶ」。安定入力では下限に張り付き凍結気味に
//     なり、過学習・破滅的忘却を抑える。相対値なので自己較正する。
//   - 質問する: entropy が高く（多くの候補で迷う）かつ驚きが窓平均で持続する時
//     だけ ask を立てる。1発のノイズには反応せず、クールダウンで連発を防ぐ。
//
// Meta は Core を観測・制御するだけで、Core の重みには直接触れない。

const MAGIC = 0x42414d31; // "BAM1"

export class MetaCognition {
  constructor({
    baseLr = 0.05, alpha = 0.02,
    lrLo = 0.3, lrHi = 4.0,
    windowSize = 12, entropyThresh = 0.5, askSurpriseMult = 1.3, cooldown = 8,
  } = {}) {
    this.baseLr = baseLr; this.alpha = alpha;
    this.lrLo = lrLo; this.lrHi = lrHi;
    this.windowSize = windowSize; this.entropyThresh = entropyThresh;
    this.askSurpriseMult = askSurpriseMult; this.cooldown = cooldown;
    this.emaSurprise = null;
    this.window = [];
    this.sinceAsk = cooldown; // 起動直後から質問可能
    this.steps = 0;
  }

  // 1 ステップの制御。surprise/entropy を受け取り lr と ask を返す。
  control(surprise, entropy, V) {
    this.steps++;
    // EMA 更新（平常 surprise の基準）
    this.emaSurprise = this.emaSurprise === null ? surprise
      : (1 - this.alpha) * this.emaSurprise + this.alpha * surprise;
    // 直近窓
    this.window.push(surprise);
    if (this.window.length > this.windowSize) this.window.shift();

    const base = this.emaSurprise + 1e-6;
    const lrFactor = Math.max(this.lrLo, Math.min(this.lrHi, surprise / base));
    const lr = this.baseLr * lrFactor;

    const normEntropy = V > 1 ? entropy / Math.log(V) : 0;
    const winMean = this.window.reduce((s, x) => s + x, 0) / this.window.length;
    const persistent = winMean > this.emaSurprise * this.askSurpriseMult;
    this.sinceAsk++;
    const ask = normEntropy > this.entropyThresh && persistent && this.sinceAsk >= this.cooldown;
    if (ask) this.sinceAsk = 0;

    return { lr, ask, lrFactor, normEntropy, emaSurprise: this.emaSurprise };
  }

  serialize() {
    // 統計のみ永続化（emaSurprise, sinceAsk, steps, 窓）。
    const w = this.window;
    const buf = new ArrayBuffer(20 + w.length * 4);
    const dv = new DataView(buf);
    dv.setInt32(0, MAGIC, true);
    dv.setFloat32(4, this.emaSurprise === null ? -1 : this.emaSurprise, true);
    dv.setInt32(8, this.sinceAsk, true);
    dv.setInt32(12, this.steps, true);
    dv.setInt32(16, w.length, true);
    for (let i = 0; i < w.length; i++) dv.setFloat32(20 + i * 4, w[i], true);
    return buf;
  }

  static deserialize(buf, opts) {
    const dv = new DataView(buf);
    if (dv.getInt32(0, true) !== MAGIC) throw new Error('Meta MAGIC 不一致');
    const m = new MetaCognition(opts);
    const ema = dv.getFloat32(4, true);
    m.emaSurprise = ema < 0 ? null : ema;
    m.sinceAsk = dv.getInt32(8, true);
    m.steps = dv.getInt32(12, true);
    const n = dv.getInt32(16, true);
    m.window = [];
    for (let i = 0; i < n; i++) m.window.push(dv.getFloat32(20 + i * 4, true));
    return m;
  }
}
