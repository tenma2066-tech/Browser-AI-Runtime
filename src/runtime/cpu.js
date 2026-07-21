// CPU バックエンド（純 JS）。WebGPU が使えない環境のフォールバック。
// Phase 0 では正しさと単純さを最優先し、SIMD/WASM 最適化は行わない。

export class CPUBackend {
  constructor() {
    this.kind = 'cpu';
  }

  async init() {
    // CPU 実装に初期化は不要。契約を揃えるためだけの no-op。
  }

  // a: row-major (aRows × aCols), b: row-major (bRows × bCols)
  // 戻り値: row-major (aRows × bCols)
  async matmul(a, aRows, aCols, b, bRows, bCols) {
    if (aCols !== bRows) {
      throw new Error(`matmul 次元不一致: aCols=${aCols} bRows=${bRows}`);
    }
    const out = new Float32Array(aRows * bCols);
    for (let i = 0; i < aRows; i++) {
      for (let k = 0; k < aCols; k++) {
        const aik = a[i * aCols + k];
        if (aik === 0) continue;
        const bRow = k * bCols;
        const oRow = i * bCols;
        for (let j = 0; j < bCols; j++) {
          out[oRow + j] += aik * b[bRow + j];
        }
      }
    }
    return out;
  }

  dispose() {}
}
