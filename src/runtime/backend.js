// Runtime 抽象（Phase 0 版・最小）
//
// Phase 0 の目的は「同一契約で WebGPU 実装と CPU 実装を差し替え可能」だと
// 実証すること。汎用テンソルライブラリは作らない。学習ループが本当に必要と
// する最小演算 = 行列積(matmul) だけを Backend の責務とする。
//
// forward/backward/update といった MLP のロジックは backend 非依存にして
// trainStep.js 側に置き、行列積だけを backend に投げる。これにより「GPU が
// 学習ループの中で実際に計算している」ことを証明しつつ、実装差し替えの境界を
// matmul 1 点に絞れる。
//
// Backend 契約:
//   kind: 'webgpu' | 'cpu'
//   init(): Promise<void>                 // デバイス取得。失敗時は例外を投げる
//   matmul(a, aRows, aCols, b, bRows, bCols): Promise<Float32Array>
//        a: row-major (aRows × aCols), b: row-major (bRows × bCols)
//        戻り値: row-major (aRows × bCols)。aCols === bRows が前提。
//   dispose(): void
//
// 注: ADR-0001 で「WASM フォールバック」と記したが、Phase 0 では純 JS の CPU
//     実装を採用する。WASM/SIMD 化は Phase 1 以降の最適化と位置づける（ビルド
//     ツールへの依存を Phase 0 で持ち込まないため）。

import { WebGPUBackend } from './webgpu.js';
import { CPUBackend } from './cpu.js';

// WebGPU を試し、失敗したら CPU へ自動フォールバックする。
// forceCpu=true のときは WebGPU を試さず CPU を返す（受け入れ条件#4の検証用）。
export async function selectBackend({ forceCpu = false } = {}) {
  if (!forceCpu && typeof navigator !== 'undefined' && navigator.gpu) {
    try {
      const be = new WebGPUBackend();
      await be.init();
      return { backend: be, reason: 'webgpu-ok' };
    } catch (err) {
      const cpu = new CPUBackend();
      await cpu.init();
      return { backend: cpu, reason: 'webgpu-init-failed:' + (err && err.message) };
    }
  }
  const cpu = new CPUBackend();
  await cpu.init();
  return {
    backend: cpu,
    reason: forceCpu ? 'forced-cpu' : 'no-navigator-gpu',
  };
}
