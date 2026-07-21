// WebGPU バックエンド（最小）。行列積を WGSL compute shader で GPU 実行する。
//
// Phase 0 の狙いは速度ではなく「iOS standalone PWA 上で WebGPU の compute が
// 正しく走り、学習ループの計算を担えること」の実証。よって naive matmul で
// 十分。タイル化などの最適化は Phase 1 以降。

const WGSL_MATMUL = /* wgsl */ `
struct Dims { aRows: u32, aCols: u32, bCols: u32, _pad: u32 };

@group(0) @binding(0) var<storage, read>       a   : array<f32>;
@group(0) @binding(1) var<storage, read>       b   : array<f32>;
@group(0) @binding(2) var<storage, read_write> out : array<f32>;
@group(0) @binding(3) var<uniform>             d   : Dims;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let row = gid.x;
  let col = gid.y;
  if (row >= d.aRows || col >= d.bCols) { return; }
  var sum : f32 = 0.0;
  for (var k : u32 = 0u; k < d.aCols; k = k + 1u) {
    sum = sum + a[row * d.aCols + k] * b[k * d.bCols + col];
  }
  out[row * d.bCols + col] = sum;
}
`;

export class WebGPUBackend {
  constructor() {
    this.kind = 'webgpu';
    this.device = null;
    this.pipeline = null;
    this.adapterInfo = null;
  }

  async init() {
    if (!navigator.gpu) throw new Error('navigator.gpu なし');
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error('requestAdapter が null');
    // 端末制約を記録（Phase 0 の観測項目）
    try {
      this.adapterInfo = adapter.info || (adapter.requestAdapterInfo
        ? await adapter.requestAdapterInfo() : null);
    } catch { this.adapterInfo = null; }
    this.limits = adapter.limits;
    this.device = await adapter.requestDevice();
    this.device.lost.then((info) => {
      // デバイスロストは iOS で起こりやすい。無視せず記録できるよう保持。
      this._lost = info;
    });
    const module = this.device.createShaderModule({ code: WGSL_MATMUL });
    this.pipeline = this.device.createComputePipeline({
      layout: 'auto',
      compute: { module, entryPoint: 'main' },
    });
  }

  async matmul(a, aRows, aCols, b, bRows, bCols) {
    if (aCols !== bRows) {
      throw new Error(`matmul 次元不一致: aCols=${aCols} bRows=${bRows}`);
    }
    const dev = this.device;
    const U = GPUBufferUsage;
    const outLen = aRows * bCols;

    const aBuf = dev.createBuffer({ size: a.byteLength, usage: U.STORAGE | U.COPY_DST });
    const bBuf = dev.createBuffer({ size: b.byteLength, usage: U.STORAGE | U.COPY_DST });
    const outBuf = dev.createBuffer({ size: outLen * 4, usage: U.STORAGE | U.COPY_SRC });
    const dimBuf = dev.createBuffer({ size: 16, usage: U.UNIFORM | U.COPY_DST });
    const readBuf = dev.createBuffer({ size: outLen * 4, usage: U.COPY_DST | U.MAP_READ });

    dev.queue.writeBuffer(aBuf, 0, a);
    dev.queue.writeBuffer(bBuf, 0, b);
    dev.queue.writeBuffer(dimBuf, 0, new Uint32Array([aRows, aCols, bCols, 0]));

    const bind = dev.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: aBuf } },
        { binding: 1, resource: { buffer: bBuf } },
        { binding: 2, resource: { buffer: outBuf } },
        { binding: 3, resource: { buffer: dimBuf } },
      ],
    });

    const enc = dev.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, bind);
    pass.dispatchWorkgroups(Math.ceil(aRows / 8), Math.ceil(bCols / 8));
    pass.end();
    enc.copyBufferToBuffer(outBuf, 0, readBuf, 0, outLen * 4);
    dev.queue.submit([enc.finish()]);

    await readBuf.mapAsync(GPUMapMode.READ);
    const res = new Float32Array(readBuf.getMappedRange()).slice();
    readBuf.unmap();

    aBuf.destroy();
    bBuf.destroy();
    outBuf.destroy();
    dimBuf.destroy();
    readBuf.destroy();
    return res;
  }

  dispose() {
    if (this.device) this.device.destroy();
  }
}
