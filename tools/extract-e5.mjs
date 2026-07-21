// Phase 2b-2 抽出＋剪定パイプライン（Node）。
//
// intfloat/multilingual-e5-small（vanilla BERT, Unigram SentencePiece）から
//   - word 埋め込みを日本語サブセットに剪定して int8 量子化
//   - 12層の重み（大きな行列は int8 per-row、小物は fp32）
//   - position/token_type/LayerNorm、トークナイザ語彙
// を取り出し、/models/substrate-e5-small-ja/ に compact な凍結 substrate を書く。
//
// 事前に model.safetensors と tokenizer.json をローカルに用意（download 済み）。
// 使い方: node tools/extract-e5.mjs <safetensors> <tokenizer.json>

import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';

const ST = process.argv[2];
const TKJSON = process.argv[3];
const OUT = new URL('../models/substrate-e5-small-ja/', import.meta.url);
mkdirSync(OUT, { recursive: true });

// --- safetensors 読み込み ---
const raw = readFileSync(ST);
const headerLen = Number(new DataView(raw.buffer, raw.byteOffset, 8).getBigUint64(0, true));
const header = JSON.parse(raw.subarray(8, 8 + headerLen).toString('utf8'));
const DATA0 = 8 + headerLen;
function tensorF32(name) {
  const t = header[name];
  const [s, e] = t.data_offsets;
  return new Float32Array(raw.buffer, raw.byteOffset + DATA0 + s, (e - s) / 4);
}

// --- 語彙剪定 ---
const tk = JSON.parse(readFileSync(TKJSON));
const vocab = tk.model.vocab; // [[piece, score], ...]
const pieces = vocab.map((x) => x[0]);
const scores = vocab.map((x) => x[1]);
const N = pieces.length;

const jp = /[぀-ヿ㐀-䶿一-鿿々〆ヶーｦ-ﾟ]/;
function keepPiece(p, id) {
  if (id < 4) return true;             // <s> <pad> </s> <unk>
  const body = p.startsWith('▁') ? p.slice(1) : p;
  if (body === '') return true;        // '▁' 単体
  if (jp.test(body)) return true;      // 日本語を含む
  // ASCII は単字のみ残す（数字・英字1文字・記号）。多字ラテンは日本語アプリに不要。
  if (/^[\x21-\x7E]$/.test(body)) return true;
  return false;
}
const keptOld = [];
for (let id = 0; id < N; id++) if (keepPiece(pieces[id], id)) keptOld.push(id);
const K = keptOld.length;
const newIdOf = new Map(keptOld.map((old, ni) => [old, ni]));
console.log(`vocab pruned: ${N} -> ${K}`);

// --- word 埋め込みを剪定して int8 per-row ---
const dim = 384;
const wordEmb = tensorF32('embeddings.word_embeddings.weight'); // [250037,384]
const embI8 = new Int8Array(K * dim);
const embScale = new Float32Array(K);
for (let ni = 0; ni < K; ni++) {
  const old = keptOld[ni];
  let mx = 0; const src = old * dim;
  for (let d = 0; d < dim; d++) { const a = Math.abs(wordEmb[src + d]); if (a > mx) mx = a; }
  const s = mx > 0 ? mx / 127 : 1; embScale[ni] = s;
  const dst = ni * dim;
  for (let d = 0; d < dim; d++) {
    let q = Math.round(wordEmb[src + d] / s); q = q > 127 ? 127 : q < -127 ? -127 : q;
    embI8[dst + d] = q;
  }
}
writeFileSync(new URL('emb.i8', OUT), Buffer.from(embI8.buffer));
writeFileSync(new URL('emb.scale', OUT), Buffer.from(embScale.buffer));

// --- 剪定後トークナイザ ---
const newPieces = keptOld.map((old) => pieces[old]);
const newScores = keptOld.map((old) => scores[old]);
writeFileSync(new URL('tokenizer.json', OUT), JSON.stringify({
  type: 'unigram', metaspace: '▁', pieces: newPieces, scores: newScores,
  bos: 0, eos: newIdOf.get(2), pad: newIdOf.get(1), unk: newIdOf.get(3),
}));

// --- 層の重み: 大きな行列は int8 per-row、小物は fp32 ---
const QUANT = /encoder\.layer\.\d+\.(attention\.self\.(query|key|value)|attention\.output\.dense|intermediate\.dense|output\.dense)\.weight$/;
const KEEP_F32 = [
  'embeddings.LayerNorm.weight', 'embeddings.LayerNorm.bias',
  'embeddings.position_embeddings.weight', 'embeddings.token_type_embeddings.weight',
];
const chunks = [];
let off = 0;
const index = [];
function push(name, dtype, shape, dataBuf, scaleBuf) {
  const dataOffset = off; chunks.push(dataBuf); off += dataBuf.byteLength;
  let scaleOffset = -1, scaleLen = 0;
  if (scaleBuf) { scaleOffset = off; chunks.push(scaleBuf); off += scaleBuf.byteLength; scaleLen = scaleBuf.byteLength; }
  index.push({ name, dtype, shape, dataOffset, dataLen: dataBuf.byteLength, scaleOffset, scaleLen });
}
function quantRows(f32, rows, cols) {
  const i8 = new Int8Array(rows * cols); const sc = new Float32Array(rows);
  for (let r = 0; r < rows; r++) {
    let mx = 0; const base = r * cols;
    for (let c = 0; c < cols; c++) { const a = Math.abs(f32[base + c]); if (a > mx) mx = a; }
    const s = mx > 0 ? mx / 127 : 1; sc[r] = s;
    for (let c = 0; c < cols; c++) { let q = Math.round(f32[base + c] / s); i8[base + c] = q > 127 ? 127 : q < -127 ? -127 : q; }
  }
  return { i8, sc };
}
for (const name of Object.keys(header)) {
  if (name === '__metadata__') continue;
  if (name === 'embeddings.word_embeddings.weight') continue;
  if (name === 'embeddings.position_ids') continue;         // I64、forward で不要
  if (name.startsWith('pooler.')) continue;                  // E5 は mean pooling
  const t = header[name];
  const f32 = tensorF32(name);
  if (QUANT.test(name) && !process.env.E5_FP32) {
    const [rows, cols] = t.shape;
    const { i8, sc } = quantRows(f32, rows, cols);
    push(name, 'i8', t.shape, Buffer.from(i8.buffer), Buffer.from(sc.buffer));
  } else if (KEEP_F32.includes(name) || name.includes('.bias') || name.includes('LayerNorm')) {
    push(name, 'f32', t.shape, Buffer.from(f32.buffer, f32.byteOffset, f32.byteLength));
  } else {
    // その他（想定外）は fp32 で保存
    push(name, 'f32', t.shape, Buffer.from(f32.buffer, f32.byteOffset, f32.byteLength));
  }
}
writeFileSync(new URL('layers.bin', OUT), Buffer.concat(chunks));
writeFileSync(new URL('layers.meta.json', OUT), JSON.stringify(index));

// --- meta ---
writeFileSync(new URL('meta.json', OUT), JSON.stringify({
  model: 'intfloat/multilingual-e5-small', arch: 'bert', hidden: dim, layers: 12,
  heads: 12, intermediate: 1536, vocab: K, pooling: 'mean', l2norm: true,
  e5_prefix: { query: 'query: ', passage: 'passage: ' },
  quant: { emb: 'int8-perrow', layerWeights: 'int8-perrow', rest: 'f32' },
}, null, 2));

// サイズ集計
const fs2 = readFileSync;
const sz = (p) => fs2(new URL(p, OUT)).length;
const total = sz('emb.i8') + sz('emb.scale') + sz('layers.bin') + sz('tokenizer.json') + sz('meta.json');
console.log(`emb.i8 ${(sz('emb.i8')/1e6).toFixed(1)}MB, layers.bin ${(sz('layers.bin')/1e6).toFixed(1)}MB, tokenizer ${(sz('tokenizer.json')/1e6).toFixed(1)}MB`);
console.log(`TOTAL ${(total/1e6).toFixed(1)}MB (vocab ${K}, dim ${dim})`);
