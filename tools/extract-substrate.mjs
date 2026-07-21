// Phase 2b オフライン抽出パイプライン（Node）。
//
// 実モデル ku-nlp/deberta-v2-tiny-japanese から
//   - 事前学習済みサブワード埋め込み word_embeddings.weight [32000,192]
//   - SentencePiece(unigram) 語彙 pieces/scores
// を取り出し、int8 量子化して /models 以下に compact な凍結 substrate として書き出す。
//
// フルの Transformer は取り込まない（Phase 2b の方針、docs/phase-2b-spec.md）。
//
// 使い方: node tools/extract-substrate.mjs

import { writeFileSync, mkdirSync } from 'node:fs';

const REPO = 'ku-nlp/deberta-v2-tiny-japanese';
const BASE = `https://huggingface.co/${REPO}/resolve/main`;
const OUT = new URL('../models/substrate-deberta-tiny-ja/', import.meta.url);
const TENSOR = 'deberta.embeddings.word_embeddings.weight';

async function range(url, start, end) {
  const res = await fetch(url, { headers: { Range: `bytes=${start}-${end}` } });
  if (!res.ok && res.status !== 206) throw new Error(`HTTP ${res.status} for ${url}`);
  const buf = Buffer.from(await res.arrayBuffer());
  return { buf, partial: res.status === 206 };
}

async function main() {
  mkdirSync(OUT, { recursive: true });
  const stUrl = `${BASE}/model.safetensors`;

  // 1) ヘッダ長（先頭8バイト uint64 LE）
  const head8 = (await range(stUrl, 0, 7)).buf;
  const headerLen = Number(head8.readBigUint64LE(0));
  console.log('header length =', headerLen);

  // 2) ヘッダ JSON
  const headerBuf = (await range(stUrl, 8, 8 + headerLen - 1)).buf;
  const header = JSON.parse(headerBuf.toString('utf8'));
  const t = header[TENSOR];
  if (!t) throw new Error(`${TENSOR} が見つからない`);
  console.log('tensor', TENSOR, t.dtype, t.shape, 'offsets', t.data_offsets);
  const [rows, dim] = t.shape;
  const dataStart = 8 + headerLen + t.data_offsets[0];
  const dataEnd = 8 + headerLen + t.data_offsets[1] - 1;

  // 3) 埋め込み本体のみ range-download
  console.log(`downloading embeddings ${((dataEnd - dataStart + 1) / 1e6).toFixed(1)}MB ...`);
  const embBuf = (await range(stUrl, dataStart, dataEnd)).buf;
  const emb = new Float32Array(embBuf.buffer, embBuf.byteOffset, rows * dim);
  console.log('loaded floats:', emb.length, '(expected', rows * dim, ')');

  // 4) 行ごと int8 量子化
  const i8 = new Int8Array(rows * dim);
  const scale = new Float32Array(rows);
  for (let r = 0; r < rows; r++) {
    let mx = 0; const base = r * dim;
    for (let d = 0; d < dim; d++) { const a = Math.abs(emb[base + d]); if (a > mx) mx = a; }
    const s = mx > 0 ? mx / 127 : 1;
    scale[r] = s;
    for (let d = 0; d < dim; d++) {
      let q = Math.round(emb[base + d] / s);
      if (q > 127) q = 127; else if (q < -127) q = -127;
      i8[base + d] = q;
    }
  }
  writeFileSync(new URL('emb.i8', OUT), Buffer.from(i8.buffer));
  writeFileSync(new URL('emb.scale', OUT), Buffer.from(scale.buffer));

  // 5) トークナイザ（unigram pieces/scores）
  console.log('fetching tokenizer.json ...');
  const tk = await (await fetch(`${BASE}/tokenizer.json`)).json();
  const model = tk.model || {};
  const vocab = model.vocab || []; // [[piece, score], ...]
  const pieces = vocab.map((x) => x[0]);
  const scores = vocab.map((x) => x[1]);
  const unkId = model.unk_id ?? 0;
  // メタスペース記号（Metaspace pre-tokenizer の replacement, 既定 ▁）
  let metaspace = '▁';
  const pre = tk.pre_tokenizer;
  if (pre && pre.replacement) metaspace = pre.replacement;
  writeFileSync(new URL('tokenizer.json', OUT), JSON.stringify({
    type: 'unigram', unk_id: unkId, metaspace, pieces, scores,
  }));

  // 6) メタ情報
  const meta = {
    model: REPO, vocab: rows, dim, quant: 'int8-perrow',
    files: { emb: 'emb.i8', scale: 'emb.scale', tokenizer: 'tokenizer.json' },
    note: 'frozen pretrained subword embeddings only (no transformer forward)',
  };
  writeFileSync(new URL('meta.json', OUT), JSON.stringify(meta, null, 2));

  console.log('done. vocab', rows, 'dim', dim, 'pieces', pieces.length);
}

main().catch((e) => { console.error(e); process.exit(1); });
