// Phase 0 の検証タスク: 2層 MLP で XOR を学習する。
//
// 目的は「学習ループが回り、loss が下がる」ことの実証のみ。言語である必要は
// ない。XOR は正解が自明で、loss 低下が一目で分かるため選定。
//
// ネットワーク:
//   入力 x(2) -> W1(2×H)+b1 -> tanh -> A1(H) -> W2(H×1)+b2 -> sigmoid -> y(1)
//   損失: MSE = mean((y - t)^2)
//
// 行列積(matmul)だけ backend に投げる。活性化・要素積・転置は JS で行う。
// これにより「GPU が学習ループ内で実際に計算している」ことを担保しつつ、
// backend の責務を matmul 1 点に絞る。

export const H = 4; // 隠れ層ユニット数

// XOR データセット（N=4）
export const X = new Float32Array([0, 0, 0, 1, 1, 0, 1, 1]); // 4×2
export const T = new Float32Array([0, 1, 1, 0]); // 4×1
export const N = 4;

// --- パラメータ生成 -------------------------------------------------------

// 小さめの乱数で初期化。seed で再現可能にする（検証の再現性のため）。
export function initParams(seed = 12345) {
  let s = seed >>> 0;
  const rnd = () => {
    // xorshift32 -> [-0.5, 0.5)
    s ^= s << 13; s >>>= 0;
    s ^= s >> 17;
    s ^= s << 5; s >>>= 0;
    return (s / 0xffffffff) - 0.5;
  };
  const scale = 1.0;
  return {
    H,
    step: 0,
    W1: Float32Array.from({ length: 2 * H }, () => rnd() * scale), // 2×H
    b1: new Float32Array(H),
    W2: Float32Array.from({ length: H * 1 }, () => rnd() * scale), // H×1
    b2: new Float32Array(1),
  };
}

// --- 補助 -----------------------------------------------------------------

const tanh = (v) => Math.tanh(v);
const sigmoid = (v) => 1 / (1 + Math.exp(-v));

// row-major (rows×cols) を転置して (cols×rows) を返す
function transpose(m, rows, cols) {
  const out = new Float32Array(rows * cols);
  for (let i = 0; i < rows; i++)
    for (let j = 0; j < cols; j++)
      out[j * rows + i] = m[i * cols + j];
  return out;
}

// --- 1 バッチ分の forward + backward + update ------------------------------
// backend.matmul を使うので async。lr は学習率。
// 戻り値: { loss, params(更新後) }
export async function trainStep(backend, p, lr = 0.5) {
  const { W1, b1, W2, b2 } = p;

  // forward
  // Z1 = X(N×2) · W1(2×H) => N×H
  const Z1 = await backend.matmul(X, N, 2, W1, 2, H);
  const A1 = new Float32Array(N * H);
  for (let i = 0; i < N; i++)
    for (let j = 0; j < H; j++)
      A1[i * H + j] = tanh(Z1[i * H + j] + b1[j]);

  // Z2 = A1(N×H) · W2(H×1) => N×1
  const Z2 = await backend.matmul(A1, N, H, W2, H, 1);
  const Y = new Float32Array(N);
  for (let i = 0; i < N; i++) Y[i] = sigmoid(Z2[i] + b2[0]);

  // loss = mean((Y - T)^2)
  let loss = 0;
  for (let i = 0; i < N; i++) {
    const d = Y[i] - T[i];
    loss += d * d;
  }
  loss /= N;

  // backward
  // dZ2 = dLoss/dZ2 = (2/N)(Y-T) * sigmoid'(Z2),  sigmoid' = Y(1-Y)
  const dZ2 = new Float32Array(N);
  for (let i = 0; i < N; i++)
    dZ2[i] = (2 / N) * (Y[i] - T[i]) * Y[i] * (1 - Y[i]);

  // gW2 = A1^T(H×N) · dZ2(N×1) => H×1
  const A1T = transpose(A1, N, H); // H×N
  const gW2 = await backend.matmul(A1T, H, N, dZ2, N, 1);
  // gb2 = sum(dZ2)
  let gb2 = 0;
  for (let i = 0; i < N; i++) gb2 += dZ2[i];

  // dA1 = dZ2(N×1) · W2^T(1×H) => N×H  （H×1 の row-major は 1×H と同一データ）
  const dA1 = await backend.matmul(dZ2, N, 1, W2, 1, H);
  // dZ1 = dA1 * tanh'(Z1),  tanh' = 1 - A1^2
  const dZ1 = new Float32Array(N * H);
  for (let i = 0; i < N * H; i++)
    dZ1[i] = dA1[i] * (1 - A1[i] * A1[i]);

  // gW1 = X^T(2×N) · dZ1(N×H) => 2×H
  const XT = transpose(X, N, 2); // 2×N
  const gW1 = await backend.matmul(XT, 2, N, dZ1, N, H);
  // gb1 = sum over N of dZ1  => H
  const gb1 = new Float32Array(H);
  for (let i = 0; i < N; i++)
    for (let j = 0; j < H; j++)
      gb1[j] += dZ1[i * H + j];

  // update（in-place）
  for (let i = 0; i < W1.length; i++) W1[i] -= lr * gW1[i];
  for (let j = 0; j < H; j++) b1[j] -= lr * gb1[j];
  for (let i = 0; i < W2.length; i++) W2[i] -= lr * gW2[i];
  b2[0] -= lr * gb2;
  p.step += 1;

  return { loss, params: p };
}
