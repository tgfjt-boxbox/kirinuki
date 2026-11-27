import { Tensor } from "./vendor/litertjs/core/dist/index.js";

export const SIZE = 256;

// MediaPipe feeds this model RGB in [-1, 1]; [0, 1] drops the mask agreement with MediaPipe to person IoU 0.985.
export function preprocess(smallCtx, frame, host) {
  smallCtx.drawImage(frame, 0, 0, SIZE, SIZE);
  const px = smallCtx.getImageData(0, 0, SIZE, SIZE).data;
  for (let i = 0, j = 0; i < px.length; i += 4, j += 3) {
    host[j] = px[i] / 127.5 - 1;
    host[j + 1] = px[i + 1] / 127.5 - 1;
    host[j + 2] = px[i + 2] / 127.5 - 1;
  }
}

export async function infer(model, environment, host) {
  const input = new Tensor(host, [1, SIZE, SIZE, 3], environment);
  const gpuIn = await input.moveTo("webgpu");
  const [res] = await model.run(gpuIn);
  gpuIn.delete();
  const cpu = await res.moveTo("wasm");
  const all = cpu.toTypedArray();
  cpu.delete();
  // The model outputs logits for 6 classes; MediaPipe's confidence mask is the softmax of channel 0 (background).
  const bg = new Float32Array(SIZE * SIZE);
  for (let i = 0; i < bg.length; i++) {
    const o = i * 6;
    let m = all[o];
    for (let c = 1; c < 6; c++) m = Math.max(m, all[o + c]);
    let t = 0;
    for (let c = 0; c < 6; c++) t += Math.exp(all[o + c] - m);
    bg[i] = Math.exp(all[o] - m) / t;
  }
  return bg;
}
