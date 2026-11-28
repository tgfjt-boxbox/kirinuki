import { Tensor } from "./vendor/litertjs/core/dist/index.js";
import { SIZE } from "./litert_core.js";

const PREPROCESS = /* wgsl */ `
@group(0) @binding(0) var frame: texture_2d<f32>;
@group(0) @binding(1) var smp: sampler;
@group(0) @binding(2) var<storage, read_write> rgb: array<f32>;
@group(0) @binding(3) var<uniform> box: vec4f;
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= ${SIZE}u || id.y >= ${SIZE}u) { return; }
  let uv = box.xy + (vec2f(id.xy) + 0.5) / ${SIZE}.0 * box.zw;
  let c = textureSampleLevel(frame, smp, uv, 0.0).rgb * 2.0 - 1.0;
  let o = (id.y * ${SIZE}u + id.x) * 3u;
  rgb[o] = c.r;
  rgb[o + 1u] = c.g;
  rgb[o + 2u] = c.b;
}`;

// The model outputs logits for 6 classes; MediaPipe's confidence mask is the softmax of channel 0 (background).
const SOFTMAX = /* wgsl */ `
@group(0) @binding(0) var<storage, read> logits: array<f32>;
@group(0) @binding(1) var<storage, read_write> bg: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= ${SIZE * SIZE}u) { return; }
  let o = i * 6u;
  var m = logits[o];
  for (var c = 1u; c < 6u; c++) { m = max(m, logits[o + c]); }
  var t = 0.0;
  for (var c = 0u; c < 6u; c++) { t += exp(logits[o + c] - m); }
  bg[i] = exp(logits[o] - m) / t;
}`;

export const FULL_FRAME = [0, 0, 1, 1];

// Runs resize to SIZE×SIZE, the [-1, 1] mapping, inference and the softmax on the GPU, and reads back only the background channel.
// box = [x, y, width, height] in normalized frame coordinates selects the region that is resized to the model input.
// The result is written into `out` (default: one array that every call overwrites).
export function gpuSegmenter(device, model, environment) {
  const rgb = device.createBuffer({ size: SIZE * SIZE * 3 * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const bg = device.createBuffer({ size: SIZE * SIZE * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const readback = device.createBuffer({ size: SIZE * SIZE * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const boxBuf = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

  const prePipe = device.createComputePipeline({ layout: "auto", compute: { module: device.createShaderModule({ code: PREPROCESS }), entryPoint: "main" } });
  const sampler = device.createSampler({ magFilter: "linear", minFilter: "linear" });
  let frameTex, preGroup;
  const frameTexture = (width, height) => {
    if (frameTex?.width === width && frameTex?.height === height) return;
    frameTex?.destroy();
    frameTex = device.createTexture({
      size: [width, height], format: "rgba8unorm",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
    });
    preGroup = device.createBindGroup({
      layout: prePipe.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: frameTex.createView() }, { binding: 1, resource: sampler }, { binding: 2, resource: { buffer: rgb } },
        { binding: 3, resource: { buffer: boxBuf } }],
    });
  };
  const smPipe = device.createComputePipeline({ layout: "auto", compute: { module: device.createShaderModule({ code: SOFTMAX }), entryPoint: "main" } });
  const smGroups = new WeakMap();
  const smGroup = (logits) => {
    if (!smGroups.has(logits))
      smGroups.set(logits, device.createBindGroup({
        layout: smPipe.getBindGroupLayout(0),
        entries: [{ binding: 0, resource: { buffer: logits } }, { binding: 1, resource: { buffer: bg } }],
      }));
    return smGroups.get(logits);
  };
  const input = new Tensor(rgb, [1, SIZE, SIZE, 3], "float32", environment);
  const shared = new Float32Array(SIZE * SIZE);

  return async (source, box = FULL_FRAME, out = shared, upload = true) => {
    if (upload) {
      const width = source.naturalWidth ?? source.width, height = source.naturalHeight ?? source.height;
      frameTexture(width, height);
      device.queue.copyExternalImageToTexture({ source }, { texture: frameTex }, [width, height]);
    }
    device.queue.writeBuffer(boxBuf, 0, new Float32Array(box));
    let enc = device.createCommandEncoder();
    let pass = enc.beginComputePass();
    pass.setPipeline(prePipe);
    pass.setBindGroup(0, preGroup);
    pass.dispatchWorkgroups(SIZE / 8, SIZE / 8);
    pass.end();
    device.queue.submit([enc.finish()]);

    const [res] = await model.run(input);
    const logits = res.toGpuBuffer();
    enc = device.createCommandEncoder();
    pass = enc.beginComputePass();
    pass.setPipeline(smPipe);
    pass.setBindGroup(0, smGroup(logits));
    pass.dispatchWorkgroups(SIZE * SIZE / 64);
    pass.end();
    enc.copyBufferToBuffer(bg, 0, readback, 0, SIZE * SIZE * 4);
    device.queue.submit([enc.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    out.set(new Float32Array(readback.getMappedRange()));
    readback.unmap();
    res.delete();
    return out;
  };
}

// Box around the person (background < 0.5) in a SIZE×SIZE background mask, padded by `margin` of its size on each side and widened
// to the frame's aspect ratio (equal normalized width and height). Returns null when there is no person or the box covers more
// than `maxArea` of the frame, in which case a second pass would add nothing.
export function zoomBox(bg, { margin = 0.1, maxArea = 0.95 } = {}) {
  let x0 = SIZE, y0 = SIZE, x1 = -1, y1 = -1;
  for (let y = 0; y < SIZE; y++)
    for (let x = 0; x < SIZE; x++)
      if (bg[y * SIZE + x] < 0.5) {
        if (x < x0) x0 = x;
        if (x > x1) x1 = x;
        if (y < y0) y0 = y;
        if (y > y1) y1 = y;
      }
  if (x1 < 0) return null;
  let l = x0 / SIZE, r = (x1 + 1) / SIZE, t = y0 / SIZE, b = (y1 + 1) / SIZE;
  const mx = margin * (r - l), my = margin * (b - t);
  l = Math.max(0, l - mx); r = Math.min(1, r + mx); t = Math.max(0, t - my); b = Math.min(1, b + my);
  const side = Math.min(1, Math.max(r - l, b - t));
  if (side * side > maxArea) return null;
  const cx = Math.min(Math.max((l + r) / 2, side / 2), 1 - side / 2), cy = Math.min(Math.max((t + b) / 2, side / 2), 1 - side / 2);
  return [cx - side / 2, cy - side / 2, side, side];
}

// Two passes: the whole frame, then the box around the person at full model resolution. The mask is
// min(first, second) inside the box (max of the person probability), the first pass outside it; see zoomedAt.
// boxEma < 1 blends each box with the previous frame's (weight boxEma on the new one) so a jumping box does not flicker on
// video; a frame without a box drops the history. The default keeps no history, so stills are independent.
export function gpuZoomSegmenter(device, model, environment, { boxEma = 1, ...options } = {}) {
  const segment = gpuSegmenter(device, model, environment);
  const first = new Float32Array(SIZE * SIZE), second = new Float32Array(SIZE * SIZE);
  let prev = null;
  return async (source) => {
    await segment(source, FULL_FRAME, first);
    let box = zoomBox(first, options);
    if (box && prev) box = box.map((v, i) => boxEma * v + (1 - boxEma) * prev[i]);
    prev = box;
    if (!box) return { first, second: null, box: null };
    await segment(source, box, second, false);
    return { first, second, box };
  };
}

const bilinear = (m, u, v) => {
  const fx = Math.min(SIZE - 1, Math.max(0, u * SIZE - 0.5)), fy = Math.min(SIZE - 1, Math.max(0, v * SIZE - 0.5));
  const x0 = Math.floor(fx), y0 = Math.floor(fy), x1 = Math.min(SIZE - 1, x0 + 1), y1 = Math.min(SIZE - 1, y0 + 1), wx = fx - x0, wy = fy - y0;
  return (m[y0 * SIZE + x0] * (1 - wx) + m[y0 * SIZE + x1] * wx) * (1 - wy) + (m[y1 * SIZE + x0] * (1 - wx) + m[y1 * SIZE + x1] * wx) * wy;
};

// Background probability at normalized frame coordinates (u, v) of a gpuZoomSegmenter result; the compositor shader does the same.
export function zoomedAt({ first, second, box }, u, v) {
  const a = bilinear(first, u, v);
  if (!box) return a;
  const su = (u - box[0]) / box[2], sv = (v - box[1]) / box[3];
  if (su < 0 || su > 1 || sv < 0 || sv > 1) return a;
  return Math.min(a, bilinear(second, su, sv));
}
