import { Tensor } from "./vendor/litertjs/core/dist/index.js";
import { SIZE } from "./litert_core.js";

const PREPROCESS = /* wgsl */ `
@group(0) @binding(0) var frame: texture_2d<f32>;
@group(0) @binding(1) var smp: sampler;
@group(0) @binding(2) var<storage, read_write> rgb: array<f32>;
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= ${SIZE}u || id.y >= ${SIZE}u) { return; }
  let uv = (vec2f(id.xy) + 0.5) / ${SIZE}.0;
  let c = textureSampleLevel(frame, smp, uv, 0.0).rgb * 2.0 - 1.0;
  let o = (id.y * ${SIZE}u + id.x) * 3u;
  rgb[o] = c.r;
  rgb[o + 1u] = c.g;
  rgb[o + 2u] = c.b;
}`;

// The model outputs logits for 6 classes; MediaPipe's confidence mask is the softmax of channel 0 (background).
const SOFTMAX = /* wgsl */ `
struct Params { offset: u32 }
@group(0) @binding(0) var<storage, read> logits: array<f32>;
@group(0) @binding(1) var<storage, read_write> bg: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= ${SIZE * SIZE}u) { return; }
  let o = params.offset + i * 6u;
  var m = logits[o];
  for (var c = 1u; c < 6u; c++) { m = max(m, logits[o + c]); }
  var t = 0.0;
  for (var c = 0u; c < 6u; c++) { t += exp(logits[o + c] - m); }
  bg[i] = exp(logits[o] - m) / t;
}`;

// Runs resize to SIZE×SIZE, the [-1, 1] mapping, inference and the softmax on the GPU, and reads back only the background channel.
// Each call overwrites the Float32Array that the previous call returned.
export function gpuSegmenter(device, model, environment) {
  const rgb = device.createBuffer({ size: SIZE * SIZE * 3 * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const bg = device.createBuffer({ size: SIZE * SIZE * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const readback = device.createBuffer({ size: SIZE * SIZE * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const params = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

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
      entries: [{ binding: 0, resource: frameTex.createView() }, { binding: 1, resource: sampler }, { binding: 2, resource: { buffer: rgb } }],
    });
  };
  const smPipe = device.createComputePipeline({ layout: "auto", compute: { module: device.createShaderModule({ code: SOFTMAX }), entryPoint: "main" } });
  const smGroups = new WeakMap();
  const smGroup = (logits) => {
    if (!smGroups.has(logits))
      smGroups.set(logits, device.createBindGroup({
        layout: smPipe.getBindGroupLayout(0),
        entries: [{ binding: 0, resource: { buffer: logits } }, { binding: 1, resource: { buffer: bg } }, { binding: 2, resource: { buffer: params } }],
      }));
    return smGroups.get(logits);
  };
  const input = new Tensor(rgb, [1, SIZE, SIZE, 3], "float32", environment);
  const out = new Float32Array(SIZE * SIZE);

  return async (source) => {
    const width = source.naturalWidth ?? source.width, height = source.naturalHeight ?? source.height;
    frameTexture(width, height);
    device.queue.copyExternalImageToTexture({ source }, { texture: frameTex }, [width, height]);
    let enc = device.createCommandEncoder();
    let pass = enc.beginComputePass();
    pass.setPipeline(prePipe);
    pass.setBindGroup(0, preGroup);
    pass.dispatchWorkgroups(SIZE / 8, SIZE / 8);
    pass.end();
    device.queue.submit([enc.finish()]);

    const [res] = await model.run(input);
    const logits = res.toGpuBuffer();
    device.queue.writeBuffer(params, 0, new Uint32Array([res.liteRtTensorHandle.offset() / 4]));
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
