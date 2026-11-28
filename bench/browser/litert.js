import { loadLiteRt, loadAndCompile, Environment } from "./vendor/litertjs/core/dist/index.js";
import { SIZE, preprocess, infer } from "./litert_core.js";
import { gpuSegmenter, gpuZoomSegmenter, zoomBox } from "./litert_gpu.js";
import { FilesetResolver, ImageSegmenter } from "./vendor/package/vision_bundle.mjs";

const [hashInput, hashRepeat, hashF16] = location.hash.slice(1).split("/");
const MODEL = "selfie_multiclass_256x256";
const INPUTS = hashInput ? [hashInput] : ["static", "moving", "small"];
const REPEAT = Number(hashRepeat) || 1;
const F16 = hashF16 !== "nof16";
const BACKGROUND = [0, 1, 0];
const WARMUP = 20;
const FRAMES = 100;
const BOX_EMA = 0.3;

const pct = (xs, p) => xs.slice().sort((a, b) => a - b)[Math.min(xs.length - 1, Math.floor(p * xs.length))];
const stat = (xs) => ({ p50: +pct(xs, 0.5).toFixed(2), p95: +pct(xs, 0.95).toFixed(2) });

function drawFrame(ctx, img, input, i) {
  const s = Math.max(640 / img.width, 480 / img.height) * { static: 1, moving: 1.15, small: 0.5 }[input];
  const dx = input === "static" ? 0 : 40 * Math.sin(i / 7);
  const dy = input === "static" ? 0 : 20 * Math.cos(i / 11);
  ctx.drawImage(img, (640 - img.width * s) / 2 + dx, (480 - img.height * s) / 2 + dy, img.width * s, img.height * s);
}

function compositor(gl) {
  const shader = (type, src) => {
    const sh = gl.createShader(type);
    gl.shaderSource(sh, src);
    gl.compileShader(sh);
    if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(sh));
    return sh;
  };
  const prog = gl.createProgram();
  gl.attachShader(prog, shader(gl.VERTEX_SHADER, `#version 300 es
    out vec2 uv;
    void main() {
      vec2 p = vec2(gl_VertexID & 1, gl_VertexID >> 1);
      uv = vec2(p.x, 1.0 - p.y);
      gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
    }`));
  gl.attachShader(prog, shader(gl.FRAGMENT_SHADER, `#version 300 es
    precision highp float;
    uniform sampler2D frame;
    uniform highp sampler2D mask;
    uniform highp sampler2D zoomMask;
    uniform vec4 box;
    uniform vec3 background;
    in vec2 uv;
    out vec4 color;
    void main() {
      float bg = texture(mask, uv).r;
      vec2 zuv = (uv - box.xy) / box.zw;
      if (box.z > 0.0 && all(greaterThanEqual(zuv, vec2(0.0))) && all(lessThanEqual(zuv, vec2(1.0)))) bg = min(bg, texture(zoomMask, zuv).r);
      float a = 1.0 - bg;
      color = vec4(mix(background, texture(frame, uv).rgb, a), 1.0);
    }`));
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(prog));
  const vao = gl.createVertexArray();
  const tex = (filter) => {
    const t = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, t);
    for (const [k, v] of [[gl.TEXTURE_MIN_FILTER, filter], [gl.TEXTURE_MAG_FILTER, filter],
      [gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE], [gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE]]) gl.texParameteri(gl.TEXTURE_2D, k, v);
    return t;
  };
  const frameTex = tex(gl.LINEAR);
  const maskTex = tex(gl.LINEAR);
  const zoomTex = tex(gl.LINEAR);
  gl.bindTexture(gl.TEXTURE_2D, null);
  const loc = (n) => gl.getUniformLocation(prog, n);

  // mask: a background Float32Array, or a gpuZoomSegmenter result { first, second, box }.
  return (source, mask) => {
    const zoom = mask instanceof Float32Array ? { first: mask, second: null, box: null } : mask;
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, gl.drawingBufferWidth, gl.drawingBufferHeight);
    gl.useProgram(prog);
    gl.bindVertexArray(vao);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, maskTex);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 4);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R16F, SIZE, SIZE, 0, gl.RED, gl.FLOAT, zoom.first);
    if (zoom.box) {
      gl.activeTexture(gl.TEXTURE2);
      gl.bindTexture(gl.TEXTURE_2D, zoomTex);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.R16F, SIZE, SIZE, 0, gl.RED, gl.FLOAT, zoom.second);
    }
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, frameTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, source);
    gl.uniform1i(loc("frame"), 0);
    gl.uniform1i(loc("mask"), 1);
    gl.uniform1i(loc("zoomMask"), 2);
    gl.uniform4fv(loc("box"), zoom.box ?? [0, 0, 0, 0]);
    gl.uniform3fv(loc("background"), BACKGROUND);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    gl.bindVertexArray(null);
    gl.bindTexture(gl.TEXTURE_2D, null);
  };
}

function agreement(bg, mpMask, w, h) {
  let same = 0, inter = 0, union = 0;
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      const fy = Math.min(SIZE - 1, Math.max(0, (y + 0.5) * SIZE / h - 0.5)), fx = Math.min(SIZE - 1, Math.max(0, (x + 0.5) * SIZE / w - 0.5));
      const y0 = Math.floor(fy), x0 = Math.floor(fx), y1 = Math.min(SIZE - 1, y0 + 1), x1 = Math.min(SIZE - 1, x0 + 1), wy = fy - y0, wx = fx - x0;
      const v = (bg[y0 * SIZE + x0] * (1 - wx) + bg[y0 * SIZE + x1] * wx) * (1 - wy) + (bg[y1 * SIZE + x0] * (1 - wx) + bg[y1 * SIZE + x1] * wx) * wy;
      const a = v < 0.5, b = mpMask[y * w + x] < 0.5;
      same += a === b;
      inter += a && b;
      union += a || b;
    }
  return { pixel: +(same / (w * h)).toFixed(4), person_iou: +(inter / union).toFixed(4) };
}

async function main() {
  const out = { userAgent: navigator.userAgent, model: MODEL, f16: F16, box_ema: BOX_EMA, results: [], ema_boxes: {} };
  try {
    const adapter = await navigator.gpu.requestAdapter();
    out.gpu = `${adapter.info.vendor} ${adapter.info.architecture}`;
    const device = await adapter.requestDevice({ requiredFeatures: F16 ? [...adapter.features].filter((f) => f === "shader-f16") : [] });
    out.device_features = [...device.features].sort();
    await loadLiteRt("./vendor/litertjs/core/wasm/");
    const environment = await Environment.create({ webGpuDevice: device });
    const model = await loadAndCompile(`../../models/${MODEL}.tflite`, { accelerator: "webgpu", environment });
    out.accelerator = model.options.accelerator;

    const img = new Image();
    img.src = "../materials/public/office-peace.png";
    await img.decode();
    const canvas = Object.assign(document.createElement("canvas"), { width: 640, height: 480 });
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    const smallCtx = Object.assign(document.createElement("canvas"), { width: SIZE, height: SIZE }).getContext("2d", { willReadFrequently: true });
    const glCanvas = Object.assign(document.createElement("canvas"), { width: 640, height: 480 });
    const gl = glCanvas.getContext("webgl2");
    const composite = compositor(gl);
    const pixel = new Uint8Array(4);
    const host = new Float32Array(SIZE * SIZE * 3);

    drawFrame(ctx, img, "static", 0);
    const fileset = await FilesetResolver.forVisionTasks("./vendor/package/wasm");
    const seg = await ImageSegmenter.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: `../../models/${MODEL}.tflite`, delegate: "GPU" },
      runningMode: "IMAGE", outputConfidenceMasks: true, outputCategoryMask: false,
    });
    const mp = seg.segment(canvas).confidenceMasks[0].getAsFloat32Array().slice();
    seg.close();
    preprocess(smallCtx, canvas, host);
    out.mask_agreement_vs_mediapipe = agreement(await infer(model, environment, host), mp, 640, 480);
    const segmentOnGpu = gpuSegmenter(device, model, environment);
    const zoomOnGpu = gpuZoomSegmenter(device, model, environment);
    const zoomAlwaysOnGpu = gpuZoomSegmenter(device, model, environment, { maxArea: Infinity });
    const zoomEmaOnGpu = gpuZoomSegmenter(device, model, environment, { boxEma: BOX_EMA });
    const zoomEmaAlwaysOnGpu = gpuZoomSegmenter(device, model, environment, { boxEma: BOX_EMA, maxArea: Infinity });
    out.gpu_mask_agreement_vs_mediapipe = agreement(await segmentOnGpu(canvas), mp, 640, 480);

    const PATHS = {
      LITERT_WEBGPU_COMPOSITE: async () => {
        preprocess(smallCtx, canvas, host);
        return infer(model, environment, host);
      },
      LITERT_WEBGPU_GPUPRE_COMPOSITE: () => segmentOnGpu(canvas),
      LITERT_WEBGPU_ZOOM_COMPOSITE: () => zoomOnGpu(canvas),
      LITERT_WEBGPU_ZOOM_ALWAYS_COMPOSITE: () => zoomAlwaysOnGpu(canvas),
      LITERT_WEBGPU_ZOOM_EMA_COMPOSITE: () => zoomEmaOnGpu(canvas),
      LITERT_WEBGPU_ZOOM_EMA_ALWAYS_COMPOSITE: () => zoomEmaAlwaysOnGpu(canvas),
    };
    for (const input of INPUTS)
      for (let k = 0; k < REPEAT; k++)
        for (const [mode, segment] of Object.entries(PATHS)) {
          const seg = [], total = [];
          let zoomed = 0;
          for (let i = 0; i < WARMUP + FRAMES; i++) {
            ctx.fillStyle = "#000";
            ctx.fillRect(0, 0, 640, 480);
            drawFrame(ctx, img, input, i);
            const t0 = performance.now();
            const bg = await segment();
            const t1 = performance.now();
            composite(canvas, bg);
            gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, pixel);
            const t2 = performance.now();
            if (i >= WARMUP) { seg.push(t1 - t0); total.push(t2 - t0); zoomed += !!bg.box; }
            if (mode === "LITERT_WEBGPU_ZOOM_EMA_ALWAYS_COMPOSITE" && k === 0)
              (out.ema_boxes[input] ??= []).push({ raw: zoomBox(bg.first, { maxArea: Infinity }), box: bg.box && [...bg.box] });
          }
          out.results.push({ mode, input, frames: FRAMES, zoomed_frames: zoomed, segment_ms: stat(seg), with_readback_ms: stat(total) });
        }
    model.delete();
  } catch (e) {
    out.error = String(e && e.stack || e);
  }
  await fetch("/result", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(out) });
  document.body.append("RESULT " + JSON.stringify(out));
}
main();
