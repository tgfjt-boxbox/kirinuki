import { loadLiteRt, loadAndCompile, Environment } from "./vendor/litertjs/core/dist/index.js";
import { SIZE, preprocess, infer } from "./litert_core.js";
import { gpuSegmenter, gpuZoomSegmenter, zoomedAt } from "./litert_gpu.js";

const MASK_MAX = 1024;
const [source, list, framing, path = "cpu"] = location.hash.slice(1).split("/");
const SOURCES = {
  hagrid: { list: `../../data/hagrid_v1/${list}.txt`, url: (line) => `../../data/hagrid_v1/${line}` },
  ep: { list: `../../data/easyportrait/${list}.txt`, url: (id) => `../../data/easyportrait/images/${id}.jpg` },
};

function cropBox(w, h) {
  const target = 640 / 480;
  if (w / h > target) {
    const cw = Math.round(h * target);
    return [Math.floor((w - cw) / 2), 0, cw, h];
  }
  const ch = Math.round(w / target);
  return [0, Math.floor((h - ch) / 2), w, ch];
}

async function main() {
  const out = { userAgent: navigator.userAgent, source, list, framing, path, images: 0, errors: [] };
  try {
    if (!SOURCES[source] || !["crop43", "whole"].includes(framing) || !["cpu", "gpu", "zoom"].includes(path)) throw new Error(`bad hash: ${location.hash}`);
    const adapter = await navigator.gpu.requestAdapter();
    const device = await adapter.requestDevice({ requiredFeatures: [...adapter.features].filter((f) => f === "shader-f16") });
    await loadLiteRt("./vendor/litertjs/core/wasm/");
    const environment = await Environment.create({ webGpuDevice: device });
    const model = await loadAndCompile("../../models/selfie_multiclass_256x256.tflite", { accelerator: "webgpu", environment });
    const lines = (await (await fetch(SOURCES[source].list)).text()).split("\n").map((l) => l.trim()).filter(Boolean);
    const frameCtx = Object.assign(document.createElement("canvas"), { width: 640, height: 480 }).getContext("2d", { willReadFrequently: true });
    const smallCtx = Object.assign(document.createElement("canvas"), { width: SIZE, height: SIZE }).getContext("2d", { willReadFrequently: true });
    const maskCanvas = Object.assign(document.createElement("canvas"), { width: SIZE, height: SIZE });
    const maskCtx = maskCanvas.getContext("2d");
    const host = new Float32Array(SIZE * SIZE * 3);
    const segment = path === "gpu" ? gpuSegmenter(device, model, environment) : path === "zoom" ? gpuZoomSegmenter(device, model, environment) : async (frame) => {
      preprocess(smallCtx, frame, host);
      return infer(model, environment, host);
    };
    const fullCanvas = document.createElement("canvas");
    let zoomed = 0;
    for (const line of lines) {
      const stem = line.split("/").pop().replace(/\.[^.]+$/, "");
      try {
        const img = new Image();
        img.src = SOURCES[source].url(line);
        await img.decode();
        let frame = img;
        if (framing === "crop43") {
          const [sx, sy, sw, sh] = cropBox(img.naturalWidth, img.naturalHeight);
          frameCtx.drawImage(img, sx, sy, sw, sh, 0, 0, 640, 480);
          frame = frameCtx.canvas;
        }
        const bg = await segment(frame);
        let target = maskCanvas;
        if (path === "zoom") {
          // The second pass has finer detail than SIZE×SIZE inside the box, so the mask is written at frame resolution
          // (long side capped at MASK_MAX; the evaluators resize it to the image).
          zoomed += !!bg.box;
          const fw = frame.naturalWidth ?? frame.width, fh = frame.naturalHeight ?? frame.height;
          const scale = Math.min(1, MASK_MAX / Math.max(fw, fh));
          const w = Math.round(fw * scale), h = Math.round(fh * scale);
          Object.assign(fullCanvas, { width: w, height: h });
          const ctx = fullCanvas.getContext("2d");
          const pixels = ctx.createImageData(w, h);
          for (let y = 0; y < h; y++)
            for (let x = 0; x < w; x++) {
              const v = Math.round((1 - zoomedAt(bg, (x + 0.5) / w, (y + 0.5) / h)) * 255);
              pixels.data.set([v, v, v, 255], (y * w + x) * 4);
            }
          ctx.putImageData(pixels, 0, 0);
          target = fullCanvas;
        } else {
          const pixels = maskCtx.createImageData(SIZE, SIZE);
          for (let i = 0; i < bg.length; i++) {
            const v = Math.round((1 - bg[i]) * 255);
            pixels.data.set([v, v, v, 255], i * 4);
          }
          maskCtx.putImageData(pixels, 0, 0);
        }
        const blob = await new Promise((resolve) => target.toBlob(resolve, "image/png"));
        const res = await fetch(`/mask/${stem}.png`, { method: "POST", body: blob });
        if (!res.ok) throw new Error(`POST mask ${res.status}`);
        out.images++;
      } catch (e) {
        out.errors.push(`${line}: ${e}`);
      }
    }
    out.zoomed = zoomed;
    model.delete();
  } catch (e) {
    out.error = String(e && e.stack || e);
  }
  await fetch("/result", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(out) });
  document.body.append("RESULT " + JSON.stringify({ images: out.images, errors: out.errors.length, error: out.error }));
}
main();
