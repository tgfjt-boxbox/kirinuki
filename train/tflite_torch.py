"""Run a float32 TFLite graph as a PyTorch module, with its weights as trainable parameters.

Covers the 11 builtin ops of selfie_multiclass_256x256.tflite. Activations stay NHWC like TFLite, so RESHAPE, TRANSPOSE
and SUM keep their meaning; convolutions permute to NCHW and back.

Needs the `tflite` package (flatbuffer schema bindings, Apache-2.0) in .venv: pip install tflite==2.18.0.

usage: .venv/bin/python -I train/tflite_torch.py MODEL.tflite IMAGE...
compares the softmax of the module's output with MediaPipe's confidence masks for each 256x256-resized image.
"""
import math
import sys

import numpy as np
import tflite
import torch
import torch.nn as nn
import torch.nn.functional as F

OP = {v: k for k, v in vars(tflite.BuiltinOperator).items() if not k.startswith("_")}
OPTIONS = {
    "CONV_2D": tflite.Conv2DOptions, "DEPTHWISE_CONV_2D": tflite.DepthwiseConv2DOptions, "ADD": tflite.AddOptions,
    "MUL": tflite.MulOptions, "TRANSPOSE_CONV": tflite.TransposeConvOptions, "SOFTMAX": tflite.SoftmaxOptions,
    "SUM": tflite.ReducerOptions, "RESIZE_BILINEAR": tflite.ResizeBilinearOptions,
    "RESIZE_NEAREST_NEIGHBOR": tflite.ResizeNearestNeighborOptions, "RESHAPE": tflite.ReshapeOptions,
}
SAME = tflite.Padding.SAME
ACT = {tflite.ActivationFunctionType.NONE: lambda x: x, tflite.ActivationFunctionType.RELU: F.relu}


def same_pad(x, k, s, d=1):
    """Pad NCHW x the way TFLite SAME padding does (the extra pixel goes after)."""
    pads = []
    for size, kk in ((x.shape[3], k[1]), (x.shape[2], k[0])):
        total = max((math.ceil(size / s) - 1) * s + (kk - 1) * d + 1 - size, 0)
        pads += [total // 2, total - total // 2]
    return F.pad(x, pads)


class TFLiteModule(nn.Module):
    def __init__(self, path):
        super().__init__()
        buf = open(path, "rb").read()
        model = tflite.Model.GetRootAsModel(buf, 0)
        g = model.Subgraphs(0)
        self.inputs, self.outputs = list(g.InputsAsNumpy()), list(g.OutputsAsNumpy())
        self.params = nn.ParameterDict()
        self.ops = []
        for i in range(g.TensorsLength()):
            t = g.Tensors(i)
            data = model.Buffers(t.Buffer()).DataAsNumpy()
            if not isinstance(data, np.ndarray) or data.size == 0:
                continue
            dtype = {tflite.TensorType.FLOAT32: np.float32, tflite.TensorType.INT32: np.int32}[t.Type()]
            arr = np.frombuffer(data.tobytes(), dtype=dtype).reshape(t.ShapeAsNumpy())
            if dtype == np.float32:
                self.params[str(i)] = nn.Parameter(torch.from_numpy(arr.copy()))
            else:
                setattr(self, f"const{i}", arr.copy())
        for i in range(g.OperatorsLength()):
            op = g.Operators(i)
            code = model.OperatorCodes(op.OpcodeIndex())
            name = OP[max(code.BuiltinCode(), code.DeprecatedBuiltinCode())]
            opts = None
            if name in OPTIONS and op.BuiltinOptions() is not None:
                opts = OPTIONS[name]()
                opts.Init(op.BuiltinOptions().Bytes, op.BuiltinOptions().Pos)
            self.ops.append((name, list(op.InputsAsNumpy()), list(op.OutputsAsNumpy()), opts))

    def tensor(self, env, i):
        if i in env:
            return env[i]
        if str(i) in self.params:
            return self.params[str(i)]
        return getattr(self, f"const{i}")

    def forward(self, x):
        """x: NHWC float32 in the model's input range. Returns the NHWC output (logits for multiclass)."""
        env = {self.inputs[0]: x}
        for name, ins, outs, o in self.ops:
            a = [self.tensor(env, i) if i >= 0 else None for i in ins]
            env[outs[0]] = getattr(self, f"op_{name.lower()}")(a, o)
        return env[self.outputs[0]]

    def op_conv_2d(self, a, o):
        x, w = a[0].permute(0, 3, 1, 2), a[1].permute(0, 3, 1, 2)
        s, d = o.StrideH(), o.DilationHFactor()
        if o.Padding() == SAME:
            x = same_pad(x, w.shape[2:], s, d)
        y = F.conv2d(x, w, a[2] if len(a) > 2 else None, stride=s, dilation=d)
        return ACT[o.FusedActivationFunction()](y.permute(0, 2, 3, 1))

    def op_depthwise_conv_2d(self, a, o):
        x, w = a[0].permute(0, 3, 1, 2), a[1].permute(3, 0, 1, 2)
        s, d = o.StrideH(), o.DilationHFactor()
        if o.Padding() == SAME:
            x = same_pad(x, w.shape[2:], s, d)
        y = F.conv2d(x, w, a[2] if len(a) > 2 else None, stride=s, dilation=d, groups=x.shape[1])
        return ACT[o.FusedActivationFunction()](y.permute(0, 2, 3, 1))

    def op_transpose_conv(self, a, o):
        shape, w, x = a[0], a[1], a[2].permute(0, 3, 1, 2)
        s, (kh, kw) = o.StrideH(), w.shape[1:3]
        full = F.conv_transpose2d(x, w.permute(3, 0, 1, 2), stride=s)
        oh, ow = int(shape[1]), int(shape[2])
        top = max((x.shape[2] - 1) * s + kh - oh, 0) // 2 if o.Padding() == SAME else 0
        left = max((x.shape[3] - 1) * s + kw - ow, 0) // 2 if o.Padding() == SAME else 0
        y = full[:, :, top:top + oh, left:left + ow]
        if len(a) > 3 and a[3] is not None:
            y = y + a[3].view(1, -1, 1, 1)
        return ACT[o.FusedActivationFunction()](y.permute(0, 2, 3, 1))

    def op_add(self, a, o):
        return ACT[o.FusedActivationFunction()](a[0] + a[1])

    def op_mul(self, a, o):
        return ACT[o.FusedActivationFunction()](a[0] * a[1])

    def op_reshape(self, a, o):
        shape = [int(v) for v in (a[1] if len(a) > 1 and a[1] is not None else o.NewShapeAsNumpy())]
        # The graph is traced at batch 1. Some reshapes fold image blocks into dim 0; the batch stays outermost there.
        batch = a[0].numel() // math.prod(shape)
        return a[0].reshape([shape[0] * batch] + shape[1:])

    def op_transpose(self, a, o):
        perm = [int(v) for v in a[1]]
        assert perm[0] == 0, "dim 0 holds the batch and must stay first"
        return a[0].permute(*perm)

    def op_softmax(self, a, o):
        return torch.softmax(a[0] * o.Beta(), dim=-1)

    def op_sum(self, a, o):
        axes = [int(v) for v in np.atleast_1d(a[1])]
        assert 0 not in axes, "summing over the batch dim"
        return a[0].sum(dim=axes, keepdim=bool(o.KeepDims()))

    def _resize(self, a, o, mode):
        assert o.HalfPixelCenters() and not o.AlignCorners(), "only half-pixel resizing is implemented"
        x = a[0].permute(0, 3, 1, 2)
        size = [int(v) for v in a[1]]
        y = F.interpolate(x, size=size, mode=mode, **({"align_corners": False} if mode == "bilinear" else {}))
        return y.permute(0, 2, 3, 1)

    def op_resize_bilinear(self, a, o):
        return self._resize(a, o, "bilinear")

    def op_resize_nearest_neighbor(self, a, o):
        return self._resize(a, o, "nearest-exact")


def main():
    import mediapipe as mp
    from PIL import Image

    path, images = sys.argv[1], sys.argv[2:]
    net = TFLiteModule(path).eval()
    seg = mp.tasks.vision.ImageSegmenter.create_from_options(mp.tasks.vision.ImageSegmenterOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=path), output_confidence_masks=True))
    worst = 0.0
    for p in images:
        rgb = np.asarray(Image.open(p).convert("RGB").resize((256, 256), Image.BILINEAR))
        ref = np.stack([np.squeeze(m.numpy_view(), -1) for m in seg.segment(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))).confidence_masks], -1)
        with torch.no_grad():
            out = torch.softmax(net(torch.from_numpy(rgb[None].astype(np.float32) / 127.5 - 1)), -1)[0].numpy()
        diff = np.abs(out - ref)
        agree = (out.argmax(-1) == ref.argmax(-1)).mean()
        worst = max(worst, diff.max())
        print(f"{p}: max |torch - mediapipe| {diff.max():.2e}, mean {diff.mean():.2e}, argmax agreement {agree:.4f}")
    print(f"worst max diff {worst:.2e}")


if __name__ == "__main__":
    main()
