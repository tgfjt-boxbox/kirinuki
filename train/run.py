"""Distill the two-pass zoom teacher (#58 masks) into a single-pass student on MPS.

Run through agent/train-mps, which sandboxes this file: no network, writes only under .cache/train-out ($HOME).
Each --data entry is SOURCE:LIST[:LIMIT]. A list line names an image under the source's root; its teacher mask is
.cache/train-data/<source>_teacher/<stem>.png (person probability in R).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

KIRINUKI = Path(__file__).resolve().parents[1]
ORG = KIRINUKI.parent
OUT = ORG / ".cache/train-out"


SOURCES = {
    "hagrid": lambda line: KIRINUKI / "data/hagrid_v1" / line,
    "ep_train": lambda line: KIRINUKI / "data/easyportrait_train/images" / f"{line}.jpg",
}


def load_pairs(data, size):
    images, targets = [], []
    for entry in data:
        source, list_file, *limit = entry.split(":")
        lines = (KIRINUKI / list_file).read_text().split()
        for line in lines[:int(limit[0])] if limit else lines:
            rgb = Image.open(SOURCES[source](line)).convert("RGB").resize((size, size), Image.BILINEAR)
            mask = Image.open(ORG / f".cache/train-data/{source}_teacher/{Path(line).stem}.png").getchannel("R")
            images.append(np.asarray(rgb))
            targets.append(np.asarray(mask.resize((size, size), Image.BILINEAR)))
    x = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).contiguous()
    y = torch.from_numpy(np.stack(targets))[:, None].contiguous()
    return x, y


def block(cin, cout, stride):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class Student(nn.Module):
    def __init__(self, width):
        super().__init__()
        w = width
        self.e1, self.e2, self.e3, self.e4 = block(3, w, 2), block(w, 2 * w, 2), block(2 * w, 4 * w, 2), block(4 * w, 8 * w, 2)
        self.d3, self.d2, self.d1 = block(12 * w, 4 * w, 1), block(6 * w, 2 * w, 1), block(3 * w, w, 1)
        self.head = nn.Conv2d(w, 1, 1)

    def forward(self, x):
        up = lambda a, b: torch.cat([F.interpolate(a, size=b.shape[-2:], mode="bilinear", align_corners=False), b], 1)
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        d = self.d1(up(self.d2(up(self.d3(up(e4, e3)), e2)), e1))
        return F.interpolate(self.head(d), size=x.shape[-2:], mode="bilinear", align_corners=False)


def batches(x, y, batch, device, shuffle, generator=None):
    order = torch.randperm(len(x), generator=generator) if shuffle else torch.arange(len(x))
    for i in range(0, len(x), batch):
        idx = order[i:i + batch]
        yield (x[idx].to(device).float() / 127.5 - 1), (y[idx].to(device).float() / 255)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", default=["hagrid:data/hagrid_v1/train_500.txt:9000"])
    ap.add_argument("--val", type=int, default=200)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--width", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default="smoke")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    t0 = time.time()
    x, y = load_pairs(args.data, args.size)
    split = torch.randperm(len(x), generator=torch.Generator().manual_seed(args.seed))
    xv, yv, xt, yt = x[split[:args.val]], y[split[:args.val]], x[split[args.val:]], y[split[args.val:]]
    model = Student(args.width).to(device)
    params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    print(f"device {device} train {len(xt)} val {len(xv)} params {params} load {time.time() - t0:.1f}s", flush=True)

    def val_loss():
        model.eval()
        total, agree = 0.0, 0.0
        with torch.no_grad():
            for xb, yb in batches(xv, yv, args.batch, device, False):
                logits = model(xb)
                total += F.binary_cross_entropy_with_logits(logits, yb, reduction="sum").item()
                agree += ((logits > 0) == (yb >= 0.5)).float().sum().item()
        n = yv.numel()
        model.train()
        return total / n, agree / n

    p = (yv.float() / 255).clamp(1e-6, 1 - 1e-6)
    floor = float(-(p * p.log() + (1 - p) * (1 - p).log()).mean())
    majority = float(max((yv >= 128).float().mean(), (yv < 128).float().mean()))
    history = [{"epoch": 0, "val_bce": val_loss()[0], "val_bce_floor": floor, "val_majority_agreement": majority}]
    print(f"epoch 0 val_bce {history[0]['val_bce']:.4f} floor {floor:.4f} majority_agree {majority:.4f}", flush=True)
    g = torch.Generator().manual_seed(args.seed)
    for epoch in range(1, args.epochs + 1):
        t = time.time()
        losses = []
        for xb, yb in batches(xt, yt, args.batch, device, True, g):
            loss = F.binary_cross_entropy_with_logits(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        vb, va = val_loss()
        history.append({"epoch": epoch, "train_bce": float(np.mean(losses)), "val_bce": vb, "val_pixel_agreement": va, "seconds": time.time() - t})
        print(f"epoch {epoch} train_bce {np.mean(losses):.4f} val_bce {vb:.4f} val_agree {va:.4f} {time.time() - t:.1f}s", flush=True)

    out = OUT / args.name
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "student.pt")
    (out / "history.json").write_text(json.dumps({"args": vars(args), "device": str(device), "params": params, "history": history}, indent=1))
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
