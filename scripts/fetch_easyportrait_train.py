"""Fetch the first N EasyPortrait train images and masks into data/easyportrait_train/ for training.
Source: the EasyPortrait v2 train.zip on the dataset's sbercloud storage (68.3 GB). Masks come from the local data/easyportrait/annotations.zip.
The images are read by HTTP Range from the remote zip, so only the central directory and the chosen files are downloaded, not all 68 GB.
The sample is the first N ids in sorted order that have both an image and a mask. Ids are UUIDs, so this is a reproducible random sample.
usage: .venv/bin/python -I scripts/fetch_easyportrait_train.py N [--max-side 512]
"""

import argparse
import http.client
import io
import os
import urllib.request
import zipfile
from pathlib import Path

from PIL import Image

URL = "https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/easyportrait/v2/train.zip"
ROOT = Path(__file__).resolve().parent.parent
ANNOTATIONS = ROOT / "data/easyportrait/annotations.zip"
TEST_IDS = ROOT / "data/easyportrait/ids_all.txt"
OUT = ROOT / "data/easyportrait_train"
CHUNK = 1 << 20


class RangeFile(io.RawIOBase):
    def __init__(self, url):
        self.url = url
        self.pos = 0
        self.buf_start = 0
        self.buf = b""
        self.fetched = 0
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req) as r:
            self.size = int(r.headers["Content-Length"])

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        base = {0: 0, 1: self.pos, 2: self.size}[whence]
        self.pos = base + offset
        return self.pos

    def _get(self, start, end):
        # The server drops long responses midway, so resume from the last byte received.
        data = b""
        for _ in range(10):
            req = urllib.request.Request(self.url, headers={"Range": f"bytes={start + len(data)}-{end - 1}"})
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    assert r.status == 206, r.status
                    data += r.read()
            except http.client.IncompleteRead as e:
                data += e.partial
            except OSError:
                pass
            if len(data) == end - start:
                break
        assert len(data) == end - start, (len(data), start, end)
        self.fetched += len(data)
        return data

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0:
            return b""
        end = self.pos + n
        buf_end = self.buf_start + len(self.buf)
        if not (self.buf_start <= self.pos and end <= buf_end):
            kept = self.buf[self.pos - self.buf_start :] if self.buf_start <= self.pos < buf_end else b""
            start = self.pos + len(kept)
            self.buf = kept + self._get(start, min(self.size, max(end, start + CHUNK)))
            self.buf_start = self.pos
        out = self.buf[self.pos - self.buf_start : end - self.buf_start]
        self.pos = end
        return out


def open_remote():
    remote = RangeFile(URL)
    rzip = zipfile.ZipFile(remote)
    jpgs = {Path(m).stem: m for m in rzip.namelist() if m.lower().endswith(".jpg")}
    return remote, rzip, jpgs


def write_atomic(path, save):
    tmp = path.with_name(path.name + ".part")
    save(tmp)
    os.replace(tmp, path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("n", type=int)
    p.add_argument("--max-side", type=int, default=512)
    args = p.parse_args()

    images_dir, masks_dir = OUT / "images", OUT / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    list_path = OUT / f"ids_{args.n}.txt"

    ann = zipfile.ZipFile(ANNOTATIONS)
    masks = {Path(m).stem: m for m in ann.namelist() if m.startswith("train/") and m.endswith(".png")}
    remote = None
    if list_path.exists():
        ids = list_path.read_text().split()
    else:
        remote, rzip, jpgs = open_remote()
        ids = sorted(jpgs.keys() & masks.keys())[: args.n]
        write_atomic(list_path, lambda t: t.write_text("".join(i + "\n" for i in ids)))
    test_ids = set(TEST_IDS.read_text().split())
    assert not test_ids & set(ids), "train ids overlap the test split"

    for i in ids:
        img_out, mask_out = images_dir / f"{i}.jpg", masks_dir / f"{i}.png"
        if img_out.exists() and mask_out.exists():
            continue
        if remote is None:
            remote, rzip, jpgs = open_remote()
        img = Image.open(io.BytesIO(rzip.read(jpgs[i]))).convert("RGB")
        mask = Image.open(io.BytesIO(ann.read(masks[i])))
        # train.zip holds originals up to about 3000 px while the masks are at most 1920 px, so match shapes, not sizes.
        assert abs(img.width * mask.height / img.height - mask.width) <= 1, (i, img.size, mask.size)
        scale = min(1, args.max_side / max(mask.size))
        size = (round(mask.width * scale), round(mask.height * scale))
        if img.size != size:
            img = img.resize(size, Image.LANCZOS)
        if mask.size != size:
            mask = mask.resize(size, Image.NEAREST)
        write_atomic(img_out, lambda t: img.save(t, "JPEG", quality=95))
        write_atomic(mask_out, lambda t: mask.save(t, "PNG"))

    on_disk = sum((images_dir / f"{i}.jpg").exists() and (masks_dir / f"{i}.png").exists() for i in ids)
    print(f"listed {len(ids)}, on disk {on_disk}, bytes fetched {remote.fetched if remote else 0}")


if __name__ == "__main__":
    main()
