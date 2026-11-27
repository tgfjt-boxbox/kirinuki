"""List HaGRID v1 images for distillation training, leaving out every image the evaluation uses, and unzip them.

usage: python3 -I scripts/train_list.py CLASS... [--per-class N]
Writes data/hagrid_v1/train_<N>.txt (paths relative to data/hagrid_v1, as litert_masks.html reads them): per class, the
first N names in sorted order that are not in any evaluation list data/hagrid_v1/<class>_<n>.txt (docs/datasets.md).
"""
import argparse
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "data" / "hagrid_v1"
ZIP = ROOT / "hagrid-sample-30k-384p.zip"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("classes", nargs="+")
    p.add_argument("--per-class", type=int, default=500)
    args = p.parse_args()
    held_out = {line for f in ROOT.glob("*_*.txt") if not f.name.startswith("train_") for line in f.read_text().split()}
    with zipfile.ZipFile(ZIP) as z:
        names = sorted(n for n in z.namelist() if n.endswith(".jpg"))
        assert held_out <= set(names), "an evaluation list names images that are not in the zip, so it would not be left out"
        picked = []
        for c in dict.fromkeys(args.classes):
            in_class = [n for n in names if f"/train_val_{c}/" in n and n not in held_out]
            assert len(in_class) >= args.per_class, f"{c}: only {len(in_class)} images outside the evaluation lists"
            picked += in_class[: args.per_class]
        for n in picked:
            if not (ROOT / n).exists():
                z.extract(n, ROOT)
    out = ROOT / f"train_{args.per_class}.txt"
    out.write_text("".join(f"{n}\n" for n in picked))
    print(f"{len(picked)} images from {len(args.classes)} classes, {len(held_out)} evaluation images left out -> {out}")


if __name__ == "__main__":
    main()
