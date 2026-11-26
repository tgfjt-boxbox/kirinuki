"""Write meta.json for one benchmark run folder, so a result can be traced to the code, models and machine.

usage: .venv/bin/python bench/run_meta.py OUT_DIR [--note TEXT]
Run it right after the evaluation, before committing, so `commit` is the code that produced the results.
`dirty` lists tracked files changed since that commit; a non-empty list means the results came from uncommitted code.
"""
import argparse
import datetime
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ("mediapipe", "numpy", "pillow")


def git(*args):
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True, check=True).stdout.strip()


def version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out", type=Path)
    p.add_argument("--note", default="")
    args = p.parse_args()
    meta = {
        "commit": git("rev-parse", "HEAD"),
        "dirty": git("diff", "--name-only", "HEAD").splitlines(),
        "created_utc": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        "machine": {"platform": platform.platform(), "processor": platform.machine()},
        "python": sys.version.split()[0],
        "packages": {name: version(name) for name in PACKAGES},
        "models": {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted((ROOT / "models").glob("*"))
                   if f.is_file()},
        "note": args.note,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
