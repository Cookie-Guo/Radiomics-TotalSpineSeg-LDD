#!/usr/bin/env python3
"""
Download the public SPIDER release (Zenodo 10159290) to SPIDER_ROOT.
Skips files that already exist at the expected size.
"""

from __future__ import annotations
import os

import hashlib
import json
import ssl
import sys
import time
import urllib.request
import zipfile
from datetime import datetime
from pathlib import Path

DEST = Path(os.environ.get("SPIDER_ROOT", "<spider_root>"))
RECORD_API = "https://zenodo.org/api/records/10159290"
WANTED = ("images.zip", "masks.zip", "overview.csv", "radiological_gradings.csv")


def md5_file(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_record() -> dict:
    ctx = ssl.create_default_context()
    req = urllib.request.Request(RECORD_API, headers={"User-Agent": "LDD-revision-SPIDER/1.0"})
    with urllib.request.urlopen(req, context=ctx, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


def download(url: str, dest: Path, expected: int | None) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and expected and dest.stat().st_size == expected:
        print(f"  skip (size ok) {dest.name} {expected/1e9:.2f} GB")
        return
    if dest.exists() and expected and dest.stat().st_size != expected:
        print(f"  size mismatch {dest.name}: have {dest.stat().st_size}, want {expected}; re-download")
        dest.unlink()
    print(f"  downloading {dest.name} …")
    ctx = ssl.create_default_context()
    req = urllib.request.Request(url, headers={"User-Agent": "LDD-revision-SPIDER/1.0"})
    t0 = time.time()
    with urllib.request.urlopen(req, context=ctx, timeout=300) as r, open(dest, "wb") as f:
        n = 0
        last = 0
        while True:
            buf = r.read(1 << 20)
            if not buf:
                break
            f.write(buf)
            n += len(buf)
            if n - last >= 80 << 20:
                print(f"    {dest.name}: {n/1e9:.2f} GB")
                last = n
    elapsed = time.time() - t0
    got = dest.stat().st_size
    print(f"  wrote {dest.name} {got/1e9:.3f} GB in {elapsed:.0f}s")
    if expected and got != expected:
        raise SystemExit(f"size fail {dest.name}: {got} != {expected}")


def unzip(zpath: Path, outdir: Path) -> None:
    marker = outdir / f".unzipped_{zpath.stem}"
    if marker.exists():
        print(f"  unzip skip {zpath.name}")
        return
    print(f"  unzip {zpath.name} → {outdir}")
    with zipfile.ZipFile(zpath, "r") as zf:
        zf.extractall(outdir)
    marker.write_text(datetime.now().isoformat(timespec="seconds"), encoding="utf-8")


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    print("Zenodo record", RECORD_API)
    rec = fetch_record()
    files = {f["key"]: f for f in rec.get("files") or []}
    missing = [k for k in WANTED if k not in files]
    if missing:
        raise SystemExit(f"record missing files: {missing}; have {list(files)}")

    meta = {"record": 10159290, "downloaded_at": datetime.now().isoformat(timespec="seconds"), "files": {}}
    for key in WANTED:
        f = files[key]
        url = (f.get("links") or {}).get("content") or (f.get("links") or {}).get("self")
        size = int(f.get("size") or 0)
        dest = DEST / key
        download(url, dest, size)
        meta["files"][key] = {
            "size": dest.stat().st_size,
            "md5": md5_file(dest) if dest.stat().st_size < 80_000_000 else "skipped_large",
            "checksum": f.get("checksum"),
        }

    unzip(DEST / "images.zip", DEST / "images")
    unzip(DEST / "masks.zip", DEST / "masks")
    (DEST / "download.meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("D0 done →", DEST)


if __name__ == "__main__":
    main()
