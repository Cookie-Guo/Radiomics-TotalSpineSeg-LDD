#!/usr/bin/env python3
"""
Download the official RadImageNet ResNet50 Keras notop weights (single ZIP member).
Records URL, member name, SHA256 and byte size for official_weights_provenance.json.
"""

from __future__ import annotations

import hashlib
import json
import re
import struct
import sys
import zlib
from datetime import datetime
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
WEIGHTS_DIR = ROOT / "results" / "05_compare_cnn" / "weights"
OUT = ROOT / "results" / "05_compare_cnn"

DRIVE_FILE_ID = "1UgYviv2K6QPM1SCexqqab5-yTgwoAFEc"
SOURCE_PAGE = "https://github.com/BMEII-AI/RadImageNet"
WANT = re.compile(r"resnet50.*notop.*\.h5$", re.I)


def open_stream(session: requests.Session) -> str:
    """Pass the Google Drive virus-scan confirm page; return a Range-capable URL."""
    r = session.get("https://drive.usercontent.google.com/download",
                    params={"id": DRIVE_FILE_ID, "export": "download"}, timeout=60)
    r.raise_for_status()
    action = re.search(r'<form[^>]+action="([^"]+)"', r.text)
    fields = dict(re.findall(r'<input type="hidden" name="([^"]+)" value="([^"]*)"', r.text))
    url = action.group(1) if action else "https://drive.usercontent.google.com/download"
    req = requests.Request("GET", url, params=fields).prepare()
    return req.url


def fetch_range(session: requests.Session, url: str, start: int, end: int) -> bytes:
    r = session.get(url, headers={"Range": f"bytes={start}-{end}"}, timeout=180)
    if r.status_code not in (200, 206):
        raise SystemExit(f"range request failed: {r.status_code}")
    return r.content


def total_size(session: requests.Session, url: str) -> int:
    r = session.get(url, headers={"Range": "bytes=0-0"}, timeout=60)
    r.raise_for_status()
    return int(r.headers["Content-Range"].split("/")[-1])


def find_central_directory(session, url, size):
    """Read the ZIP tail, parse EOCD (ZIP64 if needed); return (cd_offset, cd_size)."""
    tail_len = min(1 << 20, size)
    tail = fetch_range(session, url, size - tail_len, size - 1)
    i = tail.rfind(b"PK\x05\x06")
    if i < 0:
        raise SystemExit("EOCD not found")
    cd_size, cd_off = struct.unpack("<II", tail[i + 12:i + 20])
    if cd_off == 0xFFFFFFFF or cd_size == 0xFFFFFFFF:
        j = tail.rfind(b"PK\x06\x06")
        if j < 0:
            raise SystemExit("ZIP64 EOCD not found")
        cd_size, cd_off = struct.unpack("<QQ", tail[j + 40:j + 56])
    return cd_off, cd_size


def parse_central_directory(cd: bytes):
    entries, p = [], 0
    while p + 46 <= len(cd) and cd[p:p + 4] == b"PK\x01\x02":
        (method, _t, _d, crc, csize, usize, nlen, elen, clen,
         _dsk, _ia, _ea, lho) = struct.unpack("<HHHIIIHHHHHII", cd[p + 10:p + 46])
        name = cd[p + 46:p + 46 + nlen].decode("utf-8", "replace")
        extra = cd[p + 46 + nlen:p + 46 + nlen + elen]
        if 0xFFFFFFFF in (csize, usize, lho):        # ZIP64 extra field
            q = 0
            while q + 4 <= len(extra):
                hid, hsz = struct.unpack("<HH", extra[q:q + 4])
                if hid == 0x0001:
                    vals, r = [], q + 4
                    for cur in (usize, csize, lho):
                        if cur == 0xFFFFFFFF and r + 8 <= q + 4 + hsz:
                            vals.append(struct.unpack("<Q", extra[r:r + 8])[0]); r += 8
                        else:
                            vals.append(cur)
                    usize, csize, lho = vals
                    break
                q += 4 + hsz
        entries.append({"name": name, "method": method, "crc": crc,
                        "csize": csize, "usize": usize, "local_header_offset": lho})
        p += 46 + nlen + elen + clen
    return entries


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    s = requests.Session()
    url = open_stream(s)
    size = total_size(s, url)
    print(f"official zip: {size:,} bytes")

    cd_off, cd_size = find_central_directory(s, url, size)
    print(f"central directory @ {cd_off:,} ({cd_size:,} bytes)")
    entries = parse_central_directory(fetch_range(s, url, cd_off, cd_off + cd_size - 1))
    print(f"{len(entries)} zip members")
    h5s = [e for e in entries if e["name"].lower().endswith(".h5")]
    for e in h5s:
        print(f"   {e['name']}  ({e['usize']:,} B)")

    target = next((e for e in entries if WANT.search(e["name"])), None)
    if target is None:
        raise SystemExit(f"no ResNet50 notop .h5 among: {[e['name'] for e in h5s]}")
    print(f"\nselected: {target['name']}  compressed {target['csize']:,} B")

    lh = fetch_range(s, url, target["local_header_offset"], target["local_header_offset"] + 29)
    if lh[:4] != b"PK\x03\x04":
        raise SystemExit("bad local file header")
    nlen, elen = struct.unpack("<HH", lh[26:30])
    data_off = target["local_header_offset"] + 30 + nlen + elen
    blob = fetch_range(s, url, data_off, data_off + target["csize"] - 1)
    if len(blob) != target["csize"]:
        raise SystemExit(f"short read: {len(blob)} != {target['csize']}")
    raw = zlib.decompress(blob, -15) if target["method"] == 8 else blob
    if len(raw) != target["usize"]:
        raise SystemExit(f"size mismatch after inflate: {len(raw)} != {target['usize']}")
    if zlib.crc32(raw) & 0xFFFFFFFF != target["crc"]:
        raise SystemExit("CRC mismatch")
    print(f"inflated {len(raw):,} B, CRC OK")

    out_h5 = WEIGHTS_DIR / Path(target["name"]).name
    out_h5.write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()

    prov = {
        "source_repo": SOURCE_PAGE,
        "drive_file_id": DRIVE_FILE_ID,
        "zip_name": "RadImageNet_models-20230414T114049Z-001.zip",
        "zip_bytes": size,
        "zip_member": target["name"],
        "extraction": "HTTP Range read of the ZIP central directory; only this member fetched",
        "file": str(out_h5),
        "bytes": len(raw),
        "sha256": digest,
        "crc32_verified": True,
        "all_h5_members_in_zip": [e["name"] for e in h5s],
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    (OUT / "official_weights_provenance.json").write_text(
        json.dumps(prov, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {out_h5}")
    print(f"sha256 {digest}")


if __name__ == "__main__":
    sys.exit(main())
