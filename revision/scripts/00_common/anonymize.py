#!/usr/bin/env python3
"""
Replace patient names in strings with anonymous P### identifiers.
Reads the offline name-to-ID map from the PHI_MAP environment variable.
Not used by modelling scripts that already consume de-identified tables.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path

import pandas as pd

PHI_MAP = Path(os.environ.get("PHI_MAP", "<not published>"))
RAD = Path(os.environ.get("IMAGE_ROOT", "<image_root>"))
BATCHES = ["employee", "student", "patient1", "patient2", "patient3"]


class Redactor:
    """Replace patient names with anonymous IDs."""

    def __init__(self, phi_map: Path = PHI_MAP) -> None:
        if not phi_map.exists():
            raise SystemExit(f"[error] missing {phi_map}; set PHI_MAP or run make_labels.py")
        m = pd.read_excel(phi_map)
        self.map: dict[str, str] = dict(zip(m["name"].astype(str), m["patient_id"].astype(str)))

        extra = set()
        for b in BATCHES:
            d = RAD / b / f"{b}_L3-4_label"
            if d.is_dir():
                for f in glob.glob(os.path.join(str(d), "*.nrrd")):
                    extra.add(os.path.basename(f).split("_L3-4_")[0])
        for i, n in enumerate(sorted(extra - set(self.map)), 1):
            self.map[n] = f"N{i:03d}"

        self._ordered = sorted(self.map.items(), key=lambda kv: -len(kv[0]))

    def text(self, s: str) -> str:
        """Replace every patient name in the string."""
        if not isinstance(s, str):
            return s
        for name, pid in self._ordered:
            if name in s:
                s = s.replace(name, pid)
        return s

    def series(self, s: pd.Series) -> pd.Series:
        return s.map(self.text)

    def name_of(self, pid: str) -> str:
        """Look up the real name for a P### id. Use only to locate files on disk; never write the name to outputs."""
        for name, p in self.map.items():
            if p == pid:
                return name
        raise KeyError(f"unknown id {pid}")

    def check(self, s: str) -> list[str]:
        """Return names still present in the string (including case/space variants). Empty means clean."""
        hits = []
        for name in self.map:
            for v in {name, name.lower(), name.replace(" ", ""), name.replace(" ", "").lower()}:
                if v and v in s:
                    hits.append(name)
                    break
        return sorted(set(hits))
