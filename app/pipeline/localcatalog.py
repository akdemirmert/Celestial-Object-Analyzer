"""Offline deep-sky catalog: OpenNGC (data/NGC.csv, user-downloaded).

Source: https://github.com/mattiaverga/OpenNGC (CC-BY-SA-4.0) - 13k+ NGC/IC
objects with authoritative types, positions, sizes and magnitudes. Optional:
the app works without the file; SIMBAD (online) remains the primary source.
"""
from __future__ import annotations

import csv
import math
import threading
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "NGC.csv"

TYPE_LABELS = {
    "G": "Galaxy",
    "GPair": "Galaxy pair",
    "GTrpl": "Galaxy triplet",
    "GGroup": "Galaxy group",
    "OCl": "Open cluster",
    "GCl": "Globular cluster",
    "Cl+N": "Cluster with nebulosity",
    "PN": "Planetary nebula",
    "HII": "H II region (emission nebula)",
    "EmN": "Emission nebula",
    "Neb": "Nebula",
    "RfN": "Reflection nebula",
    "SNR": "Supernova remnant",
    "DrkN": "Dark nebula",
    "*": "Star",
    "**": "Double star",
    "*Ass": "Stellar association",
    "NonEx": "Non-existent (catalog error)",
    "Dup": "Duplicate entry",
    "Other": "Other object",
}

_cache: list[dict] | None = None
_lock = threading.Lock()


def _hms_to_deg(hms: str) -> float | None:
    try:
        h, m, s = hms.split(":")
        return (float(h) + float(m) / 60 + float(s) / 3600) * 15.0
    except (ValueError, AttributeError):
        return None


def _dms_to_deg(dms: str) -> float | None:
    try:
        sign = -1.0 if dms.strip().startswith("-") else 1.0
        d, m, s = dms.strip().lstrip("+-").split(":")
        return sign * (float(d) + float(m) / 60 + float(s) / 3600)
    except (ValueError, AttributeError):
        return None


def available() -> bool:
    return DATA_PATH.exists()


def _load() -> list[dict]:
    global _cache
    with _lock:
        if _cache is not None:
            return _cache
        rows: list[dict] = []
        if DATA_PATH.exists():
            try:
                with open(DATA_PATH, encoding="utf-8") as fh:
                    for rec in csv.DictReader(fh, delimiter=";"):
                        if rec.get("Type") in ("NonEx", "Dup"):
                            continue
                        ra = _hms_to_deg(rec.get("RA", ""))
                        dec = _dms_to_deg(rec.get("Dec", ""))
                        if ra is None or dec is None:
                            continue
                        rows.append({
                            "name": rec.get("Name", ""),
                            "type": rec.get("Type", ""),
                            "ra": ra,
                            "dec": dec,
                            "major_arcmin": _float(rec.get("MajAx")),
                            "v_mag": _float(rec.get("V-Mag")) or _float(rec.get("B-Mag")),
                            "messier": rec.get("M", "").strip(),
                            "common_names": rec.get("Common names", "").strip(),
                        })
            except (OSError, csv.Error):
                rows = []
        _cache = rows
        return rows


def _float(v) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def cone_search(ra: float, dec: float, radius_deg: float, limit: int = 10) -> list[dict]:
    """Nearest cataloged objects; big/bright objects ranked first."""
    rows = _load()
    if not rows:
        return []
    cos_dec = math.cos(math.radians(dec))
    hits = []
    for r in rows:
        d_ra = (r["ra"] - ra) * cos_dec
        d_dec = r["dec"] - dec
        sep = math.hypot(d_ra, d_dec)
        if sep <= radius_deg:
            hits.append((sep, r))
    hits.sort(key=lambda t: (-(t[1]["major_arcmin"] or 0), t[0]))
    out = []
    for sep, r in hits[:limit]:
        name = r["name"]
        if r["messier"]:
            name = f"M{int(r['messier'])} ({name})"
        if r["common_names"]:
            name = f"{name} - {r['common_names']}"
        out.append({
            "name": name,
            "type_label": TYPE_LABELS.get(r["type"], r["type"]),
            "ra": round(r["ra"], 4),
            "dec": round(r["dec"], 4),
            "angular_size_arcmin": r["major_arcmin"],
            "v_mag": r["v_mag"],
            "center_distance_deg": round(sep, 4),
            "source": "OpenNGC (offline)",
        })
    return out
