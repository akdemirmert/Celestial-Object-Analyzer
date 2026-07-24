"""Curated EXOTIC-object reference gallery (rogue planets, brown dwarfs,
famous oddball stars): the frames people share for these are tiny survey
crops with a handful of faint stars - no blind solver can anchor them, and
the press archives never covered them.

The general mechanism: for each curated object, fetch survey cutouts from
CDS hips2fits (free) at several fields of view. Unlike harvested press
images, the WCS of a hips2fits cutout is EXACT by construction (we request
center/FOV/projection), so each reference enters avm_wcs.json as a
quality-Full entry and the existing press-avm machinery does the rest:
visual match -> pixel alignment -> inherited WCS -> identity injection ->
per-star naming.

Re-run safe: existing files are kept, only missing ones are fetched.
Afterwards run scripts/build_visual_embeddings.py (incremental).
"""
import csv
import json
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.pipeline import catalogs  # noqa: E402

INDEX_DIR = ROOT / "data" / "visual_index"
IMG = INDEX_DIR / "img"
PRESS_META = INDEX_DIR / "press_meta.csv"
AVM_PATH = INDEX_DIR / "avm_wcs.json"

H2F = "https://alasky.cds.unistra.fr/hips-image-services/hips2fits"
SIZE = 768
FOVS = [0.05, 0.12, 0.30]  # deg: tight finder crop -> wide context
PS1_LIMIT_DEC = -29.5      # Pan-STARRS coverage floor

# (display name, ident candidates tried in order; the FIRST that resolves
# becomes the reference object name, so runtime re-resolution is guaranteed)
EXOTICS = [
    ("PSO J318.5-22", ["PSO J318.5-22", "PSO J318.5338-22.8603"]),
    ("KIC 8462852", ["KIC 8462852"]),                    # Tabby's Star
    ("TRAPPIST-1", ["TRAPPIST-1"]),
    ("Proxima Centauri", ["Proxima Centauri"]),
    ("Barnard's star", ["Barnard's star", "GJ 699"]),
    ("Teegarden's Star", ["Teegarden's Star", "GAT 1370"]),
    ("Luhman 16", ["Luhman 16", "WISE J104915.57-531906.1"]),
    ("WISE J085510.83-071442.5", ["WISE J085510.83-071442.5"]),
    ("HD 101065", ["HD 101065"]),                        # Przybylski's Star
    ("Hoag's Object", ["Hoag's Object", "PGC 54559"]),
    ("Hanny's Voorwerp", ["Hanny's Voorwerp", "SDSS J094103.80+344334.2"]),
    ("OTS 44", ["OTS 44"]),
    ("Cygnus X-1", ["Cyg X-1", "Cygnus X-1"]),
    ("V404 Cygni", ["V404 Cyg"]),
    ("SS 433", ["SS 433"]),
]


def resolve(cands: list[str]) -> tuple[str, float, float] | None:
    for c in cands:
        try:
            o = catalogs.object_by_name(c, 0.0, 0.0)
        except Exception:
            o = None
        if o and o.get("ra") is not None:
            return c, float(o["ra"]), float(o["dec"])
        time.sleep(0.4)
    return None


def surveys_for(dec: float) -> list[tuple[str, str]]:
    out = []
    if dec > PS1_LIMIT_DEC:
        # two PS1 palettes: discovery finder charts circulate in both styles
        out.append(("CDS/P/PanSTARRS/DR1/color-z-zg-g", "ps1z"))
        out.append(("CDS/P/PanSTARRS/DR1/color-i-r-g", "ps1i"))
    out.append(("CDS/P/DSS2/color", "dss2"))
    return out


def main() -> None:
    IMG.mkdir(parents=True, exist_ok=True)
    avm = json.loads(AVM_PATH.read_text(encoding="utf-8")) if AVM_PATH.exists() else {}
    meta_rows = []
    added = 0

    with httpx.Client(timeout=60) as cli:
        for display, cands in EXOTICS:
            r = resolve(cands)
            if not r:
                print(f"COZULEMEDI (SIMBAD): {display} - atlandi")
                continue
            name, ra, dec = r
            stem = "PRESS_" + name.replace(" ", "_").replace("*", "s")
            n = 0
            for hips, tag in surveys_for(dec):
                for fov in FOVS:
                    fname = f"{stem}__{n}.jpg"
                    n += 1
                    dest = IMG / fname
                    key = dest.stem
                    if dest.exists():
                        if key not in avm:
                            avm[key] = _entry(hips, ra, dec, fov)
                        continue
                    try:
                        resp = cli.get(H2F, params={
                            "hips": hips, "ra": ra, "dec": dec, "fov": fov,
                            "width": SIZE, "height": SIZE,
                            "projection": "TAN", "format": "jpg"})
                        resp.raise_for_status()
                        if len(resp.content) < 4000:
                            raise ValueError("bos/kirik kesit")
                    except Exception as e:
                        print(f"  {fname}: indirilemedi ({e})")
                        continue
                    dest.write_bytes(resp.content)
                    avm[key] = _entry(hips, ra, dec, fov)
                    meta_rows.append((fname, name))
                    added += 1
                    print(f"  + {fname}  ({tag}, fov {fov} deg)")
                    time.sleep(0.3)
            print(f"{display} -> {name}  RA {ra:.4f} Dec {dec:.4f}")

    if meta_rows:
        exists = PRESS_META.exists()
        have = set()
        if exists:
            with open(PRESS_META, encoding="utf-8") as f:
                have = {row["file"] for row in csv.DictReader(f)}
        with open(PRESS_META, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if not exists:
                w.writerow(["file", "object"])
            for fname, obj in meta_rows:
                if fname not in have:
                    w.writerow([fname, obj])

    AVM_PATH.write_text(json.dumps(avm), encoding="utf-8")
    print(f"BITTI: {added} yeni referans. Simdi build_visual_embeddings.py calistir.")


def _entry(hips: str, ra: float, dec: float, fov: float) -> dict:
    # exact-by-construction TAN WCS of the requested cutout; same field
    # names the harvested archive entries use (see harvest_avm_wcs.py)
    step = fov / SIZE
    return {"archive": "hips2fits", "id": hips, "quality": "Full",
            "proj": "TAN", "ra": ra, "dec": dec,
            "scale": [-step, step],
            "crpix": [SIZE / 2.0, SIZE / 2.0],
            "dim": [float(SIZE), float(SIZE)], "rot": 0.0}


if __name__ == "__main__":
    main()
