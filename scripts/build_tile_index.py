"""Quad-code index over the uniform sky tiles: the searchable half of the
local mini-astrometry.net. Incremental: skips tiles already indexed."""
import csv
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from build_pattern_index import quad_codes  # noqa: E402
from app.pipeline.visualmatch import _blobs  # noqa: E402

TILES = ROOT / "data" / "sky_tiles"
OUT = TILES / "tile_index.npz"


def main() -> None:
    from PIL import Image

    meta = {}
    with open(TILES / "tiles.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            meta[int(r["index"])] = (float(r["ra"]), float(r["dec"]),
                                     float(r["fov_deg"]))
    files = sorted((TILES / "img").glob("t*.jpg"),
                   key=lambda p: int(p.stem[1:]))
    old_ids: list[int] = []
    old_codes = old_refs = None
    if OUT.exists():
        z = np.load(OUT)
        old_codes, old_refs = z["codes"], z["refs"]
        old_ids = list(z["tile_ids"])
    known = set(old_ids)
    todo = [f for f in files if int(f.stem[1:]) not in known]
    print(f"{len(files)} karo, {len(todo)} yeni indekslenecek")
    t0 = time.time()
    codes_l, refs_l, ids = [], [], list(old_ids)
    for i, f in enumerate(todo):
        tid = int(f.stem[1:])
        if tid not in meta:
            continue
        try:
            g = np.asarray(Image.open(f).convert("L"), dtype=np.float32) / 255.0
        except Exception:
            continue
        pts, _ = _blobs(g, n=12)
        if len(pts) < 4:
            continue
        c = quad_codes(pts)
        if not len(c):
            continue
        codes_l.append(c)
        refs_l.append(np.full(len(c), len(ids), dtype=np.int32))
        ids.append(tid)
        if (i + 1) % 3000 == 0:
            print(f"  {i + 1}/{len(todo)} ({time.time() - t0:.0f}s)", flush=True)
    new_codes = np.vstack(codes_l) if codes_l else np.zeros((0, 4), np.float32)
    new_refs = (np.concatenate(refs_l) if refs_l
                else np.zeros(0, np.int32))
    codes = (np.vstack([old_codes, new_codes])
             if old_codes is not None else new_codes)
    refs = (np.concatenate([old_refs, new_refs])
            if old_refs is not None else new_refs)
    ras = np.array([meta[t][0] for t in ids], dtype=np.float64)
    decs = np.array([meta[t][1] for t in ids], dtype=np.float64)
    fovs = np.array([meta[t][2] for t in ids], dtype=np.float64)
    np.savez_compressed(OUT, codes=codes.astype(np.float32), refs=refs,
                        tile_ids=np.array(ids, dtype=np.int64),
                        ras=ras, decs=decs, fovs=fovs)
    print(f"BITTI: {len(ids)} karo, {len(codes)} kod, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
