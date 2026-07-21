"""Geometric pattern index over ALL reference cutouts (astrometry.net's quad
idea applied to our own 13,962-object library).

For each reference image: brightest 8 blobs -> every 4-point subset -> a
scale/rotation-invariant 4D code (the two most distant points define the
frame; the other two, expressed in it, ARE the code). At query time the same
codes from the photo vote for candidate references directly by GEOMETRY -
appearance embeddings ranked the correct field #7,834 of 13,962 for a framing
mismatch; geometry does not care about framing.

Output: data/visual_index/pattern_index.npz  (codes float32 [N,4], refs int32
[N], names list)
"""
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.pipeline.visualmatch import INDEX_DIR, _blobs  # noqa: E402


def quad_codes(pts: np.ndarray) -> np.ndarray:
    """All valid canonical 4-point codes for one blob set."""
    out = []
    n = len(pts)
    for a, b, c, d in combinations(range(n), 4):
        quad = pts[[a, b, c, d]]
        # frame = the most distant pair
        dist = ((quad[:, None, :] - quad[None, :, :]) ** 2).sum(-1)
        i, j = np.unravel_index(np.argmax(dist), dist.shape)
        others = [k for k in range(4) if k not in (i, j)]
        A, B = quad[i], quad[j]
        L2 = float(dist[i, j])
        if L2 < 3600:  # frame < 60 px -> relative jitter sinks the code
            continue
        AB = B - A
        # coordinates in the (AB, AB_perp) frame, normalized by |AB|
        perp = np.array([-AB[1], AB[0]])
        rel = (quad[others] - A)
        cx = float(rel[0] @ AB) / L2
        cy = float(rel[0] @ perp) / L2
        dx = float(rel[1] @ AB) / L2
        dy = float(rel[1] @ perp) / L2
        # canonical orientation: force cx+dx <= 1 (A<->B swap symmetry) and
        # cx <= dx (C<->D swap symmetry)
        if cx + dx > 1.0:
            cx, cy, dx, dy = 1 - cx, -cy, 1 - dx, -dy
        if cx > dx:
            cx, cy, dx, dy = dx, dy, cx, cy
        if not (-0.6 <= cy <= 0.6 and -0.6 <= dy <= 0.6
                and -0.2 <= cx <= 1.2 and -0.2 <= dx <= 1.2):
            continue
        out.append((cx, cy, dx, dy))
    return np.array(out, dtype=np.float32) if out else np.zeros((0, 4), np.float32)


def main() -> None:
    from PIL import Image

    files = sorted((INDEX_DIR / "img").glob("*.jpg"))
    print(f"{len(files)} referans taranacak")
    all_codes, all_refs, names = [], [], []
    t0 = time.time()
    for ri, f in enumerate(files):
        try:
            g = np.asarray(Image.open(f).convert("L"), dtype=np.float32) / 255.0
        except Exception:
            continue
        pts, _ = _blobs(g, n=8)
        if len(pts) < 4:
            continue
        codes = quad_codes(pts)
        if not len(codes):
            continue
        all_codes.append(codes)
        all_refs.append(np.full(len(codes), len(names), dtype=np.int32))
        names.append(f.stem)
        if (ri + 1) % 2000 == 0:
            print(f"  {ri + 1}/{len(files)} ({time.time() - t0:.0f}s)", flush=True)
    codes = np.vstack(all_codes)
    refs = np.concatenate(all_refs)
    np.savez_compressed(INDEX_DIR / "pattern_index.npz",
                        codes=codes, refs=refs,
                        names=np.array(names, dtype=object))
    print(f"BITTI: {len(names)} referans, {len(codes)} quad kodu, "
          f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
