"""Find pixel-duplicate press refs living under DIFFERENT object names.

Archives reuse the same frame across releases (the Antennae portrait was also
published inside the SN UDS10Wil release and harvested under that name), which
splits the visual vote and kills the identification gate's lead condition.

For every pair of refs with embedding cosine >= 0.995 under different objects,
suppress the copy whose object is the worse owner (no local coordinates, then
fewer total refs). Suppressed filenames go to suppressed_refs.txt - the index
loader skips them; nothing is deleted.

Usage: .venv/Scripts/python.exe scripts/dedupe_press_refs.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
IDX = ROOT / "data" / "visual_index"
OUT = IDX / "suppressed_refs.txt"
DUP_COS = 0.995


def main() -> None:
    from app.pipeline import visualmatch as vm

    z = np.load(IDX / "embeddings.npz", allow_pickle=True)
    names = list(z["names"])
    vecs = z["vecs"].astype(np.float32)

    press = [(i, n) for i, n in enumerate(names) if n.startswith("PRESS_")]
    idxs = np.array([i for i, _ in press])
    pvec = vecs[idxs]
    pobj = [n[6:].rsplit("__", 1)[0] for _, n in press]

    obj_count: dict[str, int] = {}
    for o in pobj:
        obj_count[o] = obj_count.get(o, 0) + 1

    def has_coords(obj: str) -> bool:
        disp = obj.replace("_", " ")
        idx = vm._load_index()
        coords = idx[3] if idx else {}
        for key in vm._coord_keys(obj, disp):
            if key in coords:
                return True
        return disp in vm._FAMOUS_COORDS

    coords_cache = {o: has_coords(o) for o in set(pobj)}

    suppressed: set[str] = set()
    n = len(press)
    chunk = 1500
    pairs = 0
    for a0 in range(0, n, chunk):
        a1 = min(a0 + chunk, n)
        sims = pvec[a0:a1] @ pvec.T
        ii, jj = np.nonzero(sims >= DUP_COS)
        for i_l, j in zip(ii, jj):
            i = a0 + i_l
            if j <= i:
                continue
            oi, oj = pobj[i], pobj[j]
            if oi == oj:
                continue
            ni, nj = press[i][1], press[j][1]
            if ni in suppressed or nj in suppressed:
                continue
            pairs += 1
            # keep the better owner: local coords first, then ref count
            ci, cj = coords_cache[oi], coords_cache[oj]
            if ci != cj:
                loser = nj if ci else ni
            elif obj_count[oi] != obj_count[oj]:
                loser = nj if obj_count[oi] > obj_count[oj] else ni
            else:
                loser = nj
            suppressed.add(loser)
            print(f"cift: {ni} <-> {nj} (cos {sims[i_l, j]:.4f}) -> bastir: {loser}",
                  flush=True)

    OUT.write_text("\n".join(sorted(suppressed)) + ("\n" if suppressed else ""),
                   encoding="utf-8")
    print(f"\nTAMAM: {pairs} farkli-adli cift, {len(suppressed)} referans bastirildi "
          f"-> {OUT.name}")


if __name__ == "__main__":
    main()
