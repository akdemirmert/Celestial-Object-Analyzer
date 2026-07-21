"""Uniform sky-tile library: the local 'mini astrometry.net' image base.

The WHOLE sky as ~21k overlapping 2-degree DSS2 tiles (1.4-degree grid step).
Unlike per-object cutouts, any 0.5-6 degree query field shares its FULL point
set with the tile that contains it - which is what makes geometric quad
voting actually work. Resume-safe, 10 parallel workers, 429-aware.

Output: data/sky_tiles/img/t<index>.jpg + tiles.csv (index, ra, dec, fov)
"""
import csv
import math
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "sky_tiles"
IMG = OUT / "img"
IMG.mkdir(parents=True, exist_ok=True)

FOV = 2.0
STEP = 1.4
SIZE = 256
WORKERS = 10


def grid() -> list[tuple[int, float, float]]:
    tiles = []
    i = 0
    dec = -90.0 + STEP / 2
    while dec < 90.0:
        cosd = max(math.cos(math.radians(dec)), 0.05)
        n_ra = max(int(math.ceil(360.0 * cosd / STEP)), 1)
        for k in range(n_ra):
            tiles.append((i, k * 360.0 / n_ra, dec))
            i += 1
        dec += STEP
    return tiles


def main() -> None:
    tiles = grid()
    print(f"{len(tiles)} karo hedefi")
    meta_path = OUT / "tiles.csv"
    write_header = not meta_path.exists()
    todo = [t for t in tiles if not (IMG / f"t{t[0]}.jpg").exists()]
    print(f"mevcut {len(tiles) - len(todo)}, kalan {len(todo)}", flush=True)
    lock = threading.Lock()
    slow = threading.Event()
    done = fail = scanned = 0
    with (httpx.Client(timeout=60, follow_redirects=True) as client,
          open(meta_path, "a", newline="", encoding="utf-8") as mf):
        w = csv.writer(mf)
        if write_header:
            w.writerow(["index", "ra", "dec", "fov_deg"])

        def fetch(t) -> None:
            nonlocal done, fail, scanned
            idx, ra, dec = t
            dest = IMG / f"t{idx}.jpg"
            url = ("https://alasky.cds.unistra.fr/hips-image-services/hips2fits"
                   f"?hips=CDS%2FP%2FDSS2%2Fcolor&width={SIZE}&height={SIZE}"
                   f"&fov={FOV:g}&projection=TAN&coordsys=icrs"
                   f"&ra={ra:.4f}&dec={dec:.4f}&format=jpg")
            ok = False
            try:
                if slow.is_set():
                    time.sleep(1.5)
                r = client.get(url)
                if r.status_code == 429:
                    slow.set()
                    time.sleep(5.0)
                    r = client.get(url)
                if r.status_code == 200 and len(r.content) > 2000 \
                        and r.headers.get("content-type", "").startswith("image"):
                    dest.write_bytes(r.content)
                    ok = True
            except Exception:
                time.sleep(2.0)
            with lock:
                scanned += 1
                if ok:
                    done += 1
                    w.writerow([idx, f"{ra:.4f}", f"{dec:.4f}", FOV])
                    mf.flush()
                else:
                    fail += 1
                if done in (1, 10) or scanned % 500 == 0:
                    print(f"  {done} yeni ({scanned}/{len(todo)}, {fail} hata)"
                          + (" [yavas]" if slow.is_set() else ""), flush=True)

        with ThreadPoolExecutor(WORKERS) as ex:
            list(ex.map(fetch, todo))
    print(f"BITTI: +{done} yeni, {fail} hata; toplam "
          f"{len(list(IMG.glob('*.jpg')))}/{len(tiles)}")


if __name__ == "__main__":
    main()
