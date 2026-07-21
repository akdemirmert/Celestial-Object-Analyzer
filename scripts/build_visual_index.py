"""Build the visual identification index: one reference image per cataloged
deep-sky object, downloaded from real sky surveys (DSS2 via the free CDS
hips2fits service), for appearance-based identification of close-up photos
that cannot plate-solve.

Coverage = the ENTIRE OpenNGC catalog (all NGC + IC objects, ~13,900) - not a
hand-picked famous subset. Objects are processed brightest/largest first so a
useful index exists within minutes while the long tail downloads.

Resume-safe: re-running skips images that already exist. Rate-limited to stay
polite to the free CDS service.

Output:
  data/visual_index/img/<name>.jpg   256px reference cutouts
  data/visual_index/meta.csv         name, type, ra, dec, size_arcmin, fov_deg
"""
import csv
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "visual_index"
IMG = OUT / "img"
IMG.mkdir(parents=True, exist_ok=True)

HIPS = "CDS/P/DSS2/color"
SIZE = 256


def load_openngc() -> list[dict]:
    rows = []
    with open(ROOT / "data" / "NGC.csv", encoding="utf-8") as f:
        rdr = csv.DictReader(f, delimiter=";")
        for r in rdr:
            name = (r.get("Name") or "").strip()
            ra, dec = r.get("RA"), r.get("Dec")
            if not name or not ra or not dec:
                continue
            # sexagesimal HH:MM:SS.s / +DD:MM:SS -> degrees
            try:
                h, m, s = [float(x) for x in ra.split(":")]
                ra_deg = (h + m / 60 + s / 3600) * 15.0
                sign = -1.0 if dec.strip().startswith("-") else 1.0
                dd, dm, ds = [abs(float(x)) for x in dec.split(":")]
                dec_deg = sign * (dd + dm / 60 + ds / 3600)
            except (ValueError, AttributeError):
                continue
            try:
                majax = float(r.get("MajAx") or 0)  # arcmin
            except ValueError:
                majax = 0.0
            try:
                vmag = float(r.get("V-Mag") or 99)
            except ValueError:
                vmag = 99.0
            rows.append({
                "name": name, "type": (r.get("Type") or "").strip(),
                "ra": ra_deg, "dec": dec_deg, "majax": majax, "vmag": vmag,
                "common": (r.get("Common names") or "").strip(),
            })
    # famous first: big and bright objects lead the queue
    rows.sort(key=lambda r: (-(r["majax"] or 0), r["vmag"]))
    return rows


WORKERS = 10  # paralel istek sayisi; 429 gelirse kendini yavaslatir


def main() -> None:
    import threading
    from concurrent.futures import ThreadPoolExecutor

    objs = load_openngc()
    print(f"OpenNGC: {len(objs)} objects with coordinates")
    existing = len(list(IMG.glob("*.jpg")))
    todo = [o for o in objs
            if not (IMG / (o["name"].replace("/", "_") + ".jpg")).exists()]
    print(f"mevcut: {existing} referans zaten inmis - {len(todo)} kaldi, "
          f"{WORKERS} paralel indiriliyor", flush=True)
    meta_path = OUT / "meta.csv"
    write_header = not meta_path.exists()
    lock = threading.Lock()
    slow = threading.Event()  # 429 gorulunce herkes yavaslar
    done = fail = scanned = 0

    with (httpx.Client(timeout=60, follow_redirects=True) as client,
          open(meta_path, "a", newline="", encoding="utf-8") as mf):
        w = csv.writer(mf)
        if write_header:
            w.writerow(["name", "type", "ra", "dec", "majax_arcmin",
                        "fov_deg", "common"])

        def fetch(o) -> None:
            nonlocal done, fail, scanned
            dest = IMG / (o["name"].replace("/", "_") + ".jpg")
            fov = min(max((o["majax"] or 2.0) / 60.0 * 2.2, 0.05), 8.0)
            url = ("https://alasky.cds.unistra.fr/hips-image-services/hips2fits"
                   f"?hips={HIPS.replace('/', '%2F')}&width={SIZE}&height={SIZE}"
                   f"&fov={fov:g}&projection=TAN&coordsys=icrs"
                   f"&ra={o['ra']:.5f}&dec={o['dec']:.5f}&format=jpg")
            ok = False
            try:
                if slow.is_set():
                    time.sleep(1.5)
                r = client.get(url)
                if r.status_code == 429:
                    slow.set()  # servis sikildi: temkinli moda gec
                    time.sleep(5.0)
                    r = client.get(url)
                if r.status_code == 200 and len(r.content) > 3000 \
                        and r.headers.get("content-type", "").startswith("image"):
                    dest.write_bytes(r.content)
                    ok = True
            except Exception:
                time.sleep(2.0)
            with lock:
                scanned += 1
                if ok:
                    done += 1
                    w.writerow([o["name"], o["type"], f"{o['ra']:.5f}",
                                f"{o['dec']:.5f}", o["majax"], f"{fov:.3f}",
                                o["common"]])
                    mf.flush()
                else:
                    fail += 1
                if done in (1, 10) or scanned % 200 == 0:
                    print(f"  bu calistirmada {done} yeni "
                          f"({scanned}/{len(todo)} denendi, {fail} hata) - "
                          f"toplam {existing + done}"
                          + (" [yavas mod]" if slow.is_set() else ""), flush=True)

        with ThreadPoolExecutor(WORKERS) as ex:
            list(ex.map(fetch, todo))

    print(f"BITTI: {done} yeni referans, {fail} hata; "
          f"toplam {len(list(IMG.glob('*.jpg')))} goruntu")


if __name__ == "__main__":
    main()
