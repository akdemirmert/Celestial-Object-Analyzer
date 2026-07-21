# -*- coding: utf-8 -*-
"""Basin kutuphanesindeki her referansin YAYINCI WCS'ini (AVM) topla.

djangoplicity arsivleri (esahubble/esawebb/eso/noirlab) her goruntunun
/api/json/ sayfasinda Spatial.* alanlarini yayinlar: merkez RA/Dec, piksel
olcegi, donme, TAN projeksiyon - yani tam WCS, yayincinin kendisinden.
Gorsel eslesen bir yukleme bu WCS'i miras alinca uzay-teleskobu yakin
cekimleri de tek tek isimlenebilir (press-avm cozucu katmani).

Cikti: data/visual_index/avm_wcs.json  {ref_adi(.jpg'siz): {...}}
Devam ettirilebilir: mevcut anahtarlar (basarisizlar dahil) atlanir.
"""
import concurrent.futures as cf
import json
import sys
import threading
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
VI = ROOT / "data" / "visual_index"
OUT = VI / "avm_wcs.json"

URL = {
    "esahubble": "https://esahubble.org/images/{}/api/json/",
    "esawebb": "https://esawebb.org/images/{}/api/json/",
    "eso": "https://www.eso.org/public/images/{}/api/json/",
    "noirlab": "https://noirlab.edu/public/images/{}/api/json/",
}
WORKERS = 12

_lock = threading.Lock()


def fetch(job) -> tuple[str, dict | None]:
    archive, img_id, ref = job
    key = ref[:-4] if ref.lower().endswith(".jpg") else ref
    try:
        r = httpx.get(URL[archive].format(img_id), timeout=45,
                      follow_redirects=True)
        if r.status_code != 200:
            return key, {"quality": None, "err": f"http {r.status_code}"}
        d = r.json()
        rv = d.get("Spatial.ReferenceValue")
        sc = d.get("Spatial.Scale")
        rp = d.get("Spatial.ReferencePixel")
        dim = d.get("Spatial.ReferenceDimension")
        if not (rv and sc and rp and dim):
            return key, {"quality": None}
        return key, {
            "archive": archive, "id": img_id,
            "quality": d.get("Spatial.Quality"),
            "proj": d.get("Spatial.CoordsystemProjection"),
            "ra": float(rv[0]), "dec": float(rv[1]),
            "scale": [float(sc[0]), float(sc[1])],
            "crpix": [float(rp[0]), float(rp[1])],
            "dim": [float(dim[0]), float(dim[1])],
            "rot": float(d.get("Spatial.Rotation") or 0.0),
        }
    except Exception as e:
        return key, {"quality": None, "err": repr(e)[:80]}


def main() -> None:
    manifest = json.loads((VI / "archive_manifest.json").read_text("utf-8"))
    out: dict = {}
    if OUT.exists():
        out = json.loads(OUT.read_text("utf-8"))
    jobs = []
    for archive, items in manifest.items():
        if archive not in URL:
            continue
        for img_id, ref in items.items():
            if not isinstance(ref, str) or ref == "isimsiz":
                continue
            key = ref[:-4] if ref.lower().endswith(".jpg") else ref
            if key in out:
                continue
            jobs.append((archive, img_id, ref))
    # oncelik: bu gecenin test hedefleri en one
    jobs.sort(key=lambda j: 0 if ("Cha" in j[2] or "NGC_4254" in j[2]
                                  or "NGC_628" in j[2]) else 1)
    print(f"toplam {len(jobs)} referans cekilecek "
          f"(mevcut {len(out)})", flush=True)
    done = 0
    with cf.ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for key, entry in ex.map(fetch, jobs):
            with _lock:
                out[key] = entry
                done += 1
                if done % 200 == 0 or done == len(jobs):
                    OUT.write_text(json.dumps(out), "utf-8")
                    ok = sum(1 for v in out.values() if v.get("quality"))
                    print(f"[{done}/{len(jobs)}] kayitli {len(out)} "
                          f"(WCS'li {ok})", flush=True)
    OUT.write_text(json.dumps(out), "utf-8")
    ok = sum(1 for v in out.values() if v.get("quality"))
    print(f"BITTI: {len(out)} kayit, {ok} tam WCS", flush=True)


if __name__ == "__main__":
    sys.exit(main())
