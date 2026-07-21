"""Complete press-archive harvest: ESA/Hubble + ESA/Webb + ESO + NOIRLab.

Downloads EVERY observation image that names an astronomical object from the
four Data2Dome feeds into the visual index as PRESS_<Object>__<n>.jpg refs.
Resume-safe: data/visual_index/archive_manifest.json remembers every feed ID
already handled (downloaded OR skipped), so re-running continues where it
stopped. After each site completes, embeddings are rebuilt incrementally so
finished sites become usable immediately.

Usage: .venv/Scripts/python.exe scripts/collect_press_archives.py
"""
from __future__ import annotations

import csv
import json
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

# 14 parallel downloads ~= 25-35 req/s aggregate: 10-15x faster than serial
# while staying inside what a public CDN tolerates without banning the IP
# (a ban would kill the whole harvest - that is why not 100-200)
WORKERS = 14

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
IDX = ROOT / "data" / "visual_index"
IMG = IDX / "img"
META = IDX / "press_meta.csv"
MANIFEST = IDX / "archive_manifest.json"

SITES = [
    ("esahubble", "https://esahubble.org/images/d2d/"),
    ("esawebb", "https://esawebb.org/images/d2d/"),
    ("eso", "https://www.eso.org/public/images/d2d/"),
    ("noirlab", "https://noirlab.edu/public/images/d2d/"),
]

# non-observation content: hardware, people, renders, charts. Applied to the
# TITLE only - descriptions legitimately mention "telescope" for real photos.
BAD_TITLE = re.compile(
    r"artist|impression|illustrat|render|diagram|chart|infograph|logo|"
    r"anniversar|poster|banner|brochure|screenshot|simulat|animation|"
    r"telescope|observator|dome|mirror|instrument|detector|spacecraft|"
    r"satellite dish|launch|control room|laborator|clean room|facility|"
    r"team|staff|crowd|visitor|ceremony|conference|exhibit|event|"
    r"groundbreaking|inauguration|timelapse|time-lapse|fisheye|all-sky|"
    r"aerial|drone view|construction|paranal|la silla|armazones|kitt peak|"
    r"cerro|summit|panorama of the|sunset|sunrise|moonset|moonrise over",
    re.I)

# catalog designations recognizable inside a title when Subject.Name is empty
TITLE_ID = re.compile(
    r"\b(NGC\s?\d{1,4}|IC\s?\d{1,4}|Messier\s?\d{1,3}|M\s?\d{1,3}(?!\d)|"
    r"Arp\s?\d{1,3}|UGC\s?\d{1,5}|HCG\s?\d{1,3}|Abell\s?\d{1,4}|"
    r"Barnard\s?\d{1,3}|Sh2-\d{1,3}|Gum\s?\d{1,2}|RCW\s?\d{1,3}|"
    r"Omega Centauri|Antennae|Whirlpool|Sombrero|Crab Nebula|Orion Nebula|"
    r"Carina Nebula|Eagle Nebula|Lagoon Nebula|Trifid|Helix Nebula|"
    r"Ring Nebula|Tarantula|Centaurus A|Pinwheel|Andromeda|Triangulum|"
    r"Cartwheel|Stephan.s Quintet|Butterfly Nebula|Horsehead|Veil Nebula|"
    r"Bubble Nebula|Large Magellanic Cloud|Small Magellanic Cloud)\b", re.I)

PER_OBJECT_CAP = 80
MIN_BYTES = 25_000
MAX_BYTES = 8_000_000


def safe_name(obj: str) -> str:
    s = obj.strip().replace("*", "s").replace("/", "_").replace("\\", "_")
    s = re.sub(r"[^A-Za-z0-9 _.+-]", "", s).replace(" ", "_")
    return s[:60] or "Unknown"


def pick_object(item: dict) -> str | None:
    subj = item.get("Subject") or {}
    names = subj.get("Name") or []
    if names and isinstance(names, list) and str(names[0]).strip():
        return str(names[0]).strip()
    m = TITLE_ID.search(item.get("Title") or "")
    return m.group(0).strip() if m else None


def pick_url(item: dict) -> str | None:
    order = {"Small": 0, "Screen": 1, "Medium": 2, "Large": 3}
    best = None
    for a in item.get("Assets") or []:
        for r in a.get("Resources") or []:
            rt, url = r.get("ResourceType"), r.get("URL")
            if not url or not str(url).lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            rank = order.get(rt)
            if rank is None:
                continue
            if best is None or rank < best[0]:
                best = (rank, url)
    return best[1] if best else None


def main() -> None:
    IMG.mkdir(parents=True, exist_ok=True)
    manifest: dict = (json.loads(MANIFEST.read_text(encoding="utf-8"))
                      if MANIFEST.exists() else {})

    # per-object counters + caps from the existing library
    counters: dict[str, int] = {}
    if META.exists():
        with open(META, encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                stem = Path(row["file"]).stem
                pre = stem[6:].rsplit("__", 1)[0]
                counters[pre] = counters.get(pre, 0) + 1

    from app.pipeline import ingest, ml_gate  # lazy heavy imports

    def cnn_ok(data: bytes) -> bool:
        try:
            im = ingest.load_image(data, max_dim=512)
            p = ml_gate.astro_probability(im.rgb)
            return p is None or p >= 0.35
        except Exception:
            return False

    meta_fh = open(META, "a", newline="", encoding="utf-8")
    meta_w = csv.writer(meta_fh)
    client = httpx.Client(follow_redirects=True, timeout=60,
                          headers={"User-Agent": "Mozilla/5.0 (CelestialAnalyzer archive mirror; contact local)"})

    def save_manifest():
        MANIFEST.write_text(json.dumps(manifest), encoding="utf-8")

    lock = threading.Lock()

    for site, feed in SITES:
        done: dict = manifest.setdefault(site, {})
        if done.get("__complete__"):
            print(f"[{site}] zaten tam - atlandi", flush=True)
            continue
        url = feed
        page = 0
        added = skipped = errors = 0
        t0 = time.time()

        def handle(it):
            nonlocal added, skipped, errors
            fid = str(it.get("ID") or "")
            title = it.get("Title") or ""
            if BAD_TITLE.search(title):
                with lock:
                    done[fid] = "baslik"
                    skipped += 1
                return
            obj = pick_object(it)
            if not obj:
                with lock:
                    done[fid] = "isimsiz"
                    skipped += 1
                return
            pre = safe_name(obj)
            with lock:
                if counters.get(pre, 0) >= PER_OBJECT_CAP:
                    done[fid] = "tavan"
                    skipped += 1
                    return
            src = pick_url(it)
            if not src:
                with lock:
                    done[fid] = "url-yok"
                    skipped += 1
                return
            try:
                data = client.get(src).content
            except Exception:
                with lock:
                    done[fid] = "indirme"
                    errors += 1
                return
            if not MIN_BYTES <= len(data) <= MAX_BYTES:
                with lock:
                    done[fid] = "boyut"
                    skipped += 1
                return
            if not cnn_ok(data):
                with lock:
                    done[fid] = "cnn"
                    skipped += 1
                return
            with lock:
                n = counters.get(pre, 0)
                if n >= PER_OBJECT_CAP:
                    done[fid] = "tavan"
                    skipped += 1
                    return
                fn = f"PRESS_{pre}__{n}.jpg"
                counters[pre] = n + 1
                (IMG / fn).write_bytes(data)
                meta_w.writerow([fn, obj])
                meta_fh.flush()
                done[fid] = fn
                added += 1
                if added % 100 == 0:
                    save_manifest()
                    el = time.time() - t0
                    print(f"[{site}] +{added} indirildi / {skipped} atlandi "
                          f"({el:.0f}sn, sayfa {page}, {added / max(el, 1):.1f}/sn)",
                          flush=True)

        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            while url:
                page += 1
                try:
                    d = client.get(url).json()
                except Exception as exc:
                    print(f"[{site}] sayfa {page} HATA {type(exc).__name__} - 30sn bekle", flush=True)
                    time.sleep(30)
                    errors += 1
                    if errors > 40:
                        break
                    continue
                items = [it for it in (d.get("Collections") or [])
                         if str(it.get("ID") or "") and str(it.get("ID")) not in done]
                list(pool.map(handle, items))
                url = d.get("Next")
        done["__complete__"] = True
        save_manifest()
        print(f"[{site}] TAMAM: +{added} yeni, {skipped} atlandi, "
              f"{time.time() - t0:.0f}sn", flush=True)
        # bu site kullanilabilir olsun: embeddings'i hemen guncelle
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / "build_visual_embeddings.py")],
                           capture_output=True, text=True)
        print(f"[{site}] embeddings: {(r.stdout or r.stderr).strip()[-120:]}", flush=True)

    meta_fh.close()
    print("HEPSI TAMAM", flush=True)


if __name__ == "__main__":
    main()
