# -*- coding: utf-8 -*-
"""Astrometry.net 5000/5001 dar-alan indekslerini indir (devam ettirilebilir).

Serinin en dar iki kademesi: 5001 (~2.8-4' dortgenler) ve 5000 (~2-2.8').
Bunlar olmadan kor cozum ~0.2 derecenin altina inemiyor - unlu/rastgele
ayrimi olmadan TUM dar yakin cekimlerin genel cozumu bu tabani indirmek.
Nazik indirme: 5 paralel, mevcut-tam dosyalar atlanir (boyut dogrulamali).
"""
import concurrent.futures as cf
import sys
from pathlib import Path

import httpx

BASE = "https://data.astrometry.net/5000/"
DEST = Path(__file__).resolve().parent.parent / "data" / "astrometry_index"
SERIES = ["5001", "5000"]  # 5001 once: 5002 ile boslugu kopruler
WORKERS = 5

def fetch_one(name: str) -> str:
    url = BASE + name
    dest = DEST / name
    try:
        with httpx.stream("GET", url, timeout=None, follow_redirects=True) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            if dest.exists() and total and dest.stat().st_size == total:
                return f"SKIP {name}"
            tmp = dest.with_suffix(".part")
            with open(tmp, "wb") as f:
                for chunk in r.iter_bytes(chunk_size=1 << 20):
                    f.write(chunk)
            if total and tmp.stat().st_size != total:
                tmp.unlink(missing_ok=True)
                return f"EKSIK {name} (yeniden denenecek)"
            tmp.replace(dest)
            return f"OK {name} ({tmp_size_mb(total)} MB)" if total else f"OK {name}"
    except Exception as e:
        return f"HATA {name}: {e!r}"

def tmp_size_mb(b: int) -> int:
    return round(b / 1e6)

def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    for series in SERIES:
        names = [f"index-{series}-{i:02d}.fits" for i in range(48)]
        todo = []
        for n in names:
            p = DEST / n
            # boyutu bilinmeyen mevcut dosyayi da dogrulamak icin hepsi kuyruga
            todo.append(n)
        print(f"--- seri {series}: {len(todo)} dosya ---", flush=True)
        done = 0
        with cf.ThreadPoolExecutor(max_workers=WORKERS) as ex:
            for res in ex.map(fetch_one, todo):
                done += 1
                print(f"[{series} {done}/{len(todo)}] {res}", flush=True)
    print("BITTI", flush=True)

if __name__ == "__main__":
    sys.exit(main())
