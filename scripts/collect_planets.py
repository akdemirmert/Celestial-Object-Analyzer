"""Collect FULL-DISK planet/moon positives from the NASA Image Library.

The gate learned to reject 'large colorful sphere on black' because vignetted
COCO balls look exactly like that - and the positive set held ~2000 galaxies
but almost no full-disk planets (Mars mosaic scored P=0.045). Only the CNN can
tell Mars from a disco ball - both are crisp circles - it just needs to have
SEEN planets.
"""
import time
from pathlib import Path

import httpx

OUT = Path(__file__).resolve().parent.parent / "data" / "astro"
OUT.mkdir(parents=True, exist_ok=True)

QUERIES = [
    "jupiter full disk", "jupiter hubble", "jupiter voyager",
    "mars full disk", "mars globe", "mars hubble",
    "saturn full disk", "saturn cassini", "saturn voyager",
    "neptune full disk", "neptune voyager", "uranus voyager",
    "venus full disk", "mercury messenger full", "pluto new horizons",
    "full moon photo", "moon full disk", "earth full disk apollo",
    "ganymede full disk", "europa full disk", "io jupiter moon",
    "titan cassini", "ceres dawn", "vesta dawn",
]
PER_QUERY = 25

total = 0
with httpx.Client(timeout=40, follow_redirects=True) as client:
    for q in QUERIES:
        try:
            r = client.get("https://images-api.nasa.gov/search",
                           params={"q": q, "media_type": "image"})
            items = r.json()["collection"]["items"][:PER_QUERY]
        except Exception as exc:
            print(f"  [{q:<26}] failed: {exc}")
            continue
        got = 0
        for item in items:
            data = (item.get("data") or [{}])[0]
            nasa_id = (data.get("nasa_id") or "")[:60].replace("/", "_")
            href = next((l.get("href") for l in (item.get("links") or [])
                         if l.get("href", "").lower().endswith((".jpg", ".jpeg"))), None)
            if not href or not nasa_id:
                continue
            dest = OUT / f"planet_{nasa_id}.jpg"
            if dest.exists() or (OUT / f"{nasa_id}.jpg").exists():
                continue
            try:
                img = client.get(href).content
                if len(img) > 5000:
                    dest.write_bytes(img)
                    got += 1
            except Exception:
                continue
            time.sleep(0.1)
        total += got
        print(f"  [{q:<26}] +{got}")
print(f"TOTAL new planet positives: {total}")
