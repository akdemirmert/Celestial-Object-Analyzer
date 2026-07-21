"""Collect positive (astronomical) images from the NASA Image Library into
data/astro/. Diverse queries so the detector learns 'sky-ness', not one class."""
import sys
import time
from pathlib import Path

import httpx

OUT = Path(__file__).resolve().parent.parent / "data" / "astro"
OUT.mkdir(parents=True, exist_ok=True)

QUERIES = [
    "nebula hubble", "galaxy hubble", "star cluster hubble", "globular cluster",
    "milky way stars", "moon surface", "jupiter planet", "mars surface",
    "saturn rings", "comet", "supernova remnant", "night sky stars",
    "aurora station", "solar eclipse", "planetary nebula",
]
PER_QUERY = 40

total = 0
with httpx.Client(timeout=30, follow_redirects=True) as client:
    for q in QUERIES:
        try:
            resp = client.get("https://images-api.nasa.gov/search",
                              params={"q": q, "media_type": "image"})
            items = resp.json()["collection"]["items"][:PER_QUERY]
        except Exception as exc:
            print(f"[{q}] search failed: {exc}")
            continue
        got = 0
        for item in items:
            data = (item.get("data") or [{}])[0]
            nasa_id = data.get("nasa_id", "")
            href = next((l.get("href") for l in (item.get("links") or [])
                         if l.get("href", "").lower().endswith((".jpg", ".jpeg", ".png"))), None)
            if not href or not nasa_id:
                continue
            dest = OUT / f"{nasa_id[:60].replace('/', '_')}.jpg"
            if dest.exists():
                continue
            try:
                img = client.get(href).content
                if len(img) > 5000:
                    dest.write_bytes(img)
                    got += 1
            except Exception:
                continue
            time.sleep(0.15)  # be polite to the free API
        total += got
        print(f"[{q}] +{got}")
print(f"TOTAL new: {total}, folder now has {len(list(OUT.glob('*.jpg')))} images")
