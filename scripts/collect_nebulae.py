"""Collect BRIGHT, frame-filling nebula close-ups from the NASA Image Library.

The gate CNN learned "bright + colorful frame = not astro" because ordinary
positives sit on dark sky; Hubble's visible-light Pillars of Creation scored
P(astro)=0.004. These close-ups are the missing training mode. Staged into
data/neb_staging/ for a visual pass before joining data/astro/ as nebclose_*.
"""
import time
from pathlib import Path

import httpx

OUT = Path(__file__).resolve().parent.parent / "data" / "neb_staging"
OUT.mkdir(parents=True, exist_ok=True)

QUERIES = [
    "pillars of creation", "carina nebula", "eagle nebula", "lagoon nebula",
    "orion nebula close-up", "star forming region hubble", "tarantula nebula",
    "nebula webb telescope", "emission nebula", "nebula detail hubble",
    "monkey head nebula", "cosmic cliffs", "veil nebula", "swan nebula",
    "trifid nebula", "rosette nebula", "butterfly nebula", "crab nebula",
]
PER_QUERY = 30
# press/ceremony/diagram shots poison the positive set - skip by title
BAD_WORDS = ("astronaut", "crew", "launch", "people", "ceremony", "artist",
             "concept", "illustration", "exhibit", "model", "mirror",
             "clean room", "anniversary event", "director", "administrator",
             "visitors", "replica", "poster", "logo")

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
            text = (data.get("title", "") + " " + data.get("description", "")[:300]).lower()
            if any(wd in text for wd in BAD_WORDS):
                continue
            href = next((l.get("href") for l in (item.get("links") or [])
                         if l.get("href", "").lower().endswith((".jpg", ".jpeg", ".png"))), None)
            if not href or not nasa_id:
                continue
            dest = OUT / f"nebclose_{nasa_id[:60].replace('/', '_')}.jpg"
            if dest.exists():
                continue
            try:
                img = client.get(href).content
                if len(img) > 10000:
                    dest.write_bytes(img)
                    got += 1
            except Exception:
                continue
            time.sleep(0.15)
        total += got
        print(f"[{q}] +{got}")
print(f"TOTAL new: {total}, staging now has {len(list(OUT.glob('*.jpg')))} images")
