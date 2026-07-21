"""End-to-end local ASTAP solve test on a real sky photograph."""
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.pipeline import ingest, platesolve_local

query = sys.argv[1] if len(sys.argv) > 1 else "constellation orion night sky"
OUT = Path(__file__).resolve().parent.parent / "test_images" / "real_skyfield.jpg"

resp = httpx.get("https://images-api.nasa.gov/search",
                 params={"q": query, "media_type": "image"}, timeout=20)
items = resp.json()["collection"]["items"]
href = title = None
for item in items:
    for l in item.get("links") or []:
        if l.get("href", "").lower().endswith((".jpg", ".jpeg", ".png")):
            href, title = l["href"], item["data"][0].get("title", "")
            break
    if href:
        break
print(f"image: {title}\n  {href}")
data = httpx.get(href, timeout=60, follow_redirects=True).content
OUT.write_bytes(data)
print(f"saved {len(data)/1024:.0f} KB")

img = ingest.load_image(data)
jpeg = ingest.encode_analysis_jpeg(img)
t0 = time.time()
res = platesolve_local.solve(jpeg, img.width, img.height,
                             progress=lambda m: print("  .", m))
dt = time.time() - t0
print(f"\nsolved: {res.solved}  ({dt:.1f}s)")
if res.solved:
    print(f"  RA {res['ra']:.4f}  Dec {res['dec']:.4f}  "
          f"radius {res['radius_deg']:.3f} deg  "
          f"pixscale {res['pixscale_arcsec']:.2f} arcsec/px")
    print(f"  wcs bytes: {len(res.get('wcs_fits') or b'')}")
else:
    print("  error:", res.get("error"))
