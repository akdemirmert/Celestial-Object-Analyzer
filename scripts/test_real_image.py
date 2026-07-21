"""Fetch a real NASA image and run the offline pipeline on it."""
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.pipeline import analyzer, features, ingest
from app.pipeline.platesolve import PlateSolveResult

OUT = Path(__file__).resolve().parent.parent / "test_images"
query = sys.argv[1] if len(sys.argv) > 1 else "barred spiral galaxy hubble"
dest = OUT / "real_test.jpg"

resp = httpx.get("https://images-api.nasa.gov/search",
                 params={"q": query, "media_type": "image"}, timeout=20)
items = resp.json()["collection"]["items"]
href = None
title = None
for item in items:
    links = item.get("links") or []
    for l in links:
        if l.get("href", "").lower().endswith((".jpg", ".jpeg", ".png")):
            href = l["href"]
            title = item["data"][0].get("title", "")
            break
    if href:
        break

print(f"downloading: {title}\n  {href}")
img_data = httpx.get(href, timeout=60, follow_redirects=True).content
dest.write_bytes(img_data)
print(f"saved {len(img_data) / 1024:.0f} KB -> {dest}")

img = ingest.load_image(img_data)
feats = features.extract_features(img.rgb)
solve = PlateSolveResult(solved=False, skipped=True, error="offline test")
report = analyzer.analyze(feats, solve, [], None, img.exif)

print("=" * 70)
print(f"stars: {feats['star_count']}")
src = feats.get("main_source")
if src:
    print(f"main src: d={src['equiv_diameter_px']}px bright_d={src['bright_diameter_px']}px "
          f"elong={src['elongation']} circ={src['circularity']}")
    print(f"          conc={src['concentration']} limb={src['limb_sharpness']} "
          f"rb={src['rb_color_ratio']} sat={src['saturation']} tex={src['texture']}")
    print(f"flags: point={src['is_point_like']} disk={src['is_disk_like']} "
          f"fuzzy={src['is_extended_fuzzy']} streak={src['is_streak']}")
print(f"MODE     : {report['mode']}")
print(f"HEADLINE : {report['headline']}")
for h in report["hypotheses"]:
    print(f"  [{h['band']:^8} {h['score']:>3}] {h['label']}")
