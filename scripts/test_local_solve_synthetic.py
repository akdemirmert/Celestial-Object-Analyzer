"""Closed-loop local-solve validation: render the REAL sky from catalog
positions, then check ASTAP recovers the field center we rendered.

Renders bright stars (V<7, SIMBAD) around the Orion belt through a TAN
projection; a correct solve must return RA~84, Dec~-1.
"""
import io
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from astropy.wcs import WCS
from PIL import Image

from app.pipeline import platesolve_local, starid

RA0, DEC0 = 84.0, -1.0   # Orion belt area
FOV_W = 25.0             # degrees
W, H = 1000, 750

catalog = starid.field_star_catalog(RA0, DEC0, FOV_W * 0.75, limit=400)
stars = [c for c in catalog if c.get("vmag") is not None and c["vmag"] < 7.0]
print(f"catalog stars V<7 in field: {len(stars)}")

wcs = WCS(naxis=2)
wcs.wcs.crpix = [W / 2, H / 2]
wcs.wcs.crval = [RA0, DEC0]
scale = FOV_W / W
wcs.wcs.cdelt = [-scale, scale]
wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]

rng = np.random.default_rng(1)
img = rng.normal(0.03, 0.012, (H, W)).astype(np.float32)
yy, xx = np.mgrid[:H, :W]
n_drawn = 0
for s in stars:
    px, py = wcs.world_to_pixel_values(s["ra"], s["dec"])
    if not (5 < px < W - 5 and 5 < py < H - 5):
        continue
    amp = 10 ** (-0.4 * (s["vmag"] - 7.0)) * 0.25  # brighter star -> brighter blob
    amp = min(amp, 0.95)
    sigma = 1.6
    img += amp * np.exp(-((yy - py) ** 2 + (xx - px) ** 2) / (2 * sigma ** 2))
    n_drawn += 1
print(f"stars drawn: {n_drawn}")

rgb = np.stack([img, img, img * 1.05], axis=2)
pil = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))
buf = io.BytesIO()
pil.save(buf, format="JPEG", quality=92)
jpeg = buf.getvalue()
(Path(__file__).resolve().parent.parent / "test_images" / "orion_render.jpg").write_bytes(jpeg)

t0 = time.time()
res = platesolve_local.solve(jpeg, W, H, fov_hint_deg=None,
                             progress=lambda m: print("  .", m))
dt = time.time() - t0
print(f"\nsolved: {res.solved}  ({dt:.1f}s)")
if res.solved:
    err_ra = abs(res["ra"] - RA0)
    err_dec = abs(res["dec"] - DEC0)
    print(f"  RA {res['ra']:.3f} (expected {RA0}, err {err_ra:.3f} deg)")
    print(f"  Dec {res['dec']:.3f} (expected {DEC0}, err {err_dec:.3f} deg)")
    print(f"  pixscale {res['pixscale_arcsec']:.1f} arcsec/px "
          f"(expected {scale * 3600:.1f})")
    print(f"  wcs_fits: {len(res.get('wcs_fits') or b'')} bytes")
    print("  PASS" if err_ra < 0.5 and err_dec < 0.5 else "  POSITION MISMATCH")
else:
    print("  error:", res.get("error"))
