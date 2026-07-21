"""Star-identification chain test with a synthetic WCS over the Pleiades.

Builds a TAN WCS centered on the Pleiades, fetches the real SIMBAD field
catalog, projects three bright catalog stars into pixel space, then verifies
identify_stars() maps them back and names them correctly.
"""
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from astropy.io import fits
from astropy.wcs import WCS

from app.pipeline import starid

RA0, DEC0 = 56.75, 24.1167  # Pleiades center

w = WCS(naxis=2)
w.wcs.crpix = [400, 300]
w.wcs.crval = [RA0, DEC0]
w.wcs.cdelt = [-10 / 3600, 10 / 3600]  # 10 arcsec/px
w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
buf = io.BytesIO()
fits.PrimaryHDU(header=w.to_header()).writeto(buf)
wcs_bytes = buf.getvalue()

solve = {"ra": RA0, "dec": DEC0, "radius_deg": 1.0, "pixscale_arcsec": 10.0}

catalog = starid.field_star_catalog(RA0, DEC0, 1.0)
print(f"SIMBAD field catalog: {len(catalog)} entries")
with_v = [c for c in catalog if c.get("vmag") is not None]
with_v.sort(key=lambda c: c["vmag"])
targets = with_v[:3]
for t in targets:
    print(f"  ground truth: {t['main_id']} V={t['vmag']} at RA {t['ra']:.4f}, Dec {t['dec']:.4f}")

stars = []
for t in targets:
    px, py = w.world_to_pixel_values(t["ra"], t["dec"])
    stars.append({"x": float(px), "y": float(py), "peak_snr": 60.0,
                  "flux": 1.0, "rb_ratio": 0.9, "tier": "bright"})

out, matched, use_flip = starid.identify_stars(wcs_bytes, stars, solve, image_height=600)
print(f"\nmatched {matched}/3:")
ok = 0
for s, t in zip(out, targets):
    got = (s.get("id") or {}).get("name", "-- no match --")
    hit = got == t["main_id"].strip()
    ok += hit
    print(f"  {'OK ' if hit else 'MISS'} expected {t['main_id']!r} -> got {got!r} "
          f"(sep {(s.get('id') or {}).get('match_arcsec', '?')}\", "
          f"dist {(s.get('id') or {}).get('distance_ly', '?')} ly, "
          f"url: {(s.get('id') or {}).get('url', '-')[:60]})")
print(f"\n{'ALL OK' if ok == 3 else 'FAILURES PRESENT'}")
