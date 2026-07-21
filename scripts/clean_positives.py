"""Quarantine non-sky images from data/astro/.

The NASA search queries pulled in press photos: crowds watching an eclipse,
exhibition booths, clean rooms, Apollo surface shots. Training on those taught
the gate that a crowd is a sky image. Verified by eye on a contact sheet:
~38 of the 41 images the gate rejects are genuinely not sky photos.

Quarantine (not delete, not relabel): a few real ones - a total eclipse, an
Earthrise - are in there too, and mislabelling those as negatives would be
worse than losing them.
"""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.pipeline import ingest, ml_gate

ROOT = Path(__file__).resolve().parent.parent
QUARANTINE = ROOT / "data" / "astro_quarantine"
QUARANTINE.mkdir(parents=True, exist_ok=True)

moved = 0
for p in sorted((ROOT / "data" / "astro").glob("*.jpg")):
    try:
        im = ingest.load_image(p.read_bytes(), max_dim=512)
        if ml_gate.astro_probability(im.rgb) < 0.35:
            shutil.move(str(p), str(QUARANTINE / p.name))
            moved += 1
    except Exception:
        continue
print(f"quarantined {moved} non-sky images")
print(f"positives left: {len(list((ROOT / 'data' / 'astro').glob('*.jpg')))}")
