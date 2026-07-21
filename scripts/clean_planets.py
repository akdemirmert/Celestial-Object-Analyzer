"""Auto-clean the new planet positives with the RULE-based gate only.

The junk in the collection is daylight/indoor material (clean rooms, press
events, monuments, diagrams) - exactly what the brightness/vividness rules
reject. Full-disk planets on black sky pass them. The CNN cannot be used here:
its current planet-rejecting bias is the very thing we are fixing.
"""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.pipeline import features, ingest

ROOT = Path(__file__).resolve().parent.parent
Q = ROOT / "data" / "astro_quarantine"
Q.mkdir(exist_ok=True)

kept = moved = 0
for p in sorted((ROOT / "data" / "astro").glob("planet_*.jpg")):
    try:
        im = ingest.load_image(p.read_bytes(), max_dim=512)
        f = features.extract_features(im.rgb)
        if f["looks_astronomical"]:
            kept += 1
        else:
            shutil.move(str(p), str(Q / p.name))
            moved += 1
    except Exception:
        shutil.move(str(p), str(Q / p.name))
        moved += 1
print(f"kept {kept}, quarantined {moved}")
