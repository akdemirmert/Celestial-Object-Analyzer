"""Vignette-augmented negatives could plausibly teach the gate to reject REAL
planet/Moon photos (a bright disk surrounded by black is their geometry too).
Check every real solar-system image we downloaded from NASA."""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.pipeline import ingest, ml_gate

ROOT = Path(__file__).resolve().parent.parent
astro = sorted((ROOT / "data" / "astro").glob("*.jpg"))

scores = []
for p in astro:
    try:
        im = ingest.load_image(p.read_bytes(), max_dim=512)
        scores.append((ml_gate.astro_probability(im.rgb), p.name))
    except Exception:
        continue

scores.sort()
rejected = [s for s in scores if s[0] < 0.35]
print(f"real NASA astro images: {len(scores)}")
print(f"REJECTED by the gate (P<0.35): {len(rejected)}  "
      f"({100*len(rejected)/max(len(scores),1):.1f}%)")
print("\nlowest 15 (would be refused):")
for p, n in scores[:15]:
    print(f"  {p:.3f}  {n[:60]}")
print(f"\nmedian P = {scores[len(scores)//2][0]:.3f}")
