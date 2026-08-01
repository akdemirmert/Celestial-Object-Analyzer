"""Offline pipeline test over test_images/ (plate solving disabled).

Mirrors the gate order used by jobs.py - ML gate, transparency, render check -
so the offline result matches what the app actually reports. (It did not
before, which hid real failures behind passing tests.)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.pipeline import analyzer, features, ingest, ml_gate
from app.pipeline.platesolve import PlateSolveResult

TEST_DIR = Path(__file__).resolve().parent.parent / "test_images"
EXTS = (".jpg", ".jpeg", ".png", ".webp")


def gated_report(path: Path):
    img = ingest.load_image(path.read_bytes())
    feats = features.extract_features(img.rgb)

    # --- same gates as jobs.py, same order ---
    try:
        p = ml_gate.astro_probability(img.rgb)
    except Exception:
        p = None
    if p is not None:
        feats["astro_probability"] = round(p, 3)
        if p < ml_gate.REJECT_THRESHOLD and feats.get("looks_astronomical", True):
            feats["looks_astronomical"] = False
            feats["plausibility_reasons"] = [
                f"neural gate scores this {p * 100:.0f}% likely to be a sky image"
            ] + (feats.get("plausibility_reasons") or [])
        elif p > ml_gate.RESCUE_THRESHOLD and not feats.get("looks_astronomical", True):
            feats["looks_astronomical"] = True
            feats["plausibility_reasons"] = []
    if (img.transparent_fraction > 0.005
            and img.interior_transparent_fraction > 0.005):
        feats["looks_astronomical"] = False
        feats["plausibility_reasons"] = ["transparent background"] + (
            feats.get("plausibility_reasons") or [])

    solve = PlateSolveResult(solved=False, skipped=True, error="offline test")
    return feats, analyzer.analyze(feats, solve, [], None, img.exif), p


skipped = []
for img_path in sorted(p for p in TEST_DIR.iterdir() if p.suffix.lower() in EXTS):
    try:
        feats, report, prob = gated_report(img_path)
    except Exception as e:
        # one unreadable file (a saved robots-policy page named .jpg) killed
        # the whole dump for two days - report it loudly and keep going
        skipped.append((img_path.name, repr(e)))
        print("=" * 78)
        print(f"IMAGE: {img_path.name}  !! SKIPPED: {e!r}")
        continue
    print("=" * 78)
    print(f"IMAGE: {img_path.name}")
    print(f"  mode      : {report['mode']}")
    print(f"  headline  : {report['headline']}")
    print(f"  P(astro)  : {prob if prob is None else round(prob, 2)}")
    src = feats.get("main_source")
    if src:
        print(f"  main src  : d={src['equiv_diameter_px']}px elong={src['elongation']} "
              f"conc={src['concentration']} qcirc={src.get('bright_circularity')} "
              f"band={src.get('band_contrast')} border={src.get('border_contact')}")
        print(f"  flags     : point={src['is_point_like']} disk={src['is_disk_like']} "
              f"fuzzy={src['is_extended_fuzzy']} streak={src['is_streak']}")
    print(f"  stars     : {feats['star_count']}")
    for h in report["hypotheses"]:
        print(f"  [{h['band']:^8} {h['score']:>3}] {h['label']}")
print("=" * 78)
if skipped:
    print(f"WARNING: {len(skipped)} file(s) skipped as unreadable: "
          + ", ".join(n for n, _ in skipped))
print("ALL TESTS RAN")
