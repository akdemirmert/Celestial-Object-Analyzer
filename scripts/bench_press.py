"""Stress bench over the famous-object press library: the answer to "I can't
hand-find every failure". Samples N images per object, runs the OFFLINE
pipeline (features + gates + analyzer), and flags anomalies:

  COK_TESPIT   detections > 1500 on a frame whose main source fills > 25%
               (interior texture leaking through as "stars")
  SURE         a single image taking > 25 s offline
  RET          a real astro press image rejected as not-astronomical

Writes bench_report.csv; prints a summary + the worst offenders. Run after
any pipeline change: `python scripts/bench_press.py [N_per_object]`.
"""
import csv
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.pipeline import analyzer, features, ingest, ml_gate  # noqa: E402

IMG = ROOT / "data" / "visual_index" / "img"
OUT = ROOT / "data" / "bench_report.csv"


def main() -> None:
    per_obj = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    groups: dict[str, list[Path]] = {}
    for p in sorted(IMG.glob("PRESS_*.jpg")):
        obj = p.stem[6:].rsplit("__", 1)[0]
        groups.setdefault(obj, []).append(p)
    files = [p for paths in groups.values() for p in paths[:per_obj]]
    print(f"{len(groups)} cisim, {len(files)} goruntu taranacak")

    rows = []
    flags = Counter()
    t_all = time.time()
    for i, p in enumerate(files):
        t0 = time.time()
        try:
            im = ingest.load_image(p.read_bytes())
            prob = ml_gate.astro_probability(
                ingest.load_image(p.read_bytes(), max_dim=512).rgb)
            f = features.extract_features(im.rgb)
            if prob > ml_gate.RESCUE_THRESHOLD and not f["looks_astronomical"]:
                f["looks_astronomical"] = True
                f["plausibility_reasons"] = []
            f["astro_probability"] = prob
            rep = analyzer.analyze(f, {}, [], None, {})
            dt = time.time() - t0
            src = f.get("main_source") or {}
            n_star = len(f.get("stars") or [])
            fill = src.get("fill_fraction") or 0
            flag = []
            if n_star > 1500 and fill > 0.25:
                flag.append("COK_TESPIT")
            if dt > 25:
                flag.append("SURE")
            if rep.get("mode") == "not_astronomical" and prob >= 0.6:
                flag.append("RET")
            for fl in flag:
                flags[fl] += 1
            rows.append([p.stem, p.stem[6:].rsplit("__", 1)[0],
                         round(prob, 2), round(fill, 3), n_star,
                         round(dt, 1), rep.get("mode"),
                         (rep.get("headline") or "")[:70], "|".join(flag)])
        except Exception as exc:
            flags["HATA"] += 1
            rows.append([p.stem, "?", "", "", "", "", "HATA",
                         f"{type(exc).__name__}: {exc}"[:70], "HATA"])
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(files)} ({time.time() - t_all:.0f}s) "
                  f"bayraklar={dict(flags)}", flush=True)

    with open(OUT, "w", newline="", encoding="utf-8") as fo:
        w = csv.writer(fo)
        w.writerow(["dosya", "cisim", "P", "fill", "tespit", "sure_s",
                    "mod", "baslik", "bayrak"])
        w.writerows(rows)
    print(f"\nBITTI {time.time() - t_all:.0f}s -> {OUT.name}")
    print("bayrak ozeti:", dict(flags) or "TEMIZ")
    bad = [r for r in rows if r[-1]]
    for r in sorted(bad, key=lambda r: r[1])[:25]:
        print("  ", r[0][:38], "|", r[-1], "|", r[7][:48])


if __name__ == "__main__":
    main()
