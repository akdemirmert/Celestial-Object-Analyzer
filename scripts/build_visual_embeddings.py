"""Compute embeddings for every downloaded visual-index reference image.
Re-run after build_visual_index.py adds more references (incremental: keeps
existing vectors, embeds only new images)."""
import csv
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.pipeline import visualmatch  # noqa: E402
from app.pipeline.visualmatch import EMB_PATH, INDEX_DIR  # noqa: E402


def main() -> None:
    from PIL import Image

    commons: dict[str, str] = {}
    meta = INDEX_DIR / "meta.csv"
    if meta.exists():
        with open(meta, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                commons[r["name"]] = r.get("common") or ""

    old_names: list[str] = []
    old_vecs = None
    if EMB_PATH.exists():
        z = np.load(EMB_PATH, allow_pickle=True)
        old_names = list(z["names"])
        old_vecs = z["vecs"]
    known = set(old_names)

    files = sorted((INDEX_DIR / "img").glob("*.jpg"))
    todo = [f for f in files if f.stem not in known]
    print(f"referans: {len(files)} goruntu, {len(todo)} yeni embed edilecek")
    t0 = time.time()
    new_names, new_vecs = [], []
    for i, f in enumerate(todo):
        try:
            rgb = np.asarray(Image.open(f).convert("RGB"), dtype=np.float32) / 255.0
        except Exception:
            continue
        new_names.append(f.stem)
        new_vecs.append(visualmatch.embed(rgb))
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(todo)} ({time.time() - t0:.0f}s)", flush=True)

    names = old_names + new_names
    vecs = (np.vstack([old_vecs, np.stack(new_vecs)]) if old_vecs is not None and new_vecs
            else (old_vecs if old_vecs is not None else np.stack(new_vecs)))
    np.savez_compressed(
        EMB_PATH, names=np.array(names, dtype=object),
        commons=np.array([commons.get(n, "") for n in names], dtype=object),
        vecs=vecs.astype(np.float32))
    print(f"kaydedildi: {EMB_PATH.name} ({len(names)} nesne, {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
