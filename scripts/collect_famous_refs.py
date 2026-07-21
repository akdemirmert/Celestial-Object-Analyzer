"""Multi-photo reference library for FAMOUS objects (the user's demand:
"don't train on one photo - 100-200 per very famous object, 30-40 for the
rest"). Sources: NASA Image Library, filtered by title words AND the astro
gate CNN, saved as PRESS_<Object>__<n>.jpg into the visual index so the
appearance matcher learns every processing style of each object.

Resume-safe; re-run scripts/build_visual_embeddings.py afterwards.
"""
import csv
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

IMG = ROOT / "data" / "visual_index" / "img"
PRESS_META = ROOT / "data" / "visual_index" / "press_meta.csv"

BAD = ("astronaut", "crew", "launch", "people", "ceremony", "artist",
       "illustration", "concept", "animation", "artist's", "rendering",
       "exhibit", "model", "mirror", "clean room", "director", "training",
       "suit", "simulator", "replica", "poster", "logo", "diagram", "chart",
       "infographic", "map of", "visualization", "visualisation")

SOLAR = [("Earth", ["earth from space full disk", "blue marble earth",
                    "earth planet space view", "crescent earth space"]),
         ("Moon", ["moon full disk", "lunar surface orbit", "crescent moon space",
                   "moon far side"]),
         ("Sun", ["sun sdo", "solar flare sun", "sun corona eclipse",
                  "sun full disk"]),
         ("Mercury", ["mercury messenger planet", "mercury globe"]),
         ("Venus", ["venus planet global", "venus magellan"]),
         ("Mars", ["mars full disk", "mars globe hubble", "mars planet viking orbiter"]),
         ("Jupiter", ["jupiter full disk", "jupiter juno", "jupiter hubble",
                      "jupiter great red spot"]),
         ("Saturn", ["saturn full disk", "saturn cassini planet", "saturn hubble",
                     "saturn rings planet"]),
         ("Uranus", ["uranus planet", "uranus voyager", "uranus webb"]),
         ("Neptune", ["neptune planet", "neptune voyager", "neptune webb"]),
         ("Pluto", ["pluto new horizons"])]
SOLAR_TARGET = 150

SPECIALS = [("Polaris", ["polaris star"]), ("Betelgeuse", ["betelgeuse"]),
            ("Sirius", ["sirius star"]), ("Eta Carinae", ["eta carinae"]),
            ("Milky Way", ["milky way galactic center", "milky way band night"]),
            ("Sagittarius A*", ["sagittarius a black hole"]),
            ("TON 618", ["TON 618 quasar"])]
SPECIAL_TARGET = 25
MESSIER_TARGET = 30


def load_famous_ngc() -> list[tuple[str, list[str]]]:
    out = []
    try:
        with open(ROOT / "data" / "NGC.csv", encoding="utf-8") as f:
            for r in csv.DictReader(f, delimiter=";"):
                cn = (r.get("Common names") or "").strip()
                name = (r.get("Name") or "").strip()
                if cn and name:
                    first = cn.split(",")[0].strip()
                    if len(first) >= 5:
                        out.append((name, [first]))
    except Exception:
        pass
    return out[:200]


def main() -> None:
    from app.pipeline import ingest, ml_gate

    targets: list[tuple[str, list[str], int]] = []
    targets += [(n, qs, SOLAR_TARGET) for n, qs in SOLAR]
    targets += [(f"M{i}", [f"Messier {i}"], MESSIER_TARGET)
                for i in range(1, 111)]
    targets += [(n, qs, SPECIAL_TARGET) for n, qs in SPECIALS]
    targets += [(n, qs, 25) for n, qs in load_famous_ngc()]
    print(f"{len(targets)} unlu cisim hedefi")

    meta_new = not PRESS_META.exists()
    total = 0
    with (httpx.Client(timeout=40, follow_redirects=True) as client,
          open(PRESS_META, "a", newline="", encoding="utf-8") as mf):
        w = csv.writer(mf)
        if meta_new:
            w.writerow(["file", "object"])
        for oi, (obj, queries, target) in enumerate(targets):
            safe = obj.replace(" ", "_").replace("*", "s").replace("/", "_")
            have = len(list(IMG.glob(f"PRESS_{safe}__*.jpg")))
            if have >= target:
                continue
            got = have
            seen_ids: set[str] = set()
            for q in queries:
                if got >= target:
                    break
                for page in range(1, 6):
                    if got >= target:
                        break
                    try:
                        r = client.get("https://images-api.nasa.gov/search",
                                       params={"q": q, "media_type": "image",
                                               "page": page})
                        items = r.json()["collection"]["items"]
                    except Exception:
                        break
                    if not items:
                        break
                    for item in items:
                        if got >= target:
                            break
                        d = (item.get("data") or [{}])[0]
                        nid = d.get("nasa_id", "")
                        if not nid or nid in seen_ids:
                            continue
                        seen_ids.add(nid)
                        text = (d.get("title", "") + " "
                                + d.get("description", "")[:300]).lower()
                        if any(b in text for b in BAD):
                            continue
                        href = next((l.get("href") for l in (item.get("links") or [])
                                     if l.get("href", "").lower().endswith(
                                         (".jpg", ".jpeg"))), None)
                        if not href:
                            continue
                        dest = IMG / f"PRESS_{safe}__{got}.jpg"
                        try:
                            data = client.get(href).content
                            if len(data) < 15000:
                                continue
                            im = ingest.load_image(data, max_dim=512)
                            if ml_gate.astro_probability(im.rgb) < 0.5:
                                continue  # press junk: podiums, hardware...
                            dest.write_bytes(data)
                            w.writerow([dest.name, obj])
                            mf.flush()
                            got += 1
                            total += 1
                        except Exception:
                            continue
                        time.sleep(0.12)
            if got > have:
                print(f"  [{oi + 1}/{len(targets)}] {obj}: {got} referans",
                      flush=True)
    print(f"BITTI: +{total} yeni unlu-cisim referansi")


if __name__ == "__main__":
    main()
