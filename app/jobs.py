"""Background analysis jobs: each upload runs the full pipeline in a thread
while the frontend polls for stage-by-stage progress."""
from __future__ import annotations

import io
import json
import re
import threading
import time
import traceback
import uuid
from pathlib import Path

from .config import CONFIG
from .pipeline import (analyzer, catalogs, ephemeris, features, ingest,
                       localcatalog, ml_gate, photometry, platesolve,
                       platesolve_local, starid)

STAGES = ["ingest", "features", "photometry", "platesolve", "catalogs",
          "ephemeris", "analysis"]

MAX_KEPT_JOBS = 20

_jobs: dict[str, dict] = {}
_order: list[str] = []
_lock = threading.Lock()


def _set_stage(job_id: str, stage: str, status: str, detail: str = "") -> None:
    with _lock:
        job = _jobs.get(job_id)
        if job:
            job["stages"][stage] = {"status": status, "detail": detail}


HISTORY_DIR = Path(__file__).resolve().parents[1] / "data" / "jobs"
HISTORY_KEEP = 50
_SAFE_ID = re.compile(r"^[0-9a-f]{12}$")


def get_job(job_id: str) -> dict | None:
    with _lock:
        job = _jobs.get(job_id)
        if job:
            return {k: v for k, v in job.items() if k != "_analysis_jpeg"}
    # finished sessions survive on disk - a refresh or restart no longer
    # loses the analysis
    if _SAFE_ID.match(job_id):
        p = HISTORY_DIR / f"{job_id}.json"
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                return None
    return None


def get_job_image(job_id: str) -> bytes | None:
    with _lock:
        job = _jobs.get(job_id)
        if job and job.get("_analysis_jpeg"):
            return job["_analysis_jpeg"]
    if _SAFE_ID.match(job_id):
        p = HISTORY_DIR / f"{job_id}.jpg"
        if p.exists():
            return p.read_bytes()
    return None


def get_job_thumb(job_id: str) -> bytes | None:
    if _SAFE_ID.match(job_id):
        for name in (f"{job_id}_thumb.jpg", f"{job_id}.jpg"):
            p = HISTORY_DIR / name
            if p.exists():
                return p.read_bytes()
    return get_job_image(job_id)


def list_history(limit: int = HISTORY_KEEP) -> list[dict]:
    """Newest-first light entries for the recent-analyses panel."""
    out = []
    try:
        metas = sorted(HISTORY_DIR.glob("*.meta.json"),
                       key=lambda p: p.stat().st_mtime, reverse=True)
        for p in metas[:limit]:
            try:
                out.append(json.loads(p.read_text(encoding="utf-8")))
            except Exception:
                continue
    except OSError:
        pass
    return out


def _persist_job(job_id: str) -> None:
    """Write a finished job (report + image + tiny meta) into data/jobs/,
    keeping the newest HISTORY_KEEP analyses."""
    try:
        with _lock:
            job = _jobs.get(job_id)
            if not job or job.get("status") != "done":
                return
            payload = {k: v for k, v in job.items() if k != "_analysis_jpeg"}
            jpeg = job.get("_analysis_jpeg")
        payload["saved_at"] = time.time()
        HISTORY_DIR.mkdir(parents=True, exist_ok=True)
        (HISTORY_DIR / f"{job_id}.json").write_text(
            json.dumps(payload), encoding="utf-8")
        if jpeg:
            (HISTORY_DIR / f"{job_id}.jpg").write_bytes(jpeg)
            try:
                from PIL import Image as _Img
                im = _Img.open(io.BytesIO(jpeg))
                im.thumbnail((320, 320))
                im.convert("RGB").save(HISTORY_DIR / f"{job_id}_thumb.jpg",
                                       quality=80)
            except Exception:
                pass
        rep = payload.get("report") or {}
        (HISTORY_DIR / f"{job_id}.meta.json").write_text(json.dumps({
            "id": job_id,
            "filename": payload.get("filename") or "upload",
            "headline": rep.get("headline") or "",
            "mode": rep.get("mode") or "",
            "saved_at": payload["saved_at"],
        }), encoding="utf-8")
        # prune beyond the cap - drop the whole file family
        metas = sorted(HISTORY_DIR.glob("*.meta.json"),
                       key=lambda p: p.stat().st_mtime, reverse=True)
        for p in metas[HISTORY_KEEP:]:
            stem = p.name[:-len(".meta.json")]
            for q in (p, HISTORY_DIR / f"{stem}.json",
                      HISTORY_DIR / f"{stem}.jpg",
                      HISTORY_DIR / f"{stem}_thumb.jpg"):
                try:
                    q.unlink(missing_ok=True)
                except OSError:
                    pass
    except Exception:
        traceback.print_exc()


# approximate constellation centers (RA deg, Dec deg) - a user saying "I was
# pointing at Orion" is a ~20-degree hint, so a few degrees of error is fine
_CONSTELLATIONS = {
    "andromeda": (11, 38), "antlia": (150, -33), "apus": (245, -76),
    "aquarius": (335, -10), "aquila": (295, 3), "ara": (262, -55),
    "aries": (40, 20), "auriga": (85, 42), "bootes": (218, 30),
    "caelum": (70, -38), "camelopardalis": (90, 70), "cancer": (130, 20),
    "canes venatici": (195, 40), "canis major": (103, -22),
    "canis minor": (114, 6), "capricornus": (315, -18), "carina": (140, -62),
    "cassiopeia": (15, 62), "centaurus": (200, -47), "cepheus": (330, 70),
    "cetus": (25, -7), "chamaeleon": (160, -79), "circinus": (222, -63),
    "columba": (86, -35), "coma berenices": (190, 23),
    "corona australis": (280, -41), "corona borealis": (235, 32),
    "corvus": (186, -18), "crater": (170, -15), "crux": (187, -60),
    "cygnus": (307, 42), "delphinus": (309, 13), "dorado": (80, -60),
    "draco": (240, 65), "equuleus": (318, 8), "eridanus": (55, -20),
    "fornax": (42, -30), "gemini": (105, 25), "grus": (335, -46),
    "hercules": (255, 30), "horologium": (48, -53), "hydra": (160, -20),
    "hydrus": (35, -70), "indus": (315, -55), "lacerta": (335, 46),
    "leo": (160, 15), "leo minor": (155, 32), "lepus": (83, -19),
    "libra": (230, -15), "lupus": (233, -43), "lynx": (120, 47),
    "lyra": (283, 36), "mensa": (83, -77), "microscopium": (315, -37),
    "monoceros": (105, -3), "musca": (188, -70), "norma": (243, -51),
    "octans": (330, -85), "ophiuchus": (257, -6), "orion": (83, 3),
    "pavo": (295, -65), "pegasus": (340, 20), "perseus": (50, 45),
    "phoenix": (15, -48), "pictor": (85, -52), "pisces": (15, 12),
    "piscis austrinus": (340, -30), "puppis": (115, -32), "pyxis": (135, -30),
    "reticulum": (60, -60), "sagitta": (297, 18), "sagittarius": (285, -28),
    "scorpius": (253, -30), "sculptor": (5, -32), "scutum": (280, -10),
    "serpens": (255, 0), "sextans": (155, -2), "taurus": (67, 16),
    "telescopium": (285, -51), "triangulum": (32, 32),
    "triangulum australe": (240, -66), "tucana": (355, -64),
    "ursa major": (165, 55), "ursa minor": (230, 78), "vela": (140, -47),
    "virgo": (200, -3), "volans": (120, -68), "vulpecula": (300, 25),
    # popular asterisms people actually type
    "big dipper": (183, 57), "little dipper": (230, 78),
    "southern cross": (187, -60), "summer triangle": (295, 30),
    "pleiades": (56.75, 24.12), "polaris": (37.95, 89.26),
}


def _local_portrait(name: str) -> dict | None:
    """The identified object's portrait from OUR reference library - always
    the right object, offline. (A blind NASA first-hit search once showed a
    random galaxy under the M 16 identification.)"""
    try:
        import csv as _csv

        from .pipeline import visualmatch as _vm

        def norm(x: str) -> str:
            return re.sub(r"\s+", " ", x).strip().lower()

        base = norm(name)
        cands = {base, base.replace(" ", "")}
        m = re.fullmatch(r"(?:m|messier) ?(\d{1,3})", base)
        if m and int(m.group(1)) in _vm._MESSIER_NGC:
            num = _vm._MESSIER_NGC[int(m.group(1))]
            cands |= {f"ngc {num}", f"ngc{num}", "ngc %04d" % num,
                      "ngc%04d" % num, f"m {m.group(1)}", f"m{m.group(1)}"}
            for common, key in _vm._COMMON_KEY.items():
                if key.lower() in cands:
                    cands.add(common)
        hits = []
        with open(_vm.INDEX_DIR / "press_meta.csv", encoding="utf-8") as fh:
            for r in _csv.DictReader(fh):
                o = norm(r.get("object") or "")
                if o in cands or o.replace(" ", "") in cands:
                    mn = re.search(r"__(\d+)\.", r["file"])
                    hits.append((int(mn.group(1)) if mn else 0, r["file"]))
        if hits:
            # highest index wins: curated/corrected refs are appended last
            # (HCG 40's __0 was a mislabeled harvest frame)
            hits.sort(reverse=True)
            return {"title": f"{re.sub(r'\\s+', ' ', name).strip()} "
                             "- reference image (local library)",
                    "image_url": f"/api/ref-image/{hits[0][1]}"}
    except Exception:
        pass
    return None


def _meta_majax(name: str) -> float | None:
    """Catalog angular size (arcmin) for a matched object name, alias-aware."""
    try:
        import csv as _csv

        from .pipeline import visualmatch as _vmm
        keys = set(_vmm._coord_keys(name, name))
        keys.add(name.replace(" ", ""))
        with open(_vmm.INDEX_DIR / "meta.csv", encoding="utf-8") as fh:
            for r in _csv.DictReader(fh):
                if r["name"] in keys:
                    try:
                        return float(r["majax_arcmin"])
                    except (TypeError, ValueError):
                        return None
    except Exception:
        pass
    return None


def _identity_from_ref(ref: str) -> str:
    """Reference key -> object name: 'PRESS_Large_Magellanic_Cloud__11'
    -> 'Large Magellanic Cloud'."""
    t = re.sub(r"^PRESS_", "", (ref or "").strip())
    return re.sub(r"__\d+$", "", t).replace("_", " ").strip()


def _resolve_user_hint(text: str) -> dict | None:
    """User position hint -> {'ra','dec','label','radius_deg'} or None.

    Accepts decimal 'RA Dec' degrees, a constellation/asterism name, or any
    object name SIMBAD can resolve (Vega, M31, Barnard's Star...).
    """
    t = text.strip()
    # press archives spell some designations long-form; SIMBAD wants the
    # short ident ("Hickson Compact Group 40" -> "HCG 40")
    t = re.sub(r"^hickson\s+compact\s+group\s+(\d+)$", r"HCG \1", t,
               flags=re.I)
    # 1) two numbers = RA Dec in degrees
    m = re.match(r"^\s*([+-]?\d+(?:\.\d+)?)[,\s]+([+-]?\d+(?:\.\d+)?)\s*$", t)
    if m:
        ra, dec = float(m.group(1)), float(m.group(2))
        if 0 <= ra < 360 and -90 <= dec <= 90:
            return {"ra": ra, "dec": dec, "label": f"RA {ra:.2f} Dec {dec:.2f}",
                    "radius_deg": 15.0}
        return None
    # 2) constellation / asterism
    c = _CONSTELLATIONS.get(t.lower())
    if c:
        return {"ra": float(c[0]), "dec": float(c[1]), "label": t.title(),
                "radius_deg": 25.0}
    # 3) SIMBAD name resolution (free TAP; covers stars, DSOs, everything)
    try:
        import httpx as _hx
        safe = t.replace("'", "''")
        r = _hx.get("https://simbad.cds.unistra.fr/simbad/sim-tap/sync",
                    params={"request": "doQuery", "lang": "adql",
                            "format": "json",
                            "query": ("SELECT b.ra, b.dec FROM basic b "
                                      "JOIN ident i ON b.oid = i.oidref "
                                      f"WHERE i.id = '{safe}'")},
                    timeout=25)
        data = (r.json() or {}).get("data") or []
        if data and data[0][0] is not None:
            return {"ra": float(data[0][0]), "dec": float(data[0][1]),
                    "label": t, "radius_deg": 15.0}
    except Exception:
        pass
    return None


def _run_pipeline(job_id: str, data: bytes,
                  user_hint_text: str | None = None,
                  user_fov: float | None = None) -> None:
    try:
        # ---- 1. ingest -------------------------------------------------------
        _set_stage(job_id, "ingest", "running", "Decoding image and reading EXIF...")
        img = ingest.load_image(data, max_dim=CONFIG["max_image_dim"])
        analysis_jpeg = ingest.encode_analysis_jpeg(img)
        exif_summary = []
        if img.exif.get("datetime_original"):
            exif_summary.append("capture time found")
        if "latitude" in img.exif:
            exif_summary.append("GPS found")
        with _lock:
            _jobs[job_id]["_analysis_jpeg"] = analysis_jpeg
            _jobs[job_id]["image"] = {
                "analysis_width": img.width, "analysis_height": img.height,
                "original_width": img.original_width,
                "original_height": img.original_height,
            }
        _set_stage(job_id, "ingest", "done",
                   f"{img.original_width}x{img.original_height} {img.format or 'image'}"
                   + (f" ({', '.join(exif_summary)})" if exif_summary else " (no EXIF time/GPS)"))

        # ---- 2. features -----------------------------------------------------
        _set_stage(job_id, "features", "running", "Measuring sources, color, and structure...")
        # deep star pass: big originals get a second detection run at up to
        # 4096 px, recovering faint stars the analysis-size downsample erased
        # (user-visible stars measured 'missing'; costs a few extra seconds)
        hi = None
        if max(img.original_width, img.original_height) > 1.25 * max(img.width, img.height):
            _set_stage(job_id, "features", "running",
                       "Measuring sources + deep faint-star scan at high resolution...")
            try:
                hi = ingest.load_image(data, max_dim=4096)
            except Exception:
                hi = None
        feats = features.extract_features(img.rgb, hi.rgb if hi else None)
        src = feats.get("main_source")

        # Saturn's companions: bright points beside a ringed planet are its
        # moons. The brightest is almost certainly Titan (mag 8.5 vs Rhea's
        # 9.7 - Titan outshines every other moon in any framing); the rest
        # get an honest candidate label. Gives the gold ring + hover card +
        # click-through the user asked for.
        if (src or {}).get("is_ringed_disk"):
            bright_pts = [s for s in feats.get("stars") or []
                          if s.get("tier") == "bright"]
            # wide-sweep morphology = possibly an ice giant in IR (the JWST
            # Neptune frame) - its bright points are Triton & co., so the
            # Saturn-specific moon names would be WRONG names there
            _ice = bool((src or {}).get("ring_wide_sweep"))
            for i, s in enumerate(bright_pts[:4]):
                if _ice:
                    s["id"] = {
                        "name": "Moon of the ringed planet (candidate)",
                        "type_label": "Planetary moon",
                        "type_note": ("Bright points beside a ringed planet "
                                      "are its moons; WHICH planet (and so "
                                      "which moon) needs the capture time - "
                                      "for Saturn this would be Titan/Rhea, "
                                      "for Neptune Triton."),
                        "url": "https://science.nasa.gov/solar-system/moons/",
                    }
                elif i == 0:
                    s["id"] = {
                        "name": "Titan (likely)",
                        "type_label": "Moon of Saturn - its largest and brightest",
                        "type_note": ("The brightest point near Saturn is almost "
                                      "always Titan; exact confirmation needs the "
                                      "capture time."),
                        "url": "https://science.nasa.gov/saturn/moons/titan/",
                    }
                else:
                    s["id"] = {
                        "name": "Saturnian moon (candidate)",
                        "type_label": "Moon of Saturn - Rhea, Dione, Tethys or Enceladus",
                        "type_note": ("Which moon it is depends on the capture "
                                      "time; these four are the usual bright ones."),
                        "url": "https://science.nasa.gov/saturn/moons/",
                    }

        detail = f"{feats['star_count']} star-like sources"
        deep_n = sum(1 for s in feats.get("stars") or [] if s.get("deep"))
        if deep_n:
            detail += f" (+{deep_n} from the deep high-res scan)"
        if src:
            detail += f"; main source {src['equiv_diameter_px']:.0f} px"
        _set_stage(job_id, "features", "done", detail)

        # ML gate: a MobileNetV3 pretrained on ImageNet (1.2M photos, so it
        # already knows balls, stadiums, jewellery, grass...) with a head
        # trained on our astro/not-astro data. It generalises to objects we
        # never collected, which hand-written pixel rules cannot do.
        # It only REJECTS: the rule gates stay the accept authority.
        try:
            p_astro = ml_gate.astro_probability(img.rgb)
        except Exception:
            p_astro = None
        if p_astro is not None:
            feats["astro_probability"] = round(p_astro, 3)
            if p_astro < ml_gate.REJECT_THRESHOLD and feats.get("looks_astronomical", True):
                feats["looks_astronomical"] = False
                feats["plausibility_reasons"] = [
                    f"a neural network trained on real astronomical images and on "
                    f"1.2 million everyday photographs scores this only "
                    f"{p_astro * 100:.0f}% likely to be a sky image"
                ] + (feats.get("plausibility_reasons") or [])
            elif (p_astro > ml_gate.RESCUE_THRESHOLD
                    and not feats.get("looks_astronomical", True)):
                # rescue: the brightness rules reject frame-filling planets
                # (Viking Mars fills 98% of the image, so the "sky" IS the
                # planet); a confident CNN overrides those pixel statistics
                feats["looks_astronomical"] = True
                feats["plausibility_reasons"] = []

        # transparency is decisive: real photographs never contain transparent
        # pixels - this is a cut-out/render (e.g. "removebg" images)
        if img.transparent_fraction > 0.005:
            feats["looks_astronomical"] = False
            feats["plausibility_reasons"] = [
                f"the image has a transparent background "
                f"({img.transparent_fraction * 100:.0f}% of pixels) - photographs never "
                "contain transparency, so this is a digital cut-out or render"
            ] + (feats.get("plausibility_reasons") or [])

        # ---- early exit: not an astronomical image ----------------------------
        # No point running photometry/solving/catalogs on a screenshot or logo;
        # report the rejection immediately with nothing else attached.
        if not feats.get("looks_astronomical", True):
            for stage in ("photometry", "platesolve", "catalogs", "ephemeris"):
                _set_stage(job_id, stage, "skipped", "Not an astronomical image.")
            _set_stage(job_id, "analysis", "running", "Building the report...")
            solve = platesolve.PlateSolveResult(solved=False, skipped=True,
                                                error="Not an astronomical image.")
            report = analyzer.analyze(feats, solve, [], None, img.exif, {})
            _set_stage(job_id, "analysis", "done", report["headline"])
            with _lock:
                _jobs[job_id].update(
                    status="done", report=report, features=feats, photometry={},
                    exif=img.exif, solve=dict(solve),
                    overlay={"stars": [], "notable_sources": [],
                             "main_source": None, "annotations": []})
            _persist_job(job_id)
            return

        # ---- 3. photometry -----------------------------------------------------
        _set_stage(job_id, "photometry", "running", "Computing histogram and radial profile...")
        try:
            phot = photometry.measure(img.rgb, feats)
            _set_stage(job_id, "photometry", "done",
                       "Histogram + radial light profile measured"
                       if "radial_profile" in phot else "Histogram measured")
        except Exception:
            phot = {}
            _set_stage(job_id, "photometry", "failed", "Photometry failed (non-fatal).")

        # ---- 4. plate solve ------------------------------------------------------
        # local ASTAP first (seconds, offline); nova.astrometry.net as fallback
        api_key = CONFIG["astrometry_api_key"]
        astap_ok = platesolve_local.available()
        # a USER-SUPPLIED position collapses the search space, so hinted
        # attempts are worth it even on sparse phone frames (5-7 stars)
        user_pos = None
        if user_hint_text:
            _set_stage(job_id, "platesolve", "running",
                       f"Resolving position hint '{user_hint_text}'...")
            user_pos = _resolve_user_hint(user_hint_text)
            _set_stage(job_id, "platesolve", "running",
                       (f"Position hint: {user_pos['label']} "
                        f"(RA {user_pos['ra']:.2f}, Dec {user_pos['dec']:.2f})"
                        if user_pos else
                        f"Could not resolve '{user_hint_text}' - continuing without it."))
        enough_stars = (feats["star_count"] >= 8
                        or (user_pos is not None and feats["star_count"] >= 5))
        # A dominant extended object (a galaxy/nebula close-up filling the frame)
        # has too few background stars to plate-solve - confirmed by both ASTAP
        # and nova failing on such images. We still ATTEMPT a fast local solve
        # (so the user sees it tried), but skip the minutes-long online queue
        # that would only fail anyway.
        _msrc = feats.get("main_source") or {}
        # A frame-filling object - galaxy/nebula OR a resolved planet/Moon disk -
        # is a close-up that cannot plate-solve. Deep telescope frames show
        # HUNDREDS of faint points, but they are far below the magnitude limit
        # of any astrometric index (Tycho-2/Gaia reach ~mag 12-17; a Hubble crop
        # shows mag 20+), so a high star count here means nothing - gating on it
        # sent doomed images to the online queue for 7 minutes.
        # frame_filling = the emission region spans the whole frame (bright
        # nebula close-up, e.g. visible-light Pillars): same doomed-to-fail
        # situation even though the 5-sigma component itself is tiny - without
        # this it burned 75s of ASTAP + the full 210s nova timeout (292s).
        # fill > 0.30 alone also counts: whatever the type flags say, a source
        # covering a third of the frame is a close-up - a stock Earth photo
        # slipped past every type flag and burned 15 minutes in solver queues
        dominant_dso = bool(((_msrc.get("is_extended_fuzzy") or _msrc.get("is_disk_like")
                              or _msrc.get("is_ringed_disk"))
                             and _msrc.get("fill_fraction", 0) > 0.05)
                            or _msrc.get("frame_filling")
                            or _msrc.get("fill_fraction", 0) > 0.30)
        vm_matches: list[dict] = []
        solve = None
        if not feats.get("looks_astronomical", True):
            solve = platesolve.PlateSolveResult(
                solved=False, skipped=True, error="Not an astronomical image.")
            _set_stage(job_id, "platesolve", "skipped",
                       "Skipped - this doesn't look like a night-sky photo.")
        elif not enough_stars:
            solve = platesolve.PlateSolveResult(
                solved=False, skipped=True,
                error="Too few stars for astrometric solving.")
            _set_stage(job_id, "platesolve", "skipped",
                       f"Only {feats['star_count']} stars detected - naming individual objects "
                       "needs a star field (~10+ stars).")
        elif not astap_ok and not api_key:
            solve = platesolve.PlateSolveResult(solved=False, skipped=True,
                                                error="No solver available.")
            _set_stage(job_id, "platesolve", "skipped",
                       "No local ASTAP and no astrometry.net API key configured.")
        else:
            hint = ingest.field_of_view_hint(img.exif, img.width)

            # visual position guess FIRST: "this looks like M16" gives a sky
            # position that a hinted ASTAP run can verify astrometrically in
            # seconds. The user's insight: an object's background stars are
            # fixed, so knowing WHAT it is means knowing WHERE it is - and
            # then every star in frame can be named from the catalog.
            try:
                from .pipeline import visualmatch
                if visualmatch.available() and (p_astro or 0) >= 0.5:
                    # visible progress: on a cold server this loads the torch
                    # model + the 26k-image index (~1 min once) - without a
                    # stage message the pipeline LOOKED frozen after photometry
                    _set_stage(job_id, "platesolve", "running",
                               "Matching against the reference image library...")
                    vm_matches = visualmatch.match(img.rgb, top_k=8)
            except Exception:
                traceback.print_exc()
            # harvested press names (Arp 105, LEDA ids...) mostly have no
            # local catalog entry - one SIMBAD lookup turns a visual identity
            # into a POSITION, unlocking the whole identity chain for them
            for _m in vm_matches[:2]:
                if _m.get("ra") is None and _m["similarity"] >= 0.85:
                    _rr = _resolve_user_hint(_m["name"])
                    if _rr:
                        _m["ra"], _m["dec"] = _rr["ra"], _rr["dec"]
            pos_hints = [{"ra": m["ra"], "dec": m["dec"],
                          "label": (m.get("common") or m["name"]).split(",")[0]}
                         for m in vm_matches
                         if m.get("ra") is not None and m["similarity"] >= 0.5]
            # the user's own hint is the most trustworthy - try it FIRST
            if user_pos:
                pos_hints.insert(0, {"ra": user_pos["ra"], "dec": user_pos["dec"],
                                     "label": f"{user_pos['label']} (user hint)"})

            if astap_ok:
                _set_stage(job_id, "platesolve", "running",
                           "Attempting to locate the field on the sky (local ASTAP)...")
                fov_est = None
                if hint:
                    fov_est = (hint["scale_lower"] * hint["scale_upper"]) ** 0.5
                if user_fov:
                    fov_est = user_fov  # stated by the user, beats EXIF guesswork
                # dominant DSOs won't solve; use a short fast sweep on a tight
                # budget so the attempt stays a few seconds, not a minute
                # ultra-dense frames (20k+ stars) are Hubble-crop territory:
                # the local database is hopeless there, so one auto attempt
                # only - the real chance is nova's deep indexes
                fast_list = ([0.0, 2.0] if dominant_dso
                             else [0.0] if feats["star_count"] > 8000 else None)
                # position hints are high-value: give the hinted attempts room
                # even on dominant frames (each wrong hint costs a bounded
                # ~8 s slice; the right one confirms in 0.2-2 s)
                _budget = (40.0 if pos_hints else 25.0) if dominant_dso else 75.0
                solve = platesolve_local.solve(
                    analysis_jpeg, img.width, img.height, fov_est, fov_list=fast_list,
                    budget_s=_budget,
                    pos_hints=pos_hints,
                    progress=lambda msg: _set_stage(job_id, "platesolve", "running", msg),
                )
                if solve.solved:
                    via = solve.get("via_visual_hint")
                    _set_stage(job_id, "platesolve", "done",
                               f"Located! RA {solve['ra']:.3f}, Dec {solve['dec']:.3f}"
                               + (f" (visual match '{via}' confirmed astrometrically)"
                                  if via else ""))

            # SKY-TILE blind solve: the local mini-astrometry.net. Quad codes
            # vote among ~21k uniform 2-degree DSS tiles, verified by point
            # fit + pixel correlation - no online queue, seconds not minutes.
            # (never for a resolved-object close-up: Earth's cloud blobs once
            # pattern-locked onto a random star tile and named phantom stars)
            if (solve is None or not solve.solved) and not dominant_dso:
                try:
                    from .pipeline import visualmatch as _vmt
                    _set_stage(job_id, "platesolve", "running",
                               "Searching the local sky-tile index (whole sky, 2-deg tiles)...")
                    ts = _vmt.tile_solve(img.rgb, img.width, img.height)
                except Exception:
                    traceback.print_exc()
                    ts = None
                if ts:
                    solve = platesolve.PlateSolveResult(
                        solved=True, skipped=False, solver="sky-tiles", **ts)
                    _set_stage(job_id, "platesolve", "done",
                               f"Located on the local sky-tile index (tile {ts['matched_tile']}, "
                               f"{ts['pattern_inliers']} matched points, "
                               f"pixel correlation {ts['pattern_ncc']:.2f})")

            # pattern lock: match the photo's blob constellation onto each
            # visually-similar candidate's reference cutout (known WCS). One
            # verified lock pins EVERY object's sky position - the user's
            # "one object's position finds all the rest", engineered. Wrong
            # candidates fail to lock, so this cannot false-positive quietly.
            if (solve is None or not solve.solved) and vm_matches:
                _set_stage(job_id, "platesolve", "running",
                           "Pattern-matching against visually similar reference fields...")
                try:
                    from .pipeline import visualmatch as _vm
                    # candidates from BOTH recalls: appearance similarity AND
                    # the geometric quad-vote (immune to framing differences)
                    cand_names = {c["name"] for c in vm_matches}
                    cands = list(vm_matches)
                    for gc in _vm.pattern_candidates(img.rgb):
                        if gc["name"] not in cand_names:
                            cands.append(gc)
                    ps = _vm.pattern_solve(img.rgb, cands, img.width, img.height)
                except Exception:
                    traceback.print_exc()
                    ps = None
                if ps:
                    solve = platesolve.PlateSolveResult(
                        solved=True, skipped=False, solver="visual-pattern", **ps)
                    solve["identity_name"] = _identity_from_ref(
                        ps["matched_reference"])
                    _set_stage(job_id, "platesolve", "done",
                               f"Located by pattern-locking onto the reference field of "
                               f"{ps['matched_reference']} ({ps['pattern_inliers']} matched "
                               f"points, ±{ps['pattern_rms_px']} px)")

            # PUBLISHER-WCS lock (press-avm): space-telescope close-ups have
            # no solvable star pattern (their detections are nebular knots),
            # but the ARCHIVE that published the image states exactly where
            # the frame sits (AVM Spatial metadata). A pixel-tight alignment
            # onto the matching press reference inherits that WCS - and then
            # every knot gets named from the deep catalogs like any solved
            # frame. This is the general fix for "identified the JWST image
            # but every object fell to unidentified".
            if (solve is None or not solve.solved) and vm_matches:
                try:
                    from .pipeline import visualmatch as _vma
                    _top = vm_matches[0]
                    if _top["similarity"] >= 0.90:
                        _set_stage(job_id, "platesolve", "running",
                                   f"Visual identity {_top['name']}: aligning "
                                   "onto the publisher's astrometry (AVM)...")
                        pa = _vma.press_avm_solve(img.rgb, _top["name"],
                                                  img.width, img.height)
                        if pa:
                            solve = platesolve.PlateSolveResult(
                                solved=True, skipped=False,
                                solver="press-avm", **pa)
                            solve["identity_name"] = _top["name"]
                            _set_stage(
                                job_id, "platesolve", "done",
                                f"Located via the publisher's own astrometry "
                                f"for {pa['matched_reference']} "
                                f"({pa['avm_inliers']} aligned points, pixel "
                                f"correlation {pa['avm_ncc']:.2f})")
                except Exception:
                    traceback.print_exc()

            # IDENTITY -> ASTROMETRY: a decisive visual identification pins
            # position AND scale (known angular size / measured pixel size);
            # the only free parameter left is rotation - sweep it against the
            # field's Gaia stars. Names everything AROUND an identified
            # object whose close-up could never plate-solve (user: "NGC 1300
            # moves over millions of years; its background is fixed").
            if ((solve is None or not solve.solved) and len(vm_matches) >= 2
                    and vm_matches[0].get("ra") is not None
                    and vm_matches[0]["similarity"] >= 0.88
                    and vm_matches[0]["similarity"]
                        - vm_matches[1]["similarity"] >= 0.03
                    and _msrc.get("major_axis_px")):
                try:
                    import numpy as _np
                    from .pipeline import visualmatch as _vmi
                    top = vm_matches[0]
                    # scale estimate: catalog angular size over measured size.
                    # Alias names must normalize to the catalog key ("Antennae
                    # Galaxies" -> NGC4038, "Messier 88" -> NGC4501), else the
                    # rotation solver silently skipped every alias-named match
                    _keys = set(_vmi._coord_keys(top["name"], top["name"]))
                    _keys.add(top["name"].replace(" ", ""))
                    majax = None
                    import csv as _csv
                    with open(_vmi.INDEX_DIR / "meta.csv", encoding="utf-8") as _f:
                        for _r in _csv.DictReader(_f):
                            if _r["name"] in _keys:
                                try:
                                    majax = float(_r["majax_arcmin"])
                                except (TypeError, ValueError):
                                    majax = None
                                break
                    if majax:
                        _set_stage(job_id, "platesolve", "running",
                                   f"Visual identity {top['name']}: fitting field "
                                   "rotation against Gaia...")
                        scale_est = (majax / 60.0) / max(_msrc["major_axis_px"], 1)
                        # brightest FIRST: the sweep matches against Gaia's
                        # brightest, and SNR saturates - flux is the real rank
                        # (unsorted, the top-250 slice was arbitrary and a
                        # 23k-star frame never locked)
                        _st_sorted = sorted(feats.get("stars") or [],
                                            key=lambda s2: -(s2.get("flux") or 0))
                        det_xy = _np.array([[s2["x"], s2["y"]]
                                            for s2 in _st_sorted])
                        ids = _vmi.identity_rotation_solve(
                            det_xy, top["ra"], top["dec"],
                            _msrc["center_x"], _msrc["center_y"],
                            scale_est, img.width, img.height)
                        if ids:
                            solve = platesolve.PlateSolveResult(
                                solved=True, skipped=False,
                                solver="visual-id+rotation", **ids)
                            solve["identity_name"] = top["name"]
                            _set_stage(job_id, "platesolve", "done",
                                       f"Located via the {top['name']} identification "
                                       f"({ids['id_matches']} Gaia stars aligned at "
                                       f"rotation {ids['id_rotation_deg']} deg)")
                except Exception:
                    traceback.print_exc()

            # USER-HINT rotation solve: a typed identity ("Arp 105") plus a
            # stated field width IS position+scale from the observer - only
            # rotation is unknown, the same math as the visual-identity path.
            # Assumes the target is roughly centered (people frame their aim).
            if ((solve is None or not solve.solved) and user_pos and user_fov
                    and feats["star_count"] >= 6):
                try:
                    from .pipeline import visualmatch as _vmu
                    import numpy as _np3
                    _set_stage(job_id, "platesolve", "running",
                               f"Position hint {user_pos['label']}: fitting "
                               "field rotation against Gaia...")
                    _st_sorted = sorted(feats.get("stars") or [],
                                        key=lambda s2: -(s2.get("flux") or 0))
                    det_xy = _np3.array([[s2["x"], s2["y"]]
                                         for s2 in _st_sorted])
                    ids = _vmu.identity_rotation_solve(
                        det_xy, user_pos["ra"], user_pos["dec"],
                        img.width / 2.0, img.height / 2.0,
                        user_fov / max(img.width, 1), img.width, img.height)
                    if ids:
                        solve = platesolve.PlateSolveResult(
                            solved=True, skipped=False,
                            solver="user-hint+rotation", **ids)
                        _set_stage(job_id, "platesolve", "done",
                                   f"Located from your position hint "
                                   f"({ids['id_matches']} Gaia stars aligned at "
                                   f"rotation {ids['id_rotation_deg']} deg)")
                except Exception:
                    traceback.print_exc()

            # online fallback: only for images that plausibly have a star field.
            # Skip the minutes-long queue for dominant galaxy/nebula close-ups,
            # AND when the local solver failed despite thousands of "stars" -
            # real fields that dense solve locally in seconds, so those points
            # are detail/noise (artwork), not stars.
            # "thousands of stars but no local solve = fake-star artwork" was
            # written before the CNN gate existed. A real Hubble deep field
            # also holds 20k+ stars (and CAN solve on nova's deep 2MASS
            # indexes), so only skip when the CNN doubts the image too.
            junk_stars = (astap_ok and feats["star_count"] > 1500
                          and (p_astro if p_astro is not None else 1.0) < 0.85)

            # LOCAL ENGINE: the real astrometry.net running inside WSL with
            # on-disk index files - nova-grade blind solving with ZERO queue.
            # Identification must never depend on the public queue's mood
            # (the same M31 mosaic solved one day and failed four retries the
            # next; the local engine pins it deterministically in ~2 min).
            # position hint for the engine: the user's own, else a decisive
            # visual identity - "this IS NGC 1672" turns a blind search into
            # a targeted one, and the identity's angular size bounds the scale
            _eng_pos = None
            _eng_hint = hint
            if user_pos:
                _eng_pos = {"ra": user_pos["ra"], "dec": user_pos["dec"],
                            "radius_deg": user_pos["radius_deg"]}
            elif (vm_matches and vm_matches[0].get("ra") is not None
                  and vm_matches[0]["similarity"] >= 0.88):
                _eng_pos = {"ra": vm_matches[0]["ra"],
                            "dec": vm_matches[0]["dec"], "radius_deg": 2.0}
                _mj = _meta_majax(vm_matches[0]["name"])
                if _mj and _msrc.get("major_axis_px"):
                    _fw = ((_mj / 60.0) / max(_msrc["major_axis_px"], 1)
                           * img.width)
                    _eng_hint = {"scale_units": "degwidth",
                                 "scale_lower": max(_fw * 0.45, 0.01),
                                 "scale_upper": _fw * 2.2}
            if user_fov:
                _eng_hint = {"scale_units": "degwidth",
                             "scale_lower": user_fov * 0.5,
                             "scale_upper": user_fov * 2.0}
            if (solve is None or not solve.solved) and not junk_stars \
                    and (not dominant_dso or feats["star_count"] >= 1500
                         or _eng_pos is not None):
                try:
                    from .pipeline import platesolve_engine
                    if platesolve_engine.available():
                        es = platesolve_engine.solve(
                            analysis_jpeg, img.width, img.height,
                            scale_hint=_eng_hint, pos_hint=_eng_pos,
                            timeout_s=200 if dominant_dso else 120,
                            nebulous=dominant_dso,
                            progress=lambda msg: _set_stage(
                                job_id, "platesolve", "running", msg))
                        if es.get("solved"):
                            solve = platesolve.PlateSolveResult(**es)
                            _set_stage(job_id, "platesolve", "done",
                                       f"Located by the LOCAL astrometry engine "
                                       f"(offline)! RA {es['ra']:.3f}, "
                                       f"Dec {es['dec']:.3f}")
                except Exception:
                    traceback.print_exc()
            # a DOMINANT frame that still resolves thousands of stars is a
            # star cloud / nearby dwarf (NGC 6822 measured 23k real stars) -
            # nova's deep indexes solve those even when local ASTAP chokes on
            # the nebulous background, so dominance alone must not skip it
            _nova_worth = (not dominant_dso) or feats["star_count"] >= 1500
            if (solve is None or not solve.solved) and api_key \
                    and _nova_worth and not junk_stars:
                prefix = "Local solve failed; " if astap_ok else ""
                _set_stage(job_id, "platesolve", "running",
                           f"{prefix}trying the online solver (nova.astrometry.net)"
                           + (" with an EXIF field-of-view hint..." if hint else "..."))
                # nova is the LAST chance once we're here, and the public
                # queue alone can eat 3 minutes - twice we cut the solver
                # while it was actively matching. Give it a real window.
                _nova_timeout = max(CONFIG["astrometry_timeout_s"], 360)
                # NO position hint from visual matches: appearance similarity
                # proved unreliable for position (a wrong 0.92-similar field
                # would pin nova to the wrong sky area and doom the solve).
                # The EXIF scale hint stays - it is measured, not guessed.
                # the USER's hint is different from a visual guess: it is
                # ground truth from the observer - pass it to nova as a
                # center+radius constraint, plus their stated field width
                _nova_hint = dict(hint or {})
                if user_fov:
                    _nova_hint.update({"scale_units": "degwidth",
                                       "scale_lower": user_fov * 0.5,
                                       "scale_upper": user_fov * 2.0})
                if user_pos:
                    _nova_hint.update({"center_ra": user_pos["ra"],
                                       "center_dec": user_pos["dec"],
                                       "radius": user_pos["radius_deg"]})
                solve = platesolve.solve(
                    analysis_jpeg, api_key, timeout_s=_nova_timeout,
                    scale_hint=_nova_hint or None,
                    progress=lambda msg: _set_stage(job_id, "platesolve", "running", msg),
                )
                if solve.solved:
                    _set_stage(job_id, "platesolve", "done",
                               f"Located! RA {solve['ra']:.3f}, Dec {solve['dec']:.3f}")

            if not solve.solved:
                if dominant_dso:
                    if _msrc.get("is_disk_like"):
                        subj = "the planet/Moon disk fills the frame"
                    elif feats["star_count"] >= 400:
                        subj = ("this is a deep telescope field packed with faint stars, "
                                "nebulosity or a cluster")
                    else:
                        subj = "the galaxy/nebula fills the frame"
                    extra = ""
                    if feats["star_count"] > 40:
                        extra = (f" The {feats['star_count']} points detected are far fainter "
                                 "than the catalog stars solvers match against (deep telescope "
                                 "frames reach magnitudes no all-sky index contains), so they "
                                 "cannot anchor a position.")
                    _set_stage(job_id, "platesolve", "failed",
                               f"Tried to locate it on the sky: no star-pattern match - "
                               f"{subj}.{extra} The specific object can't be named, but its "
                               "class is analysed below.")
                elif junk_stars:
                    _set_stage(job_id, "platesolve", "failed",
                               f"No star-pattern match. {feats['star_count']} point-like details "
                               "were detected but they don't form real sky patterns - fields this "
                               "dense solve locally in seconds when genuine, so the online solver "
                               "was skipped.")
                else:
                    _set_stage(job_id, "platesolve", "failed", solve.get("error") or "Solve failed.")
        wcs_fits = solve.pop("wcs_fits", None)  # bytes; never JSON-serialized

        # ---- 5. catalogs ------------------------------------------------------------
        matches: list[dict] = []
        nasa_image = None
        if solve.solved:
            src_names = []
            _set_stage(job_id, "catalogs", "running", "Querying SIMBAD for objects at this position...")
            try:
                matches = catalogs.cone_search(
                    solve["ra"], solve["dec"], solve.get("radius_deg") or 0.5)
                src_names.append("SIMBAD")
            except Exception:
                matches = []
            if localcatalog.available():
                try:
                    local = localcatalog.cone_search(
                        solve["ra"], solve["dec"], solve.get("radius_deg") or 0.5)
                    known = {m["name"] for m in matches}
                    matches.extend(m for m in local if m["name"] not in known)
                    src_names.append("OpenNGC offline")
                except Exception:
                    pass
            named = [m for m in matches if m.get("name")]

            # visually-established identity (press-avm / pattern lock /
            # id+rotation): the object we ALIGNED onto must appear in the
            # field list even when its cataloged center falls outside the
            # small cone - a frame inside the LMC never saw 'NAME LMC' in
            # its 0.06-deg search (center 0.17 deg away, and no M/NGC/IC
            # designation for the famous-extended pass), so the analyzer
            # crowned a random double star instead of the galaxy itself.
            _idn = solve.get("identity_name")
            if _idn:
                def _nrm(s):
                    return re.sub(r"\s+", "", (s or "").lower())
                if _nrm(_idn) not in {_nrm(m.get("name")) for m in matches}:
                    try:
                        _io = catalogs.object_by_name(
                            _idn, solve["ra"], solve["dec"])
                    except Exception:
                        _io = None
                    if _io and _nrm(_io["name"]) not in {
                            _nrm(m.get("name")) for m in matches}:
                        # pixel-verified identity: the analyzer crowns this
                        # entry outright (a point-like exotic has no angular
                        # size, so the extended-overlap rule can't see it)
                        _io["identity"] = True
                        matches.insert(0, _io)
                        named = [m for m in matches if m.get("name")]

            # per-star identification: WCS maps each detected star to RA/Dec,
            # one batched SIMBAD query names the cataloged ones (hover/click UI)
            identified_count = 0
            if wcs_fits:
                try:
                    feats["stars"], identified_count, use_flip = starid.identify_stars(
                        wcs_fits, feats["stars"], solve, img.height)
                    by_pos = {(s["x"], s["y"]): s for s in feats["stars"]}
                    feats["notable_sources"] = [
                        by_pos.get((n["x"], n["y"]), n)
                        for n in feats["notable_sources"]]
                    # a notable with no catalog counterpart should still SAY
                    # something - five obvious background galaxies rendered
                    # with empty labels
                    for n in feats["notable_sources"]:
                        if not n.get("id") and not n.get("label"):
                            n["label"] = ("Background galaxy or compact "
                                          "source - no catalog entry at "
                                          "this position")
                    # sky position of the dominant source: anchors the primary-
                    # object choice to what is actually IN the image
                    src = feats.get("main_source")
                    if src:
                        y = (img.height - 1 - src["center_y"]) if use_flip else src["center_y"]
                        (mra, mdec), = starid.pixels_to_sky(
                            wcs_fits, [(src["center_x"], y)])
                        solve["main_source_ra"] = round(mra, 5)
                        solve["main_source_dec"] = round(mdec, 5)
                    # name each big detected source from the field catalog:
                    # "NGC 6285 (Galaxy)" on the numbered overlay instead of
                    # "Source 2 - unidentified"
                    if feats.get("sources") and matches:
                        import math
                        pts = [(s["center_x"],
                                (img.height - 1 - s["center_y"]) if use_flip
                                else s["center_y"]) for s in feats["sources"]]
                        s_coords = starid.pixels_to_sky(wcs_fits, pts)
                        diag = (img.width ** 2 + img.height ** 2) ** 0.5
                        deg_per_px = 2.0 * float(solve.get("radius_deg") or 0.5) / diag
                        for s, (sra, sdec) in zip(feats["sources"], s_coords):
                            s["ra"], s["dec"] = round(sra, 5), round(sdec, 5)
                            tol = max(s["equiv_diameter_px"] * deg_per_px * 0.6, 0.01)
                            best, best_d = None, tol
                            for m in matches:
                                if m.get("ra") is None or not m.get("name"):
                                    continue
                                dd = math.hypot(
                                    (m["ra"] - sra) * math.cos(math.radians(sdec)),
                                    m["dec"] - sdec)
                                if dd < best_d:
                                    best, best_d = m, dd
                            if best is not None:
                                s["name"] = best["name"]
                                s["type_label"] = best.get("type_label")
                except Exception:
                    traceback.print_exc()

            detail_extra = f"; {identified_count} stars identified by position" if identified_count else ""
            if named:
                _set_stage(job_id, "catalogs", "done",
                           f"{len(named)} objects ({' + '.join(src_names)}){detail_extra}")
            else:
                _set_stage(job_id, "catalogs", "done",
                           f"No cataloged objects at this position{detail_extra}.")
        else:
            _set_stage(job_id, "catalogs", "skipped",
                       "Catalog lookup requires a plate-solved position.")

        # ---- 6. ephemeris --------------------------------------------------------------
        _set_stage(job_id, "ephemeris", "running",
                   "Computing solar-system positions and sky context...")
        try:
            sky = ephemeris.sky_context(solve, img.exif)
            if sky.get("solar_system_matches"):
                _set_stage(job_id, "ephemeris", "done",
                           f"Ephemeris match: {sky['solar_system_matches'][0]['body']}!")
            elif sky.get("constellation"):
                _set_stage(job_id, "ephemeris", "done",
                           f"Field is in {sky['constellation']}"
                           + (f"; Moon: {sky['moon']['phase_name']}" if sky.get("moon") else ""))
            elif sky.get("moon"):
                _set_stage(job_id, "ephemeris", "done",
                           f"Moon at capture: {sky['moon']['phase_name']} "
                           f"({sky['moon']['illumination'] * 100:.0f}% lit)")
            elif not img.exif.get("datetime_original"):
                _set_stage(job_id, "ephemeris", "skipped",
                           "Needs EXIF capture time (and GPS for horizon checks).")
            else:
                _set_stage(job_id, "ephemeris", "done", "Sky context computed.")
        except Exception:
            traceback.print_exc()
            sky = {}
            _set_stage(job_id, "ephemeris", "failed", "Ephemeris computation failed (non-fatal).")

        # if the ephemeris pinned a solar-system body, fetch its portrait
        if solve.solved and sky.get("solar_system_matches") and nasa_image is None:
            nasa_image = catalogs.nasa_image_lookup(
                sky["solar_system_matches"][0]["body"])

        # ---- 7. analysis -----------------------------------------------------------------
        _set_stage(job_id, "analysis", "running", "Building the hypothesis report...")
        report = analyzer.analyze(feats, solve, matches, nasa_image, img.exif, sky)

        # unsolved but the user told us WHERE they pointed: list what the
        # catalog says should be visible there - honest planetarium fallback
        # for sparse phone frames no solver can lock
        if user_pos and not solve.solved:
            try:
                _fr = min(max((user_fov or 20.0) / 2.0, 5.0), 15.0)
                _cand = starid.field_star_catalog(
                    user_pos["ra"], user_pos["dec"], _fr, limit=60)
                _bright = [c for c in _cand if c.get("vmag") is not None][:10]
                if _bright:
                    _names = "; ".join(
                        f"{(c.get('main_id') or '?').strip()} (V {c['vmag']:.1f})"
                        for c in _bright)
                    report.setdefault("observations", []).append(
                        f"Position hint '{user_pos['label']}': the brightest "
                        f"cataloged stars within ~{_fr:.0f} deg of that spot are "
                        f"{_names}. The brightest points in your photo are most "
                        "likely among these.")
                    report.setdefault("caveats", []).append(
                        "The star list above comes from your position hint, not "
                        "from an astrometric solution - it says what SHOULD be "
                        "there, not which detected point is which.")
            except Exception:
                traceback.print_exc()

        # visual similarity against the all-OpenNGC reference index. v1 is a
        # SUGGESTER, not an identifier (ImageNet features see "spiral galaxy
        # texture", not identity - measured: same-object similarity does not
        # reliably beat similar-object), so wording stays hedged and it only
        # runs when the sky position could not name things exactly.
        # FAMOUS-IMAGE identification: press photos of iconic objects repeat
        # across the web - if the upload matches ONE object's reference photos
        # decisively (>=0.90 similarity with a clear lead), that IS the
        # object. This names close-ups that no astrometry can solve
        # (NGC 1300's Hubble portrait spans 5 arcmin - unsolvable locally).
        try:
            _vm_ok = False
            # RELATIVE gap: near the top the embedding scale saturates, so a
            # 0.971-vs-0.950 race is decisive even though the absolute lead
            # is small - the rival sits 1.7x farther from a perfect match.
            # (AG Car's false M31 measured ~1.25x; the Antennae/Earth tie 1.0x
            # - both stay rejected.)
            _rel_gap = (len(vm_matches) >= 2
                        and vm_matches[0]["similarity"] >= 0.95
                        and (1.0 - vm_matches[1]["similarity"])
                            >= 1.6 * (1.0 - vm_matches[0]["similarity"]))
            if (not solve.solved and len(vm_matches) >= 2
                    and report.get("mode") == "probabilistic"
                    and report.get("object") is None and _rel_gap):
                _vm_ok = True
            if (not _vm_ok and not solve.solved and len(vm_matches) >= 2
                    and report.get("mode") == "probabilistic"
                    and report.get("object") is None
                    and vm_matches[0]["similarity"] >= 0.90
                    and vm_matches[0]["similarity"]
                        - vm_matches[1]["similarity"] >= 0.04):
                # second factor: several reference variants must agree (AG
                # Carinae's stellar shell once matched a single M31 frame and
                # got crowned "M31")
                from .pipeline import visualmatch as _vms
                _sup = _vms.object_support(img.rgb, vm_matches[0]["name"])
                _vm_ok = (_sup >= 2 or (vm_matches[0]["similarity"] >= 0.96
                                        and vm_matches[0]["similarity"]
                                        - vm_matches[1]["similarity"] >= 0.05))
                if not _vm_ok:
                    # third path, PIXEL-verified: embedding similarity cannot
                    # separate "same frame, recompressed" (user's M64: 0.94,
                    # single ref) from a lookalike of a different object (AG
                    # Car scored 0.88 on several M31 refs). Alignment + NCC
                    # can: same frame 0.983, lookalikes <= 0.40.
                    _ncc = _vms.press_ncc(img.rgb, vm_matches[0]["name"])
                    _vm_ok = _ncc >= 0.55
                    if _vm_ok:
                        _set_stage(job_id, "analysis", "running",
                                   f"Visual match pixel-verified "
                                   f"(correlation {_ncc:.2f})...")
            if _vm_ok:
                top = vm_matches[0]
                nm = top["name"]
                report["headline"] = f"Probable identification (visual match): {nm}"
                report["object"] = {
                    "name": nm,
                    "type_label": "Identified by image similarity",
                    "type_note": (f"This photo matches our reference imagery of {nm} "
                                  f"with {top['similarity']:.0%} similarity and a clear "
                                  "lead over every other object - almost certainly a "
                                  "published image of it. Not an astrometric proof."),
                }
                report["hypotheses"].insert(0, _mk_vm_hyp(nm, top["similarity"]))
                # identified but unsolvable close-up: pull what the catalog
                # holds AT that spot, so the frame's bright points still get
                # honest candidate names (HP Tau's crop identified as its
                # nebula code and then named nothing at all)
                if top.get("ra") is None:
                    _rr2 = _resolve_user_hint(nm)
                    if _rr2:
                        top["ra"], top["dec"] = _rr2["ra"], _rr2["dec"]
                if top.get("ra") is not None:
                    # cone sized to the FRAME (identity scale), not a fixed
                    # 0.35 deg - a 6-arcmin close-up was listing stars far
                    # outside its own field of view
                    _fr3 = 0.35
                    _mj2 = _meta_majax(nm)
                    if _mj2 and _msrc.get("major_axis_px"):
                        _fr3 = min(max((_mj2 / 60.0)
                                       / max(_msrc["major_axis_px"], 1)
                                       * img.width * 0.7, 0.03), 0.35)
                    _cand2 = starid.field_star_catalog(
                        top["ra"], top["dec"], _fr3, limit=40)
                    _br2 = [c for c in _cand2 if c.get("vmag") is not None][:8]
                    if _br2:
                        _names2 = "; ".join(
                            f"{(c.get('main_id') or '?').strip()} "
                            f"(V {c['vmag']:.1f})" for c in _br2)
                        report.setdefault("observations", []).append(
                            f"Around {nm}, the brightest cataloged objects "
                            f"are {_names2} - the bright points in this "
                            "close-up are most likely among these.")
                        # archives label frames by obscure catalog codes;
                        # when the spot's brightest star carries a household
                        # designation, put it in the headline ("GN 04.32.8"
                        # meant nothing to the user - it is the HP Tau region)
                        from .pipeline.visualmatch import _name_prestige
                        if _name_prestige(nm) <= 1:
                            # region name from a TIGHT cone (6 arcmin): the
                            # subject star, not a bright field star half a
                            # degree away; pick the most RECOGNIZABLE id
                            _tight = starid.field_star_catalog(
                                top["ra"], top["dec"], 0.1, limit=25)
                            _cands3 = []
                            for c in _tight:
                                _d3 = re.sub(r"^(V\* +|NAME +|\* +|CoKu +)", "",
                                             (c.get("main_id") or "").strip())
                                _d3 = re.sub(r"\s+", " ", _d3)
                                if _d3:
                                    _cands3.append(
                                        (_name_prestige(_d3),
                                         -(c.get("vmag") or 99), _d3))
                            _cands3.sort(reverse=True)
                            if _cands3 and _cands3[0][0] >= 3:
                                _alt_disp = _cands3[0][2]
                                report["headline"] += f" - the {_alt_disp} region"
                                if report.get("object"):
                                    report["object"]["name"] = (
                                        f"{nm} ({_alt_disp} region)")
        except Exception:
            traceback.print_exc()

        try:
            if (not solve.solved and vm_matches
                    and report.get("mode") == "probabilistic"):
                vm = [m for m in vm_matches if m["similarity"] >= 0.60]
                if vm:
                    report["visual_matches"] = vm
                    names = ", ".join(
                        f"{m['name']}" + (f" ({m['common']})" if m["common"] else "")
                        + f" {m['similarity']:.0%}" for m in vm)
                    report.setdefault("observations", []).append(
                        f"Visually similar cataloged objects (experimental "
                        f"appearance search over the full NGC/IC index): {names}. "
                        "Similarity suggests the same CLASS of object, not a "
                        "confirmed identity.")
        except Exception:
            traceback.print_exc()
        # fetch the NASA portrait for whatever the analyzer actually crowned
        if report.get("object") and not report["object"].get("nasa_image"):
            try:
                _oname = report["object"]["name"].split(" - ")[0]
                # our own 26k-image library first: always the right object
                ni = _local_portrait(_oname)
                if ni is None:
                    q = catalogs.sanitize_name_for_search(_oname)
                    ni = catalogs.nasa_image_lookup(q)
                    # VALIDATE the hit: the title must actually mention the
                    # designation, else no portrait beats a wrong one
                    if ni:
                        _qc = re.sub(r"\s+", "", q).lower()
                        _tc = re.sub(r"\s+", "", ni.get("title") or "").lower()
                        if _qc not in _tc:
                            ni = None
                if ni:
                    report["object"]["nasa_image"] = ni
            except Exception:
                pass
        _set_stage(job_id, "analysis", "done", report["headline"])

        # context tags: a detection sitting at the CORE of a galaxy-kind
        # source is that object's nucleus, not a separate star - the tooltip
        # must stop calling it one ("3 orange blobs are galaxies, but we
        # interpreted them as stars" - direct user feedback)
        try:
            ext_sources = [s for s in (report.get("sources") or [])
                           if s.get("kind") in ("galaxy", "nebula", "main", "body")]
            if ext_sources:
                import math
                for st in feats.get("stars") or []:
                    for s in ext_sources:
                        r_core = max(s.get("major_axis_px", 0) * 0.5 * 0.45, 12)
                        if math.hypot(st["x"] - s["center_x"],
                                      st["y"] - s["center_y"]) <= r_core:
                            st["ctx"] = {"n": s["index"], "label": s["label"]}
                            break
        except Exception:
            traceback.print_exc()

        # the Moon disk (if any) gets its own marked, labeled spot
        if feats.get("moon_disk"):
            _md = feats["moon_disk"]
            feats.setdefault("notable_sources", []).append(
                {"x": _md["x"], "y": _md["y"],
                 "label": "The Moon (likely) - resolved lunar disk"})
        overlay = {
            "stars": feats.get("stars", []),
            "notable_sources": feats.get("notable_sources", []),
            "main_source": feats.get("main_source"),
            "dark_structure": feats.get("dark_structure"),
            "sources": report.get("sources") or [],
            "annotations": (solve.get("annotations") or []) if solve.solved else [],
        }
        with _lock:
            _jobs[job_id].update(status="done", report=report, features=feats,
                                 photometry=phot, exif=img.exif, overlay=overlay,
                                 solve={k: v for k, v in solve.items() if k != "annotations"})
        _persist_job(job_id)
    except Exception as exc:  # surface real errors to the UI instead of hanging
        traceback.print_exc()
        with _lock:
            if job_id in _jobs:
                _jobs[job_id].update(status="error", error=f"{type(exc).__name__}: {exc}")


def _mk_vm_hyp(name: str, sim: float) -> dict:
    return {
        "label": f"{name} (matched to its published imagery)",
        "score": 90, "band": "Strong",
        "evidence": [f"{sim:.0%} visual similarity to reference photos of {name}, "
                     "with a decisive lead over every other cataloged object"],
        "notes": ["Identification is by image similarity, not sky position - "
                  "a novel photo of a different object that merely LOOKS "
                  "similar would not reach this similarity level."],
    }


def start_job(data: bytes, filename: str, hint: str | None = None,
              hint_fov: float | None = None) -> str:
    job_id = uuid.uuid4().hex[:12]
    with _lock:
        _jobs[job_id] = {
            "id": job_id,
            "filename": filename,
            "status": "running",
            "stages": {s: {"status": "pending", "detail": ""} for s in STAGES},
            "report": None,
            "error": None,
        }
        _order.append(job_id)
        while len(_order) > MAX_KEPT_JOBS:
            dead = _order.pop(0)
            _jobs.pop(dead, None)
    threading.Thread(target=_run_pipeline, args=(job_id, data, hint, hint_fov),
                     daemon=True).start()
    return job_id
