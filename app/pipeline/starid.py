"""Per-star identification: WCS pixel->sky mapping + batched SIMBAD matching.

Requires a successful plate solve (whose FITS WCS header we download). One
single TAP query fetches every cataloged star in the field (respecting CDS
rate limits); detected stars are then matched locally by angular separation.
Proper motion is negligible at consumer pixel scales over decades, so a
positional match is a reliable identification.
"""
from __future__ import annotations

import io
import math
import urllib.parse

import httpx

SIMBAD_TAP = "https://simbad.cds.unistra.fr/simbad/sim-tap/sync"
VIZIER_TAP = "https://tapvizier.cds.unistra.fr/TAPVizieR/tap/sync"


def _load_wcs(wcs_fits: bytes):
    from astropy.io import fits
    from astropy.wcs import WCS
    with fits.open(io.BytesIO(wcs_fits)) as hdul:
        return WCS(hdul[0].header)


def pixels_to_sky(wcs_fits: bytes, points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """(x, y) pixel positions -> [(ra_deg, dec_deg)] via the real WCS solution."""
    wcs = _load_wcs(wcs_fits)
    if not points:
        return []
    world = wcs.pixel_to_world_values([p[0] for p in points], [p[1] for p in points])
    return list(zip([float(r) for r in world[0]], [float(d) for d in world[1]]))


def _tap_json(adql: str) -> list[dict]:
    try:
        resp = httpx.get(SIMBAD_TAP, params={
            "request": "doQuery", "lang": "adql", "format": "json", "query": adql,
        }, timeout=30)
        resp.raise_for_status()
        payload = resp.json()
    except (httpx.HTTPError, ValueError):
        return []
    cols = [c["name"] for c in payload.get("metadata", [])]
    return [dict(zip(cols, row)) for row in payload.get("data", [])
            if row and row[0]]


def field_star_catalog(ra: float, dec: float, radius_deg: float,
                       limit: int = 400) -> list[dict]:
    """All cataloged objects in the field, in TWO passes: brightest stars by
    V magnitude, plus EXTENDED objects (galaxies, nebulae, clusters) queried
    separately - most galaxies have no V flux, so 'ORDER BY vmag' pushed them
    past the TOP cutoff and the overlay could never name them."""
    radius = min(max(radius_deg * 1.15, 0.05), 15.0)
    where = (f"CONTAINS(POINT('ICRS', b.ra, b.dec), "
             f"CIRCLE('ICRS', {ra:.6f}, {dec:.6f}, {radius:.4f})) = 1 "
             "AND b.ra IS NOT NULL AND b.dec IS NOT NULL")
    rows = _tap_json(f"""
        SELECT TOP {limit} b.main_id, b.ra, b.dec, b.otype, b.sp_type, b.plx_value,
               f.V AS vmag
        FROM basic b
        LEFT JOIN allfluxes f ON b.oid = f.oidref
        WHERE {where}
        ORDER BY vmag ASC
    """)
    rows += _tap_json(f"""
        SELECT TOP 200 b.main_id, b.ra, b.dec, b.otype, b.sp_type, b.plx_value,
               NULL AS vmag
        FROM basic b
        WHERE {where} AND b.galdim_majaxis IS NOT NULL
        ORDER BY b.galdim_majaxis DESC
    """)
    if radius <= 0.35:
        # SMALL fields (Hubble close-ups): the vmag/galdim orderings both
        # exclude the anonymous background galaxies (LEDA entries carry
        # neither) - the user's frame showed five OBVIOUS galaxies and every
        # one came back unnamed. A tiny cone holds few hundred rows total,
        # so just take everything.
        rows += _tap_json(f"""
            SELECT TOP 1500 b.main_id, b.ra, b.dec, b.otype, b.sp_type,
                   b.plx_value, NULL AS vmag
            FROM basic b
            WHERE {where}
        """)
    seen: set[str] = set()
    out = []
    for rec in rows:
        name = (rec.get("main_id") or "").strip()
        if name and name not in seen:
            seen.add(name)
            out.append(rec)
    return out


def gaia_field_catalog(ra: float, dec: float, radius_deg: float,
                       limit: int = 60000,
                       order_bright: bool = False) -> list[dict]:
    """DEEP star catalog for the solved field from ESA Gaia DR3 via the free
    VizieR TAP service. SIMBAD's per-field TOP-400 named ~13 of 20,000
    detections on an M31 frame; Gaia's 1.8 BILLION stars reach mag ~21 and
    let the position match name essentially everything real (measured:
    18,107 rows over M31's core in 1.7 s)."""
    radius = min(max(radius_deg * 1.1, 0.05), 4.0)

    def _cone(cra: float, cdec: float, r: float) -> list | None:
        # order_bright: the rotation solver needs the field's TRUE brightest
        # stars - an unordered TOP over a dense cone returns arbitrary
        # magnitudes and the sweep never finds its counterparts (NGC 6822's
        # 0.55-deg Milky-Way-edge cone silently failed this way)
        q = (f'SELECT TOP {limit} "Source", "RA_ICRS", "DE_ICRS", "Gmag", "Plx" '
             f'FROM "I/355/gaiadr3" '
             f"WHERE 1=CONTAINS(POINT('ICRS',\"RA_ICRS\",\"DE_ICRS\"), "
             f"CIRCLE('ICRS', {cra:.6f}, {cdec:.6f}, {r:.4f})) "
             f'AND "Gmag" < 20.8'
             + (' ORDER BY "Gmag"' if order_bright else ""))
        try:
            resp = httpx.get("https://tapvizier.cds.unistra.fr/TAPVizieR/tap/sync",
                             params={"request": "doQuery", "lang": "adql",
                                     "format": "json", "query": q}, timeout=90)
            resp.raise_for_status()
            return resp.json().get("data", [])
        except (httpx.HTTPError, ValueError):
            return None

    # QUADTREE cones: an unordered truncated cone is SPATIALLY biased (every
    # identification once piled onto one half of an M31 frame), and a global
    # magnitude cut costs depth. When a cone fills its row quota, split it
    # into four sub-cones with their OWN quotas: full depth AND full coverage.
    seen: set[int] = set()
    out: list[dict] = []

    def _collect(rows) -> None:
        for row in rows:
            try:
                sid = int(row[0])
                if sid in seen:
                    continue
                seen.add(sid)
                out.append({"source": sid, "ra": float(row[1]),
                            "dec": float(row[2]),
                            "gmag": (float(row[3]) if row[3] is not None else None),
                            "plx": (float(row[4]) if row[4] is not None else None)})
            except (TypeError, ValueError):
                continue

    cosd0 = max(abs(math.cos(math.radians(dec))), 0.05)
    stack = [(ra, dec, radius, 0)]
    while stack:
        cra, cdec, r, depth = stack.pop()
        rows = _cone(cra, cdec, r)
        if rows is None:
            continue
        if len(rows) >= limit and depth < 2 and not order_bright:
            h = r / 2.0
            for dra, ddec in ((-h, -h), (-h, h), (h, -h), (h, h)):
                stack.append((cra + dra / cosd0, cdec + ddec, r * 0.72,
                              depth + 1))
        else:
            _collect(rows)
    return out


def deep_survey_catalog(ra: float, dec: float, radius_deg: float,
                        limit: int = 20000) -> list[dict]:
    """DEEPER than Gaia: the big free ground surveys via VizieR TAP.

    Gaia stops at G~20.8 while a Hubble crop resolves to mag 25+ - a solved
    IC 3225 frame named 9 of 2569 detections and the user rightly asked how
    that is possible. Pan-STARRS DR1 (r~23, dec > -30) and DES DR2 (r~24,
    south) push the naming floor two magnitudes deeper at zero cost."""
    radius = min(max(radius_deg * 1.1, 0.02), 1.0)
    if dec > -29.5:
        table, cra, cde, cmag = '"II/349/ps1"', '"RAJ2000"', '"DEJ2000"', '"rmag"'
        survey = "Pan-STARRS DR1"
        vsrc = "II/349/ps1"
    else:
        table, cra, cde, cmag = '"II/371/des_dr2"', '"RA_ICRS"', '"DE_ICRS"', '"rmag"'
        survey = "DES DR2"
        vsrc = "II/371/des_dr2"
    # ORDER BY brightness, never a bare TOP: VizieR returns rows in spatial
    # storage order, so an unordered TOP over a crowded cone is a CONTIGUOUS
    # SKY CHUNK - 6,213 PS1 names piled into the bottom-left quadrant of a
    # COSMOS frame (44% in one quadrant) while Gaia/SIMBAD stayed uniform.
    # Third appearance of the unordered-TOP lesson. Sorting the FULL cone
    # times out server-side, so it runs in two ordered passes: the bright
    # population first (usually fills the quota), then - only in sparse
    # fields - the faint band to keep the old depth.
    def _fetch(mag_where: str, top: int) -> list:
        # NOT SIMBAD's TAP - the survey tables live on the VizieR TAP server
        q = (f"SELECT TOP {top} {cra}, {cde}, {cmag} FROM {table} "
             f"WHERE 1=CONTAINS(POINT('ICRS',{cra},{cde}), "
             f"CIRCLE('ICRS', {ra:.6f}, {dec:.6f}, {radius:.4f})) "
             f"{mag_where} ORDER BY {cmag}")
        try:
            resp = httpx.get(VIZIER_TAP, params={
                "request": "doQuery", "lang": "adql", "format": "json",
                "query": q,
            }, timeout=90)
            resp.raise_for_status()
            return resp.json().get("data", []) or []
        except (httpx.HTTPError, ValueError):
            return []

    data = _fetch(f"AND {cmag} < 21.5", limit)
    if len(data) < int(limit * 0.6):
        data += _fetch(f"AND {cmag} >= 21.5 AND {cmag} < 23.8",
                       limit - len(data))
    out = []
    for r in data:
        try:
            if r[0] is None or r[1] is None:
                continue
            out.append({"ra": float(r[0]), "dec": float(r[1]),
                        "mag": (float(r[2]) if r[2] is not None else None),
                        "survey": survey, "vsrc": vsrc})
        except (TypeError, ValueError):
            continue
    return out


def hubble_source_catalog(ra: float, dec: float, radius_deg: float,
                          limit: int = 50000) -> list[dict]:
    """Hubble Source Catalog v3 via the free MAST API - mag ~26 depth.

    Built from HST observations up to ~2019, so it covers exactly the famous
    press-image fields (M16 Pillars: 14,537 sources in a 0.03 deg cone).
    Newer HST frames (e.g. the 2024 IC 3225 image) fall outside it - that is
    a coverage fact, not a bug. NOTE: the API radius unit is DEGREES."""
    radius = min(max(radius_deg * 1.1, 0.01), 0.5)
    try:
        resp = httpx.get(
            "https://catalogs.mast.stsci.edu/api/v0.1/hsc/v3/summary",
            params={"ra": ra, "dec": dec, "radius": round(radius, 4),
                    "format": "json", "pagesize": limit},
            timeout=120)
        resp.raise_for_status()
        payload = resp.json()
    except (httpx.HTTPError, ValueError):
        return []
    cols = [c.get("name") for c in payload.get("info", [])]
    try:
        i_ra = cols.index("MatchRA")
        i_de = cols.index("MatchDec")
        i_id = cols.index("MatchID")
        i_ni = cols.index("NumImages")
    except ValueError:
        return []
    out = []
    for r in payload.get("data", []):
        try:
            out.append({"ra": float(r[i_ra]), "dec": float(r[i_de]),
                        "match_id": r[i_id], "n_images": r[i_ni]})
        except (TypeError, ValueError, IndexError):
            continue
    return out


# SIMBAD main_ids are cryptic ("* alf Lyr") - the user saw Vega identified
# correctly and still read it as nonsense. Famous stars get their common name
# up front; the catalog id stays in parentheses so the SIMBAD link still works.
_COMMON_STAR_NAMES = {
    "* alf Lyr": "Vega", "* alf CMa": "Sirius", "* alf Car": "Canopus",
    "* alf Boo": "Arcturus", "* bet Ori": "Rigel", "* alf CMi": "Procyon",
    "* alf Ori": "Betelgeuse", "* alf Eri": "Achernar", "* alf Aql": "Altair",
    "* alf Tau": "Aldebaran", "* alf Sco": "Antares", "* alf Vir": "Spica",
    "* bet Gem": "Pollux", "* alf PsA": "Fomalhaut", "* alf Cyg": "Deneb",
    "* alf Leo": "Regulus", "* alf Aur": "Capella", "* alf Gem": "Castor",
    "* alf UMi": "Polaris", "* alf Cen": "Alpha Centauri",
    "* bet Cen": "Hadar", "* alf Cru": "Acrux", "* bet Cru": "Mimosa",
    "* eps UMa": "Alioth", "* eta UMa": "Alkaid", "* alf Per": "Mirfak",
    "* alf And": "Alpheratz", "* bet And": "Mirach", "* alf Oph": "Rasalhague",
}


def _pretty_star_name(main_id: str) -> str:
    """'* alf Lyr' -> 'Vega (alf Lyr)'; unknown ids pass through unchanged."""
    mid = main_id.strip()
    common = _COMMON_STAR_NAMES.get(mid)
    if not common:
        return mid
    return f"{common} ({mid.lstrip('* ').strip()})"


def _iau_designation(prefix: str, ra: float, dec: float) -> str:
    """Official coordinate-based designation (PS1 Jhhmmss.ss+ddmmss.s)."""
    rh = ra / 15.0
    hh = int(rh)
    mm = int((rh - hh) * 60)
    ss = ((rh - hh) * 60 - mm) * 60
    sign = "+" if dec >= 0 else "-"
    ad = abs(dec)
    dd = int(ad)
    dm = int((ad - dd) * 60)
    ds = ((ad - dd) * 60 - dm) * 60
    return (f"{prefix} J{hh:02d}{mm:02d}{ss:05.2f}"
            f"{sign}{dd:02d}{dm:02d}{ds:04.1f}")


def _sep_arcsec(ra1, dec1, ra2, dec2) -> float:
    cosd = math.cos(math.radians((dec1 + dec2) / 2))
    return math.hypot((ra1 - ra2) * cosd, dec1 - dec2) * 3600.0


def simbad_url(name: str) -> str:
    return ("https://simbad.cds.unistra.fr/simbad/sim-id?Ident="
            + urllib.parse.quote(name))


def identify_stars(wcs_fits: bytes, stars: list[dict], solve: dict,
                   image_height: int, max_named: int = 50000) -> tuple[list[dict], int]:
    """Attach ra/dec to every star and a SIMBAD identity where one exists.

    Solvers disagree on the pixel-origin convention (FITS puts y at the
    bottom-left; images put it at the top-left) - a mismatch mirrors every
    off-center position. We settle it EMPIRICALLY: map the brightest stars
    under both conventions and keep whichever matches the catalog better.

    Mutates/returns copies of the star dicts; returns
    (stars, matched_count, use_flip) - use_flip tells the caller which pixel
    convention won, so other pixel positions can be mapped consistently.
    """
    if not wcs_fits or not stars:
        return stars, 0, False

    import numpy as np
    from scipy.spatial import cKDTree

    catalog = field_star_catalog(solve["ra"], solve["dec"],
                                 solve.get("radius_deg") or 0.5)
    gaia = gaia_field_catalog(solve["ra"], solve["dec"],
                              solve.get("radius_deg") or 0.5)
    try:
        deep = deep_survey_catalog(solve["ra"], solve["dec"],
                                   solve.get("radius_deg") or 0.5)
    except Exception:
        deep = []
    # Hubble Source Catalog: mag ~26 depth, but only where HST pointed
    # before ~2019 - exactly the famous press-image fields. Close-ups only.
    hsc: list[dict] = []
    if (solve.get("radius_deg") or 0.5) <= 0.3:
        try:
            hsc = hubble_source_catalog(solve["ra"], solve["dec"],
                                        solve.get("radius_deg") or 0.3)
        except Exception:
            hsc = []
    # PHYSICAL PLAUSIBILITY GATE: a wide-field frame (phone lens, tens of
    # arcsec per pixel) cannot record faint stars - matching its noise blobs
    # against G=16-20 Gaia entries produced confident garbage names on a
    # phone shot of Vega. Cap the catalog magnitude by what the optics that
    # took THIS frame could plausibly detect. Extended objects (no vmag) are
    # kept: a phone genuinely records M31 or the Orion nebula.
    pixscale = solve.get("pixscale_arcsec") or 2.0
    if pixscale >= 20.0:      # phone / ultra-wide lens
        mag_cap = 9.0
    elif pixscale >= 5.0:     # camera lens / wide field
        mag_cap = 13.0
    else:                     # telescope - deep frames legitimately reach 20+
        mag_cap = None
    if mag_cap is not None:
        gaia = [g for g in gaia
                if g.get("gmag") is not None and g["gmag"] <= mag_cap]
        catalog = [c for c in catalog
                   if c.get("vmag") is None or c["vmag"] <= mag_cap]
        deep, hsc = [], []

    # generous but safe tolerance: a few pixels or 10 arcsec, whichever larger
    tol = max(10.0, 3.0 * (solve.get("pixscale_arcsec") or 2.0))

    cosd = math.cos(math.radians(solve["dec"]))

    def _tree(rows) -> cKDTree | None:
        if not rows:
            return None
        pts_c = np.array([((r["ra"]) * cosd, r["dec"]) for r in rows])
        return cKDTree(pts_c)

    simbad_tree = _tree(catalog)
    gaia_tree = _tree(gaia)
    deep_tree = _tree(deep)
    hsc_tree = _tree(hsc)
    tol_deg = tol / 3600.0
    # deep layers are DENSE (HSC: thousands per close-up cone), so the 10"
    # SIMBAD/Gaia tolerance would attach names by chance alignment - use a
    # pixel-scale-aware tight radius instead (never output a wrong name)
    tol_deep_deg = min(tol_deg,
                       max(3.0, 4.0 * (solve.get("pixscale_arcsec") or 2.0))
                       / 3600.0)

    def _match_count(coords_subset) -> int:
        if simbad_tree is None and gaia_tree is None:
            return 0
        n = 0
        for ra, dec in coords_subset:
            q = ((ra) * cosd, dec)
            for tr in (simbad_tree, gaia_tree):
                if tr is not None:
                    d, _ = tr.query(q)
                    if d < tol_deg:
                        n += 1
                        break
        return n

    probe = stars[:30]  # brightest first (features sorts by flux)
    normal = pixels_to_sky(wcs_fits, [(s["x"], s["y"]) for s in probe])
    flipped = pixels_to_sky(wcs_fits,
                            [(s["x"], image_height - 1 - s["y"]) for s in probe])
    use_flip = _match_count(flipped) > _match_count(normal)

    pts = [(s["x"], image_height - 1 - s["y"]) if use_flip else (s["x"], s["y"])
           for s in stars]
    coords = pixels_to_sky(wcs_fits, pts)

    from .catalogs import describe_otype
    matched = 0
    out = []
    used_simbad: set[int] = set()
    used_gaia: set[int] = set()
    used_deep: set[int] = set()
    used_hsc: set[int] = set()
    for s, (ra, dec) in zip(stars, coords):
        s = dict(s)
        s["ra"] = round(ra, 5)
        s["dec"] = round(dec, 5)
        if matched >= max_named:
            out.append(s)
            continue
        q = (ra * cosd, dec)
        ident = None
        # NAMED SIMBAD objects take priority over Gaia's numeric designations
        if simbad_tree is not None:
            d, i = simbad_tree.query(q)
            # EXTENDED entries (galaxies) get 3x tolerance: the detection
            # centroid of an asymmetric galaxy sits well off its catalog
            # center, and the frame's obvious background galaxies stayed
            # unnamed at the stellar 10-arcsec radius
            _otype_i = (catalog[int(i)].get("otype") or "") if i < len(catalog) else ""
            _tol_i = tol_deg * (3.0 if _otype_i.startswith(("G", "AGN", "EmG",
                                                            "LIN", "SBG",
                                                            "GiG", "GiP"))
                                else 1.0)
            if d < _tol_i and int(i) not in used_simbad:
                used_simbad.add(int(i))
                best = catalog[int(i)]
                label, note = describe_otype((best.get("otype") or "").strip())
                ident = {
                    "name": _pretty_star_name(best["main_id"]),
                    "type_label": label,
                    "type_note": note,
                    "spectral_type": (best.get("sp_type") or "").strip() or None,
                    "v_mag": best.get("vmag"),
                    "match_arcsec": round(d * 3600.0, 1),
                    "url": simbad_url(best["main_id"].strip()),
                }
                plx = best.get("plx_value")
                if plx and plx > 0:
                    ident["distance_ly"] = round(3261.6 / plx, 1)
        if ident is None and gaia_tree is not None:
            d, i = gaia_tree.query(q)
            if d < tol_deg and int(i) not in used_gaia:
                used_gaia.add(int(i))
                g = gaia[int(i)]
                name = f"Gaia DR3 {g['source']}"
                ident = {
                    "name": name,
                    "type_label": "Star (Gaia DR3 catalog)",
                    "type_note": ("Identified against ESA Gaia DR3 - the deep "
                                  "all-sky survey of 1.8 billion stars."),
                    "v_mag": g.get("gmag"),
                    "match_arcsec": round(d * 3600.0, 1),
                    # SIMBAD does NOT hold most Gaia stars (17M vs 1.8B) - a
                    # coordinate query there returned "no object found". Link
                    # the star's OWN catalog record on VizieR instead.
                    "url": ("https://vizier.cds.unistra.fr/viz-bin/VizieR-6?"
                            f"-source=I/355/gaiadr3&Source={g['source']}"),
                }
                plx = g.get("plx")
                if plx and plx > 0.05:
                    ident["distance_ly"] = round(3261.6 / plx, 1)
        # Hubble Source Catalog first among the deep layers: mag ~26 and
        # measured on HST images themselves, so close-up frames match best
        if ident is None and hsc_tree is not None:
            d, i = hsc_tree.query(q)
            if d < tol_deep_deg and int(i) not in used_hsc:
                used_hsc.add(int(i))
                hv = hsc[int(i)]
                ident = {
                    "name": _iau_designation("HSC", hv["ra"], hv["dec"]),
                    "type_label": "Source (Hubble Source Catalog v3)",
                    "type_note": ("Matched against the Hubble Source Catalog "
                                  "(MAST MatchID "
                                  f"{hv.get('match_id')}, seen in "
                                  f"{hv.get('n_images') or 1} HST image(s)). "
                                  "Reaches ~mag 26; star/galaxy nature is "
                                  "not distinguished."),
                    "match_arcsec": round(d * 3600.0, 1),
                    "url": ("https://sky.esa.int/esasky/?target="
                            f"{hv['ra']:.5f}%20{hv['dec']:+.5f}"
                            "&fov=0.05&sci=true"),
                }
        # DEEPER ground surveys: Pan-STARRS/DES reach mag ~23-24 where Gaia
        # stops at 20.8 - this is what turns "9 named of 2569" into hundreds
        if ident is None and deep_tree is not None:
            d, i = deep_tree.query(q)
            if d < tol_deep_deg and int(i) not in used_deep:
                used_deep.add(int(i))
                sv = deep[int(i)]
                prefix = "PS1" if sv["survey"].startswith("Pan") else "DES"
                ident = {
                    "name": _iau_designation(prefix, sv["ra"], sv["dec"]),
                    "type_label": f"Source ({sv['survey']} survey)",
                    "type_note": (f"Matched against {sv['survey']} - a deep "
                                  "ground survey reaching ~2 magnitudes "
                                  "beyond Gaia. Star/galaxy nature is not "
                                  "distinguished at this depth."),
                    "v_mag": sv.get("mag"),
                    "match_arcsec": round(d * 3600.0, 1),
                    "url": ("https://vizier.cds.unistra.fr/viz-bin/VizieR-4?"
                            f"-source={sv['vsrc']}"
                            f"&-c={sv['ra']:.5f}%20{sv['dec']:+.5f}&-c.rs=2"),
                }
        if ident is not None:
            s["id"] = ident
            matched += 1
        out.append(s)

    # BRIGHT-STAR RESCUE: a defocused very bright star (Vega on a phone) is
    # a wide soft disc whose centroid sits several pixels from the true star
    # position - outside the normal tolerance, so it either stays unnamed
    # or, worse, grabs a chance faint neighbour. Stars with V <= 2.5 are so
    # rare (~25 on the whole sky) that a strong detection within a generous
    # radius of one IS that star with near-certainty.
    # Two tiers: V <= 2.5 (~25 stars on the whole sky) get a generous radius;
    # V <= 4.5 (naked-eye) get a moderate one - beta Lyrae sat at the frame
    # EDGE of a phone shot, where wide-angle lens distortion pushes real
    # stars several px off the TAN projection (measured: 3.2 px, tol was 3).
    # Chance-alignment risk at these magnitudes is ~1e-3 per star.
    bright = sorted(
        (c for c in catalog
         if c.get("vmag") is not None and c["vmag"] <= 4.5
         and c.get("ra") is not None and c.get("dec") is not None),
        key=lambda c: c["vmag"])
    if bright and out:
        fluxes = sorted((s.get("flux") or 0.0) for s in out)
        f_hi = fluxes[int(0.75 * (len(fluxes) - 1))]
        for c in bright:
            rescue_deg = (max(4.0 * tol, 25.0 * pixscale)
                          if c["vmag"] <= 2.5 else 1.5 * tol) / 3600.0
            best, best_d = None, rescue_deg
            for s in out:
                if s.get("ra") is None or (s.get("flux") or 0.0) < f_hi:
                    continue
                d = math.hypot((s["ra"] - c["ra"]) * cosd, s["dec"] - c["dec"])
                if d < best_d:
                    best, best_d = s, d
            if best is None:
                continue
            cur = best.get("id")
            if cur and cur.get("v_mag") is not None \
                    and cur["v_mag"] <= c["vmag"]:
                continue  # already named with an equal-or-brighter star
            label, note = describe_otype((c.get("otype") or "").strip())
            if cur is None:
                matched += 1
            best["id"] = {
                "name": _pretty_star_name(c["main_id"]),
                "type_label": label,
                "type_note": note,
                "spectral_type": (c.get("sp_type") or "").strip() or None,
                "v_mag": c.get("vmag"),
                "match_arcsec": round(best_d * 3600.0, 1),
                "rescued": True,
                "url": simbad_url(c["main_id"].strip()),
            }
            plx = c.get("plx_value")
            if plx and plx > 0:
                best["id"]["distance_ly"] = round(3261.6 / plx, 1)
    return out, matched, use_flip
