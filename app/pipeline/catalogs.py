"""Catalog enrichment: SIMBAD TAP cone search + NASA Image Library.

Both services are free and keyless. SIMBAD is shared academic infrastructure:
keep queries rare and cache-friendly (one cone search + one detail query per
analysis is well under the ~5 queries/s limit).
"""
from __future__ import annotations

import urllib.parse

import httpx

SIMBAD_TAP = "https://simbad.cds.unistra.fr/simbad/sim-tap/sync"
NASA_IMAGES = "https://images-api.nasa.gov/search"

# SIMBAD otype (short code) -> human-readable label + one-line science note.
OTYPE_INFO = {
    "G": ("Galaxy", "A gravitationally bound system of stars, gas, dust and dark matter."),
    "GiG": ("Galaxy in a group", "A galaxy that is a member of a small galaxy group."),
    "GiC": ("Galaxy in a cluster", "A galaxy belonging to a larger galaxy cluster."),
    "AGN": ("Active galactic nucleus", "A galaxy whose center hosts an actively accreting supermassive black hole."),
    "SyG": ("Seyfert galaxy", "A galaxy with a bright active nucleus powered by a supermassive black hole."),
    "QSO": ("Quasar", "An extremely luminous active galactic nucleus at cosmological distance."),
    "SBG": ("Starburst galaxy", "A galaxy undergoing an exceptionally high rate of star formation."),
    "EmG": ("Emission-line galaxy", "A galaxy with strong emission lines, indicating active star formation or an AGN."),
    "PN": ("Planetary nebula", "A glowing shell of ionized gas ejected by a dying Sun-like star."),
    "SNR": ("Supernova remnant", "The expanding debris cloud left behind by a stellar explosion."),
    "HII": ("H II region (emission nebula)", "A cloud of ionized hydrogen glowing red (H-alpha) around young hot stars."),
    "RNe": ("Reflection nebula", "Dust scattering the light of nearby stars; typically blue."),
    "DNe": ("Dark nebula", "A dense dust cloud seen in silhouette against a brighter background."),
    "GiP": ("Galaxy in a pair", "A galaxy gravitationally paired with a companion."),
    "IG": ("Interacting galaxies", "Galaxies close enough to distort each other by gravity."),
    "LIN": ("Galaxy with an active nucleus (LINER)",
            "A galaxy whose center shows low-ionization emission - a gently feeding black hole."),
    "AGN": ("Active galactic nucleus", "A galaxy center powered by a feeding supermassive black hole."),
    "SyG": ("Seyfert galaxy", "A galaxy with a bright active nucleus around its central black hole."),
    "Sy1": ("Seyfert 1 galaxy", "An active galaxy whose bright nucleus is seen face-on."),
    "Sy2": ("Seyfert 2 galaxy", "An active galaxy whose nucleus is veiled by dust."),
    "QSO": ("Quasar", "An extremely luminous feeding black hole in a distant galaxy."),
    "EmG": ("Emission-line galaxy", "A galaxy glowing strongly in specific spectral lines."),
    "SBG": ("Starburst galaxy", "A galaxy forming stars at an extreme rate."),
    "GrG": ("Galaxy group", "A small gravitationally bound set of galaxies."),
    "Neb": ("Nebula", "An interstellar cloud of gas and dust."),
    "OpC": ("Open cluster", "A loose group of tens to thousands of young stars born together."),
    "GlC": ("Globular cluster", "A dense, ancient sphere of hundreds of thousands of stars."),
    "Cl*": ("Star cluster", "A gravitationally bound group of stars."),
    "As*": ("Stellar association", "A loose group of young stars with a common origin."),
    "*": ("Star", "A self-luminous ball of plasma powered by nuclear fusion."),
    "**": ("Double or multiple star", "Two or more stars orbiting a common center of mass."),
    "V*": ("Variable star", "A star whose brightness changes over time."),
    "Pu*": ("Pulsating variable star", "A star that rhythmically expands and contracts, changing brightness."),
    "PM*": ("High proper-motion star", "A relatively nearby star moving quickly across the sky."),
    "LM*": ("Low-mass star", "A star smaller and cooler than the Sun."),
    "BD*": ("Brown dwarf", "An object between planet and star: too light to fuse hydrogen steadily."),
    "WD*": ("White dwarf", "The dense, Earth-sized remnant core of a dead Sun-like star."),
    "N*": ("Neutron star", "An ultra-dense stellar remnant; a teaspoon would weigh billions of tons."),
    "Psr": ("Pulsar", "A rotating neutron star sweeping beams of radiation across space."),
    "No*": ("Nova", "A white dwarf undergoing a thermonuclear surface explosion."),
    "SN*": ("Supernova", "A catastrophic stellar explosion, briefly outshining its whole galaxy."),
    "X": ("X-ray source", "An object detected primarily by its X-ray emission."),
    "Rad": ("Radio source", "An object detected primarily at radio wavelengths."),
    "IR": ("Infrared source", "An object detected primarily in the infrared."),
    "gLe": ("Gravitational lens", "A massive object bending and magnifying light from behind it."),
}


def describe_otype(otype: str) -> tuple[str, str]:
    if otype in OTYPE_INFO:
        return OTYPE_INFO[otype]
    # fall back on prefix matches (SIMBAD otypes have many subtypes like "Em*")
    for code, info in OTYPE_INFO.items():
        base = code.rstrip("*")
        # base must be non-empty: "**" stripped to "" prefix-matched EVERY
        # unknown otype, so galaxies (GiP) were labeled "double star"
        if base and code != "*" and otype.startswith(base) and len(code) > 1:
            return info
    if otype.endswith("*"):
        return OTYPE_INFO["*"]
    return (otype or "Unknown type", "No description available for this SIMBAD object type.")


def _tap_query(adql: str, timeout: float = 25.0) -> list[dict]:
    params = {
        "request": "doQuery",
        "lang": "adql",
        "format": "json",
        "query": adql,
    }
    with httpx.Client() as client:
        resp = client.get(SIMBAD_TAP, params=params, timeout=timeout)
        resp.raise_for_status()
        payload = resp.json()
    cols = [c["name"] for c in payload.get("metadata", [])]
    return [dict(zip(cols, row)) for row in payload.get("data", [])]


def cone_search(ra: float, dec: float, radius_deg: float) -> list[dict]:
    """Objects in the solved field, in two passes: the LARGEST extended
    objects anywhere in the field (a famous nebula 4 deg off-center must not
    lose to anonymous stars near the center), then the nearest objects."""
    radius = min(max(radius_deg, 0.05), 15.0)
    cols = (f"main_id, otype, ra, dec, galdim_majaxis, plx_value, rvz_redshift, sp_type, "
            f"DISTANCE(POINT('ICRS', ra, dec), POINT('ICRS', {ra:.6f}, {dec:.6f})) AS center_dist")
    where = (f"CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {ra:.6f}, {dec:.6f}, "
             f"{radius:.4f})) = 1 AND ra IS NOT NULL AND dec IS NOT NULL")
    rows = []
    try:
        # famous extended objects only: Messier/NGC/IC-designated, plausible size
        # (survey catalogs contain "clusters" spanning tens of degrees - noise here)
        rows += _tap_query(f"""
            SELECT TOP 15 b.main_id AS main_id, b.otype AS otype, b.ra AS ra,
                   b.dec AS dec, b.galdim_majaxis AS galdim_majaxis,
                   b.plx_value AS plx_value, b.rvz_redshift AS rvz_redshift,
                   b.sp_type AS sp_type,
                   DISTANCE(POINT('ICRS', b.ra, b.dec),
                            POINT('ICRS', {ra:.6f}, {dec:.6f})) AS center_dist
            FROM basic b JOIN ident i ON i.oidref = b.oid
            WHERE CONTAINS(POINT('ICRS', b.ra, b.dec),
                           CIRCLE('ICRS', {ra:.6f}, {dec:.6f}, {radius:.4f})) = 1
              AND b.ra IS NOT NULL AND b.dec IS NOT NULL
              AND b.galdim_majaxis IS NOT NULL AND b.galdim_majaxis < 700
              AND (i.id LIKE 'M %' OR i.id LIKE 'NGC %' OR i.id LIKE 'IC %')
            ORDER BY galdim_majaxis DESC""")
    except (httpx.HTTPError, ValueError, KeyError):
        pass
    try:
        rows += _tap_query(
            f"SELECT TOP 20 {cols} FROM basic WHERE {where} ORDER BY center_dist ASC")
    except (httpx.HTTPError, ValueError, KeyError):
        pass
    if not rows:
        return []
    seen: set[str] = set()
    rows = [r for r in rows
            if not (r.get("main_id") in seen or seen.add(r.get("main_id")))]

    def notability(row):
        size = row.get("galdim_majaxis") or 0.0  # arcmin; big named objects rank first
        dist_penalty = (row.get("center_dist") or 0.0)
        return (-size, dist_penalty)

    rows.sort(key=notability)
    return [_entry_from_row(row) for row in rows]


def _entry_from_row(row: dict) -> dict:
    from .starid import _pretty_star_name
    otype = (row.get("otype") or "").strip()
    label, note = describe_otype(otype)
    main_id = row.get("main_id", "")
    if main_id.startswith("NAME "):
        main_id = main_id[5:]
    entry = {
        "name": _pretty_star_name(main_id),
        "otype": otype,
        "type_label": label,
        "type_note": note,
        "ra": row.get("ra"),
        "dec": row.get("dec"),
        "angular_size_arcmin": row.get("galdim_majaxis"),
        "spectral_type": (row.get("sp_type") or "").strip() or None,
        "center_distance_deg": round(row.get("center_dist") or 0.0, 4),
    }
    plx = row.get("plx_value")
    if plx and plx > 0:
        entry["distance_ly"] = round(3261.6 / plx, 1)  # mas -> light-years
    z = row.get("rvz_redshift")
    if z and z > 0.001:
        # rough comoving-free estimate: d ~ z * c / H0 (fine for z << 1)
        entry["distance_mly"] = round(z * 299792.458 / 70.0 * 3.2616, 1)
        entry["redshift"] = z
    return entry


def object_by_name(name: str, ra0: float, dec0: float) -> dict | None:
    """Resolve ONE known object by identifier into the cone_search entry
    shape. Needed to inject a visually-established identity into the field
    list: a frame INSIDE the LMC never sees 'NAME LMC' in its small solved
    cone (the object's cataloged center is outside the cone, and it carries
    no M/NGC/IC designation for the famous-extended pass)."""
    import re as _re
    t = (name or "").strip()
    if not t:
        return None
    cands = {t, f"NAME {t}"}
    m = _re.match(r"^([A-Za-z]+)\s*(\d+)$", t)
    if m:
        cands.add(f"{m.group(1).upper()} {m.group(2)}")
        cands.add(f"{m.group(1).upper()}{m.group(2)}")
    ids = ", ".join("'" + c.replace("'", "''") + "'" for c in cands)
    try:
        rows = _tap_query(f"""
            SELECT TOP 1 b.main_id AS main_id, b.otype AS otype, b.ra AS ra,
                   b.dec AS dec, b.galdim_majaxis AS galdim_majaxis,
                   b.plx_value AS plx_value, b.rvz_redshift AS rvz_redshift,
                   b.sp_type AS sp_type,
                   DISTANCE(POINT('ICRS', b.ra, b.dec),
                            POINT('ICRS', {ra0:.6f}, {dec0:.6f})) AS center_dist
            FROM basic b JOIN ident i ON i.oidref = b.oid
            WHERE i.id IN ({ids}) AND b.ra IS NOT NULL AND b.dec IS NOT NULL""")
    except (httpx.HTTPError, ValueError, KeyError):
        return None
    return _entry_from_row(rows[0]) if rows else None


def estimate_physical_size(angular_size_arcmin: float | None,
                           distance_ly: float | None) -> float | None:
    """Physical size in light-years from angular size + distance."""
    if not angular_size_arcmin or not distance_ly:
        return None
    import math
    return round(2 * distance_ly * math.tan(math.radians(angular_size_arcmin / 60.0) / 2), 2)


def nasa_image_lookup(query: str) -> dict | None:
    """First matching image from NASA's free image library (no key)."""
    try:
        with httpx.Client() as client:
            resp = client.get(
                NASA_IMAGES,
                params={"q": query, "media_type": "image"},
                timeout=15,
            )
            resp.raise_for_status()
            items = resp.json().get("collection", {}).get("items", [])
        for item in items[:5]:
            data = (item.get("data") or [{}])[0]
            links = item.get("links") or []
            href = next((l.get("href") for l in links if l.get("href")), None)
            if href:
                return {
                    "title": data.get("title", ""),
                    "description": (data.get("description", "") or "")[:400],
                    "image_url": href,
                    "nasa_id": data.get("nasa_id", ""),
                }
    except (httpx.HTTPError, ValueError, KeyError):
        pass
    return None


def sanitize_name_for_search(name: str) -> str:
    """SIMBAD main_ids like 'M  31' or 'NGC  224' -> friendlier search term."""
    import re
    clean = urllib.parse.unquote(" ".join(name.split()))
    # 'M 42' is too ambiguous for the NASA image search; 'Messier 42' is not
    return re.sub(r"^M (\d+)$", r"Messier \1", clean)
