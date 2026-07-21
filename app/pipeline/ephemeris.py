"""Solar-system reasoning from EXIF capture time (+ optional GPS).

Uses astropy's built-in ephemerides (no network, arcminute-level accuracy -
far better than needed to match a plate-solved field or rank candidates).

Three capabilities:
  1. solved position + time  -> exact planet/Moon/Sun identification
  2. time (+ GPS)            -> which bright solar-system bodies were up
  3. time                    -> lunar phase (illumination + phase name)
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from astropy.coordinates import (AltAz, EarthLocation, SkyCoord, get_body,
                                 get_constellation, solar_system_ephemeris)
from astropy.time import Time
import astropy.units as u

BODIES = ["moon", "mercury", "venus", "mars", "jupiter", "saturn", "sun"]

BODY_FACTS = {
    "sun": "The Sun - a G2V main-sequence star, 1.39 million km across. NEVER observe it without a certified solar filter.",
    "moon": "The Moon - Earth's only natural satellite, 3,474 km across, mean distance 384,400 km.",
    "mercury": "Mercury - the smallest planet (4,879 km), a cratered rocky world with almost no atmosphere.",
    "venus": "Venus - Earth-sized (12,104 km) with a crushing CO2 atmosphere under permanent sulfuric-acid clouds; the brightest planet in our sky.",
    "mars": "Mars - the red planet (6,779 km); its color comes from oxidized iron (rust) dust covering the surface.",
    "jupiter": "Jupiter - the largest planet (139,820 km), a gas giant of hydrogen and helium with banded ammonia clouds.",
    "saturn": "Saturn - a gas giant (116,460 km) famous for its bright water-ice rings.",
}


def parse_exif_datetime(exif: dict) -> datetime | None:
    """EXIF 'YYYY:MM:DD HH:MM:SS' -> aware datetime (assumed local ~ UTC caveat).

    ONLY DateTimeOriginal counts: the generic DateTime tag is the file's last
    EDIT date (web images carry processing dates), and claiming a Moon phase
    from an edit date would be confidently wrong."""
    raw = exif.get("datetime_original")
    if not raw:
        return None
    for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(raw.strip(), fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def _location(exif: dict) -> EarthLocation | None:
    if "latitude" in exif and "longitude" in exif:
        return EarthLocation(lat=exif["latitude"] * u.deg, lon=exif["longitude"] * u.deg)
    return None


def moon_phase(when: datetime) -> dict:
    """Illuminated fraction + phase name from Sun-Moon elongation."""
    t = Time(when)
    with solar_system_ephemeris.set("builtin"):
        sun = get_body("sun", t)
        moon = get_body("moon", t)
    elongation = sun.separation(moon)
    # phase angle ~ 180 deg - elongation (geocentric approximation)
    phase_angle = 180.0 - elongation.deg
    illumination = (1 + np.cos(np.radians(phase_angle))) / 2.0

    # waxing if the Moon is east of the Sun (ecliptic longitude difference)
    dlon = (moon.geocentricmeanecliptic.lon.deg - sun.geocentricmeanecliptic.lon.deg) % 360
    waxing = dlon < 180

    if illumination < 0.03:
        name = "New Moon"
    elif illumination < 0.35:
        name = "Waxing Crescent" if waxing else "Waning Crescent"
    elif illumination < 0.65:
        name = "First Quarter" if waxing else "Last Quarter"
    elif illumination < 0.97:
        name = "Waxing Gibbous" if waxing else "Waning Gibbous"
    else:
        name = "Full Moon"
    return {"illumination": round(float(illumination), 3), "phase_name": name}


def identify_at_position(ra_deg: float, dec_deg: float, radius_deg: float,
                         when: datetime) -> list[dict]:
    """Which solar-system bodies fall inside the solved field at capture time."""
    t = Time(when)
    center = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg)
    hits = []
    with solar_system_ephemeris.set("builtin"):
        for body in BODIES:
            try:
                pos = get_body(body, t)
            except Exception:
                continue
            sep = center.separation(SkyCoord(ra=pos.ra, dec=pos.dec)).deg
            if sep <= max(radius_deg * 1.2, 0.5):
                hits.append({
                    "body": body.capitalize(),
                    "separation_deg": round(float(sep), 3),
                    "fact": BODY_FACTS.get(body, ""),
                })
    hits.sort(key=lambda h: h["separation_deg"])
    return hits


def visible_bright_bodies(when: datetime, exif: dict) -> list[dict]:
    """Bodies above the horizon at capture time (needs GPS for the horizon)."""
    loc = _location(exif)
    if loc is None:
        return []
    t = Time(when)
    frame = AltAz(obstime=t, location=loc)
    out = []
    with solar_system_ephemeris.set("builtin"):
        for body in BODIES:
            try:
                altaz = get_body(body, t).transform_to(frame)
            except Exception:
                continue
            if altaz.alt.deg > 5:
                out.append({
                    "body": body.capitalize(),
                    "altitude_deg": round(float(altaz.alt.deg), 1),
                    "azimuth_deg": round(float(altaz.az.deg), 1),
                    "fact": BODY_FACTS.get(body, ""),
                })
    out.sort(key=lambda h: -h["altitude_deg"])
    return out


def constellation_at(ra_deg: float, dec_deg: float) -> str:
    """IAU constellation containing the position (authoritative boundaries)."""
    return str(get_constellation(SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg)))


def sky_context(solve: dict, exif: dict) -> dict:
    """Everything time/position-based the report can use, best-effort."""
    ctx: dict = {}
    when = parse_exif_datetime(exif)

    if solve.get("solved"):
        ctx["constellation"] = constellation_at(solve["ra"], solve["dec"])
        if when:
            ctx["solar_system_matches"] = identify_at_position(
                solve["ra"], solve["dec"], solve.get("radius_deg") or 0.5, when)

    if when:
        ctx["capture_time_utc"] = when.strftime("%Y-%m-%d %H:%M:%S")
        ctx["moon"] = moon_phase(when)
        ctx["visible_bodies"] = visible_bright_bodies(when, exif)
        ctx["time_caveat"] = ("EXIF time is assumed to be UTC; if the camera clock was set "
                              "to local time, positions can shift slightly.")
    return ctx
