"""Local plate solving via ASTAP (astap_cli.exe) - no network, no queue.

ASTAP solves in ~1-5 s against its local star database (D50/D80 for normal
fields, W08 for very wide fields). Unlike nova's fully blind solver it wants a
field-of-view estimate, so without an EXIF hint we sweep candidate FOVs.

Solution files: for image.jpg ASTAP writes image.ini (key=value, PLTSOLVD=T,
CRVAL1/2, CDELT2, CROTA2, CD1_1..CD2_2) and image.wcs (FITS-style header) next
to it. We convert the CD matrix into a synthetic FITS WCS header so the rest
of the pipeline (starid, astropy) works identically to the nova path.
"""
from __future__ import annotations

import io
import subprocess
import tempfile
from pathlib import Path

from ..config import PROJECT_ROOT
from .platesolve import PlateSolveResult

_SEARCH_PATHS = [
    PROJECT_ROOT / "astap" / "astap_cli.exe",
    PROJECT_ROOT / "astap" / "astap.exe",
    Path(r"C:\Program Files\astap\astap_cli.exe"),
    Path(r"C:\Program Files\astap\astap.exe"),
]

# 0 = ASTAP's own FOV auto-detection - with a healthy solver it blind-solves
# a 60-deg-offset field in ~1 s, so it goes FIRST. The explicit sweep stays
# as the fallback for images whose header/scale confuses the auto mode.
# D50 is rated for ~0.2-6 deg; much narrower needs the online solver.
FOV_CANDIDATES = [0.0, 20.0, 5.0, 1.0, 0.3]


def find_astap() -> Path | None:
    for p in _SEARCH_PATHS:
        if p.exists():
            return p
    return None


def available() -> bool:
    return find_astap() is not None


def _parse_ini(ini_path: Path) -> dict | None:
    if not ini_path.exists():
        return None
    kv = {}
    for line in ini_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" in line:
            k, _, v = line.partition("=")
            kv[k.strip().upper()] = v.strip()
    if kv.get("PLTSOLVD") != "T":
        return None
    try:
        return {
            "ra": float(kv["CRVAL1"]),
            "dec": float(kv["CRVAL2"]),
            "crpix1": float(kv.get("CRPIX1", 0)) or None,
            "crpix2": float(kv.get("CRPIX2", 0)) or None,
            "scale_deg_per_px": abs(float(kv.get("CDELT2") or kv.get("CDELT1"))),
            "rotation_deg": float(kv.get("CROTA2", kv.get("CROTA1", 0)) or 0),
            "cd": [float(kv[c]) for c in ("CD1_1", "CD1_2", "CD2_1", "CD2_2")]
                  if all(c in kv for c in ("CD1_1", "CD1_2", "CD2_1", "CD2_2")) else None,
        }
    except (KeyError, ValueError):
        return None


def _synthesize_wcs_fits(sol: dict, width: int, height: int) -> bytes | None:
    """Build a minimal FITS header (TAN projection) astropy can consume."""
    try:
        from astropy.io import fits

        hdr = fits.Header()
        hdr["CTYPE1"] = "RA---TAN"
        hdr["CTYPE2"] = "DEC--TAN"
        hdr["CRPIX1"] = sol.get("crpix1") or width / 2.0
        hdr["CRPIX2"] = sol.get("crpix2") or height / 2.0
        hdr["CRVAL1"] = sol["ra"]
        hdr["CRVAL2"] = sol["dec"]
        if sol.get("cd"):
            hdr["CD1_1"], hdr["CD1_2"], hdr["CD2_1"], hdr["CD2_2"] = sol["cd"]
        else:
            import math
            scale = sol["scale_deg_per_px"]
            rot = math.radians(sol.get("rotation_deg", 0.0))
            hdr["CD1_1"] = -scale * math.cos(rot)
            hdr["CD1_2"] = scale * math.sin(rot)
            hdr["CD2_1"] = scale * math.sin(rot)
            hdr["CD2_2"] = scale * math.cos(rot)
        buf = io.BytesIO()
        fits.PrimaryHDU(header=hdr).writeto(buf)
        return buf.getvalue()
    except Exception:
        return None


def solve(image_bytes: bytes, image_width: int, image_height: int,
          fov_hint_deg: float | None = None, fov_list: list | None = None,
          budget_s: float = 75.0, pos_hints: list | None = None,
          progress=lambda msg: None) -> PlateSolveResult:
    """Solve with local ASTAP. Mirrors platesolve.solve()'s result shape.

    fov_list overrides the sweep (e.g. a short list for a fast best-effort try).
    budget_s is a HARD total time budget: a solvable field solves in 1-5 s per
    attempt, so grinding for minutes means the image won't solve at all
    (detail-rich non-sky images once burned 15 minutes here)."""
    import time
    result = PlateSolveResult(solved=False, error=None, skipped=False, solver="astap")
    astap = find_astap()
    if astap is None:
        result.update(skipped=True, error="ASTAP not installed.")
        return result

    deadline = time.monotonic() + budget_s
    # even with an EXIF hint, try auto-FOV afterwards - it is nearly free
    fovs = [fov_hint_deg, 0.0] if fov_hint_deg else (fov_list or FOV_CANDIDATES)
    with tempfile.TemporaryDirectory(prefix="astap_") as td:
        img_path = Path(td) / "field.jpg"
        img_path.write_bytes(image_bytes)
        ini_path = img_path.with_suffix(".ini")

        has_w08 = any(astap.parent.glob("w08*"))

        # POSITION-HINTED attempts first: a visual match against the object
        # index supplies a sky-position guess ("this looks like M16"), and a
        # hinted ASTAP run verifies it astrometrically in ~0.1-10 s. A wrong
        # guess simply fails to solve - the hint can't cause a false positive.
        # 5 slots, not 3: SIMBAD now grants coordinates to lookalike matches
        # too, and a junk hint in front must not evict the right one (the M31
        # WISE frame lost its solve when "Hockey Stick Galaxy" outranked
        # "Andromeda"). A wrong hint costs ~1-3 s; the right one solves in 0.2.
        # dedupe hints within 5 deg (M110/M106 point at the same solve region
        # as Andromeda - -r 20 covers them all in one attempt)
        _hints = []
        for hint in (pos_hints or []):
            if all(abs(hint["ra"] - h2["ra"]) + abs(hint["dec"] - h2["dec"]) > 5
                   for h2 in _hints):
                _hints.append(hint)
            if len(_hints) >= 5:
                break
        for h_i, hint in enumerate(_hints):
            remaining = deadline - time.monotonic()
            if remaining < 3:
                break
            # FAIR SLICE per hint: one wrong first guess used to grind the
            # whole budget with its auto-FOV sweep, starving the correct hint
            # behind it (the M31 WISE frame lost its solve exactly this way -
            # a right hint completes in 0.2-2 s, so ~8 s is generous)
            slice_s = min(10.0, max(remaining / max(len(_hints) - h_i, 1), 3.0))
            label = hint.get("label") or "candidate"
            progress(f"Trying the visual-match position hint ({label})...")
            args = [str(astap), "-f", str(img_path),
                    "-ra", f"{hint['ra'] / 15.0:.4f}",
                    "-spd", f"{hint['dec'] + 90.0:.4f}",
                    "-r", "20", "-fov", "0", "-z", "2"]
            try:
                subprocess.run(args, capture_output=True,
                               timeout=slice_s,
                               cwd=str(astap.parent))
            except subprocess.TimeoutExpired:
                continue
            except OSError:
                break
            sol = _parse_ini(ini_path)
            if sol:
                scale = sol["scale_deg_per_px"]
                radius = scale * (image_width ** 2 + image_height ** 2) ** 0.5 / 2
                result.update(
                    solved=True, ra=sol["ra"], dec=sol["dec"],
                    radius_deg=round(radius, 4),
                    pixscale_arcsec=round(scale * 3600, 3),
                    orientation_deg=sol.get("rotation_deg"),
                    annotations=[],
                    wcs_fits=_synthesize_wcs_fits(sol, image_width, image_height),
                    via_visual_hint=label,
                )
                progress(f"Solved via the {label} position hint!")
                return result

        for fov in fovs:
            remaining = deadline - time.monotonic()
            if remaining < 3:
                result["error"] = (f"Local solve time budget ({budget_s:.0f}s) exhausted "
                                   "without a match.")
                return result
            progress("Local ASTAP solve (auto field of view)..." if not fov else
                     f"Local ASTAP solve, trying field of view {fov:g} deg...")
            # -z 2 downsamples for star extraction: much faster, still reliable
            args = [str(astap), "-f", str(img_path), "-r", "180",
                    "-fov", f"{fov:g}", "-z", "2"]
            if fov >= 20 and has_w08:
                args += ["-D", "W08"]  # wide fields need the wide-field database
            try:
                subprocess.run(
                    args,
                    capture_output=True, timeout=min(25, max(remaining, 3)),
                    cwd=str(astap.parent),
                )
            except subprocess.TimeoutExpired:
                continue  # this FOV is hopeless; try the next within budget
            except OSError as exc:
                result["error"] = f"ASTAP failed to run: {exc}"
                return result
            sol = _parse_ini(ini_path)
            if sol:
                scale = sol["scale_deg_per_px"]
                radius = scale * (image_width ** 2 + image_height ** 2) ** 0.5 / 2
                result.update(
                    solved=True,
                    ra=sol["ra"],
                    dec=sol["dec"],
                    radius_deg=round(radius, 4),
                    pixscale_arcsec=round(scale * 3600, 3),
                    orientation_deg=sol.get("rotation_deg"),
                    annotations=[],
                    wcs_fits=_synthesize_wcs_fits(sol, image_width, image_height),
                )
                progress(f"ASTAP solved at FOV {fov:g} deg.")
                return result

    result["error"] = ("ASTAP could not match the star pattern "
                       "(tried FOVs: " + ", ".join(f"{f:g}" for f in fovs) + " deg).")
    return result
