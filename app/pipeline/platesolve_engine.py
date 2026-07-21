"""Local astrometry.net ENGINE via WSL: nova-grade blind solving, zero queue.

The free nova.astrometry.net queue proved to be a lottery - the same M31
mosaic solved one day and failed four times the next, and identification must
never depend on luck. The engine behind nova is open source, so it now runs
LOCALLY (WSL Ubuntu + the 4100-series index files in data/astrometry_index):
deterministic, offline, still zero cost.
"""
from __future__ import annotations

import subprocess
import tempfile
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INDEX_DIR = PROJECT_ROOT / "data" / "astrometry_index"
TMP_DIR = PROJECT_ROOT / "data" / "tmp_engine"

_available: bool | None = None


def _has_narrow_indexes() -> bool:
    """True once any 5000/5001 narrow-quad tile (2-4') is on disk -
    checked per call (cheap glob) so freshly fetched tiles start helping
    without a restart."""
    return any(INDEX_DIR.glob("index-5000-*.fits")) \
        or any(INDEX_DIR.glob("index-5001-*.fits"))


# ------------------- on-demand narrow index tiles ------------------------- #
# The 5000/5001 series (2-4' quads) unlock blind solving of close-up frames
# for ANY object, famous or not - but the full set is ~50 GB. The series is
# split into 48 healpix sky tiles, and a narrow solve always comes with a
# position hint (visual identity, user hint), so only the tile covering that
# position is ever needed: fetch it on demand (~0.5 GB), keep a bounded LRU
# cache. The pre-installed 4100/5002+ series are never evicted.
NARROW_BASE = "https://data.astrometry.net/5000/"
NARROW_SERIES = ("5001", "5000")
NARROW_CACHE_GB = 14.0


def _healpix_xy(ra: float, dec: float) -> int | None:
    """Tile number for a sky position in astrometry.net's own XY scheme
    (the scheme its index files are named in - verified via get-healpix)."""
    r = _wsl(f"get-healpix -N 2 -- {ra:.6f} {dec:.6f}")
    if not r or r.returncode != 0:
        return None
    import re
    m = re.search(r"Healpix=(\d+) in the XY scheme", r.stdout or "")
    return int(m.group(1)) if m else None


def _ensure_narrow_tiles(ra: float, dec: float, radius_deg: float,
                         progress=lambda msg: None) -> None:
    """Fetch the 5000/5001 tiles covering the hinted position (+ the search
    radius corners, for hints near a tile border). Best-effort: any failure
    just means the solve proceeds without narrow indexes."""
    import httpx

    r = max(min(radius_deg, 5.0), 0.5)
    import math
    cosd = max(abs(math.cos(math.radians(dec))), 0.05)
    probes = [(ra, dec), (ra + r / cosd, dec), (ra - r / cosd, dec),
              (ra, min(dec + r, 89.9)), (ra, max(dec - r, -89.9))]
    tiles = {t for p_ra, p_dec in probes
             if (t := _healpix_xy(p_ra % 360.0, p_dec)) is not None}
    if not tiles:
        return
    wanted = [f"index-{s}-{t:02d}.fits" for s in NARROW_SERIES
              for t in sorted(tiles)]
    missing = [n for n in wanted if not (INDEX_DIR / n).exists()]
    for name in missing:
        try:
            progress(f"Fetching narrow-field index tile {name} "
                     "(one-time, ~0.5 GB)...")
            tmp = INDEX_DIR / (name + ".part")
            with httpx.stream("GET", NARROW_BASE + name, timeout=180,
                              follow_redirects=True) as resp:
                resp.raise_for_status()
                total = int(resp.headers.get("content-length", 0))
                with open(tmp, "wb") as f:
                    for chunk in resp.iter_bytes(chunk_size=1 << 20):
                        f.write(chunk)
            if total and tmp.stat().st_size != total:
                tmp.unlink(missing_ok=True)
                continue
            tmp.replace(INDEX_DIR / name)
        except Exception:
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass
    # LRU eviction over the ON-DEMAND series only (5000/5001): tiles the
    # current request needs are exempt, oldest-accessed go first
    try:
        cached = [p for p in INDEX_DIR.glob("index-500[01]-*.fits")
                  if p.name not in wanted]
        keep = [p for p in INDEX_DIR.glob("index-500[01]-*.fits")
                if p.name in wanted]
        total_gb = sum(p.stat().st_size for p in cached + keep) / 1e9
        cached.sort(key=lambda p: p.stat().st_atime)
        while total_gb > NARROW_CACHE_GB and cached:
            victim = cached.pop(0)
            total_gb -= victim.stat().st_size / 1e9
            victim.unlink()
    except Exception:
        pass


def _wsl(cmd: str, timeout: int = 30) -> subprocess.CompletedProcess | None:
    try:
        return subprocess.run(
            ["wsl", "-d", "Ubuntu", "-u", "root", "--", "bash", "-lc", cmd],
            capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return None


def available() -> bool:
    """solve-field reachable inside WSL and index files on disk (cached)."""
    global _available
    if _available is None:
        has_idx = INDEX_DIR.exists() and any(INDEX_DIR.glob("index-*.fits"))
        r = _wsl("which solve-field") if has_idx else None
        _available = bool(r and r.returncode == 0 and r.stdout.strip())
    return _available


def _win_to_wsl(p: Path) -> str:
    s = str(p.resolve()).replace("\\", "/")
    return "/mnt/" + s[0].lower() + s[2:]


def solve(image_bytes: bytes, image_width: int, image_height: int,
          scale_hint: dict | None = None, timeout_s: int = 120,
          nebulous: bool = False, pos_hint: dict | None = None,
          progress=lambda msg: None) -> dict:
    """Blind-solve with the local engine. Mirrors the nova result shape."""
    result: dict = {"solved": False, "skipped": False, "error": None,
                    "solver": "local-engine"}
    if not available():
        result.update(skipped=True, error="Local engine not installed.")
        return result

    TMP_DIR.mkdir(parents=True, exist_ok=True)
    stem = uuid.uuid4().hex[:10]
    img = TMP_DIR / f"{stem}.jpg"
    img.write_bytes(image_bytes)
    wsl_img = _win_to_wsl(img)

    if scale_hint and scale_hint.get("scale_lower") and scale_hint.get("scale_upper"):
        scale_args = (f"--scale-units degwidth "
                      f"--scale-low {max(scale_hint['scale_lower'], 0.02):.3f} "
                      f"--scale-high {min(scale_hint['scale_upper'], 60):.3f} ")
    else:
        # broad-but-bounded beats fully blind: unscaled search wastes the CPU
        # budget crawling implausible scales. The blind FLOOR follows the
        # installed indexes: with the 5000/5001 narrow series (2-4' quads)
        # on disk, close-up frames down to ~0.05 deg become blind-solvable -
        # famous or not - so don't fence them out.
        floor = 0.05 if _has_narrow_indexes() else 0.2
        scale_args = (f"--scale-units degwidth --scale-low {floor} "
                      "--scale-high 40 ")
    if pos_hint and pos_hint.get("ra") is not None:
        # a visual identity or user hint pins the sky region - the engine
        # then only searches quads there (close-up frames of identified
        # objects solve in seconds instead of failing blind)
        scale_args += (f"--ra {pos_hint['ra']:.5f} --dec {pos_hint['dec']:.5f} "
                       f"--radius {pos_hint.get('radius_deg', 2.0):.2f} ")
        # a pinned region also tells us WHICH narrow-index sky tile could
        # serve it - fetch on demand instead of hosting the full 50 GB set
        _ensure_narrow_tiles(pos_hint["ra"], pos_hint["dec"],
                             pos_hint.get("radius_deg", 2.0), progress)

    solved_marker = TMP_DIR / f"{stem}.solved"
    wcs_file = TMP_DIR / f"{stem}.wcs"
    r = None
    # downsample 2 solves clean optical fields in ~6 s; nebulous/IR mosaics
    # need the aggressive 4 (the M31 WISE frame only solved there: 507 clean
    # sources, log-odds 138) - try both within the budget. For frames the
    # caller KNOWS are nebulous close-ups, start at 4: the ds2 pass wasted
    # ~80 s on every M31-class mosaic. SMALL images (phone/stock, <1600 px)
    # keep every pixel: ds4 left 6 sources in a 1200 px nightscape
    small = max(image_width, image_height) < 1600
    order = (4, 2) if nebulous else ((1, 2) if small else (2, 4))
    budgets = (max(int(timeout_s * 0.4), 25), max(int(timeout_s * 0.6), 40))
    try:
        for ds, bud in zip(order, budgets):
            progress(f"Local astrometry engine: matching quads "
                     f"(downsample {ds})...")
            cmd = (f"solve-field --overwrite --no-plots --crpix-center "
                   f"--downsample {ds} --cpulimit {bud - 5} "
                   f"{scale_args}'{wsl_img}'")
            r = _wsl(cmd, timeout=bud)
            if solved_marker.exists():
                break
        if r is None or not solved_marker.exists() or not wcs_file.exists():
            tail = (r.stdout or "").strip().splitlines()[-1] if (r and r.stdout) else ""
            result["error"] = f"Engine could not solve the field. {tail}".strip()
            return result

        from astropy.io import fits
        from astropy.wcs import WCS
        hdr = fits.getheader(wcs_file)
        # solve-field ran on a --downsample'd copy but its .wcs is in ORIGINAL
        # pixel coordinates already; feed our true image size for the center
        wcs = WCS(hdr)
        ra, dec = (float(v) for v in wcs.pixel_to_world_values(
            image_width / 2.0, image_height / 2.0))
        cd = wcs.pixel_scale_matrix
        import numpy as np
        scale_deg = float(np.sqrt(abs(np.linalg.det(cd))))
        radius = scale_deg * (image_width ** 2 + image_height ** 2) ** 0.5 / 2
        hdr["IMAGEW"] = image_width
        hdr["IMAGEH"] = image_height
        import io
        buf = io.BytesIO()
        fits.PrimaryHDU(header=hdr).writeto(buf)
        result.update(
            solved=True, ra=ra, dec=dec,
            radius_deg=round(radius, 4),
            pixscale_arcsec=round(scale_deg * 3600, 3),
            wcs_fits=buf.getvalue(),
        )
        progress(f"Local engine solved it: RA {ra:.3f}, Dec {dec:.3f}")
        return result
    except Exception as exc:
        result["error"] = f"Engine result parse failed: {type(exc).__name__}: {exc}"
        return result
    finally:
        for q in TMP_DIR.glob(f"{stem}*"):
            try:
                q.unlink()
            except OSError:
                pass
