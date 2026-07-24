"""Appearance-based object lookup against the visual reference index.

For photos that cannot plate-solve (extreme close-ups, single-object frames),
compare the image against survey reference cutouts of EVERY OpenNGC object
(built by scripts/build_visual_index.py + build_visual_embeddings.py) using
ImageNet-pretrained MobileNetV3 features and cosine similarity.

Honesty contract: press images differ wildly from survey cutouts (palette,
band, framing), so this reports "visually similar known objects" with scores -
it upgrades to an identification claim ONLY on a decisive score+gap, and the
UI wording stays hedged.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from ..config import PROJECT_ROOT

INDEX_DIR = PROJECT_ROOT / "data" / "visual_index"
EMB_PATH = INDEX_DIR / "embeddings.npz"

_model = None
_index = None  # (names: list[str], commons: list[str], vecs: np.ndarray)


def _backbone():
    global _model
    if _model is None:
        import torch
        from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
        m = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        m.eval()
        _model = m
    return _model


_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def embed(rgb: np.ndarray) -> np.ndarray:
    """L2-normalized 576-d feature for one RGB float array (any size)."""
    import torch
    from PIL import Image

    im = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))
    im = im.resize((224, 224), Image.BILINEAR)
    x = (np.asarray(im, dtype=np.float32) / 255.0 - _MEAN) / _STD
    t = torch.from_numpy(x.transpose(2, 0, 1)[None])
    m = _backbone()
    with torch.no_grad():
        f = m.avgpool(m.features(t)).flatten(1).numpy()[0]
    n = np.linalg.norm(f)
    return (f / n if n > 0 else f).astype(np.float32)


def available() -> bool:
    return EMB_PATH.exists()


# Messier -> NGC aliases: press images are named "M82" but the coordinate
# catalog (OpenNGC cutouts) keys on zero-padded NGC ids, so an M-name match
# silently lost its position hint - and with it the whole identity chain.
_MESSIER_NGC = {
    1: 1952, 2: 7089, 3: 5272, 4: 6121, 5: 5904, 6: 6405, 7: 6475, 8: 6523,
    9: 6333, 10: 6254, 11: 6705, 12: 6218, 13: 6205, 14: 6402, 15: 7078,
    16: 6611, 17: 6618, 18: 6613, 19: 6273, 20: 6514, 21: 6531, 22: 6656,
    23: 6494, 26: 6694, 27: 6853, 28: 6626, 29: 6913, 30: 7099, 31: 224,
    32: 221, 33: 598, 34: 1039, 35: 2168, 36: 1960, 37: 2099, 38: 1912,
    39: 7092, 41: 2287, 42: 1976, 43: 1982, 44: 2632, 46: 2437, 47: 2422,
    48: 2548, 49: 4472, 50: 2323, 51: 5194, 52: 7654, 53: 5024, 54: 6715,
    55: 6809, 56: 6779, 57: 6720, 58: 4579, 59: 4621, 60: 4649, 61: 4303,
    62: 6266, 63: 5055, 64: 4826, 65: 3623, 66: 3627, 67: 2682, 68: 4590,
    69: 6637, 70: 6681, 71: 6838, 72: 6981, 73: 6994, 74: 628, 75: 6864,
    76: 650, 77: 1068, 78: 2068, 79: 1904, 80: 6093, 81: 3031, 82: 3034,
    83: 5236, 84: 4374, 85: 4382, 86: 4406, 87: 4486, 88: 4501, 89: 4552,
    90: 4569, 91: 4548, 92: 6341, 93: 2447, 94: 4736, 95: 3351, 96: 3368,
    97: 3587, 98: 4192, 99: 4254, 100: 4321, 101: 5457, 102: 5866, 103: 581,
    104: 4594, 105: 3379, 106: 4258, 107: 6171, 108: 3556, 109: 3992,
    110: 205,
}

# Famous objects with no NGC id (or none the cutout catalog carries):
# direct ICRS coordinates so their identifications still seed the solver.
_FAMOUS_COORDS = {
    "M24": (274.68, -18.55), "M25": (277.94, -19.12),
    "M40": (185.55, 58.08), "M45": (56.75, 24.12),
    "M87* black hole": (187.7059, 12.3911),
    "Sagittarius A* black hole": (266.4168, -29.0078),
    "AG Carinae": (164.0479, -60.4536),
    "WR 124": (287.8786, 16.8606),
}


# famous common names -> the catalog key the coordinate index actually holds
# (press feeds label images "Antennae Galaxies", the index knows "NGC4038")
_COMMON_KEY = {
    "omega centauri": "NGC5139", "antennae": "NGC4038",
    "antennae galaxies": "NGC4038", "whirlpool galaxy": "NGC5194",
    "sombrero galaxy": "NGC4594", "crab nebula": "NGC1952",
    "orion nebula": "NGC1976", "carina nebula": "NGC3372",
    "eagle nebula": "NGC6611", "lagoon nebula": "NGC6523",
    "trifid nebula": "NGC6514", "helix nebula": "NGC7293",
    "ring nebula": "NGC6720", "tarantula nebula": "NGC2070",
    "centaurus a": "NGC5128", "pinwheel galaxy": "NGC5457",
    "andromeda galaxy": "NGC0224", "triangulum galaxy": "NGC0598",
    "butterfly nebula": "NGC6302", "veil nebula": "NGC6960",
    "bubble nebula": "NGC7635", "cat's eye nebula": "NGC6543",
    "dumbbell nebula": "NGC6853", "sculptor galaxy": "NGC0253",
    "bode's galaxy": "NGC3031", "cigar galaxy": "NGC3034",
    "black eye galaxy": "NGC4826", "stephan's quintet": "NGC7318",
    "barnards galaxy": "NGC6822", "barnard's galaxy": "NGC6822",
    "evil eye galaxy": "NGC4826", "antennae galaxies": "NGC4038",
    "whirlpool": "NGC5194", "sombrero": "NGC4594",
}


_OBSCURE = re.compile(r"^(GN|LBN|LDN|GAL|IRAS|2MASS|SDSS|GALEX|UCAC|TYC|"
                      r"Gaia|PGC|LEDA|UGCA|ESO|MCG|CGCG|KUG|Ced|DG|VdB)\b", re.I)


def _name_prestige(name: str) -> int:
    """How recognizable a designation is - merged aliases keep the best one
    (the HP Tau frame was shown under its nebula code 'GN 04.32.8')."""
    # SIMBAD pads ids with double spaces ('HD  28976') - normalize first
    n = re.sub(r"\s+", " ", name.strip())
    if re.fullmatch(r"(M|Messier)\s?\d{1,3}", n, re.I):
        return 6
    if re.fullmatch(r"(NGC|IC)\s?\d{1,4}[A-B]?", n, re.I):
        return 5
    if not any(ch.isdigit() for ch in n):
        return 5  # proper names: Antennae Galaxies, Orion Nebula...
    if re.fullmatch(r"[A-Z]{1,2}\d? [A-Z][a-z]{2}( [A-Z]?\d)?", n):
        return 4  # variable-star style: HP Tau, RS Pup, HP Tau G2
    if re.fullmatch(r"(HD|HR|SAO) ?\d+[A-B]?", n, re.I):
        return 3
    if _OBSCURE.match(n):
        return 0
    return 1


def _coord_keys(raw: str, disp: str):
    """Candidate coordinate-catalog keys for a match name, aliases included."""
    keys = [raw, disp, disp.replace(" ", "")]
    m = re.fullmatch(r"(?:M|Messier)\s?(\d{1,3})", disp.strip(), re.I)
    if m and int(m.group(1)) in _MESSIER_NGC:
        keys.append("NGC%04d" % _MESSIER_NGC[int(m.group(1))])
    ck = _COMMON_KEY.get(disp.strip().lower())
    if ck:
        keys.append(ck)
    # zero-pad NGC/IC ids: the cutout catalog keys are NGC0224-style
    m = re.fullmatch(r"(NGC|IC)\s?(\d{1,4})", disp.strip(), re.I)
    if m:
        keys.append("%s%04d" % (m.group(1).upper(), int(m.group(2))))
    return keys


_index_mtime = None


def _load_index():
    global _index, _index_mtime
    # RELOAD when the archive harvest rewrites embeddings.npz: a running
    # server was matching against a 16k index while 24k sat on disk
    if EMB_PATH.exists() and _index is not None:
        try:
            if EMB_PATH.stat().st_mtime != _index_mtime:
                _index = None
        except OSError:
            pass
    if _index is None and EMB_PATH.exists():
        _index_mtime = EMB_PATH.stat().st_mtime
        z = np.load(EMB_PATH, allow_pickle=True)
        # archives republish the same frame under other subjects (the Antennae
        # portrait also shipped inside a supernova release) - those duplicate
        # refs split the vote and kill the identification lead, so the dedupe
        # scan's suppression list is honored here
        _sup_path = INDEX_DIR / "suppressed_refs.txt"
        if _sup_path.exists():
            _sup = {s.strip() for s in
                    _sup_path.read_text(encoding="utf-8").splitlines() if s.strip()}
            if _sup:
                _keep = [i for i, n in enumerate(z["names"]) if n not in _sup]
                z = {"names": [z["names"][i] for i in _keep],
                     "commons": [z["commons"][i] for i in _keep],
                     "vecs": z["vecs"][_keep]}
        # sky coordinates from meta.csv: a visual match doubles as a POSITION
        # GUESS, which a hinted plate solve can then verify exactly
        coords: dict[str, tuple[float, float]] = {}
        meta = INDEX_DIR / "meta.csv"
        if meta.exists():
            import csv
            with open(meta, encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    try:
                        coords[r["name"]] = (float(r["ra"]), float(r["dec"]))
                    except (KeyError, ValueError):
                        continue
        _index = (list(z["names"]), list(z["commons"]),
                  z["vecs"].astype(np.float32), coords)
    return _index


def _blobs(gray: np.ndarray, n: int = 12):
    """Brightest compact blobs (galaxies + stars) as (x, y) points."""
    from scipy import ndimage as ndi

    sm = ndi.gaussian_filter(gray, 1.5)
    res = sm - ndi.gaussian_filter(gray, 12)
    peak = ndi.maximum_filter(res, size=7)
    thr = float(res.std()) * 2.5
    ys, xs = np.nonzero((res == peak) & (res > thr))
    if len(ys) == 0:
        return np.zeros((0, 2)), np.zeros(0)
    vals = res[ys, xs]
    order = np.argsort(-vals)[:n]
    ys, xs, vals = ys[order], xs[order], vals[order]
    # sub-pixel centroids: quad codes need ~1 px repeatability to be
    # discriminative (integer peaks jittered codes by 0.03-0.06)
    h, w = res.shape
    cx, cy = [], []
    for y, x in zip(ys, xs):
        y0, y1 = max(y - 3, 0), min(y + 4, h)
        x0, x1 = max(x - 3, 0), min(x + 4, w)
        win = np.clip(res[y0:y1, x0:x1], 0, None)
        s = win.sum()
        if s > 0:
            gy, gx = np.mgrid[y0:y1, x0:x1]
            cy.append(float((win * gy).sum() / s))
            cx.append(float((win * gx).sum() / s))
        else:
            cy.append(float(y)); cx.append(float(x))
    return (np.column_stack([cx, cy]).astype(np.float64),
            vals.astype(np.float64))


def _fit_pattern(qp: np.ndarray, rp: np.ndarray,
                 smin: float = 0.3, smax: float = 3.5):
    """RANSAC similarity transform (scale+rotation+shift) mapping query blob
    points onto reference blob points. Returns (A 2x2, t 2, inliers, rms)."""
    best = None
    nq, nr = len(qp), len(rp)
    if nq < 5 or nr < 5:
        return None
    for i in range(nq):
        for j in range(i + 1, nq):
            dq = qp[j] - qp[i]
            lq = np.hypot(*dq)
            if lq < 18:
                continue
            for k in range(nr):
                for l in range(nr):
                    if k == l:
                        continue
                    dr = rp[l] - rp[k]
                    lr = np.hypot(*dr)
                    s = lr / lq
                    if not smin <= s <= smax:
                        continue
                    ca = (dq[0] * dr[0] + dq[1] * dr[1]) / (lq * lr)
                    sa = (dq[0] * dr[1] - dq[1] * dr[0]) / (lq * lr)
                    A = s * np.array([[ca, -sa], [sa, ca]])
                    t = rp[k] - A @ qp[i]
                    proj = qp @ A.T + t
                    d = np.sqrt(((proj[:, None, :] - rp[None, :, :]) ** 2).sum(-1))
                    mind = d.min(axis=1)
                    inl = mind < 4.0
                    n_in = int(inl.sum())
                    if n_in >= 5:
                        rms = float(np.sqrt((mind[inl] ** 2).mean()))
                        score = n_in - rms / 4.0
                        if best is None or score > best[0]:
                            best = (score, A, t, n_in, rms)
    if best is None:
        return None
    _, A, t, n_in, rms = best
    return A, t, n_in, rms


def _warp_ncc(q: np.ndarray, ref: np.ndarray, A: np.ndarray,
              t: np.ndarray) -> float:
    """Normalized cross-correlation of the query warped into the reference
    frame. The decisive second factor against chance point alignments: two
    images of the SAME sky correlate strongly; six accidentally-aligned blobs
    between different fields do not (a wrong field once locked with 6 points
    and named everything from the wrong position)."""
    from scipy import ndimage as ndi

    try:
        Ainv = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        return 0.0
    # ref pixel (x,y) -> query pixel: q = Ainv @ (r - t); affine_transform maps
    # output[y,x] = input[(M @ [y,x] + off)], row-major
    M = np.array([[Ainv[1, 1], Ainv[1, 0]], [Ainv[0, 1], Ainv[0, 0]]])
    off = Ainv @ (-t)
    warped = ndi.affine_transform(q, M, offset=[off[1], off[0]],
                                  output_shape=ref.shape, order=1, cval=-1.0)
    valid = warped >= 0
    if valid.sum() < 800:
        return 0.0
    a = warped[valid]
    b = ref[valid]
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / denom) if denom > 0 else 0.0


_pattern_idx = None
_tile_idx = None


def identity_rotation_solve(det_xy: np.ndarray, obj_ra: float, obj_dec: float,
                            obj_cx: float, obj_cy: float, scale_est: float,
                            full_w: int, full_h: int) -> dict | None:
    """The user's insight completed: once the OBJECT is identified, its
    catalog position pins the field and its known angular size pins the
    scale - the only unknown left is ROTATION (plus mirror parity). Sweep it:
    project the field's Gaia stars into pixel space at every angle and keep
    the one where they land on our detections. Self-verifying: a wrong
    identity or scale simply never beats the match threshold.

    det_xy: detection pixel coords (brightest first). scale_est: deg/px.
    Returns a solve dict (wcs_fits etc.) or None."""
    from scipy.spatial import cKDTree

    from .starid import gaia_field_catalog

    if len(det_xy) < 6:
        return None
    diag = (full_w ** 2 + full_h ** 2) ** 0.5
    gaia = gaia_field_catalog(obj_ra, obj_dec,
                              min(scale_est * diag * 0.7, 1.0), limit=3000,
                              order_bright=True)
    if len(gaia) < 6:
        return None
    # brightest catalog stars only: they are the ones our detector sees
    gaia.sort(key=lambda g: (g.get("gmag") if g.get("gmag") is not None else 99))
    gaia = gaia[:400]
    d0 = np.radians(obj_dec)
    ra_arr = np.radians(np.array([g["ra"] for g in gaia]))
    de_arr = np.radians(np.array([g["dec"] for g in gaia]))
    ra0 = np.radians(obj_ra)
    # gnomonic (TAN) projection about the object centre, in degrees
    cosc = (np.sin(d0) * np.sin(de_arr)
            + np.cos(d0) * np.cos(de_arr) * np.cos(ra_arr - ra0))
    xi = np.degrees(np.cos(de_arr) * np.sin(ra_arr - ra0) / cosc)
    eta = np.degrees((np.cos(d0) * np.sin(de_arr)
                      - np.sin(d0) * np.cos(de_arr) * np.cos(ra_arr - ra0)) / cosc)

    tree = cKDTree(det_xy[:250])

    def _count(s, parity, th_deg, tol):
        th = np.radians(th_deg)
        c, sn = np.cos(th), np.sin(th)
        # sky (xi, eta) -> pixel offsets; xi flips with RA direction
        px = obj_cx + (parity * (-xi) * c - eta * sn) / s
        py = obj_cy + (parity * (-xi) * sn + eta * c) / s * -1.0
        inside = ((px >= -20) & (px < full_w + 20)
                  & (py >= -20) & (py < full_h + 20))
        if inside.sum() < 5:
            return 0, 0
        d, _ = tree.query(np.column_stack([px[inside], py[inside]]),
                          distance_upper_bound=tol)
        return int(np.isfinite(d).sum()), int(inside.sum())

    # COARSE-TO-FINE: one loose sweep only RANKS candidate poses (a loose
    # tolerance saturates with chance hits, so it must not accept anything);
    # each finalist is then refined on a dense local grid and judged at a
    # TIGHT tolerance against its own chance expectation. The old single
    # sweep used a 12-arcsec tolerance that became 80 px on HST close-ups -
    # chance-saturated - while a tight tolerance alone missed real poses
    # sitting between the coarse 15-20% scale steps.
    # cap at 28 px, not 12: Hubble press mosaics are drizzle-warped, so even
    # the TRUE pose lands its counterparts 10-30 px off a rigid gnomonic
    # (measured on NGC 1300: 26 hits at 28 px, <=7 at 12 px)
    tol_fine = min(max(4.0, 12.0 / (scale_est * 3600.0)), 28.0)
    tol_coarse = 45.0
    cands = []
    for s_mul in (0.62, 0.72, 0.85, 1.0, 1.18, 1.38, 1.6):
        s = scale_est * s_mul
        for parity in (1.0, -1.0):
            for th_deg in range(0, 360, 3):
                n_hit, n_in = _count(s, parity, th_deg, tol_coarse)
                if n_hit >= 6:
                    cands.append((n_hit, s, parity, th_deg, n_in))
    if not cands:
        return None
    cands.sort(reverse=True)
    _chance_frac = (min(len(det_xy), 250) * np.pi * tol_fine ** 2
                    / (full_w * full_h))
    best = None
    for _, s0, parity, th0, _ in cands[:10]:
        for ds in (-0.10, -0.07, -0.045, -0.03, -0.015, 0.0,
                   0.015, 0.03, 0.045, 0.07, 0.10):
            for dth in (-2.25, -1.5, -0.75, 0.0, 0.75, 1.5, 2.25):
                s = s0 * (1.0 + ds)
                n_hit, n_in = _count(s, parity, th0 + dth, tol_fine)
                if best is None or n_hit > best[0]:
                    best = (n_hit, s, parity, th0 + dth, n_in)
    if best is None:
        return None
    n_hit, s, parity, th_deg, n_in = best
    # statistical acceptance: the pose must beat its own chance expectation
    # by >=4 sigma (Poisson). The historic true lock measured 26 hits vs ~10
    # expected by chance - significant, yet any fixed multiple would reject it
    _chance_hits = n_in * _chance_frac
    need = int(_chance_hits + max(6.0, 4.0 * np.sqrt(max(_chance_hits, 1.0))))
    if n_hit < need:
        return None
    tol_px = tol_fine
    # rather than deriving CD analytically (parity/rotation algebra is easy
    # to get subtly wrong), FIT it from correspondences: matched pairs of
    # (pixel, xi/eta) give the linear map directly
    th = np.radians(th_deg)
    c, sn = np.cos(th), np.sin(th)
    px = obj_cx + (parity * (-xi) * c - eta * sn) / s
    py = obj_cy + (parity * (-xi) * sn + eta * c) / s * -1.0
    d, idx = tree.query(np.column_stack([px, py]), distance_upper_bound=tol_px)
    ok = np.isfinite(d)
    if ok.sum() < 5:
        return None
    P = np.column_stack([det_xy[:250][idx[ok].astype(int), 0] - obj_cx,
                         det_xy[:250][idx[ok].astype(int), 1] - obj_cy,
                         np.ones(int(ok.sum()))])
    X = np.linalg.lstsq(P, xi[ok], rcond=None)[0]
    E = np.linalg.lstsq(P, eta[ok], rcond=None)[0]
    try:
        import io as _io

        from astropy.io import fits
        hdr = fits.Header()
        hdr["CTYPE1"], hdr["CTYPE2"] = "RA---TAN", "DEC--TAN"
        hdr["CRVAL1"], hdr["CRVAL2"] = obj_ra, obj_dec
        # CRPIX where xi=eta=0: solve the tiny 2x2 system
        A2 = np.array([[X[0], X[1]], [E[0], E[1]]])
        b2 = -np.array([X[2], E[2]])
        off = np.linalg.solve(A2, b2)
        hdr["CRPIX1"], hdr["CRPIX2"] = obj_cx + off[0], obj_cy + off[1]
        hdr["CD1_1"], hdr["CD1_2"] = X[0], X[1]
        hdr["CD2_1"], hdr["CD2_2"] = E[0], E[1]
        buf = _io.BytesIO()
        fits.PrimaryHDU(header=hdr).writeto(buf)
        pixscale = float(np.sqrt(abs(np.linalg.det(A2)))) * 3600.0
        # scale consistency: the FITTED scale must agree with the identity's
        # catalog-size estimate - a degenerate chance fit drifts far off
        if not (0.4 * scale_est * 3600.0 <= pixscale <= 2.5 * scale_est * 3600.0):
            return None
        return {"ra": obj_ra, "dec": obj_dec,
                "radius_deg": round(pixscale / 3600.0 * diag / 2, 4),
                "pixscale_arcsec": round(pixscale, 3),
                "wcs_fits": buf.getvalue(), "annotations": [],
                "id_rotation_deg": th_deg, "id_matches": n_hit,
                "id_parity": parity}
    except Exception:
        return None


def nucleus_polar_solve(rgb: np.ndarray, obj_ra: float, obj_dec: float,
                        scale_est: float, obj_cx: float, obj_cy: float,
                        det_xy: np.ndarray | None, full_w: int,
                        full_h: int) -> dict | None:
    """Close-up WCS via NUCLEUS-anchored log-polar morphology + Gaia proof.

    The user's argument, engineered: "I hand you the Maiden's Tower and you
    can't point at the Bosphorus." An identified galaxy pins position and
    scale; its nucleus is a sharp anchor in BOTH our photo and the survey
    image (translation solved by anchoring), and in log-polar coordinates
    around that anchor rotation and scale become simple shifts of the arm
    pattern. Per-radius normalization cancels the band-to-band brightness
    profile, so only the STRUCTURE angle matters. The resulting pose is then
    verified INDEPENDENTLY by projecting Gaia stars: at a 6 px tolerance even
    5-6 coincidences are decisive, so the star-scarcity that killed the pure
    point-matching solver is no longer fatal."""
    import io as _io

    import httpx
    from scipy import ndimage as ndi

    # --- survey reference with exact WCS (free hips2fits) ---------------
    fov = scale_est * max(full_w, full_h) * 1.6
    hips_list = (["CDS/P/DES-DR2/r", "CDS/P/DSS2/red"] if obj_dec < -29.5
                 else ["CDS/P/PanSTARRS/DR1/r", "CDS/P/DSS2/red"])
    ref = ref_wcs = None
    for hips in hips_list:
        try:
            r = httpx.get(
                "https://alasky.cds.unistra.fr/hips-image-services/hips2fits",
                params={"hips": hips, "width": 400, "height": 400,
                        "fov": fov, "projection": "TAN", "coordsys": "icrs",
                        "ra": obj_ra, "dec": obj_dec, "format": "fits"},
                timeout=60)
            r.raise_for_status()
            from astropy.io import fits as _f
            from astropy.wcs import WCS as _W
            hdu = _f.open(_io.BytesIO(r.content))[0]
            cand = np.nan_to_num(np.asarray(hdu.data, dtype=np.float32))
            if cand.ndim == 2 and float(cand.std()) > 0:
                ref, ref_wcs = cand, _W(hdu.header)
                break
        except Exception:
            continue
    if ref is None:
        return None
    lo, hi = np.percentile(ref, (2, 99.8))
    ref = np.clip((ref - lo) / max(hi - lo, 1e-6), 0, 1)

    gray = np.clip(rgb, 0, 1).mean(axis=2).astype(np.float32)

    def _nucleus(img, cx0, cy0, rad):
        sm = ndi.gaussian_filter(img, 3.0)
        y0, y1 = int(max(cy0 - rad, 0)), int(min(cy0 + rad, img.shape[0]))
        x0, x1 = int(max(cx0 - rad, 0)), int(min(cx0 + rad, img.shape[1]))
        sub = sm[y0:y1, x0:x1]
        if sub.size == 0:
            return cx0, cy0
        iy, ix = np.unravel_index(int(np.argmax(sub)), sub.shape)
        return x0 + ix, y0 + iy

    q_nx, q_ny = _nucleus(gray, obj_cx, obj_cy,
                          0.15 * max(full_w, full_h))
    r_nx, r_ny = _nucleus(ref, 200, 200, 80)

    # --- band-pass then log-polar sample around each nucleus ------------
    def _prep(img):
        return (ndi.gaussian_filter(img, 1.5)
                - ndi.gaussian_filter(img, 10.0))

    ref_scale = fov / 400.0                      # deg/px of the reference
    # CRITICAL: resample the query to the reference's SKY scale before the
    # band-pass - filtering both in their own pixel units kept structures of
    # 0.3-2 arcsec in the HST query vs 3-15 arcsec in the survey reference:
    # non-overlapping bands, so the true pose correlated at nothing while a
    # spurious peak won the sweep
    _shrink = scale_est / ref_scale
    q_small = ndi.zoom(gray, _shrink, order=1)
    qs_nx, qs_ny = None, None  # nucleus recomputed below in small coords
    qb, rb = _prep(q_small), _prep(ref)
    n_th, n_r = 180, 80
    thetas = np.linspace(0, 2 * np.pi, n_th, endpoint=False)

    def _polar(img, nx, ny, r_min_px, r_max_px):
        rr = np.exp(np.linspace(np.log(r_min_px), np.log(r_max_px), n_r))
        tt, rg = np.meshgrid(thetas, rr)
        xs = nx + rg * np.cos(tt)
        ys = ny + rg * np.sin(tt)
        out = ndi.map_coordinates(img, [ys.ravel(), xs.ravel()], order=1,
                                  cval=0.0).reshape(n_r, n_th)
        # per-radius normalization: kills the cross-band radial brightness
        # profile, keeps the arm pattern along theta
        out = out - out.mean(axis=1, keepdims=True)
        sd = out.std(axis=1, keepdims=True)
        return out / np.maximum(sd, 1e-5)

    # radial range: 4%..55% of the reference frame (galaxy disk annulus);
    # both images are now at the SAME sky scale, so the radii match directly
    r_ref_min, r_ref_max = 8.0, 110.0
    P_ref = _polar(rb, r_nx, r_ny, r_ref_min, r_ref_max)
    qs_nx, qs_ny = q_nx * _shrink, q_ny * _shrink
    P_q = _polar(qb, qs_nx, qs_ny, r_ref_min, r_ref_max)

    dlnr = (np.log(r_ref_max) - np.log(r_ref_min)) / (n_r - 1)
    best = None
    scores = []
    for parity in (1, -1):
        Q = P_q if parity == 1 else P_q[:, ::-1]
        for k in range(-12, 13):          # log-r shift: scale +-45%
            if k >= 0:
                A_, B_ = Q[k:, :], P_ref[:n_r - k, :]
            else:
                A_, B_ = Q[:n_r + k, :], P_ref[-k:, :]
            # circular cross-correlation along theta via FFT
            fa = np.fft.rfft(A_, axis=1)
            fb = np.fft.rfft(B_, axis=1)
            cc = np.fft.irfft(fa.conj() * fb, n=n_th, axis=1)
            prof = cc.mean(axis=0) / n_th
            j = int(np.argmax(prof))
            sc = float(prof[j])
            scores.append(sc)
            if best is None or sc > best[0]:
                best = (sc, j, k, parity)
    sc, j_th, k_r, parity = best
    arr = np.array(scores)
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med))) + 1e-9
    prominence = (sc - med) / (1.4826 * mad)
    if prominence < 5.0:
        return None

    scale_corr = np.exp(k_r * dlnr)      # true = est * corr
    s_true = scale_est * scale_corr
    zoom = s_true / ref_scale

    # --- candidate poses, GAIA arbitrates --------------------------------
    # Deriving the rotation/parity algebra analytically is exactly the trap
    # the project's own history warns about - so build BOTH theta-sign
    # candidates and let the independent Gaia check pick the real one (a
    # wrong convention projects the stars into nonsense and simply fails).
    if det_xy is None or len(det_xy) < 6:
        return None
    try:
        from astropy.io import fits as _f2
        from astropy.wcs import WCS as _W2
        from scipy.spatial import cKDTree

        from .starid import gaia_field_catalog
        diag = (full_w ** 2 + full_h ** 2) ** 0.5
        gaia = gaia_field_catalog(obj_ra, obj_dec,
                                  min(s_true * diag * 0.7, 1.0),
                                  limit=1500, order_bright=True)
        if len(gaia) < 4:
            return None
        g_ra = np.array([g["ra"] for g in gaia])
        g_de = np.array([g["dec"] for g in gaia])
        tree = cKDTree(det_xy[:2000])
        tol = 6.0
        cd_ref = ref_wcs.pixel_scale_matrix
        crv = ref_wcs.pixel_to_world_values(r_nx, r_ny)
        F = (np.eye(2) if parity == 1
             else np.array([[1.0, 0.0], [0.0, -1.0]]))
        best_pose = None
        for th_sign in (1.0, -1.0):
            th = th_sign * thetas[j_th]
            c, sn = np.cos(th), np.sin(th)
            R = np.array([[c, -sn], [sn, c]])
            for order_ in (R @ F, F @ R):
                cd_q = cd_ref @ (zoom * order_)
                hdr = _f2.Header()
                hdr["CTYPE1"], hdr["CTYPE2"] = "RA---TAN", "DEC--TAN"
                hdr["CRVAL1"], hdr["CRVAL2"] = float(crv[0]), float(crv[1])
                hdr["CRPIX1"], hdr["CRPIX2"] = q_nx + 1.0, q_ny + 1.0
                hdr["CD1_1"], hdr["CD1_2"] = cd_q[0, 0], cd_q[0, 1]
                hdr["CD2_1"], hdr["CD2_2"] = cd_q[1, 0], cd_q[1, 1]
                buf = _io.BytesIO()
                _f2.PrimaryHDU(header=hdr).writeto(buf)
                wb = buf.getvalue()
                w2 = _W2(_f2.open(_io.BytesIO(wb))[0].header)
                gx, gy = w2.world_to_pixel_values(g_ra, g_de)
                ok = ((gx >= -10) & (gx < full_w + 10)
                      & (gy >= -10) & (gy < full_h + 10))
                n_in = int(ok.sum())
                if n_in < 4:
                    continue
                # the polar pose is COARSE (2 deg/bin = 35 px at the frame
                # edge): pair generously, REFIT the WCS from the pairs (the
                # rotation solver's own trick), then verify tightly
                d30, i30 = tree.query(np.column_stack([gx[ok], gy[ok]]),
                                      distance_upper_bound=30.0)
                pair_ok = np.isfinite(d30)
                if int(pair_ok.sum()) < 5:
                    continue
                gi = np.nonzero(ok)[0][pair_ok]
                det_i = i30[pair_ok].astype(int)
                # gnomonic (xi, eta) about the object; pixel -> sky LSQ
                d0r = np.radians(obj_dec)
                ra0r = np.radians(obj_ra)
                gr = np.radians(g_ra[gi])
                gd = np.radians(g_de[gi])
                cosc = (np.sin(d0r) * np.sin(gd)
                        + np.cos(d0r) * np.cos(gd) * np.cos(gr - ra0r))
                xi_p = np.degrees(np.cos(gd) * np.sin(gr - ra0r) / cosc)
                eta_p = np.degrees((np.cos(d0r) * np.sin(gd)
                                    - np.sin(d0r) * np.cos(gd)
                                    * np.cos(gr - ra0r)) / cosc)
                P = np.column_stack([det_xy[:2000][det_i, 0] - q_nx,
                                     det_xy[:2000][det_i, 1] - q_ny,
                                     np.ones(len(det_i))])
                X_, *_ = np.linalg.lstsq(P, xi_p, rcond=None)
                E_, *_ = np.linalg.lstsq(P, eta_p, rcond=None)
                hdr2 = _f2.Header()
                hdr2["CTYPE1"], hdr2["CTYPE2"] = "RA---TAN", "DEC--TAN"
                hdr2["CRVAL1"], hdr2["CRVAL2"] = obj_ra, obj_dec
                A2 = np.array([[X_[0], X_[1]], [E_[0], E_[1]]])
                b2 = -np.array([X_[2], E_[2]])
                try:
                    off = np.linalg.solve(A2, b2)
                except np.linalg.LinAlgError:
                    continue
                hdr2["CRPIX1"] = q_nx + off[0] + 1.0
                hdr2["CRPIX2"] = q_ny + off[1] + 1.0
                hdr2["CD1_1"], hdr2["CD1_2"] = X_[0], X_[1]
                hdr2["CD2_1"], hdr2["CD2_2"] = E_[0], E_[1]
                buf2 = _io.BytesIO()
                _f2.PrimaryHDU(header=hdr2).writeto(buf2)
                wb2 = buf2.getvalue()
                w3 = _W2(_f2.open(_io.BytesIO(wb2))[0].header)
                gx2, gy2 = w3.world_to_pixel_values(g_ra, g_de)
                ok2 = ((gx2 >= -10) & (gx2 < full_w + 10)
                       & (gy2 >= -10) & (gy2 < full_h + 10))
                if int(ok2.sum()) < 4:
                    continue
                d8, _ = tree.query(np.column_stack([gx2[ok2], gy2[ok2]]),
                                   distance_upper_bound=5.0)
                hits = int(np.isfinite(d8).sum())
                chance = (int(ok2.sum()) * min(len(det_xy), 2000)
                          * np.pi * 25.0 / (full_w * full_h))
                need = chance + max(4.0, 3.5 * np.sqrt(max(chance, 1.0)))
                if hits >= need and (best_pose is None
                                     or hits > best_pose[0]):
                    best_pose = (hits, wb2, float(np.degrees(th)))
        if best_pose is None:
            return None
        gaia_hits, wcs_fits, th_deg_out = best_pose
    except Exception:
        return None

    pixscale = s_true * 3600.0
    return {"ra": float(crv[0]), "dec": float(crv[1]),
            "radius_deg": round(s_true * diag / 2, 4),
            "pixscale_arcsec": round(pixscale, 3),
            "wcs_fits": wcs_fits, "annotations": [],
            "morph_prominence": round(prominence, 1),
            "morph_rotation_deg": round(th_deg_out, 1),
            "morph_parity": int(parity), "gaia_hits": gaia_hits}


def morphology_align_solve(rgb: np.ndarray, obj_ra: float, obj_dec: float,
                           scale_est: float, full_w: int,
                           full_h: int) -> dict | None:
    """Identity -> WCS via MORPHOLOGY, for close-ups where star quads die.

    A 6-arcmin extragalactic frame holds too few catalog stars for any quad
    solver (local engine, nova and the Gaia rotation sweep all fail), and a
    DSS cutout shares no point landmarks with an HST close-up. But the
    GALAXY ITSELF is the landmark: fetch the survey image of the identified
    object WITH its WCS (hips2fits, free) and find the rotation/parity that
    correlates the two images' structure. Self-verifying - a wrong identity
    or scale shows no correlation peak."""
    import io as _io

    import httpx
    from PIL import Image as _Img

    fov = scale_est * max(full_w, full_h) * 1.35
    try:
        r = httpx.get("https://alasky.cds.unistra.fr/hips-image-services/hips2fits",
                      params={"hips": "CDS/P/DSS2/red", "width": 300,
                              "height": 300, "fov": fov, "projection": "TAN",
                              "coordsys": "icrs", "ra": obj_ra, "dec": obj_dec,
                              "format": "fits"}, timeout=60)
        r.raise_for_status()
        from astropy.io import fits as _fits
        from astropy.wcs import WCS as _WCS
        hdu = _fits.open(_io.BytesIO(r.content))[0]
        ref = np.asarray(hdu.data, dtype=np.float32)
        ref_wcs = _WCS(hdu.header)
    except Exception:
        return None
    if ref.ndim != 2 or not np.isfinite(ref).any():
        return None
    ref = np.nan_to_num(ref)
    lo, hi = np.percentile(ref, (5, 99.5))
    ref = np.clip((ref - lo) / max(hi - lo, 1e-6), 0, 1)
    # HIGH-PASS both images before correlating: the shared bright-core-on-
    # dark-sky blob correlates at EVERY angle (floor 0.65, peak 0.74 -
    # prominence 2), while the bar/arm STRUCTURE is what actually encodes
    # orientation
    from scipy import ndimage as _ndi
    ref = ref - _ndi.gaussian_filter(ref, 8.0)

    # query -> gray thumb at the SAME sky scale as the reference
    ref_scale = fov / 300.0  # deg/px
    zoom = scale_est / ref_scale
    qw, qh = max(int(full_w * zoom), 40), max(int(full_h * zoom), 40)
    gray = (np.clip(rgb, 0, 1).mean(axis=2) * 255).astype("uint8")
    qim = _Img.fromarray(gray).resize((qw, qh), _Img.LANCZOS)

    # correlate on mean-removed overlap, center-aligned (the identified
    # object sits at both centers by construction)
    def _ncc_at(arr):
        ah, aw = arr.shape
        y0, x0 = (300 - ah) // 2, (300 - aw) // 2
        if y0 < 0 or x0 < 0:
            ys, xs = max(-y0, 0), max(-x0, 0)
            arr = arr[ys:ys + 300, xs:xs + 300]
            ah, aw = arr.shape
            y0, x0 = (300 - ah) // 2, (300 - aw) // 2
        sub = ref[y0:y0 + ah, x0:x0 + aw]
        a = arr - arr.mean()
        b = sub - sub.mean()
        den = float(np.sqrt((a * a).sum() * (b * b).sum()))
        return float((a * b).sum() / den) if den > 0 else 0.0

    scores = []
    for flip in (False, True):
        base = qim.transpose(_Img.FLIP_LEFT_RIGHT) if flip else qim
        for th in range(0, 360, 3):
            rot = base.rotate(th, expand=True, resample=_Img.BILINEAR)
            arr = np.asarray(rot, dtype=np.float32) / 255.0
            arr = arr - _ndi.gaussian_filter(arr, 8.0)
            scores.append((_ncc_at(arr), th, flip))
    scores.sort(reverse=True)
    best_ncc, best_th, best_flip = scores[0]
    others = [s[0] for s in scores[12:]]  # skip the peak's neighbours
    med = float(np.median(others))
    mad = float(np.median(np.abs(np.array(others) - med))) + 1e-6
    prominence = (best_ncc - med) / (1.4826 * mad)
    # acceptance: real correlation AND a clear peak over the angle sweep
    if best_ncc < 0.35 or prominence < 5.0:
        return None

    # build the query WCS: reference WCS composed with the fitted similarity
    # (rotation about both centers; parity = X flip before rotation)
    th_r = np.radians(best_th)
    c, sn = np.cos(th_r), np.sin(th_r)
    # query pixel -> reference pixel: scale by zoom, optional flip, rotate.
    # PIL rotates counter-clockwise about the image center with y DOWN in
    # pixel space, so express the map explicitly
    par = -1.0 if best_flip else 1.0
    M = zoom * np.array([[par * c, sn], [-par * sn, c]])
    try:
        from astropy.io import fits as _fits2
        cd_ref = ref_wcs.pixel_scale_matrix
        cd_q = cd_ref @ M
        crv = ref_wcs.pixel_to_world_values(149.5, 149.5)
        hdr = _fits2.Header()
        hdr["CTYPE1"], hdr["CTYPE2"] = "RA---TAN", "DEC--TAN"
        hdr["CRVAL1"], hdr["CRVAL2"] = float(crv[0]), float(crv[1])
        hdr["CRPIX1"], hdr["CRPIX2"] = full_w / 2.0, full_h / 2.0
        hdr["CD1_1"], hdr["CD1_2"] = cd_q[0, 0], cd_q[0, 1]
        hdr["CD2_1"], hdr["CD2_2"] = cd_q[1, 0], cd_q[1, 1]
        buf = _io.BytesIO()
        _fits2.PrimaryHDU(header=hdr).writeto(buf)
        pixscale = float(np.sqrt(abs(np.linalg.det(cd_q)))) * 3600.0
        diag = (full_w ** 2 + full_h ** 2) ** 0.5
        return {"ra": float(crv[0]), "dec": float(crv[1]),
                "radius_deg": round(pixscale / 3600.0 * diag / 2, 4),
                "pixscale_arcsec": round(pixscale, 3),
                "wcs_fits": buf.getvalue(), "annotations": [],
                "morph_ncc": round(best_ncc, 3),
                "morph_rotation_deg": best_th,
                "morph_prominence": round(prominence, 1),
                "morph_parity": ("flip" if best_flip else "normal")}
    except Exception:
        return None


def tile_solve(rgb: np.ndarray, full_w: int, full_h: int) -> dict | None:
    """Local blind identification against the uniform sky-tile library: the
    photo's quad codes vote for candidate 2-degree tiles, each candidate is
    verified by point-fit + pixel correlation, and a verified lock converts
    the tile's known WCS into a WCS for the photo. Covers ~0.5-6 degree
    fields with NO online queue - built after the free nova queue proved to
    be a lottery (one run queued 175 s, the next got no slot in 360 s)."""
    global _tile_idx
    idx_path = PROJECT_ROOT / "data" / "sky_tiles" / "tile_index.npz"
    if not idx_path.exists():
        return None
    if _tile_idx is None:
        from scipy.spatial import cKDTree
        z = np.load(idx_path)
        _tile_idx = (cKDTree(z["codes"]), z["refs"], z["tile_ids"],
                     z["ras"], z["decs"], z["fovs"])
    tree, refs, tile_ids, ras, decs, fovs = _tile_idx

    import sys as _sys
    from PIL import Image
    _sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from build_pattern_index import quad_codes

    im = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)).convert("L")
    st = max(im.size) / 256.0
    if st > 1:
        im = im.resize((max(int(im.width / st), 8), max(int(im.height / st), 8)),
                       Image.BILINEAR)
    q = np.asarray(im, dtype=np.float32) / 255.0
    qp, _ = _blobs(q, n=12)
    if len(qp) < 5:
        return None
    qc = quad_codes(qp)
    if not len(qc):
        return None
    votes: dict[int, int] = {}
    for hits in tree.query_ball_point(qc, r=0.02):
        if len(hits) > 400:
            continue  # non-discriminative code
        for h in set(int(refs[i]) for i in hits):
            votes[h] = votes.get(h, 0) + 1
    ranked = sorted(votes.items(), key=lambda kv: -kv[1])[:12]

    best = None
    for ref_i, v in ranked:
        if v < 2:
            continue
        tid = int(tile_ids[ref_i])
        tpath = PROJECT_ROOT / "data" / "sky_tiles" / "img" / f"t{tid}.jpg"
        if not tpath.exists():
            continue
        try:
            tg = np.asarray(Image.open(tpath).convert("L"),
                            dtype=np.float32) / 255.0
        except Exception:
            continue
        tp, _ = _blobs(tg, n=12)
        fit = _fit_pattern(qp, tp, smin=0.22, smax=4.6)
        if fit is None:
            continue
        A, t, n_in, rms = fit
        ratio = n_in / max(min(len(qp), len(tp)), 1)
        if not ((n_in >= 6 or (n_in >= 5 and ratio >= 0.6)) and rms <= 2.5):
            continue
        ncc = _warp_ncc(q, tg, A, t)
        # verified TRUE tile locks measure 0.90-1.00; a dense IR mosaic
        # chance-locked a random tile at 0.517 with 6 points - keep a wide
        # margin under the real ones (same reasoning as the pattern solver)
        # real tile locks measure 0.90-1.00; a cloud-blob false lock slipped
        # through at ~0.4, so the bar sits well above chance yet far below
        # every genuine lock observed
        if ncc < 0.65:
            continue
        score = n_in + ncc * 6.0 - rms + v * 0.2
        if best is None or score > best[0]:
            best = (score, ref_i, A, t, n_in, rms, ncc, v)
    if best is None:
        return None
    _, ref_i, A, t, n_in, rms, ncc, v = best
    try:
        from astropy.io import fits
        from astropy.wcs import WCS

        ref_scale = float(fovs[ref_i]) / 256.0
        wr = WCS(naxis=2)
        wr.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        wr.wcs.crval = [float(ras[ref_i]), float(decs[ref_i])]
        wr.wcs.crpix = [128.5, 128.5]
        wr.wcs.cd = np.array([[-ref_scale, 0.0], [0.0, -ref_scale]])
        scale_thumb = max(full_w, full_h) / 256.0 if max(full_w, full_h) > 256 else 1.0
        cx_t, cy_t = (full_w / scale_thumb) / 2.0, (full_h / scale_thumb) / 2.0
        rx, ry = A @ np.array([cx_t, cy_t]) + t
        sky = wr.pixel_to_world_values(rx, ry)
        cd_q = wr.wcs.cd @ (A / scale_thumb)
        hdr = fits.Header()
        hdr["CTYPE1"], hdr["CTYPE2"] = "RA---TAN", "DEC--TAN"
        hdr["CRVAL1"], hdr["CRVAL2"] = float(sky[0]), float(sky[1])
        hdr["CRPIX1"], hdr["CRPIX2"] = full_w / 2.0, full_h / 2.0
        hdr["CD1_1"], hdr["CD1_2"] = cd_q[0, 0], cd_q[0, 1]
        hdr["CD2_1"], hdr["CD2_2"] = cd_q[1, 0], cd_q[1, 1]
        import io as _io
        buf = _io.BytesIO()
        fits.PrimaryHDU(header=hdr).writeto(buf)
        pixscale = float(np.sqrt(abs(np.linalg.det(cd_q)))) * 3600.0
        return {
            "ra": float(sky[0]), "dec": float(sky[1]),
            "radius_deg": round(pixscale / 3600.0
                                * (full_w ** 2 + full_h ** 2) ** 0.5 / 2, 4),
            "pixscale_arcsec": round(pixscale, 3),
            "wcs_fits": buf.getvalue(),
            "annotations": [],
            "matched_tile": int(tile_ids[ref_i]),
            "pattern_inliers": n_in,
            "pattern_rms_px": round(rms, 2),
            "pattern_ncc": round(ncc, 3),
            "tile_votes": v,
        }
    except Exception:
        return None


def pattern_candidates(rgb: np.ndarray, top_n: int = 25) -> list[dict]:
    """GEOMETRIC recall: the photo's own quad codes vote for reference fields
    directly, immune to framing/palette differences that sink embedding
    similarity (the correct field ranked #7,834 of 13,962 by appearance)."""
    global _pattern_idx
    pi_path = INDEX_DIR / "pattern_index.npz"
    if not pi_path.exists():
        return []
    if _pattern_idx is None:
        from scipy.spatial import cKDTree
        z = np.load(pi_path, allow_pickle=True)
        _pattern_idx = (cKDTree(z["codes"]), z["refs"], list(z["names"]))
    tree, refs, names = _pattern_idx

    import sys
    from PIL import Image
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from build_pattern_index import quad_codes

    im = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)).convert("L")
    st = max(im.size) / 256.0
    if st > 1:
        im = im.resize((max(int(im.width / st), 8), max(int(im.height / st), 8)),
                       Image.BILINEAR)
    g = np.asarray(im, dtype=np.float32) / 255.0
    qp, _ = _blobs(g, n=8)
    if len(qp) < 4:
        return []
    qc = quad_codes(qp)
    if not len(qc):
        return []
    votes: dict[int, int] = {}
    for hits in tree.query_ball_point(qc, r=0.015):
        for h in set(int(refs[i]) for i in hits):
            votes[h] = votes.get(h, 0) + 1
    ranked = sorted(votes.items(), key=lambda kv: -kv[1])[:top_n]
    return [{"name": names[r], "votes": v} for r, v in ranked if v >= 2]


def pattern_solve(rgb: np.ndarray, candidates: list[dict],
                  full_w: int, full_h: int) -> dict | None:
    """The user's principle, engineered: 'one object's position pins all the
    rest'. Try to MATCH the photo's blob pattern onto each visually-similar
    candidate's reference cutout (whose sky coordinates we know exactly).
    A verified pattern lock yields a WCS for the photo without ASTAP - and a
    wrong candidate simply fails to lock (self-verifying, like plate solving).
    Positions are approximate (reference is 256 px), good enough to NAME
    galaxies and bright stars."""
    import csv as _csv

    from PIL import Image

    meta: dict[str, dict] = {}
    mp = INDEX_DIR / "meta.csv"
    if not mp.exists():
        return None
    with open(mp, encoding="utf-8") as f:
        for r in _csv.DictReader(f):
            meta[r["name"]] = r

    im = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)).convert("L")
    scale_thumb = max(im.size) / 256.0
    if scale_thumb > 1:
        im = im.resize((max(int(im.width / scale_thumb), 8),
                        max(int(im.height / scale_thumb), 8)), Image.BILINEAR)
    q = np.asarray(im, dtype=np.float32) / 255.0
    qp, _ = _blobs(q)
    if len(qp) < 5:
        return None

    # evaluate EVERY candidate and keep the best verified lock - accepting
    # the first passable one let a wrong field through
    best = None
    for cand in candidates:
        m = meta.get(cand["name"])
        ref_path = INDEX_DIR / "img" / f"{cand['name'].replace('/', '_')}.jpg"
        if not m or not ref_path.exists():
            continue
        try:
            ref = np.asarray(Image.open(ref_path).convert("L"),
                             dtype=np.float32) / 255.0
        except Exception:
            continue
        rp, _ = _blobs(ref)
        fit = _fit_pattern(qp, rp)
        if fit is None:
            continue
        A, t, n_in, rms = fit
        ratio = n_in / max(min(len(qp), len(rp)), 1)
        if not ((n_in >= 6 or (n_in >= 5 and ratio >= 0.6)) and rms <= 2.5):
            continue
        ncc = _warp_ncc(q, ref, A, t)
        # verified TRUE locks measure 0.90-1.00; a 904-star composite chance-
        # aligned 7 blobs onto a galaxy cutout at NCC 0.643 and 0.35 let it
        # through (whole frame misidentified as IC0272)
        if ncc < 0.72:
            continue  # points aligned by chance; the PIXELS disagree
        score = n_in + ncc * 6.0 - rms
        if best is None or score > best[0]:
            best = (score, cand, m, A, t, n_in, rms, ncc)

    if best is None:
        return None
    _, cand, m, A, t, n_in, rms, ncc = best
    # reference WCS: TAN, centered on the object, fov over 256 px,
    # north-up with RA increasing left (hips2fits convention)
    try:
        from astropy.io import fits
        from astropy.wcs import WCS

        ra0, dec0 = float(m["ra"]), float(m["dec"])
        ref_scale = float(m["fov_deg"]) / 256.0
        wr = WCS(naxis=2)
        wr.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        wr.wcs.crval = [ra0, dec0]
        wr.wcs.crpix = [128.5, 128.5]
        wr.wcs.cd = np.array([[-ref_scale, 0.0], [0.0, -ref_scale]])
        cx_t, cy_t = (full_w / scale_thumb) / 2.0, (full_h / scale_thumb) / 2.0
        rx, ry = A @ np.array([cx_t, cy_t]) + t
        sky = wr.pixel_to_world_values(rx, ry)
        # full-res query WCS in the same image-y convention as wr; the
        # star-naming layer tests both pixel conventions empirically
        # (use_flip), so only the composed geometry has to be right
        cd_q = wr.wcs.cd @ (A / scale_thumb)
        hdr = fits.Header()
        hdr["CTYPE1"], hdr["CTYPE2"] = "RA---TAN", "DEC--TAN"
        hdr["CRVAL1"], hdr["CRVAL2"] = float(sky[0]), float(sky[1])
        hdr["CRPIX1"], hdr["CRPIX2"] = full_w / 2.0, full_h / 2.0
        hdr["CD1_1"], hdr["CD1_2"] = cd_q[0, 0], cd_q[0, 1]
        hdr["CD2_1"], hdr["CD2_2"] = cd_q[1, 0], cd_q[1, 1]
        import io as _io
        buf = _io.BytesIO()
        fits.PrimaryHDU(header=hdr).writeto(buf)
        pixscale = float(np.sqrt(abs(np.linalg.det(cd_q)))) * 3600.0
        return {
            "ra": float(sky[0]), "dec": float(sky[1]),
            "radius_deg": round(pixscale / 3600.0
                                * (full_w ** 2 + full_h ** 2) ** 0.5 / 2, 4),
            "pixscale_arcsec": round(pixscale, 3),
            "wcs_fits": buf.getvalue(),
            "annotations": [],
            "matched_reference": cand["name"],
            "pattern_inliers": n_in,
            "pattern_rms_px": round(rms, 2),
            "pattern_ncc": round(ncc, 3),
        }
    except Exception:
        return None


def press_ncc(rgb: np.ndarray, obj_name: str) -> float:
    """Pixel-level verification of a borderline famous match: blob-align the
    query onto the object's best press reference and correlate the PIXELS.

    Embedding similarity alone cannot separate "same frame, recompressed"
    (user's M64 at 0.94) from "different object that merely looks alike"
    (AG Car scored 0.88 against several M31 refs). Alignment + NCC can:
    the same frame correlates strongly, a lookalike fails to even align.
    """
    idx = _load_index()
    if idx is None:
        return 0.0
    names, _, vecs, _ = idx
    q = embed(rgb)
    stem_pre = "PRESS_" + obj_name.replace(" ", "_").replace("*", "s") + "__"
    best_i, best_s = None, -1.0
    for i, n in enumerate(names):
        if n.startswith(stem_pre):
            s = float(vecs[i] @ q)
            if s > best_s:
                best_i, best_s = i, s
    if best_i is None:
        return 0.0
    ref_path = INDEX_DIR / "img" / f"{names[best_i]}.jpg"
    if not ref_path.exists():
        return 0.0
    try:
        from PIL import Image as _I
        qg = np.asarray(_I.fromarray(
            (np.clip(rgb, 0, 1) * 255).astype("uint8")).convert("L").resize(
                (256, 256)), dtype=np.float32) / 255.0
        rg = np.asarray(_I.open(ref_path).convert("L").resize((256, 256)),
                        dtype=np.float32) / 255.0
        qp, _ = _blobs(qg)
        rp, _ = _blobs(rg)
        if len(qp) < 4 or len(rp) < 4:
            # too few landmarks to align: fall back to direct correlation
            # (both are 256-square, same-frame pairs line up closely enough)
            a, b = qg - qg.mean(), rg - rg.mean()
            den = float(np.sqrt((a * a).sum() * (b * b).sum()))
            return float((a * b).sum() / den) if den > 0 else 0.0
        fit = _fit_pattern(qp, rp, smin=0.5, smax=2.0)
        if fit is None:
            # NOTE: a raw direct-correlation fallback here false-confirmed
            # lookalike tilted galaxies at 0.87 (blob-shape correlation) and
            # a high-passed one separated nothing - alignment failure means
            # the pixels cannot judge, so answer honestly
            return 0.0
        A, t, n_in, rms = fit
        return float(_warp_ncc(qg, rg, A, t))
    except Exception:
        return 0.0


_avm_cache: tuple[float, dict] | None = None

# AVM ReferencePixel y-origin: FITS convention counts from the BOTTOM-left,
# image pixels from the top-left. Calibrated empirically against a frame the
# local engine solved independently (see memory v4.6+); flip if that test
# says so.
_AVM_Y_FLIP = True


def _load_avm() -> dict:
    """Publisher AVM WCS per press reference (harvested from the archives'
    api/json Spatial.* fields); mtime-cached so the background harvest's
    growing file is picked up without a restart."""
    global _avm_cache
    import json as _json
    p = INDEX_DIR / "avm_wcs.json"
    if not p.exists():
        return {}
    mt = p.stat().st_mtime
    if _avm_cache and _avm_cache[0] == mt:
        return _avm_cache[1]
    try:
        d = _json.loads(p.read_text("utf-8"))
    except Exception:
        return {}
    _avm_cache = (mt, d)
    return d


def _query_prep(rgb: np.ndarray):
    """Query side of the alignment (grayscale, 640px downscale, blobs).
    Independent of the reference, so a candidate SCAN computes it once
    instead of once per attempt - that repeat cost was 80% of an 8-minute
    scan over 80 candidate references."""
    from PIL import Image as _I

    qg_full = np.asarray(_I.fromarray(
        (np.clip(rgb, 0, 1) * 255).astype("uint8")).convert("L"),
        dtype=np.float32) / 255.0
    qh, qw = qg_full.shape
    q_s = 640.0 / max(qw, qh)
    qg = np.asarray(_I.fromarray((qg_full * 255).astype("uint8")).resize(
        (max(int(qw * q_s), 8), max(int(qh * q_s), 8)), _I.LANCZOS),
        dtype=np.float32) / 255.0
    qp, _ = _blobs(qg, n=40)
    return qg, qp


def _avm_align(rgb: np.ndarray, ref_name: str, e: dict,
               full_w: int, full_h: int, q_cache=None) -> dict | None:
    """Align the query onto a press reference pixel-to-pixel and inherit the
    publisher's AVM WCS. Returns a solve dict or None."""
    import io as _io

    from astropy.io import fits as _fits
    from astropy.wcs import WCS as _WCS
    from PIL import Image as _I

    ref_path = INDEX_DIR / "img" / f"{ref_name}.jpg"
    if not ref_path.exists():
        return None
    try:
        rimg = _I.open(ref_path).convert("L")
        rw, rh = rimg.size
        dim_w, dim_h = e["dim"]
        s_full = dim_w / rw
        # the stored 'small' variant must be a uniform downscale of the
        # full-res frame the AVM describes - aspect mismatch means a crop
        if abs(dim_h / rh - s_full) > 0.02 * s_full:
            return None
        r_s = 640.0 / max(rw, rh)
        # geometry of the query is free from its shape; only the grayscale
        # downscale + blob extraction are worth caching across candidates
        qh, qw = rgb.shape[:2]
        q_s = 640.0 / max(qw, qh)
        qg, qp = q_cache if q_cache is not None else _query_prep(rgb)
        rg = np.asarray(rimg.resize(
            (max(int(rw * r_s), 8), max(int(rh * r_s), 8)), _I.LANCZOS),
            dtype=np.float32) / 255.0
        rp, _ = _blobs(rg, n=40)
        if len(qp) < 5 or len(rp) < 5:
            return None
        fit = _fit_pattern(qp, rp, smin=0.3, smax=3.5)
        if fit is None:
            return None
        A, t, n_in, rms = fit
        if n_in < 6:
            return None
        ncc = float(_warp_ncc(qg, rg, A, t))
        if ncc < 0.55:
            return None

        # reference full-res WCS from the publisher's AVM fields
        rot = np.radians(e["rot"])
        cd = np.array([
            [e["scale"][0] * np.cos(rot), -e["scale"][1] * np.sin(rot)],
            [e["scale"][0] * np.sin(rot), e["scale"][1] * np.cos(rot)]])
        hdr_r = _fits.Header()
        hdr_r["CTYPE1"], hdr_r["CTYPE2"] = "RA---TAN", "DEC--TAN"
        hdr_r["CRVAL1"], hdr_r["CRVAL2"] = e["ra"], e["dec"]
        hdr_r["CRPIX1"], hdr_r["CRPIX2"] = e["crpix"][0], e["crpix"][1]
        hdr_r["CD1_1"], hdr_r["CD1_2"] = cd[0, 0], cd[0, 1]
        hdr_r["CD2_1"], hdr_r["CD2_2"] = cd[1, 0], cd[1, 1]
        w_ref = _WCS(hdr_r)

        # control points: query corners+center -> ref-small -> ref-full ->
        # (FITS y convention) -> sky
        ctrl = np.array([[qw * fx, qh * fy]
                         for fx in (0.1, 0.5, 0.9) for fy in (0.1, 0.5, 0.9)],
                        dtype=float)
        ref_small = (ctrl * q_s) @ A.T + t
        ref_native = ref_small / r_s
        ref_full_x = ref_native[:, 0] * s_full
        ref_full_y = ref_native[:, 1] * s_full
        fits_x = ref_full_x + 1.0
        fits_y = (dim_h - ref_full_y) if _AVM_Y_FLIP else (ref_full_y + 1.0)
        world = w_ref.wcs_pix2world(np.column_stack([fits_x, fits_y]), 1)

        # fit the QUERY's own TAN WCS from the control correspondences
        ra0, dec0 = e["ra"], e["dec"]
        r0, d0 = np.radians(ra0), np.radians(dec0)
        rr, dd = np.radians(world[:, 0]), np.radians(world[:, 1])
        cosc = np.sin(d0) * np.sin(dd) + np.cos(d0) * np.cos(dd) * np.cos(rr - r0)
        xi = np.degrees(np.cos(dd) * np.sin(rr - r0) / cosc)
        eta = np.degrees((np.cos(d0) * np.sin(dd)
                          - np.sin(d0) * np.cos(dd) * np.cos(rr - r0)) / cosc)
        P = np.column_stack([ctrl[:, 0], ctrl[:, 1], np.ones(len(ctrl))])
        X, *_ = np.linalg.lstsq(P, xi, rcond=None)
        E, *_ = np.linalg.lstsq(P, eta, rcond=None)
        hdr = _fits.Header()
        hdr["CTYPE1"], hdr["CTYPE2"] = "RA---TAN", "DEC--TAN"
        hdr["CRVAL1"], hdr["CRVAL2"] = ra0, dec0
        A2 = np.array([[X[0], X[1]], [E[0], E[1]]])
        b2 = -np.array([X[2], E[2]])
        off = np.linalg.solve(A2, b2)
        hdr["CRPIX1"], hdr["CRPIX2"] = off[0], off[1]
        hdr["CD1_1"], hdr["CD1_2"] = X[0], X[1]
        hdr["CD2_1"], hdr["CD2_2"] = E[0], E[1]
        buf = _io.BytesIO()
        _fits.PrimaryHDU(header=hdr).writeto(buf)
        pixscale = float(np.sqrt(abs(np.linalg.det(A2)))) * 3600.0
        if not (0.005 <= pixscale <= 120.0):
            return None
        w_q = _WCS(hdr)
        cra, cdec = w_q.wcs_pix2world([[full_w / 2.0, full_h / 2.0]], 0)[0]
        diag = (full_w ** 2 + full_h ** 2) ** 0.5
        return {"ra": float(cra), "dec": float(cdec),
                "radius_deg": round(pixscale / 3600.0 * diag / 2, 4),
                "pixscale_arcsec": round(pixscale, 4),
                "wcs_fits": buf.getvalue(), "annotations": [],
                "matched_reference": ref_name,
                "avm_inliers": int(n_in), "avm_ncc": round(ncc, 3),
                "avm_rms_px": round(float(rms), 2)}
    except Exception:
        return None


def press_avm_solve(rgb: np.ndarray, obj_name: str,
                    full_w: int, full_h: int) -> dict | None:
    """PUBLISHER-WCS lock: when the query is (a variant of) a press image,
    align it pixel-to-pixel onto the matching reference and inherit the
    archive's own published WCS (AVM Spatial.* metadata).

    This is what names every object in a space-telescope close-up that no
    blind solver can touch: the frame's detections are nebular knots, not
    catalog stars, but the PUBLISHER knows exactly where the frame sits.
    Triple gate against wrong identity: embedding sim >= 0.90 on the same
    object, >= 6 alignment inliers, warp NCC >= 0.55 (measured thresholds -
    lookalike pairs fail alignment at <= 0.4)."""
    avm = _load_avm()
    if not avm:
        return None
    idx = _load_index()
    if idx is None:
        return None
    names, _, vecs, _ = idx
    q = embed(rgb)
    stem = "PRESS_" + obj_name.replace(" ", "_").replace("*", "s") + "__"
    # Identity confidence comes from the object's BEST reference (any AVM
    # quality); the alignment attempt then extends to lower-sim Full-AVM
    # refs. Rationale (measured on the Crab timelapse composite): the top
    # match hit 0.942 but carried Position-only AVM, while the Full-AVM
    # variants sat at 0.85-0.88 - below the old 0.90 gate - yet one of them
    # pixel-aligned cleanly. The ALIGNMENT is the decisive verification
    # (lookalike different objects measure <=0.4 NCC vs the 0.55 gate);
    # embedding sim only needs to establish WHICH object, once.
    cands = []
    best_sim = 0.0
    for i, n in enumerate(names):
        if n.startswith(stem):
            s = float(vecs[i] @ q)
            best_sim = max(best_sim, s)
            entry = avm.get(n)
            if entry and entry.get("quality") == "Full":
                cands.append((s, n, entry))
    if best_sim < 0.90:
        return None          # identity itself is not confident enough
    cands.sort(key=lambda c: -c[0])
    for sim, n, entry in cands[:5]:
        if sim < 0.80:
            break
        r = _avm_align(rgb, n, entry, full_w, full_h)
        if r:
            r["avm_similarity"] = round(sim, 3)
            return r
    return None


def object_support(rgb: np.ndarray, obj_name: str, floor: float = 0.85) -> int:
    """How many DIFFERENT reference images of this object the photo matches.
    A true famous image agrees with several variants of itself; a lookalike
    (AG Carinae's shell once matched a single M31 frame) agrees with one."""
    idx = _load_index()
    if idx is None:
        return 0
    names, commons, vecs, coords = idx
    q = embed(rgb)
    key = obj_name.replace(" ", "_")
    n = 0
    for i, raw in enumerate(names):
        if raw.startswith("PRESS_") and raw[6:].rsplit("__", 1)[0] == key:
            if float(vecs[i] @ q) >= floor:
                n += 1
        elif raw == obj_name.replace(" ", ""):
            if float(vecs[i] @ q) >= floor:
                n += 1
    return n


def _prescreen(qp: np.ndarray, ref_name: str, n: int = 16,
               min_inliers: int = 4) -> bool:
    """Fast reject for a candidate reference: fit only the brightest blobs.
    Cheap enough to run over a long candidate list before paying for the
    full alignment."""
    from PIL import Image as _I

    ref_path = INDEX_DIR / "img" / f"{ref_name}.jpg"
    if not ref_path.exists():
        return False
    try:
        rimg = _I.open(ref_path).convert("L")
        rw, rh = rimg.size
        r_s = 640.0 / max(rw, rh)
        rg = np.asarray(rimg.resize(
            (max(int(rw * r_s), 8), max(int(rh * r_s), 8)), _I.LANCZOS),
            dtype=np.float32) / 255.0
        rp, _ = _blobs(rg, n=n)
        if len(qp) < 5 or len(rp) < 5:
            return False
        fit = _fit_pattern(qp[:n], rp[:n], smin=0.3, smax=3.5)
        return bool(fit and fit[2] >= min_inliers)
    except Exception:
        return False


def align_candidates_solve(rgb: np.ndarray, full_w: int,
                           full_h: int) -> dict | None:
    """Identity-by-ALIGNMENT for frames whose appearance cannot pick the
    object. Tiny survey crops (rogue planets, brown dwarfs, finder charts)
    all look alike to the embedding - the WRONG object can out-score the
    right one (PSO J318's own field measured 0.867 while a CFBDSIR lookalike
    hit 0.945). Blob alignment is the separator: the true field locked at
    NCC 0.72 / 13 inliers while 8 wrong-object references ALL failed to
    align. So: try to pixel-align every visually plausible Full-AVM
    candidate and accept ONLY if exactly one distinct object aligns."""
    avm = _load_avm()
    if not avm:
        return None
    idx = _load_index()
    if idx is None:
        return None
    names, _, vecs, _ = idx
    q = embed(rgb)
    sims = vecs @ q
    by_obj: dict[str, list] = {}
    for i, n in enumerate(names):
        if not n.startswith("PRESS_"):
            continue
        s = float(sims[i])
        if s < 0.72:
            continue
        e = avm.get(n)
        if not e or e.get("quality") != "Full":
            continue
        disp = (n[6:].rsplit("__", 1)[0].replace("_", " ")
                .replace("M87s", "M87*")
                .replace("Sagittarius As", "Sagittarius A*"))
        by_obj.setdefault(disp, []).append((s, n, e))
    if not by_obj:
        return None
    ranked = sorted(by_obj.items(), key=lambda kv: -max(s for s, _, _ in kv[1]))
    hits: dict[str, dict] = {}
    tried = 0
    # deep the candidate list on purpose: for a tiny survey crop the RIGHT
    # object ranks low on appearance (PSO J318's own field sat 29th, behind
    # dozens of unrelated deep-field references). Alignment is what decides,
    # so it must actually reach the true candidate. Also try several refs
    # per object - PSO's best-scoring reference failed to align while its
    # third one locked cleanly.
    qg_c, qp_c = _query_prep(rgb)
    q_cache = (qg_c, qp_c)
    for disp, refs in ranked[:40]:
        refs.sort(key=lambda t: -t[0])
        for s, n, e in refs[:3]:
            if tried >= 80:
                break
            tried += 1
            # CHEAP PRESCREEN first: the full fit is O(points^4) and costs
            # ~6s per candidate, which made an 80-candidate scan take eight
            # minutes. On the brightest 16 blobs it costs 0.4s and separates
            # cleanly anyway - the true field scored 7 inliers where three
            # wrong-object references all scored 0.
            if not _prescreen(qp_c, n, min_inliers=4):
                continue
            r = _avm_align(rgb, n, e, full_w, full_h, q_cache=q_cache)
            # tighter than the press-avm gates (0.55/6): identity here rests
            # on the alignment ALONE, with no similarity gate above it
            if r and r["avm_ncc"] >= 0.60 and r["avm_inliers"] >= 10:
                r["avm_similarity"] = round(s, 3)
                hits[disp] = r
                break
        if len(hits) > 1:
            return None  # two different objects align: ambiguous, refuse
    if len(hits) != 1:
        return None
    disp, r = next(iter(hits.items()))
    r["identity"] = disp
    return r


def match(rgb: np.ndarray, top_k: int = 3) -> list[dict]:
    """Top visually-similar cataloged objects for this image. PRESS_* entries
    (multi-photo famous-object references) collapse onto their object name,
    so 40 Saturn press photos count as ONE candidate named Saturn."""
    idx = _load_index()
    if idx is None:
        return []
    names, commons, vecs, coords = idx
    q = embed(rgb)
    sims = vecs @ q
    order = np.argsort(-sims)[:top_k * 8]
    out = []
    seen: set[str] = set()
    for i in order:
        raw = names[i]
        press = raw.startswith("PRESS_")
        disp = (raw[6:].rsplit("__", 1)[0].replace("_", " ") if press else raw)
        # filenames can't hold '*': restore it for display (M87*, Sgr A*)
        disp = disp.replace("M87s", "M87*").replace("Sagittarius As",
                                                    "Sagittarius A*")
        if disp in seen:
            continue
        seen.add(disp)
        entry = {"name": disp, "common": (None if press else commons[i] or None),
                 "similarity": round(float(sims[i]), 3), "press": press,
                 "_vec_i": int(i)}
        # coords lookup must survive the display normalization: press entries
        # show "NGC 1300" while the catalog key is "NGC1300" - the space cost
        # us every position hint for press-identified objects
        for key in _coord_keys(raw, disp):
            if key in coords:
                entry["ra"], entry["dec"] = coords[key]
                break
        else:
            if disp in _FAMOUS_COORDS:
                entry["ra"], entry["dec"] = _FAMOUS_COORDS[disp]
        out.append(entry)
        if len(out) >= top_k * 3:
            break

    # SAME-OBJECT MERGE: one physical object often ranks several times under
    # alias names ("Antennae Galaxies" vs "NGC 4038" vs a mislabeled dupe) -
    # as separate rivals they nuke the identification gate's lead condition.
    # A later candidate is absorbed when it shares a normalized catalog key,
    # sits within 0.5 deg, or its best reference is pixel-near-identical.
    merged: list[dict] = []
    for e in out:
        dup = False
        ekeys = set(_coord_keys(e["name"], e["name"]))
        for m in merged:
            if ekeys & set(_coord_keys(m["name"], m["name"])):
                dup = True
            elif (e.get("ra") is not None and m.get("ra") is not None):
                import math as _math
                dra = abs(e["ra"] - m["ra"]) * _math.cos(_math.radians(m["dec"]))
                if _math.hypot(dra, e["dec"] - m["dec"]) < 0.5:
                    dup = True
            if not dup and float(vecs[e["_vec_i"]] @ vecs[m["_vec_i"]]) >= 0.985:
                dup = True
            if dup:
                # the absorbed alias may carry the coordinates the winner lacks
                if m.get("ra") is None and e.get("ra") is not None:
                    m["ra"], m["dec"] = e["ra"], e["dec"]
                # ...and may carry the FAMOUS name (HP Tau beats GN 04.32.8)
                if _name_prestige(e["name"]) > _name_prestige(m["name"]):
                    m["name"] = e["name"]
                break
        if not dup:
            merged.append(e)
        if len(merged) >= top_k:
            break
    for e in merged:
        e.pop("_vec_i", None)
    return merged
