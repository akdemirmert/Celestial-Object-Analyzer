"""Feature extraction: source detection and quantitative measurements.

Everything downstream (the hypothesis engine) reasons ONLY from numbers
produced here, so each measurement is kept explicit and unit-annotated.

Detection design (learned from real astrophotos):
- STARS are found on a *local-background-subtracted* map, so faint stars
  sitting on top of galaxy glow or sky gradients are still detected.
- The MAIN SOURCE is segmented twice: a bright core mask (5 sigma) to locate
  it and measure color, and an extended mask (1.5 sigma, the connected
  component containing the core) to measure true shape, edge and profile.
  Thresholding a galaxy at 5 sigma sees only its bright inner plateau and
  misreads it as a flat disk - the extended mask fixes exactly that.
"""
from __future__ import annotations

import base64
import io

import numpy as np
from scipy import ndimage


def _luminance(rgb: np.ndarray) -> np.ndarray:
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


def _sigma_clipped_stats(lum: np.ndarray, iters: int = 3) -> tuple[float, float]:
    """Global sky level + noise, robust against large bright objects."""
    vals = lum.ravel()
    for _ in range(iters):
        med = np.median(vals)
        sigma = np.median(np.abs(vals - med)) * 1.4826
        keep = np.abs(vals - med) < 3.5 * max(sigma, 1e-5)
        if keep.all():
            break
        vals = vals[keep]
    med = float(np.median(vals))
    sigma = float(max(np.median(np.abs(vals - med)) * 1.4826, 1e-4))
    return med, sigma


def _text_mask(lum: np.ndarray) -> np.ndarray | None:
    """Caption/label detector: press images carry titles like
    'Saturn | Webb Infrared Light' whose letters were being circled as stars.

    Text signature: MANY small sharp bright glyphs whose horizontal dilation
    merges into thin wide bars (lines of text). Star rows lack the tight
    vertical alignment and per-glyph sharpness."""
    h, w = lum.shape
    cand = lum > 0.45
    labels, n = ndimage.label(cand)
    if not n:
        return None
    areas = ndimage.sum_labels(np.ones_like(lum), labels, index=range(1, n + 1))
    objs = ndimage.find_objects(labels)
    glyph = np.zeros_like(cand)
    max_gh = max(int(0.05 * h), 8)
    for i, sl in enumerate(objs, start=1):
        gh = sl[0].stop - sl[0].start
        gw = sl[1].stop - sl[1].start
        if 3 <= gh <= max_gh and gw <= 2.5 * max_gh and areas[i - 1] <= gh * gw:
            glyph[sl][labels[sl] == i] = True
    if not glyph.any():
        return None
    # merge glyphs horizontally into candidate text lines
    gap = max(int(0.015 * w), 6)
    bar = ndimage.binary_dilation(glyph, structure=np.ones((1, gap), bool))
    bl, bn = ndimage.label(bar)
    if not bn:
        return None
    out = np.zeros_like(cand)
    for i, sl in enumerate(ndimage.find_objects(bl), start=1):
        bh = sl[0].stop - sl[0].start
        bw = sl[1].stop - sl[1].start
        if bh > max_gh * 1.6 or bw < 2.5 * bh or bw < 0.04 * w:
            continue
        # a real text line holds several separate glyphs
        n_glyphs = ndimage.label(glyph[sl])[1]
        if n_glyphs >= 4:
            out[sl] |= bl[sl] == i
    if not out.any():
        return None
    return ndimage.binary_dilation(out, iterations=3)


def _local_background(lum: np.ndarray, step: int = 4, box: int = 16) -> np.ndarray:
    """Smooth local background (median-filtered, downsampled grid) - models
    galaxy glow and sky gradients so stars can be detected on the residual."""
    small = lum[::step, ::step]
    bg_small = ndimage.median_filter(small, size=box)
    bg = ndimage.zoom(bg_small, (lum.shape[0] / bg_small.shape[0],
                                 lum.shape[1] / bg_small.shape[1]), order=1)
    bg = bg[: lum.shape[0], : lum.shape[1]]
    if bg.shape != lum.shape:  # zoom rounding
        pad_y = lum.shape[0] - bg.shape[0]
        pad_x = lum.shape[1] - bg.shape[1]
        bg = np.pad(bg, ((0, max(pad_y, 0)), (0, max(pad_x, 0))), mode="edge")
        bg = bg[: lum.shape[0], : lum.shape[1]]
    return ndimage.gaussian_filter(bg, 3)


def _detect_stars(lum: np.ndarray, rgb: np.ndarray,
                  exclude_mask: np.ndarray | None,
                  snr_min: float = 3.5, peak_floor: float = 0.012) -> list[dict]:
    """Point sources on the local-background residual, in two tiers:
    tier 'bright' (peak >= 7 sigma, reliable) and tier 'faint' (3.5-7 sigma,
    honest candidates - some may be noise). Each star carries its color."""
    local_bg = _local_background(lum)
    residual = lum - local_bg
    _, noise = _sigma_clipped_stats(residual)

    det = residual > snr_min * noise
    labels, n = ndimage.label(det)
    if not n:
        return []
    stars = []
    objs = ndimage.find_objects(labels)
    # (cy, cx, peak) of bright cores - their glare zone eats compact candidates
    glare_cores: list[tuple[float, float, float]] = []
    compact_cands: list[tuple[float, float, float, float, float, float, float]] = []

    def _emit(cy: float, cx: float, snr: float, flux: float,
              r_mean: float, b_mean: float) -> None:
        if exclude_mask is not None and exclude_mask[int(cy), int(cx)]:
            return
        stars.append({
            "x": round(float(cx), 1), "y": round(float(cy), 1),
            "peak_snr": round(float(snr), 1),
            "flux": round(float(flux), 4),
            "rb_ratio": round(float(r_mean) / max(float(b_mean), 1e-4), 2),
            "tier": "bright" if snr >= 7.0 else "faint",
        })

    for i, sl in enumerate(objs, start=1):
        box_h = sl[0].stop - sl[0].start
        box_w = sl[1].stop - sl[1].start
        box_max = max(box_h, box_w)
        comp = labels[sl] == i
        area = int(comp.sum())
        aspect = box_max / max(min(box_h, box_w), 1)
        peak = float(residual[sl][comp].max())
        snr = peak / noise

        # A compact footprint is a normal/faint star: measure it directly.
        compact = (box_max <= 25 and 2 <= area <= 120 and aspect <= 2.5)
        if compact:
            # sigma test alone fails on ultra-clean images (JPEG ripples become
            # "significant"); an absolute brightness floor kills those regardless
            if snr < snr_min or peak < peak_floor:
                continue
            if snr >= 7.0 and peak < 0.02:
                snr = 6.9  # bright tier additionally requires real brightness
            px = rgb[sl][comp]
            # deferred: glare suppression around saturated cores runs first
            compact_cands.append((sl[0].start + box_h / 2, sl[1].start + box_w / 2,
                                  snr, float(residual[sl][comp].sum()), peak,
                                  float(px[:, 0].mean()), float(px[:, 2].mean())))
            continue

        # Otherwise the footprint is oversized or ragged. A BRIGHT star lives
        # here: its halo and diffraction spikes swell the 3.5-sigma footprint
        # (measured 99-270 px boxes at SNR 40-100 on real Hubble frames) and
        # merge it with neighbours. Shape cuts throw exactly the most obvious
        # stars away, so identify them by what a star really is - a compact
        # local MAXIMUM - instead of by the shape of its footprint.
        if snr < 12.0 or peak < 0.08:
            continue  # big and faint = real extended structure, not a star
        for cy, cx, pk in _peaks_in_component(residual, noise, sl, comp):
            y0, y1 = max(int(cy) - 3, 0), int(cy) + 4
            x0, x1 = max(int(cx) - 3, 0), int(cx) + 4
            box = rgb[y0:y1, x0:x1]
            if box.size == 0:
                continue
            if pk >= 0.4:
                glare_cores.append((cy, cx, pk))
            _emit(cy, cx, pk / noise, float(residual[y0:y1, x0:x1].sum()),
                  box[..., 0].mean(), box[..., 2].mean())

    # Glare suppression: local-background subtraction shreds a saturated
    # star's mottled halo into 1-5 px fragments that pass the compact filters
    # (measured: peaks at 8-18% of the core, out to ~45 px on a SNR-40 star).
    # A real neighbouring star keeps >=20% of the core's peak (the true double
    # measured 100%), so kill only the fragments: fainter than 0.2x the core
    # inside a glare radius that scales with core brightness.
    keep = np.ones(len(compact_cands), dtype=bool)
    if compact_cands and glare_cores:
        cyx = np.array([(c[0], c[1]) for c in compact_cands])
        cpk = np.array([c[4] for c in compact_cands])
        for gy, gx, gp in glare_cores:
            r = 65.0 * min(gp, 1.0)
            near = (np.abs(cyx[:, 0] - gy) < r) & (np.abs(cyx[:, 1] - gx) < r)
            near &= cpk < 0.2 * gp
            if near.any():
                idx = np.nonzero(near)[0]
                d = np.hypot(cyx[idx, 0] - gy, cyx[idx, 1] - gx)
                keep[idx[d < r]] = False
    for ok, (cy, cx, snr, flux, peak, r_mean, b_mean) in zip(keep, compact_cands):
        if ok:
            _emit(cy, cx, snr, flux, r_mean, b_mean)

    stars.sort(key=lambda s: -s["flux"])
    return stars


def _peaks_in_component(residual: np.ndarray, noise: float, sl, comp: np.ndarray):
    """Local maxima inside one connected component -> one entry per bright star.
    Smoothing first suppresses single-pixel noise; the maximum filter then keeps
    only genuine peaks, so a star merged with its neighbours still yields one
    detection each instead of a single rejected blob.

    Diffraction spikes carry periodic intensity beads, and each bead is a local
    maximum too - a bright X-spiked star used to come back as 5+ "stars"
    (centre + one per spike arm). Real neighbouring stars are separated by a
    dark valley; a spike bead is connected to its core by a bright ridge. So a
    fainter peak is kept only if the brightness along the line to every kept
    brighter peak drops well below the fainter peak's own level (SExtractor-
    style contrast deblending)."""
    sub = np.where(comp, residual[sl], 0.0)
    smooth = ndimage.gaussian_filter(sub, 1.2)
    peak_map = ndimage.maximum_filter(smooth, size=9)
    hits = (smooth == peak_map) & (smooth > max(12.0 * noise, 0.06)) & comp
    ys, xs = np.nonzero(hits)
    vals = smooth[ys, xs]
    order = np.argsort(-vals)
    kept: list[tuple[float, float, float]] = []
    for idx in order:
        y, x, v = float(ys[idx]), float(xs[idx]), float(vals[idx])
        ridge = False
        for ky, kx, _ in kept:
            d = float(np.hypot(y - ky, x - kx))
            if d > 90.0:
                continue  # spikes never bead this far without a valley
            n = max(int(d), 2)
            line_y = np.clip(np.round(np.linspace(y, ky, n)).astype(int), 0, smooth.shape[0] - 1)
            line_x = np.clip(np.round(np.linspace(x, kx, n)).astype(int), 0, smooth.shape[1] - 1)
            valley = float(smooth[line_y, line_x].min())
            if valley > 0.5 * v:
                ridge = True  # no real valley: same star's spike/halo structure
                break
        if not ridge:
            kept.append((y, x, v))
    return [(sl[0].start + y, sl[1].start + x, float(sub[int(y), int(x)]))
            for y, x, _ in kept]


def _scene_signatures(lum: np.ndarray, bg: float, sigma: float,
                      rgb: np.ndarray | None = None) -> dict:
    """Whole-frame signatures for the remaining appearance classes: star
    trails (many concentric arcs), a total-eclipse corona ring, auroral
    green glow, and the Milky Way's diffuse band."""
    h, w = lum.shape
    out = {"star_trail_count": 0, "star_trail_angle_std": 0.0,
           "star_trail_concentric": False, "eclipse_ring": False,
           "green_glow_frac": 0.0, "green_curtain_ratio": 0.0,
           "green_border": False, "diffuse_band": False}
    if rgb is not None:
        try:
            # AURORA metrics on the WHOLE frame: the glow often reads as
            # "background" to the sigma-clipped source masks, so the
            # main-source box never sees it
            green = ((rgb[..., 1] > rgb[..., 0] * 1.12)
                     & (rgb[..., 1] > rgb[..., 2] * 1.05) & (lum > 0.08))
            gf = float(green.mean())
            out["green_glow_frac"] = round(gf, 4)
            if gf > 0.02:
                gmap = np.where(green, rgb[..., 1] - np.maximum(
                    rgb[..., 0], rgb[..., 2]), 0.0)
                colv = float(np.var(gmap.mean(axis=0)))
                rowv = float(np.var(gmap.mean(axis=1)))
                out["green_curtain_ratio"] = round(colv / max(rowv, 1e-8), 2)
                gd = ndimage.binary_dilation(green, iterations=2)
                out["green_border"] = bool(gd[0, :].any() or gd[-1, :].any()
                                           or gd[:, 0].any() or gd[:, -1].any())
            # MILKY WAY band: heavy smoothing melts stars into the diffuse
            # glow; a band-shaped surplus crossing most of the frame remains.
            # Threshold on the SMOOTHED map's own dynamic range - the raw
            # sigma is inflated by grain and killed the real band (measured:
            # MW p98-p50 = 0.14, flat control frames <= 0.03)
            sm = ndimage.gaussian_filter(lum, 30.0)
            med = float(np.median(sm))
            p98 = float(np.percentile(sm, 98))
            band = (sm > med + 0.5 * (p98 - med)) if (p98 - med) > 0.06 \
                else np.zeros_like(sm, dtype=bool)
            if 0.05 * h * w < band.sum() < 0.6 * h * w:
                lb, nb = ndimage.label(band)
                if nb:
                    bs = ndimage.sum(band, lb, range(1, nb + 1))
                    bb = lb == (int(np.argmax(bs)) + 1)
                    ys, xs = np.nonzero(bb)
                    span = np.hypot(ys.max() - ys.min(), xs.max() - xs.min())
                    mxx = np.var(xs); myy = np.var(ys)
                    mxy = float(np.mean((xs - xs.mean()) * (ys - ys.mean())))
                    tr_, det_ = mxx + myy, mxx * myy - mxy * mxy
                    disc = max(tr_ * tr_ / 4 - det_, 0) ** 0.5
                    el = ((tr_ / 2 + disc) / max(tr_ / 2 - disc, 1e-6)) ** 0.5
                    # measured margins: MW pano span 0.56/elong 3.0 vs the
                    # Pillars nebula 0.49/1.5 - band must be LONG and thin
                    if span > 0.52 * (h * h + w * w) ** 0.5 and el > 2.0:
                        out["diffuse_band"] = True
        except Exception:
            pass
    try:
        mask = lum > bg + max(3.5 * sigma, 0.04)
        lab, n = ndimage.label(mask)
        min_len = 0.03 * max(h, w)
        chords = []
        for i, sl in enumerate(ndimage.find_objects(lab), start=1):
            hh, ww = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
            if max(hh, ww) < min_len:
                continue
            comp = lab[sl] == i
            area = int(comp.sum())
            if area < 20:
                continue
            ys, xs = np.nonzero(comp)
            ys = ys + sl[0].start
            xs = xs + sl[1].start
            cx, cy = xs.mean(), ys.mean()
            mxx = ((xs - cx) ** 2).mean()
            myy = ((ys - cy) ** 2).mean()
            mxy = ((xs - cx) * (ys - cy)).mean()
            tr, det = mxx + myy, mxx * myy - mxy * mxy
            disc = max(tr * tr / 4 - det, 0.0) ** 0.5
            l1, l2 = tr / 2 + disc, max(tr / 2 - disc, 1e-6)
            if (l1 / l2) ** 0.5 < 3.0:
                continue
            ang = 0.5 * np.degrees(np.arctan2(2 * mxy, mxx - myy))
            chords.append((cx, cy, np.radians(ang)))
        out["star_trail_count"] = len(chords)
        if len(chords) >= 8:
            angs = np.array([c[2] for c in chords])
            # circular std over the 180-degree orientation space
            out["star_trail_angle_std"] = float(np.degrees(
                np.sqrt(-2 * np.log(max(abs(np.exp(2j * angs).mean()), 1e-9))) / 2))
            # concentric-arc test: each arc's perpendicular bisector should
            # pass near a common center (the celestial pole)
            A_rows, b_rows = [], []
            for cx, cy, a in chords:
                dx, dy = np.cos(a), np.sin(a)
                A_rows.append([dx, dy])
                b_rows.append(dx * cx + dy * cy)
            sol, res, *_ = np.linalg.lstsq(np.array(A_rows),
                                           np.array(b_rows), rcond=None)
            resid = np.abs(np.array(A_rows) @ sol - np.array(b_rows))
            out["star_trail_concentric"] = bool(
                np.median(resid) < 0.10 * max(h, w))
        # --- total-eclipse corona: bright ring with a dark round hole -----
        bright = lum > bg + max(6 * sigma, 0.15)
        bright = ndimage.binary_opening(bright, iterations=2)
        lab2, n2 = ndimage.label(bright)
        if n2:
            sizes = ndimage.sum(bright, lab2, range(1, n2 + 1))
            big = lab2 == (int(np.argmax(sizes)) + 1)
            filled = ndimage.binary_fill_holes(big)
            hole = filled & ~big
            if hole.sum() > 0.25 * big.sum() and big.sum() > 0.002 * h * w:
                hl, hn = ndimage.label(hole)
                hs = ndimage.sum(hole, hl, range(1, hn + 1))
                hbig = hl == (int(np.argmax(hs)) + 1)
                ys, xs = np.nonzero(hbig)
                d = ((ys.max() - ys.min()) + (xs.max() - xs.min())) / 2.0
                circ = hbig.sum() / (np.pi * (d / 2.0) ** 2 + 1e-6)
                if circ > 0.55 and float(lum[hbig].mean()) < 0.4 * float(lum[big].mean()):
                    out["eclipse_ring"] = True
    except Exception:
        pass
    return out


def _horizon_row(rgb: np.ndarray, lum: np.ndarray) -> int | None:
    """Skyline row of a NIGHTSCAPE (city lights / terrain under a starry
    sky), or None for pure-sky frames.

    A Moon-over-the-city photo picked the city glow band as its main source
    and called the frame a satellite trail. Signature of a real nightscape:
    the lower band is WARM and bright-textured (sodium lights, terrain) while
    everything above is dark sky - the transition is sharp and persistent."""
    h, w = lum.shape
    warm = ((lum > 0.22) & (rgb[..., 0] > rgb[..., 2] * 1.12)).mean(axis=1)
    csum = np.concatenate([[0.0], np.cumsum(warm)])
    win = max(h // 8, 30)

    def mean_window(y):
        # WINDOW below the candidate row, not mean-to-bottom: a dark
        # foreground (road, trees) under the lit city band diluted the
        # full-tail mean below threshold and the horizon went undetected
        y1 = min(y + win, h)
        return (csum[y1] - csum[y]) / max(y1 - y, 1)

    def mean_above(y):
        y0 = max(0, y - h // 8)
        return (csum[y] - csum[y0]) / max(y - y0, 1)

    for y in range(int(h * 0.35), int(h * 0.96)):
        if mean_window(y) > 0.12 and mean_above(y) < 0.03:
            # granularity check: city lights are POINT texture, a warm
            # nebula band is smooth - demand many local peaks somewhere in
            # the wide zone below (the lights often sit UNDER a smooth
            # atmospheric-glow band, so keep scanning instead of bailing)
            band = lum[y:min(y + max(int(h * 0.25), 60), h)]
            hp = band - ndimage.gaussian_filter(band, 4.0)
            if float((hp > 0.08).mean()) > 0.003:
                # a giant celestial disc is ALSO warm+bright+granular (a full
                # Moon's craters pass the point-texture test, and one got its
                # top sliced off here and called a Meteor) - but its warm
                # pixels form ONE round compact blob, while city glow is a
                # wide irregular band of many separate lights. Reject and
                # keep scanning: a real skyline may still sit lower.
                y1b = min(y + max(int(h * 0.25), 60), h)
                wm = ((lum[y:y1b] > 0.22)
                      & (rgb[y:y1b, :, 0] > rgb[y:y1b, :, 2] * 1.12))
                lab_w, n_w = ndimage.label(wm)
                if n_w:
                    sizes = ndimage.sum_labels(
                        np.ones_like(wm, dtype=np.float32), lab_w,
                        index=range(1, n_w + 1))
                    k = int(np.argmax(sizes))
                    frac = float(sizes[k]) / max(float(wm.sum()), 1.0)
                    sl = ndimage.find_objects(lab_w)[k]
                    bh = sl[0].stop - sl[0].start
                    bw = sl[1].stop - sl[1].start
                    if (frac >= 0.6 and bw <= 0.75 * w
                            and bw / max(bh, 1) <= 2.5):
                        continue
                return y
    return None


def _moon_disk(lum: np.ndarray, rgb: np.ndarray,
               sky_limit: int, strict: bool = False) -> dict | None:
    """A resolved Moon anywhere on the dark sky: small saturated near-white
    disk with a dark surround - independent of main-source selection (the
    huge obvious Moon lost to the city glow and went unmentioned)."""
    sky = lum[:sky_limit]
    bright = sky > 0.75
    bright = ndimage.binary_opening(bright, iterations=1)
    lab, n = ndimage.label(bright)
    best = None
    for i, sl in enumerate(ndimage.find_objects(lab), start=1):
        comp = lab[sl] == i
        area = int(comp.sum())
        if not 30 <= area <= 40000:
            continue
        hh, ww = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        d = (hh + ww) / 2.0
        # without a detected horizon a round street lamp can sneak in -
        # demand a clearly RESOLVED disk then
        if d < (14 if strict else 7) or max(hh, ww) / max(min(hh, ww), 1) > 1.5:
            continue
        circ = area / (np.pi * (d / 2.0) ** 2 + 1e-6)
        if circ < 0.6:
            continue
        cy = (sl[0].start + sl[0].stop) / 2.0
        cx = (sl[1].start + sl[1].stop) / 2.0
        # dark, empty surround (annulus 1.6-2.6 radii)
        yy0 = int(max(cy - d * 1.4, 0))
        yy1 = int(min(cy + d * 1.4, sky_limit))
        xx0 = int(max(cx - d * 1.4, 0))
        xx1 = int(min(cx + d * 1.4, lum.shape[1]))
        ring = lum[yy0:yy1, xx0:xx1].copy()
        ry, rx = np.mgrid[yy0:yy1, xx0:xx1]
        inner = ((ry - cy) ** 2 + (rx - cx) ** 2) < (d * 0.75) ** 2
        ring[inner] = np.nan
        if not np.isfinite(ring).any():
            continue
        # 0.40, not 0.30: the Moon carries its own glow halo (measured 0.32
        # on a hazy night) - lamps are excluded by the horizon split instead
        if float(np.nanmean(ring)) > 0.40:
            continue
        # near-white core (the Moon is colorless)
        core = rgb[int(max(cy - 2, 0)):int(cy + 3),
                   int(max(cx - 2, 0)):int(cx + 3)]
        if core.size:
            mx = core.reshape(-1, 3).mean(axis=0)
            sat = (mx.max() - mx.min()) / max(mx.max(), 1e-4)
            if sat > 0.35:
                continue
        score = area * circ
        if best is None or score > best["score"]:
            best = {"x": round(cx, 1), "y": round(cy, 1),
                    "diameter_px": round(d, 1), "score": score}
    return best


def _ring_moon_rescue(lum: np.ndarray, region: np.ndarray,
                      txt: np.ndarray | None) -> list[dict]:
    """Moons sitting ON the bright rings, recovered from inside the exclusion.

    The ring exclusion is correct (its sparkle texture is not stars) but it
    also swallowed every moon that touches the ring pixels. A grey OPENING
    keeps the smooth ring strip and erases point-like bumps, so the
    difference map holds the moons: the real moon measured area 61 / box
    fill 0.79 / aspect 1.6 while ring leftovers stay small or stripe-shaped."""
    selem = np.zeros((11, 11), bool)
    yy, xx = np.mgrid[:11, :11]
    selem[(yy - 5) ** 2 + (xx - 5) ** 2 <= 25] = True
    res = lum - ndimage.grey_opening(lum, footprint=selem)
    mask = (res > 0.25) & region
    if txt is not None:
        mask &= ~txt
    lab, n = ndimage.label(mask)
    out = []
    for i, sl in enumerate(ndimage.find_objects(lab), start=1):
        comp = lab[sl] == i
        area = int(comp.sum())
        h2, w2 = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        aspect = max(h2, w2) / max(min(h2, w2), 1)
        boxfill = area / (h2 * w2)
        peak = float(res[sl][comp].max())
        # two tiers: a big bright moon ON the ring, and a SMALL sharp one
        # hovering just off it (area 7, peak 0.41, box-fill 0.88 measured) -
        # ring leftovers fail on stripe shape or low fill either way
        big_ok = (40 <= area <= 500 and aspect <= 2.2
                  and boxfill >= 0.60 and peak >= 0.30)
        small_ok = (4 <= area < 40 and aspect <= 2.0
                    and boxfill >= 0.65 and peak >= 0.38)
        if not (big_ok or small_ok):
            continue
        cy = (sl[0].start + sl[0].stop) / 2.0
        cx = (sl[1].start + sl[1].stop) / 2.0
        out.append({"x": round(cx, 1), "y": round(cy, 1),
                    "peak_snr": round(30.0 + peak * 40.0, 1),
                    "flux": round(float(res[sl][comp].sum()), 4),
                    "rb_ratio": 1.0, "tier": "bright"})
    return out


def _merge_deep_stars(stars: list[dict], hi_rgb: np.ndarray,
                      exclude_mask: np.ndarray | None,
                      low_shape: tuple[int, int]) -> list[dict]:
    """Second detection pass at HIGHER resolution, merged into the base list.

    Downsampling to the analysis size averages a 2 px faint star into the
    noise floor (a 6780 px Hubble frame loses ~8x of its flux per star at
    2400 px) - the very stars users see 'missing'. Re-detecting on a larger
    version recovers them; existing detections stay authoritative (cell-based
    dedup, ~4 px in analysis space)."""
    h, w = low_shape
    hh, hw = hi_rgb.shape[:2]
    scale = hw / w
    if scale < 1.25:
        return stars
    lum_hi = _luminance(hi_rgb)
    ex = None
    if exclude_mask is not None:
        ex = ndimage.zoom(exclude_mask.astype(np.uint8),
                          (hh / h, hw / w), order=0).astype(bool)
        if ex.shape != (hh, hw):
            ex = np.pad(ex[:hh, :hw],
                        ((0, max(hh - ex.shape[0], 0)), (0, max(hw - ex.shape[1], 0))),
                        mode="edge")[:hh, :hw]
    # the deep pass digs deeper on purpose: higher resolution + slightly
    # looser thresholds recover the tiniest stars users still see missing
    deep = _detect_stars(lum_hi, hi_rgb, ex, snr_min=3.0, peak_floor=0.009)
    cell = 4.0
    occupied = {(int(s["x"] // cell), int(s["y"] // cell)) for s in stars}
    out = list(stars)
    inv2 = 1.0 / (scale * scale)  # flux scales with pixel area
    for s in deep:
        x, y = s["x"] / scale, s["y"] / scale
        cx_, cy_ = int(x // cell), int(y // cell)
        if any((cx_ + dx, cy_ + dy) in occupied
               for dx in (-1, 0, 1) for dy in (-1, 0, 1)):
            continue
        s = dict(s)
        s["x"], s["y"] = round(x, 1), round(y, 1)
        s["flux"] = round(s["flux"] * inv2, 4)
        s["deep"] = True
        out.append(s)
        occupied.add((cx_, cy_))
    out.sort(key=lambda s: -s["flux"])
    return out


def _ring_gap_signature(lum: np.ndarray, mask: np.ndarray) -> dict | None:
    """Ringed-planet test: brightness profile along the MAJOR axis.

    Saturn reads ring(peak) - gap(deep valley) - planet(WIDE bright plateau) -
    gap - ring: two symmetric deep valleys flanking a broad central plateau.
    Nothing else produces this: galaxies decline monotonically from the
    nucleus (valleys 0.14-0.34 measured, never <0.15 on both sides), the
    Hourglass nebula's centre is a NARROW star spike (plateau 0.04), N44's
    centre is dark (0.33). Saturn measured valleys 0.05/0.03, plateau 0.40,
    centre 0.94."""
    ys, xs = np.nonzero(mask)
    if len(ys) < 200:
        return None
    cy, cx = ys.mean(), xs.mean()
    elong, major, _minor, ang = _elongation(mask)
    # the ring PLANE's angle, not the blob's: rings are the brightest
    # structure in every Saturn image, so the principal axis of the top
    # brightness decile tracks them (the whole-mask moments axis measured
    # 66 deg on a frame whose rings sit at -8 deg)
    mref = float(np.percentile(lum[mask], 97))
    bmask = mask & (lum >= 0.7 * mref)
    if bmask.sum() >= 200:
        _, _, _, ang = _elongation(bmask)
        bys, bxs = np.nonzero(bmask)
        cy, cx = float(bys.mean()), float(bxs.mean())
    th = np.radians(ang)
    dx, dy = np.cos(th), np.sin(th)
    half = int(major / 2 * 1.15)
    if half < 20:
        return None
    ts = np.arange(-half, half + 1)
    # the centroid line can miss the planet on a tilted/offset framing (JWST
    # Saturn: centre sampled 0.106 - dark sky beside the disk). Sweep a few
    # parallel lines; the geometry only needs to show on ONE of them.
    nx, ny = -dy, dx  # unit normal to the major axis
    best = {"ringed": False, "plateau_w": 0.0, "center": 0.0}

    def _side(seg: np.ndarray):
        """Returns the ring-peak DISTANCE from the plateau edge, or None."""
        if len(seg) < 6:
            return None
        head = seg[:max(len(seg) // 2, 3)]
        vmin_i = int(np.argmin(head))
        vmin = float(head.min())
        tail = seg[vmin_i:]
        if not len(tail):
            return None
        peak = float(tail.max())
        # valley 0.40: ring-gap depth varies with processing (Hubble 0.03,
        # JWST clean 0.26, NASA captioned crop 0.36). The REAL discriminators
        # are the ring peak (>=0.40) and its contrast (diff >= 0.35): the
        # strongest galaxy side ever measured reached only diff 0.31
        if vmin < 0.40 and peak >= 0.40 and (peak - vmin) >= 0.35:
            return vmin_i + int(np.argmax(tail)), peak
        return None

    # angle sweep too: a tilted thin ring is an ELLIPSE - the straight
    # moments-axis line grazes it on one side and misses the other arm
    # (JWST Saturn's left arm measured peak 0.04 on the straight line)
    sweep = [(a_off, o_frac) for a_off in (0.0, -10.0, 10.0, -20.0, 20.0)
             for o_frac in (0.0, -0.15, 0.15, -0.3, 0.3)]
    hits = 0  # real rings show on MANY lines; chance symmetry shows on one
    for a_off, off_frac in sweep:
        th2 = th + np.radians(a_off)
        dx, dy = np.cos(th2), np.sin(th2)
        nx, ny = -dy, dx
        oy = cy + off_frac * (_minor / 2.0) * ny
        ox = cx + off_frac * (_minor / 2.0) * nx
        py = np.clip(np.round(oy + ts * dy).astype(int), 0, lum.shape[0] - 1)
        px = np.clip(np.round(ox + ts * dx).astype(int), 0, lum.shape[1] - 1)
        prof = ndimage.uniform_filter1d(lum[py, px], 5)
        mx = float(prof.max())
        if mx <= 0:
            continue
        p = prof / mx
        n = len(p)
        # the PLANET is the widest bright run near the middle - the global
        # maximum is usually a RING tip (rings outshine the disk on Saturn),
        # which is exactly the wrong anchor
        runs = []
        in_run = False
        start = 0
        for i2, v2 in enumerate(p > 0.5):
            if v2 and not in_run:
                in_run, start = True, i2
            elif not v2 and in_run:
                in_run = False
                runs.append((start, i2 - 1))
        if in_run:
            runs.append((start, n - 1))
        mids = [(a, b) for a, b in runs
                if 0.2 * n <= (a + b) / 2 <= 0.8 * n]
        if not mids:
            continue
        left, right = max(mids, key=lambda r2: r2[1] - r2[0])
        plateau_w = (right - left) / n
        mid = float(p[(left + right) // 2])
        sideL = _side(p[left::-1])
        sideR = _side(p[right:])
        # SYMMETRY is Saturn's fingerprint the fakes lack: both ring peaks
        # sit at near-equal distances from the planet (nebula lobes that
        # slipped through as "Saturn" on the stress bench are asymmetric)
        sym = (sideL is not None and sideR is not None
               and abs(sideL[0] - sideR[0]) <= 0.35 * max(sideL[0], sideR[0], 1))
        ringed = (1.25 <= elong <= 3.2 and 0.26 <= plateau_w <= 0.68
                  and mid >= 0.70 and sym)
        if ringed:
            hits += 1
            # the 25-line sweep is a multiple-comparisons trap (globulars
            # produced ONE chance-symmetric line on the stress bench).
            # Accept on 2+ independent lines, OR on a single line whose ring
            # peaks BOTH shine near maximum (every real Saturn measures
            # 0.98-1.00; chance star-pairs rarely do)
            if hits >= 2 or min(sideL[1], sideR[1]) >= 0.85:
                return {"ringed": True, "plateau_w": round(plateau_w, 3),
                        "center": round(mid, 3)}
        if plateau_w > best["plateau_w"]:
            best = {"ringed": False, "plateau_w": round(plateau_w, 3),
                    "center": round(mid, 3)}

    # FULL-ANGLE RESCUE: the bright-decile anchor assumes rings outshine the
    # planet - JWST's Neptune inverts that (saturated pearl disk, fainter
    # rings) and a glow flare tipped the axis to 59 deg while the rings sit
    # at ~0 deg; the +-20 sweep never crossed them. Sweep the whole circle,
    # but with STRICTER acceptance than pass 1 (60+ lines is a bigger
    # multiple-comparisons surface): both sides must clear diff >= 0.32
    # (measured: Neptune 0.34/0.36, strongest galaxy side ever 0.31), ring
    # peaks must be near-equal (0.44/0.48 vs asymmetric nebula lobes), the
    # plateau must stay bright and flat (mid >= 0.85), and 2+ lines must
    # agree.
    if not (1.15 <= elong <= 4.0):  # rings are never perfectly face-on
        return best

    # line length from the CORE, not the ext major axis: diffraction spikes
    # and glow arms inflate ext (major 534 px around a ~90 px planet pushed
    # plateau_w to 0.15 and every line failed the 0.26 gate on the tilted
    # Neptune crop). Rings live within ~1.3 core diameters, so +-1.8
    # diameters covers ring-gap-planet-gap-ring at the right scale - and a
    # shorter line stays ON the thin ring arms instead of straying off them.
    d_core = 0.0
    if bmask.sum() >= 200:
        bl_, bn_ = ndimage.label(bmask)
        if bn_:
            bareas_ = ndimage.sum_labels(np.ones_like(lum), bl_,
                                         index=range(1, bn_ + 1))
            d_core = 2.0 * float(np.sqrt(float(np.max(bareas_)) / np.pi))
    if d_core >= 25:
        ts2 = np.arange(-int(1.8 * d_core), int(1.8 * d_core) + 1)
        off_scale = d_core
    else:
        ts2 = ts
        off_scale = _minor / 2.0

    def _wide_line(ang_deg: float, off_frac: float):
        """One profile line of the wide sweep; returns (plateau_w, mid) on a
        full pass, None otherwise. Stricter than pass 1: both sides must
        clear diff >= 0.32 (Neptune 0.34/0.36, best galaxy side 0.31), ring
        peaks near-equal, plateau bright and flat."""
        th2 = np.radians(ang_deg)
        dx2, dy2 = np.cos(th2), np.sin(th2)
        nx2, ny2 = -dy2, dx2
        oy = cy + off_frac * off_scale * ny2
        ox = cx + off_frac * off_scale * nx2
        py = np.clip(np.round(oy + ts2 * dy2).astype(int), 0, lum.shape[0] - 1)
        px = np.clip(np.round(ox + ts2 * dx2).astype(int), 0, lum.shape[1] - 1)
        prof = ndimage.uniform_filter1d(lum[py, px], 5)
        mx = float(prof.max())
        if mx <= 0:
            return None
        p = prof / mx
        n = len(p)
        runs = []
        in_run = False
        start = 0
        for i2, v2 in enumerate(p > 0.5):
            if v2 and not in_run:
                in_run, start = True, i2
            elif not v2 and in_run:
                in_run = False
                runs.append((start, i2 - 1))
        if in_run:
            runs.append((start, n - 1))
        mids = [(a, b) for a, b in runs
                if 0.2 * n <= (a + b) / 2 <= 0.8 * n]
        if not mids:
            return None
        left, right = max(mids, key=lambda r2: r2[1] - r2[0])
        plateau_w = (right - left) / n
        mid = float(p[(left + right) // 2])
        if not (0.26 <= plateau_w <= 0.55 and mid >= 0.70):
            return None

        def _side2(seg: np.ndarray):
            # STRUCTURAL gap-then-ring walk. The old "min of the inner half,
            # then max after it" heuristic broke whenever the sky BEYOND the
            # ring (darker than the ring gap) entered the window: the
            # minimum landed out there and the peak search skipped the ring
            # entirely (measured: gap 0.10 -> ring 0.44 -> outer sky 0.09).
            # Contrast floor 0.24, not 0.32: glare fills the gap on the
            # tilted Neptune crop (gap 0.20-0.26 vs ring 0.46-0.53). The
            # compensating wall against galaxies is THINNESS: the >=0.40
            # ring run must stay under 0.7x the plateau - rings are thin,
            # spiral-arm bumps are broad.
            n3 = len(seg)
            if n3 < 6:
                return None
            i = 0
            while i < n3 and seg[i] >= 0.40:
                i += 1
            if i >= n3:
                return None          # never enters a gap
            j = i
            while j < n3 and seg[j] < 0.40:
                j += 1
            if j >= n3:
                return None          # gap never rises into a ring
            gap_min = float(seg[i:j].min())
            k2 = j
            while k2 < n3 and seg[k2] >= 0.40:
                k2 += 1
            if (k2 - j) > 0.7 * max(right - left, 8):
                return None          # broad bump = arm/lobe, not a ring
            ring = seg[j:k2]
            peak = float(ring.max())
            if (peak - gap_min) < 0.24:
                return None
            return j + int(np.argmax(ring)), peak

        sL = _side2(p[left::-1])
        sR = _side2(p[right:])
        if sL is None or sR is None:
            return None
        if abs(sL[0] - sR[0]) > 0.35 * max(sL[0], sR[0], 1):
            return None
        if abs(sL[1] - sR[1]) > 0.35 * max(sL[1], sR[1]):
            return None
        return plateau_w, mid

    # thin, shallow-tilt rings: only the near-exact ring-plane line crosses
    # both arms, so the coarse grid rarely lands 2 hits on its own. Real
    # geometry is CONTINUOUS in angle - a coarse hit is confirmed by a fine
    # re-sweep around it (+-6 deg, small parallel offsets); an isolated
    # chance-symmetric line (the globular trap on the stress bench) is not.
    for ang_abs in range(0, 180, 5):
        hit = _wide_line(float(ang_abs), 0.0)
        if hit is None:
            continue
        confirms = 0
        for d_ang in (-6.0, -4.0, -2.0, 2.0, 4.0, 6.0):
            for off in (0.0, -0.06, 0.06, -0.12, 0.12):
                if _wide_line(ang_abs + d_ang, off) is not None:
                    confirms += 1
                    break  # one confirming line per neighbour angle
        if confirms >= 2:
            return {"ringed": True, "plateau_w": round(hit[0], 3),
                    "center": round(hit[1], 3), "wide_sweep": True}
    return best


def _solidity(mask: np.ndarray) -> float:
    """Component area / convex-hull area. A solid body (planet, even a
    crescent) fills its hull (~0.9); a galaxy GROUP is a sparse constellation
    of blobs with sky inside the hull (~0.45)."""
    ys, xs = np.nonzero(mask)
    if len(ys) < 50:
        return 1.0
    step = max(len(ys) // 4000, 1)
    pts = np.column_stack([xs[::step], ys[::step]]).astype(np.float64)
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(pts)
        return float(mask.sum()) / max(float(hull.volume), 1.0)
    except Exception:
        return 1.0


def _split_blended(lum: np.ndarray, mask: np.ndarray, min_d: float,
                   depth: int = 0, bg_split: float = 0.0,
                   earthlike: bool = False) -> list[np.ndarray]:
    """Multi-threshold deblend of ONE bright component into its real objects.

    A compact galaxy group merges into a single 5-sigma blob (HCG 40: five
    galaxies came back as one d=1269 component). Raising the threshold inside
    the blob until it falls apart into >=2 pieces of 'obvious-object' size
    recovers the individual members - SExtractor's deblending idea applied at
    object scale."""
    vals = lum[mask]
    if len(vals) < 400 or depth >= 3:
        return [mask]
    # the solidity gate protects EARTH-LIKE bodies only: ocean "valleys"
    # between cloud lobes fool every brightness criterion. But a tightly
    # cropped galaxy group also fills its hull (HCG 40 crop: solidity 0.98!)
    # and must still split - its members separate on the valley test, which
    # gray/colorless solid bodies (Moon, Venus) also pass safely.
    if earthlike and _solidity(mask) >= 0.70:
        return [mask]
    for q in (55, 65, 75, 85, 92):
        t = float(np.percentile(vals, q))
        sl, sn = ndimage.label(mask & (lum > t))
        if sn < 2:
            continue
        ar = ndimage.sum_labels(np.ones_like(lum), sl, index=range(1, sn + 1))
        big = [i + 1 for i, a in enumerate(ar) if 2.0 * np.sqrt(a / np.pi) >= min_d]
        if len(big) > 10:
            # tens of "objects" = a percolated star-field web, not a discrete
            # group; splitting it would flood the source list with junk
            return [mask]
        if len(big) >= 2:
            # split acceptance: real separate objects have near-dark VALLEYS
            # between them (HCG 40 galaxies: sky at 3% of galaxy brightness);
            # bright parts of ONE object don't (Earth's cloud lobes stay
            # connected through ~40%-bright ocean - splitting them made Earth
            # a "galaxy group")
            pieces = [sl == b for b in big]
            cents, refs = [], []
            for m in pieces:
                ys, xs = np.nonzero(m)
                cents.append((float(ys.mean()), float(xs.mean())))
                refs.append(float(np.percentile(lum[m], 90)))
            ok = True
            for i in range(len(pieces)):
                for j in range(i + 1, len(pieces)):
                    n_s = max(int(np.hypot(cents[i][0] - cents[j][0],
                                           cents[i][1] - cents[j][1])), 2)
                    ly = np.clip(np.round(np.linspace(cents[i][0], cents[j][0], n_s)).astype(int),
                                 0, lum.shape[0] - 1)
                    lx = np.clip(np.round(np.linspace(cents[i][1], cents[j][1], n_s)).astype(int),
                                 0, lum.shape[1] - 1)
                    valley = float(lum[ly, lx].min())
                    # real separate objects dim to a small fraction of their
                    # brightness between each other (HCG 40 pairs: 2-29%)
                    if valley > 0.32 * min(refs[i], refs[j]):
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                continue  # this quantile cuts through one object - try higher
            out: list[np.ndarray] = []
            for m in pieces:  # a piece may itself hold two objects - recurse
                out.extend(_split_blended(lum, m, min_d, depth + 1, bg_split,
                                          earthlike))
            return out
    return [mask]


def _measure_source(lum: np.ndarray, rgb: np.ndarray, mask: np.ndarray,
                    bg: float, index: int) -> dict:
    """Per-source metrics for the multi-source list (galaxy vs star vs nebula)."""
    ys, xs = np.nonzero(mask)
    cy, cx = float(ys.mean()), float(xs.mean())
    area = float(mask.sum())
    equiv_d = 2.0 * np.sqrt(area / np.pi)
    elong, major, minor, angle = _elongation(mask)
    net = np.clip(lum[ys, xs] - bg, 0, None)
    total = float(net.sum())
    # half-light radius vs footprint radius: a star packs its light into a
    # tiny saturated core inside a big spike/halo footprint (ratio < ~0.35);
    # a galaxy spreads it (>= ~0.45)
    r = np.hypot(ys - cy, xs - cx)
    order = np.argsort(r)
    cum = np.cumsum(net[order])
    r_half = float(r[order][int(np.searchsorted(cum, total * 0.5))]) if total > 0 else 0.0
    half_ratio = r_half / max(equiv_d / 2.0, 1.0)
    px = rgb[ys, xs]
    unclipped = px[px.max(axis=1) < 0.98]
    cpx = unclipped if len(unclipped) > 20 else px
    mean_r, mean_g, mean_b = (float(cpx[:, i].mean()) for i in range(3))
    maxc, minc = max(mean_r, mean_g, mean_b), min(mean_r, mean_g, mean_b)
    # Earth's fingerprint: BLUE ocean pixels + WHITE cloud patches coexisting.
    # Mean color washes to near-neutral on a cloudy globe (Galileo Earth
    # measured rb=1.01), so measure the fractions directly.
    px_max = px.max(axis=1)
    px_sat = (px_max - px.min(axis=1)) / np.maximum(px_max, 1e-4)
    blue_fraction = float(np.mean((px[:, 2] > px[:, 0] * 1.15) & (px_sat > 0.15)
                                  & (px_max > 0.08)))
    white_fraction = float(np.mean(np.min(px, axis=1) > 0.62))
    return {
        "index": index,
        "center_x": round(cx, 1), "center_y": round(cy, 1),
        "equiv_diameter_px": round(equiv_d, 1),
        "major_axis_px": round(major, 1), "minor_axis_px": round(minor, 1),
        "angle_deg": round(angle, 1), "elongation": round(elong, 2),
        "half_light_ratio": round(half_ratio, 3),
        "peak": round(float(lum[ys, xs].max()), 3),
        "mean_rgb": [round(mean_r, 3), round(mean_g, 3), round(mean_b, 3)],
        "rb_ratio": round(mean_r / max(mean_b, 1e-4), 2),
        "saturation": round((maxc - minc) / max(maxc, 1e-4), 3),
        "blue_fraction": round(blue_fraction, 3),
        "white_fraction": round(white_fraction, 3),
    }


def _mask_overlay_png(mask: np.ndarray, color: tuple[int, int, int],
                      max_dim: int = 960) -> str:
    """Boolean mask -> base64 PNG OUTLINE tracing the source's TRUE shape.

    Outline only, no interior fill (user feedback), and anti-aliased: the
    old nearest-neighbour 480 px edge upscaled into visibly blocky pixel
    steps. Smoothing the mask and feathering the 0.5-contour renders a clean
    continuous line at any zoom."""
    from PIL import Image

    h, w = mask.shape
    scale = max(h, w) / max_dim
    m = (ndimage.zoom(mask.astype(np.float32), 1.0 / scale, order=1)
         if scale > 1 else mask.astype(np.float32))
    # smoothing scales with the object: a big smooth body (Saturn's ring
    # ellipse) gets its wobble ironed flat, small structures keep detail
    obj_d = 2.0 * np.sqrt(max(float((m > 0.5).sum()), 1.0) / np.pi)
    sm = ndimage.gaussian_filter(m, float(np.clip(obj_d / 90.0, 2.0, 7.0)))
    # feathered band around the 0.5 contour = a smooth ~3 px outline
    edge_alpha = np.clip(1.0 - np.abs(sm - 0.5) / 0.16, 0.0, 1.0) * 235.0
    rgba = np.zeros((*sm.shape, 4), dtype=np.uint8)
    rgba[..., 0], rgba[..., 1], rgba[..., 2] = color
    rgba[..., 3] = edge_alpha.astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _radial_concentration(lum: np.ndarray, bg: float, cy: float, cx: float,
                          r_max: float) -> float:
    """Fraction of background-subtracted flux inside r_max/3.
    Uniform disk ~0.11 (area ratio); galaxies/PSF cores >= ~0.22."""
    h, w = lum.shape
    yy, xx = np.ogrid[:h, :w]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    net = np.clip(lum - bg, 0, None)
    total = net[r <= r_max].sum()
    if total <= 0:
        return 0.0
    inner = net[r <= max(r_max / 3.0, 1.0)].sum()
    return float(inner / total)


def _mask_perimeter(mask: np.ndarray) -> float:
    return float(np.count_nonzero(mask & ~ndimage.binary_erosion(mask)))


def _elongation(mask: np.ndarray) -> tuple[float, float, float, float]:
    """(elongation, major_px, minor_px, angle_deg) from image moments."""
    ys, xs = np.nonzero(mask)
    if len(ys) < 4:
        return 1.0, float(len(ys)), float(len(ys)), 0.0
    y0, x0 = ys.mean(), xs.mean()
    cyy = np.mean((ys - y0) ** 2)
    cxx = np.mean((xs - x0) ** 2)
    cxy = np.mean((ys - y0) * (xs - x0))
    evals, evecs = np.linalg.eigh(np.array([[cxx, cxy], [cxy, cyy]]))
    evals = np.clip(evals, 1e-6, None)
    major = 4.0 * np.sqrt(evals[1])
    minor = 4.0 * np.sqrt(evals[0])
    angle = float(np.degrees(np.arctan2(evecs[1, 1], evecs[0, 1])))
    return float(np.sqrt(evals[1] / evals[0])), float(major), float(minor), angle


def _sky_plausibility(median_lum: float, dark_fraction: float,
                      vivid_fraction: float, pure_color_fraction: float) -> dict:
    """Decide whether the image plausibly shows the night sky.

    Real astrophotos sit on a mostly-dark sky, are largely low-saturation, and
    contain essentially no PURE-color pixels (optics/sensors always mix
    channels; measured 0.00-0.01% across the whole test set, vs 0.9% for a
    game logo with glowing red eyes). Rejection needs a decisive flag, so a
    bright Moon shot or a subtle colorful nebula still passes."""
    reasons = []
    not_dark = median_lum > 0.30 and dark_fraction < 0.35
    if not_dark:
        reasons.append(f"the frame is bright and not sitting on a dark sky "
                       f"(median brightness {median_lum:.2f}, only "
                       f"{dark_fraction * 100:.0f}% dark background)")
    if vivid_fraction > 0.25:
        reasons.append(f"{vivid_fraction * 100:.0f}% of the frame is vividly colored; real "
                       "celestial images are mostly grayscale with only subtle color")
    elif vivid_fraction > 0.12:
        reasons.append(f"an unusually large {vivid_fraction * 100:.0f}% of the frame is "
                       "vividly colored for an astronomical photo")
    # NOTE: pure-color fraction was tried as an artwork detector and RETRACTED:
    # Hubble narrowband nebulae measure 5%+ pure-color pixels (6x more than the
    # game logo that motivated it). Rule-based color gates cannot separate
    # narrowband astro imagery from digital art - that needs the ML classifier.
    ok = not (not_dark or vivid_fraction > 0.25
              or (vivid_fraction > 0.12 and median_lum > 0.20))
    return {"ok": ok, "reasons": reasons}


def extract_features(rgb: np.ndarray, hi_rgb: np.ndarray | None = None) -> dict:
    h, w = rgb.shape[:2]
    lum = _luminance(rgb)

    # ---- NIGHTSCAPE split: sky above, city/terrain below -----------------
    # A Moon-over-the-city photo chose the city glow band as its main source
    # ("satellite trail") and counted thousands of street lights as stars.
    # If a skyline exists, neutralize everything below it and analyze the SKY.
    horizon_y = _horizon_row(rgb, lum)
    moon = None
    if horizon_y is not None:
        sky_med = float(np.median(lum[:horizon_y]))
        lum = lum.copy()
        rgb = rgb.copy()
        lum[horizon_y:] = sky_med
        rgb[horizon_y:] = sky_med
        if hi_rgb is not None:
            hi_rgb = hi_rgb.copy()
            hy = int(horizon_y * hi_rgb.shape[0] / h)
            hi_rgb[hy:] = sky_med
    moon = _moon_disk(lum, rgb, horizon_y if horizon_y is not None else h,
                      strict=horizon_y is None)

    bg, sigma = _sigma_clipped_stats(lum)

    # ---- global measurements ---------------------------------------------------
    lap = ndimage.laplace(lum)
    blur_score = float(np.var(lap))
    clipped_fraction = float(np.mean(np.max(rgb, axis=2) >= 0.99))
    bright_fraction = float(np.mean(lum > bg + 5 * sigma))

    # ---- "is this even a night-sky photo?" plausibility metrics ------------------
    # Real astrophotos sit on a DARK sky and are mostly low-saturation (grayscale
    # stars, subtle red/blue nebulae). Screenshots, memes, daytime photos and
    # pixel art are bright and/or vividly colored across large areas.
    median_lum = float(np.median(lum))
    dark_fraction = float(np.mean(lum < 0.12))
    maxc = np.max(rgb, axis=2)
    minc = np.min(rgb, axis=2)
    sat = (maxc - minc) / np.maximum(maxc, 1e-4)
    # vivid = saturated AND bright enough to be a real color block (not dark noise)
    vivid_fraction = float(np.mean((sat > 0.5) & (maxc > 0.25)))
    # pure-color pixels (~#FF0000-style): the signature of digital art/logos.
    # Optics, sensors and the atmosphere always mix channels, so real
    # astrophotos measure 0.00-0.01% here even for colorful nebulae.
    pure_color_fraction = float(np.mean((sat > 0.9) & (maxc > 0.35)))
    plausibility = _sky_plausibility(median_lum, dark_fraction, vivid_fraction,
                                     pure_color_fraction)

    # frame-filling emission: a narrowband nebula close-up covers the frame
    # with bright, color-saturated, CONNECTED structure. No star field can -
    # points on dark sky measure <0.06 on both metrics, while a Hubble-
    # palette NGC 3576 frame measured 0.71 colorful-bright with one
    # component spanning 0.94 of the image.
    emission_colorful_fraction = float(np.mean((maxc > 0.25) & (sat > 0.25)))
    biggest_bright_component = 0.0
    if emission_colorful_fraction > 0.05:
        _lab_e, _n_e = ndimage.label(maxc > 0.25)
        if _n_e:
            biggest_bright_component = float(
                np.max(np.bincount(_lab_e.ravel())[1:]) / _lab_e.size)

    features: dict = {
        "emission_colorful_fraction": round(emission_colorful_fraction, 4),
        "biggest_bright_component": round(biggest_bright_component, 4),
        "nightscape_horizon_y": horizon_y,
        "moon_disk": moon,
        **_scene_signatures(lum, bg, sigma, rgb),
        "image_width": w,
        "image_height": h,
        "background_level": round(bg, 4),
        "noise_sigma": round(sigma, 5),
        "blur_score": round(blur_score, 6),
        "clipped_fraction": round(clipped_fraction, 4),
        "bright_fraction": round(bright_fraction, 4),
        "median_luminance": round(median_lum, 4),
        "dark_fraction": round(dark_fraction, 4),
        "vivid_fraction": round(vivid_fraction, 4),
        "looks_astronomical": plausibility["ok"],
        "plausibility_reasons": plausibility["reasons"],
        "star_count": 0,
        "faint_star_count": 0,
        "stars": [],
        "notable_sources": [],
        "main_source": None,
    }

    # ---- main source: bright mask locates it -------------------------------------
    # sigma floor at one 8-bit quantization step: heavy JPEG smoothing can
    # crush the measured noise to ~0.0001, degenerating the 5-sigma cut to
    # "any pixel above the median" (a phone frame's mask became 9,600 px of
    # near-black noise)
    bright = lum > bg + 5.0 * max(sigma, 1.0 / 255.0)
    bright_labels, n_bright = ndimage.label(bright)
    main_bright_mask = None
    if n_bright:
        idx = range(1, n_bright + 1)
        areas = ndimage.sum_labels(np.ones_like(lum), bright_labels, index=idx)
        # integrated excess flux, not raw area: on that same degenerate frame
        # the largest component was the noise patch while the actual subject
        # (Vega's defocus disc) held 27x its flux in a sixth of the area. The
        # subject of a photograph is where the LIGHT is, not where the most
        # pixels crawled over the threshold.
        excess = ndimage.sum_labels(lum - bg, bright_labels, index=idx)
        # eligibility stays area >= 12 (a sub-12px speck can win raw flux and
        # would leave the frame main-source-less); the CHOICE among eligible
        # components is by flux
        eligible = np.nonzero(np.asarray(areas) >= 12)[0]
        if len(eligible):
            main_label = int(eligible[np.argmax(np.asarray(excess)[eligible])]) + 1
            main_bright_mask = bright_labels == main_label

    # Fallback for OBJECT-DOMINATED frames: a globe filling most of the image
    # makes the clipped stats bimodal (Mars: bg=0.36, sigma=0.14) so the
    # 5-sigma mask comes back empty and the planet becomes invisible to the
    # classifier. Only then: take the sky level from the darkest 15% of pixels
    # (by count) and threshold halfway up to the object.
    # ... and for frames where the 5-sigma mask caught only a bright SLIVER
    # of the real object: a red SOHO Sun frame put bg at the glow level and
    # the "main source" became 0.5% of the disk, so no dominant-object logic
    # ever fired and the granulation drew thousands of circles. The candidate
    # replaces the sliver only if it is a SOLID body (solidity >= 0.7) - a
    # nebula's emission web must keep its knot-based measurement instead.
    _tiny_main = (main_bright_mask is not None
                  and float(main_bright_mask.sum()) < 0.02 * w * h)
    if (main_bright_mask is None or _tiny_main) and bg > 0.10:
        flat = np.sort(lum.ravel())
        k = max(int(len(flat) * 0.15), 200)
        dark_med = float(np.median(flat[:k]))
        if dark_med < 0.4 * bg:
            sigma2 = float(max(np.median(np.abs(flat[:k] - dark_med)) * 1.4826, 1e-3))
            bright2 = lum > dark_med + 0.25 * (float(np.median(lum)) - dark_med)
            bl2, nb2 = ndimage.label(bright2)
            if nb2:
                areas2 = ndimage.sum_labels(np.ones_like(lum), bl2,
                                            index=range(1, nb2 + 1))
                ml2 = int(np.argmax(areas2)) + 1
                cand = bl2 == ml2
                # 0.9: the Sun's disk measures 1.00, N44's nebula blob 0.795 -
                # only truly solid bodies may replace an existing measurement
                if areas2[ml2 - 1] >= 12 and \
                        (not _tiny_main or _solidity(cand) >= 0.9):
                    main_bright_mask = cand
                    bright_labels, n_bright = bl2, nb2
                    bg, sigma = dark_med, sigma2

    # ---- extended mask: grow the main source to its true extent --------------------
    ext_mask = None
    if main_bright_mask is not None:
        # On a near-black background sigma collapses, so a pure 1.5-sigma cut
        # swallows the JPEG halo around a bright object and the mask bleeds to
        # the frame edges (Jupiter measured a 1337 px "diameter" in a 1328 px
        # image). Floor the threshold at a small fraction of the source's own
        # brightness so the mask follows real light, not compression fog.
        src_ref = float(np.percentile(lum[main_bright_mask], 90))
        low_thresh = max(bg + 1.5 * sigma, bg + 0.04 * max(src_ref - bg, 0.0))
        low = lum > low_thresh
        low = ndimage.binary_opening(low, iterations=1)  # break noise bridges
        low_labels, _ = ndimage.label(low)
        ys, xs = np.nonzero(main_bright_mask)
        cy0, cx0 = int(ys.mean()), int(xs.mean())
        ext_label = low_labels[cy0, cx0]
        if ext_label > 0:
            ext_mask = low_labels == ext_label
            # a low threshold can percolate through noise across the whole frame;
            # fall back to the bright mask if the region is implausibly large
            if ext_mask.sum() > 0.65 * w * h:
                ext_mask = main_bright_mask
        else:
            ext_mask = main_bright_mask
        ext_mask = ndimage.binary_fill_holes(ext_mask)
        # PERCOLATION GUARD: on a noisy near-black phone frame the low
        # threshold sits inside the noise floor and the region grows THROUGH
        # noise bridges into a frame-spanning octopus - the "main source"
        # centroid then lands on empty sky far from the actual subject
        # (measured on a Vega phone shot: 257 px disc -> 10,029 px blob,
        # centroid moved 633 px). Growth must expand a source, not move it.
        bys, bxs = np.nonzero(main_bright_mask)
        eys, exs = np.nonzero(ext_mask)
        if len(eys) and len(bys):
            r_b = float(np.sqrt(len(bys) / np.pi))
            shift = float(np.hypot(eys.mean() - bys.mean(),
                                   exs.mean() - bxs.mean()))
            if len(eys) > 8 * len(bys) and shift > max(3.0 * r_b, 15.0):
                ext_mask = ndimage.binary_fill_holes(main_bright_mask)

    # caption text: letters/digits must never be detected as anything
    txt_mask = _text_mask(lum)

    def _stars_and_notables(exclude_mask) -> None:
        """Star detection + notable list + render check. Runs AFTER the main
        source is understood, so the exclusion zone can depend on its type."""
        if txt_mask is not None:
            exclude_mask = (txt_mask if exclude_mask is None
                            else (exclude_mask | txt_mask))
        # dominant close-up regime: on a frame-filling object the "sky" is
        # its dim glow, where JPEG blocks pass the normal floors (the Sun's
        # red corona grew hundreds of fake stars and tripped the star-cluster
        # rule). Demand REAL significance there.
        # COOL fuzzy giants are exempt: a resolved galaxy filling the frame
        # (NGC 1672 measured fill 0.86, rb 1.05, sat 0.04) carries REAL
        # interior sources; the strict regime exists for the Sun's hot glow
        # (rb 21-34, sat 0.95) and saturated nebula filaments
        _sunlike_glow = (rb_ratio >= 3.0 or saturation >= 0.5
                         or not is_extended_fuzzy)
        _dominant = (main_bright_mask is not None
                     and ext_mask is not None
                     and float(ext_mask.sum()) > 0.30 * w * h
                     and _sunlike_glow)
        stars = _detect_stars(lum, rgb, exclude_mask,
                              snr_min=6.0 if _dominant else 3.5,
                              peak_floor=0.045 if _dominant else 0.012)
        # SMALL originals get the inverse treatment: their faintest stars are
        # single pixels, and the compact-footprint test (area >= 2) rejects
        # those wholesale. A 2x upsample spreads each PSF over enough pixels
        # for the same quality filters to accept it (Jewel Box at 720 px:
        # 955 -> 1551 detections - exactly the visibly-unmarked stars).
        # (_hi, not hi_rgb: assigning the closure variable would shadow it)
        _hi = hi_rgb
        if _hi is None and max(w, h) < 1100 and not _dominant:
            from PIL import Image as _Img
            _up = _Img.fromarray((np.clip(rgb, 0, 1) * 255).astype("uint8")
                                 ).resize((w * 2, h * 2), _Img.LANCZOS)
            _hi = np.asarray(_up, dtype=np.float32) / 255.0
        # the deep pass only pays off when there is open sky to dig in - on a
        # frame-filling planet it just grinds over excluded surface texture
        if _hi is not None and (exclude_mask is None
                                or float(exclude_mask.mean()) < 0.5):
            stars = _merge_deep_stars(stars, _hi, exclude_mask, (h, w))
        # moons ON the rings live inside the exclusion zone - rescue the
        # compact round bumps the grey-opening difference isolates there
        if is_ringed_disk and qm is not None and main_bright_mask is not None:
            _moon_region = ndimage.binary_dilation(
                qm | main_bright_mask, iterations=6)
            for m_ in _ring_moon_rescue(lum, _moon_region, txt_mask):
                if all((m_["x"] - s2["x"]) ** 2 + (m_["y"] - s2["y"]) ** 2 > 100
                       for s2 in stars):
                    stars.append(m_)
        bright_stars = [s for s in stars if s["tier"] == "bright"]
        features["star_count"] = len(bright_stars)
        features["faint_star_count"] = len(stars) - len(bright_stars)
        # Keep ALL of them: a rich Hubble/Webb field holds 30k+ real stars and
        # every cap we tried (600, then 6000) silently hid most of the
        # detections from the viewer. The bound below is only a runaway guard.
        features["stars"] = stars[:500000]

        notable = []
        main_cx = main_cy = None
        if main_bright_mask is not None:
            mys, mxs = np.nonzero(main_bright_mask)
            main_cy, main_cx = float(mys.mean()), float(mxs.mean())
        for s in bright_stars:
            if main_cx is not None and np.hypot(s["x"] - main_cx, s["y"] - main_cy) < 30:
                continue
            notable.append(s)
            if len(notable) >= 5:
                break
        features["notable_sources"] = notable

        # render/cut-out detection: a composited black background is PERFECT
        # zeros with no faint stars; real night photos always carry noise+stars
        zero_fraction = float(np.mean(np.all(rgb == 0.0, axis=2)))
        features["zero_fraction"] = round(zero_fraction, 4)
        # EXEMPTION: a frame dominated by one resolved disk (the Moon filling
        # a phone shot) legitimately has a starless, denoised-to-zero sky -
        # the bright disk forces a short exposure that hides every star, and
        # phone night modes crush the empty background to perfect black
        _disk_dominant = (is_disk_like and ext_mask is not None
                          and float(ext_mask.sum()) >= 0.03 * w * h)
        if (features["looks_astronomical"] and zero_fraction > 0.6
                and not _disk_dominant
                and features["star_count"] < 10 and features["faint_star_count"] < 20):
            features["looks_astronomical"] = False
            reasons = list(features.get("plausibility_reasons") or [])
            reasons.append(
                f"the background is perfectly empty ({zero_fraction * 100:.0f}% pure-zero "
                "pixels) yet contains almost no stars - real night-sky photos always carry "
                "sensor noise and faint stars; this looks like a render or cut-out placed "
                "on a black background")
            features["plausibility_reasons"] = reasons

    if main_bright_mask is None:
        # the star closure's dominant-regime test reads rb_ratio/saturation/
        # is_extended_fuzzy - on a source-less frame they were never assigned
        # and the NameError killed empty_noise/mars_viking (hidden for two
        # days behind an earlier suite crash). Neutral defaults: no main
        # source means no dominant-close-up regime.
        rb_ratio, saturation, is_extended_fuzzy = 1.0, 0.0, False
        is_ringed_disk, qm, is_disk_like = False, None, False
        _stars_and_notables(None)
        return features

    ys, xs = np.nonzero(ext_mask)
    cy, cx = float(ys.mean()), float(xs.mean())
    area = float(ext_mask.sum())
    equiv_diameter = float(2.0 * np.sqrt(area / np.pi))
    bright_area = float(main_bright_mask.sum())
    bright_diameter = float(2.0 * np.sqrt(bright_area / np.pi))

    perimeter = _mask_perimeter(ext_mask)
    circularity = float(4 * np.pi * area / (perimeter ** 2)) if perimeter > 0 else 0.0
    circularity = min(circularity, 1.2)
    elong, major_px, minor_px, angle_deg = _elongation(ext_mask)

    # disk-ness: measure shape at QUARTER-MAX of the source's own brightness.
    # Sigma-based masks are uselessly low on near-black backgrounds (they wrap
    # the JPEG fog around a planet); a threshold tied to the source brightness
    # follows the real luminous edge instead. A planet's quarter-max contour is
    # its crisp disk; lamp-lit ground or a galaxy's inner oval stay irregular
    # or elongated.
    ref = float(np.percentile(lum[ext_mask], 90))
    qm = (lum > max(ref * 0.25, bg + 5.0 * sigma)) & ext_mask
    qm = ndimage.binary_fill_holes(ndimage.binary_closing(qm, iterations=2))
    qm_labels, qn = ndimage.label(qm)
    if qn:
        qm_areas = ndimage.sum_labels(np.ones_like(lum), qm_labels,
                                      index=range(1, qn + 1))
        qm = qm_labels == (int(np.argmax(qm_areas)) + 1)
    qm_area = float(qm.sum())
    b_perim = _mask_perimeter(qm)
    bright_circularity = (min(float(4 * np.pi * qm_area / (b_perim ** 2)), 1.2)
                          if b_perim > 0 and qm_area >= 12 else 0.0)
    bright_elong, _, _, _ = _elongation(qm)

    # banding: a gas giant's cloud decks vary strongly ACROSS latitude but stay
    # smooth ALONG each band. Colour cannot see this (press images are often
    # false-colour/UV), but the geometry survives any processing.
    band_contrast = 0.0
    interior_rows = ndimage.binary_erosion(qm, iterations=4) if qm.any() else qm
    if interior_rows.sum() > 200:
        rows = []
        ys_i, xs_i = np.nonzero(interior_rows)
        y_lo, y_hi = int(ys_i.min()), int(ys_i.max())
        for yy_ in range(y_lo, y_hi + 1):
            m = interior_rows[yy_]
            if m.sum() >= 12:
                rows.append(float(lum[yy_][m].mean()))
        if len(rows) >= 20:
            prof = np.array(rows)
            smooth_prof = ndimage.uniform_filter1d(prof, 5)
            # variation between latitudes, normalised by the disk's brightness
            band_contrast = float(smooth_prof.std() / max(smooth_prof.mean(), 1e-4))

    # terrestrial signal: how much of the image border the "object" covers.
    # Deep-sky objects are framed (0% across every real test image); lamp-lit
    # ground/walls in night street photos bleed off-frame (25-44% measured).
    border = np.concatenate([ext_mask[0, :], ext_mask[-1, :],
                             ext_mask[:, 0], ext_mask[:, -1]])
    border_contact = float(border.mean())

    # color: bright interior, unclipped pixels (the object's own light)
    src_px = rgb[main_bright_mask]
    unclipped = src_px[np.max(src_px, axis=1) < 0.98]
    color_px = unclipped if len(unclipped) > 20 else src_px
    mean_r, mean_g, mean_b = (float(np.mean(color_px[:, i])) for i in range(3))
    rb_ratio = mean_r / max(mean_b, 1e-4)
    maxc, minc = max(mean_r, mean_g, mean_b), min(mean_r, mean_g, mean_b)
    saturation = (maxc - minc) / max(maxc, 1e-4)

    # limb sharpness on the TRUE outer edge (extended mask boundary)
    grad = np.hypot(*np.gradient(lum))
    boundary = ext_mask & ~ndimage.binary_erosion(ext_mask, iterations=2)
    interior_mean = float(lum[ext_mask].mean())
    limb_sharpness = float(grad[boundary].mean() / max(interior_mean, 1e-4)) if boundary.any() else 0.0

    interior = ndimage.binary_erosion(ext_mask, iterations=3)
    texture = float(lum[interior].std() / max(interior_mean, 1e-4)) if interior.any() else 0.0

    concentration = _radial_concentration(lum, bg, cy, cx, max(major_px / 2, 3.0))

    def _centroid(channel):
        vals = np.clip(rgb[..., channel] - bg, 0, None) * main_bright_mask
        s = vals.sum()
        if s <= 0:
            return cy, cx
        yy = float((vals.sum(axis=1) * np.arange(h)).sum() / s)
        xx = float((vals.sum(axis=0) * np.arange(w)).sum() / s)
        return yy, xx

    ry, rx = _centroid(0)
    by, bx = _centroid(2)
    chroma_shift_px = float(np.hypot(ry - by, rx - bx))
    src_clipped = float(np.mean(np.max(src_px, axis=1) >= 0.99))

    def _fit_limb_circle(mask):
        """LSQ circle (Kasa) on the mask's outer boundary, outliers rejected.

        A planet's lit limb is a circular ARC even when the night side breaks
        every whole-mask shape test (crescent / terminator frames). Returns
        the fitted disk when the arc is real: enough boundary points on the
        circle and enough angular coverage around its center.
        """
        # OUTER contour only: cloud texture riddles the mask with holes whose
        # edges drown the limb arc (66k boundary points, fit collapsed)
        lab, nlab = ndimage.label(mask)
        if nlab < 1:
            return None
        sizes = ndimage.sum(mask, lab, range(1, nlab + 1))
        solid = ndimage.binary_fill_holes(lab == (int(np.argmax(sizes)) + 1))
        b = solid & ~ndimage.binary_erosion(solid, iterations=2)
        py, px = np.nonzero(b)
        if len(py) < 60:
            return None
        pts = np.column_stack([px, py]).astype(np.float64)
        keep = np.ones(len(pts), dtype=bool)
        fit = None
        for _ in range(3):
            P = pts[keep]
            if len(P) < 40:
                return None
            A = np.column_stack([P[:, 0], P[:, 1], np.ones(len(P))])
            bb = (P ** 2).sum(axis=1)
            try:
                sol, *_ = np.linalg.lstsq(A, bb, rcond=None)
            except np.linalg.LinAlgError:
                return None
            fcx, fcy = sol[0] / 2, sol[1] / 2
            r2 = sol[2] + fcx ** 2 + fcy ** 2
            if r2 <= 0:
                return None
            fr = float(np.sqrt(r2))
            resid = np.abs(np.hypot(pts[:, 0] - fcx, pts[:, 1] - fcy) - fr)
            keep = resid < 0.06 * fr
            fit = (float(fcx), float(fcy), fr)
        fcx, fcy, fr = fit
        inlier = float(keep.mean())
        ang = np.degrees(np.arctan2(pts[keep][:, 1] - fcy,
                                    pts[keep][:, 0] - fcx))
        cover = float(len(np.unique((ang // 5).astype(int))) / 72.0)
        if (inlier < 0.45 or cover < 0.35 or fr < 25
                or fr > 1.5 * max(w, h)):
            return None
        return {"cx": fcx, "cy": fcy, "r": fr,
                "inlier": round(inlier, 3), "cover": round(cover, 3)}

    is_streak = bool(elong > 4.0 and major_px > 0.08 * max(w, h))
    # streak PROFILE: satellites are uniform, meteors taper/flare, aircraft
    # strobe - the fixed 55/30/25 scores called every meteor a satellite
    streak_taper, streak_dashes = None, None
    if is_streak:
        th_ = np.radians(angle_deg)
        n_s = 60
        ts = np.linspace(-0.48, 0.48, n_s) * major_px
        sx_ = np.clip(cx + ts * np.cos(th_), 0, w - 1)
        sy_ = np.clip(cy + ts * np.sin(th_), 0, h - 1)
        prof = ndimage.map_coordinates(lum, [sy_, sx_], order=1)
        pmax = float(prof.max()) if prof.size else 0.0
        if pmax > 0:
            ends = float(np.mean(np.concatenate([prof[:9], prof[-9:]])))
            streak_taper = round(ends / pmax, 3)
            below = prof < 0.3 * pmax
            streak_dashes = int(np.diff(below.astype(int)).clip(min=0).sum())
    # curtain ratio (aurora): vertical rays vary ACROSS columns but stay
    # smooth along each column
    curtain_ratio = None
    try:
        bb = np.nonzero(ext_mask)
        y0b, y1b = bb[0].min(), bb[0].max() + 1
        x0b, x1b = bb[1].min(), bb[1].max() + 1
        sub = lum[y0b:y1b, x0b:x1b]
        if sub.shape[0] > 20 and sub.shape[1] > 20:
            colv = float(np.var(sub.mean(axis=0)))
            rowv = float(np.var(sub.mean(axis=1)))
            curtain_ratio = round(colv / max(rowv, 1e-6), 2)
    except Exception:
        pass
    is_point_like = bool(bright_diameter < 20 and not is_streak)
    # Earth-signature blue fraction of the extended region: cloud patches
    # shred the quarter-max mask (qcirc 0.15-0.45 on real Earth photos), so a
    # blue-planet disk must be recognizable from its OUTER limb instead
    epx = rgb[ext_mask]
    emax = epx.max(axis=1)
    esat = (emax - epx.min(axis=1)) / np.maximum(emax, 1e-4)
    ext_blue = float(np.mean((epx[:, 2] > epx[:, 0] * 1.15)
                             & (esat > 0.15) & (emax > 0.08)))
    # concentration is the physical discriminator: solid disks have flat
    # profiles (<~0.22) at any blur level; galaxies are always centrally
    # condensed (>=~0.22). Additionally, a crisply CIRCULAR bright core is a
    # disk even with limb darkening pushing concentration a bit higher
    # (Hubble's Jupiter measured conc 0.235 with a perfect circle).
    is_disk_like = bool(not is_point_like and equiv_diameter >= 20
                        and ((circularity > 0.7 and elong < 1.6
                              and concentration < 0.22)
                             or (bright_diameter >= 20
                                 and bright_circularity > 0.8
                                 and bright_elong < 1.4
                                 and concentration < 0.34)
                             # blue planet: round outer limb + ocean-blue
                             # pixels (galaxies measure circularity < 0.3
                             # here, so 0.55 splits them cleanly)
                             or (equiv_diameter >= 60 and circularity > 0.55
                                 and elong < 1.6 and ext_blue >= 0.12)))
    # crescent/terminator rescue: a night-side bite breaks the shape tests
    # above, but the LIT limb still traces a circular arc. Ocean-blue floor
    # at 0.28 keeps round face-on galaxies out (their record is 0.217).
    disk_fit = None
    # concentration guard: solid planets have FLAT profiles (<0.16 measured,
    # 0.22 is the galaxy boundary) - without it a round blue-white dwarf
    # galaxy (star cloud, conc 0.225) got rescued into "planet/Moon disk"
    if (not is_disk_like and not is_point_like and not is_streak
            and equiv_diameter >= 60 and ext_blue >= 0.28 and elong < 2.4
            and concentration < 0.20):
        disk_fit = _fit_limb_circle(ext_mask)
        if disk_fit:
            is_disk_like = True
            # the lit-mask moments describe the crescent, not the planet:
            # report the fitted disk so the overlay draws the true circle
            cx, cy = disk_fit["cx"], disk_fit["cy"]
            major_px = minor_px = equiv_diameter = 2 * disk_fit["r"]
            elong, angle_deg = 1.0, 0.0
    elif is_disk_like and equiv_diameter >= 60:
        # already a disk: fit anyway so the exclusion below can cover the
        # full circle (dark limb + city lights) instead of the lit mask only
        disk_fit = _fit_limb_circle(ext_mask)
    # ringed planet (Saturn): the ring system breaks every circularity test,
    # so detect it by its unique major-axis profile instead
    ring_sig = (_ring_gap_signature(lum, ext_mask)
                if not (is_disk_like or is_point_like or is_streak)
                and equiv_diameter >= 60 else None)
    is_ringed_disk = bool(ring_sig and ring_sig["ringed"])

    is_extended_fuzzy = bool(not is_point_like and not is_streak
                             and equiv_diameter >= 20 and not is_disk_like
                             and not is_ringed_disk
                             and concentration >= 0.18)

    # ---- stars, with a type-aware exclusion zone --------------------------------
    # A SOLID disk (planet/Moon) physically cannot show stars through it: its
    # storm spots and surface details are NOT stars, so the whole disk region
    # is excluded. Galaxies/nebulae keep only a thin edge annulus excluded -
    # foreground stars genuinely appear superimposed on them.
    exclude = None
    if (not (is_disk_like or is_ringed_disk)
            and ext_blue >= 0.30 and area > 0.2 * w * h):
        # BLUE PLANETARY SURFACE, whatever the shape tests said: a night-side
        # bite can break every disk criterion, but 30%+ ocean-blue pixels over
        # a fifth of the frame is a planet - its cloud texture is never stars
        # (a stock Earth photo drew 15k circles without this)
        exclude = ndimage.binary_dilation(ext_mask, iterations=6)
    elif is_ringed_disk and area > 400:
        # exclude the BRIGHT structure (planet + rings, the quarter-max
        # mask), NOT the faint glow envelope: moons sitting in the dark gap
        # or just off the rings were being swallowed whole (only Titan
        # survived on the JWST frame)
        exclude = ndimage.binary_dilation(
            qm | main_bright_mask,
            iterations=max(4, int(0.004 * max(w, h))))
    elif (is_disk_like
          or (area > 0.30 * w * h
              and (rb_ratio >= 3.0 or saturation >= 0.5
                   or not is_extended_fuzzy))) and area > 400:
        # the "30%+" clause is for HOT/solid close-ups: any single object at
        # that coverage whose glow is Sun-like (rb>=3 or sat>=0.5) or not
        # fuzzy hides no stars behind its texture. COOL resolved galaxies
        # (NGC 1672: fill 0.86, rb 1.05, sat 0.04) are exempt - their
        # interior knots and stars are REAL sources the user wants marked.
        # the frame is a close-up whose interior texture is surface detail,
        # not stars (the Sun's granulation drew 4k circles; every solvable
        # sky field measures fill <= 0.29)
        # rings sparkle with compression texture that is NOT stars; moons
        # OUTSIDE the disk+ring region remain legitimate detections.
        # The exclusion rim scales with resolution: ring-edge speckles sat
        # 6-7px outside the mask at 2400px while the real moon sat 32px out.
        exclude = ndimage.binary_dilation(
            ext_mask, iterations=max(6, int(0.006 * max(w, h))))
        if disk_fit:
            # the lit mask misses the NIGHT side of the disk - city lights
            # there are surface features, never stars (101 phantom stars on
            # a terminator Earth without this)
            # 12% limb margin: atmospheric fade / limb darkening trims the
            # luminance mask ~10% inside the true limb, and city lights sit
            # right on that dark rim
            yy2, xx2 = np.mgrid[:h, :w]
            full_disk = ((yy2 - disk_fit["cy"]) ** 2
                         + (xx2 - disk_fit["cx"]) ** 2
                         ) <= (disk_fit["r"] * 1.12) ** 2
            exclude |= full_disk
    elif area > 400:
        edge = ext_mask & ~ndimage.binary_erosion(ext_mask, iterations=4)
        exclude = ndimage.binary_dilation(edge, iterations=6)
        core = ndimage.binary_dilation(main_bright_mask, iterations=3) & ~ext_mask
        exclude |= core  # bright fragments just outside the extended region
    _stars_and_notables(exclude)

    features["main_source"] = {
        "center_x": round(cx, 1),
        "center_y": round(cy, 1),
        "area_px": int(area),
        "equiv_diameter_px": round(equiv_diameter, 1),
        "bright_diameter_px": round(bright_diameter, 1),
        "major_axis_px": round(major_px, 1),
        "minor_axis_px": round(minor_px, 1),
        "angle_deg": round(angle_deg, 1),
        "elongation": round(elong, 2),
        "circularity": round(circularity, 3),
        "fill_fraction": round(area / (w * h), 4),
        "mean_rgb": [round(mean_r, 3), round(mean_g, 3), round(mean_b, 3)],
        "rb_color_ratio": round(rb_ratio, 3),
        "saturation": round(saturation, 3),
        "clipped_fraction": round(src_clipped, 3),
        "limb_sharpness": round(limb_sharpness, 3),
        "texture": round(texture, 3),
        "concentration": round(concentration, 3),
        "bright_circularity": round(bright_circularity, 3),
        "band_contrast": round(band_contrast, 3),
        "border_contact": round(border_contact, 3),
        "chroma_shift_px": round(chroma_shift_px, 2),
        "is_streak": is_streak,
        "streak_taper": streak_taper,
        "streak_dashes": streak_dashes,
        "curtain_ratio": curtain_ratio,
        "is_point_like": is_point_like,
        "is_disk_like": is_disk_like,
        "disk_fit": disk_fit,
        "is_ringed_disk": is_ringed_disk,
        # wide-sweep detections mark the planet-outshines-rings morphology
        # (JWST ice giants) - the classifier's naming honesty depends on it
        "ring_wide_sweep": bool(ring_sig and ring_sig.get("wide_sweep")),
        "is_extended_fuzzy": is_extended_fuzzy,
    }

    # ---- multiple obvious sources ---------------------------------------------------
    # "At a glance this photo clearly holds five galaxies" - list every source
    # big enough to see immediately (not the star-scale points), deblending
    # merged groups. Powers the numbered-source overlay + per-source analysis.
    sources: list[dict] = []
    if is_disk_like or is_ringed_disk:
        # a resolved planet/Moon IS one object: its surface albedo patches and
        # ring arcs must never be "sources" (Mars split into a fake "galaxy
        # group", Saturn's rings became "edge-on galaxies" without this)
        sources = [_measure_source(lum, rgb, ext_mask, bg, 1)]
        features["sources"] = sources
        src_n = 0
    else:
        src_labels, src_n = ndimage.label(lum > bg + 5.0 * sigma)
    if src_n:
        src_areas = ndimage.sum_labels(np.ones_like(lum), src_labels,
                                       index=range(1, src_n + 1))
        min_d = max(30.0, 0.022 * max(w, h))
        pieces: list[np.ndarray] = []
        for idx in np.argsort(-src_areas)[:12]:
            if 2.0 * np.sqrt(src_areas[idx] / np.pi) < min_d:
                break
            comp = src_labels == int(idx) + 1
            # Earth-signature check on the component's own pixels: blue-ocean
            # fraction >= 8% marks it a blue planet whose cloud lobes must
            # never be split into fake "sources"
            cys, cxs = np.nonzero(comp)
            cpx = rgb[cys, cxs]
            cmax = cpx.max(axis=1)
            csat = (cmax - cpx.min(axis=1)) / np.maximum(cmax, 1e-4)
            comp_blue = float(np.mean((cpx[:, 2] > cpx[:, 0] * 1.15)
                                      & (csat > 0.15) & (cmax > 0.08)))
            pieces.extend(_split_blended(lum, comp, min_d,
                                         bg_split=bg + 6.0 * sigma,
                                         earthlike=comp_blue >= 0.08))
        # significance floor: a "source" must actually be BRIGHT - noise blobs
        # on a compressed black sky measured peak 0.004-0.016 and became
        # circled "galaxies" in visibly empty space
        # (capped: 8-sigma explodes past 1.0 on bright noisy frames and was
        # deleting every real source in N44)
        src_floor = bg + min(max(8.0 * sigma, 0.05), 0.2)
        pieces = [m for m in pieces
                  if 2.0 * np.sqrt(float(m.sum()) / np.pi) >= min_d
                  and float(np.percentile(lum[m], 95)) >= src_floor]
        pieces.sort(key=lambda m: -float(m.sum()))
        sources = [_measure_source(lum, rgb, m, bg, i + 1)
                   for i, m in enumerate(pieces[:8])]
    features["sources"] = sources

    # ---- true-shape display region -------------------------------------------------
    # Disks and points render honestly as ellipses; everything extended ships
    # its measured mask so the viewer traces the real outline ("the ellipse
    # never covers the actual shape" - direct user feedback).
    if not (is_disk_like or is_point_like):
        display_mask = ext_mask
        # Nebula-dominated frame (Pillars of Creation in visible light: the
        # whole frame glows at median 0.32): the 5-sigma component is a tiny
        # knot, so show the full emission region instead of a misplaced sliver.
        if median_lum > 0.22 and area / (w * h) < 0.05:
            # smooth first: a raw threshold traces sensor noise into a
            # speckled outline; the display should read as ONE glow boundary
            emission = ndimage.gaussian_filter(lum, 3) > 0.6 * median_lum
            elabels, en = ndimage.label(emission)
            if en:
                eareas = ndimage.sum_labels(np.ones_like(lum), elabels,
                                            index=range(1, en + 1))
                emission = elabels == int(np.argmax(eareas)) + 1
            hlabels, hn = ndimage.label(~emission)
            if hn:
                hareas = ndimage.sum_labels(np.ones_like(lum), hlabels,
                                            index=range(1, hn + 1))
                small = np.nonzero(hareas < 0.002 * w * h)[0] + 1
                if len(small):
                    emission |= np.isin(hlabels, small)
            if emission.mean() > 0.35:
                display_mask = emission
                features["main_source"]["frame_filling"] = True
        features["main_source"]["region_mask_png"] = _mask_overlay_png(
            display_mask, (52, 211, 153))

    # ---- dark absorption structure --------------------------------------------------
    # On a bright nebular background the photographed SUBJECT is often a dark
    # dust silhouette (the Pillars themselves). Detected only on luminous
    # frames; the corner test rejects "dark sky around a bright object", which
    # is background, not structure.
    if not is_disk_like and median_lum > 0.22:
        dark = ndimage.binary_opening(lum < 0.6 * median_lum, iterations=2)
        dlabels, dn = ndimage.label(dark)
        if dn:
            dareas = ndimage.sum_labels(np.ones_like(lum), dlabels,
                                        index=range(1, dn + 1))
            dmask = ndimage.binary_fill_holes(dlabels == int(np.argmax(dareas)) + 1)
            frac = float(dmask.sum()) / (w * h)
            corners = sum(bool(dmask[yy, xx]) for yy, xx in
                          ((0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)))
            if 0.015 <= frac <= 0.30 and corners <= 1:
                features["dark_structure"] = {
                    "area_fraction": round(frac, 4),
                    "mask_png": _mask_overlay_png(dmask, (248, 113, 113)),
                }
    return features
