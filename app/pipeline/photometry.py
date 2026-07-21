"""Photometric measurements powering the report's charts.

Returns plain JSON-ready data; all plotting happens client-side.
"""
from __future__ import annotations

import numpy as np


def _luminance(rgb: np.ndarray) -> np.ndarray:
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


def histogram(rgb: np.ndarray, bins: int = 64) -> dict:
    lum = _luminance(rgb)
    counts, edges = np.histogram(lum, bins=bins, range=(0.0, 1.0))
    counts = counts.astype(float)
    total = counts.sum() or 1.0
    return {
        "bin_centers": [round(float((edges[i] + edges[i + 1]) / 2), 4) for i in range(bins)],
        "fractions": [round(float(c / total), 6) for c in counts],
    }


def radial_profile(rgb: np.ndarray, cy: float, cx: float, r_max: float,
                   n_bins: int = 40) -> dict:
    """Azimuthally averaged brightness vs radius around the main source."""
    lum = _luminance(rgb)
    h, w = lum.shape
    r_max = float(max(r_max, 5.0))
    yy, xx = np.ogrid[:h, :w]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    inside = r <= r_max
    r_in = r[inside]
    v_in = lum[inside]
    bin_idx = np.minimum((r_in / r_max * n_bins).astype(int), n_bins - 1)
    sums = np.bincount(bin_idx, weights=v_in, minlength=n_bins)
    counts = np.bincount(bin_idx, minlength=n_bins).astype(float)
    means = sums / np.maximum(counts, 1)
    radii = [(i + 0.5) * r_max / n_bins for i in range(n_bins)]

    # half-light radius: where cumulative flux passes 50%
    order = np.argsort(r_in)
    cumflux = np.cumsum(v_in[order])
    half_r = None
    if cumflux[-1] > 0:
        idx = int(np.searchsorted(cumflux, cumflux[-1] / 2))
        half_r = round(float(r_in[order][min(idx, len(r_in) - 1)]), 1)

    return {
        "radius_px": [round(float(x), 2) for x in radii],
        "mean_brightness": [round(float(m), 5) for m in means],
        "half_light_radius_px": half_r,
    }


def color_profile(rgb: np.ndarray, mask_center: tuple[float, float], r_max: float) -> dict:
    """Mean R/G/B within the source region - drawn as a color bar chart."""
    h, w = rgb.shape[:2]
    cy, cx = mask_center
    yy, xx = np.ogrid[:h, :w]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    inside = r <= max(r_max, 3.0)
    px = rgb[inside]
    unclipped = px[np.max(px, axis=1) < 0.98]
    use = unclipped if len(unclipped) > 20 else px
    return {
        "mean_r": round(float(np.mean(use[:, 0])), 4),
        "mean_g": round(float(np.mean(use[:, 1])), 4),
        "mean_b": round(float(np.mean(use[:, 2])), 4),
    }


def measure(rgb: np.ndarray, features: dict) -> dict:
    out = {"histogram": histogram(rgb)}
    src = features.get("main_source")
    if src:
        r_max = max(src["major_axis_px"] * 0.75, 10.0)
        out["radial_profile"] = radial_profile(
            rgb, src["center_y"], src["center_x"], r_max)
        out["color_profile"] = color_profile(
            rgb, (src["center_y"], src["center_x"]), src["major_axis_px"] / 2)
    return out
