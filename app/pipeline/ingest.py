"""Image ingestion: decoding (incl. iPhone HEIC), EXIF extraction, normalization."""
from __future__ import annotations

import io
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageOps
from PIL.ExifTags import GPSTAGS, TAGS

try:
    from pillow_heif import register_heif_opener

    register_heif_opener()
    HEIC_SUPPORTED = True
except ImportError:  # pragma: no cover
    HEIC_SUPPORTED = False


@dataclass
class ImageData:
    rgb: np.ndarray  # HxWx3 float32 in [0, 1]
    width: int
    height: int
    original_width: int
    original_height: int
    exif: dict = field(default_factory=dict)
    format: str = ""
    transparent_fraction: float = 0.0  # photos never have transparency


def _rational_to_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _parse_gps(gps_ifd: dict) -> dict:
    out = {}
    tags = {GPSTAGS.get(k, k): v for k, v in gps_ifd.items()}

    def dms_to_deg(dms, ref):
        try:
            deg = _rational_to_float(dms[0]) + _rational_to_float(dms[1]) / 60 + _rational_to_float(dms[2]) / 3600
        except (IndexError, TypeError):
            return None
        if ref in ("S", "W"):
            deg = -deg
        return round(deg, 6)

    if "GPSLatitude" in tags and "GPSLatitudeRef" in tags:
        lat = dms_to_deg(tags["GPSLatitude"], tags["GPSLatitudeRef"])
        if lat is not None:
            out["latitude"] = lat
    if "GPSLongitude" in tags and "GPSLongitudeRef" in tags:
        lon = dms_to_deg(tags["GPSLongitude"], tags["GPSLongitudeRef"])
        if lon is not None:
            out["longitude"] = lon
    return out


def _extract_exif(img: Image.Image) -> dict:
    """Pull the identification-relevant EXIF fields (time, GPS, optics)."""
    out: dict = {}
    try:
        exif = img.getexif()
    except Exception:
        return out
    if not exif:
        return out

    named = {TAGS.get(k, str(k)): v for k, v in exif.items()}
    for src, dst in [
        ("DateTimeOriginal", "datetime_original"),
        ("DateTime", "datetime"),
        ("Make", "camera_make"),
        ("Model", "camera_model"),
    ]:
        if src in named:
            out[dst] = str(named[src]).strip()

    # Sub-IFD holds exposure/focal data on most cameras.
    try:
        sub = exif.get_ifd(0x8769)
        sub_named = {TAGS.get(k, str(k)): v for k, v in sub.items()}
        if "DateTimeOriginal" in sub_named:
            out["datetime_original"] = str(sub_named["DateTimeOriginal"]).strip()
        if "FocalLength" in sub_named:
            fl = _rational_to_float(sub_named["FocalLength"])
            if fl:
                out["focal_length_mm"] = round(fl, 1)
        if "FocalLengthIn35mmFilm" in sub_named:
            fl35 = _rational_to_float(sub_named["FocalLengthIn35mmFilm"])
            if fl35:
                out["focal_length_35mm"] = round(fl35, 1)
        if "ExposureTime" in sub_named:
            et = _rational_to_float(sub_named["ExposureTime"])
            if et:
                out["exposure_s"] = et
        if "ISOSpeedRatings" in sub_named:
            iso = sub_named["ISOSpeedRatings"]
            out["iso"] = int(iso[0] if isinstance(iso, (tuple, list)) else iso)
    except Exception:
        pass

    try:
        gps_ifd = exif.get_ifd(0x8825)
        if gps_ifd:
            out.update(_parse_gps(gps_ifd))
    except Exception:
        pass

    return out


def load_image(data: bytes, max_dim: int = 2400) -> ImageData:
    """Decode bytes into a normalized RGB array plus EXIF metadata."""
    img = Image.open(io.BytesIO(data))
    fmt = img.format or ""
    exif = _extract_exif(img)
    img = ImageOps.exif_transpose(img)
    orig_w, orig_h = img.size

    # Transparency handling: default RGB conversion composites alpha onto
    # BLACK, which makes any background-removed render look like a bright
    # object on a night sky. Measure the transparency (a decisive "not a
    # photograph" signal) and composite onto WHITE instead.
    transparent_fraction = 0.0
    has_alpha = (img.mode in ("RGBA", "LA", "PA")
                 or (img.mode == "P" and "transparency" in img.info))
    if has_alpha:
        rgba = img.convert("RGBA")
        alpha = np.asarray(rgba)[..., 3]
        transparent_fraction = float(np.mean(alpha < 128))
        white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        img = Image.alpha_composite(white, rgba)

    if img.mode != "RGB":
        img = img.convert("RGB")
    if max(img.size) > max_dim:
        img.thumbnail((max_dim, max_dim), Image.LANCZOS)

    rgb = np.asarray(img, dtype=np.float32) / 255.0
    return ImageData(
        rgb=rgb,
        width=img.size[0],
        height=img.size[1],
        original_width=orig_w,
        original_height=orig_h,
        exif=exif,
        format=fmt,
        transparent_fraction=transparent_fraction,
    )


def encode_analysis_jpeg(img: ImageData, quality: int = 95) -> bytes:
    """Re-encode the (downscaled) analysis image. Uploading THIS to the plate
    solver keeps returned pixel coordinates aligned with our overlay space,
    and smaller uploads solve faster in the queue.

    Quality 95, not 90: measured on a DSS 15\"/px star field, ASTAP solved the
    q95 encode instantly and failed the IDENTICAL pixels at q90 - compression
    artifacts at faint-star scale break its quad extraction."""
    pil = Image.fromarray((np.clip(img.rgb, 0, 1) * 255).astype(np.uint8))
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def field_of_view_hint(exif: dict, image_width_px: int) -> dict | None:
    """Estimate the field of view from EXIF focal length -> plate-solve scale
    hints (dramatically speeds up blind solving)."""
    import math
    fl35 = exif.get("focal_length_35mm")
    if not fl35:
        return None
    fov_deg = 2 * math.degrees(math.atan(36.0 / (2 * fl35)))
    return {
        "scale_units": "degwidth",
        # generous margin: EXIF crop-factor quirks and digital zoom
        "scale_lower": round(max(fov_deg * 0.5, 0.05), 3),
        "scale_upper": round(min(fov_deg * 2.0, 180.0), 3),
    }
