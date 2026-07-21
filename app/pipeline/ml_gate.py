"""ML gate: is this image a photograph of the sky?

A MobileNetV3-Small pretrained on ImageNet (1.2M photos across 1000 everyday
categories) with a small head trained on our astro / not-astro data. The
pretrained features are the whole point: the negative space is unbounded -
disco balls, footballs under floodlights, jewellery, wet stones at night - and
no collection effort can enumerate it. The backbone has already seen those
things, so the head only has to learn one boundary and still generalises to
objects we never collected.

Falls back silently to the rule-based gate when no model file is installed.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

CNN_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "astro_gate_cnn.pt"
CNN_SIZE = 160
# Single source of truth for the reject decision. Calibrated on held-out
# cases: every trap scores <= 0.06, every real astro image >= 0.28 (the
# hardest being the razor-sharp Viking Mars map mosaic) - 0.20 leaves margin
# on both sides.
REJECT_THRESHOLD = 0.20
# Above this, a confident CNN may RESCUE an image the brightness rules
# rejected (frame-filling planets break pixel statistics: the "sky" is the
# planet). Kept high so weak scores never override the proven rules.
RESCUE_THRESHOLD = 0.60
# the backbone expects ImageNet-normalised input
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_cnn = None
_cnn_loaded = False


def _load_cnn():
    global _cnn, _cnn_loaded
    if _cnn_loaded:
        return _cnn
    _cnn_loaded = True
    if CNN_PATH.exists():
        try:
            import torch

            _cnn = torch.jit.load(str(CNN_PATH), map_location="cpu")
            _cnn.eval()
        except Exception:
            _cnn = None
    return _cnn


def available() -> bool:
    return _load_cnn() is not None


def astro_probability(rgb: np.ndarray) -> float | None:
    """P(image is a sky photograph) in [0, 1]; None if no model is installed."""
    cnn = _load_cnn()
    if cnn is None:
        return None

    import torch
    from PIL import Image

    u8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    thumb = np.asarray(
        Image.fromarray(u8).resize((CNN_SIZE, CNN_SIZE), Image.BILINEAR),
        dtype=np.float32) / 255.0
    thumb = ((thumb - _IMAGENET_MEAN) / _IMAGENET_STD).astype(np.float32)
    x = torch.from_numpy(thumb.transpose(2, 0, 1)).unsqueeze(0)
    with torch.no_grad():
        return float(torch.softmax(cnn(x), dim=1)[0, 1])
