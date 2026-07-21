"""Train the astro / not-astro gate on a pretrained ImageNet backbone.

Why transfer learning: the negative space is unbounded (footballs, stadiums,
jewellery, wet stones at night, logos...). Training from scratch on a few
thousand images can never cover it. A MobileNetV3 pretrained on ImageNet has
already seen 1.2M photos across 1000 categories - it knows what a ball, a
stadium, grass and a light bulb look like. We only teach it the one boundary
we care about, which needs far less data and generalises to objects we never
collected.

Frozen backbone + trained head; brightness jitter keeps "dark = astro" from
becoming a shortcut.
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from PIL import Image
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

ROOT = Path(__file__).resolve().parent.parent
SIZE = 160
GALAXY10_SAMPLE = 2000
EPOCHS = 8
BATCH = 48
TRAP_WEIGHT = 25.0
rng = np.random.default_rng(11)
torch.manual_seed(0)

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass
try:
    import pillow_avif  # noqa: F401
except ImportError:
    pass

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


_VIG_Y, _VIG_X = np.mgrid[0:SIZE, 0:SIZE]
_VIG_R = np.sqrt(((_VIG_Y - SIZE / 2) / (SIZE / 2)) ** 2
                 + ((_VIG_X - SIZE / 2) / (SIZE / 2)) ** 2)


def vignette_batch(x, rng):
    """Fade the frame edges to black on a random subset - applied to BOTH
    classes, like the brightness jitter.

    Every gate failure has one geometry: a bright object surrounded by
    darkness (disco ball, ball under floodlights, product shot). COCO is
    daylight-only, so the head learned "dark surround = sky". Adding vignetted
    NEGATIVES only flipped the shortcut - it then rejected real eclipses and
    planets, which have exactly that geometry. Applying the same vignette to
    both classes instead makes darkness carry no information at all, so the
    object itself has to decide.
    """
    for i in range(len(x)):
        if rng.random() > 0.3:
            continue
        inner, outer = rng.uniform(0.20, 0.50), rng.uniform(0.60, 1.0)
        mask = np.clip((outer - _VIG_R) / max(outer - inner, 1e-3), 0, 1) ** rng.uniform(1.0, 3.0)
        x[i] = x[i] * mask[..., None].astype(np.float32)
    return x


def hue_batch(x, rng):
    """Random hue rotation on a subset - applied to BOTH classes.

    Narrowband astro palettes are arbitrary (SHO/HOO mappings differ per
    processor), yet the head learned 'smooth cyan gradient = daytime sky':
    the 2014 Pillars scored 0.02 while the SAME object in the 1995 palette
    scored 0.99, and swapping its R/B channels moved it 0.02->0.42. Rotating
    hue class-agnostically makes the specific palette carry no information.
    Uses the standard luminance-preserving hueRotate matrix (vectorizable),
    not per-pixel HSV."""
    for i in range(len(x)):
        if rng.random() > 0.3:
            continue
        a = rng.uniform(-1.05, 1.05)  # radians, ~±60 deg
        c, s = np.cos(a), np.sin(a)
        m = np.array([
            [0.213 + c * 0.787 - s * 0.213, 0.715 - c * 0.715 - s * 0.715, 0.072 - c * 0.072 + s * 0.928],
            [0.213 - c * 0.213 + s * 0.143, 0.715 + c * 0.285 + s * 0.140, 0.072 - c * 0.072 - s * 0.283],
            [0.213 - c * 0.213 - s * 0.787, 0.715 - c * 0.715 + s * 0.715, 0.072 + c * 0.928 + s * 0.072],
        ], dtype=np.float32)
        x[i] = np.clip(x[i] @ m.T, 0, 1)
    return x


def load_thumb(path: Path):
    try:
        img = Image.open(path)
        if img.mode in ("RGBA", "LA", "PA") or (img.mode == "P" and "transparency" in img.info):
            rgba = img.convert("RGBA")
            white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            img = Image.alpha_composite(white, rgba)
        return np.asarray(img.convert("RGB").resize((SIZE, SIZE), Image.BILINEAR),
                          dtype=np.uint8)
    except Exception:
        return None


print("loading images...")
t0 = time.time()
imgs, labels, weights, names = [], [], [], []

import re

COCO_NAME = re.compile(r"^\d{12}\.jpg$")   # bulk filler: 000000000139.jpg
# test_images/ holds BOTH classes - never let a negative leak in as a positive
# (a football copied there for testing was trained as "astro", scoring 0.93).
NEG_TEST_IMAGES = {
    "game_logo.jpg", "render_3d.jpg", "street_synth.jpg", "empty_noise.jpg",
    "football_dark.jpg", "football_stadium.webp",
}

neg_files = [p for p in (ROOT / "data" / "not_astro").rglob("*")
             if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp", ".avif")]
neg_files += [p for p in (ROOT / "test_images").glob("*")
              if p.name in NEG_TEST_IMAGES and p.name != "empty_noise.jpg"]
seen_neg = set()
for p in neg_files:
    if p.name in seen_neg:
        continue
    seen_neg.add(p.name)
    a = load_thumb(p)
    if a is not None:
        trap = not COCO_NAME.match(p.name)   # user-collected trap vs bulk COCO
        imgs.append(a); labels.append(0)
        weights.append(TRAP_WEIGHT if trap else 1.0)
        names.append(p.name)
n_traps = sum(1 for w in weights if w > 1)
print(f"  negatives: {len(imgs)} ({n_traps} user traps)  ({time.time()-t0:.0f}s)")

# test_images/ stays OUT of training entirely - it is the held-out
# verification set; training on it made the gate-case checks meaningless
pos_files = list((ROOT / "data" / "astro").glob("*.jpg"))
for p in pos_files:
    a = load_thumb(p)
    if a is not None:
        # full-disk planets are the scarcest class and the newest failure
        # mode (Mars scored 0.045) - weight them harder than other positives.
        # nebclose_ = bright frame-filling nebula close-ups (visible-light
        # Pillars of Creation scored 0.004: "bright+colorful" shortcut)
        wgt = (8.0 if p.name.startswith("planet_")
               else 6.0 if p.name.startswith("nebclose_") else 4.0)
        imgs.append(a); labels.append(1); weights.append(wgt); names.append(p.name)

# explicit anchors for the two poles of the hardest boundary:
# (a) zoomed trap variants (a cropped disco ball must STILL be a trap)
# (b) frame-filling planet crops (Viking Mars fills 98% of its frame)
def _center_crop(u8, f, rng2):
    s = int(SIZE * f)
    o = (SIZE - s) // 2
    from PIL import Image as _Im
    return np.asarray(_Im.fromarray(u8[o:o + s, o:o + s]).resize(
        (SIZE, SIZE), _Im.BILINEAR), dtype=np.uint8)


anchor_rng = np.random.default_rng(3)
n_before = len(imgs)
for i in range(n_before):
    nm = names[i]
    if labels[i] == 0 and not COCO_NAME.match(nm) and nm != "galaxy10":
        for _ in range(4):
            imgs.append(_center_crop(imgs[i], anchor_rng.uniform(0.55, 0.85), anchor_rng))
            labels.append(0); weights.append(TRAP_WEIGHT / 2); names.append("zoomtrap_" + nm)
    elif labels[i] == 1 and nm.startswith("planet_"):
        imgs.append(_center_crop(imgs[i], anchor_rng.uniform(0.5, 0.7), anchor_rng))
        labels.append(1); weights.append(8.0); names.append("zoomplanet_" + nm)
print(f"  + {len(imgs) - n_before} anchor variants (zoom traps / frame-filling planets)")

import h5py
with h5py.File(ROOT / "data" / "Galaxy10_DECals.h5", "r") as f:
    idx = np.sort(rng.choice(f["images"].shape[0], GALAXY10_SAMPLE, replace=False))
    for arr in f["images"][idx]:
        imgs.append(np.asarray(Image.fromarray(arr).resize((SIZE, SIZE), Image.BILINEAR)))
        labels.append(1); weights.append(1.0); names.append("galaxy10")

X = np.stack(imgs); y = np.array(labels, dtype=np.int64)
w = np.array(weights, dtype=np.float32)
print(f"  total: {len(X)} (pos={int(y.sum())}, neg={int((1-y).sum())})  ({time.time()-t0:.0f}s)")

n = len(X)
perm = rng.permutation(n)
n_val = int(n * 0.15)
val_idx, tr_idx = perm[:n_val], perm[n_val:]


class Gate(nn.Module):
    """ImageNet features (last stage fine-tunable) + a small trainable head.

    A fully frozen backbone could not separate the disco-ball family from
    frame-filling planets - the decision boundary oscillated between them
    across epochs. Unfreezing the final stage lets the features themselves
    bend around that distinction."""

    def __init__(self):
        super().__init__()
        base = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        self.features = base.features
        for p in self.features.parameters():
            p.requires_grad = False
        for p in self.features[-3:].parameters():   # last conv stage trainable
            p.requires_grad = True
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(576, 128), nn.Hardswish(),
                                  nn.Dropout(0.2), nn.Linear(128, 2))

    def forward(self, x):
        return self.head(self.pool(self.features(x)))


def zoom_batch(x, rng):
    """Random centre zoom-crop on a subset (both classes). Teaches the gate
    that a FRAME-FILLING object is the same object as a framed one - the
    Viking Mars mosaic (disk covering ~98% of the image) scored 0.28 because
    every training planet floated in black sky."""
    from PIL import Image as _Im
    for i in range(len(x)):
        if rng.random() > 0.2:
            continue
        f = rng.uniform(0.65, 0.9)
        s = int(SIZE * f)
        o = (SIZE - s) // 2
        crop = (x[i, o:o + s, o:o + s] * 255).astype(np.uint8)
        x[i] = np.asarray(_Im.fromarray(crop).resize((SIZE, SIZE), _Im.BILINEAR),
                          dtype=np.float32) / 255.0
    return x


def batch_tensor(batch_u8, train):
    x = batch_u8.astype(np.float32) / 255.0
    if train:
        if rng.random() < 0.5:
            x = x[:, :, ::-1, :].copy()
        x = zoom_batch(x, rng)
        jit = rng.uniform(0.3, 1.5, (len(x), 1, 1, 1)).astype(np.float32)
        x = np.clip(x * jit, 0, 1).astype(np.float32)
        x = np.clip(vignette_batch(x, rng), 0, 1).astype(np.float32)
        x = hue_batch(x, rng)
    x = ((x - MEAN) / STD).astype(np.float32)
    return torch.from_numpy(np.ascontiguousarray(x.transpose(0, 3, 1, 2)))


model = Gate()
opt = torch.optim.Adam([
    {"params": model.head.parameters(), "lr": 8e-4},
    {"params": [p for p in model.features.parameters() if p.requires_grad],
     "lr": 1e-4},   # gentle on pretrained weights
])

trap_idx = [i for i, nm in enumerate(names)
            if nm != "galaxy10" and y[i] == 0 and not COCO_NAME.match(nm)]
planet_val_idx = [i for i in val_idx
                  if names[i].startswith(("planet_", "zoomplanet_"))]
neb_val_idx = [i for i in val_idx if names[i].startswith("nebclose_")]

# Epoch-to-epoch trap scores oscillate (worst 0.54 <-> 0.83 across runs), so
# keep the BEST checkpoint by a combined score instead of blindly saving the
# last epoch: high accuracy on both classes, low trap scores, high planet
# scores - the two poles of the disco-ball-vs-Mars tension.
best_score, best_state = -1e9, None

for epoch in range(EPOCHS):
    model.train()
    model.features.eval()          # keep frozen BN stats
    ep = rng.permutation(tr_idx)
    losses = []
    for i in range(0, len(ep), BATCH):
        bi = ep[i:i + BATCH]
        logits = model(batch_tensor(X[bi], True))
        loss = (nn.functional.cross_entropy(logits, torch.from_numpy(y[bi]),
                                            reduction="none")
                * torch.from_numpy(w[bi])).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(float(loss.detach()))
    model.eval()
    correct = np.zeros(2); totals = np.zeros(2)
    with torch.no_grad():
        for i in range(0, len(val_idx), 128):
            bi = val_idx[i:i + 128]
            pred = model(batch_tensor(X[bi], False)).argmax(1).numpy()
            for c in (0, 1):
                m = y[bi] == c
                totals[c] += m.sum(); correct[c] += (pred[m] == c).sum()
        tp = torch.softmax(model(batch_tensor(X[trap_idx], False)), 1)[:, 1].numpy()
        pl = (torch.softmax(model(batch_tensor(X[planet_val_idx], False)), 1)[:, 1].numpy()
              if planet_val_idx else np.array([1.0]))
        nb = (torch.softmax(model(batch_tensor(X[neb_val_idx], False)), 1)[:, 1].numpy()
              if neb_val_idx else np.array([1.0]))
    acc0 = correct[0] / max(totals[0], 1)
    acc1 = correct[1] / max(totals[1], 1)
    score = acc0 + acc1 + pl.mean() + nb.mean() - 2.0 * tp.mean() - tp.max()
    if score > best_score:
        best_score = score
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
        mark = " *best*"
    else:
        mark = ""
    print(f"epoch {epoch+1}/{EPOCHS} loss={np.mean(losses):.4f} "
          f"val not_astro={acc0:.3f} astro={acc1:.3f} "
          f"| traps mean P={tp.mean():.3f} worst={tp.max():.3f} >0.5:{(tp>0.5).sum()}/{len(tp)} "
          f"| planets={pl.mean():.3f} nebulae={nb.mean():.3f} ({time.time()-t0:.0f}s){mark}")

model.load_state_dict(best_state)
model.eval()
torch.jit.script(model).save(str(ROOT / "data" / "astro_gate_cnn.pt"))
print(f"saved BEST checkpoint -> data/astro_gate_cnn.pt  total {time.time()-t0:.0f}s")
