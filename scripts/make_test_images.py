"""Generate synthetic test images exercising each pipeline branch."""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

OUT = Path(__file__).resolve().parent.parent / "test_images"
OUT.mkdir(exist_ok=True)
rng = np.random.default_rng(42)


def save(arr, name):
    img = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))
    img.save(OUT / name, quality=92)
    print("wrote", OUT / name)


def gaussian_blob(h, w, cy, cx, sigma, amp=1.0):
    yy, xx = np.mgrid[:h, :w]
    return amp * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma**2))


H, W = 800, 1000

# 1. blurry red planet-like disk (the user's canonical scenario)
img = np.zeros((H, W, 3), dtype=np.float32)
disk = gaussian_blob(H, W, 400, 500, 6) * 0  # placeholder
yy, xx = np.mgrid[:H, :W]
r = np.sqrt((yy - 400) ** 2 + (xx - 500) ** 2)
disk_mask = 1 / (1 + np.exp((r - 80) / 6.0))  # soft-edged disk radius 80
texture = 0.15 * np.sin(yy / 17.0) * disk_mask  # faint banding
img[..., 0] = 0.75 * disk_mask + texture
img[..., 1] = 0.38 * disk_mask + texture * 0.6
img[..., 2] = 0.22 * disk_mask + texture * 0.4
img += rng.normal(0.02, 0.008, img.shape).astype(np.float32)
save(img, "red_planet_blurry.jpg")

# 2. star field (many point sources)
img = rng.normal(0.03, 0.01, (H, W, 3)).astype(np.float32)
for _ in range(120):
    cy, cx = rng.uniform(10, H - 10), rng.uniform(10, W - 10)
    amp = rng.uniform(0.3, 1.0)
    sigma = rng.uniform(1.0, 2.2)
    b = gaussian_blob(H, W, cy, cx, sigma, amp)
    tint = rng.uniform(0.85, 1.15)
    img[..., 0] += b * tint
    img[..., 1] += b
    img[..., 2] += b / tint
save(img, "star_field.jpg")

# 3. fuzzy elliptical galaxy-like glow
img = rng.normal(0.03, 0.01, (H, W, 3)).astype(np.float32)
yy, xx = np.mgrid[:H, :W]
a, b_ax = 160.0, 70.0
re = np.sqrt(((xx - 500) / a) ** 2 + ((yy - 400) / b_ax) ** 2)
glow = np.exp(-2.5 * re**0.5)  # de Vaucouleurs-ish falloff
img[..., 0] += glow * 0.55
img[..., 1] += glow * 0.50
img[..., 2] += glow * 0.38
for _ in range(25):  # sprinkle a few field stars
    cy, cx = rng.uniform(10, H - 10), rng.uniform(10, W - 10)
    bl = gaussian_blob(H, W, cy, cx, rng.uniform(1.0, 1.8), rng.uniform(0.3, 0.8))
    img += bl[..., None] * np.array([1.0, 0.95, 0.9])
save(img, "fuzzy_galaxy.jpg")

# 4. satellite streak
img = rng.normal(0.03, 0.01, (H, W, 3)).astype(np.float32)
for t in np.linspace(0, 1, 3000):
    cy, cx = 150 + t * 500, 100 + t * 800
    if 0 <= cy < H and 0 <= cx < W:
        img[int(cy), int(cx)] += 0.6
from scipy import ndimage
img = ndimage.gaussian_filter(img, (1.5, 1.5, 0))
for _ in range(15):
    cy, cx = rng.uniform(10, H - 10), rng.uniform(10, W - 10)
    img += gaussian_blob(H, W, cy, cx, 1.5, 0.5)[..., None] * np.array([1.0, 1.0, 1.0])
save(img, "satellite_streak.jpg")

# 5. single orange point source (cool star)
img = rng.normal(0.02, 0.008, (H, W, 3)).astype(np.float32)
star = gaussian_blob(H, W, 400, 500, 3.5, 0.95)
img[..., 0] += star * 1.0
img[..., 1] += star * 0.6
img[..., 2] += star * 0.35
save(img, "orange_point_star.jpg")

# 6. gray textured moon-like disk
img = rng.normal(0.02, 0.008, (H, W, 3)).astype(np.float32)
r = np.sqrt((yy - 400) ** 2 + (xx - 500) ** 2)
disk_mask = (r < 220).astype(np.float32)
crater_noise = ndimage.gaussian_filter(rng.normal(0, 1, (H, W)), 8)
lunar = (0.55 + 0.18 * crater_noise) * disk_mask
img += lunar[..., None] * np.array([1.0, 0.98, 0.94])
save(img, "moon_like_disk.jpg")

# 7. near-empty noise frame (indeterminate case)
img = rng.normal(0.02, 0.006, (H, W, 3)).astype(np.float32)
save(img, "empty_noise.jpg")

print("done")
