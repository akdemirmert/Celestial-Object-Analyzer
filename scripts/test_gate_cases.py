"""Run the trained astro gate over our known tricky cases."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.pipeline import ingest, ml_gate

CASES = [
    ("game_logo.jpg", "NOT astro (oyun logosu)"),
    ("render_3d.jpg", "NOT astro (3D bina)"),
    ("street_synth.jpg", "NOT astro (sokak)"),
    ("football_stadium.webp", "NOT astro (stadyum topu)"),
    ("football_dark.jpg", "NOT astro (siyah zemin top)"),
    ("disco_ball.jpg", "NOT astro (disko topu)"),
    ("mars_viking.jpg", "NOT astro (kagit harita mozaigi)"),
    ("mars_globe.jpg", "ASTRO (Mars kuresi - kullanici vakasi)"),
    ("saturn_full.jpg", "ASTRO (Saturn)"),
    ("nebula_n44.jpg", "ASTRO (N44 bulutsusu)"),
    ("pillars_visible.jpg", "ASTRO (Yaratilis Sutunlari - parlak kare)"),
    ("pillars_ir.jpg", "ASTRO (Yaratilis Sutunlari kizilotesi)"),
    ("saturn_moons.jpg", "ASTRO (Saturn + uydular - kullanici vakasi)"),
    ("galaxy_group.jpg", "ASTRO (HCG 40 bes galaksi - kullanici vakasi)"),
    ("earth_moon_mgs.jpg", "ASTRO (Dunya+Ay Mars'tan - kullanici vakasi)"),
    ("earth_moon_galileo.jpg", "ASTRO (Dunya+Ay Galileo)"),
    ("nebula_hourglass.jpg", "ASTRO (Kum Saati Bulutsusu)"),
    ("jupiter_uv.jpg", "ASTRO (Jupiter UV)"),
    ("user_galaxy.jpg", "ASTRO (sarmal galaksi)"),
    ("real_test.jpg", "ASTRO (Hubble galaksi)"),
    ("real_skyfield.jpg", "ASTRO (yildiz alani)"),
    ("cluster_bigstars.jpg", "ASTRO (kume)"),
    ("moon_like_disk.jpg", "ASTRO-sentetik (ay diski)"),
    ("star_field.jpg", "ASTRO-sentetik (yildiz alani)"),
    ("orion_render.jpg", "ASTRO-sentetik (Orion)"),
    ("red_planet_blurry.jpg", "ASTRO-sentetik (kizil gezegen)"),
]

T = ml_gate.REJECT_THRESHOLD
print(f"{'goruntu':<24} {'P(astro)':>9}  karar(esik {T})   beklenti")
print("-" * 78)
fails = 0
for name, expect in CASES:
    p = Path(__file__).resolve().parent.parent / "test_images" / name
    if not p.exists():
        continue
    im = ingest.load_image(p.read_bytes(), max_dim=512)
    prob = ml_gate.astro_probability(im.rgb)
    decision = "RET" if prob < T else "KABUL"
    ok = (decision == "RET") == expect.startswith("NOT")
    fails += not ok
    print(f"{name:<24} {prob:>8.2f}  {decision:<6} {'OK ' if ok else 'FAIL'}  {expect}")
print(f"\n{'HEPSI DOGRU' if not fails else str(fails) + ' HATA'}")
