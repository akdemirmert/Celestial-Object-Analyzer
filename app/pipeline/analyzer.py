"""Rule-based hypothesis engine (v2).

Combines image measurements, the astrometric solution, catalog data and
solar-system ephemerides into an evidence-linked report. Each hypothesis
carries a *consistency score* (0-100; how well the evidence fits that class -
NOT a calibrated probability) and a qualitative band. "Cannot be determined"
remains a first-class answer.
"""
from __future__ import annotations

STRONG, MODERATE, WEAK = "Strong", "Moderate", "Weak"


def _band(score: float) -> str:
    if score >= 70:
        return STRONG
    if score >= 40:
        return MODERATE
    return WEAK


def _hyp(label: str, score: float, evidence: list[str],
         notes: list[str] | None = None) -> dict:
    score = max(5.0, min(98.0, score))
    return {"label": label, "score": round(score), "band": _band(score),
            "evidence": evidence, "notes": notes or []}


# --------------------------------------------------------------------------- #
# Color physics
# --------------------------------------------------------------------------- #

def _star_temperature_class(rb_ratio: float) -> tuple[str, str]:
    if rb_ratio < 0.85:
        return ("blue-white", "hot star (spectral class B/A, roughly 7,500-20,000+ K)")
    if rb_ratio < 1.10:
        return ("white", "Sun-like temperature (spectral class F/G, roughly 5,000-7,500 K)")
    if rb_ratio < 1.40:
        return ("yellow-orange", "cooler than the Sun (spectral class K, roughly 3,900-5,300 K)")
    return ("orange-red", "cool star (spectral class M, roughly 2,500-3,900 K)")


def _planet_surface_hypotheses(src: dict, features_ctx: dict | None = None) -> list[dict]:
    rb = src["rb_color_ratio"]
    sat = src["saturation"]
    texture = src["texture"]
    hyps = []
    if rb > 1.3 and sat > 0.15:
        hyps.append(_hyp(
            "Rocky body with iron-oxide (rust-colored) surface material",
            55 + min(20, (rb - 1.3) * 20),
            [f"Red-dominant color (R/B flux ratio {rb:.2f})",
             f"Color saturation {sat:.2f} suggests intrinsic surface color, not white light"],
            ["Analogy: Mars owes its color to oxidized iron (rust) dust covering its surface.",
             "A cool red STAR out of focus can produce the same appearance - see caveats."],
        ))
    # Banding is a morphological signature, not a colour one - it survives the
    # false-colour/UV processing that press images use, so don't gate it on hue.
    if texture > 0.06 and src.get("band_contrast", 0) > 0.05:
        bc = src["band_contrast"]
        hyps.append(_hyp(
            "Gas giant with banded cloud structure (Jupiter/Saturn-like)",
            48 + min(22, int(bc * 250)),
            [f"The disk is crossed by horizontal bands: brightness varies "
             f"{bc * 100:.0f}% between latitudes but stays smooth along each one",
             f"Internal structure across the disk (texture index {texture:.2f})"],
            ["Only Jupiter and Saturn show this banded cloud pattern among the planets; "
             "the bands are ammonia cloud decks stretched by fast zonal winds.",
             "Naming which one needs the capture time and sky position."],
        ))
    elif 1.05 <= rb <= 1.6 and texture > 0.06 and sat >= 0.12:
        hyps.append(_hyp(
            "Gas giant with banded cloud structure",
            45 + min(15, texture * 100),
            [f"Visible brightness structure across the disk (texture index {texture:.2f})",
             f"Warm hue (R/B ratio {rb:.2f}) consistent with ammonia/hydrocarbon clouds"],
            ["Analogy: Jupiter and Saturn show banded ammonia cloud decks at these hues."],
        ))
    if sat < 0.12 and texture > 0.08:
        hyps.append(_hyp(
            "Airless cratered body (Moon-like surface)",
            60,
            [f"Low color saturation ({sat:.2f}) - essentially gray surface",
             f"Surface brightness variation (texture index {texture:.2f}) consistent with maria/craters"],
        ))
    if sat < 0.10 and texture <= 0.05:
        hyps.append(_hyp(
            "Cloud-covered body or featureless disk",
            30,
            [f"Nearly uniform, colorless disk (saturation {sat:.2f}, texture {texture:.2f})"],
            ["Analogy: Venus appears as a brilliant, nearly featureless white disk in visible light.",
             "An overexposed or defocused object produces the same appearance."],
        ))

    # Earth / ice giants: measured on the per-source color fractions (mean
    # color washes out on a cloudy globe). The banding guard matters: Hubble's
    # UV Jupiter is blue with white zones too, but it is strongly banded.
    s0 = ((features_ctx.get("sources") or [{}])[0] if features_ctx else {})
    blue, white = s0.get("blue_fraction", 0.0), s0.get("white_fraction", 0.0)
    # band guard at 0.22: Earth's own cloud bands (ITCZ, jet streams) measure
    # up to ~0.2, while Jupiter's true banding is 0.29 - keep them apart
    if blue >= 0.15 and white >= 0.03 and src.get("band_contrast", 0) < 0.22:
        hyps.insert(0, _hyp(
            "Earth (blue oceans + white clouds)", 78,
            [f"{blue * 100:.0f}% of the disk is ocean-blue and "
             f"{white * 100:.0f}% cloud-white with no latitudinal banding - "
             "Earth is the only known body with this combination"],
            ["If this is a processed/false-color image of another body, the "
             "color argument does not apply - see caveats."],
        ))
    elif blue >= 0.30 and sat >= 0.20 and texture <= 0.06:
        hyps.append(_hyp(
            "Ice giant (Uranus/Neptune-like featureless blue disk)", 55,
            [f"Uniform, strongly blue disk (blue fraction {blue * 100:.0f}%, "
             f"texture {texture:.2f}) - methane absorption tints both ice "
             "giants this way"],
        ))
    return hyps


# --------------------------------------------------------------------------- #
# Observations, caveats, notes
# --------------------------------------------------------------------------- #

def _notable_sources(features: dict) -> list[dict]:
    """Human summaries for the brightest secondary sources - so a striking
    orange dot at the frame edge is never silently ignored."""
    out = []
    for s in features.get("notable_sources", []):
        ident = s.get("id")
        color_name, temp_desc = _star_temperature_class(s.get("rb_ratio", 1.0))
        if ident:
            text = f"{ident['type_label']}"
            if ident.get("spectral_type"):
                text += f", spectral type {ident['spectral_type']}"
            if ident.get("distance_ly"):
                text += f", {ident['distance_ly']:,.0f} light-years away"
            if ident.get("v_mag") is not None:
                text += f", magnitude V={ident['v_mag']:.1f}"
            if ident.get("match_arcsec") is not None:
                text += f" (matched by sky position to {ident['match_arcsec']}\" precision)."
            elif ident.get("type_note"):
                # appearance-based identifications (e.g. Saturn's moons)
                text += f". {ident['type_note']}"
        else:
            text = (f"Point source with {color_name} color - if a star, {temp_desc}. "
                    "At point scale a distant planet or even a quasar would look identical; "
                    "a plate-solved image lets us match it against catalogs by position.")
        out.append({
            "x": s["x"], "y": s["y"],
            "name": ident["name"] if ident else None,
            "url": ident["url"] if ident else None,
            "peak_snr": s.get("peak_snr"),
            "text": text,
        })
    return out


def _observations(features: dict, solve: dict, exif: dict, sky: dict) -> list[str]:
    obs = [f"Detected {features['star_count']} reliable point sources"
           + (f" plus {features.get('faint_star_count', 0)} faint candidates (3.5-7 sigma; "
              "some may be noise)" if features.get("faint_star_count") else "")
           + " in the frame."]
    src = features.get("main_source")
    if src:
        obs.append(
            f"Main source: {src['equiv_diameter_px']:.0f} px across, "
            f"elongation {src['elongation']:.1f}, circularity {src['circularity']:.2f}."
        )
        r, g, b = src["mean_rgb"]
        obs.append(
            f"Main source mean color RGB = ({r:.2f}, {g:.2f}, {b:.2f}); "
            f"R/B flux ratio {src['rb_color_ratio']:.2f}; saturation {src['saturation']:.2f}."
        )
        obs.append(
            f"Light concentration {src['concentration']:.2f} "
            f"(uniform disk ~0.11; centrally condensed objects >0.22; PSF cores ~0.3)."
        )
        if solve.get("solved") and solve.get("pixscale_arcsec"):
            ang = src["equiv_diameter_px"] * solve["pixscale_arcsec"]
            obs.append(f"Measured angular size of main source: {ang:.1f} arcseconds.")
    if solve.get("solved"):
        obs.append(
            f"Astrometric solution: field center RA {solve['ra']:.4f} deg, "
            f"Dec {solve['dec']:.4f} deg; field radius {solve.get('radius_deg', 0):.2f} deg; "
            f"pixel scale {solve.get('pixscale_arcsec', 0):.2f} arcsec/px."
        )
    if sky.get("constellation"):
        obs.append(f"Field lies in the constellation {sky['constellation']} (IAU boundaries).")
    if sky.get("moon"):
        m = sky["moon"]
        obs.append(f"At capture time the Moon was {m['phase_name']} "
                   f"({m['illumination'] * 100:.0f}% illuminated).")
    if features["clipped_fraction"] > 0.001:
        obs.append(f"{features['clipped_fraction'] * 100:.1f}% of the frame is overexposed (clipped).")
    if exif.get("datetime_original"):
        obs.append(f"EXIF capture time: {exif['datetime_original']}.")
    if "latitude" in exif:
        obs.append(f"EXIF GPS position: {exif['latitude']:.4f}, {exif['longitude']:.4f}.")
    if exif.get("camera_model"):
        obs.append(f"Camera: {exif.get('camera_make', '')} {exif['camera_model']}"
                   + (f", {exif['focal_length_mm']} mm" if exif.get("focal_length_mm") else "") + ".")
    return obs


def _caveats(features: dict, solve: dict, exif: dict, sky: dict) -> list[str]:
    caveats = [
        "Exact chemical composition, atmosphere and temperature can only be established by "
        "spectroscopy - no single photograph, of any quality, can prove them.",
        "Camera color is not calibrated: white balance, atmospheric conditions and JPEG "
        "processing all shift the recorded colors.",
    ]
    src = features.get("main_source")
    if src:
        if src["chroma_shift_px"] > max(0.1 * src["equiv_diameter_px"], 1.5):
            caveats.append(
                "Strong chromatic aberration detected (red/blue channels displaced by "
                f"{src['chroma_shift_px']:.1f} px). Lens aberration can make an ordinary white star "
                "look like a colored disk - treat color-based hypotheses with extra suspicion."
            )
        if src["clipped_fraction"] > 0.3:
            caveats.append(
                "The main source is heavily overexposed; its recorded color and shape are unreliable."
            )
        if features["blur_score"] < 1e-4 and src["equiv_diameter_px"] > 15:
            caveats.append(
                "The image is very soft/blurry. A defocused bright star can mimic a planetary disk "
                "(a defocused Venus and a defocused Sirius can look identical)."
            )
    if not exif.get("datetime_original"):
        caveats.append(
            "No capture time in EXIF (common for images passed through messaging apps). With a "
            "timestamp and location, solar-system objects can be identified exactly from "
            "ephemerides - re-upload the original file if possible."
        )
    elif sky.get("time_caveat"):
        caveats.append(sky["time_caveat"])
    if not solve.get("solved") and not solve.get("skipped"):
        caveats.append(
            "Plate solving failed, so the exact sky position is unknown; identifications here "
            "are class-level hypotheses, not names."
        )
    if not solve.get("solved") and features.get("star_count", 0) > 1000:
        caveats.append(
            f"{features['star_count']} point-like details were detected, yet none of them "
            "matched real sky patterns during astrometric solving. If this image is artwork "
            "or not a photo of the sky, the class analysis does not apply."
        )
    return caveats


SCIENCE_NOTES = [
    "Star colors are real physics: a star's color reveals its surface temperature "
    "(blue = hot, red = cool) - the same principle as metal glowing red versus white-hot.",
    "Galaxy shapes follow the Hubble sequence: ellipticals (smooth), spirals (disk + arms), "
    "and irregulars. Blue tones indicate active star formation; yellow-red tones an older population.",
    "Red glowing nebulae are usually ionized hydrogen (H-alpha emission at 656 nm); "
    "blue nebulae are typically dust reflecting starlight.",
    "Physical size requires angular size PLUS an independent distance - it can never be read "
    "from one image alone.",
    "Plate solving matches the star pattern in your photo against all-sky catalogs; a match is "
    "essentially never wrong (Lang et al. 2010), though blurry images often fail to match at all.",
]


# --------------------------------------------------------------------------- #
# Classification rules (probabilistic mode)
# --------------------------------------------------------------------------- #

def _classify(features: dict, sky: dict) -> list[dict]:
    src = features.get("main_source")
    stars = features["star_count"]
    hyps: list[dict] = []
    if src is None:
        return hyps

    d = src["equiv_diameter_px"]
    fill = src["fill_fraction"]
    visible = sky.get("visible_bodies") or []

    # --- streak -----------------------------------------------------------------
    if src["is_streak"]:
        # score from the MEASURED axis profile, not fixed numbers - the old
        # constant 55/30/25 called every meteor a satellite
        taper = src.get("streak_taper")
        dashes = src.get("streak_dashes")
        sat_s, met_s, air_s = 55, 25, 30
        ev_sat = ["Satellites leave straight, uniform-brightness trails"]
        ev_met = ["Meteor trails taper at the ends and often flare mid-track"]
        ev_air = ["Aircraft trails show dashed/beaded structure from strobes"]
        if dashes is not None and dashes >= 3:
            air_s, sat_s, met_s = 72, 35, 20
            ev_air.insert(0, f"The trail breaks into {dashes} bright segments "
                             "- a strobe pattern")
        elif taper is not None and taper < 0.35:
            met_s, sat_s = 72, 35
            ev_met.insert(0, f"Trail brightness fades to {taper * 100:.0f}% "
                             "at the ends - the taper of an ablating meteor")
        elif taper is not None and taper >= 0.55:
            sat_s = 65
            ev_sat.insert(0, f"Near-uniform brightness along the trail "
                             f"(ends at {taper * 100:.0f}% of peak)")
        ordered = sorted([("Satellite trail (e.g. ISS or Starlink)", sat_s, ev_sat,
                           ["With capture time + GPS the exact satellite could be "
                            "identified from orbital data (planned feature)."]),
                          ("Meteor", met_s, ev_met, None),
                          ("Aircraft trail", air_s, ev_air, None)],
                         key=lambda t: -t[1])
        for nm_, sc_, ev_, notes_ in ordered:
            hyps.append(_hyp(nm_, sc_,
                             [f"Strongly elongated source (elongation "
                              f"{src['elongation']:.1f}) - a light trail"] + ev_,
                             notes_))
        return hyps

    # --- Sun warning --------------------------------------------------------------
    if fill > 0.15 and src["clipped_fraction"] > 0.6:
        sun_up = any(b["body"] == "Sun" for b in visible)
        hyps.append(_hyp(
            "The Sun (overexposed disk)", 70 if sun_up else 50,
            [f"Disk fills {fill * 100:.0f}% of the frame, {src['clipped_fraction'] * 100:.0f}% overexposed"]
            + (["EXIF time/GPS confirm the Sun was above the horizon"] if sun_up else []),
            ["SAFETY: never observe or photograph the Sun without a certified solar filter."],
        ))

    # --- resolved disk --------------------------------------------------------------
    if src["is_disk_like"] and not src["is_extended_fuzzy"]:
        # A large, crisply circular, flat-profile disk is about as unambiguous
        # as single-image astronomy gets: stars are points, galaxies/nebulae are
        # centrally condensed with soft edges. Score the evidence, don't hedge.
        base = 55 + (15 if stars < 15 else 0) + (10 if src["limb_sharpness"] > 0.05 else 0)
        base += 10 if src.get("bright_circularity", 0) > 0.85 else 0  # true circle
        base += 8 if src["concentration"] < 0.16 else 0               # genuinely flat
        base += min(8, int(fill * 20))                                # resolved, not a dot
        base_evidence = [
            f"Resolved disk with a defined edge ({d:.0f} px across, "
            f"circularity {src.get('bright_circularity', src['circularity']):.2f})",
            f"Flat light profile (concentration {src['concentration']:.2f} - a uniform disk; "
            "stars concentrate light in a point, galaxies in a bulge)",
            f"{stars} background stars visible (planets/Moon usually dominate short exposures)",
        ]
        moon_boost = 0
        moon_notes = []
        if sky.get("moon") and src["saturation"] < 0.12:
            m = sky["moon"]
            moon_boost = 10
            moon_notes.append(
                f"At capture time the Moon was {m['phase_name']} ({m['illumination'] * 100:.0f}% lit) - "
                "compare with the disk's illuminated shape.")
        # sat cap 0.15 (not 0.12): warm low-altitude atmosphere tints a phone
        # Moon to sat 0.127; the blood-Moon gate starts at 0.15 so no overlap.
        # A soft limb (atmospheric halo, measured 0.017) is accepted only
        # together with a flat, truly circular disk - galaxies stay excluded
        # by their central bulge (concentration) and ragged outline.
        if (src["saturation"] < 0.15 and src["texture"] > 0.08 and fill > 0.01
                and (src["limb_sharpness"] > 0.05
                     or (src.get("bright_circularity", 0) > 0.9
                         and src["concentration"] < 0.16))):
            hyps.append(_hyp("The Moon", base + 10 + moon_boost,
                             base_evidence + ["Gray, textured surface consistent with lunar maria and craters",
                                              f"Sharp, well-defined limb (edge sharpness {src['limb_sharpness']:.2f}) - "
                                              "typical of an airless body, unlike the soft edges of galaxies/nebulae"],
                             moon_notes))
        # BLOOD MOON: a totally eclipsed Moon is deep red-orange with the
        # maria pattern intact - the old sat<0.12 gate shut the Moon out and
        # crowned it "iron-oxide rocky body (Mars-like)"
        _rb_bm = src["rb_color_ratio"]
        if (0.15 <= src["saturation"] <= 0.72 and 1.25 <= _rb_bm <= 4.5
                and src["texture"] > 0.03 and fill > 0.01
                and src["limb_sharpness"] > 0.04):
            # honest ambiguity: pixels alone cannot split a blood Moon from
            # Mars (both warm, dim, textured disks - Mars measured lum 0.38,
            # sat 0.63); EXIF time + the lunar ephemeris settles it via
            # moon_boost, so both hypotheses are always LISTED
            # never outrank the generic disk verdict on pixels alone - only
            # the lunar ephemeris (EXIF moon_boost) may break the tie
            _ps = next((h_["score"] for h_ in hyps
                        if h_["label"].startswith("Planet (resolved")), 99)
            _bm_score = min(base + 9, _ps - 2) + moon_boost
            hyps.append(_hyp(
                "Totally eclipsed Moon (blood Moon)", _bm_score,
                base_evidence + [
                    f"Deep warm hue (R/B {_rb_bm:.2f}) over a textured disk - "
                    "sunlight refracted through Earth's atmosphere onto the "
                    "eclipsed Moon",
                    "Surface texture matches lunar maria rather than a "
                    "planet's cloud bands"],
                ["During totality the Moon turns copper-red; the capture "
                 "time (EXIF) would confirm against the eclipse catalog.",
                 "Mars shows a similar hue but is far too small to resolve "
                 "this large without a telescope."]))
        planet_notes = ["Which planet cannot be determined from appearance alone without the "
                        "capture time and sky position."]
        planet_boost = 0
        planet_up = [b for b in visible if b["body"] not in ("Sun", "Moon")]
        if planet_up:
            names = ", ".join(f"{b['body']} (alt {b['altitude_deg']:.0f} deg)" for b in planet_up[:4])
            planet_notes = [f"Planets above the horizon at capture time: {names}."]
            planet_boost = 12
        hyps.append(_hyp("Planet (resolved disk)",
                         base + planet_boost - (8 if hyps else 0),
                         base_evidence, planet_notes))
        hyps.extend(_planet_surface_hypotheses(src, features))
        return hyps

    # --- point-like -------------------------------------------------------------------
    if src["is_point_like"]:
        color_name, temp_desc = _star_temperature_class(src["rb_color_ratio"])
        hyps.append(_hyp(
            f"Star ({color_name} - {temp_desc})", 55,
            [f"Point-like light profile (concentration {src['concentration']:.2f})",
             f"Apparent color: {color_name} (R/B ratio {src['rb_color_ratio']:.2f})"],
            ["Color-to-temperature mapping assumes roughly neutral white balance.",
             "At point-source scale a star, a distant planet and even a quasar are visually "
             "indistinguishable in a single image; 'star' is the most common case, not a certainty."],
        ))
        planet_up = [b for b in (sky.get("visible_bodies") or [])
                     if b["body"] not in ("Sun", "Moon")]
        if planet_up:
            names = ", ".join(f"{b['body']} (alt {b['altitude_deg']:.0f} deg, az {b['azimuth_deg']:.0f} deg)"
                              for b in planet_up[:4])
            hyps.append(_hyp(
                "Planet appearing as a bright point", 45,
                [f"EXIF time + GPS: planets above the horizon at capture: {names}",
                 "Very bright, steady point sources are more often planets than stars"],
                ["Point the camera direction (azimuth) at one of these candidates to confirm."],
            ))
        elif src["rb_color_ratio"] > 1.25:
            hyps.append(_hyp(
                "Planet appearing as a bright point (e.g. Mars when distant)", 30,
                ["Bright warm-colored point sources low over the horizon are often planets"],
                ["Capture time + location would settle this immediately via ephemeris lookup."],
            ))
        return hyps

    # --- extended fuzzy -----------------------------------------------------------------
    # A rich resolved-star frame with LOW concentration misses the fuzzy flag
    # (its conc floor is 0.18) yet is exactly the star-forming-region case -
    # Hubble's near-IR Pillars measured conc 0.161 with 24k stars and fell
    # through to "Unclassified". Let it into the branch: the rich-field rule
    # inside handles it and returns before any galaxy call.
    rich_low_conc = (features["star_count"] >= 400
                     and src["concentration"] < 0.28
                     and not src["is_disk_like"] and not src["is_point_like"]
                     and not src["is_streak"] and src["equiv_diameter_px"] >= 20)
    if src["is_extended_fuzzy"] or rich_low_conc:
        rb = src["rb_color_ratio"]
        sat = src["saturation"]
        conc = src["concentration"]
        soft_edge = src["limb_sharpness"] < 0.03
        dominant = fill > 0.05           # object clearly dominates the frame
        colorful = sat >= 0.3            # saturated color -> emission/reflection nebula

        # Evidence-strength score: how firmly do the measurements rule out the
        # alternatives? A large, centrally-condensed, soft-edged blob is NOT a
        # star (point-like), NOT a planet/Moon (sharp limb), NOT a streak - so
        # confidence in "deep-sky extended object" is genuinely high.
        def _extended_score(base: int) -> int:
            s = base
            s += min(16, fill * 60)                        # size rules out a star
            s += 10 if conc >= 0.28 else (6 if conc >= 0.22 else 0)  # central concentration
            s += 8 if soft_edge else 0                     # soft edge rules out planet/Moon
            s += 4 if features["star_count"] > 30 else 0   # resolved field
            return int(s)

        # A galaxy has a BULGE: its light piles up in the centre (concentration
        # 0.33-0.45 measured). A star-forming region or a cluster has no
        # nucleus - light stays spread (0.18-0.19) - and the frame is full of
        # resolved stars because you are looking *inside* a galaxy, not at one.
        # Without this, Hubble's N44 nebula and a globular cluster both came
        # back as "Galaxy".
        rich_field = features["star_count"] >= 400
        if rich_field and conc < 0.28:
            neb_kind = ("emission nebula" if rb > 1.1 and sat > 0.1
                        else "reflection nebula / gas" if rb < 0.95 and sat > 0.1
                        else "nebulosity")
            hyps.append(_hyp(
                "Star-forming region / nebula with an embedded star cluster",
                _extended_score(64) + 6,
                [f"{features['star_count']} resolved stars spread across the frame - "
                 "you are looking at stars inside a galaxy, not at a distant galaxy",
                 f"The glow is NOT centrally concentrated (concentration {conc:.2f}); "
                 "a galaxy piles its light into a bright nucleus (0.30+)",
                 f"Patchy, filamentary {neb_kind} rather than a smooth disk "
                 f"(texture {src['texture']:.2f})"],
                ["Regions like this are where new stars are born: hot young stars "
                 "light up the gas that formed them and blow cavities in it.",
                 "Naming the specific region requires plate solving."],
            ))
            hyps.append(_hyp(
                "Globular / open star cluster", 45,
                [f"A dense concentration of resolved stars ({features['star_count']}) "
                 "can also be a star cluster",
                 "Clusters show little or no surrounding gas; nebulosity here argues "
                 "for a star-forming region"],
            ))
            hyps.append(_hyp(
                "Galaxy (alternative)", 30,
                ["A galaxy would concentrate its light into a nucleus and would not "
                 f"resolve into {features['star_count']} individual stars at this scale"],
            ))
            return hyps

        if not colorful and src["elongation"] < 4.0:
            # neutral/low-saturation concentrated glow -> galaxy is the strong call
            structured = src["texture"] >= 0.30
            shape_note = (("Elongated glow - consistent with an edge-on spiral or an "
                           "inclined disk galaxy") if src["elongation"] > 1.8 else
                          "Roundish glow - consistent with an elliptical or a face-on spiral galaxy")
            pop_note = ("Bluish tones suggest active star formation (a younger stellar population)."
                        if rb < 1.0 else "Yellow-red tones suggest an older stellar population.")
            evidence = [f"Large extended source with a centrally concentrated brightness "
                        f"profile (concentration {conc:.2f}) - the hallmark of a galaxy",
                        shape_note]
            if dominant:
                evidence.append(f"Dominates the frame ({fill * 100:.0f}% coverage) - far too "
                                "large and diffuse to be a star or planet")
            if soft_edge:
                evidence.append("Soft, gradually fading edge - rules out the Moon or a planet, "
                                "which show sharp, well-defined limbs")
            label = "Galaxy"
            score = _extended_score(62)
            if structured:
                label = "Galaxy (spiral / structured)"
                evidence.append(f"Visible internal brightness structure (texture {src['texture']:.2f}) "
                                "- consistent with spiral arms and dust lanes")
                score += 8
            if features["star_count"] > 200:
                evidence.append(f"{features['star_count']} point sources - resolved stars and "
                                "star-forming knots, typical of telescope images of nearby galaxies")
            hyps.append(_hyp(label, score, evidence,
                             [pop_note, "Naming the specific galaxy requires plate solving; "
                                        "typical galaxies span tens of thousands of light-years."]))
            # nebula as a genuine-but-secondary possibility (galaxy vs large nebula);
            # low color saturation argues strongly against a nebula, so keep it low
            hyps.append(_hyp(
                "Large nebula (alternative)", min(52, 25 + int(sat * 130)),
                ["Some nebulae also appear as soft extended glows",
                 f"Argues against it here: low color saturation ({sat:.2f}); nebulae usually "
                 "show strong red (emission) or blue (reflection) color"],
            ))

        elif colorful:
            # saturation alone must not outvote the galaxy signature: false-
            # color IR press images of galaxies are intensely saturated, yet a
            # bright NUCLEUS (conc 0.32+), a tilted-disk shape and thousands
            # of resolved member stars are things no nebula shows together
            # (the WISE M31 mosaic was filed as "Emission nebula" without this)
            galaxy_sig = (conc >= 0.32 and src["elongation"] >= 1.8
                          and features["star_count"] >= 300)
            neb_kind = ("Emission nebula (ionized hydrogen, H-alpha)" if rb > 1.15
                        else "Reflection nebula (dust scattering starlight)")
            if galaxy_sig:
                hyps.append(_hyp(
                    "Galaxy (strongly colored / false-color image)",
                    _extended_score(62) + 6,
                    [f"Centrally concentrated brightness profile (concentration "
                     f"{conc:.2f}) - the hallmark of a galactic nucleus; nebulae "
                     "measure 0.18-0.19",
                     f"Elongated disk shape (elongation {src['elongation']:.1f}) "
                     "- an inclined galaxy, not a gas cloud",
                     f"{features['star_count']} resolved point sources across the "
                     "disk - member stars of a nearby galaxy",
                     f"The strong color (saturation {sat:.2f}) is typical of "
                     "false-color infrared/narrowband processing, not of the "
                     "object itself"],
                    ["Naming the specific galaxy requires plate solving."]))
                hyps.append(_hyp(f"{neb_kind} (alternative)", 40,
                                 ["The saturated palette alone would suggest a nebula, "
                                  "but nebulae show no nuclear light concentration"]))
                return hyps
            evidence = [f"Extended glow with strongly saturated color (saturation {sat:.2f})",
                        f"Dominant hue is {'red - typical of H-alpha emission from ionized hydrogen' if rb > 1.15 else 'blue - typical of dust scattering nearby starlight'}"]
            if soft_edge:
                evidence.append("Soft diffuse edges, typical of interstellar gas and dust")
            hyps.append(_hyp(neb_kind, _extended_score(60), evidence))
            hyps.append(_hyp("Galaxy (alternative)", max(28, _extended_score(60) - 30),
                             ["Strongly colored extended objects are more often nebulae than galaxies"]))

        else:
            hyps.append(_hyp(
                "Extended deep-sky object (galaxy or nebula)", _extended_score(55),
                [f"Large diffuse source (concentration {conc:.2f}), but its very elongated "
                 f"shape (elongation {src['elongation']:.1f}) leaves galaxy vs nebula open"]))

        # only offer the artifact alternative for small, low-concentration blobs
        # (a defocused star can't be a dominant, centrally-condensed object)
        if not dominant and conc < 0.28:
            hyps.append(_hyp(
                "Out-of-focus star or lens artifact", 24,
                ["Small soft blobs can also come from defocus or internal lens reflections"],
            ))
        return hyps

    hyps.append(_hyp(
        "Unclassified source", 20,
        [f"Measurements (size {d:.0f} px, elongation {src['elongation']:.1f}, "
         f"concentration {src['concentration']:.2f}) do not match a clear category"],
    ))
    return hyps


def _star_field_hypotheses(features: dict) -> list[dict]:
    stars = features["star_count"]
    # frame-filling nebula guard: when the frame IS one connected sheet of
    # bright, color-saturated emission, "star field" is measurably wrong -
    # star fields are points on dark sky (<0.06 on both metrics), while a
    # narrowband NGC 3576 close-up measured 0.71 / 0.94 and still fell here
    # because no single main source could be segmented out of it.
    ecf = features.get("emission_colorful_fraction") or 0.0
    big = features.get("biggest_bright_component") or 0.0
    if ecf > 0.30 and big > 0.50:
        return [_hyp(
            "Emission nebula / star-forming region (frame-filling close-up)", 85,
            [f"{ecf * 100:.0f}% of the frame is bright, color-saturated emission "
             f"and one connected structure spans {big * 100:.0f}% of the image - "
             "a star field is point sources on a dark background, never this",
             f"{stars} embedded point sources detected within the nebulosity"],
            ["Without a plate-solved position the specific nebula cannot be named"],
        ), _hyp(
            "Star field", 15,
            ["Many point sources present, but the dominant signal is the "
             "extended emission, not the stars"],
        )]
    hyps = [_hyp(
        "Star field", 75 if stars >= 50 else 55,
        [f"{stars} point sources detected across the frame with no single dominant object"],
    )]
    if stars >= 40:
        hyps.append(_hyp(
            "Open or globular star cluster in the field", 30,
            ["Dense star concentrations can indicate a cluster; plate solving would name it"],
        ))
    return hyps


def _summary(report: dict, features: dict, solve: dict, sky: dict) -> str:
    """One professional paragraph capturing the verdict."""
    if report["mode"] == "identified" and report.get("object"):
        o = report["object"]
        parts = [f"Astrometric plate solving located this field at "
                 f"RA {solve['ra']:.3f} deg, Dec {solve['dec']:.3f} deg"]
        if sky.get("constellation"):
            parts.append(f" in the constellation {sky['constellation']}")
        parts.append(f". The primary object is {o['name']} - {o.get('type_label', 'object')}")
        if o.get("distance_ly"):
            parts.append(f", approximately {o['distance_ly']:,.0f} light-years away")
        elif o.get("distance_mly"):
            parts.append(f", approximately {o['distance_mly']:,.1f} million light-years away")
        if o.get("physical_size_ly"):
            parts.append(f", spanning roughly {o['physical_size_ly']:,.0f} light-years")
        parts.append(". This identification is astrometric and effectively certain.")
        return "".join(parts)

    # solved frames WITHOUT a primary object must never claim "could not be
    # pinned" - the position IS known (a solved phone frame carried both
    # texts at once and the user rightly called it nonsense)
    if report["mode"] == "identified" and solve and solve.get("solved"):
        parts = [f"Astrometric plate solving located this field at "
                 f"RA {solve['ra']:.3f} deg, Dec {solve['dec']:.3f} deg"]
        if sky.get("constellation"):
            parts.append(f" in the constellation {sky['constellation']}")
        parts.append(". No single cataloged object sits at the exact center; "
                     "the overlay names every source that matched a catalog "
                     "position.")
        return "".join(parts)

    hyps = report.get("hypotheses") or []
    if not hyps:
        return "No conclusion could be drawn from this image."
    top = hyps[0]
    s = (f"The image could not be pinned to an exact sky position, so this is a "
         f"probabilistic analysis. The measurements are most consistent with: "
         f"{top['label']} (consistency {top['score']}/100). ")
    if len(hyps) > 1:
        alts = ", ".join(f"{h['label']} ({h['score']})" for h in hyps[1:3])
        s += f"Alternatives considered: {alts}. "
    s += "See the caveats - honest uncertainty is part of the method."
    return s


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def _source_entries(features: dict) -> list[dict]:
    """Per-source classification for the multi-source list. Size + softness
    carry the signal (photometric star-vs-small-galaxy separation measurably
    fails at press-image scales - concentration 0.23 vs 0.25); small compacts
    get an honest either/or label unless a catalog names them."""
    dim = float(max(features.get("image_width", 1), features.get("image_height", 1)))
    out = []
    for s in features.get("sources") or []:
        d_frac = s["equiv_diameter_px"] / dim
        elong = s["elongation"]
        if elong >= 2.6 and d_frac >= 0.03 and s["saturation"] < 0.35:
            # saturation guard: galaxies are nearly colorless (max 0.30
            # measured across HCG 40) while nebular filaments are intensely
            # colored (Hourglass lobes 0.84-0.93) - without it the Hourglass
            # Nebula's lobes became a "group of 7 interacting galaxies"
            kind, label = "galaxy", "Edge-on / highly inclined galaxy"
            ev = (f"Needle-like shape (elongation {elong:.1f}) with a soft, "
                  "extended light profile - a galactic disk seen from the side")
        elif elong >= 2.6 and d_frac >= 0.03:
            kind, label = "nebula", "Nebular filament (colorful gas structure)"
            ev = (f"Elongated but intensely colored (saturation "
                  f"{s['saturation']:.2f}) - glowing gas, not a stellar disk")
        elif d_frac >= 0.05:
            if s["saturation"] >= 0.30:
                kind, label = "nebula", "Nebula (diffuse colorful gas)"
                ev = (f"Large diffuse source with strongly saturated color "
                      f"(saturation {s['saturation']:.2f})")
            elif elong < 1.5:
                kind, label = "galaxy", "Galaxy (smooth, elliptical-like)"
                ev = (f"Large, round, smoothly fading light distribution "
                      f"({s['equiv_diameter_px']:.0f} px across)")
            else:
                kind, label = "galaxy", "Galaxy (inclined spiral-like)"
                ev = (f"Large oval source ({s['equiv_diameter_px']:.0f} px, "
                      f"elongation {elong:.1f}) with structured light")
        elif s["peak"] >= 0.97 and elong < 1.5:
            kind, label = "star", "Bright foreground star (Milky Way)"
            ev = ("Compact, round and saturated to full brightness - a nearby "
                  "star sitting in front of the deep field")
        else:
            kind, label = "compact", "Compact source - distant galaxy or faint star"
            ev = (f"Small ({s['equiv_diameter_px']:.0f} px) source; at this size "
                  "a star and a distant galaxy look alike without a catalog match")
        entry = dict(s)
        entry["kind"] = kind
        # a catalog identification (set by the WCS matching in jobs) overrides
        # the class guess: "NGC 6285 (Galaxy)" instead of a generic label
        if s.get("name"):
            entry["label"] = f"{s['name']} ({s.get('type_label') or label})"
            entry["identified"] = True
        else:
            entry["label"] = label
            entry["identified"] = False
        entry["evidence"] = ev
        out.append(entry)
    return out


def analyze(features: dict, solve: dict, catalog_matches: list[dict],
            nasa_image: dict | None, exif: dict, sky: dict | None = None) -> dict:
    """Public entry: core analysis + the multi-source layer on top."""
    # DEEP-FIELD STREAK VETO, applied before ANY branch reads the flag.
    # Every streak class (satellite trail, meteor, aircraft, and the streak
    # arms of the Moon and star-trail branches) assumes a sparse sky holding
    # one linear artifact. A deep telescope frame breaks that assumption:
    # dust pillars and nebular filaments measure as elongated blobs. The
    # Pillars of Creation (7,078 point sources) walked down the chain one
    # branch at a time - "star trails", then "the Moon", then "satellite
    # trail" - as each gate was closed on its own. Clearing the flag here
    # closes all of them at once.
    _dsrc = features.get("main_source") or {}
    if ((features.get("star_count", 0)
         + features.get("faint_star_count", 0)) > 6000
            and _dsrc.get("is_streak")):
        _dsrc["is_streak"] = False

    report = _analyze_core(features, solve, catalog_matches, nasa_image, exif, sky)
    if report["mode"] == "not_astronomical":
        report["sources"] = []
        return report

    entries = _source_entries(features)
    report["sources"] = entries

    _ssrc = features.get("main_source") or {}

    # ---- star trails: many concentric arcs = Earth's rotation ---------------
    # point-source veto: Earth's rotation drags EVERY star into an arc, so a
    # true trail frame keeps almost no point sources (measured: 43 arcs vs
    # 112 points). A deep telescope frame full of points cannot be a trail -
    # JWST's diffraction spikes were counted as 12 "arcs" beside 4,328 point
    # sources and the Pillars of Creation came back as "star trails".
    _pts = features.get("star_count", 0)
    if (report["mode"] == "probabilistic"
            and features.get("star_trail_count", 0) >= max(8, _pts * 0.02)
            and (features.get("star_trail_concentric")
                 or features.get("star_trail_angle_std", 0) > 25)):
        n_arc = features["star_trail_count"]
        report["headline"] = ("Probabilistic analysis: star trails "
                              "(long-exposure Earth rotation)")
        report["hypotheses"] = [
            _hyp("Star trails (long exposure)", 85,
                 [f"{n_arc} elongated light arcs with diverse orientations"
                  + (" converging on a common center - the celestial pole"
                     if features.get("star_trail_concentric") else ""),
                  "Only Earth's rotation drags EVERY star into an arc; a "
                  "satellite leaves one or two straight trails"],
                 ["The arc center marks the celestial pole; arc length gives "
                  "the exposure duration (15 deg per hour)."]),
            _hyp("Multiple satellite trails (alternative)", 15,
                 ["A satellite train (e.g. Starlink) leaves PARALLEL streaks, "
                  "not concentric arcs"]),
        ]
        report["summary"] = (
            f"The frame holds {n_arc} star arcs - a long exposure recording "
            "Earth's rotation, not a sky anomaly.")
        return report

    # ---- total solar eclipse: bright corona around a dark disk --------------
    if report["mode"] == "probabilistic" and features.get("eclipse_ring"):
        report["headline"] = ("Probabilistic analysis: total solar eclipse "
                              "(corona)")
        report["object"] = {
            "name": "Total solar eclipse (likely)",
            "type_label": "The Sun's corona around the new Moon",
            "type_note": ("A bright ring with a dark, circular interior - "
                          "the Moon covering the solar disk with the corona "
                          "streaming around it."),
        }
        report["hypotheses"] = [
            _hyp("Total solar eclipse", 85,
                 ["A bright ring surrounds a DARK circular hole - no galaxy, "
                  "nebula or camera artifact shows this geometry",
                  "The interior darkness matches the new Moon's silhouette"],
                 ["EXIF time + location would confirm against eclipse "
                  "catalogs.",
                  "SAFETY: totality is the only phase safe to view "
                  "unfiltered."]),
            _hyp("Annular ring of fire (alternative)", 25,
                 ["An annular eclipse leaves a thinner, even ring of "
                  "sunlight"]),
        ]
        report["summary"] = (
            "The frame shows a luminous corona around a dark lunar "
            "silhouette - the signature of a total solar eclipse.")
        return report

    # ---- aurora: green curtains, usually border-touching --------------------
    # scene-level metrics, NOT the main-source box: the glow reads as
    # "background" to the source masks and the box never contains it
    _mrgb = _ssrc.get("mean_rgb") or [0, 0, 0]
    _gfrac = features.get("green_glow_frac", 0)
    if (report["mode"] == "probabilistic" and _gfrac > 0.04
            and features.get("green_border")
            and (features.get("green_curtain_ratio", 0) > 2.0
                 or _gfrac > 0.12
                 or features.get("nightscape_horizon_y") is not None)):
        report["headline"] = "Probabilistic analysis: aurora (auroral emission)"
        report["hypotheses"] = [
            _hyp("Aurora (oxygen 557.7 nm green emission)", 82,
                 [f"GREEN-dominant glow covers {_gfrac * 100:.0f}% of the "
                  "frame - atomic oxygen's auroral line, not a celestial "
                  "color",
                  ("Vertical curtain structure detected"
                   if features.get("green_curtain_ratio", 0) > 2.0 else
                   "The glow spreads from the frame edge like a curtain, "
                   "unlike any compact celestial source")],
                 ["Red/purple fringes appear at high altitudes during strong "
                  "storms.",
                  "A comet's green coma is COMPACT with a condensed core - "
                  "this glow is diffuse and anchored to the horizon."]),
            _hyp("Airglow band (alternative)", 25,
                 ["Faint green airglow forms smooth bands but lacks curtain "
                  "structure"]),
        ]
        report["summary"] = (
            "The green, curtain-like glow matches auroral oxygen emission - "
            "an atmospheric light show, not a deep-sky object.")
        return report

    # ---- the Milky Way band -------------------------------------------------
    # scene-level: heavy smoothing melts the stars into their diffuse glow;
    # a band-shaped surplus crossing the frame + a rich star field is the
    # galaxy's disk seen from inside (the main source is often just one
    # bright star, so main-source shape tests never fire)
    # guard: a frame-filling GALAXY also leaves an elongated smooth band -
    # in true Milky Way panoramas the band is background glow, never the
    # dominant compact source (M31's fill measures 0.29+, panoramas < 0.2)
    if (report["mode"] == "probabilistic"
            and features.get("diffuse_band")
            and features.get("star_count", 0) >= 800
            and _ssrc.get("fill_fraction", 1.0) < 0.20):
        report["headline"] = ("Probabilistic analysis: the Milky Way "
                              "(galactic band)")
        report["hypotheses"] = [
            _hyp("The Milky Way's band", 80,
                 [f"A broad diffuse band crosses the whole frame "
                  f"({features['star_count']} resolved stars riding on it)",
                  "Dark dust lanes and star clouds along a band this wide "
                  "belong to our own galaxy's disk seen edge-on from inside"],
                 ["Plate solving the star field names the exact region "
                  "(e.g. Cygnus, Sagittarius)."]),
            _hyp("Large diffuse nebula (alternative)", 25,
                 ["A single nebula rarely spans the entire frame at this "
                  "star density"]),
        ]
        report["summary"] = (
            "A frame-crossing diffuse band dense with stars - the Milky "
            "Way itself. Plate solving can name the exact constellation "
            "region.")
        return report

    # ---- the Sun by appearance ----------------------------------------------
    # A frame-dominating, intensely warm-saturated disk: the Sun is the only
    # star any camera resolves as a surface. H-alpha/EUV views are deep red
    # to orange (r/b ratio 2.5+ measured 20+ on SOHO 304A frames).
    # thresholds measured: SOHO/SDO suns rb 21-34, sat 0.95+; Mars only
    # reaches rb 2.7-3.1, sat 0.63-0.68 - a canyon between them
    if (report["mode"] == "probabilistic"
            and _ssrc.get("fill_fraction", 0) > 0.30
            and _ssrc.get("rb_color_ratio", 0) >= 6.0
            and _ssrc.get("saturation", 0) >= 0.85
            # a red planetary nebula close-up passed every color test on the
            # stress bench; but it carries THOUSANDS of interior points where
            # the excluded solar disk carries almost none
            and features.get("star_count", 0) < 300):
        report["headline"] = "Probabilistic analysis: the Sun (appearance match)"
        report["object"] = {
            "name": "The Sun (likely)",
            "type_label": "Star - our own",
            "type_note": ("A resolved stellar surface in an intensely "
                          "warm-saturated narrowband palette (H-alpha/EUV). "
                          "No other star resolves as a disk at any camera."),
        }
        report["hypotheses"] = [
            _hyp("The Sun (H-alpha / EUV solar image)", 86,
                 [f"A single disk covers {_ssrc.get('fill_fraction', 0) * 100:.0f}% "
                  "of the frame in a deeply saturated warm palette "
                  f"(R/B ratio {_ssrc.get('rb_color_ratio', 0):.1f})",
                  "Granular surface texture and limb structures (prominences) "
                  "are solar-imaging signatures",
                  "The Sun is the only star whose surface any camera resolves"],
                 ["Solar telescope images are false-color by nature; the exact "
                  "wavelength (H-alpha, 304 A, 171 A...) can't be recovered "
                  "from the photo alone."]),
            _hyp("A red giant star (alternative)", 10,
                 ["No other star resolves as a surface - a red giant appears "
                  "only as a point in any real photograph"]),
        ]
        report["summary"] = (
            "The frame is filled by a single resolved stellar disk in an "
            "intensely warm palette with granular texture - the visual "
            "signature of solar imaging. This is almost certainly the Sun.")
        for e in entries:
            if e.get("kind") in ("galaxy", "nebula", "main"):
                e["label"], e["kind"] = "The Sun (likely) - resolved stellar disk", "body"
                break
        return report

    # ---- the Moon by appearance (nightscapes included) -----------------------
    # A resolved, near-white, sharply-bounded disk on dark sky is the Moon in
    # any camera. Detected independently of main-source selection, because a
    # city-glow band once stole the main-source slot and the huge obvious
    # Moon went unmentioned ("satellite trail").
    _moon = features.get("moon_disk")
    _hz = features.get("nightscape_horizon_y")
    # without a horizon, the moon candidate must BE the main subject: a
    # saturated round star blob inside a rich deep-sky frame (Antennae crop,
    # 3287 stars) passed the size gate as a 98 px "Moon" while the actual
    # subject held 100x its light two hundred pixels away
    _moon_is_main = bool(_moon and _ssrc and (
        ((_moon["x"] - _ssrc.get("center_x", 1e9)) ** 2
         + (_moon["y"] - _ssrc.get("center_y", 1e9)) ** 2) ** 0.5
        < _moon["diameter_px"] * 1.5))
    # no-horizon arm, three physics gates (all measured):
    # - CORE size, not ext: atmospheric glow inflated a real Moon's ext to
    #   444 px around a 129 px disk and the old equiv gate rejected it as
    #   "Gas giant"; the bright core (200 px) is what must match the disk
    # - sparse sky: a bright-Moon exposure cannot record a deep star field
    #   (M78's blue nebula core passed every disk test but sat in a
    #   1,323-source frame; real Moon frames measured 0-627)
    # - lunar color: the Moon is never blue (M78 rb 0.764 vs Moon 0.9-1.5)
    _moon_core_ok = bool(_moon and (
        (_ssrc.get("bright_diameter_px") or _ssrc.get("equiv_diameter_px", 0))
        < _moon["diameter_px"] * 3))
    _sky_sparse = (features.get("star_count", 0)
                   + features.get("faint_star_count", 0)) < 900
    _lunar_color = (_ssrc.get("rb_color_ratio") or 1.0) >= 0.85
    # the streak arm needs the sparse-sky gate too: the Pillars of Creation
    # (7,078 point sources) reached this branch because its dust columns
    # measured as elongated "streaks", and came back as "the Moon". The
    # HORIZON arm is exempt - a detected horizon is itself proof of a
    # ground-based landscape (a real moonscape measured 3,280 points over
    # its skyline), so it only carries a far ceiling against deep frames.
    _deep_field = (features.get("star_count", 0)
                   + features.get("faint_star_count", 0)) > 6000
    if (report["mode"] == "probabilistic" and _moon
            and ((_hz is not None and not _deep_field)
                 or (_ssrc.get("is_streak") and _sky_sparse)
                 or (_moon_is_main and _moon_core_ok
                     and _sky_sparse and _lunar_color))):
        scene = ("over a night landscape (city lights below the horizon)"
                 if _hz is not None else "on the night sky")
        report["headline"] = ("Probabilistic analysis: the Moon "
                              "(appearance match)")
        report["object"] = {
            "name": "The Moon (likely)",
            "type_label": "Earth's Moon",
            "type_note": (f"A resolved, colorless, sharply-bounded disk "
                          f"({_moon['diameter_px']:.0f} px) with an empty "
                          f"dark surround, {scene} - the Moon's signature "
                          "look in any camera."),
        }
        report["hypotheses"] = [
            _hyp("The Moon", 87,
                 [f"Saturated near-white disk of {_moon['diameter_px']:.0f} px "
                  "with a clean circular edge and a dark, empty surround",
                  "No other body shows a resolved colorless disk this bright "
                  "to ordinary cameras"],
                 ["The illuminated phase and exact position need the capture "
                  "time (EXIF) to confirm against the lunar ephemeris.",
                  "A strongly defocused bright star or planet can mimic a "
                  "small disk - the sharp edge argues against that here."]),
            _hyp("Bright planet, heavily overexposed (alternative)", 15,
                 ["Venus or Jupiter can bloom into a small disk when "
                  "overexposed, though without this sharp an edge"]),
        ] + ([_hyp("Night landscape scene", 60,
                   ["City lights and terrain detected below the skyline - "
                    "the analysis above covers only the sky region"])]
             if _hz is not None else [])
        report["summary"] = (
            f"The frame holds a resolved lunar disk {scene}. "
            "The sky region was analyzed separately from the landscape.")
        return report

    # ---- comet by appearance -------------------------------------------------
    # A soft GREEN-dominant coma is the diatomic-carbon (C2/CN) glow no other
    # common object shows: nebulae are red/blue/teal, galaxies colorless.
    # (A green comet was being reported as "nebula".)
    # A very smooth coma can miss the fuzzy flag, so a compact round source
    # (small fill, no elongation) qualifies too.
    _coma_shape = (_ssrc.get("is_extended_fuzzy")
                   or (_ssrc.get("elongation", 9) <= 1.6
                       and _ssrc.get("fill_fraction", 0) < 0.18))
    if (report["mode"] == "probabilistic" and _ssrc and _coma_shape
            and _ssrc.get("fill_fraction", 0) < 0.25
            and _ssrc.get("border_contact", 0) < 0.08
            and features.get("green_glow_frac", 0) < 0.05
            and features.get("nightscape_horizon_y") is None
            and _ssrc.get("mean_rgb")):
        _mr, _mg, _mb = _ssrc["mean_rgb"]
        if _mg > _mr * 1.12 and _mg > _mb * 1.12 and _ssrc.get("saturation", 0) >= 0.15:
            report["headline"] = "Probabilistic analysis: comet (green coma)"
            report["hypotheses"] = [
                _hyp("Comet (diatomic-carbon green coma)", 78,
                     [f"A soft, compact glow whose color is GREEN-dominant "
                      f"(G/R {_mg / max(_mr, 1e-3):.2f}, G/B {_mg / max(_mb, 1e-3):.2f}) - "
                      "the C2/CN emission signature unique to comet comas",
                      "Round, diffuse envelope around a condensed core - a coma, "
                      "not a resolved nebula"],
                     ["Naming the specific comet requires the capture date: comets "
                      "move against the stars daily, so a sky position alone dates "
                      "rather than names them.",
                      "A tail may or may not be visible depending on geometry."]),
            ] + [h for h in report["hypotheses"] if "nebula" in h["label"].lower()][:1]
            report["summary"] = (
                "The dominant source is a compact, green-glowing coma - the "
                "diatomic-carbon emission that only comets show. The starfield "
                "behind it can still be plate-solved to date the sighting.")
            return report

    # ---- planetary pair: Earth + Moon by appearance -------------------------
    # Two solid bodies on an empty sky, the larger carrying Earth's unique
    # blue-ocean + white-cloud signature, the companion neutral gray: no other
    # resolvable pairing looks like this. Without the check, the famous
    # Mars-orbit Earth+Moon photo came back as a "galaxy group".
    dim = float(max(features.get("image_width", 1), features.get("image_height", 1)))
    if (report["mode"] == "probabilistic" and 2 <= len(entries) <= 3
            and all(e["peak"] >= 0.25 for e in entries)
            # empty sky, OR a frame-dominating cloud-covered body (grainy old
            # scans count thousands of compression specks as "stars", e.g.
            # Galileo's Earth-Moon portrait measured star_count 2585)
            and (features.get("star_count", 0) < 40
                 or (entries[0]["equiv_diameter_px"] / dim >= 0.45
                     and entries[0]["white_fraction"] >= 0.05))):
        big = entries[0]
        grays = [e for e in entries[1:]
                 if e["saturation"] < 0.30 and e["blue_fraction"] < 0.05]
        if big["blue_fraction"] >= 0.15 and grays:
            moon = grays[0]
            report["headline"] = ("Probabilistic analysis: Earth and the Moon "
                                  "(appearance match)")
            report["object"] = {
                "name": "Earth and the Moon (likely)",
                "type_label": "Planetary pair",
                "type_note": ("Appearance-based identification: a blue planet with "
                              "white cloud patches next to a smaller neutral-gray "
                              "companion - the Earth-Moon system's signature look."),
            }
            report["hypotheses"] = [
                _hyp("Earth and the Moon (planetary pair)", 85,
                     [f"Two solid bodies on an otherwise empty sky "
                      f"({features.get('star_count', 0)} faint points only) - "
                      "a planetary scene, not a deep-sky field",
                      f"The larger body is blue with white cloud cover "
                      f"({big['blue_fraction'] * 100:.0f}% blue-ocean pixels, "
                      f"{big['white_fraction'] * 100:.0f}% cloud-white) - Earth is "
                      "the only known planet with this signature",
                      f"The companion is neutral gray (saturation "
                      f"{moon['saturation']:.2f}) - exactly how the Moon appears"],
                     ["Identification is appearance-based; the capture position "
                      "(e.g. photographed from Mars orbit) cannot be recovered "
                      "without EXIF or mission metadata."]),
                _hyp("Another planet-moon pairing (alternative)", 20,
                     ["No other solar-system pairing shows a blue-white planet "
                      "with a single large gray companion at this size ratio"]),
            ]
            report["summary"] = (
                "The frame holds two solid bodies: a blue planet with white "
                "clouds and a smaller neutral-gray companion. That combination "
                "matches one thing in the known sky - Earth and its Moon.")
            big["label"], big["kind"] = "Earth (likely) - blue planet with clouds", "body"
            moon["label"], moon["kind"] = "The Moon (likely) - neutral gray companion", "body"
            for e in entries:
                if e is not big and e is not moon and e["kind"] == "galaxy":
                    e["kind"], e["label"] = "compact", ("Compact source - "
                                                        "unidentified companion")
            return report

    # ---- Earth alone by appearance ------------------------------------------
    # A resolved disk that is ocean-blue with white cloud cover and no
    # latitudinal banding: no other known body shows this combination, so the
    # single-disk case deserves the same certainty as the pair above. NO
    # star-count guard on purpose: press composites paste Earth onto artificial
    # starfields, and those stars must not demote the obvious planet.
    _esrc = features.get("main_source") or {}
    _e0 = (features.get("sources") or [{}])[0]
    if (report["mode"] == "probabilistic" and _esrc.get("is_disk_like")
            and _e0.get("blue_fraction", 0) >= 0.25
            and _e0.get("white_fraction", 0) >= 0.05
            and _esrc.get("band_contrast", 1) < 0.22
            and _esrc.get("saturation", 1) < 0.40):
        report["headline"] = "Probabilistic analysis: Earth (appearance match)"
        report["object"] = {
            "name": "Earth (likely)",
            "type_label": "Planet - our own",
            "type_note": ("Appearance-based identification: a resolved disk "
                          "that is ocean-blue with white cloud patches and no "
                          "latitudinal banding - the Earth's signature look."),
        }
        report["hypotheses"] = [
            _hyp("Earth (blue oceans + white clouds)", 88,
                 [f"{_e0.get('blue_fraction', 0) * 100:.0f}% of the disk is "
                  f"ocean-blue and {_e0.get('white_fraction', 0) * 100:.0f}% "
                  "cloud-white, with no latitudinal banding "
                  f"(band contrast {_esrc.get('band_contrast', 0):.2f}) - "
                  "Earth is the only known body with this combination",
                  "Continents, cloud swirls and the terminator are surface "
                  "features of one body - not separate objects"],
                 ["If this is a processed or artist-composited image, the "
                  "color argument applies to the Earth imagery it contains.",
                  "Ice giants are blue but carry no white cloud patches; "
                  "Jupiter's blue-white UV views are strongly banded."]),
            _hyp("Another cloud-covered planet (alternative)", 8,
                 ["No other known planet combines ocean-blue color with "
                  "patchy white clouds at zero banding"]),
        ]
        report["summary"] = (
            "The frame is dominated by a resolved planetary disk that is "
            "ocean-blue with white cloud cover and no banding. That "
            "combination matches exactly one known body: Earth.")
        for e in entries:
            if e.get("kind") in ("galaxy", "nebula", "main", "body", "compact"):
                e["label"], e["kind"] = "Earth (likely) - blue planet with clouds", "body"
                break
        return report

    # compact galaxy group: >=3 clearly-resolved galaxies is its own verdict -
    # without this, HCG 40's five galaxies merged into one blob and the frame
    # came back as a single "star-forming region"
    gals = [e for e in entries if e["kind"] == "galaxy"]
    # group members must be COMPARABLE in size: real compact groups measure
    # 33-72% of their biggest member, while a shredded nebula yields one
    # dominant blob plus 7-8% knots (the Crab became a "group of 7")
    if len(gals) >= 3:
        _dmax = max(g["equiv_diameter_px"] for g in gals)
        _peers = sum(1 for g in gals
                     if g["equiv_diameter_px"] >= 0.25 * _dmax) - 1
        if _peers < 2:
            gals = []
    if len(gals) >= 3 and report["mode"] == "probabilistic":
        n = len(gals)
        sizes = ", ".join(f"{g['equiv_diameter_px']:.0f}" for g in gals[:5])
        report["headline"] = (f"Probabilistic analysis: compact group of "
                              f"{n} interacting galaxies")
        report["hypotheses"] = [
            _hyp(f"Compact galaxy group ({n} galaxies resolved)", 88,
                 [f"{n} separate large, soft-profiled sources resolved in one "
                  f"frame ({sizes} px across) - each individually shaped like "
                  "a galaxy (smooth ellipticals, inclined disks, an edge-on)",
                  "Their tight mutual spacing is the signature of a compact "
                  "group - gravitationally bound and often interacting"],
                 ["Compact groups like Stephan's Quintet or the Hickson catalog "
                  "members eventually merge into a single giant galaxy.",
                  "Naming the specific group requires plate solving."]),
            _hyp("Chance line-of-sight alignment (alternative)", 25,
                 ["Unrelated galaxies at different distances can appear close "
                  "together, though a grouping this tight is statistically rare"]),
        ]
        report["summary"] = (
            f"The frame resolves {n} clearly separated galaxies at close mutual "
            "spacing - most consistent with a compact interacting galaxy group. "
            "Each member is listed under Detected Sources with its own shape-based "
            "classification.")
    elif len(entries) >= 1 and report["hypotheses"]:
        # single-object frames: the source list should agree with the verdict,
        # so the entry containing the main source inherits the top hypothesis
        src = features.get("main_source")
        if src:
            top = (report["object"]["name"] if report.get("object")
                   else report["hypotheses"][0]["label"])
            for e in entries:
                if (abs(e["center_x"] - src["center_x"]) <= max(src.get("major_axis_px", 0), 60) / 2
                        and abs(e["center_y"] - src["center_y"]) <= max(src.get("major_axis_px", 0), 60) / 2
                        and not e.get("identified")):
                    e["label"] = f"Main source - {top}"
                    e["kind"] = "main"
                    break
    _rank_hypotheses(report)
    return report


# hypotheses that cannot describe the same object at the same time. Key words
# are matched against the leading hypothesis; the listed rivals are then
# dropped outright rather than merely demoted.
_INCOMPATIBLE = [
    # a lunar disk is not a gas giant: banded ammonia cloud decks are a
    # Jupiter/Saturn feature, and lunar maria measure as "bands" by accident
    (("moon",), ("gas giant",)),
    # ... and the reverse
    (("gas giant",), ("moon",)),
]


def _rank_hypotheses(report: dict) -> None:
    """Turn independently-scored class fits into a ranked verdict.

    Each hypothesis is scored on its own merits ("how well does this class
    match the measurements?"), which is right for the physics but wrong for
    the reader: a blood Moon at 97 sat beside "gas giant" at 70 and BOTH
    printed STRONG, as though the frame held four objects at once. The leader
    now caps its rivals - the more decisive it is, the harder they are capped
    - and physically impossible rivals are removed entirely.
    """
    hyps = report.get("hypotheses") or []
    if len(hyps) < 2:
        return
    hyps = sorted(hyps, key=lambda h: -h.get("score", 0))
    lead = (hyps[0].get("label") or "").lower()
    for keys, rivals in _INCOMPATIBLE:
        if any(k in lead for k in keys):
            hyps = [hyps[0]] + [h for h in hyps[1:]
                                if not any(r in (h.get("label") or "").lower()
                                           for r in rivals)]
    top = hyps[0].get("score", 0)
    # a near-certain leader leaves little room for anything else; a weak one
    # leaves room for a genuine rival (honest ambiguity must survive)
    ratio = 0.55 if top >= 90 else (0.72 if top >= 75 else 0.88)
    cap = top * ratio
    for h in hyps[1:]:
        if h.get("score", 0) > cap:
            h["score"] = round(cap)
            h["band"] = _band(h["score"])
    report["hypotheses"] = hyps


def _analyze_core(features: dict, solve: dict, catalog_matches: list[dict],
                  nasa_image: dict | None, exif: dict, sky: dict | None = None) -> dict:
    sky = sky or {}
    report = {
        "mode": "probabilistic",
        "headline": "",
        "summary": "",
        "object": None,
        "field_objects": [],
        "hypotheses": [],
        "notable_sources": _notable_sources(features),
        "observations": _observations(features, solve, exif, sky),
        "caveats": _caveats(features, solve, exif, sky),
        "science_notes": SCIENCE_NOTES,
        "sky_context": sky,
    }

    # ---------- gate: is this even a night-sky image? ----------
    # Unless plate solving actually located it on the sky (definitive proof it
    # IS astronomical), reject images that clearly aren't astrophotos so the
    # engine never confidently calls a screenshot or a daytime photo a nebula.
    if not solve.get("solved") and not features.get("looks_astronomical", True):
        reasons = features.get("plausibility_reasons") or []
        report["mode"] = "not_astronomical"
        report["headline"] = "This does not look like an astronomical image"
        report["identity_status"] = "not_astronomical"
        report["hypotheses"] = [_hyp(
            "Not an astronomical image", 88,
            ["The app analyses photos of the night sky and celestial objects"]
            + [r.capitalize() for r in reasons],
            ["If this really is an astronomy image (for example an unusual false-color "
             "render), it may still be analysable - but the app won't guess a celestial "
             "type from content that looks like a screenshot, a daytime photo or artwork."],
        )]
        report["notable_sources"] = []
        report["observations"] = [
            f"Median brightness {features.get('median_luminance', 0):.2f} "
            f"(dark sky is typically < 0.05).",
            f"Dark-sky coverage {features.get('dark_fraction', 0) * 100:.0f}% of the frame.",
            f"Vividly colored area {features.get('vivid_fraction', 0) * 100:.0f}% of the frame.",
        ]
        report["caveats"] = [
            "The app deliberately refuses to classify non-astronomical images rather than "
            "produce a confident but meaningless answer.",
        ]
        report["summary"] = ("This image doesn't appear to be a photograph of the night sky. "
                             "Upload a photo of a celestial object (a star field, the Moon, a "
                             "planet, a galaxy or a nebula) for analysis.")
        return report

    # ---------- gate: terrestrial night scene ----------
    # A lamp-lit street/indoor photo passes the darkness gates, but its lit
    # ground/walls BLEED OFF-FRAME (25-44% border contact measured) while real
    # deep-sky objects are framed in dark sky (0% across the whole test set).
    _src = features.get("main_source") or {}
    # CNN exception: a close-up of a bright nebula (Egg Nebula fills its whole
    # frame, border contact ~100%) is geometrically identical to lamp glow
    # bleeding off-frame - but the CNN separates them cleanly (nebula 0.99 vs
    # street/render traps <=0.06), so a confident CNN vetoes this gate.
    if (not solve.get("solved") and _src.get("is_extended_fuzzy")
            and _src.get("border_contact", 0) > 0.12
            and features.get("star_count", 0) < 50
            and (features.get("astro_probability") or 0.0) < 0.9):
        bc = _src.get("border_contact", 0)
        report["mode"] = "not_astronomical"
        report["headline"] = "This looks like a nighttime photo of a terrestrial scene"
        report["hypotheses"] = [_hyp(
            "Terrestrial night scene (artificial lighting)", 85,
            [f"The main bright region runs off the edge of the frame "
             f"({bc * 100:.0f}% of the image border) - deep-sky objects sit framed "
             "in dark sky, while lamp-lit ground, walls or foreground bleed off-frame",
             f"Only {features.get('star_count', 0)} star-like points detected - a real "
             "night-sky exposure this dark would show a rich star field",
             "Bright compact sources in such scenes are typically artificial lights"],
            ["If this is genuinely an astronomical image (e.g. a very unusual framing), "
             "crop it so the celestial object sits inside a dark-sky frame and re-upload."],
        )]
        report["notable_sources"] = []
        report["observations"] = [
            f"Main bright region covers {bc * 100:.0f}% of the image border.",
            f"{features.get('star_count', 0)} reliable point sources detected.",
        ]
        report["caveats"] = [
            "The app deliberately refuses to classify non-astronomical images rather than "
            "produce a confident but meaningless answer.",
        ]
        report["summary"] = ("This appears to be a nighttime photograph of a terrestrial "
                             "scene lit by artificial lights, not an image of the sky. "
                             "Upload a photo of a celestial object for analysis.")
        return report

    # ---------- exact solar-system identification (ephemeris + solved field) ----------
    ss = sky.get("solar_system_matches") or []
    if solve.get("solved") and ss:
        body = ss[0]
        report["mode"] = "identified"
        report["headline"] = f"Identified: {body['body']} (ephemeris match)"
        report["object"] = {
            "name": body["body"],
            "type_label": "Solar-system body",
            "type_note": body["fact"],
            "ra": solve["ra"], "dec": solve["dec"],
        }
        if nasa_image:
            report["object"]["nasa_image"] = nasa_image
        # "Jupiter (planet)" - the identified NAME leads, its kind follows
        _kind = {"Moon": "moon", "Sun": "star"}.get(body["body"], "planet")
        report["hypotheses"] = [_hyp(
            f"{body['body']} ({_kind})", 96,
            [f"Plate-solved position matches {body['body']}'s computed ephemeris position "
             f"to {body['separation_deg']:.2f} deg at the EXIF capture time",
             "Astrometry + ephemeris cross-check is the gold-standard identification method"],
            [body["fact"]],
        )]
        report["field_objects"] = [{"name": m.get("name", ""), "type_label": m.get("type_label")}
                                   for m in catalog_matches[:6]]
        report["summary"] = _summary(report, features, solve, sky)
        return report

    # ---------- identified deep-sky object ----------
    if solve.get("solved"):
        named = [m for m in catalog_matches if m.get("name")]
        annotations = solve.get("annotations") or []

        # a primary object must be EARNED: an extended cataloged object that
        # OVERLAPS the dominant source in the image, or the identified star
        # sitting exactly on it. Never crown a random object elsewhere in the
        # field (a sparse 9-degree "cluster" is context, not identification).
        import math
        extended = [m for m in named if m.get("angular_size_arcmin")]
        primary = None
        # a visually-established identity (press-avm / pattern lock) is
        # pixel-verified: it wins outright. Without this, a point-like
        # exotic (rogue planet, brown dwarf - no angular size) loses the
        # headline to whatever bright field star sits on the main source.
        for m in named:
            if m.get("identity"):
                primary = m
                break
        m_ra, m_dec = solve.get("main_source_ra"), solve.get("main_source_dec")
        if primary is None and extended and m_ra is not None:
            overlapping = []
            for m in extended:
                cosd = math.cos(math.radians(m_dec))
                sep_deg = math.hypot((m["ra"] - m_ra) * cosd, m["dec"] - m_dec)
                reach = max(m["angular_size_arcmin"] / 60.0 * 0.75, 0.35)
                if sep_deg <= reach:
                    overlapping.append((sep_deg, m))
            if overlapping:
                # normalize separation by object SIZE: with M31 in frame, a
                # tiny embedded cluster a hair closer to the centroid must not
                # outrank the galaxy that IS the picture
                overlapping.sort(
                    key=lambda t: t[0] / max((t[1].get("angular_size_arcmin")
                                              or 1.0), 1.0))
                primary = overlapping[0][1]
        elif primary is None and extended and m_ra is None:
            # no dominant-source anchor (e.g. pure DSO frame): fall back to the
            # object nearest the field center among the extended candidates
            extended.sort(key=lambda m: m.get("center_distance_deg") or 99)
            if (extended[0].get("center_distance_deg") or 99) < \
                    max((solve.get("radius_deg") or 1) * 0.5, 0.3):
                primary = extended[0]
        src = features.get("main_source")
        if primary is None and src:
            import math
            best = None
            for s in features.get("stars", []):
                if not s.get("id"):
                    continue
                d = math.hypot(s["x"] - src["center_x"], s["y"] - src["center_y"])
                if d <= max(src["equiv_diameter_px"], 20) and (best is None or d < best[0]):
                    best = (d, s)
            if best:
                i = best[1]["id"]
                primary = {
                    "name": i["name"],
                    "type_label": i.get("type_label", "Star"),
                    "type_note": (i.get("type_note") or "")
                    + " Identified as the dominant source in this frame.",
                    "spectral_type": i.get("spectral_type"),
                    "distance_ly": i.get("distance_ly"),
                    "v_mag": i.get("v_mag"),
                    "ra": best[1].get("ra"), "dec": best[1].get("dec"),
                }

        if primary:
            report["mode"] = "identified"
            report["headline"] = f"Identified: {primary['name']} - {primary['type_label']}"
            obj = dict(primary)
            from .catalogs import estimate_physical_size
            size_ly = estimate_physical_size(
                obj.get("angular_size_arcmin"),
                obj.get("distance_ly") or ((obj.get("distance_mly") or 0) * 1e6 or None),
            )
            if size_ly:
                obj["physical_size_ly"] = size_ly
            if nasa_image:
                obj["nasa_image"] = nasa_image
            report["object"] = obj
            report["field_objects"] = named[1:8]
            report["hypotheses"] = [_hyp(
                f"{primary['name']} ({primary['type_label']})", 95,
                ["Star pattern matched against all-sky catalogs (plate solving) - "
                 "near-zero false-positive rate",
                 f"Solved field center: RA {solve['ra']:.3f}, Dec {solve['dec']:.3f} "
                 f"(field radius {solve.get('radius_deg', 0):.2f} deg)"],
                [primary.get("type_note") or ""],
            )]
            report["summary"] = _summary(report, features, solve, sky)
            return report

        # A solved frame with no primary catalog object at its center can
        # still have an unmistakable subject: the brightest detection matched
        # to a naked-eye star (a phone shot of Vega solved perfectly, then
        # reported "Specific object: Not determined" - nonsense to the user).
        beacon = None
        for s in features.get("stars") or []:
            i = s.get("id")
            if not i or i.get("v_mag") is None or i["v_mag"] > 3.0:
                continue
            if (i.get("name") or "").startswith(("Gaia", "PS1", "DES", "HSC")):
                continue
            if beacon is None or (s.get("flux") or 0) > (beacon[0].get("flux") or 0):
                beacon = (s, i)
        if beacon is not None:
            bs, bi = beacon
            report["mode"] = "identified"
            report["headline"] = f"Field solved - brightest star: {bi['name']}"
            report["object"] = {
                "name": bi["name"],
                "type_label": bi.get("type_label"),
                "type_note": bi.get("type_note"),
                "spectral_type": bi.get("spectral_type"),
                "v_mag": bi.get("v_mag"),
                "distance_ly": bi.get("distance_ly"),
                "ra": bs.get("ra"), "dec": bs.get("dec"),
            }
            report["field_objects"] = named[:8]
            report["hypotheses"] = [_hyp(
                f"{bi['name']} - brightest star in the solved field", 92,
                [f"Astrometric solution at RA {solve['ra']:.3f}, "
                 f"Dec {solve['dec']:.3f} "
                 f"(field radius {solve.get('radius_deg', 0):.2f} deg)",
                 f"V magnitude {bi.get('v_mag')} - the dominant point source "
                 "in this frame"],
                [bi.get("type_note") or ""],
            )]
            report["summary"] = _summary(report, features, solve, sky)
            return report

        if annotations:
            names = [", ".join(a["names"]) for a in annotations[:6] if a.get("names")]
            report["mode"] = "identified"
            report["headline"] = "Field solved - known objects located in the image"
            report["field_objects"] = [{"name": n} for n in names]
            report["hypotheses"] = [_hyp(
                "Solved star field containing cataloged objects", 90,
                [f"Astrometric solution at RA {solve['ra']:.3f}, Dec {solve['dec']:.3f}",
                 f"Objects annotated by astrometry.net: {'; '.join(names) or 'stars only'}"],
            )]
            report["summary"] = _summary(report, features, solve, sky)
            return report

        const = sky.get("constellation")
        report["mode"] = "identified"
        report["headline"] = (f"Field solved: star field in {const}" if const
                              else "Field solved: star field")
        report["field_objects"] = named[:8]
        solved_hyp = _hyp(
            "Solved star field" + (f" in {const}" if const else ""), 92,
            [f"Astrometric solution at RA {solve['ra']:.3f}, Dec {solve['dec']:.3f} "
             f"(field radius {solve.get('radius_deg', 0):.2f} deg)",
             "Hover the gold-ringed stars in the viewer for their identities"],
        )
        report["hypotheses"] = [solved_hyp] + (_classify(features, sky) or [])
        report["summary"] = _summary(report, features, solve, sky)
        return report

    # ---------- ringed planet = Saturn (appearance identification) ----------
    # The ring-gap profile (bright ring - dark gap - wide planet plateau -
    # gap - ring, symmetric) is unique: no galaxy, nebula or cluster in the
    # measurement set reproduces it. And among objects a camera can resolve,
    # only Saturn shows rings this prominent - Jupiter's ring is invisible
    # and Uranus/Neptune need professional IR. So the geometry alone names it.
    # A frame that resolves Saturn's rings is a close-up, so its sky is
    # nearly empty - a rich starfield means the "ring" is galaxy structure
    # (a dust-lane galaxy like Centaurus A can mimic the profile).
    _rsrc = features.get("main_source") or {}
    if _rsrc.get("is_ringed_disk") and features.get("star_count", 0) < 200:
        report["mode"] = "identified"
        # WHICH ringed planet? Classic anchored detections (rings outshine
        # the disk, elongation 2.3-2.6) are Saturn - the only ringed planet
        # any optical setup resolves. Wide-sweep detections mark the
        # OPPOSITE morphology: a saturated near-white disk outshining its
        # rings inside a big glow halo - that is how JWST renders Neptune
        # and Uranus in the infrared, and measured colors cannot split them
        # from Saturn (rb 1.01 vs 1.07-1.15). Naming Saturn there once put
        # "Saturn" on the famous JWST Neptune frame - a wrong name. The
        # qualified label is the honest one.
        _ice_ir = bool(_rsrc.get("ring_wide_sweep"))
        if _ice_ir:
            report["headline"] = ("Identified by appearance: ringed planet "
                                  "(Saturn or an ice giant in infrared)")
            report["object"] = {
                "name": "Ringed planet",
                "type_label": "Planet with a prominent ring system",
                "type_note": ("The ring-gap brightness signature is "
                              "unmistakable, but this frame's morphology - "
                              "a saturated, neutral-white disk outshining "
                              "its rings inside a wide glow halo - matches "
                              "professional infrared imaging (JWST), where "
                              "Neptune and Uranus show rings just as "
                              "prominent as Saturn's. Pixels alone cannot "
                              "say which planet this is; the capture time "
                              "(EXIF) would settle it via the ephemerides."),
            }
        else:
            report["headline"] = "Identified by appearance: Saturn"
            report["object"] = {
                "name": "Saturn",
                "type_label": "Planet (gas giant with rings)",
                "type_note": ("Identified from its ring geometry: the brightness profile "
                              "along the long axis shows ring - gap - planet - gap - ring, "
                              "a pattern no other resolvable celestial object produces."),
            }
        if _ice_ir:
            report["hypotheses"] = [
                _hyp("Ringed planet (Saturn, or Neptune/Uranus in IR)", 90,
                     ["Full-circle brightness sweep found the ring-gap "
                      "signature: symmetric bright rings flanking a wide "
                      "saturated disk",
                      "Neutral-white saturated disk inside a broad glow "
                      "halo - the morphology of professional infrared "
                      "frames, where ice-giant rings rival Saturn's"],
                     ["Without the capture time no photograph can name "
                      "WHICH ringed planet this is - Saturn, Neptune and "
                      "Uranus all produce this geometry at these scales.",
                      "Point sources near the rings are most likely the "
                      "planet's moons."]),
                _hyp("Edge-on galaxy with a dust lane (alternative)", 12,
                     ["An edge-on galaxy declines smoothly from the centre "
                      "- it cannot produce two symmetric dark gaps between "
                      "separate bright components"]),
            ]
            report["summary"] = (
                "The main source shows the ring-gap brightness signature of "
                "a ringed planet: bright symmetric rings separated from a "
                "wide saturated disk by dark gaps. The neutral-white disk "
                "and broad glow halo match professional infrared imaging, "
                "where Neptune and Uranus show rings as prominent as "
                "Saturn's - so the exact planet honestly cannot be named "
                "from pixels alone. The capture time (EXIF) would settle it.")
            return report
        report["hypotheses"] = [
            _hyp("Saturn (planet with a prominent ring system)", 93,
                 ["Major-axis brightness profile shows the ring-gap signature: bright "
                  "rings flanking a wide planetary disk with dark gaps between them "
                  f"(elongation {_rsrc.get('elongation', 0):.1f}, "
                  f"{_rsrc.get('fill_fraction', 0) * 100:.0f}% frame coverage)",
                  "Smooth, sharp-edged structure on a dark sky - a solid body, not "
                  "diffuse gas",
                  "Saturn is the only object whose rings appear like this at any "
                  "amateur or observatory scale"],
                 ["Point sources just outside the rings are most likely Saturn's moons "
                  "(Titan, Rhea, Dione...) - naming them exactly needs the capture time."]),
            _hyp("Edge-on galaxy with a dust lane (alternative)", 15,
                 ["An edge-on galaxy is elongated too, but its brightness declines "
                  "smoothly from the centre - it cannot produce two symmetric dark "
                  "gaps between separate bright components"]),
        ]
        # appearance-based identification has no astrometric solution, so the
        # generic identified-mode summary (which cites RA/Dec) does not apply
        report["summary"] = (
            "The main source shows the ring-gap brightness signature that only "
            "Saturn produces at resolvable scales: bright rings separated from a "
            "wide planetary disk by dark gaps, symmetric about the centre. "
            "Point sources just outside the rings are most likely Saturnian moons.")
        return report

    # ---------- probabilistic mode ----------
    src = features.get("main_source")
    if src is None and features["star_count"] == 0:
        report["mode"] = "indeterminate"
        report["headline"] = "No celestial source could be detected in this image"
        report["hypotheses"] = [_hyp(
            "Cannot be determined from this image", 90,
            ["No source rises significantly above the background noise"],
            ["Possible causes: extreme underexposure, heavy compression, or the image may "
             "not contain a celestial object."],
        )]
        report["summary"] = _summary(report, features, solve, sky)
        return report

    # frame_filling = the measured emission spans the whole luminous frame
    # (visible-light Pillars: the 5-sigma component is a tiny knot but the
    # nebula IS the picture) - that is a star-forming region, not a mere
    # "star field", so let it flow to the extended-source branch below
    if src is None or (features["star_count"] >= 30 and src["fill_fraction"] < 0.002
                       and not src.get("frame_filling")):
        report["hypotheses"] = _star_field_hypotheses(features)
        # headline follows the TOP hypothesis: the frame-filling-nebula guard
        # can outrank "star field" here, and the headline must not contradict
        # the 85-scored hypothesis directly beneath it
        _top_h = report["hypotheses"][0]["label"]
        report["headline"] = (
            "Probabilistic analysis: star field" if _top_h == "Star field"
            else f"Probabilistic analysis: most consistent with {_top_h}")
        # the dominant point source deserves its own assessment (e.g. a big
        # orange star standing out of the field) - it IS the "main source"
        # here, so the notable-source list would otherwise skip it
        if src is not None and src.get("bright_diameter_px", 99) < 40:
            color_name, temp_desc = _star_temperature_class(src["rb_color_ratio"])
            report["notable_sources"].insert(0, {
                "x": src["center_x"], "y": src["center_y"],
                "name": None, "url": None, "peak_snr": None,
                "text": (f"The dominant (brightest) source in this field has a striking "
                         f"{color_name} color - if a star, {temp_desc}. Strongly colored "
                         "bright points are usually cool giant stars or planets; a black "
                         "hole itself emits no visible light (only its accretion disk "
                         "could, and none is resolvable at this scale)."),
            })
        report["summary"] = _summary(report, features, solve, sky)
        return report

    report["hypotheses"] = sorted(_classify(features, sky),
                                  key=lambda h: -h["score"])
    if report["hypotheses"]:
        top = report["hypotheses"][0]
        report["headline"] = f"Probabilistic analysis: most consistent with {top['label']}"
    else:
        report["mode"] = "indeterminate"
        report["headline"] = "The source resists classification from this image alone"
        report["hypotheses"] = [_hyp("Cannot be determined from this image", 85,
                                     ["Measurements do not discriminate between object classes"])]
    report["summary"] = _summary(report, features, solve, sky)
    return report
