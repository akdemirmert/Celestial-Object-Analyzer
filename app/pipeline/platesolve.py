"""nova.astrometry.net client: blind plate solving over the free JSON API.

Flow: login -> upload -> poll submission -> poll job -> calibration + annotations.
The service is best-effort (no SLA); every step degrades gracefully.
"""
from __future__ import annotations

import json
import time

import httpx

API_BASE = "https://nova.astrometry.net/api"


class PlateSolveResult(dict):
    """dict with .solved convenience."""

    @property
    def solved(self) -> bool:
        return bool(self.get("solved"))


def _post_json(client: httpx.Client, url: str, payload: dict, files=None,
               retries: int = 3) -> dict:
    data = {"request-json": json.dumps(payload)}
    last = None
    for attempt in range(retries):
        try:
            resp = client.post(url, data=data, files=files, timeout=60)
            resp.raise_for_status()
            return resp.json()
        except (httpx.HTTPError, json.JSONDecodeError) as exc:
            last = exc
            time.sleep(2 * (attempt + 1))
    raise last


def _get_json_retry(client: httpx.Client, url: str, retries: int = 4):
    """GET + JSON that tolerates transient disconnects (nova drops connections
    under load). Returns None if every attempt fails - the caller keeps polling
    rather than aborting the whole solve on one hiccup."""
    for attempt in range(retries):
        try:
            resp = client.get(url, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except (httpx.HTTPError, json.JSONDecodeError):
            time.sleep(2 * (attempt + 1))
    return None


def solve(image_bytes: bytes, api_key: str, timeout_s: int = 180,
          scale_hint: dict | None = None,
          progress=lambda msg: None) -> PlateSolveResult:
    result = PlateSolveResult(solved=False, error=None, skipped=False)
    if not api_key:
        result.update(skipped=True, error="No astrometry.net API key configured.")
        return result

    deadline = time.monotonic() + timeout_s
    try:
        with httpx.Client() as client:
            progress("Logging in to nova.astrometry.net...")
            login = _post_json(client, f"{API_BASE}/login", {"apikey": api_key})
            if login.get("status") != "success":
                result["error"] = f"Login failed: {login.get('errormessage', 'unknown error')}"
                return result
            session = login["session"]

            progress("Uploading image for blind plate solving...")
            payload = {
                "session": session,
                "publicly_visible": "n",
                "allow_modifications": "d",
                "allow_commercial_use": "d",
            }
            if scale_hint:
                payload.update(scale_hint)
            upload = _post_json(
                client,
                f"{API_BASE}/upload",
                payload,
                files={"file": ("upload.jpg", image_bytes, "application/octet-stream")},
            )
            if upload.get("status") != "success":
                result["error"] = f"Upload failed: {upload.get('errormessage', 'unknown error')}"
                return result
            subid = upload["subid"]

            progress("Queued on astrometry.net, waiting for a solver slot "
                     "(the public queue can take several minutes)...")
            job_id = None
            while time.monotonic() < deadline:
                sub = _get_json_retry(client, f"{API_BASE}/submissions/{subid}")
                if sub is not None:
                    jobs = [j for j in sub.get("jobs", []) if j]
                    if jobs:
                        job_id = jobs[0]
                        break
                time.sleep(4)
            if job_id is None:
                result["error"] = "Timed out waiting for the solver queue."
                return result

            progress("Solver running (matching star patterns)...")
            status = None
            while time.monotonic() < deadline:
                job = _get_json_retry(client, f"{API_BASE}/jobs/{job_id}")
                if job is not None:
                    status = job.get("status")
                    if status in ("success", "failure"):
                        break
                time.sleep(4)

            if status == "failure":
                result["error"] = ("Solver finished: no star-pattern match. "
                                   "The image likely lacks a resolvable star field.")
                return result
            if status != "success":
                result["error"] = "Timed out while the solver was running."
                return result

            progress("Solved! Fetching calibration and known objects...")
            cal = _get_json_retry(client, f"{API_BASE}/jobs/{job_id}/calibration/") or {}
            ann = _get_json_retry(client, f"{API_BASE}/jobs/{job_id}/annotations/") or {}
            annotations = ann.get("annotations", [])
            # full WCS solution (FITS header) - enables pixel <-> sky mapping
            # for per-star identification. Non-fatal if unavailable.
            wcs_fits = None
            for attempt in range(3):
                try:
                    wcs_resp = client.get(
                        f"https://nova.astrometry.net/wcs_file/{job_id}", timeout=30)
                    if wcs_resp.status_code == 200:
                        wcs_fits = wcs_resp.content
                    break
                except httpx.HTTPError:
                    time.sleep(2 * (attempt + 1))

            result.update(
                solved=True,
                wcs_fits=wcs_fits,  # bytes; jobs.py pops this before JSON serialization
                ra=cal.get("ra"),
                dec=cal.get("dec"),
                radius_deg=cal.get("radius"),
                pixscale_arcsec=cal.get("pixscale"),
                orientation_deg=cal.get("orientation"),
                annotations=[
                    {
                        "names": a.get("names", []),
                        "type": a.get("type", ""),
                        "pixel_x": a.get("pixelx"),
                        "pixel_y": a.get("pixely"),
                        "radius_px": a.get("radius"),
                    }
                    for a in annotations
                ],
            )
            return result
    except httpx.HTTPError as exc:
        result["error"] = f"Network error talking to astrometry.net: {exc}"
        return result


def verify_key(api_key: str) -> tuple[bool, str]:
    """Login-only check used at startup / from the UI."""
    if not api_key:
        return False, "No API key configured."
    try:
        with httpx.Client() as client:
            login = _post_json(client, f"{API_BASE}/login", {"apikey": api_key})
        if login.get("status") == "success":
            return True, "API key OK."
        return False, login.get("errormessage", "Login rejected.")
    except httpx.HTTPError as exc:
        return False, f"Network error: {exc}"
