"""Celestial Object Analyzer - local web application entry point."""
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from . import jobs
from .config import CONFIG
from .pipeline import localcatalog, ml_gate, platesolve, platesolve_local

app = FastAPI(title="Celestial Object Analyzer")


@app.on_event("startup")
def _warm_up() -> None:
    """Preload the heavy bits (torch, the embedding model, the 26k-image
    index) in the background so the FIRST analysis doesn't silently eat a
    ~1-minute cold start - it looked like the pipeline froze after
    photometry."""
    import threading

    def _load():
        try:
            import numpy as _np

            from .pipeline import visualmatch
            if visualmatch.available():
                visualmatch.match(_np.zeros((64, 64, 3), dtype=_np.float32),
                                  top_k=1)
        except Exception:
            pass

    threading.Thread(target=_load, daemon=True).start()

STATIC_DIR = Path(__file__).resolve().parent / "static"

MAX_UPLOAD_BYTES = 50 * 1024 * 1024


@app.middleware("http")
async def no_stale_assets(request, call_next):
    """Browsers happily serve a cached app.js/style.css for hours, so a restarted
    server appears to 'ignore' fixes. This is a local single-user app - always
    revalidate the page and its assets."""
    response = await call_next(request)
    if not request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-cache, must-revalidate"
    return response


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...),
                  hint: str = Form(None),
                  hint_fov: str = Form(None)):
    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty upload.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Image larger than 50 MB.")
    fov = None
    try:
        fov = float(hint_fov) if hint_fov else None
    except ValueError:
        fov = None
    job_id = jobs.start_job(data, file.filename or "upload",
                            hint=(hint or "").strip() or None, hint_fov=fov)
    return {"job_id": job_id}


@app.get("/api/job/{job_id}")
def job_status(job_id: str):
    job = jobs.get_job(job_id)
    if job is None:
        raise HTTPException(404, "Unknown job id.")
    return job


@app.get("/api/job/{job_id}/image")
def job_image(job_id: str):
    data = jobs.get_job_image(job_id)
    if data is None:
        raise HTTPException(404, "No image for this job (yet).")
    return Response(content=data, media_type="image/jpeg")


@app.get("/api/job/{job_id}/thumb")
def job_thumb(job_id: str):
    data = jobs.get_job_thumb(job_id)
    if data is None:
        raise HTTPException(404, "No thumbnail for this job.")
    return Response(content=data, media_type="image/jpeg")


@app.get("/api/history")
def history():
    return {"jobs": jobs.list_history()}


@app.get("/api/ref-image/{fn}")
def ref_image(fn: str):
    """Serve a reference image from the local press library (the identified
    object's portrait) - strictly whitelisted filenames, no traversal."""
    import re as _re
    if not _re.fullmatch(r"PRESS_[A-Za-z0-9 ._+\-]+__\d+\.(jpg|jpeg|png)", fn):
        raise HTTPException(404, "No such reference image.")
    p = (Path(__file__).resolve().parents[1] / "data" / "visual_index"
         / "img" / fn)
    if not p.exists():
        raise HTTPException(404, "No such reference image.")
    return FileResponse(p, media_type="image/jpeg")


@app.get("/api/status")
def status():
    key = CONFIG["astrometry_api_key"]
    return {
        "astrometry_key_configured": bool(key),
        "astap_available": platesolve_local.available(),
        "openngc_available": localcatalog.available(),
        "ml_gate_available": ml_gate.available(),
        "app": "Celestial Object Analyzer",
        "version": "0.4.0",
    }


@app.get("/api/verify-key")
def verify_key():
    ok, msg = platesolve.verify_key(CONFIG["astrometry_api_key"])
    return {"ok": ok, "message": msg}


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
