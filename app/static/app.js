/* Celestial Object Analyzer - frontend (v2) */

const $ = (id) => document.getElementById(id);

const STAGE_LABELS = {
  ingest: "Ingest",
  features: "Measure",
  photometry: "Photometry",
  platesolve: "Plate Solve",
  catalogs: "Catalogs",
  ephemeris: "Ephemeris",
  analysis: "Analysis",
};

const STAGE_ICONS = { pending: "○", done: "✔", failed: "✖", skipped: "↷" };

let pollTimer = null;
let currentJob = null; // last full job payload (for overlay + export)
let analysisActive = false; // true from upload until "Analyze another image"
let aladinLoaded = false;

/* ============================ navigation ============================ */

function showMainView() {
  $("methodology-section").classList.add("hidden");
  if (analysisActive) {
    $("upload-section").classList.add("hidden");
    $("analysis-section").classList.remove("hidden");
    // canvases render at 0x0 while hidden - redraw now that we're visible
    setTimeout(() => {
      drawOverlay();
      if (currentJob) renderPhotometry(currentJob.photometry || {});
    }, 50);
  } else {
    $("analysis-section").classList.add("hidden");
    $("upload-section").classList.remove("hidden");
  }
}

$("nav-methodology").addEventListener("click", () => {
  $("upload-section").classList.add("hidden");
  $("analysis-section").classList.add("hidden");
  $("methodology-section").classList.remove("hidden");
});
$("methodology-back").addEventListener("click", showMainView);
$("nav-home").addEventListener("click", showMainView);
$("nav-new").addEventListener("click", () => {
  resetUI();
});

/* ============================ upload wiring ============================ */

const dropzone = $("dropzone");
const fileInput = $("file-input");

dropzone.addEventListener("click", () => fileInput.click());
dropzone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropzone.classList.add("dragover");
});
dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropzone.classList.remove("dragover");
  if (e.dataTransfer.files.length) stageFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener("change", () => {
  if (fileInput.files.length) stageFile(fileInput.files[0]);
});

/* staged upload: picking a photo no longer fires the analysis instantly -
   the user can type a position hint first, then press Start analysis */
let stagedFile = null;

function stageFile(file) {
  stagedFile = file;
  $("dz-staged-name").textContent = file.name;
  $("dz-staged").classList.remove("hidden");
  $("start-btn").classList.remove("hidden");
  const hintEl = $("hint-input");
  if (hintEl) hintEl.focus({ preventScroll: true });
}

$("start-btn").addEventListener("click", () => {
  if (stagedFile) startAnalysis(stagedFile);
});
$("hint-input").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && stagedFile) startAnalysis(stagedFile);
});

$("analyze-another").addEventListener("click", resetUI);
$("error-retry").addEventListener("click", resetUI);

/* black-hole logo: spin while analyzing, single flash when done */
function logoSpin(on) {
  const m = $("brand-mark");
  if (m) m.classList.toggle("analyzing", on);
  favSpin(on);
}
function logoFlash() {
  const m = $("brand-mark");
  if (m) {
    m.classList.remove("analyzing");
    m.classList.add("done-flash");
    setTimeout(() => m.classList.remove("done-flash"), 1300);
  }
  favFlash();
}

/* ---- browser-tab icon mirrors the brand mark: it spins while an analysis
   runs and flashes bright when the verdict lands, so even a backgrounded
   tab tells the user what the app is doing ---- */
const favLink = document.querySelector("link[rel='icon']");
const favBase = favLink ? favLink.href : "";
let favTimer = null;
let favAngle = 0;

function favDraw(angle, glow) {
  if (!favLink) return;
  const halo = glow
    ? `<circle cx='60' cy='64' r='50' fill='none' stroke='#ffe9c9' stroke-width='9' opacity='${(0.9 * glow).toFixed(2)}'/>` +
      `<circle cx='60' cy='64' r='57' fill='none' stroke='#ffb45c' stroke-width='5' opacity='${(0.55 * glow).toFixed(2)}'/>`
    : "";
  const svg =
    `<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 120 120'>` +
    `<defs><radialGradient id='g'><stop offset='0.29' stop-color='#04070d'/><stop offset='0.33' stop-color='#ffe9c9' stop-opacity='0.95'/><stop offset='0.4' stop-color='#ff9d3c' stop-opacity='0.35'/><stop offset='0.62' stop-color='#5ba8ff' stop-opacity='0.14'/><stop offset='1' stop-color='#5ba8ff' stop-opacity='0'/></radialGradient>` +
    `<linearGradient id='d' x1='0' y1='0' x2='1' y2='0'><stop offset='0' stop-color='#7cc4ff'/><stop offset='0.28' stop-color='#ffe0ab'/><stop offset='0.55' stop-color='#ffb45c'/><stop offset='1' stop-color='#a78bfa'/></linearGradient></defs>` +
    `<g transform='rotate(${angle} 60 64)'>` +
    `<path d='M 8 64 A 52 8 0 0 1 112 64' fill='none' stroke='url(#d)' stroke-width='5' stroke-linecap='round' opacity='0.75'/>` +
    `<path d='M 33 66 A 27 29 0 0 1 87 66' fill='none' stroke='url(#d)' stroke-width='5.5' stroke-linecap='round' opacity='0.95'/>` +
    `<circle cx='60' cy='64' r='41' fill='url(#g)'/>` +
    `<circle cx='60' cy='64' r='19' fill='#04070d'/>` +
    `<circle cx='60' cy='64' r='20' fill='none' stroke='#ffe9c9' stroke-width='2' opacity='0.9'/>` +
    `<path d='M 8 64 A 52 8 0 0 0 112 64' fill='none' stroke='url(#d)' stroke-width='6.5' stroke-linecap='round'/>` +
    `</g>${halo}</svg>`;
  favLink.href = "data:image/svg+xml," + encodeURIComponent(svg);
}

function favSpin(on) {
  clearInterval(favTimer);
  favTimer = null;
  if (!on) {
    if (favLink) favLink.href = favBase;
    return;
  }
  // time-based angle at ~60fps: butter-smooth in the foreground, and when
  // the browser throttles a background tab the icon still lands on the
  // CORRECT elapsed angle instead of visibly stepping between frames
  const t0 = performance.now() - (favAngle / 360) * 2400;
  favTimer = setInterval(() => {
    favAngle = ((performance.now() - t0) / 2400) * 360 % 360;
    favDraw(favAngle, 0);
  }, 16);
}

function favFlash() {
  clearInterval(favTimer);
  // bright halo that decays over ~1.3s - the tab-icon twin of done-flash
  const steps = [1, 0.8, 0.6, 0.42, 0.26, 0.12, 0];
  let i = 0;
  favTimer = setInterval(() => {
    favDraw(0, steps[i]);
    if (++i >= steps.length) {
      clearInterval(favTimer);
      favTimer = null;
      if (favLink) favLink.href = favBase;
    }
  }, 190);
}

function resetUI() {
  clearInterval(pollTimer);
  logoSpin(false);
  history.replaceState(null, "", location.pathname);
  currentJob = null;
  analysisActive = false;
  stagedFile = null;
  $("dz-staged").classList.add("hidden");
  $("start-btn").classList.add("hidden");
  $("nav-new").classList.add("hidden");
  fileInput.value = "";
  refreshHistory();
  $("analysis-section").classList.add("hidden");
  $("methodology-section").classList.add("hidden");
  ["verdict-card", "object-card", "tabs", "error-card", "analyze-another",
   "field-objects-card", "id-status", "sources-card"].forEach((id) => $(id).classList.add("hidden"));
  $("viewer-img").removeAttribute("src");
  clearOverlay();
  $("aladin-div").innerHTML = "";
  $("upload-section").classList.remove("hidden");
}

function showError(msg) {
  clearInterval(pollTimer);
  logoSpin(false);
  $("error-message").textContent = msg;
  $("error-card").classList.remove("hidden");
  $("analyze-another").classList.remove("hidden");
}

/* ============================ recent analyses ============================ */

async function refreshHistory() {
  let items;
  try {
    const resp = await fetch("/api/history");
    if (!resp.ok) return;
    items = (await resp.json()).jobs || [];
  } catch {
    return;
  }
  const card = $("history-card"), list = $("history-list");
  if (!card || !list) return;
  if (!items.length) { card.classList.add("hidden"); return; }
  list.innerHTML = "";
  items.forEach((it, i) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "history-item";
    btn.innerHTML =
      `<span class="h-num"></span><img class="h-thumb" loading="lazy" alt="">` +
      `<span class="h-meta"><span class="h-name"></span><span class="h-head"></span></span>`;
    btn.querySelector(".h-num").textContent = i + 1;
    btn.querySelector(".h-thumb").src = `/api/job/${it.id}/thumb`;
    btn.querySelector(".h-name").textContent = it.filename || "upload";
    btn.querySelector(".h-head").textContent = it.headline || "";
    btn.addEventListener("click", () => loadHistoricJob(it.id));
    list.appendChild(btn);
  });
  card.classList.remove("hidden");
}

async function loadHistoricJob(id) {
  let job;
  try {
    const resp = await fetch(`/api/job/${id}`);
    if (!resp.ok) throw new Error(resp.statusText);
    job = await resp.json();
  } catch (err) {
    showError("Could not load that analysis: " + err.message);
    return;
  }
  // render exactly like a freshly finished analysis - same code path
  clearInterval(pollTimer);
  analysisActive = false;
  $("nav-new").classList.remove("hidden");
  $("upload-section").classList.add("hidden");
  $("methodology-section").classList.add("hidden");
  $("analysis-section").classList.remove("hidden");
  ["verdict-card", "object-card", "tabs", "error-card",
   "field-objects-card", "id-status", "sources-card"]
    .forEach((x) => $(x).classList.add("hidden"));
  renderStages(job.stages);
  const im = $("viewer-img");
  im.onload = () => drawOverlay();
  im.src = `/api/job/${id}/image`;
  currentJob = job;
  prepareRegionImages(job);
  renderReport(job);
  drawOverlay();
  history.replaceState(null, "", `#job=${id}`);
  window.scrollTo({ top: 0, behavior: "smooth" });
}

/* ============================ analysis flow ============================ */

async function startAnalysis(file) {
  clearInterval(pollTimer); // never let two polls fight over the UI
  currentJob = null;
  analysisActive = true;
  $("nav-new").classList.remove("hidden");
  $("upload-section").classList.add("hidden");
  $("methodology-section").classList.add("hidden");
  $("analysis-section").classList.remove("hidden");
  ["verdict-card", "object-card", "tabs", "error-card",
   "analyze-another", "field-objects-card", "id-status"].forEach((id) => $(id).classList.add("hidden"));
  // restore bits a previous "not a sky image" run may have hidden
  ["viewer-controls", "viewer-hint"].forEach((id) => $(id).classList.remove("hidden"));
  // the PREVIOUS image's toggle counts must never linger over a new analysis:
  // blank them now, refill when this job's report lands
  ["ct-identified", "ct-stars", "ct-faint", "ct-sources"].forEach((id) => ($(id).textContent = ""));
  $("sources-card").classList.add("hidden");
  document.querySelectorAll("#viewer-controls .toggle").forEach((t) => (t.style.display = ""));
  regionImgs = { main: null, dark: null };
  $("file-label").textContent = `${file.name} · ${(file.size / 1024 / 1024).toFixed(2)} MB`;

  // local preview first (may fail for HEIC; backend JPEG replaces it after ingest)
  logoSpin(true);
  const url = URL.createObjectURL(file);
  const img = $("viewer-img");
  img.onerror = () => img.removeAttribute("src");
  img.src = url;
  clearOverlay();
  renderStages(null);

  const form = new FormData();
  form.append("file", file);
  const hintEl = $("hint-input"), fovEl = $("hint-fov");
  if (hintEl && hintEl.value.trim()) form.append("hint", hintEl.value.trim());
  if (fovEl && fovEl.value) form.append("hint_fov", fovEl.value);

  let jobId;
  try {
    const resp = await fetch("/api/analyze", { method: "POST", body: form });
    if (!resp.ok) throw new Error((await resp.json()).detail || resp.statusText);
    jobId = (await resp.json()).job_id;
  } catch (err) {
    showError("Upload failed: " + err.message);
    return;
  }

  // the job id lives in the URL so a reload (or a server restart while the
  // page was open) can re-attach instead of dumping the user on page one
  history.replaceState(null, "", `#job=${jobId}`);
  startPolling(jobId);
}

function startPolling(jobId) {
  let swappedToServerImage = false;
  let notFound = 0;
  clearInterval(pollTimer);
  pollTimer = setInterval(async () => {
    let job;
    try {
      const resp = await fetch(`/api/job/${jobId}`);
      if (resp.status === 404) {
        // the server restarted mid-run: the in-memory job is gone for good -
        // an endless silent spinner here just looks like a freeze
        if (++notFound >= 6) {
          clearInterval(pollTimer);
          history.replaceState(null, "", location.pathname);
          showError("This analysis no longer exists on the server (it was " +
                    "likely restarted mid-run). Please upload the photo again.");
        }
        return;
      }
      if (!resp.ok) throw new Error(resp.statusText);
      job = await resp.json();
      notFound = 0;
    } catch {
      return; // transient poll failure; keep trying
    }
    renderStages(job.stages);

    // once ingest is done the backend has a browser-friendly JPEG (fixes HEIC)
    if (!swappedToServerImage && job.stages.ingest.status === "done") {
      swappedToServerImage = true;
      const im = $("viewer-img");
      im.onload = () => drawOverlay();
      im.src = `/api/job/${jobId}/image`;
    }

    if (job.status === "done") {
      clearInterval(pollTimer);
      logoFlash();
      currentJob = job;
      prepareRegionImages(job);
      renderReport(job);
      drawOverlay();
      refreshHistory();
    } else if (job.status === "error") {
      clearInterval(pollTimer);
      showError(job.error || "Analysis failed.");
    }
  }, 900);
}

async function resumeFromHash() {
  const m = location.hash.match(/^#job=([0-9a-f]{12})$/);
  if (!m) return;
  const jobId = m[1];
  let job;
  try {
    const resp = await fetch(`/api/job/${jobId}`);
    if (!resp.ok) throw new Error(resp.statusText);
    job = await resp.json();
  } catch {
    history.replaceState(null, "", location.pathname);
    return; // job lost (server restarted) - stay on the upload page
  }
  if (job.status === "done") {
    loadHistoricJob(jobId);
    return;
  }
  // still running: re-enter the analysis view and keep polling
  analysisActive = true;
  $("nav-new").classList.remove("hidden");
  $("upload-section").classList.add("hidden");
  $("methodology-section").classList.add("hidden");
  $("analysis-section").classList.remove("hidden");
  logoSpin(true);
  renderStages(job.stages);
  const im = $("viewer-img");
  im.onload = () => drawOverlay();
  im.src = `/api/job/${jobId}/image`;
  startPolling(jobId);
}

function renderStages(stages) {
  const list = $("stage-list");
  list.innerHTML = "";
  for (const key of Object.keys(STAGE_LABELS)) {
    const st = stages ? stages[key] : { status: "pending", detail: "" };
    const div = document.createElement("div");
    div.className = `stage ${st.status}`;
    const icon =
      st.status === "running"
        ? '<span class="spinner"></span>'
        : `<span>${STAGE_ICONS[st.status] || "○"}</span>`;
    div.innerHTML = `
      <span class="s-icon">${icon}</span>
      <span class="s-name">${STAGE_LABELS[key]}</span>
      <span class="s-detail">${escapeHtml(st.detail || "")}</span>`;
    list.appendChild(div);
  }
}

/* ============================ overlay ============================ */

["ov-identified", "ov-stars", "ov-faint", "ov-notable", "ov-main", "ov-sources",
 "ov-dark", "ov-ann"].forEach((id) =>
  $(id).addEventListener("change", drawOverlay)
);
window.addEventListener("resize", () => drawOverlay());

let hitTargets = []; // {sx, sy, star} in canvas space, for hover/click
let regionImgs = { main: null, dark: null }; // decoded region-mask overlays

/* Region masks arrive as base64 PNGs; decode them once per job and redraw
   when ready so the outline appears without user interaction. */
function prepareRegionImages(job) {
  regionImgs = { main: null, dark: null };
  const ov = job.overlay || {};
  const load = (key, b64) => {
    if (!b64) return;
    const img = new Image();
    img.onload = () => drawOverlay();
    img.src = "data:image/png;base64," + b64;
    regionImgs[key] = img;
  };
  load("main", ov.main_source && ov.main_source.region_mask_png);
  load("dark", ov.dark_structure && ov.dark_structure.mask_png);
}

function clearOverlay() {
  const canvas = $("overlay-canvas");
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  hitTargets = [];
}

/* Fill the per-class counts and hide toggles for empty classes, so the legend
   reflects what's actually in the image (10k unnamed stars, 0 identified). */
function updateOverlayLegend(job) {
  const stars = (job.overlay && job.overlay.stars) || [];
  // classes must be mutually exclusive, exactly like the draw partition:
  // a FAINT detection that got a catalog name was counted in both "identified"
  // and "faint", double-subtracting into "Unidentified (-1)"
  const idN = stars.filter((s) => s.id).length;
  const faintN = stars.filter((s) => !s.id && s.tier === "faint").length;
  const unidN = stars.length - idN - faintN;

  const setRow = (toggleId, countId, n) => {
    $(countId).textContent = n ? `(${n.toLocaleString("en-US")})` : "";
    // hide a class with nothing in it (e.g. "Identified" when nothing solved)
    $(toggleId).closest(".toggle").style.display = n ? "" : "none";
  };
  setRow("ov-identified", "ct-identified", idN);
  setRow("ov-stars", "ct-stars", unidN);
  setRow("ov-faint", "ct-faint", faintN);
  // dark-structure layer only exists on bright nebular frames
  $("ov-dark").closest(".toggle").style.display =
    job.overlay && job.overlay.dark_structure ? "" : "none";
  // numbered sources only when the frame holds several obvious objects
  const srcN = ((job.overlay && job.overlay.sources) || []).length;
  $("ct-sources").textContent = srcN >= 2 ? `(${srcN})` : "";
  $("ov-sources").closest(".toggle").style.display = srcN >= 2 ? "" : "none";
}

function drawOverlay() {
  const canvas = $("overlay-canvas");
  const img = $("viewer-img");
  if (!currentJob || !img.src || !img.naturalWidth) return clearOverlay();

  const ov = currentJob.overlay || {};
  const dims = currentJob.image || {};
  const aw = dims.analysis_width || img.naturalWidth;
  const ah = dims.analysis_height || img.naturalHeight;

  canvas.width = img.clientWidth;
  canvas.height = img.clientHeight;
  const sx = canvas.width / aw;
  const sy = canvas.height / ah;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  hitTargets = [];

  const notableSet = new Set((ov.notable_sources || []).map((n) => `${n.x},${n.y}`));
  const showId = $("ov-identified").checked;   // gold rings, named catalog stars
  const showUnid = $("ov-stars").checked;      // blue rings, detected but unnamed
  const showFaint = $("ov-faint").checked;
  const showNotable = $("ov-notable").checked;

  if (ov.stars) {
    for (const s of ov.stars) {
      const identified = !!s.id;
      // an IDENTIFIED object is never also "notable" - the orange ring was
      // stacking on top of the gold one
      const isNotable = !identified && notableSet.has(`${s.x},${s.y}`);
      const faint = s.tier === "faint";

      // decide visibility per class
      let visible;
      if (isNotable) visible = showNotable || showUnid;
      else if (identified) visible = showId;
      else if (faint) visible = showFaint;
      else visible = showUnid;
      if (!visible) continue;

      const px = s.x * sx, py = s.y * sy;
      if (isNotable && showNotable) {
        ctx.strokeStyle = "rgba(251, 146, 60, 0.95)"; // orange: notable
        ctx.lineWidth = 2;
        ctx.beginPath(); ctx.arc(px, py, 9, 0, Math.PI * 2); ctx.stroke();
      } else if (identified) {
        ctx.strokeStyle = "rgba(250, 204, 21, 0.95)"; // gold: identified
        ctx.lineWidth = 1.8;
        ctx.beginPath(); ctx.arc(px, py, 6.5, 0, Math.PI * 2); ctx.stroke();
      } else {
        ctx.strokeStyle = faint ? "rgba(148, 161, 195, 0.4)" : "rgba(91, 168, 255, 0.85)";
        ctx.lineWidth = 1;
        ctx.beginPath(); ctx.arc(px, py, faint ? 3.5 : 5, 0, Math.PI * 2); ctx.stroke();
      }
      hitTargets.push({ sx: px, sy: py, star: s });
    }
  }

  if ($("ov-main").checked && ov.main_source) {
    const m = ov.main_source;
    ctx.font = "11px monospace";
    ctx.fillStyle = "rgba(52, 211, 153, 0.95)";
    if (m.frame_filling) {
      // the emission fills the FRAME: its mask contour is just the mask's
      // HOLES (dark pillars, dim corners) - drawing it looked like random
      // noodles around the corners. A label says it all.
      ctx.fillText("main source: diffuse emission across the entire frame",
                   10, canvas.height - 10);
    } else if (m.region_mask_png && regionImgs.main && regionImgs.main.complete) {
      // irregular source: trace its TRUE measured shape, not a fitted ellipse
      ctx.drawImage(regionImgs.main, 0, 0, canvas.width, canvas.height);
      ctx.fillText("main source", 10, canvas.height - 10);
    } else {
      // disks and point sources really are elliptical - keep the crisp ellipse
      ctx.strokeStyle = "rgba(52, 211, 153, 0.95)";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.ellipse(
        m.center_x * sx, m.center_y * sy,
        Math.max((m.major_axis_px / 2) * sx, 8),
        Math.max((m.minor_axis_px / 2) * sy, 8),
        ((m.angle_deg || 0) * Math.PI) / 180, 0, Math.PI * 2
      );
      ctx.stroke();
      ctx.fillText("main source",
        m.center_x * sx + (m.major_axis_px / 2) * sx + 8,
        m.center_y * sy);
    }
  }

  // numbered rings for every clearly-visible source (galaxies, big stars...)
  const srcList = ov.sources || [];
  if ($("ov-sources").checked && srcList.length >= 2) {
    ctx.strokeStyle = "rgba(34, 211, 238, 0.9)";
    ctx.fillStyle = "rgba(34, 211, 238, 0.95)";
    ctx.lineWidth = 2;
    ctx.font = "bold 13px monospace";
    for (const s of srcList) {
      const px = s.center_x * sx, py = s.center_y * sy;
      const r = Math.max((s.major_axis_px / 2) * sx * 1.25, 16);
      ctx.beginPath();
      ctx.arc(px, py, r, 0, Math.PI * 2);
      ctx.stroke();
      const tag = s.name ? `${s.index} · ${s.name}` : String(s.index);
      ctx.fillText(tag, px - r + 4, py - r - 5);
    }
  }

  if ($("ov-dark").checked && ov.dark_structure &&
      regionImgs.dark && regionImgs.dark.complete) {
    ctx.drawImage(regionImgs.dark, 0, 0, canvas.width, canvas.height);
    ctx.font = "11px monospace";
    ctx.fillStyle = "rgba(248, 113, 113, 0.95)";
    ctx.fillText("dark structure (absorbing dust)", 10, 16);
  }

  if ($("ov-ann").checked && ov.annotations) {
    ctx.strokeStyle = "rgba(167, 139, 250, 0.95)";
    ctx.fillStyle = "rgba(167, 139, 250, 0.95)";
    ctx.lineWidth = 1.5;
    ctx.font = "12px monospace";
    for (const a of ov.annotations) {
      if (a.pixel_x == null) continue;
      const r = Math.max((a.radius_px || 20) * sx, 12);
      ctx.beginPath();
      ctx.arc(a.pixel_x * sx, a.pixel_y * sy, r, 0, Math.PI * 2);
      ctx.stroke();
      const label = (a.names || []).join(", ");
      if (label) ctx.fillText(label, a.pixel_x * sx + r + 5, a.pixel_y * sy + 4);
    }
  }
}

/* ---- hover tooltip + click-through to SIMBAD ---- */

const viewerWrap = $("viewer-wrap");
const tooltip = $("star-tooltip");
let hoveredStar = null;

function nearestTarget(mx, my) {
  let best = null, bestD = 14; // px hit radius
  for (const t of hitTargets) {
    const d = Math.hypot(t.sx - mx, t.sy - my);
    if (d < bestD) { best = t; bestD = d; }
  }
  return best;
}

viewerWrap.addEventListener("mousemove", (e) => {
  if (!hitTargets.length) return;
  const rect = viewerWrap.getBoundingClientRect();
  const mx = e.clientX - rect.left, my = e.clientY - rect.top;
  const t = nearestTarget(mx, my);
  hoveredStar = t ? t.star : null;
  viewerWrap.style.cursor = t && t.star.id ? "pointer" : "default";
  if (!t) { tooltip.classList.add("hidden"); return; }

  const s = t.star;
  let html;
  if (s.id) {
    const rows = [`<div class="tt-name">${escapeHtml(s.id.name)}</div>`,
                  `<div class="tt-type">${escapeHtml(s.id.type_label || "")}</div>`];
    if (s.id.spectral_type) rows.push(`<div>Spectral type: <b>${escapeHtml(s.id.spectral_type)}</b></div>`);
    if (s.id.v_mag != null) rows.push(`<div>Magnitude V: <b>${Number(s.id.v_mag).toFixed(1)}</b></div>`);
    if (s.id.distance_ly) rows.push(`<div>Distance: <b>${fmt(s.id.distance_ly)} ly</b></div>`);
    rows.push(`<div class="tt-hint">click → full SIMBAD page</div>`);
    html = rows.join("");
  } else if (s.ctx) {
    // this detection is the bright CORE of a bigger detected source, not a
    // separate star
    html = `<div class="tt-name">Core of Source ${s.ctx.n}</div>
      <div class="tt-phys">This bright point is the nucleus of
      <b>${escapeHtml(s.ctx.label)}</b> — not a separate star.</div>
      <div>Detection: <b>${s.peak_snr}σ</b></div>` +
      (s.ra != null ? `<div>RA ${s.ra.toFixed(4)}°, Dec ${s.dec.toFixed(4)}°</div>` : "");
  } else {
    // NO star assumption: a compact dot can be a star, a distant galaxy or a
    // quasar. Give the star reading as CONDITIONAL, plus the honest caveat.
    const rb = s.rb_ratio;
    let phys = "";
    if (rb != null) {
      if (rb < 0.85)
        phys = "If a star: blue-white → HOT (class B/A, 7,500–20,000+ K), typically young and massive.";
      else if (rb < 1.10)
        phys = "If a star: white → Sun-like temperature (class F/G, 5,000–7,500 K).";
      else if (rb < 1.40)
        phys = "If a star: yellow-orange → cooler than the Sun (class K, 3,900–5,300 K).";
      else
        phys = "If a star: orange-red → COOL (class M) — an aged red giant, a red dwarf, or dust-reddened. If slightly fuzzy, an orange blob this size can equally be a distant galaxy.";
    }
    html = `<div class="tt-name">Unidentified object</div>
      <div>Detection: <b>${s.peak_snr}σ</b> (${s.tier})</div>
      <div>Color R/B: <b>${rb ?? "?"}</b></div>` +
      (phys ? `<div class="tt-phys">${phys}</div>` : "") +
      `<div class="tt-hint">a point this small can be a star, a distant galaxy or a quasar —
       a successful sky-position match is what settles it</div>` +
      (s.ra != null ? `<div>RA ${s.ra.toFixed(4)}°, Dec ${s.dec.toFixed(4)}°</div>` : "");
  }
  tooltip.innerHTML = html;
  tooltip.classList.remove("hidden");
  const ttw = tooltip.offsetWidth, tth = tooltip.offsetHeight;
  let tx = t.sx + 14, ty = t.sy - tth / 2;
  if (tx + ttw > rect.width - 4) tx = t.sx - ttw - 14;
  ty = Math.max(4, Math.min(ty, rect.height - tth - 4));
  tooltip.style.left = `${tx}px`;
  tooltip.style.top = `${ty}px`;
});

viewerWrap.addEventListener("mouseleave", () => {
  tooltip.classList.add("hidden");
  hoveredStar = null;
});

viewerWrap.addEventListener("click", () => {
  if (hoveredStar && hoveredStar.id && hoveredStar.id.url) {
    window.open(hoveredStar.id.url, "_blank", "noopener");
  }
});

/* ============================ report ============================ */

function renderReport(job) {
  const report = job.report;
  $("verdict-card").classList.remove("hidden");
  $("analyze-another").classList.remove("hidden");

  // Non-astronomical image: show ONLY the rejection message and the way to
  // upload a new photo - no markers, no tabs, no analysis clutter.
  if (report.mode === "not_astronomical") {
    clearOverlay();
    ["tabs", "object-card", "notable-card", "field-objects-card",
     "viewer-controls", "viewer-hint", "id-status", "sources-card"].forEach((id) =>
      $(id).classList.add("hidden"));
    const badge = $("mode-badge");
    badge.className = "badge not_astronomical";
    badge.textContent = "Not a Sky Image";
    $("headline").textContent = report.headline;
    $("summary").textContent = report.summary || "";
    $("identity-line").classList.add("hidden");
    return;
  }
  $("tabs").classList.remove("hidden");
  $("viewer-controls").classList.remove("hidden");
  $("viewer-hint").classList.remove("hidden");

  const badge = $("mode-badge");
  badge.className = `badge ${report.mode}`;
  badge.textContent =
    report.mode === "identified" ? "Identified" :
    report.mode === "probabilistic" ? "Probabilistic Analysis" :
    report.mode === "not_astronomical" ? "Not a Sky Image" : "Indeterminate";
  $("headline").textContent = report.headline;
  $("summary").textContent = report.summary || "";

  renderIdentityLine(job, report);
  renderIdStatus(job);
  updateOverlayLegend(job);

  // object card
  const objCard = $("object-card");
  if (report.object) {
    objCard.classList.remove("hidden");
    objCard.innerHTML = buildObjectCard(report.object);
  }

  // per-source list: "1. NGC 6285 (Galaxy)" / "2. Bright foreground star..."
  const srcEntries = report.sources || [];
  const srcCard = $("sources-card");
  if (srcEntries.length >= 2) {
    srcCard.classList.remove("hidden");
    $("sources-list").innerHTML = srcEntries.map((s) => `
      <div class="source-item">
        <div class="source-num">${s.index}</div>
        <div class="source-body">
          <div class="source-label${s.identified ? " named" : ""}">${escapeHtml(s.label)}</div>
          <div class="source-text">${escapeHtml(s.evidence || "")}</div>
        </div>
      </div>`).join("");
  } else {
    srcCard.classList.add("hidden");
  }

  // hypotheses
  const hypEl = $("hypotheses");
  hypEl.innerHTML = "";
  for (const h of report.hypotheses) {
    const div = document.createElement("div");
    div.className = "hyp";
    const evid = h.evidence.map((e) => `<li>${escapeHtml(e)}</li>`).join("");
    const notes = (h.notes || []).filter(Boolean)
      .map((n) => `<div class="note">${escapeHtml(n)}</div>`).join("");
    div.innerHTML = `
      <div class="hyp-top">
        <span class="hyp-label">${escapeHtml(h.label)}</span>
        <span class="hyp-right"><span class="conf ${h.band.toLowerCase()}">${h.band}</span></span>
      </div>
      <div class="score-wrap">
        <div class="score-bar"><div class="score-fill" style="width:0%"></div></div>
        <span class="score-num">${h.score}/100</span>
      </div>
      <ul>${evid}</ul>${notes}`;
    hypEl.appendChild(div);
    setTimeout(() => {
      div.querySelector(".score-fill").style.width = `${h.score}%`;
    }, 60);
  }

  fillList("observations", report.observations);
  fillList("caveats", report.caveats);
  fillList("science-notes", report.science_notes);

  // notable secondary sources
  const notable = report.notable_sources || [];
  const notableCard = $("notable-card");
  if (notable.length) {
    notableCard.classList.remove("hidden");
    $("notable-list").innerHTML = notable.map((n) => {
      const head = n.name
        ? `<a href="${encodeURI(n.url)}" target="_blank" rel="noopener">${escapeHtml(n.name)}</a>`
        : `Source at (${Math.round(n.x)}, ${Math.round(n.y)})`;
      const snr = n.peak_snr != null ? `<span class="notable-snr">${n.peak_snr}σ</span>` : "";
      return `<div class="notable-item">
        <div class="notable-head">◉ ${head}${snr}</div>
        <div class="notable-text">${escapeHtml(n.text)}</div>
      </div>`;
    }).join("");
  } else {
    notableCard.classList.add("hidden");
  }

  renderPhotometry(job.photometry || {});
  renderSkyContext(job, report);
}

/* One explicit line answering "which object is this specifically?" - a name
   when we located it, an honest "not determined" (with the reason) when we
   couldn't. This is the "tanımlıysa şu, değilse tanımsız" the user asked for. */
function renderIdentityLine(job, report) {
  const el = $("identity-line");
  el.classList.remove("named", "undetermined");
  if (report.mode === "not_astronomical") {
    el.classList.add("hidden");
    return;
  }
  el.classList.remove("hidden");
  const obj = report.object;
  if (obj && obj.name) {
    el.classList.add("named");
    // the identified object's name links to its SIMBAD page (long-standing
    // user request: "let me click the main source and land on its page")
    const simbad = "https://simbad.cds.unistra.fr/simbad/sim-id?Ident="
      + encodeURIComponent(obj.name.split(" (")[0].trim());
    el.innerHTML = `<span class="id-key">Specific object:</span>
      <a class="obj-link" href="${simbad}" target="_blank" rel="noopener"
         title="Open this object's SIMBAD page"><b>${escapeHtml(obj.name)}</b></a>${obj.type_label ? " — " + escapeHtml(obj.type_label) : ""}`;
  } else {
    el.classList.add("undetermined");
    const solved = job.solve && job.solve.solved;
    const reason = solved
      ? "located on the sky, but no catalog object sits at this exact position"
      : "it couldn't be located on the sky — the frame lacks a resolvable star field, so there is no position to match against catalogs";
    el.innerHTML = `<span class="id-key">Specific object:</span>
      <b>Not determined</b> — ${reason}.`;
  }
}

/* Explains, in one clear banner, whether per-object identification is available
   and why - so a failed plate solve reads as an honest limitation, not a bug. */
function renderIdStatus(job) {
  const el = $("id-status");
  const stage = (job.stages && job.stages.platesolve) || {};
  const solve = job.solve || {};
  const idCount = (job.overlay?.stars || []).filter((s) => s.id).length;
  el.classList.remove("hidden", "ok", "warn", "info");

  if (job.report && job.report.mode === "not_astronomical") {
    el.classList.add("hidden");
    return;
  }

  if (solve.solved && idCount > 0) {
    el.classList.add("ok");
    el.innerHTML = `<b>✔ ${idCount} objects identified by sky position.</b>
      Gold rings are named catalog stars — hover for details, click to open SIMBAD.`;
  } else if (idCount > 0) {
    // appearance-based identifications (e.g. Saturn's moons) need no solve
    el.classList.add("ok");
    el.innerHTML = `<b>✔ ${idCount} objects identified by appearance.</b>
      Gold rings are named objects — hover for details, click to open their page.`;
  } else if (solve.solved) {
    el.classList.add("ok");
    el.innerHTML = `<b>✔ Field solved.</b> Sky position found, but no cataloged stars fell
      on the detected points (they may be too faint for the catalog).`;
  } else if (stage.status === "skipped") {
    el.classList.add("info");
    el.innerHTML = `<b>Individual objects not named.</b> ${escapeHtml(stage.detail ||
      "Plate solving was skipped.")} Naming individual stars requires a solvable star field.`;
  } else {
    // Answer the obvious question head-on: "you drew rings on hundreds of
    // stars, why is every one of them unnamed?"
    const detected = (job.overlay?.stars || []).length;
    const manyStars = detected > 40
      ? `The ${detected} points marked here are real detections, but they are far fainter
         than the catalog stars a solver matches against — deep telescope images reach
         magnitudes no all-sky index contains, so they cannot anchor a sky position.`
      : `Plate solving found no star-pattern match — the frame is a close-up of a single
         object rather than a wide star field.`;
    el.classList.add("warn");
    el.innerHTML = `<b>Individual objects can't be named for this image.</b>
      ${manyStars} The class-level analysis below still applies. Naming works on wider
      shots that include ordinary catalog stars around the target.`;
  }
}

function buildObjectCard(obj) {
  const rows = [];
  if (obj.type_label) rows.push(["Type", obj.type_label]);
  if (obj.spectral_type) rows.push(["Spectral type", obj.spectral_type]);
  if (obj.distance_ly) rows.push(["Distance", `${fmt(obj.distance_ly)} light-years`]);
  if (obj.distance_mly) rows.push(["Distance", `≈ ${fmt(obj.distance_mly)} million light-years`]);
  if (obj.redshift) rows.push(["Redshift (z)", Number(obj.redshift).toFixed(4)]);
  if (obj.angular_size_arcmin)
    rows.push(["Angular size", `${Number(obj.angular_size_arcmin).toFixed(1)} arcmin`]);
  if (obj.physical_size_ly)
    rows.push(["Estimated size", `≈ ${fmt(obj.physical_size_ly)} light-years across`]);
  if (obj.v_mag) rows.push(["Magnitude (V)", Number(obj.v_mag).toFixed(1)]);
  if (obj.ra != null && obj.dec != null)
    rows.push(["Position (ICRS)", `RA ${Number(obj.ra).toFixed(3)}°, Dec ${Number(obj.dec).toFixed(3)}°`]);
  if (obj.source) rows.push(["Catalog", obj.source]);

  const table = rows
    .map(([k, v]) => `<tr><td>${k}</td><td>${escapeHtml(String(v))}</td></tr>`).join("");
  const img = obj.nasa_image
    ? `<img src="${encodeURI(obj.nasa_image.image_url)}" alt="NASA image of ${escapeHtml(obj.name)}"
         title="${escapeHtml(obj.nasa_image.title || "")}">` : "";

  const simbadUrl = "https://simbad.cds.unistra.fr/simbad/sim-id?Ident="
    + encodeURIComponent(obj.name.split(" (")[0].trim());
  return `${img}
    <div class="object-facts">
      <div class="obj-name"><a class="obj-link" href="${simbadUrl}"
        target="_blank" rel="noopener"
        title="Open this object's SIMBAD page">${escapeHtml(obj.name)}</a></div>
      <div class="obj-type">${escapeHtml(obj.type_note || "")}</div>
      <table>${table}</table>
    </div>`;
}

/* ============================ photometry charts ============================ */

function renderPhotometry(phot) {
  drawHistogram($("chart-histogram"), phot.histogram);
  drawProfile($("chart-profile"), phot.radial_profile);

  const cbEl = $("color-bars");
  cbEl.innerHTML = "";
  const cp = phot.color_profile;
  if (cp) {
    for (const [label, val, color] of [
      ["Red", cp.mean_r, "#f87171"],
      ["Green", cp.mean_g, "#34d399"],
      ["Blue", cp.mean_b, "#5ba8ff"],
    ]) {
      const row = document.createElement("div");
      row.className = "cbar-row";
      row.innerHTML = `
        <span class="cbar-label">${label}</span>
        <div class="cbar-track"><div class="cbar-fill" style="width:${(val * 100).toFixed(1)}%;background:${color}"></div></div>
        <span class="cbar-val">${val.toFixed(3)}</span>`;
      cbEl.appendChild(row);
    }
  }
  const stats = [];
  if (phot.radial_profile && phot.radial_profile.half_light_radius_px != null)
    stats.push(`half-light radius: ${phot.radial_profile.half_light_radius_px} px`);
  $("phot-stats").textContent = stats.join(" · ");
}

function chartAxes(ctx, W, H, pad) {
  ctx.strokeStyle = "#232e4a";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(pad, 8);
  ctx.lineTo(pad, H - pad);
  ctx.lineTo(W - 8, H - pad);
  ctx.stroke();
}

function drawHistogram(canvas, hist) {
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const { width: W, height: H } = canvas;
  ctx.clearRect(0, 0, W, H);
  if (!hist) return;
  const pad = 34;
  chartAxes(ctx, W, H, pad);
  const n = hist.fractions.length;
  const maxF = Math.max(...hist.fractions, 1e-9);
  const bw = (W - pad - 12) / n;
  for (let i = 0; i < n; i++) {
    // log scale: astro images are dominated by near-black background pixels
    const f = hist.fractions[i];
    const hFrac = f > 0 ? Math.max(0.02, (Math.log10(f) + 6) / (Math.log10(maxF) + 6)) : 0;
    const h = hFrac * (H - pad - 14);
    const shade = Math.round(40 + 200 * (i / n));
    ctx.fillStyle = `rgb(${shade},${shade},${Math.min(shade + 30, 255)})`;
    ctx.fillRect(pad + i * bw, H - pad - h, Math.max(bw - 1, 1), h);
  }
  ctx.fillStyle = "#94a1c3";
  ctx.font = "11px monospace";
  ctx.fillText("dark", pad, H - pad + 14);
  ctx.fillText("bright", W - 48, H - pad + 14);
  ctx.fillText("pixel fraction (log)", pad + 4, 16);
}

function drawProfile(canvas, prof) {
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const { width: W, height: H } = canvas;
  ctx.clearRect(0, 0, W, H);
  if (!prof) {
    ctx.fillStyle = "#94a1c3";
    ctx.font = "12px monospace";
    ctx.fillText("No dominant source detected.", 20, H / 2);
    return;
  }
  const pad = 40;
  chartAxes(ctx, W, H, pad);
  const xs = prof.radius_px;
  const ys = prof.mean_brightness;
  const maxX = xs[xs.length - 1] || 1;
  const maxY = Math.max(...ys, 1e-9);

  ctx.strokeStyle = "#5ba8ff";
  ctx.lineWidth = 2;
  ctx.beginPath();
  for (let i = 0; i < xs.length; i++) {
    const x = pad + (xs[i] / maxX) * (W - pad - 14);
    const y = H - pad - (ys[i] / maxY) * (H - pad - 16);
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  }
  ctx.stroke();

  if (prof.half_light_radius_px != null) {
    const hx = pad + (prof.half_light_radius_px / maxX) * (W - pad - 14);
    ctx.strokeStyle = "rgba(167,139,250,0.7)";
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(hx, 12);
    ctx.lineTo(hx, H - pad);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = "#a78bfa";
    ctx.font = "11px monospace";
    ctx.fillText("r½", hx + 4, 22);
  }
  ctx.fillStyle = "#94a1c3";
  ctx.font = "11px monospace";
  ctx.fillText("radius (px) →", W - 110, H - pad + 14);
  ctx.fillText("brightness", pad + 4, 16);
}

/* ============================ sky context ============================ */

function renderSkyContext(job, report) {
  const sky = report.sky_context || {};
  const holder = $("sky-cards");
  holder.innerHTML = "";

  const addCard = (title, html) => {
    const div = document.createElement("div");
    div.className = "card";
    div.innerHTML = `<h3>${title}</h3>${html}`;
    holder.appendChild(div);
  };

  if (sky.constellation)
    addCard("Constellation",
      `<p class="sky-fact"><span class="big">${escapeHtml(sky.constellation)}</span><br>
       determined from the plate-solved position using official IAU boundaries</p>`);

  if (sky.moon)
    addCard("Moon at Capture Time",
      `<p class="sky-fact"><span class="big">${escapeHtml(sky.moon.phase_name)}</span><br>
       ${(sky.moon.illumination * 100).toFixed(0)}% illuminated (computed from EXIF timestamp)</p>`);

  if (sky.capture_time_utc)
    addCard("Capture Time",
      `<p class="sky-fact"><span class="big">${escapeHtml(sky.capture_time_utc)}</span><br>UTC (from EXIF)</p>`);

  if (sky.visible_bodies && sky.visible_bodies.length) {
    const rows = sky.visible_bodies.map((b) =>
      `<tr><td>${escapeHtml(b.body)}</td><td>${b.altitude_deg}°</td><td>${b.azimuth_deg}°</td></tr>`).join("");
    addCard("Solar-System Bodies Above the Horizon",
      `<table class="sky-table"><tr><th>Body</th><th>Altitude</th><th>Azimuth</th></tr>${rows}</table>`);
  }

  // field objects list
  const fo = report.field_objects || [];
  if (fo.length) {
    $("field-objects-card").classList.remove("hidden");
    fillList("field-objects",
      fo.map((o) => (o.type_label ? `${o.name} — ${o.type_label}` : o.name)));
  }

  // Aladin
  const solve = job.solve || {};
  if (solve.solved) {
    $("aladin-note").textContent = "Live DSS2 sky survey centered on the solved position.";
    loadAladin(solve);
  } else {
    $("aladin-note").textContent =
      "Not available: the image could not be plate-solved, so its sky position is unknown.";
  }
}

function loadAladin(solve) {
  const init = () => {
    /* global A */
    A.init.then(() => {
      $("aladin-div").innerHTML = "";
      A.aladin("#aladin-div", {
        target: `${solve.ra} ${solve.dec}`,
        fov: Math.max((solve.radius_deg || 0.5) * 2.5, 0.3),
        survey: "P/DSS2/color",
        showFullscreenControl: false,
      });
    });
  };
  if (aladinLoaded) return init();
  const s = document.createElement("script");
  s.src = "https://aladin.cds.unistra.fr/AladinLite/api/v3/latest/aladin.js";
  s.onload = () => { aladinLoaded = true; init(); };
  s.onerror = () => {
    $("aladin-note").textContent = "Sky atlas could not load (offline?).";
  };
  document.head.appendChild(s);
}

/* ============================ tabs ============================ */

document.querySelectorAll(".tab-h").forEach((btn) =>
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab-h").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    document.querySelectorAll(".tab-body").forEach((tb) => tb.classList.add("hidden"));
    $(`tab-${btn.dataset.tab}`).classList.remove("hidden");
  })
);

/* ============================ export ============================ */

$("export-json").addEventListener("click", () => {
  if (!currentJob) return;
  download(`analysis-${currentJob.id}.json`,
    new Blob([JSON.stringify(currentJob, null, 2)], { type: "application/json" }));
});

$("export-html").addEventListener("click", () => {
  if (!currentJob) return;
  const r = currentJob.report;
  const esc = escapeHtml;
  const hyps = r.hypotheses.map((h) => `
    <div style="border:1px solid #ccc;border-radius:8px;padding:12px;margin:10px 0">
      <b>${esc(h.label)}</b> — ${h.band} (${h.score}/100)
      <ul>${h.evidence.map((e) => `<li>${esc(e)}</li>`).join("")}</ul>
      ${(h.notes || []).map((n) => `<p><i>${esc(n)}</i></p>`).join("")}
    </div>`).join("");
  const lists = (title, arr) =>
    `<h3>${title}</h3><ul>${(arr || []).map((x) => `<li>${esc(x)}</li>`).join("")}</ul>`;
  const objBlock = r.object ? `<h3>Identified Object</h3><p><b>${esc(r.object.name)}</b> — ${esc(r.object.type_label || "")}<br>${esc(r.object.type_note || "")}</p>` : "";
  const html = `<!DOCTYPE html><html><head><meta charset="utf-8">
    <title>Celestial Analysis Report</title>
    <style>body{font-family:Georgia,serif;max-width:800px;margin:40px auto;padding:0 20px;color:#1a1a2e}
    h1{border-bottom:2px solid #1a1a2e;padding-bottom:8px}</style></head><body>
    <h1>Celestial Object Analysis Report</h1>
    <p><i>${esc(currentJob.filename)} — generated by Celestial Object Analyzer</i></p>
    <h2>${esc(r.headline)}</h2><p>${esc(r.summary || "")}</p>
    ${objBlock}<h3>Hypotheses</h3>${hyps}
    ${lists("Measured Observations", r.observations)}
    ${lists("Caveats & Honest Limits", r.caveats)}
    ${lists("Scientific Background", r.science_notes)}
    </body></html>`;
  download(`report-${currentJob.id}.html`, new Blob([html], { type: "text/html" }));
});

function download(name, blob) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 5000);
}

/* ============================ helpers ============================ */

function fillList(id, items) {
  $(id).innerHTML = (items || []).map((x) => `<li>${escapeHtml(x)}</li>`).join("");
}

function fmt(n) {
  return Number(n).toLocaleString("en-US", { maximumFractionDigits: 1 });
}

function escapeHtml(s) {
  const div = document.createElement("div");
  div.textContent = s == null ? "" : String(s);
  return div.innerHTML;
}

/* ============================ startup ============================ */

(async () => {
  try {
    const resp = await fetch("/api/status");
    const st = await resp.json();
    const a = $("chip-astrometry");
    if (st.astap_available) {
      a.textContent = "solver: local ASTAP" + (st.astrometry_key_configured ? " + nova" : "");
      a.classList.add("on");
      a.title = "Plate solving runs locally in seconds; nova.astrometry.net is the online fallback.";
    } else {
      a.textContent = st.astrometry_key_configured
        ? "solver: nova online" : "solver: none";
      a.classList.add(st.astrometry_key_configured ? "on" : "off");
      a.title = st.astrometry_key_configured
        ? "Solving uses the free nova.astrometry.net queue (can take minutes). Install ASTAP + D50/W08 databases into the astap/ folder for instant local solving."
        : "No solver available - install ASTAP or add an astrometry.net API key.";
    }
    const n = $("chip-ngc");
    n.textContent = st.openngc_available
      ? "OpenNGC: offline catalog loaded" : "OpenNGC: not installed";
    n.classList.add(st.openngc_available ? "on" : "off");
    if (!st.openngc_available)
      n.title = "Optional: download NGC.csv from github.com/mattiaverga/OpenNGC into the data/ folder for offline deep-sky data.";
  } catch { /* backend warming up */ }
  refreshHistory();
  resumeFromHash();
})();
