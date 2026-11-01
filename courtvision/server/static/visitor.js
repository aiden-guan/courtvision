/* Public-site sessions. Video bytes are only read by a blob URL and a local canvas.
 * The server's shared simulator is buffered per viewer; it never controls another visitor's playback.
 * Browser detection is deliberately image-space only: no invented game stats or court coordinates. */
(() => {
  "use strict";
  const $ = id => document.getElementById(id);
  const studio = () => window.CVStudio;
  const cv = () => window.CV;
  const video = () => $("local-video");
  const text = (id, value) => { $(id).textContent = value; };
  let enabled = false, base = null, view = null, mode = "idle", epoch = 0, sampleEpoch = 0;
  let url = null, model = null, modelPromise = null, detectorState = "idle", detectorError = "";
  let lens = null, demoPaused = false, delay = 0, pausedAt = 0, buffer = [], bufferTimer = 0;
  let sampleTimer = 0, inferenceBusy = false, lastSample = -1, nextTrack = 1, tracks = [];
  let samples = [], totalSamples = 0, peak = 0, ballSamples = 0, lastBoxes = [];
  let autoPlayPending = false;
  let frameHistory = [], captureHandle = 0, choice = 0;
  const MAX_SAMPLES = 3600; // bounded exports, including for multi-hour recordings
  const input = document.createElement("canvas");

  function render() {
    if (!view) return;
    studio().renderWorkspace(view, true);
    document.body.dataset.visitor = mode;
    $("local-summary").hidden = mode !== "video";
    $("demo-canvas").hidden = mode !== "demo";
    $("local-video").hidden = mode !== "video";
    $("local-overlay").hidden = mode !== "video" || !$("studio-overlay").checked;
    if (mode !== "idle") { $("studio-frame").hidden = true; $("studio-loading").hidden = true; }
    $("studio-speed").hidden = mode !== "video";
    if (mode === "demo") text("studio-throughput", demoPaused ? "Demo paused" : "Synthetic demo");
    if (mode === "video") {
      text("monitor-badge", "ON-DEVICE DETECTION");
      text("monitor-source", "Stays on your device");
      text("studio-session-label", video().ended ? "Video ended" : video().paused ? "Paused" : "Analysing locally");
      text("studio-throughput", detectorState === "ready" ? "On-device detection" : detectorState === "loading" ? "Loading detector…" : "Video preview");
      $("studio-play").setAttribute("aria-label", video().ended ? "Replay video" : video().paused ? "Play video" : "Pause video");
      text("vision-court", "Image-space only");
    }
    text("sys-detector-label", mode === "video" ? detectorState === "ready" ? "NBA model ready" : detectorState === "error" ? "NBA model unavailable" : "Loading NBA model…" : "NBA model · loads on demand");
  }

  function emptyView() {
    return { ...base, active: null, status: null, frame: null, vision: null, videos: [], analysis: null, replay: null, error: null };
  }

  function workspaceUpdate(data) {
    if (!data.public_demo) return false;
    base = data;
    if (!enabled) {
      enabled = true;
      cv().dispatch({ type: "reset", t: 0 });
      view = emptyView();
      const copy = document.querySelector("#studio-dialog .sheet-head p");
      copy.textContent = "Choose a clip up to 30 seconds. The NBA-trained model runs on your device; nothing is uploaded.";
      text("upload-title", "Try your own video");
      text("studio-start", "Analyse on this device");
      $("studio-play").dataset.label = "Pause";
      text("studio-empty-open", "Try my video");
      text("studio-open", "Try my video");
      text("studio-file-detail", "or choose a file from this device");
      document.querySelector(".hero-lede").textContent = "Explore a simulated game, or try the NBA-trained model on your own clip. Up to 30 seconds, processed entirely on your device.";
      document.querySelector(".hero-kicker").textContent = "Basketball vision, in your browser";
      document.querySelector(".hero-meta").textContent = "Up to 30 seconds · No install · No video uploads";
      document.querySelector(".file-types").textContent = "Up to 30 seconds · MP4 (H.264) or WebM recommended";
      for (const selector of [".upload-teams", "#studio-broadcast", "#library-add-title"]) {
        const el = document.querySelector(selector);
        (el?.closest(".toggle-row") || el)?.setAttribute("hidden", "");
      }
      for (const el of document.querySelectorAll("#side-nav .nav-group")) {
        if (["Setup", "Training"].includes(el.textContent.trim())) el.hidden = true;
      }
      // A new visitor chooses a session rather than being dropped into someone else's game.
      cv().selectView("tab-live");
      render();
    }
    return true;
  }

  function clearSession() {
    epoch++; sampleEpoch++; choice++; autoPlayPending = false;
    if (captureHandle && video().cancelVideoFrameCallback) video().cancelVideoFrameCallback(captureHandle);
    captureHandle = 0; frameHistory = [];
    clearTimeout(bufferTimer); clearTimeout(sampleTimer);
    buffer = []; delay = 0; demoPaused = false;
    video().pause(); video().removeAttribute("src"); video().load();
    if (url) URL.revokeObjectURL(url);
    url = null; mode = "idle";
    tracks = []; nextTrack = 1; lastSample = -1; lastBoxes = [];
    samples = []; totalSamples = 0; peak = 0; ballSamples = 0;
    if (lens) lens.clear();
    cv().dispatch({ type: "reset", t: 0 });
    view = emptyView();
    render();
  }

  async function exit() {
    clearSession(); cv().selectView("tab-live");
    studio().openDialog();
    $("studio-empty-open").focus();
  }

  async function startDemo() {
    clearSession();
    mode = "demo";
    const generation = epoch;
    $("studio-demo").disabled = true;
    try {
      const data = await cv().api("api/studio/start", { demo: true });
      if (generation !== epoch) return;
      view = { ...data, videos: [] };
      render(); cv().selectView("tab-live"); $("rtab-plays").click();
      if (!lens) lens = new window.CourtLens($("demo-canvas"), { wrap: $("monitor-screen"), interactive: false });
      lens.setGeometry(cv().courtGeometry(cv().S.court || {}));
      lens.resize();
      if (view.frame) applyDemo(view.frame);
      cv().reconnect(); // replay the shared game's metadata and history when joining mid-game
      if ($("studio-dialog").open) $("studio-dialog").close();
    } catch (error) {
      if (generation === epoch) { clearSession(); cv().toast(error.message, "error"); }
    } finally { $("studio-demo").disabled = false; }
  }

  function applyDemo(msg) {
    cv().dispatch(msg);
    if (!view?.active) return;
    if (msg.type === "meta" && lens) { lens.readColors(); lens.setGeometry(cv().S.geo); }
    if (msg.type === "frame") {
      view.frame = msg;
      view.status = { ...view.status, t: msg.t, paused: demoPaused };
      if (lens) lens.setFrame(msg);
      studio().receiveStatus(view.status);
    }
    if (msg.type === "status") view.status = { ...msg, paused: demoPaused };
    if (msg.type === "status" || msg.type === "frame") text("studio-throughput", demoPaused ? "Demo paused" : "Synthetic demo");
  }

  function drain() {
    clearTimeout(bufferTimer);
    if (mode !== "demo" || demoPaused) return;
    const now = performance.now() - delay;
    while (buffer.length && buffer[0].at <= now) applyDemo(buffer.shift().msg);
    if (buffer.length) bufferTimer = setTimeout(drain, Math.max(10, Math.min(100, buffer[0].at - now)));
  }

  function receive(msg) {
    if (!enabled) return false;
    if (mode !== "demo") return true;
    if (demoPaused || delay > 0) {
      // One three-minute simulation is comfortably below this bound. Reconnects cannot grow it forever.
      if (buffer.length >= 12000) {
        buffer = []; delay = 0; demoPaused = false;
        cv().toast("The demo stream reconnected. Resuming at the current play.");
      } else {
        buffer.push({ at: performance.now(), msg });
        if (!demoPaused) drain();
        return true;
      }
    }
    applyDemo(msg);
    return true;
  }

  async function control(action, value) {
    if (mode === "video") {
      if (action === "pause") video().pause();
      else if (action === "resume") { try { await video().play(); } catch { cv().toast("Press play on the video to continue."); } }
      else if (action === "speed") video().playbackRate = value;
      else if (action === "mute" || action === "unmute") video().muted = action === "mute";
      else return false;
      localStatus();
      return true;
    }
    if (mode !== "demo" || !view?.active) return false;
    if (action === "pause" && !demoPaused) { demoPaused = true; pausedAt = performance.now(); clearTimeout(bufferTimer); }
    else if (action === "resume" && demoPaused) { delay += performance.now() - pausedAt; demoPaused = false; drain(); }
    else if (!["pause", "resume"].includes(action)) return false;
    view.status = { ...view.status, paused: demoPaused };
    cv().dispatch({ type: "status", ...view.status });
    render();
    return true;
  }

  async function togglePlay() {
    if (mode === "demo" && view?.status?.finished) return startDemo();
    if (mode === "video" && video().ended) video().currentTime = 0;
    return control((mode === "demo" ? demoPaused : video().paused) ? "resume" : "pause");
  }

  async function loadDetector() {
    if (model) return model;
    if (!modelPromise) modelPromise = window.CVNBA.load();
    try { model = await modelPromise; return model; }
    finally { modelPromise = null; }
  }

  async function prepareDetector() {
    const generation = epoch;
    detectorState = "loading"; detectorError = "";
    text("local-analysis-note", "Loading the NBA-trained model on this device. Your clip stays here.");
    $("local-retry").hidden = true; render();
    try {
      await loadDetector();
      if (generation !== epoch || mode !== "video") return;
      detectorState = "ready";
      text("local-analysis-note", "NBA-trained model active. Play or seek to inspect your clip; pause to review detections.");
      render(); scheduleSample();
      if (autoPlayPending) {
        autoPlayPending = false;
        video().play().catch(() => { text("local-analysis-note", "Detector ready. Press play to analyse your video on this device."); });
      }
    } catch (error) {
      if (generation !== epoch || mode !== "video") return;
      detectorState = "error"; detectorError = error.message;
      console.error("NBA detector initialization failed", error);
      text("local-analysis-note", "Detector unavailable. You can still watch your video. Check your connection and retry.");
      $("local-retry").hidden = false; render();
    }
  }

  function inspectFile(file) {
    return new Promise((resolve, reject) => {
      const probe = document.createElement("video"), probeUrl = URL.createObjectURL(file);
      let settled = false;
      const done = (error) => {
        if (settled) return;
        settled = true; clearTimeout(timeout);
        const info = { duration: probe.duration, width: probe.videoWidth, height: probe.videoHeight };
        probe.removeAttribute("src"); probe.load(); URL.revokeObjectURL(probeUrl);
        if (error) reject(new Error(error)); else resolve(info);
      };
      const timeout = setTimeout(() => done("This video could not open. Try an MP4 (H.264) or WebM clip."), 10000);
      probe.addEventListener("loadedmetadata", () => {
        if (!Number.isFinite(probe.duration) || probe.duration <= 0 || !probe.videoWidth) done("Choose a file with a readable video track.");
        else if (probe.duration > 30) done("Clips must be 30 seconds or shorter. Trim your video and choose it again.");
        else done();
      }, { once: true });
      probe.addEventListener("error", () => done("This browser cannot decode this file. Try MP4 (H.264) or WebM."), { once: true });
      probe.preload = "metadata"; probe.src = probeUrl; probe.load();
    });
  }

  async function openFile(file) {
    const attempt = ++choice;
    text("studio-upload-status", "Checking the clip (up to 30 seconds)…");
    $("studio-upload-status").classList.remove("error");
    try { await inspectFile(file); }
    catch (error) {
      if (attempt === choice) {
        text("studio-upload-status", error.message);
        $("studio-upload-status").classList.add("error");
      }
      return;
    }
    if (attempt !== choice) return;
    text("studio-upload-status", "");
    clearSession(); mode = "video"; autoPlayPending = true;
    const generation = epoch;
    url = URL.createObjectURL(file);
    const id = `browser-${epoch}`;
    view = { ...emptyView(), active: { id, run_id: id, name: file.name, title: file.name.replace(/\.[^.]+$/, ""), browser: true,
      duration_s: 0, width: 0, height: 0 }, status: { t: 0, paused: true, speed: 1 },
      model: { state: "idle", name: "NBA-trained BallPersonNet" } };
    text("local-samples", 0); text("local-peak", "—"); text("local-ball", 0);
    detectorState = model ? "ready" : "idle";
    text("local-analysis-note", "Opening your video on this device…");
    render(); cv().selectView("tab-live"); $("rtab-people").click();
    if ($("studio-dialog").open) $("studio-dialog").close();
    video().playbackRate = 1;
    video().src = url;
    // Metadata/error events below handle decoding, including when the file opens after this call returns.
    video().load();
    if (generation === epoch) $("studio-exit").focus();
  }

  function localStatus() {
    if (mode !== "video" || !view?.active) return;
    view.status = { t: video().currentTime, paused: video().paused, finished: video().ended,
      speed: video().playbackRate, muted: video().muted };
    cv().dispatch({ type: "status", ...view.status });
    render();
  }

  function overlap(a, b) {
    const [x, y, w, h] = a, [xx, yy, ww, hh] = b;
    const area = Math.max(0, Math.min(x + w, xx + ww) - Math.max(x, xx)) * Math.max(0, Math.min(y + h, yy + hh) - Math.max(y, yy));
    return area / Math.max(1, w * h + ww * hh - area);
  }

  function associate(predictions, t) {
    const available = tracks.filter(track => Math.abs(t - track.t) < 1.5);
    const next = predictions.filter(p => p.class === "person" || p.class === "sports ball").map(p => {
      let match = null, best = 0.2;
      for (const track of available) {
        const score = track.class === p.class ? overlap(track.bbox, p.bbox) : 0;
        if (score > best) { match = track; best = score; }
      }
      if (match) available.splice(available.indexOf(match), 1);
      return { id: match?.id || `${p.class === "person" ? "Person" : "Ball"} ${nextTrack++}`, class: p.class,
        bbox: p.bbox, confidence: p.score, t };
    });
    tracks = next;
    return next;
  }

  function drawOverlay() {
    const canvas = $("local-overlay"), v = video();
    if (mode !== "video" || !v.videoWidth) return;
    // The canvas has the same intrinsic aspect ratio and object-fit as the video, including portrait clips.
    canvas.width = v.videoWidth; canvas.height = v.videoHeight;
    const ctx = canvas.getContext("2d"), scale = 1 / Math.min(960 / v.videoWidth, 544 / v.videoHeight);
    ctx.lineWidth = Math.max(2, v.videoWidth / 500);
    ctx.font = `${Math.max(14, v.videoWidth / 65)}px sans-serif`;
    for (const box of lastBoxes) {
      const [x, y, w, h] = box.bbox.map(n => n * scale);
      ctx.strokeStyle = box.class === "person" ? "#74b5ff" : "#ffb65c";
      ctx.fillStyle = ctx.strokeStyle;
      ctx.strokeRect(x, y, w, h);
      const label = `${box.id} · ${Math.round(box.confidence * 100)}%`;
      const textY = Math.max(18, y - 5);
      ctx.fillStyle = "rgba(0,0,0,.8)";
      ctx.fillRect(x, textY - Math.max(16, v.videoWidth / 60), ctx.measureText(label).width + 8, Math.max(20, v.videoWidth / 50));
      ctx.fillStyle = ctx.strokeStyle; ctx.fillText(label, x + 4, textY);
    }
  }

  function scheduleSample() {
    clearTimeout(sampleTimer);
    if (mode === "video" && detectorState === "ready") sampleTimer = setTimeout(sample, 200);
  }

  function captureFrame(t = video().currentTime) {
    const v = video();
    if (mode !== "video" || v.readyState < 2 || v.seeking || !v.videoWidth) return;
    const scale = Math.min(960 / v.videoWidth, 544 / v.videoHeight);
    if (input.width !== 960 || input.height !== 544) { input.width = 960; input.height = 544; }
    const ctx = input.getContext("2d", { willReadFrequently: true });
    ctx.fillStyle = "#000"; ctx.fillRect(0, 0, 960, 544);
    ctx.drawImage(v, 0, 0, Math.round(v.videoWidth * scale), Math.round(v.videoHeight * scale));
    frameHistory.push({ t, rgba: ctx.getImageData(0, 0, 960, 544).data });
    if (frameHistory.length > 12) frameHistory.shift();
  }

  function captureLoop() {
    if (mode !== "video" || !video().requestVideoFrameCallback) return;
    captureHandle = video().requestVideoFrameCallback((_, metadata) => {
      if (!document.hidden) captureFrame(metadata.mediaTime);
      captureLoop();
    });
  }

  async function sample() {
    if (mode !== "video" || detectorState !== "ready") return;
    const v = video(), generation = epoch, seek = sampleEpoch;
    if (inferenceBusy || document.hidden || v.readyState < 2 || v.seeking ||
        (v.paused && Math.abs(lastSample - v.currentTime) < 0.15)) { scheduleSample(); return; }
    if (!v.requestVideoFrameCallback || !frameHistory.length || v.paused) captureFrame();
    const latest = frameHistory.at(-1), t = latest?.t ?? v.currentTime;
    if (!latest || Math.abs(lastSample - t) < 0.15) { scheduleSample(); return; }
    inferenceBusy = true;
    try {
      const scale = Math.min(960 / v.videoWidth, 544 / v.videoHeight);
      const start = performance.now();
      const predictions = await model.detect(window.CVNBACore.history(frameHistory, t), Math.round(v.videoHeight * scale));
      if (generation !== epoch || seek !== sampleEpoch || mode !== "video") return;
      const elapsed = performance.now() - start;
      lastSample = t;
      lastBoxes = associate(predictions, t);
      const people = lastBoxes.filter(p => p.class === "person"), ball = lastBoxes.some(p => p.class === "sports ball");
      const entry = { t, people: people.length, ball_visible: ball, detections: lastBoxes.map(({ t: _, ...box }) => ({
        ...box, bbox: box.bbox.map(n => n / scale),
      })) };
      samples.push(entry); if (samples.length > MAX_SAMPLES) samples.shift();
      totalSamples++; peak = Math.max(peak, people.length); if (ball) ballSamples++;
      text("local-samples", totalSamples); text("local-peak", peak); text("local-ball", ballSamples);
      view.vision = { people: people.length, ball_visible: ball, calibrated: false, fps: 1000 / Math.max(1, elapsed),
        inference_ms: elapsed, tracks: people };
      studio().receiveVision(view.vision);
      text("track-instruction", "People detected in the image. Track numbers are provisional.");
      text("track-foot-note", "On-device detection · no identity or team assignment");
      for (const row of $("track-list").children) row.disabled = true;
      drawOverlay(); localStatus();
    } catch (error) {
      if (generation !== epoch || mode !== "video") return;
      detectorState = "error"; detectorError = error.message;
      console.error("NBA detector inference failed", error);
      text("local-analysis-note", "Detection stopped on this device. Video playback still works. Try the detector again.");
      $("local-retry").hidden = false; render();
    } finally { inferenceBusy = false; scheduleSample(); }
  }

  function exportJson() {
    if (mode === "idle") return;
    const data = mode === "video" ? {
      source: "on-device-browser-detection", video: view.active, detector: "NBA-trained BallPersonNet (original weights)",
      summary: { frames_sampled: totalSamples, peak_people: peak, ball_detections: ballSamples },
      checkpoint_sha256: model?.checkpoint, backend: model?.backend,
      coordinates: "Source video pixels: [x, y, width, height]", retained_sample_limit: MAX_SAMPLES,
      samples, error: detectorError || null,
    } : { source: "synthetic-demo", meta: cv().S.meta, frame: cv().S.frame, stats: cv().S.stats,
      events: [...cv().S.events.values()] };
    const blob = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], { type: "application/json" }));
    const link = document.createElement("a"); link.href = blob; link.download = "courtvision-analysis.json"; link.click();
    setTimeout(() => URL.revokeObjectURL(blob), 1000);
    cv().toast("Analysis exported");
  }

  video().addEventListener("loadedmetadata", () => {
    if (mode !== "video") return;
    if (!(Number.isFinite(video().duration) && video().duration > 0 && video().videoWidth > 0)) {
      text("local-analysis-note", "This file has no readable video track. Choose an MP4 (H.264) or WebM clip.");
      return;
    }
    if (video().duration > 30) { clearSession(); studio().openDialog(); text("studio-upload-status", "Clips must be 30 seconds or shorter."); return; }
    Object.assign(view.active, { duration_s: video().duration, width: video().videoWidth, height: video().videoHeight });
    cv().dispatch({ type: "meta", t: 0, teams: { home: "Home", away: "Away" }, roster: [], court: {}, rules: null });
    localStatus(); captureLoop(); prepareDetector();
  });
  for (const name of ["play", "pause", "ended", "timeupdate", "ratechange", "volumechange"]) video().addEventListener(name, localStatus);
  video().addEventListener("play", () => { autoPlayPending = false; });
  video().addEventListener("seeking", () => {
    sampleEpoch++; tracks = []; lastBoxes = []; lastSample = -1; frameHistory = []; drawOverlay();
  });
  video().addEventListener("seeked", scheduleSample);
  video().addEventListener("error", () => {
    if (mode !== "video") return;
    sampleEpoch++; clearTimeout(sampleTimer);
    text("local-analysis-note", "This browser cannot decode this video. Try MP4 (H.264) or WebM, or choose a different clip.");
    cv().toast("Video could not open. Try an MP4 (H.264) or WebM clip.", "error");
  });
  $("local-retry").addEventListener("click", () => { lastSample = -1; prepareDetector(); });
  $("studio-dialog").addEventListener("close", () => { choice++; });
  $("studio-overlay").addEventListener("change", () => { if (enabled) render(); });
  window.addEventListener("pagehide", () => { clearTimeout(sampleTimer); clearTimeout(bufferTimer); video().pause(); if (url) URL.revokeObjectURL(url); });
  window.CVVisitor = { get enabled() { return enabled; }, get mode() { return mode; },
    workspaceUpdate, receive, control, startDemo, togglePlay, openFile, exit, exportJson };
})();
