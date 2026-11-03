/* Public-site sessions. Video bytes are only read by a blob URL and a local canvas.
 * The server's shared simulator is buffered per viewer; it never controls another visitor's playback.
 * Original Python perception runs in the worker; overlays replay recorded results on their source frames. */
(() => {
  "use strict";
  const $ = id => document.getElementById(id);
  const studio = () => window.CVStudio;
  const cv = () => window.CV;
  const video = () => $("local-video");
  const text = (id, value) => { $(id).textContent = value; };
  let enabled = false, base = null, view = null, mode = "idle", epoch = 0;
  let url = null, model = null, modelPromise = null, detectorState = "idle", detectorError = "";
  let lens = null, demoPaused = false, delay = 0, pausedAt = 0, buffer = [], bufferTimer = 0;
  let modelController = null;
  let samples = [], totalSamples = 0, peak = 0, ballSamples = 0, lastBoxes = [];
  let autoPlayPending = false;
  let captureHandle = 0, choice = 0, clip = null, analysing = false, analysisT = 0, highlight = "";
  let setup = null, setupTimer = 0;

  function renderSetup() {
    $("local-setup").hidden = mode !== "video" || !setup;
    if (!setup || mode !== "video") return;
    const stages = ["model", "runtime", "court"], index = Math.max(0, stages.indexOf(setup.stage));
    const seconds = Math.floor((performance.now() - setup.started) / 1000);
    text("local-setup-elapsed", `${Math.floor(seconds/60)}:${String(seconds%60).padStart(2,"0")} elapsed`);
    text("local-setup-label", setup.label);
    text("local-setup-step", `Step ${index+1} of 3`);
    const meter = $("local-setup-progress"), measured = Number.isFinite(setup.total) && setup.total > 0;
    if (measured) {
      meter.max = setup.total; meter.value = Math.max(0,Math.min(setup.total,setup.completed || 0));
      text("local-setup-amount", `${Math.round(100*meter.value/meter.max)}%`);
    } else {
      meter.removeAttribute("value");
      text("local-setup-amount", setup.stage === "model" && setup.completed > 0 ? `${(setup.completed/1048576).toFixed(1)} MB received` : "Working…");
    }
    for (let i = 0; i < stages.length; i++) {
      const el = $(`setup-stage-${stages[i]}`);
      el.dataset.state = i < index ? "complete" : i === index ? "active" : "pending";
      if (i === index) el.setAttribute("aria-current", "step"); else el.removeAttribute("aria-current");
    }
    text("local-setup-note", clip?.paused ? "Setup continues. Analysis will wait until you resume." :
      "Analysis starts after setup. First use can take a few minutes.");
  }

  function setupTick() {
    clearTimeout(setupTimer);
    if (setup) {
      const seconds = Math.floor((performance.now() - setup.started) / 1000);
      text("local-setup-elapsed", `${Math.floor(seconds/60)}:${String(seconds%60).padStart(2,"0")} elapsed`);
      setupTimer = setTimeout(setupTick, 1000);
    }
  }

  function updateSetup(progress) {
    setup = { started: setup?.started ?? performance.now(), ...progress };
    text("local-analysis-note", progress.label);
    renderSetup(); setupTick();
  }

  function clearSetup() {
    setup = null; clearTimeout(setupTimer); renderSetup();
  }

  function render() {
    if (!view) return;
    studio().renderWorkspace(view, true);
    document.body.dataset.visitor = mode;
    $("local-summary").hidden = mode !== "video";
    $("demo-canvas").hidden = mode !== "demo";
    $("local-video").hidden = mode !== "video" || analysing;
    $("local-analysis-frame").hidden = mode !== "video" || !analysing;
    $("local-overlay").hidden = mode !== "video" || !$("studio-overlay").checked;
    if (mode !== "idle") { $("studio-frame").hidden = true; $("studio-loading").hidden = true; }
    $("studio-speed").hidden = mode !== "video";
    if (mode === "demo") text("studio-throughput", demoPaused ? "Demo paused" : "Synthetic demo");
    if (mode === "video") {
      text("monitor-badge", "ON-DEVICE DETECTION");
      text("monitor-source", "Stays on your device");
      text("studio-session-label", clip?.paused && analysing ? "Analysis paused" : setup ? "Preparing analysis" : analysing ? "Analysing locally" : video().ended ? "Video ended" : video().paused ? "Paused" : "Replay");
      text("studio-throughput", setup ? "Preparing analysis…" : analysing ? "30 fps · local pipeline" : detectorState === "ready" ? "Recorded analysis" : detectorState === "loading" ? "Loading detector…" : "Video preview");
      video().controls = !analysing && detectorState !== "loading";
      $("studio-play").setAttribute("aria-label", video().ended ? "Replay video" : video().paused ? "Play video" : "Pause video");
      text("vision-court", view.vision?.calibrated ? "Mapped locally" : "Image-space tracking");
      if (analysing) {
        const paused = !!clip?.paused;
        $("studio-play").dataset.state = paused ? "play" : "pause";
        $("studio-play").dataset.label = paused ? "Resume" : "Pause";
        $("studio-play").setAttribute("aria-label", paused ? "Resume analysis" : "Pause analysis");
      }
      $("studio-play").disabled = detectorState === "loading";
      $("studio-speed").disabled = analysing || detectorState === "loading";
      text("track-instruction", "Select a track to highlight it in the feed.");
      text("track-foot-note", "Original local tracking · provisional identities");
    }
    text("sys-detector-label", mode === "video" ? detectorState === "ready" ? "NBA model ready" : detectorState === "error" ? "NBA model unavailable" : "Loading NBA model…" : "NBA model · loads on demand");
    renderSetup();
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
    clearSetup();
    epoch++; choice++; autoPlayPending = false;
    if (captureHandle && video().cancelVideoFrameCallback) video().cancelVideoFrameCallback(captureHandle);
    captureHandle = 0;
    clip?.close(); clip = null;
    if (analysing || detectorState === "loading") {
      modelController?.abort(); model?.dispose?.(); model = null; modelPromise = null; detectorState = "idle";
    }
    analysing = false; analysisT = 0; highlight = "";
    clearTimeout(bufferTimer);
    buffer = []; delay = 0; demoPaused = false;
    video().pause(); video().removeAttribute("src"); video().load();
    if (url) URL.revokeObjectURL(url);
    url = null; mode = "idle";
    lastBoxes = [];
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
      if (analysing && (action === "pause" || action === "resume")) {
        if (action === "pause") clip.pause(); else clip.resume();
        localStatus(); return true;
      }
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
    return control((mode === "demo" ? demoPaused : analysing ? clip?.paused : video().paused) ? "resume" : "pause");
  }

  async function loadDetector() {
    if (model) return model;
    if (!modelPromise) {
      modelController = new AbortController();
      const generation = epoch;
      modelPromise = window.CVNBA.load({ signal: modelController.signal, progress: progress => {
        if (generation === epoch && mode === "video") updateSetup(progress);
      } });
    }
    const pending = modelPromise;
    const ready = await pending;
    if (modelPromise === pending) { model = ready; modelPromise = null; }
    return ready;
  }

  async function prepareDetector() {
    const generation = epoch;
    detectorState = "loading"; detectorError = "";
    updateSetup({ stage: "model", label: "Loading the NBA model…" });
    text("local-analysis-note", "Loading the NBA-trained model on this device. Your clip stays here.");
    $("local-retry").hidden = true; render();
    try {
      await loadDetector();
      if (generation !== epoch || mode !== "video") return;
      detectorState = "ready";
      text("local-analysis-note", "Preparing the original local analysis pipeline…");
      render();
      if (autoPlayPending) { autoPlayPending = false; startAnalysis(); }
    } catch (error) {
      if (generation !== epoch || mode !== "video") return;
      detectorState = "error"; detectorError = error.message;
      clearSetup();
      modelPromise = null;
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
    view.status = { t: analysing ? analysisT : video().currentTime,
      paused: analysing ? !!clip?.paused : video().paused, finished: !analysing && video().ended,
      speed: video().playbackRate, muted: video().muted };
    cv().dispatch({ type: "status", ...view.status });
    render();
  }

  function drawOverlay() {
    const canvas = $("local-overlay"), v = video();
    if (mode !== "video" || !v.videoWidth) return;
    canvas.width = v.videoWidth; canvas.height = v.videoHeight;
    const ctx = canvas.getContext("2d"), scale = Math.max(1, v.videoWidth / 1440);
    ctx.lineWidth = 2 * scale;
    ctx.font = `${14 * scale}px sans-serif`;
    for (const box of lastBoxes) {
      const [bx,by,bw,bh] = box.bbox;
      const x = Math.max(0,Math.min(v.videoWidth-1,bx)), y = Math.max(0,Math.min(v.videoHeight-1,by));
      const w = Math.max(0,Math.min(v.videoWidth-1,bx+bw))-x, h = Math.max(0,Math.min(v.videoHeight-1,by+bh))-y;
      if (w <= 0 || h <= 0) continue;
      const ball = box.class === "sports ball", muted = highlight && highlight !== box.id && !ball;
      ctx.lineWidth = (highlight === box.id ? 3 : 2) * scale;
      ctx.strokeStyle = muted ? "#6a706e" : ball ? "#f8e069" : box.class === "referee" ? "#c9cfcc" : box.team === "home" ? "#78bbed" : box.team === "away" ? "#f6bf75" : "#d1eb97";
      if (ball) {
        const cx = x+w/2, cy = y+h/2;
        ctx.beginPath(); ctx.arc(cx,cy,Math.max(9*scale,w/2+4*scale),0,Math.PI*2); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(cx-4*scale,cy); ctx.lineTo(cx+4*scale,cy); ctx.moveTo(cx,cy-4*scale); ctx.lineTo(cx,cy+4*scale); ctx.stroke();
        continue;
      }
      const len = Math.max(3, Math.min(20*scale,w/3,h/4));
      for (const [xx,dx] of [[x,1],[x+w,-1]]) for (const [yy,dy] of [[y,1],[y+h,-1]]) {
        ctx.beginPath(); ctx.moveTo(xx,yy+dy*len); ctx.lineTo(xx,yy); ctx.lineTo(xx+dx*len,yy); ctx.stroke();
      }
      if (muted) continue;
      ctx.beginPath(); ctx.ellipse(x+w/2,y+h,Math.max(6*scale,w/3),4*scale,0,0,Math.PI*2); ctx.stroke();
      const label = `${box.id}${Number.isFinite(box.confidence) ? ` / ${Math.round(box.confidence*100)}%` : ""}`;
      const tw = ctx.measureText(label).width + 12*scale, xx = Math.max(0,Math.min(x,v.videoWidth-tw));
      const yy = Math.max(24*scale,y-5*scale);
      ctx.fillStyle = "#181e1b"; ctx.fillRect(xx,yy-22*scale,tw,24*scale);
      ctx.fillStyle = ctx.strokeStyle; ctx.fillText(label,xx+6*scale,yy-5*scale);
    }
  }

  function showFrame(result) {
    lastBoxes = result?.detections || [];
    const people = lastBoxes.filter(p => p.class === "person"), ball = lastBoxes.some(p => p.class === "sports ball");
    view.vision = { people: people.length, ball_visible: ball, calibrated: !!result?.calibrated,
      gate: result?.gate, tracks: people };
    drawOverlay();
  }

  function replayFrame(t = video().currentTime) {
    if (mode !== "video" || analysing) return;
    const record = clip?.at(t);
    showFrame(record && Math.abs(record.t-t) < .075 ? record : null);
    localStatus();
  }

  function replayLoop() {
    if (mode !== "video" || analysing || !video().requestVideoFrameCallback) return;
    captureHandle = video().requestVideoFrameCallback((_, metadata) => { replayFrame(metadata.mediaTime); replayLoop(); });
  }

  function startAnalysis() {
    const generation = epoch;
    video().pause();
    analysing = true; analysisT = 0; samples = []; totalSamples = 0; peak = 0; ballSamples = 0;
    clip?.close();
    clip = window.CVNBAClip.create(video(), model, {
      state: () => { if (generation === epoch) localStatus(); },
      phase: message => { if (generation === epoch) text("local-analysis-note", message); },
      setup: progress => { if (generation === epoch) updateSetup(progress); },
      preview: pixels => {
        if (generation !== epoch) return;
        const canvas = $("local-analysis-frame");
        canvas.width = pixels.width; canvas.height = pixels.height;
        canvas.getContext("2d").putImageData(pixels,0,0);
      },
      frame: (result, pixels, through) => {
        if (generation !== epoch) return;
        if (setup) clearSetup();
        const canvas = $("local-analysis-frame");
        canvas.width = pixels.width; canvas.height = pixels.height;
        canvas.getContext("2d").putImageData(pixels,0,0);
        analysisT = through; showFrame(result);
        const people = lastBoxes.filter(p => p.class === "person").length, ball = lastBoxes.some(p => p.class === "sports ball");
        samples.push({ ...result, people, ball_visible: ball });
        totalSamples++; peak = Math.max(peak,people); if (ball) ballSamples++;
        text("local-samples",totalSamples); text("local-peak",peak); text("local-ball",ballSamples);
        localStatus();
      },
      complete: () => {
        if (generation !== epoch) return;
        analysing = false;
        clearSetup();
        text("local-analysis-note", "Analysis complete. Playback uses the original pipeline’s recorded frame-by-frame results.");
        video().currentTime = 0; replayFrame(0); replayLoop();
        video().play().catch(() => { text("local-analysis-note", "Analysis complete. Press play to view the recorded results."); });
      },
      error: error => {
        if (generation !== epoch) return;
        analysing = false; lastBoxes = []; drawOverlay();
        clearSetup();
        detectorState = "error"; detectorError = error.message;
        console.error("Original NBA pipeline failed", error);
        text("local-analysis-note", "Analysis stopped on this device. Retry the analysis or choose another clip.");
        $("local-retry").hidden = false; localStatus();
      },
    });
    localStatus(); clip.run();
  }

  function exportJson() {
    if (mode === "idle") return;
    const data = mode === "video" ? {
      source: "on-device-browser-detection", video: view.active, detector: "NBA-trained BallPersonNet + original Python perception pipeline",
      summary: { frames_analysed: totalSamples, peak_people: peak, ball_detections: ballSamples },
      checkpoint_sha256: model?.checkpoint, backend: model?.backend,
      backend_reference_checked: !!model?.validated,
      pipeline: "Original Python preprocessing, court registration, ByteTrack and BallTracker",
      pipeline_sha256: model?.source, analysis_fps: 29.97,
      coordinates: "Source video pixels: [x, y, width, height]",
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
    localStatus(); prepareDetector();
  });
  for (const name of ["play", "pause", "ended", "timeupdate", "ratechange", "volumechange"]) video().addEventListener(name, localStatus);
  video().addEventListener("seeking", () => {
    if (!analysing) replayFrame();
  });
  video().addEventListener("seeked", () => { if (!analysing) replayFrame(); });
  video().addEventListener("error", () => {
    if (mode !== "video") return;
    text("local-analysis-note", "This browser cannot decode this video. Try MP4 (H.264) or WebM, or choose a different clip.");
    cv().toast("Video could not open. Try an MP4 (H.264) or WebM clip.", "error");
  });
  $("local-retry").addEventListener("click", () => {
    video().pause();
    clip?.close(); modelController?.abort(); model?.dispose?.(); model = null; modelPromise = null;
    analysing = false;
    autoPlayPending = true; prepareDetector();
  });
  window.addEventListener("cv-focus", event => { highlight = event.detail.id || ""; drawOverlay(); });
  video().addEventListener("timeupdate", () => { if (!analysing && !video().requestVideoFrameCallback) replayFrame(); });
  $("studio-dialog").addEventListener("close", () => { choice++; });
  $("studio-overlay").addEventListener("change", () => { if (enabled) render(); });
  window.addEventListener("pagehide", () => { clearSetup(); clip?.close(); clearTimeout(bufferTimer); video().pause(); if (url) URL.revokeObjectURL(url); });
  window.CVVisitor = { get enabled() { return enabled; }, get mode() { return mode; },
    workspaceUpdate, receive, control, startDemo, togglePlay, openFile, exit, exportJson };
})();
