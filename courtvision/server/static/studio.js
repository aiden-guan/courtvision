/* Video workspace controls. Video bytes stay on this computer; server messages are text only. */
(() => {
  "use strict";
  const $ = id => document.getElementById(id);
  let workspace = null, selected = null, busy = false, pending = null, activeId = null;
  let frameUrl = null, frameBusy = false, stopped = false, lastStatus = null, lastVision = null;
  let isStudio = true, lastVideoKey = "", processedT = 0;
  let lastMonitorKey = "";
  let focusedTrack = "", trackKey = "";
  const headers = extra => {
    let token = "";
    try { token = sessionStorage.getItem("cv-token") || ""; } catch { /* optional */ }
    return { ...extra, ...(token ? { Authorization: `Bearer ${token}` } : {}) };
  };
  const text = (id, value) => { if ($(id).textContent !== String(value)) $(id).textContent = value; };
  const setVision = (id, value, tone) => { text(id, value); $(id).dataset.tone = tone; };
  const progress = value => { $("studio-progress").value = value; $("monitor-progress").style.setProperty("--p", value); };
  const pad = n => String(n).padStart(2, "0");
  // Key plays drawn on the feed timeline, keyed by event id so revisions replace earlier marks.
  const MARK_KINDS = new Set(["shot_made", "shot_missed", "steal", "block", "turnover", "foul", "period_start", "period_end"]);
  const marks = new Map();
  let marksQueued = false;
  const clock = value => {
    if (!Number.isFinite(value)) return "--:--";
    const sec = Math.max(0, Math.floor(value));
    return `${String(Math.floor(sec / 60)).padStart(2, "0")}:${String(sec % 60).padStart(2, "0")}`;
  };
  async function api(path, data) {
    const response = await fetch(path, {
      method: data === undefined ? "GET" : "POST", headers: headers(data === undefined ? {} : { "Content-Type": "application/json" }),
      ...(data === undefined ? {} : { body: JSON.stringify(data) }), cache: "no-store",
    });
    let result;
    try { result = await response.json(); } catch { result = {}; }
    if (!response.ok) {
      const detail = typeof result.detail === "string" ? result.detail : `Request failed (${response.status}).`;
      const error = new Error(detail); error.status = response.status; throw error;
    }
    return result;
  }
  function notify(message, calibration = false) {
    const notice = $("studio-notice");
    notice.replaceChildren(document.createTextNode(message));
    if (calibration) {
      const button = document.createElement("button");
      button.type = "button"; button.textContent = "Map court";
      button.addEventListener("click", setupCourt); notice.append(button);
    }
    notice.hidden = !message;
  }
  function uploadStatus(message, error = false) {
    text("studio-upload-status", message);
    $("studio-upload-status").classList.toggle("error", error);
  }
  function openDialog() {
    if (!isStudio) { notify("Open the video workspace with: courtvision studio --open"); return; }
    if (!$("studio-dialog").open) $("studio-dialog").showModal();
  }
  function chooseFile(file) {
    if (!file || busy) return;
    if (file.size > 8 * 1024 ** 3) { uploadStatus("This video exceeds 8 GB. Trim it into smaller clips.", true); return; }
    selected = file;
    text("studio-filename", file.name);
    text("studio-file-detail", `${(file.size / 1024 ** 2).toFixed(1)} MB · Ready to analyze`);
    $("studio-start").disabled = false;
    uploadStatus("");
  }
  function receiveStatus(status) {
    if (!status) return;
    lastStatus = status;
    const hasActive = !!workspace?.active || !isStudio;
    $("studio-play").disabled = !hasActive;
    $("studio-speed").disabled = !hasActive || !!status.finished;
    $("studio-play").dataset.state = status.finished ? "restart" : status.paused ? "play" : "pause";
    $("studio-play").setAttribute("aria-label", status.finished ? "Restart analysis" : status.paused ? "Resume analysis" : "Pause analysis");
    if (Number.isFinite(status.speed)) $("studio-speed").value = String(status.speed);
    document.body.classList.toggle("has-session", hasActive);
    document.body.dataset.analysis = !hasActive ? "" : status.finished ? "complete" : status.paused ? "paused" : "live";
    const label = status.finished ? "Analysis complete" : status.paused ? "Analysis paused" : "Analysis in progress";
    text("studio-session-label", label);
    if (Number.isFinite(status.t)) processedT = status.t;
    text("studio-time", clock(processedT));
    const duration = workspace?.active?.duration_s;
    if (duration) progress(Math.min(100, processedT / duration * 100));
    if (status.finished) { progress(100); text("studio-throughput", "Analysis complete"); }
    if (status.error || workspace?.error) notify(status.error || workspace.error);
    else if (status.finished && status.needs_calibration) notify("Tracking complete. Reopen this video and map the court to produce spatial events and statistics.");
    else if (status.needs_calibration && !workspace?.active?.demo) {
      notify("Image tracking is active. Map at least four court landmarks to enable spatial events and statistics.", true);
    } else if (hasActive && !workspace?.active?.demo && lastVision?.calibrated) {
      notify(lastVision.gate && lastVision.gate !== "live" ?
        "Court is out of view. Spatial events are paused until the game camera returns." :
        "Estimated events · Generic detector · Verify team assignments, player identities and scoreboard. Replays need review.");
    } else if (workspace?.active?.demo) notify("Simulator demo · This game demonstrates the workspace using synthetic tracking and event data.");
  }
  function receiveVision(vision) {
    if (!vision || !Number.isFinite(vision.people)) return;
    lastVision = vision;
    text("vision-people", Number.isFinite(vision.people) ? vision.people : "—");
    setVision("vision-ball", vision.ball_visible ? "Visible" : "Out of view", vision.ball_visible ? "good" : "idle");
    if (!vision.calibrated) setVision("vision-court", "Needs setup", "warn");
    else if (vision.gate && vision.gate !== "live") setVision("vision-court", "Out of view", "warn");
    else setVision("vision-court", "Mapped", "good");
    text("studio-throughput", Number.isFinite(vision.fps) ? `${vision.fps.toFixed(1)} frames/s · ${Math.round(vision.inference_ms)} ms` : "Analyzing");
    renderTracks(vision.tracks || []);
  }
  function setFocus(id) {
    focusedTrack = focusedTrack === id ? "" : id;
    $("studio-clear-focus").hidden = !focusedTrack;
    if (focusedTrack) $("studio-overlay").checked = true;
    for (const row of $("track-list").children) row.setAttribute("aria-pressed", String(row.dataset.track === focusedTrack));
    lastMonitorKey = "";
  }
  function renderTracks(tracks, synthetic = false) {
    const ids = tracks.map(track => [track.id, track.team]);
    const key = JSON.stringify([ids, synthetic]);
    text("track-count", String(tracks.length).padStart(2, "0"));
    $("track-empty").hidden = !!tracks.length;
    text("track-instruction", synthetic ? "Simulated player positions from the demo." : "Select a track to highlight it in the feed.");
    text("track-foot-note", synthetic ? "Simulator data · Not camera detections" : "Image-space tracking · IDs are provisional");
    if (focusedTrack && !tracks.some(track => track.id === focusedTrack)) setFocus(focusedTrack);
    if (key !== trackKey) {
      trackKey = key;
      $("track-list").replaceChildren();
      for (const track of tracks) {
        const row = document.createElement("button"), marker = document.createElement("span");
        const info = document.createElement("span"), name = document.createElement("strong");
        const detail = document.createElement("small"), value = document.createElement("span");
        row.type = "button"; row.className = "track-row"; row.dataset.track = track.id;
        row.dataset.team = track.team || "unknown";
        row.setAttribute("aria-pressed", String(track.id === focusedTrack));
        row.disabled = synthetic || !isStudio;
        marker.className = "track-marker"; marker.textContent = "⌖"; marker.setAttribute("aria-hidden", "true");
        name.textContent = track.id;
        detail.textContent = synthetic ? "Simulated player" : track.team ? `${track.team === "home" ? "Home" : "Away"} · provisional` : "Person · unassigned team";
        value.className = "track-confidence"; value.title = "Detection confidence; does not verify identity";
        info.append(name, detail); row.append(marker, info, value);
        row.addEventListener("click", () => setFocus(track.id));
        $("track-list").append(row);
      }
    }
    for (let index = 0; index < tracks.length; index++) {
      const confidence = tracks[index].confidence;
      const row = $("track-list").children[index];
      row.lastChild.textContent = Number.isFinite(confidence) ? `${Math.round(confidence * 100)}%` : synthetic ? "SIM" : "—";
      row.style.setProperty("--conf", Number.isFinite(confidence) ? confidence : 0);
    }
  }
  function renderWorkspace(data) {
    workspace = data;
    const active = data.active;
    if (!active && activeId) {
      activeId = null; lastStatus = null; lastVision = null; processedT = 0;
      text("studio-title", "Game studio"); text("studio-subtitle", "Open game footage to start.");
      text("studio-session-label", "Ready for footage"); text("monitor-source", "No footage open");
      document.body.dataset.analysis = "";
      text("studio-time", "00:00"); text("studio-duration", "--:--"); text("studio-throughput", "Awaiting footage");
      text("vision-people", "—"); setVision("vision-ball", "Awaiting footage", "idle"); setVision("vision-court", "Needs setup", "warn");
      renderTracks([]);
      $("studio-empty").hidden = false; $("studio-frame").hidden = true; $("studio-loading").hidden = true;
      $("monitor-badge").hidden = true; $("studio-play").disabled = true; $("studio-speed").disabled = true;
      progress(0); notify(""); marks.clear();
      if (frameUrl) { URL.revokeObjectURL(frameUrl); frameUrl = null; }
    }
    $("studio-export").disabled = !active;
    $("studio-calibrate").disabled = !active;
    document.body.classList.toggle("has-session", !!active);
    if (active) {
      text("studio-title", active.name.replace(/\.[^.]+$/, ""));
      text("studio-subtitle", active.demo ? "Simulated game · synthetic tracking and events" : "Your footage · analyzed on this Mac");
      text("monitor-source", active.demo ? "SIMULATOR DEMO" : `${active.width} × ${active.height} · ${Number(active.fps).toFixed(0)} fps source`);
      text("studio-duration", clock(active.duration_s));
      $("monitor-badge").hidden = false;
      text("monitor-badge", active.demo ? "SIMULATOR DEMO" : "COMPUTER VISION · ESTIMATED");
      if (active.run_id !== activeId) {
        activeId = active.run_id; processedT = 0;
        focusedTrack = ""; $("studio-clear-focus").hidden = true; renderTracks([]);
        if (frameUrl) { URL.revokeObjectURL(frameUrl); frameUrl = null; }
        $("studio-frame").hidden = true;
        $("studio-loading").hidden = false;
        $("studio-empty").hidden = true;
      }
    }
    receiveVision(data.vision); receiveStatus(data.status); queueMarks();
    if (active?.demo && data.frame) {
      text("vision-people", (data.frame.players || []).length);
      setVision("vision-ball", data.frame.ball?.visible ? "Visible" : "Out of view", data.frame.ball?.visible ? "good" : "idle");
      setVision("vision-court", "Simulator map", "good");
      renderTracks(data.frame.players || [], true);
    }
    const model = data.model;
    text("studio-model-note", model.state === "ready" ? (model.name.includes("COCO") ? "Local detector ready · Generic person and ball detections. Review results before using them as statistics." : "Configured local detector ready. Review its detections on this footage.") :
      model.state === "loading" ? "Preparing the local detector. The first download is about 350 MB." :
      model.error || "First use prepares a local detector (about 350 MB). Generic player and ball detections need review.");
    const jevMode = data.status?.jev;
    text("studio-cloud", jevMode === "live" ? "Jev live mode · Local vision" : data.capabilities.jev ? "Jev key configured · Ready for next session" : "Local vision · Jev key not configured");
    const key = JSON.stringify((data.videos || []).map(v => v.id));
    if (key !== lastVideoKey) {
      lastVideoKey = key;
      $("studio-video-list").replaceChildren();
      for (const video of (data.videos || []).slice(0, 8)) {
        const button = document.createElement("button");
        button.type = "button"; button.className = "recent-video";
        const name = document.createElement("span"), duration = document.createElement("span");
        name.textContent = video.name; duration.textContent = `${clock(video.duration_s)}  ↗`;
        button.append(name, duration); button.addEventListener("click", () => queueStart(video.id));
        $("studio-video-list").append(button);
      }
      $("studio-recent").hidden = !(data.videos || []).length;
    }
  }
  function renderMarks() {
    marksQueued = false;
    const duration = workspace?.active?.duration_s, nodes = [];
    if (duration > 0) for (const mark of marks.values()) {
      const node = document.createElement("i");
      node.style.left = `${Math.min(100, Math.max(0, mark.t / duration * 100))}%`;
      node.dataset.kind = mark.kind;
      if (mark.team) node.dataset.team = mark.team;
      nodes.push(node);
    }
    $("timeline-marks").replaceChildren(...nodes);
  }
  function queueMarks() { if (!marksQueued) { marksQueued = true; requestAnimationFrame(renderMarks); } }
  function tick() {
    const now = new Date(), time = `${pad(now.getHours())}:${pad(now.getMinutes())}:${pad(now.getSeconds())}`;
    text("sys-clock", time);
    text("feed-clock", `${now.getFullYear()}-${pad(now.getMonth() + 1)}-${pad(now.getDate())} ${time}`);
  }
  async function poll() {
    try {
      const data = await api("api/studio");
      renderWorkspace(data);
      if (pending && data.model.state === "ready") {
        const id = pending; pending = null;
        await startVideo(id);
      } else if (pending && data.model.state === "error") {
        pending = null; busy = false; $("studio-start").disabled = false;
        uploadStatus(data.model.error, true);
      }
    } catch (error) {
      if (!workspace && error.status === 404) {
        isStudio = false; document.body.classList.add("has-session");
        text("studio-title", "Game workspace");
        $("studio-empty").hidden = true;
      }
    } finally { if (!stopped) setTimeout(poll, 1500); }
  }
  async function monitor() {
    try {
      if (frameBusy || document.hidden || (!workspace?.active && isStudio)) return;
      if ($("view-live").hidden) return;
      const monitorKey = `${activeId}:${processedT}:${$("studio-overlay").checked}:${focusedTrack}:${lastStatus?.finished}:${lastStatus?.paused}`;
      if ((lastStatus?.finished || lastStatus?.paused) && frameUrl && lastMonitorKey === monitorKey) return;
      frameBusy = true;
      const requestRun = activeId;
      const path = isStudio ? `api/studio/frame.jpg?overlay=${$("studio-overlay").checked}&focus=${encodeURIComponent(focusedTrack)}` : "api/frame.jpg";
      const response = await fetch(path, { headers: headers(), cache: "no-store" });
      if (!response.ok) return;
      const url = URL.createObjectURL(await response.blob());
      const image = $("studio-frame");
      // Decode before swapping to avoid blank frames and pair the displayed timestamp to this JPEG.
      const preload = new Image(); preload.src = url;
      try { await preload.decode(); } catch { URL.revokeObjectURL(url); return; }
      if (requestRun !== activeId) { URL.revokeObjectURL(url); return; }
      const old = frameUrl; frameUrl = url; image.src = url;
      image.alt = $("studio-overlay").checked ? "Current processed game frame with player and ball tracking" : "Current processed game frame";
      image.hidden = false; $("studio-empty").hidden = true; $("studio-loading").hidden = true;
      if (old) URL.revokeObjectURL(old);
      const videoTime = Number(response.headers.get("x-video-time"));
      if (Number.isFinite(videoTime)) {
        processedT = videoTime;
        text("studio-time", clock(processedT));
        if (workspace?.active?.duration_s) progress(lastStatus?.finished ? 100 : Math.min(100, processedT / workspace.active.duration_s * 100));
      }
      lastMonitorKey = monitorKey;
    } catch { /* Keep the last frame during reconnect. */ }
    finally { frameBusy = false; if (!stopped) setTimeout(monitor, document.hidden ? 2000 : lastStatus?.finished || lastStatus?.paused ? 500 : 125); }
  }
  function upload(file) {
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open("POST", `api/studio/upload?name=${encodeURIComponent(file.name)}`);
      for (const [key, value] of Object.entries(headers({"Content-Type":"application/octet-stream"}))) xhr.setRequestHeader(key, value);
      $("studio-upload-progress").hidden = false;
      xhr.upload.onprogress = event => {
        if (event.lengthComputable) {
          const percent = Math.round(event.loaded / event.total * 100);
          $("studio-upload-progress").value = percent;
          uploadStatus(percent === 100 ? "Checking video…" : `Copying video to your local workspace… ${percent}%`);
        }
      };
      xhr.onload = () => {
        let body = {};
        try { body = JSON.parse(xhr.responseText); } catch { /* handled below */ }
        if (xhr.status >= 200 && xhr.status < 300) resolve(body);
        else reject(new Error(typeof body.detail === "string" ? body.detail : `Upload failed (${xhr.status}).`));
      };
      xhr.onerror = () => reject(new Error("Upload interrupted. Check that the local server is running and retry."));
      xhr.send(file);
    });
  }
  async function queueStart(id) {
    if (busy && !pending) return;
    busy = true; $("studio-start").disabled = true;
    if (workspace?.model.state !== "ready") {
      pending = id;
      uploadStatus("Preparing the detector. Your video will open when it is ready.");
      try { await api("api/studio/prepare", {}); }
      catch (error) { pending = null; busy = false; $("studio-start").disabled = false; uploadStatus(error.message, true); }
    } else await startVideo(id);
  }
  async function startVideo(id) {
    uploadStatus("Opening game feed…");
    try {
      const data = await api("api/studio/start", { video_id:id, home_name:$("studio-home").value,
        away_name:$("studio-away").value, broadcast:$("studio-broadcast").checked });
      renderWorkspace(data); $("studio-dialog").close(); $("tab-live").click();
      $("studio-upload-progress").hidden = true; uploadStatus("");
    } catch (error) { uploadStatus(error.message, true); }
    finally { busy = false; $("studio-start").disabled = !selected; }
  }
  async function setupCourt() {
    if (!workspace?.active && isStudio) { openDialog(); return; }
    try {
      await api("api/control", { action:"pause" });
      $("tab-calibration").click(); $("btn-cal-load").click();
      notify("The video is paused. Mark four or more known court points spread across the visible floor. Apply setup, then resume analysis from Live analysis.");
    } catch (error) { notify(error.message); }
  }
  $("studio-open").addEventListener("click", openDialog);
  $("studio-settings").addEventListener("click", () => $("btn-settings").click());
  $("studio-clear-focus").addEventListener("click", () => setFocus(focusedTrack));
  $("studio-empty-open").addEventListener("click", openDialog);
  $("studio-close").addEventListener("click", () => $("studio-dialog").close());
  $("studio-file").addEventListener("change", e => chooseFile(e.target.files[0]));
  $("studio-calibrate").addEventListener("click", setupCourt);
  $("studio-form").addEventListener("submit", async event => {
    event.preventDefault(); if (!selected || busy) return;
    busy = true; $("studio-start").disabled = true;
    try {
      if (workspace?.model.state !== "ready") await api("api/studio/prepare", {});
      const video = await upload(selected);
      busy = false;
      await queueStart(video.id);
    } catch (error) { busy = false; $("studio-start").disabled = false; uploadStatus(error.message, true); }
  });
  for (const eventName of ["dragenter", "dragover"]) $("studio-dropzone").addEventListener(eventName, e => {
    e.preventDefault(); $("studio-dropzone").classList.add("dragging");
  });
  $("studio-dropzone").addEventListener("dragleave", () => $("studio-dropzone").classList.remove("dragging"));
  $("studio-dropzone").addEventListener("drop", event => {
    event.preventDefault(); $("studio-dropzone").classList.remove("dragging"); chooseFile(event.dataTransfer.files[0]);
  });
  $("studio-demo").addEventListener("click", async () => {
    if (busy) return; busy = true; $("studio-demo").disabled = true;
    try { renderWorkspace(await api("api/studio/start", {demo:true})); }
    catch (error) { notify(error.message); }
    finally { busy = false; $("studio-demo").disabled = false; }
  });
  $("studio-play").addEventListener("click", async () => {
    if (lastStatus?.finished && workspace?.active) {
      if (workspace.active.demo) $("studio-demo").click();
      else await queueStart(workspace.active.id);
      return;
    }
    try { const result = await api("api/control", {action:lastStatus?.paused ? "resume" : "pause"}); receiveStatus(result.result); }
    catch (error) { notify(error.message); }
  });
  $("studio-speed").addEventListener("change", async () => {
    try { await api("api/control", {action:"speed", value:Number($("studio-speed").value)}); }
    catch (error) { notify(error.message); }
  });
  $("studio-fullscreen").addEventListener("click", async () => {
    try { if (document.fullscreenElement) await document.exitFullscreen(); else await $("monitor-screen").requestFullscreen(); }
    catch { notify("Fullscreen is unavailable in this browser. Enlarge the window to expand the feed."); }
  });
  $("studio-export").addEventListener("click", async () => {
    try {
      const response = await fetch("api/studio/export", {headers:headers()});
      if (!response.ok) throw new Error("Export failed. Check the local server and retry.");
      const url = URL.createObjectURL(await response.blob()), a = document.createElement("a");
      a.href = url; a.download = "courtvision-analysis.json"; a.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch (error) { notify(error.message); }
  });
  window.addEventListener("courtvision-message", event => {
    const msg = event.detail;
    if (msg.type === "vision") receiveVision(msg);
    if (msg.type === "status") receiveStatus(msg);
    if (msg.type === "event" && msg.event?.event_id) {
      const e = msg.event;
      if (MARK_KINDS.has(e.kind) && e.status !== "retracted" && Number.isFinite(e.t)) marks.set(e.event_id, { t: e.t, kind: e.kind, team: e.team });
      else marks.delete(e.event_id);
      queueMarks();
    }
    if (msg.type === "reset") {
      focusedTrack = ""; $("studio-clear-focus").hidden = true; renderTracks([]);
      lastVision = null; processedT = 0; progress(0); marks.clear(); queueMarks();
      text("vision-people", "—"); setVision("vision-ball", "Awaiting video", "idle"); setVision("vision-court", "Not configured", "warn");
      $("studio-frame").hidden = true; $("studio-loading").hidden = false;
    }
    if (msg.type === "frame" && (workspace?.active?.demo || !isStudio)) {
      text("vision-people", (msg.players || []).length);
      setVision("vision-ball", msg.ball?.visible ? "Visible" : "Out of view", msg.ball?.visible ? "good" : "idle");
      setVision("vision-court", workspace?.active?.demo ? "Simulator map" : "Mapped", "good");
      renderTracks(msg.players || [], !!workspace?.active?.demo || !isStudio);
      processedT = msg.t;
    }
  });
  window.addEventListener("pagehide", () => { stopped = true; if (frameUrl) URL.revokeObjectURL(frameUrl); });
  poll(); monitor(); tick(); setInterval(tick, 1000);
})();
