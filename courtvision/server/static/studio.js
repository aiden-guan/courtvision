/* Video workspace controls. Video bytes stay on this computer; server messages are text only. */
(() => {
  "use strict";
  const $ = id => document.getElementById(id);
  let workspace = null, selected = null, busy = false, pending = null, activeId = null;
  let frameUrl = null, frameBusy = false, stopped = false, lastStatus = null, lastVision = null;
  let isStudio = true, lastVideoKey = "", processedT = 0;
  let lastMonitorKey = "";
  let focusedTrack = "", trackKey = "";
  let notice = { message: "", calibration: false };
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
  const changed = () => window.dispatchEvent(new CustomEvent("cv-studio"));
  async function api(path, data, method) {
    const response = await fetch(path, {
      method: method || (data === undefined ? "GET" : "POST"),
      headers: headers(data === undefined ? {} : { "Content-Type": "application/json" }),
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
  // Workspace notices are composed into the briefing (static/genui.js), not a banner.
  function notify(message, calibration = false) {
    if (notice.message === message && notice.calibration === calibration) return;
    notice = { message: message || "", calibration };
    changed();
  }
  function uploadStatus(message, error = false) {
    text("studio-upload-status", message);
    $("studio-upload-status").classList.toggle("error", error);
  }
  function openDialog() {
    if (!isStudio) { notify("Open the video workspace with: courtvision studio --open"); return; }
    if (!$("studio-dialog").open) $("studio-dialog").showModal();
    loadLibrary();
  }
  const titleOf = video => video?.title || (video?.name || "Game video").replace(/\.[^.]+$/, "");
  const size = bytes => !Number.isFinite(bytes) ? "" : bytes >= 1024 ** 3 ? `${(bytes / 1024 ** 3).toFixed(1)} GB` :
    bytes >= 1024 ** 2 ? `${Math.round(bytes / 1024 ** 2)} MB` : `${Math.max(1, Math.round(bytes / 1024))} KB`;
  function replayText(replay, duration) {
    const st = replay?.state;
    if (st === "done") return { text: "Replay ready", tone: "good" };
    if (st === "running" || st === "loading") {
      const pct = duration ? Math.floor(Math.min(100, (replay.t_done || 0) / duration * 100)) : 0;
      return { text: `Preparing replay · ${pct}%`, tone: "busy" };
    }
    if (st === "queued") return { text: "Replay queued", tone: "busy" };
    if ((st === "stopped" || st === "failed") && replay.t_done > 0) return { text: `Replay up to ${clock(replay.t_done)}`, tone: "idle" };
    return { text: "No replay yet", tone: "idle" };
  }

  // ------------------------------------------------------------------ the video library (the Videos dialog)
  let library = null, confirming = "", renaming = "";
  async function loadLibrary() {
    try { library = await api("api/studio/library"); renderLibrary(); }
    catch (error) { uploadStatus(error.message, true); }
  }
  function button(label, cls, onClick, title = "") {
    const b = document.createElement("button");
    b.type = "button"; b.className = cls; b.textContent = label; if (title) b.title = title;
    b.addEventListener("click", onClick);
    return b;
  }
  function renderLibrary() {
    const videos = library?.videos || [];
    $("library").hidden = !videos.length;
    $("library-add-title").hidden = !videos.length;
    text("library-total", videos.length ? `${videos.length} video${videos.length === 1 ? "" : "s"} · ${size(library.total_bytes)} on this Mac` : "");
    const byId = new Map(videos.map(v => [v.id, v]));
    const rows = videos.map(video => {
      const li = document.createElement("li"), main = document.createElement("div"), meta = document.createElement("p");
      const actions = document.createElement("div");
      li.className = "library-row"; li.dataset.video = video.id;
      main.className = "library-main"; actions.className = "library-actions";
      if (renaming === video.id) {
        const input = document.createElement("input");
        input.type = "text"; input.maxLength = 120; input.value = titleOf(video); input.className = "library-rename";
        input.setAttribute("aria-label", "Video name");
        const save = async () => {
          const value = input.value.trim(); renaming = "";
          if (value && value !== titleOf(video)) {
            try { await api(`api/studio/video/${video.id}/rename`, { title: value }); }
            catch (error) { window.CV?.toast(error.message, "error"); }
          }
          await loadLibrary(); poll(false);
        };
        input.addEventListener("keydown", e => {
          if (e.key === "Enter") { e.preventDefault(); save(); }
          if (e.key === "Escape") { e.preventDefault(); e.stopPropagation(); renaming = ""; renderLibrary(); }
        });
        input.addEventListener("blur", save);
        main.append(input); setTimeout(() => { input.focus(); input.select(); });
      } else {
        const name = document.createElement("strong");
        name.textContent = titleOf(video); name.title = video.name || "";
        main.append(name);
      }
      const replay = replayText(video.replay, video.duration_s);
      const status = document.createElement("span");
      status.className = "library-tag"; status.dataset.tone = replay.tone; status.textContent = replay.text;
      const bits = [clock(video.duration_s), size(video.disk_bytes)];
      meta.className = "library-meta"; meta.textContent = bits.filter(Boolean).join(" · ") + " · ";
      meta.append(status);
      if (video.open) { const t = document.createElement("span"); t.className = "library-tag"; t.dataset.tone = "good"; t.textContent = "Open now"; meta.append(" ", t); }
      if (video.copies?.length) {
        const older = video.copies.map(id => byId.get(id)).filter(Boolean).filter(w => (w.created || 0) < (video.created || 0));
        if (older.length) {
          const t = document.createElement("span"); t.className = "library-tag"; t.dataset.tone = "warn";
          t.textContent = "Duplicate"; t.title = "The same video file was added more than once. Deleting a duplicate frees its space.";
          meta.append(" ", t);
        }
      }
      if (video.used_by?.length) {
        const t = document.createElement("span"); t.className = "library-tag"; t.dataset.tone = "idle";
        t.textContent = "Kept for research"; t.title = `The game folder ${video.used_by.join(", ")} (runs/nba) uses this file, so it cannot be deleted here.`;
        meta.append(" ", t);
      }
      main.append(meta);
      if (confirming === video.id) {
        const ask = document.createElement("p");
        ask.className = "library-confirm";
        ask.textContent = `Delete this video and everything made from it (${size(video.disk_bytes + (video.saved_bytes || 0))})? This cannot be undone.`;
        main.append(ask);
        actions.append(button("Keep", "btn btn-sm", () => { confirming = ""; renderLibrary(); }),
          button("Delete", "btn btn-sm btn-danger", () => deleteVideo(video)));
      } else {
        actions.append(button(video.open ? "Showing" : "Open", "btn btn-sm btn-primary", () => openVideo(video.id)));
        actions.lastChild.disabled = !!video.open;
        const more = document.createElement("div");
        more.className = "library-more";
        more.append(button("Rename", "link-btn", () => { renaming = video.id; confirming = ""; renderLibrary(); }));
        if (video.replay_bytes > 0 && !["running", "loading", "queued"].includes(video.replay?.state)) {
          more.append(button("Delete replay", "link-btn", () => deleteReplay(video),
            `Frees ${size(video.replay_bytes)}. The video stays; you can prepare the replay again.`));
        }
        const del = button("Delete", "link-btn danger", () => { confirming = video.id; renaming = ""; renderLibrary(); });
        if (video.used_by?.length) { del.disabled = true; del.title = "Kept: a research game folder uses this file."; }
        more.append(del);
        actions.append(more);
      }
      li.append(main, actions);
      return li;
    });
    $("library-list").replaceChildren(...rows);
    const extra = library?.other_runs || 0;
    $("library-extra").hidden = !extra;
    text("library-extra-text", extra ? `Saved results of ${extra} earlier live analys${extra === 1 ? "is" : "es"} (demo or older runs) · ${size(library.other_bytes)}` : "");
  }
  async function openVideo(id) {
    if ($("studio-dialog").open) $("studio-dialog").close();
    await queueStart(id);
  }
  async function deleteVideo(video) {
    confirming = "";
    try {
      const result = await api(`api/studio/video/${video.id}`, undefined, "DELETE");
      window.CV?.toast(`Deleted “${titleOf(video)}” · ${size(result.freed_bytes)} freed`);
    } catch (error) { window.CV?.toast(error.message, "error"); }
    await loadLibrary(); poll(false);
  }
  async function deleteReplay(video) {
    try {
      const result = await api(`api/studio/replay/${video.id}`, undefined, "DELETE");
      window.CV?.toast(`Replay deleted · ${size(result.freed_bytes)} freed`);
    } catch (error) { window.CV?.toast(error.message, "error"); }
    await loadLibrary(); poll(false);
  }
  function chooseFile(file) {
    if (!file || busy) return;
    if (file.size > 8 * 1024 ** 3) { uploadStatus("This video is over 8 GB. Trim it into shorter clips.", true); return; }
    selected = file;
    text("studio-filename", file.name);
    text("studio-file-detail", `${size(file.size)} · ready to add`);
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
    text("studio-session-label", status.finished ? "Analysis complete" : status.paused ? "Paused" : "Analysing");
    if (Number.isFinite(status.t)) processedT = status.t;
    text("studio-time", clock(processedT));
    const duration = workspace?.active?.duration_s;
    if (duration) progress(Math.min(100, processedT / duration * 100));
    if (status.finished) { progress(100); text("studio-throughput", "Analysis complete"); }
    if (status.error || workspace?.error) notify(status.error || workspace.error);
    else if (status.finished && status.needs_calibration) notify("Tracking complete. Reopen this video and map the court to produce spatial plays and statistics.");
    else if (status.needs_calibration && !workspace?.active?.demo) {
      notify("Image tracking is running. Map at least four court landmarks to unlock spatial plays and statistics.", true);
    } else if (hasActive && !workspace?.active?.demo && lastVision?.calibrated) {
      notify(lastVision.gate && lastVision.gate !== "live" ?
        "The court is out of view. Spatial plays pause until the game camera returns." :
        "Estimated plays from a generic detector. Verify teams, player identities and the scoreboard; replays need review.");
    } else if (workspace?.active?.demo) notify("Simulator demo: synthetic tracking and plays that show the workspace.");
    changed();
  }
  function receiveVision(vision) {
    if (!vision || !Number.isFinite(vision.people)) return;
    lastVision = vision;
    text("vision-people", Number.isFinite(vision.people) ? vision.people : "—");
    setVision("vision-ball", vision.ball_visible ? "Visible" : "Out of view", vision.ball_visible ? "good" : "idle");
    if (!vision.calibrated) setVision("vision-court", "Needs setup", "warn");
    else if (vision.gate && vision.gate !== "live") setVision("vision-court", "Out of view", "warn");
    else setVision("vision-court", "Mapped", "good");
    const fps = workspace?.active?.fps;
    text("studio-throughput", !Number.isFinite(vision.fps) ? "Analysing" : fps > 0 ?
      `Analysing at ${(vision.fps / fps).toFixed(2)}× real time` : `Analysing ${vision.fps.toFixed(1)} frames a second`);
    $("studio-throughput").title = Number.isFinite(vision.fps) ? `${vision.fps.toFixed(1)} frames a second · ${Math.round(vision.inference_ms)} ms a frame` : "";
    renderTracks(vision.tracks || []);
  }
  function setFocus(id) {
    focusedTrack = focusedTrack === id ? "" : id;
    $("studio-clear-focus").hidden = !focusedTrack;
    if (focusedTrack) $("studio-overlay").checked = true;
    for (const row of $("track-list").children) row.setAttribute("aria-pressed", String(row.dataset.track === focusedTrack));
    lastMonitorKey = "";
    window.dispatchEvent(new CustomEvent("cv-focus", { detail: { id: focusedTrack } }));
  }
  function renderTracks(tracks, synthetic = false) {
    const ids = tracks.map(track => [track.id, track.team]);
    const key = JSON.stringify([ids, synthetic]);
    text("track-count", String(tracks.length).padStart(2, "0"));
    $("track-empty").hidden = !!tracks.length;
    text("track-instruction", synthetic ? "Simulated player positions from the demo." : "Select a track to highlight it in the feed.");
    text("track-foot-note", synthetic ? "Simulator data · not camera detections" : "Image-space tracking · IDs are provisional");
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
        marker.className = "track-marker"; marker.setAttribute("aria-hidden", "true");
        name.textContent = track.jersey != null && synthetic ? `#${track.jersey} · ${track.id}` : track.id;
        detail.textContent = synthetic ? "Simulated player" : track.team ? `${track.team === "home" ? "Home" : "Away"} · provisional` : "Person · team not assigned";
        value.className = "track-confidence"; value.title = "Detection confidence; it does not verify identity";
        info.append(name, detail); row.append(marker, info, value);
        row.addEventListener("click", () => setFocus(track.id));
        $("track-list").append(row);
      }
    }
    for (let index = 0; index < tracks.length; index++) {
      const confidence = tracks[index].confidence;
      const row = $("track-list").children[index];
      row.lastChild.textContent = Number.isFinite(confidence) ? `${Math.round(confidence * 100)}%` : synthetic ? "SIM" : "—";
      row.lastChild.style.setProperty("--conf", Number.isFinite(confidence) ? confidence : 0);
    }
  }
  function renderSystem(data) {
    const model = data.model;
    const ready = model.state === "ready", loading = model.state === "loading", failed = model.state === "error";
    $("sys-detector").dataset.tone = ready ? "good" : loading ? "busy" : failed ? "bad" : "idle";
    const nba = (model.name || "").startsWith("NBA"), coco = (model.name || "").includes("COCO");
    text("sys-detector-label", ready ? (model.name || "Detector ready") : loading ? "Loading the detector…" : failed ? "Detector setup failed" :
      nba ? "NBA detector · loads when needed" : "Detector not prepared");
    $("sys-detector").title = model.error || model.name || "";
    text("studio-model-note", ready ? (coco ? "Generic detector: person and ball detections need checking before you use them as statistics." :
      nba ? "NBA detector ready. Plays are estimates: check the flagged ones." : "Configured detector ready. Check its detections on this footage.") :
      loading ? (coco ? "Getting the detector ready. The first download is about 350 MB." : "Loading the detector: a few seconds.") :
      model.error || (nba ? "The NBA detector loads when you open a video: a few seconds, nothing to download." :
        "The first video downloads a generic detector (about 350 MB). Its player and ball detections need checking."));
    const jevMode = data.status?.jev;
    const claude = !!data.capabilities?.claude;
    text("studio-cloud", jevMode === "live" ? "Jev live · local vision" : data.capabilities?.jev ? (claude ? "Jev and Claude keys set" : "Jev key set · next session") : claude ? "Claude key set" : "Local only · no cloud keys");
    $("sys-cloud").dataset.tone = jevMode === "live" || claude || data.capabilities?.jev ? "good" : "idle";
  }
  function renderRecent(data) {
    const key = JSON.stringify((data.videos || []).map(v => [v.id, titleOf(v), replayText(v.replay, v.duration_s).text]));
    if (key === lastVideoKey) return;
    lastVideoKey = key;
    $("studio-video-list").replaceChildren();
    for (const video of (data.videos || []).slice(0, 6)) {
      const button = document.createElement("button");
      const replay = replayText(video.replay, video.duration_s);
      button.type = "button"; button.className = "recent-video"; button.title = `Open ${titleOf(video)} · ${replay.text}`;
      button.dataset.replay = replay.tone;
      const thumb = document.createElement("span"), name = document.createElement("span"), duration = document.createElement("span");
      thumb.className = "rv-thumb"; thumb.setAttribute("aria-hidden", "true");
      name.className = "rv-name"; name.textContent = titleOf(video);
      duration.className = "rv-dur"; duration.textContent = clock(video.duration_s);
      button.append(thumb, name, duration); button.addEventListener("click", () => queueStart(video.id));
      $("studio-video-list").append(button);
    }
    $("studio-recent").hidden = !(data.videos || []).length;
  }
  function renderWorkspace(data) {
    workspace = data;
    const active = data.active;
    if (!active && activeId) {
      activeId = null; lastStatus = null; lastVision = null; processedT = 0;
      text("studio-title", "Game studio");
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
      text("studio-title", titleOf(active));
      text("monitor-source", active.demo ? "Simulator demo" : `${active.width}×${active.height} · ${Number(active.fps).toFixed(0)} fps`);
      text("studio-duration", clock(active.duration_s));
      $("monitor-badge").hidden = false;
      text("monitor-badge", active.demo ? "SIMULATOR DEMO" : "ESTIMATED FROM VIDEO");
      if (active.run_id !== activeId) {
        activeId = active.run_id; processedT = 0;
        focusedTrack = ""; $("studio-clear-focus").hidden = true; renderTracks([]);
        if (frameUrl) { URL.revokeObjectURL(frameUrl); frameUrl = null; }
        $("studio-frame").hidden = true;
        $("studio-loading").hidden = false;
        $("studio-empty").hidden = true;
      }
    } else if (!isStudio) $("studio-empty").hidden = true;
    receiveVision(data.vision); receiveStatus(data.status); queueMarks();
    if (active?.demo && data.frame) {
      text("vision-people", (data.frame.players || []).length);
      setVision("vision-ball", data.frame.ball?.visible ? "Visible" : "Out of view", data.frame.ball?.visible ? "good" : "idle");
      setVision("vision-court", "Simulator map", "good");
      renderTracks(data.frame.players || [], true);
    }
    renderSystem(data);
    renderRecent(data);
    changed();
  }
  // The timeline: key-play ticks and the score margin (home above the line, away below).
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
    const svgEl = $("momentum"), series = window.CV ? window.CV.S.margin : [];
    svgEl.replaceChildren();
    if (!(duration > 0) || series.length < 2) return;
    const peak = Math.max(6, ...series.map(([, m]) => Math.abs(m)));
    const X = t => Math.min(1000, Math.max(0, t / duration * 1000)), Y = m => 12 - (m / peak) * 11;
    let d = `M${X(series[0][0]).toFixed(1)} 12`;
    for (let i = 0; i < series.length; i++) {
      const [t, m] = series[i], next = series[i + 1];
      d += ` L${X(t).toFixed(1)} ${Y(m).toFixed(2)}`;
      if (next) d += ` L${X(next[0]).toFixed(1)} ${Y(m).toFixed(2)}`;
    }
    const lastT = Math.max(series[series.length - 1][0], processedT);
    d += ` L${X(lastT).toFixed(1)} ${Y(series[series.length - 1][1]).toFixed(2)} L${X(lastT).toFixed(1)} 12 Z`;
    const ns = "http://www.w3.org/2000/svg";
    const mk = (tag, attrs) => { const el = document.createElementNS(ns, tag); for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v); return el; };
    const defs = mk("defs", {}), clipTop = mk("clipPath", { id: "mom-top" }), clipBot = mk("clipPath", { id: "mom-bot" });
    clipTop.append(mk("rect", { x: 0, y: 0, width: 1000, height: 12 }));
    clipBot.append(mk("rect", { x: 0, y: 12, width: 1000, height: 12 }));
    defs.append(clipTop, clipBot);
    svgEl.append(defs,
      mk("path", { d, class: "mom-home", "clip-path": "url(#mom-top)" }),
      mk("path", { d, class: "mom-away", "clip-path": "url(#mom-bot)" }),
      mk("line", { x1: 0, x2: 1000, y1: 12, y2: 12, class: "mom-zero" }));
  }
  function queueMarks() { if (!marksQueued) { marksQueued = true; requestAnimationFrame(renderMarks); } }
  function tick() {
    const now = new Date(), time = `${pad(now.getHours())}:${pad(now.getMinutes())}:${pad(now.getSeconds())}`;
    text("sys-clock", time);
    text("feed-clock", `${now.getFullYear()}-${pad(now.getMonth() + 1)}-${pad(now.getDate())} ${time}`);
  }
  async function poll(again = true) {
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
        changed();
      }
    } finally { if (again && !stopped && isStudio) setTimeout(poll, 1500); }
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
        window.dispatchEvent(new CustomEvent("cv-frame", { detail: { t: videoTime } }));
        text("studio-time", clock(processedT));
        if (workspace?.active?.duration_s) progress(lastStatus?.finished ? 100 : Math.min(100, processedT / workspace.active.duration_s * 100));
      }
      lastMonitorKey = monitorKey;
    } catch { /* Keep the last frame during reconnect. */ }
    finally { frameBusy = false; if (!stopped) setTimeout(monitor, document.hidden ? 2000 : lastStatus?.finished || lastStatus?.paused ? 500 : 125); }
  }
  // A video already in the library (same size, first and last MiB: nba/results.fingerprint), before copying it again.
  async function findCopy(file) {
    try {
      if (!crypto?.subtle) return null;
      const probe = 1 << 20, head = file.slice(0, probe), tail = file.slice(Math.max(0, file.size - probe), file.size);
      const bytes = await new Blob([head, tail]).arrayBuffer();
      const digest = Array.from(new Uint8Array(await crypto.subtle.digest("SHA-1", bytes)), b => b.toString(16).padStart(2, "0")).join("");
      return (await api("api/studio/find", { bytes: file.size, digest })).video || null;
    } catch { return null; }
  }
  function upload(file) {
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open("POST", `api/studio/upload?name=${encodeURIComponent(file.name)}`);
      for (const [key, value] of Object.entries(headers({"Content-Type":"application/octet-stream"}))) xhr.setRequestHeader(key, value);
      $("studio-upload-wrap").hidden = false;
      xhr.upload.onprogress = event => {
        if (event.lengthComputable) {
          const percent = Math.round(event.loaded / event.total * 100);
          $("studio-upload-progress").value = percent;
          uploadStatus(percent === 100 ? "Checking the video…" : `Copying the video to this Mac's workspace… ${percent}%`);
        }
      };
      xhr.onload = () => {
        let body = {};
        try { body = JSON.parse(xhr.responseText); } catch { /* handled below */ }
        if (xhr.status >= 200 && xhr.status < 300) resolve(body);
        else reject(new Error(typeof body.detail === "string" ? body.detail : `Upload failed (${xhr.status}).`));
      };
      xhr.onerror = () => reject(new Error("Upload interrupted. Check that the local server is running, then retry."));
      xhr.send(file);
    });
  }
  async function queueStart(id) {
    if (busy && !pending) return;
    busy = true; $("studio-start").disabled = true;
    if (workspace?.model.state !== "ready") {
      pending = id;
      uploadStatus("Preparing the detector. The video opens when it is ready.");
      if (!$("studio-dialog").open) window.CV?.toast("Preparing the detector. The video opens when it is ready.");
      try { await api("api/studio/prepare", {}); }
      catch (error) { pending = null; busy = false; $("studio-start").disabled = false; uploadStatus(error.message, true); }
    } else await startVideo(id);
  }
  async function startVideo(id) {
    uploadStatus("Opening the game feed…");
    try {
      const data = await api("api/studio/start", { video_id:id, home_name:$("studio-home").value,
        away_name:$("studio-away").value, broadcast:$("studio-broadcast").checked });
      renderWorkspace(data); if ($("studio-dialog").open) $("studio-dialog").close(); $("tab-live").click();
      $("studio-upload-wrap").hidden = true; uploadStatus("");
    } catch (error) { uploadStatus(error.message, true); window.CV?.toast(error.message, "error"); }
    finally { busy = false; $("studio-start").disabled = !selected; }
  }
  async function setupCourt() {
    if (!workspace?.active && isStudio) { openDialog(); return; }
    try {
      await api("api/control", { action:"pause" });
      $("tab-calibration").click(); $("btn-cal-load").click();
      window.CV?.toast("Paused. Mark four or more court points spread across the floor, apply, then resume.");
    } catch (error) { notify(error.message); }
  }
  async function startDemo() {
    if (busy) return; busy = true; $("studio-demo").disabled = true;
    try { renderWorkspace(await api("api/studio/start", {demo:true})); $("tab-live").click(); }
    catch (error) { notify(error.message); window.CV?.toast(error.message, "error"); }
    finally { busy = false; $("studio-demo").disabled = false; }
  }
  async function exportJson() {
    try {
      const response = await fetch("api/studio/export", {headers:headers()});
      if (!response.ok) throw new Error("Export failed. Check the local server and retry.");
      const url = URL.createObjectURL(await response.blob()), a = document.createElement("a");
      a.href = url; a.download = "courtvision-analysis.json"; a.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
      window.CV?.toast("Analysis exported");
    } catch (error) { notify(error.message); window.CV?.toast(error.message, "error"); }
  }
  async function togglePlay() {
    if (lastStatus?.finished && workspace?.active) {
      if (workspace.active.demo) await startDemo();
      else await queueStart(workspace.active.id);
      return;
    }
    try { const result = await api("api/control", {action:lastStatus?.paused ? "resume" : "pause"}); receiveStatus(result.result); }
    catch (error) { notify(error.message); }
  }
  async function fullscreen() {
    try { if (document.fullscreenElement) await document.exitFullscreen(); else await $("monitor-screen").requestFullscreen(); }
    catch { window.CV?.toast("Fullscreen is unavailable in this browser. Enlarge the window instead.", "error"); }
  }

  window.CVStudio = {
    get workspace() { return workspace; }, get status() { return lastStatus; }, get vision() { return lastVision; },
    get notice() { return notice; }, get isStudio() { return isStudio; }, get processedT() { return processedT; },
    get focus() { return focusedTrack; },
    openDialog, startDemo, setupCourt, exportJson, togglePlay, fullscreen, queueStart, setFocus,
    toggleOverlay() { $("studio-overlay").checked = !$("studio-overlay").checked; lastMonitorKey = ""; },
  };

  $("studio-open").addEventListener("click", openDialog);
  $("studio-manage").addEventListener("click", openDialog);
  $("library-extra-delete").addEventListener("click", async () => {
    try {
      const result = await api("api/studio/analyses", undefined, "DELETE");
      window.CV?.toast(`Deleted ${result.deleted_runs} saved result${result.deleted_runs === 1 ? "" : "s"} · ${size(result.freed_bytes)} freed`);
    } catch (error) { window.CV?.toast(error.message, "error"); }
    await loadLibrary();
  });
  $("studio-clear-focus").addEventListener("click", () => setFocus(focusedTrack));
  $("studio-empty-open").addEventListener("click", openDialog);
  $("studio-close").addEventListener("click", () => $("studio-dialog").close());
  $("studio-file").addEventListener("change", e => chooseFile(e.target.files[0]));
  $("studio-calibrate").addEventListener("click", setupCourt);
  $("studio-overlay").addEventListener("change", () => { lastMonitorKey = ""; });
  $("studio-form").addEventListener("submit", async event => {
    event.preventDefault(); if (!selected || busy) return;
    busy = true; $("studio-start").disabled = true;
    try {
      if (workspace?.model.state !== "ready") await api("api/studio/prepare", {});
      const known = await findCopy(selected);
      if (known) uploadStatus(`“${titleOf(known)}” is already in your videos. Opening it…`);
      const video = known || await upload(selected);
      if (video.duplicate) window.CV?.toast(`“${titleOf(video)}” was already in your videos: opened it, no second copy kept.`);
      busy = false; selected = null;
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
  $("studio-demo").addEventListener("click", startDemo);
  $("studio-play").addEventListener("click", togglePlay);
  $("studio-speed").addEventListener("change", async () => {
    try { await api("api/control", {action:"speed", value:Number($("studio-speed").value)}); }
    catch (error) { notify(error.message); }
  });
  $("studio-fullscreen").addEventListener("click", fullscreen);
  $("studio-export").addEventListener("click", exportJson);
  window.addEventListener("cv-update", ev => { if (ev.detail.keys.includes("compare")) queueMarks(); });
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
      focusedTrack = ""; $("studio-clear-focus").hidden = true; renderTracks([]); lastMonitorKey = "";
      lastVision = null; processedT = 0; progress(0); marks.clear(); queueMarks();
      text("vision-people", "—"); setVision("vision-ball", "Awaiting video", "idle"); setVision("vision-court", "Not configured", "warn");
      if (workspace?.active || !isStudio) { $("studio-frame").hidden = true; $("studio-loading").hidden = false; }
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
