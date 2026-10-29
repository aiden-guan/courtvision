/* Replay: the live analysis done ahead of time (courtvision/server/preanalysis.py), played with the recording at
   full speed. Each analysed frame's boxes are drawn over the video, and the recorded hub messages are applied as if
   they had just arrived, so the plays, the box score, the court lens and the narration follow the video; a seek
   rebuilds the view at that moment. Also the status and controls of the video's background analysis. */
(() => {
  "use strict";
  const $ = id => document.getElementById(id);
  const CHUNK_S = 60, HOLD_S = 0.5, JUMP_S = 2.5;  // recording files' length; how long boxes stay up; a seek
  const COLORS = { home: "rgb(120,187,237)", away: "rgb(246,191,117)", ball: "rgb(248,224,105)", other: "rgb(209,235,151)" };
  let status = null, vid = "", on = false, lastT = NaN, pending = 0, held = false;
  let chunks = new Map(), loading = new Set(), font = "";
  const token = () => { try { return sessionStorage.getItem("cv-token") || ""; } catch { return ""; } };
  const headers = extra => ({ ...extra, ...(token() ? { Authorization: `Bearer ${token()}` } : {}) });
  const text = (id, value) => { if ($(id).textContent !== String(value)) $(id).textContent = value; };
  const clock = s => {
    if (!Number.isFinite(s)) return "--:--";
    const v = Math.max(0, Math.round(s)), h = Math.floor(v / 3600), m = Math.floor(v / 60) % 60, sec = v % 60;
    return h ? `${h}:${String(m).padStart(2, "0")}:${String(sec).padStart(2, "0")}` : `${m}:${String(sec).padStart(2, "0")}`;
  };
  const span = s => (s >= 5400 ? `${Math.round(s / 3600)} h` : s >= 3600 ? `1 h ${Math.round((s - 3600) / 60)} min` :
    s >= 60 ? `${Math.round(s / 60)} min` : "under a minute");
  const done = () => (status ? status.state === "done" : false);
  const frontier = () => (status ? status.t_done || 0 : 0);
  // the index of the last of the sorted times at or before t (-1 when none is)
  function upTo(times, t) {
    let lo = 0, hi = times.length - 1, k = -1;
    while (lo <= hi) { const mid = (lo + hi) >> 1; if (times[mid] <= t) { k = mid; lo = mid + 1; } else hi = mid - 1; }
    return k;
  }

  // ------------------------------------------------------------------ the recording, a minute at a time

  async function load(k, again = false) {
    if (k < 0 || loading.has(k) || (chunks.has(k) && !again)) return;
    loading.add(k);
    try {
      const response = await fetch(`api/studio/replay/${vid}/chunk/${k}`, { headers: headers(), cache: "no-store" });
      if (!response.ok) return;
      const boxes = [], msgs = [];
      for (const line of (await response.text()).split("\n")) {
        if (!line) continue;
        let rec;
        try { rec = JSON.parse(line); } catch { break; }  // the line a running analysis is writing
        (rec.b ? boxes : msgs).push(rec);
      }
      const last = Math.max(boxes.length ? boxes[boxes.length - 1].vt : -1, msgs.length ? msgs[msgs.length - 1].vt : -1);
      chunks.set(k, { boxes, boxT: boxes.map(r => r.vt), msgs, msgT: msgs.map(r => r.vt), last });
    } catch { /* retried on the next tick */ } finally { loading.delete(k); }
  }

  function fresh(k, t) {  // a minute still being written: fetch again once the analysis has moved on
    const c = chunks.get(k);
    if (c && !done() && t > c.last - 2 && frontier() > c.last + 2) load(k, true);
  }

  // ------------------------------------------------------------------ the view at a moment, and playing on

  // while one is in flight (pending) the messages wait; a failed one is tried again by the next tick after 1.5 s
  async function resync(t) {
    const my = ++pending;
    try {
      const response = await fetch(`api/studio/replay/${vid}/state?t=${t.toFixed(3)}`, { headers: headers(), cache: "no-store" });
      if (!response.ok) throw new Error(String(response.status));
      const st = await response.json();
      if (my !== pending || !on) return;
      const send = window.CV.dispatch;
      send({ type: "reset", t: 0 });
      if (st.meta) send(st.meta);
      for (const m of [...st.events, ...st.insights, ...st.utterances, ...st.captions]) send(m);
      for (const m of [st.stats, st.frame, st.vision]) if (m) send(m);
      lastT = t; pending = 0;
      const k = Math.floor(t / CHUNK_S);
      load(k); load(k + 1);
    } catch {
      if (my === pending) setTimeout(() => { if (my === pending) { pending = 0; lastT = NaN; } }, 1500);
    }
  }

  // apply the messages recorded in (lastT, t]; stop at a minute not loaded yet and carry on from there next time
  function advance(t) {
    for (let k = Math.floor(lastT / CHUNK_S); k <= Math.floor(t / CHUNK_S); k++) {
      const c = chunks.get(k);
      if (!c) { load(k); lastT = Math.max(lastT, k * CHUNK_S - 1e-6); return; }
      for (let i = upTo(c.msgT, lastT) + 1; i < c.msgs.length && c.msgs[i].vt <= t; i++) window.CV.dispatch(c.msgs[i].m);
    }
    lastT = t;
  }

  function boxesAt(t) {
    for (const k of [Math.floor(t / CHUNK_S), Math.floor(t / CHUNK_S) - 1]) {
      const c = chunks.get(k), i = c ? upTo(c.boxT, t) : -1;
      if (i >= 0) return t - c.boxT[i] <= HOLD_S ? c.boxes[i].b : null;
    }
    return null;
  }

  // the boxes the live monitor draws (courtvision/server/studio_game.py), over the playing video
  function draw(t) {
    const canvas = $("replay-canvas"), v = $("watch-video"), screen = $("monitor-screen");
    const w = screen.clientWidth, h = screen.clientHeight, dpr = window.devicePixelRatio || 1;
    if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
      canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr);
    }
    const g = canvas.getContext("2d");
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, w, h);
    const boxes = Number.isFinite(t) && $("studio-overlay").checked ? boxesAt(t) : null;
    if (!boxes || !v.videoWidth) return;
    const s = Math.min(w / v.videoWidth, h / v.videoHeight), ox = (w - v.videoWidth * s) / 2, oy = (h - v.videoHeight * s) / 2;
    const focus = window.CVStudio?.focus || "";
    font = font || getComputedStyle(document.body).getPropertyValue("--font").trim() || "sans-serif";
    g.lineCap = "round"; g.font = `500 ${w < 800 ? 10.5 : 12}px ${font}`; g.textBaseline = "middle";
    for (const [bx1, by1, bx2, by2, label, team, conf] of boxes) {
      const x1 = ox + bx1 * s, y1 = oy + by1 * s, x2 = ox + bx2 * s, y2 = oy + by2 * s;
      if (x2 <= x1 || y2 <= y1) continue;
      const muted = !!(focus && focus !== label && team !== "ball");
      const color = muted ? "rgb(110,112,106)" : COLORS[team] || COLORS.other;
      g.strokeStyle = color; g.fillStyle = color;
      if (team === "ball") {
        const cx = (x1 + x2) / 2, cy = (y1 + y2) / 2;
        g.lineWidth = 2; g.beginPath(); g.arc(cx, cy, Math.max(9, (x2 - x1) / 2 + 4), 0, Math.PI * 2); g.stroke();
        continue;
      }
      // corner brackets, as on the live monitor
      const len = Math.max(3, Math.min(20, (x2 - x1) / 3, (y2 - y1) / 4));
      g.lineWidth = focus === label ? 3 : 2;
      g.beginPath();
      for (const [x, dx] of [[x1, 1], [x2, -1]]) for (const [y, dy] of [[y1, 1], [y2, -1]]) {
        g.moveTo(x + dx * len, y); g.lineTo(x, y); g.lineTo(x, y + dy * len);
      }
      g.stroke();
      if (muted) continue;
      const tag = conf == null ? String(label) : `${label} / ${Math.round(conf * 100)}%`, tw = g.measureText(tag).width;
      const lx = Math.max(0, Math.min(x1, w - tw - 12)), ly = Math.max(18, y1 - 5);
      g.fillStyle = "rgba(24,30,27,0.9)"; g.fillRect(lx, ly - 16, tw + 12, 16);
      g.fillStyle = color; g.fillRect(lx, ly - 16, 2, 16); g.fillText(tag, lx + 6, ly - 8);
    }
  }

  function tick(t) {
    const k = Math.floor(t / CHUNK_S);
    load(k); fresh(k, t);
    if (t > k * CHUNK_S + CHUNK_S - 20) load(k + 1);
    if (!done() && t > frontier() - 0.5 && !$("watch-video").paused) {  // past what has been analysed so far
      $("watch-video").pause();
      if (!held) { held = true; window.CV?.toast?.(`Analysed up to ${clock(frontier())} so far. It carries on in the background.`); }
    } else if (t < frontier() - 5) held = false;
    if (!pending) {
      if (!Number.isFinite(lastT) || t < lastT - 0.05 || t > lastT + JUMP_S) resync(t);
      else advance(t);
    }
    draw(t);
  }

  function enter() {
    on = true; lastT = NaN; held = false;
    window.CV.setReplay(true);
    $("replay-canvas").hidden = false;
    resync($("watch-video").currentTime || 0);
  }

  function leave() {
    on = false; pending = 0;
    $("replay-canvas").hidden = true;
    draw(NaN);
    window.CV.setReplay(false);  // back to the live game's state
  }

  // ------------------------------------------------------------------ the background analysis of the open video

  function label() {
    if (!status) return "Replay";
    if (status.state === "running" && status.duration_s) return `Replay · ${Math.floor(frontier() / status.duration_s * 100)}%`;
    return "Replay";
  }

  function row() {
    const ws = window.CVStudio?.workspace, active = ws?.active && !ws.active.demo ? ws.active : null;
    $("pre-row").hidden = !active;
    if (!active) return;
    const st = status, state = st ? st.state : "none", dur = st?.duration_s || active.duration_s;
    const pctDone = dur ? Math.min(100, frontier() / dur * 100) : 0;
    const every = st && st.stride > 1 ? `every ${st.stride === 2 ? "2nd" : "3rd"} frame` : "every frame";
    let title, detail, action = "", tone = "idle";
    if (state === "running" || state === "loading") {
      tone = "busy";
      title = state === "loading" ? "Preparing to analyse ahead" : `Analysing ahead · ${Math.floor(pctDone)}%`;
      detail = state === "loading" ? "Loading the detector for the background analysis." :
        `Replay is ready up to ${clock(frontier())}${st.eta_s ? ` · about ${span(st.eta_s)} left` : ""} · ${every}`;
      action = "Stop";
    } else if (state === "queued") {
      tone = "busy"; title = "Waiting to analyse ahead"; detail = "Another video is being analysed first."; action = "Cancel";
    } else if (state === "done") {
      tone = "good"; title = "Analysed ahead"; detail = `Replay plays the whole video at full speed with the analysis · ${every}`;
      action = window.CVPlayback?.mode === "replay" ? "" : "Replay";
    } else if (state === "stopped" || state === "failed") {
      tone = state === "failed" ? "bad" : "idle";
      title = `${state === "failed" ? "Analysis ahead failed" : "Analysis ahead stopped"} at ${clock(frontier())}`;
      detail = `${st.error ? `${st.error} ` : ""}Replay works up to there.`; action = "Analyse again";
    } else {
      title = "Not analysed ahead";
      detail = "Analyse the whole video in the background, then replay it at full speed with the analysis. This takes a while: it runs at the analysis's own pace.";
      action = "Analyse ahead";
    }
    text("pre-title", title); text("pre-detail", detail);
    $("pre-title").dataset.tone = tone;
    $("pre-bar").hidden = !(state === "running" || ((state === "stopped" || state === "failed") && frontier() > 0));
    $("pre-fill").style.width = `${pctDone}%`;
    $("pre-stride").hidden = !(action === "Analyse ahead" || action === "Analyse again");
    $("pre-action").hidden = !action;
    text("pre-action", action);
  }

  async function act() {
    const action = $("pre-action").textContent, ws = window.CVStudio?.workspace, id = ws?.active?.id;
    if (!id) return;
    if (action === "Replay") { window.CVPlayback?.setMode("replay"); return; }
    const stop = action === "Stop" || action === "Cancel";
    $("pre-action").disabled = true;
    try {
      const response = await fetch(`api/studio/replay/${id}/${stop ? "stop" : "start"}`, {
        method: "POST", headers: headers({ "Content-Type": "application/json" }), cache: "no-store",
        body: stop ? undefined : JSON.stringify({ stride: Number($("pre-stride").value) || 1 }),
      });
      const body = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(typeof body.detail === "string" ? body.detail : `Request failed (${response.status}).`);
      status = body.state ? body : status;
      row(); window.CVPlayback?.render();
      window.CV?.toast?.(stop ? "Stopped. Replay works up to where it got." : "Analysing ahead in the background.");
    } catch (error) { window.CV?.toast?.(error.message, "error"); } finally { $("pre-action").disabled = false; }
  }

  function sync() {
    const ws = window.CVStudio?.workspace, id = ws?.active && !ws.active.demo ? ws.active.id : "";
    if (id !== vid) { vid = id; chunks = new Map(); loading = new Set(); lastT = NaN; }
    status = id ? ws.replay || null : null;
    row();
  }

  window.CVReplay = {
    get available() { return !!vid && frontier() > 2; },
    get label() { return label(); },
    get status() { return status; },
  };
  $("pre-action").addEventListener("click", act);
  window.addEventListener("cv-studio", sync);
  window.addEventListener("cv-playback-mode", ev => {
    if (ev.detail.mode === "replay") enter();
    else if (ev.detail.was === "replay") leave();
    row();
  });
  window.addEventListener("cv-playback", ev => { if (on && ev.detail.mode === "replay") tick(ev.detail.t); });
  window.addEventListener("resize", () => { if (on) draw($("watch-video").currentTime); });
})();
