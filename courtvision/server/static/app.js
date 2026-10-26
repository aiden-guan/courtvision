/* CourtVision live dashboard: vanilla JS, no dependencies, works offline.
 *
 * Data arrives on the /ws WebSocket (ARCHITECTURE "server/" schema: meta, frame, stats, event,
 * utterance, insight, caption, status). Operator writes go to the REST API with an optional
 * bearer token. All server text is inserted with textContent (never innerHTML).
 */
(() => {
  "use strict";

  // =========================================================================== utilities

  const byId = (id) => document.getElementById(id);
  const isNum = (v) => typeof v === "number" && Number.isFinite(v);
  const num = (v, d) => (isNum(v) ? v : d);
  const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
  const SVG_NS = "http://www.w3.org/2000/svg";

  function h(tag, props, ...kids) {
    const el = document.createElement(tag);
    for (const [k, v] of Object.entries(props || {})) {
      if (v == null || v === false) continue;
      if (k === "class") el.className = v;
      else if (k === "text") el.textContent = v;
      else if (k === "dataset") Object.assign(el.dataset, v);
      else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
      else el.setAttribute(k, v === true ? "" : String(v));
    }
    for (const kid of kids.flat()) {
      if (kid == null || kid === false) continue;
      el.append(kid instanceof Node ? kid : document.createTextNode(String(kid)));
    }
    return el;
  }

  function svg(tag, attrs) {
    const el = document.createElementNS(SVG_NS, tag);
    for (const [k, v] of Object.entries(attrs || {})) if (v != null) el.setAttribute(k, String(v));
    return el;
  }

  /** replaceChildren that skips null/false children (the DOM method would print "null"). */
  function fill(el, ...kids) {
    el.replaceChildren(...kids.flat().filter((k) => k != null && k !== false));
  }

  function setText(el, text) {
    const t = String(text);
    if (el && el.textContent !== t) el.textContent = t;
  }

  const store = {
    get(area, key) {
      try { return window[area].getItem(key); } catch { return null; }
    },
    set(area, key, val) {
      try {
        if (val == null || val === "") window[area].removeItem(key);
        else window[area].setItem(key, val);
      } catch { /* storage blocked: settings just do not persist */ }
    },
  };

  // ----------------------------------------------------------------------- formatting

  function fmtClock(sec) {
    if (!isNum(sec)) return "--:--";
    const v = Math.max(0, sec);
    if (v < 60) return v.toFixed(1);
    const m = Math.floor(v / 60);
    return `${m}:${String(Math.floor(v - m * 60)).padStart(2, "0")}`;
  }
  const fmtShot = (sec) => (!isNum(sec) ? "--" : sec < 5 ? Math.max(0, sec).toFixed(1) : String(Math.ceil(sec - 1e-9)));
  function fmtVideo(t) {
    if (!isNum(t)) return "–";
    const m = Math.floor(t / 60);
    return `${m}:${(t - m * 60).toFixed(1).padStart(4, "0")}`;
  }
  function fmtPct(v, d = 1) {
    if (!isNum(v)) return "–";
    return `${(Math.abs(v) <= 1.5 ? v * 100 : v).toFixed(d)}%`;
  }
  const fmtNum = (v, d = 1) => (isNum(v) ? v.toFixed(d) : "–");
  const fmtInt = (v) => (isNum(v) ? String(Math.round(v)) : "–");
  const fmtMade = (m, a) => (isNum(a) ? `${fmtInt(num(m, 0))}-${fmtInt(a)}` : "–");
  function fmtSigned(v, d = 0) {
    if (!isNum(v)) return "–";
    const r = d ? v.toFixed(d) : String(Math.round(v));
    return v > 0 ? `+${r}` : r;
  }
  function fmtMin(sec) {
    if (!isNum(sec) || sec <= 0) return "–";
    return `${Math.floor(sec / 60)}:${String(Math.floor(sec % 60)).padStart(2, "0")}`;
  }
  function humanize(k) {
    const s = String(k ?? "").replace(/_/g, " ").trim();
    return s ? s[0].toUpperCase() + s.slice(1) : "";
  }

  // =========================================================================== state

  const KINDS = ["possession", "pass", "dribble", "shot_attempt", "shot_made", "shot_missed", "block",
    "rebound", "assist", "steal", "turnover", "foul", "substitution", "timeout", "period_start",
    "period_end", "score_correction"];
  const NEEDS_PLAYER = new Set(["pass", "assist", "steal", "block", "substitution", "shot_attempt",
    "shot_made", "shot_missed", "foul"]);
  // Mirrors server/app.py: these kinds may have no team; the pairs fix other_player_id's side.
  const TEAMLESS_OK = new Set(["period_start", "period_end", "timeout"]);
  const SAME_TEAM_PAIR = new Set(["assist", "substitution"]);
  const OPPONENT_PAIR = new Set(["steal", "block", "foul"]);
  const FG_SHOT_TYPES = new Set(["jumper", "layup", "dunk", "hook", "tip"]);
  const SCORING = new Set(["shot_made", "score_correction"]);

  const S = {
    meta: null,
    teams: { home: "Home", away: "Away" },
    roster: new Map(), // player_id -> roster entry
    rosterVersion: 0,
    court: null,
    geo: null,
    rules: null,
    landmarks: {},
    frame: null,
    frameAt: 0,
    stats: null,
    events: new Map(),
    utter: [],
    utterIds: new Set(),
    insights: new Map(),
    clock: [], // [t, period, game_clock] samples for event time labels
    held: {}, // clock field -> [video t, value] of its last scoreboard reading
    paused: false,
    finished: false,
    muted: false,
    speed: 1,
    shotFilter: "all",
  };

  // =========================================================================== theme colours

  let C = {};
  let FONT = "system-ui, sans-serif";

  function readColors() {
    const cs = getComputedStyle(document.documentElement);
    const v = (n) => cs.getPropertyValue(n).trim();
    C = {
      ink: v("--ink"), surface: v("--surface"), floor: v("--court-floor"), paint: v("--court-paint"),
      line: v("--court-line"), rim: v("--court-rim"), home: v("--home"), homeInk: v("--home-ink"),
      away: v("--away"), awayInk: v("--away-ink"), neutral: v("--neutral-team"), ball: v("--ball"),
      muted: v("--muted"), seq: [0, 1, 2, 3, 4].map((i) => v(`--seq-${i}`)),
    };
    FONT = getComputedStyle(document.body).fontFamily || FONT;
  }

  function hexRgb(hex) {
    const m = /^#?([0-9a-f]{6})$/i.exec(String(hex).trim());
    if (!m) return null;
    const n = parseInt(m[1], 16);
    return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
  }

  function inkFor(hex) {
    const rgb = hexRgb(hex);
    if (!rgb) return "#ffffff";
    const lin = rgb.map((c) => { const s = c / 255; return s <= 0.03928 ? s / 12.92 : ((s + 0.055) / 1.055) ** 2.4; });
    const lum = 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2];
    return lum > 0.3 ? "#0b0b0b" : "#ffffff";
  }

  function applyTeamColors(colors) {
    const root = document.documentElement.style;
    for (const team of ["home", "away"]) {
      const hex = colors && colors[team];
      if (hex && hexRgb(hex)) {
        root.setProperty(`--${team}`, hex);
        root.setProperty(`--${team}-ink`, inkFor(hex));
      } else {
        root.removeProperty(`--${team}`);
        root.removeProperty(`--${team}-ink`);
      }
    }
  }

  function applyTheme(theme) {
    const root = document.documentElement;
    if (theme === "light" || theme === "dark") root.dataset.theme = theme;
    else delete root.dataset.theme;
    themeChanged();
  }

  function themeChanged() {
    readColors();
    courtView.base = null;
    mark("court", "shots");
  }

  /** xFG -> sequential ramp colour (20% .. 70%+). */
  function xfgColor(x) {
    if (!isNum(x)) return C.muted;
    const t = clamp((x - 0.2) / 0.5, 0, 1) * (C.seq.length - 1);
    const i = Math.min(C.seq.length - 2, Math.floor(t));
    const a = hexRgb(C.seq[i]), b = hexRgb(C.seq[i + 1]);
    if (!a || !b) return C.seq[2] || C.muted;
    const f = t - i;
    const c = a.map((av, k) => Math.round(av + (b[k] - av) * f));
    return `rgb(${c[0]}, ${c[1]}, ${c[2]})`;
  }

  // =========================================================================== render scheduling

  const dirty = new Set();
  let rafPending = false;
  const renderers = {};

  function mark(...keys) {
    for (const k of keys) dirty.add(k);
    if (!rafPending) {
      rafPending = true;
      requestAnimationFrame(flush);
    }
  }

  function flush() {
    rafPending = false;
    const keys = [...dirty];
    dirty.clear();
    for (const k of keys) {
      try { renderers[k](); } catch (err) { console.error(`render ${k} failed`, err); }
    }
  }

  // =========================================================================== names & time

  function periodLabel(p) {
    if (!isNum(p)) return "–";
    const r = S.rules || {};
    const n = num(r.n_periods, 4);
    if (p > n) return p - n === 1 ? "OT" : `${p - n}OT`;
    return `${r.halves ? "H" : "Q"}${p}`;
  }

  function jerseyFromFrame(id) {
    const p = S.frame && (S.frame.players || []).find((q) => q.id === id);
    return p && p.jersey;
  }

  function pname(id) {
    if (!id) return "—";
    const r = S.roster.get(id);
    if (r) return r.name || `#${r.jersey}`;
    const m = /^(home|away|unk)-t(\d+)$/.exec(id);
    if (m) {
      const j = jerseyFromFrame(id);
      const team = m[1] === "unk" ? "Unidentified" : S.teams[m[1]];
      return `${team} ${j ? `#${j}` : "player"} (t${m[2]})`;
    }
    return id;
  }

  function pjersey(id) {
    const r = S.roster.get(id);
    return r ? r.jersey : jerseyFromFrame(id) || "";
  }

  /** Team of a player id: roster first, then the "home-" / "away-" id prefix (as the server). */
  function teamOfPlayer(id) {
    if (!id) return null;
    const r = S.roster.get(id);
    if (r) return r.team === "home" || r.team === "away" ? r.team : null;
    const m = /^(home|away)-/i.exec(id);
    return m ? m[1].toLowerCase() : null;
  }

  function recordClock(t, clk) {
    if (!clk || !isNum(t) || !isNum(clk.game)) return;
    const last = S.clock[S.clock.length - 1];
    if (last && t < last[0]) S.clock.length = 0; // video restarted
    if (!last || t - last[0] >= 0.5 || t < last[0]) S.clock.push([t, clk.period, clk.game]);
    if (S.clock.length > 40000) S.clock.splice(0, 10000);
  }

  /** Game-clock label for a video time, from frame clock samples; falls back to video time. */
  function timeLabel(t) {
    const arr = S.clock;
    if (isNum(t) && arr.length) {
      let lo = 0, hi = arr.length - 1, best = -1;
      while (lo <= hi) {
        const mid = (lo + hi) >> 1;
        if (arr[mid][0] <= t) { best = mid; lo = mid + 1; } else hi = mid - 1;
      }
      if (best >= 0 && t - arr[best][0] <= 3) return `${periodLabel(arr[best][1])} ${fmtClock(arr[best][2])}`;
    }
    return `t ${fmtVideo(t)}`;
  }

  // =========================================================================== API

  class ApiError extends Error {
    constructor(status, message) { super(message); this.status = status; }
  }

  const getToken = () => store.get("sessionStorage", "cv-token") || "";

  function authHeaders(extra) {
    const t = getToken();
    return Object.assign({}, extra || {}, t ? { Authorization: `Bearer ${t}` } : {});
  }

  function detailText(d) {
    if (!d) return "";
    if (typeof d === "string") return d;
    if (Array.isArray(d)) {
      return d.map((e) => {
        if (!e || !e.msg) return String(e);
        const loc = (e.loc || []).filter((x) => x !== "body").join(".");
        return `${loc ? `${loc}: ` : ""}${String(e.msg).replace(/^Value error, /, "")}`;
      }).join("; ");
    }
    return JSON.stringify(d);
  }

  async function errorText(res) {
    const prefix = res.status === 503 ? "Pipeline not attached" : res.status === 401 ? "Access token required" : "";
    try {
      const j = await res.json();
      return prefix || detailText(j.detail) || `HTTP ${res.status}`;
    } catch {
      return prefix || `HTTP ${res.status}`;
    }
  }

  async function api(path, body) {
    const post = body !== undefined;
    const opts = {
      method: post ? "POST" : "GET",
      headers: authHeaders(post ? { "Content-Type": "application/json" } : {}),
      cache: "no-store",
    };
    if (post) opts.body = JSON.stringify(body);
    let res;
    try {
      res = await fetch(path, opts);
    } catch {
      throw new ApiError(0, "Server unreachable");
    }
    if (!res.ok) {
      if (res.status === 401) setConn("unauthorized");
      throw new ApiError(res.status, await errorText(res));
    }
    return res.status === 204 ? null : res.json();
  }

  async function fetchFrameImage() {
    let res;
    try {
      res = await fetch("api/frame.jpg", { headers: authHeaders(), cache: "no-store" });
    } catch {
      throw new ApiError(0, "Server unreachable");
    }
    if (!res.ok) throw new ApiError(res.status, res.status === 404 ? "No frame available yet" : await errorText(res));
    const blob = await res.blob();
    if (window.createImageBitmap) return createImageBitmap(blob);
    const url = URL.createObjectURL(blob);
    try {
      const img = new Image();
      img.src = url;
      await img.decode();
      return img;
    } finally {
      URL.revokeObjectURL(url);
    }
  }

  function toast(message, kind) {
    const el = h("div", { class: `toast${kind === "error" ? " error" : ""}`, text: message });
    byId("toast-region").append(el);
    setTimeout(() => el.remove(), kind === "error" ? 6000 : 3500);
  }

  async function send(path, body, okMessage) {
    try {
      const out = await api(path, body);
      if (okMessage) toast(okMessage);
      return out;
    } catch (err) {
      toast(err.message || String(err), "error");
      throw err;
    }
  }

  const latestT = () => (S.frame && isNum(S.frame.t) ? S.frame.t : S.stats && isNum(S.stats.t) ? S.stats.t : undefined);

  // =========================================================================== WebSocket

  const WS = { sock: null, retry: 0, timer: 0, ping: 0, pingId: 0, pingSent: 0, lastMsg: 0, opened: false, rtt: null };

  function setConn(state, label) {
    const el = byId("conn-status");
    el.dataset.state = state;
    const text = label || { live: "Live", connecting: "Connecting", offline: "Offline", unauthorized: "Token required" }[state] || state;
    setText(byId("conn-label"), text);
  }

  function wsUrl() {
    const u = new URL("ws", location.href);
    u.protocol = location.protocol === "https:" ? "wss:" : "ws:";
    u.hash = "";
    const t = getToken();
    if (t) u.searchParams.set("token", t);
    return u.toString();
  }

  function connect() {
    clearTimeout(WS.timer);
    if (WS.sock) {
      WS.sock.onclose = null;
      try { WS.sock.close(); } catch { /* already closed */ }
    }
    setConn("connecting");
    WS.opened = false;
    let sock;
    try {
      sock = new WebSocket(wsUrl());
    } catch {
      scheduleReconnect();
      return;
    }
    WS.sock = sock;
    sock.onopen = () => {
      WS.opened = true;
      WS.retry = 0;
      WS.lastMsg = performance.now();
      resetLiveState();
      setConn("live");
    };
    sock.onmessage = (ev) => {
      WS.lastMsg = performance.now();
      let msg;
      try { msg = JSON.parse(ev.data); } catch { return; }
      dispatch(msg);
    };
    sock.onclose = (ev) => {
      WS.sock = null;
      if (!WS.opened || ev.code === 1008) probeAuth();
      else scheduleReconnect();
    };
  }

  async function probeAuth() {
    try {
      const res = await fetch("api/state", { headers: authHeaders(), cache: "no-store" });
      if (res.status === 401) {
        setConn("unauthorized");
        if (!probeAuth.prompted) {
          probeAuth.prompted = true;
          openSettings();
        }
        WS.timer = setTimeout(connect, 15000);
        return;
      }
    } catch { /* server down: fall through to reconnect */ }
    scheduleReconnect();
  }

  function scheduleReconnect() {
    const delays = [500, 1000, 2000, 4000, 8000, 10000];
    const d = delays[Math.min(WS.retry, delays.length - 1)] * (0.8 + Math.random() * 0.4);
    WS.retry += 1;
    setConn("offline", `Reconnecting in ${Math.ceil(d / 1000)}s`);
    WS.timer = setTimeout(connect, d);
  }

  function heartbeat() {
    const sock = WS.sock;
    if (!sock || sock.readyState !== WebSocket.OPEN) return;
    if (performance.now() - WS.lastMsg > 35000) {
      sock.close(4000, "stale");
      return;
    }
    WS.pingId += 1;
    WS.pingSent = performance.now();
    try { sock.send(JSON.stringify({ type: "ping", id: WS.pingId })); } catch { /* closing */ }
  }

  function onPong(msg) {
    if (msg.id === WS.pingId && WS.pingSent) {
      WS.rtt = performance.now() - WS.pingSent;
      setConn("live", `Live · ${Math.round(WS.rtt)} ms`);
    }
  }

  function resetLiveState() {
    S.events.clear();
    S.utter = [];
    S.utterIds.clear();
    S.insights.clear();
    feedNodes.forEach((n) => n.el.remove());
    feedNodes.clear();
    byId("transcript-list").replaceChildren();
    transcriptRendered = 0;
    mark("feed", "insights", "transcript");
  }

  // =========================================================================== dispatch

  function dispatch(msg) {
    window.dispatchEvent(new CustomEvent("courtvision-message", { detail: msg }));
    switch (msg && msg.type) {
      case "reset":
        resetLiveState();
        S.frame = null; S.stats = null; S.clock = []; S.held = {};
        S.meta = null; S.roster.clear(); S.paused = false; S.finished = false;
        CAL.points = [];
        if (CAL.tool) { CAL.tool.img = null; CAL.tool.empty.hidden = false; CAL.tool.draw(); }
        ROI.corners = []; ROI.cells.clear(); ROI.history = [];
        if (ROI.tool) { ROI.tool.img = null; ROI.tool.empty.hidden = false; ROI.tool.draw(); }
        byId("cal-result").replaceChildren(); byId("roi-result").replaceChildren();
        roiChanged();
        mark("court", "shots", "box", "compare", "scorebar", "status", "calibration");
        break;
      case "meta": onMeta(msg); break;
      case "frame": onFrame(msg); break;
      case "stats": onStats(msg); break;
      case "event": onEvent(msg.event); break;
      case "utterance": onUtterance(msg.utterance); break;
      case "insight": onInsight(msg.insight); break;
      case "caption": onCaption(msg); break;
      case "status": onStatus(msg); break;
      case "pong": onPong(msg); break;
      default: break;
    }
  }

  function onMeta(msg) {
    S.meta = msg;
    S.teams = { home: (msg.teams && msg.teams.home) || "Home", away: (msg.teams && msg.teams.away) || "Away" };
    S.roster = new Map((msg.roster || []).map((e) => [e.player_id, e]));
    S.rosterVersion += 1;
    S.court = msg.court || {};
    S.rules = msg.rules || null;
    S.landmarks = msg.landmarks || {};
    S.geo = courtGeometry(S.court);
    applyTeamColors(msg.colors);
    readColors();
    loadLineups();
    const ratio = `${S.geo.L + 4} / ${S.geo.W + 4}`;
    byId("court-wrap").style.aspectRatio = ratio;
    byId("shot-wrap").style.aspectRatio = ratio;
    courtView.base = null;
    shotView.base = false;
    mark("teams", "court", "shots", "box", "feed", "operator", "calibration", "scorebar");
  }

  function onFrame(msg) {
    S.frame = msg;
    S.frameAt = performance.now();
    recordClock(msg.t, msg.clock);
    holdClock(msg.t, msg.clock);
    mark("court", "scorebar");
  }

  // Scoreboard readings reach only some frames (OCR runs at 5 Hz, RESEARCH D14), so each clock
  // field keeps its last reading for a short video-time window instead of flickering to "--".
  const CLOCK_HOLD_S = 2.0;

  function holdClock(t, clk) {
    if (!clk || !isNum(t)) return;
    for (const k of ["period", "game", "shot"]) if (isNum(clk[k])) S.held[k] = [t, clk[k]];
  }

  // A clock shown from the last stats snapshot because no reading arrived for CLOCK_HOLD_S of video (the
  // scoreboard reader stopped, or the board left the frame) is dimmed and says how old it is.
  function markStale(el, k, value, label) {
    const held = S.held[k], t = S.frame && S.frame.t;
    const stale = isNum(value) && !isNum(heldClock(k));
    el.classList.toggle("stale", stale);
    if (stale) {
      const age = held && isNum(t) && t >= held[0] ? Math.round(t - held[0]) : null;
      const why = age !== null ? `not read for ${age} s` : "not read";
      el.title = `${label} ${why}: last known value`;
      el.setAttribute("aria-label", `${label} ${el.textContent}, ${why}`);
    } else {
      el.title = label;
      el.removeAttribute("aria-label");
    }
  }

  function heldClock(k) {
    const held = S.held[k], t = S.frame && S.frame.t;
    if (!held) return undefined;
    if (isNum(t) && t < held[0]) return undefined; // video restarted
    if (k !== "period" && isNum(t) && t - held[0] > CLOCK_HOLD_S) return undefined;
    return held[1];
  }

  function onStats(msg) {
    S.stats = msg.stats || null;
    mark("scorebar", "compare", "box", "shots", "decisions");
  }

  function onEvent(ev) {
    if (!ev || !ev.event_id) return;
    const old = S.events.get(ev.event_id);
    if (old && num(old.revision, 0) > num(ev.revision, 0)) return;
    S.events.set(ev.event_id, ev);
    if (S.events.size > 6000) {
      const oldest = [...S.events.values()].sort((a, b) => a.t - b.t).slice(0, 1000);
      for (const e of oldest) S.events.delete(e.event_id);
    }
    mark("feed");
  }

  function onUtterance(u) {
    if (!u || S.utterIds.has(u.utterance_id)) return;
    if (u.utterance_id) S.utterIds.add(u.utterance_id);
    S.utter.push(u);
    mark("transcript");
  }

  function onInsight(ins) {
    if (!ins) return;
    S.insights.set(ins.insight_id || `i${S.insights.size}`, ins);
    mark("insights");
  }

  function onCaption(msg) {
    setText(byId(msg.channel === "ad" ? "ad-line" : "caption-line"), msg.text || " ");
  }

  function onStatus(msg) {
    if (typeof msg.paused === "boolean") S.paused = msg.paused;
    if (typeof msg.finished === "boolean") S.finished = msg.finished;
    if (typeof msg.muted === "boolean") S.muted = msg.muted;
    if (isNum(msg.speed)) S.speed = msg.speed;
    mark("status");
  }

  // =========================================================================== court geometry

  function arc(cx, cy, r, a0, a1, n) {
    const pts = [];
    for (let i = 0; i <= n; i++) {
      const a = a0 + ((a1 - a0) * i) / n;
      pts.push([cx + r * Math.cos(a), cy + r * Math.sin(a)]);
    }
    return pts;
  }

  /** Court drawing primitives (court feet, y up) from the meta message's CourtSpec fields. */
  function courtGeometry(c) {
    const L = num(c.length, 94), W = num(c.width, 50), cy = W / 2;
    const fiba = String(c.league || "").toLowerCase() === "fiba";
    const rimR = num(c.rim_radius, 0.75), bOff = num(c.basket_offset, 5.25), bbOff = num(c.backboard_offset, 4);
    const r3 = num(c.three_pt_radius, 23.75), c3 = Math.min(num(c.corner_three_dist, 22), r3);
    const lane = num(c.lane_width, 16), ftd = num(c.free_throw_dist, 15);
    const ra = num(c.restricted_radius, 4), cc = num(c.center_circle_radius, 6);
    const ftR = fiba ? 1.8 / 0.3048 : 6; // FT circle radius is not a CourtSpec field (NBA/NCAA/NFHS 6 ft, FIBA 1.8 m)
    const bbHalf = fiba ? 0.9 / 0.3048 : 3; // half backboard width (6 ft; FIBA 1.8 m)
    const paint = [], lines = [], dashes = [], rims = [];
    lines.push([[0, 0], [L, 0], [L, W], [0, W], [0, 0]]);
    lines.push([[L / 2, 0], [L / 2, W]]);
    lines.push(arc(L / 2, cy, cc, 0, 2 * Math.PI, 64));
    for (const left of [true, false]) {
      const base = left ? 0 : L, dir = left ? 1 : -1, a0 = left ? 0 : Math.PI;
      const bx = base + dir * bOff, bbx = base + dir * bbOff, ftx = base + dir * (bbOff + ftd);
      const laneRect = [[base, cy - lane / 2], [ftx, cy - lane / 2], [ftx, cy + lane / 2], [base, cy + lane / 2]];
      paint.push(laneRect);
      lines.push([...laneRect, laneRect[0]]);
      lines.push(arc(ftx, cy, ftR, a0 - Math.PI / 2, a0 + Math.PI / 2, 32));
      dashes.push(arc(ftx, cy, ftR, a0 + Math.PI / 2, a0 + 1.5 * Math.PI, 32));
      lines.push([[bbx, cy - bbHalf], [bbx, cy + bbHalf]]);
      rims.push(arc(bx, cy, rimR, 0, 2 * Math.PI, 24));
      rims.push([[bbx, cy], [bx - dir * rimR, cy]]);
      if (ra > 0) {
        const ar = arc(bx, cy, ra, a0 - Math.PI / 2, a0 + Math.PI / 2, 24);
        lines.push([[bbx, ar[0][1]], ...ar, [bbx, ar[ar.length - 1][1]]]);
      }
      const phi = Math.asin(clamp(c3 / r3, 0, 1));
      const three = arc(bx, cy, r3, a0 - phi, a0 + phi, 48);
      lines.push([[base, three[0][1]], ...three, [base, three[three.length - 1][1]]]);
    }
    return { L, W, paint, lines, dashes, rims };
  }

  // =========================================================================== live court (canvas)

  const courtView = { canvas: null, ctx: null, base: null, w: 1, h: 1, dpr: 1, scale: 1, ox: 0, oy: 0, ariaAt: 0 };
  const pxX = (x) => courtView.ox + x * courtView.scale;
  const pxY = (y) => courtView.oy + (S.geo.W - y) * courtView.scale;

  function courtResize() {
    const r = byId("court-wrap").getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    Object.assign(courtView, { dpr, w: Math.max(1, Math.round(r.width * dpr)), h: Math.max(1, Math.round(r.height * dpr)) });
    courtView.canvas.width = courtView.w;
    courtView.canvas.height = courtView.h;
    courtView.base = null;
    renderers.court(); // draw now: resizing cleared the canvas
  }

  function tracePath(g, pts, close) {
    g.beginPath();
    pts.forEach(([x, y], i) => (i ? g.lineTo(pxX(x), pxY(y)) : g.moveTo(pxX(x), pxY(y))));
    if (close) g.closePath();
  }

  function buildCourtBase() {
    const geo = S.geo, v = courtView;
    const pad = 2;
    v.scale = Math.min(v.w / (geo.L + 2 * pad), v.h / (geo.W + 2 * pad));
    v.ox = (v.w - geo.L * v.scale) / 2;
    v.oy = (v.h - geo.W * v.scale) / 2;
    const off = document.createElement("canvas");
    off.width = v.w;
    off.height = v.h;
    const g = off.getContext("2d");
    g.fillStyle = C.floor;
    g.fillRect(0, 0, v.w, v.h);
    g.fillStyle = C.paint;
    for (const poly of geo.paint) { tracePath(g, poly, true); g.fill(); }
    g.lineJoin = "round";
    g.lineCap = "round";
    g.strokeStyle = C.line;
    g.lineWidth = Math.max(1, v.dpr * 1.25);
    for (const pl of geo.lines) { tracePath(g, pl); g.stroke(); }
    g.setLineDash([4 * v.dpr, 4 * v.dpr]);
    for (const pl of geo.dashes) { tracePath(g, pl); g.stroke(); }
    g.setLineDash([]);
    g.strokeStyle = C.rim;
    g.lineWidth = Math.max(1.5, v.dpr * 1.6);
    for (const pl of geo.rims) { tracePath(g, pl); g.stroke(); }
    v.base = off;
  }

  renderers.court = () => {
    const v = courtView;
    if (!v.ctx) return;
    if (!S.geo) S.geo = courtGeometry(S.court || {});
    if (!v.base) buildCourtBase();
    const g = v.ctx;
    g.drawImage(v.base, 0, 0);
    const f = S.frame;
    byId("court-overlay").hidden = !!f;
    if (!f) return;
    const dpr = v.dpr, r = Math.max(8 * dpr, 1.15 * v.scale);
    const holder = f.ball && f.ball.holder;
    g.textAlign = "center";
    g.textBaseline = "middle";
    g.font = `650 ${Math.round(r * 1.0)}px ${FONT}`;
    for (const p of f.players || []) {
      if (!isNum(p.x) || !isNum(p.y)) continue;
      const x = pxX(p.x), y = pxY(p.y);
      const team = p.team === "home" || p.team === "away" ? p.team : null;
      if (holder && holder === p.id) {
        g.beginPath();
        g.arc(x, y, r + 3.5 * dpr, 0, 2 * Math.PI);
        g.lineWidth = 2 * dpr;
        g.strokeStyle = C.ink;
        g.stroke();
      }
      g.beginPath();
      g.arc(x, y, r, 0, 2 * Math.PI);
      g.fillStyle = team ? C[team] : C.neutral;
      g.fill();
      g.lineWidth = 1.5 * dpr;
      g.strokeStyle = C.floor;
      g.stroke();
      g.fillStyle = team ? C[`${team}Ink`] : "#ffffff";
      g.fillText(String(p.jersey ?? "?"), x, y + 0.5 * dpr);
    }
    const b = f.ball;
    if (b && b.visible && isNum(b.x) && isNum(b.y)) {
      const x = pxX(b.x), y = pxY(b.y), br = Math.max(4 * dpr, 0.45 * v.scale);
      if (isNum(b.z) && b.z > 0.3) {
        g.beginPath();
        g.arc(x, y, br + b.z * v.scale * 0.35, 0, 2 * Math.PI);
        g.lineWidth = 1.5 * dpr;
        g.strokeStyle = C.ball;
        g.stroke();
      }
      g.beginPath();
      g.arc(x, y, br, 0, 2 * Math.PI);
      g.fillStyle = C.ball;
      g.fill();
      g.lineWidth = dpr;
      g.strokeStyle = C.ink;
      g.stroke();
    }
    const now = performance.now();
    if (now - v.ariaAt > 3000) {
      v.ariaAt = now;
      const n = (f.players || []).length;
      const poss = f.possession ? `${S.teams[f.possession]} ball` : "possession unknown";
      v.canvas.setAttribute("aria-label", `Top-down court: ${n} players tracked, ${poss}`);
      setText(byId("court-meta"), `Frame ${f.frame_idx ?? "–"} · video ${fmtVideo(f.t)} · ${n} players`);
    }
  };

  // =========================================================================== score bar / teams / status

  renderers.scorebar = () => {
    const f = S.frame, st = S.stats;
    const score = (st && st.score) || {};
    setText(byId("score-source-label"), st ? (st.score_source === "scoreboard" ? "Scoreboard" : "Observed points") : "Score unavailable");
    setText(byId("away-score"), fmtInt(score.away));
    setText(byId("home-score"), fmtInt(score.home));
    const period = num(heldClock("period"), st && st.period);
    const game = num(heldClock("game"), st && st.game_clock);
    const shotS = num(heldClock("shot"), st && st.shot_clock);
    setText(byId("period"), periodLabel(period));
    const clock = byId("game-clock");
    setText(clock, fmtClock(game));
    markStale(clock, "game", game, "Game clock");
    const shot = byId("shot-clock");
    setText(shot, fmtShot(shotS));
    shot.classList.toggle("low", isNum(shotS) && shotS <= 5);
    markStale(shot, "shot", shotS, "Shot clock");
    const poss = f && f.possession;
    byId("away-poss").classList.toggle("on", poss === "away");
    byId("home-poss").classList.toggle("on", poss === "home");
    setText(byId("poss-sr"), poss ? `${S.teams[poss]} possession` : "");
  };

  function teamAbbr(name) {
    const words = String(name || "").trim().split(/\s+/).filter(Boolean);
    return (words[0] || "").slice(0, 3).toUpperCase() || "–";
  }

  renderers.teams = () => {
    for (const team of ["away", "home"]) {
      const name = S.teams[team];
      setText(byId(`${team}-abbr`), teamAbbr(name));
      for (const id of [`${team}-name`, `legend-${team}`, `compare-${team}-name`, `box-${team}-title`, `op-${team}-label`, `sub-team-${team}`, `shot-filter-${team}`]) {
        setText(byId(id), name);
      }
      const opt = byId("ev-team").querySelector(`option[value="${team}"]`);
      if (opt) opt.textContent = name;
    }
    document.title = `${S.teams.away} at ${S.teams.home} · CourtVision`;
  };

  renderers.status = () => {
    byId("paused-flag").hidden = !S.paused;
    byId("muted-flag").hidden = !S.muted;
    const mute = byId("btn-mute");
    mute.setAttribute("aria-pressed", String(S.muted));
    mute.firstChild.textContent = S.muted ? "Unmute narration " : "Mute narration ";
    for (const b of byId("speed-group").querySelectorAll("[data-speed]")) {
      b.setAttribute("aria-pressed", String(Number(b.dataset.speed) === S.speed));
    }
    setText(byId("op-status"), `${S.paused ? "Paused" : "Running"} · ${S.speed}× · narration ${S.muted ? "muted" : "on"}`);
  };

  // =========================================================================== team comparison

  const COMPARE_ROWS = [
    ["Points", (t) => fmtInt(t.pts), (t) => t.pts],
    ["Field goals", (t) => `${fmtMade(t.fgm, t.fga)} (${fmtPct(t.fga ? t.fgm / t.fga : null)})`, (t) => (t.fga ? t.fgm / t.fga : null)],
    ["3-pointers", (t) => `${fmtMade(t.fg3m, t.fg3a)} (${fmtPct(t.fg3a ? t.fg3m / t.fg3a : null)})`, (t) => (t.fg3a ? t.fg3m / t.fg3a : null)],
    ["Free throws", (t) => `${fmtMade(t.ftm, t.fta)}`, (t) => t.ftm],
    ["eFG%", (t) => fmtPct(efgOf(t)), (t) => efgOf(t)],
    ["TS%", (t) => fmtPct(tsOf(t)), (t) => tsOf(t)],
    ["Possessions", (t) => fmtInt(possOf(t)), null],
    ["Points / poss", (t) => fmtNum(pppOf(t), 2), (t) => pppOf(t)],
    ["Pace", (t) => fmtNum(t.pace, 1), null],
    ["xPTS (avg xFG)", (t) => `${fmtNum(t.xpts, 1)} (${fmtPct(t.avg_xfg, 0)})`, (t) => t.xpts],
    ["Rebounds (off)", (t) => `${fmtInt(rebOf(t))} (${fmtInt(t.oreb)})`, (t) => rebOf(t)],
    ["Assists", (t) => fmtInt(t.ast), (t) => t.ast],
    ["Turnovers", (t) => fmtInt(t.tov), (t) => (isNum(t.tov) ? -t.tov : null)],
    ["Steals / blocks", (t) => `${fmtInt(t.stl)} / ${fmtInt(t.blk)}`, null],
    ["Fouls", (t) => fmtInt(t.pf), null],
  ];

  function efgOf(t) {
    if (isNum(t.efg)) return t.efg;
    return t.fga ? (num(t.fgm, 0) + 0.5 * num(t.fg3m, 0)) / t.fga : null;
  }
  function tsOf(t) {
    if (isNum(t.ts)) return t.ts;
    const d = 2 * (num(t.fga, 0) + 0.44 * num(t.fta, 0));
    return d ? num(t.pts, 0) / d : null;
  }
  const possOf = (t) => (isNum(t.possessions) ? t.possessions : t.possessions_est);
  function pppOf(t) {
    if (isNum(t.ppp)) return t.ppp;
    const p = possOf(t);
    return p ? num(t.pts, 0) / p : null;
  }
  // TeamLine.oreb/dreb already include team rebounds (team_reb = team_oreb + team_dreb).
  const rebOf = (t) => (isNum(t.reb) ? t.reb : num(t.oreb, 0) + num(t.dreb, 0));

  renderers.compare = () => {
    const teams = (S.stats && S.stats.teams) || {};
    const a = teams.away || {}, hm = teams.home || {};
    const body = byId("compare-body");
    if (body.children.length !== COMPARE_ROWS.length) {
      body.replaceChildren(...COMPARE_ROWS.map(([label]) => h("tr", null, h("th", { scope: "row", text: label }), h("td"), h("td"))));
    }
    COMPARE_ROWS.forEach(([, show, key], i) => {
      const [, ta, th] = body.children[i].children;
      setText(ta, show(a));
      setText(th, show(hm));
      const va = key ? key(a) : null, vh = key ? key(hm) : null;
      ta.classList.toggle("lead", isNum(va) && isNum(vh) && va > vh);
      th.classList.toggle("lead", isNum(va) && isNum(vh) && vh > va);
    });
    const run = S.stats && S.stats.run;
    let meta = " ";
    if (run && run.team && num(run.points, 0) >= 6) {
      meta = `${S.teams[run.team] || run.team} on a ${fmtInt(run.points)}-${fmtInt(num(run.opp_points, 0))} run`;
    }
    setText(byId("compare-meta"), meta);
  };

  // =========================================================================== coach insights

  const SAMPLE_KEYS = ["n", "sample_size", "n_poss", "possessions", "attempts", "n_shots", "shots", "fga"];

  function sampleSize(ev) {
    for (const k of SAMPLE_KEYS) {
      if (ev && isNum(ev[k])) return `n = ${fmtInt(ev[k])}${/poss/.test(k) ? " poss" : /shot|fga|attempt/.test(k) ? " shots" : ""}`;
    }
    return "n not reported";
  }

  renderers.insights = () => {
    const list = byId("insight-list");
    const all = [...S.insights.values()].sort((a, b) => num(b.priority, 0) - num(a.priority, 0) || num(b.t, 0) - num(a.t, 0));
    const top = all.slice(0, 3);
    if (!top.length) {
      list.replaceChildren(h("li", { class: "empty", text: "No insights yet. Cards appear when the numbers support them." }));
    } else {
      list.replaceChildren(...top.map((ins) => h("li", { class: `insight${ins.team ? ` team-${ins.team}` : ""}` },
        h("div", { class: "insight-top" },
          h("span", { text: `${humanize(ins.kind)}${ins.team ? ` · ${S.teams[ins.team] || ins.team}` : ""}` }),
          h("span", { class: "insight-n", text: `${sampleSize(ins.evidence)} · ${timeLabel(ins.t)}` })),
        h("p", { class: "insight-text", text: ins.text || "" }))));
    }
    setText(byId("insight-more"), all.length > 3 ? `${all.length - 3} older` : " ");
  };

  // =========================================================================== transcript

  let transcriptRendered = 0;
  const CHANNELS = { play_by_play: "Play-by-play", color: "Colour", coach: "Coach", accessibility: "AD", correction: "Correction" };

  function utteranceNode(u) {
    const badges = [h("span", { class: "badge", text: CHANNELS[u.channel] || humanize(u.channel) })];
    badges.push(h("span", { class: `badge${u.source === "llm" ? " badge-llm" : ""}`, text: u.source === "llm" ? "Claude" : "Template" }));
    if (u.hedge || u.hedged) badges.push(h("span", { class: "badge badge-hedge", text: "Hedged" }));
    if (u.validated === false) badges.push(h("span", { class: "badge badge-unvalidated", text: "Unvalidated" }));
    if (isNum(u.latency_ms)) badges.push(h("span", { text: `${(u.latency_ms / 1000).toFixed(1)} s` }));
    badges.push(h("span", { text: timeLabel(u.t) }));
    return h("li", { class: `line ch-${u.channel || "other"}` },
      h("p", { class: "line-text", text: u.text || "" }),
      h("div", { class: "line-meta" }, badges));
  }

  renderers.transcript = () => {
    const list = byId("transcript-list");
    for (; transcriptRendered < S.utter.length; transcriptRendered++) {
      list.prepend(utteranceNode(S.utter[transcriptRendered]));
    }
    while (list.children.length > 200) list.lastChild.remove();
    if (S.utter.length > 1000) {
      S.utter.splice(0, S.utter.length - 500);
      transcriptRendered = S.utter.length;
    }
    setText(byId("transcript-count"), `${S.utterIds.size} lines`);
  };

  // =========================================================================== event feed

  const feedNodes = new Map(); // event_id -> {el, sig}

  function feedFilter(e, mode) {
    if (mode === "all") return true;
    if (mode === "review") return !!(e.attrs && e.attrs.needs_review) || e.kind === "score_correction";
    return e.kind !== "dribble" && e.kind !== "pass";
  }

  function valueWord(v) {
    return v === 1 ? "a free throw" : v === 2 ? "a 2" : v === 3 ? "a 3" : "a shot";
  }

  function describe(e) {
    const p = pname(e.player_id), o = pname(e.other_player_id);
    const tm = e.team ? S.teams[e.team] || e.team : "Team";
    const a = e.attrs || {};
    switch (e.kind) {
      case "possession": return `${tm} ball${a.cause ? ` · ${humanize(a.cause).toLowerCase()}` : ""}`;
      case "pass": return `${p} pass to ${o}${a.completed === false ? " (not completed)" : ""}`;
      case "dribble": return `${p} dribble`;
      case "shot_attempt": {
        if (a.shot_type === "free_throw") return `${p} free throw attempt`;
        const dist = isNum(a.distance_ft) ? `, ${a.distance_ft.toFixed(0)} ft` : "";
        return `${p} ${e.value || "?"}PT ${a.shot_type ? humanize(a.shot_type).toLowerCase() : "shot"}${dist}`;
      }
      case "shot_made": return `${p} makes ${valueWord(e.value)}`;
      case "shot_missed": return `${p} misses`;
      case "block": return `${p} blocks ${o}`;
      case "rebound": return `${e.player_id ? p : `${tm} team`} ${a.offensive ? "offensive" : "defensive"} rebound`;
      case "assist": return `${p} assist to ${o}`;
      case "steal": return `${p} steals from ${o}`;
      case "turnover": return `${e.player_id ? p : tm} turnover${a.type ? ` (${humanize(a.type).toLowerCase()})` : ""}`;
      case "foul": return `Foul on ${p}${e.other_player_id ? `, ${o} fouled` : ""}`;
      case "substitution": return `${p} in for ${o}`;
      case "timeout": return `${tm} timeout`;
      case "period_start": return "Period start";
      case "period_end": return "Period end";
      case "score_correction": return `Score correction ${tm} ${fmtSigned(e.value)}${a.reason ? `: ${a.reason}` : ""}`;
      default: return humanize(e.kind);
    }
  }

  function fillEventRow(li, e) {
    const retracted = e.status === "retracted";
    li.className = `ev${e.team ? ` team-${e.team}` : ""}${retracted ? " retracted" : ""}${num(e.revision, 0) > 0 ? " revised" : ""}${SCORING.has(e.kind) ? " ev-score" : ""}`;
    const badges = [h("span", { class: `badge badge-${e.status}`, text: e.status || "provisional" })];
    if (e.source && e.source !== "cv") badges.push(h("span", { class: "badge", text: e.source }));
    if (num(e.revision, 0) > 0) badges.push(h("span", { class: "badge badge-rev", text: `rev ${e.revision}` }));
    if (isNum(e.confidence) && e.confidence < 1 && !retracted) badges.push(h("span", { class: "badge", text: `${Math.round(e.confidence * 100)}%` }));
    if (e.attrs && e.attrs.needs_review) badges.push(h("span", { class: "badge badge-review", text: "Review" }));
    const label = describe(e);
    li.replaceChildren(
      h("span", { class: "ev-time", text: timeLabel(e.t) }),
      h("div", { class: "ev-desc" }, h("span", { class: "tdot", "aria-hidden": "true" }), h("span", { class: "ev-label", text: label, title: label })),
      h("div", { class: "ev-meta" }, badges),
      h("div", { class: "ev-actions" }, h("button", {
        type: "button", class: "link-btn", text: "Correct", "aria-label": `Correct event: ${label}`,
        onclick: () => openEventDialog(S.events.get(e.event_id) || e),
      })),
    );
  }

  renderers.feed = () => {
    const list = byId("feed-list");
    const mode = byId("feed-filter").value;
    const evs = [...S.events.values()].filter((e) => feedFilter(e, mode))
      .sort((a, b) => num(b.t, 0) - num(a.t, 0) || String(b.event_id).localeCompare(String(a.event_id)))
      .slice(0, 200);
    const empty = list.querySelector(".empty");
    if (empty) empty.remove();
    const keep = new Set();
    let ref = list.firstChild;
    for (const e of evs) {
      keep.add(e.event_id);
      let entry = feedNodes.get(e.event_id);
      const sig = [e.revision, e.status, e.kind, e.player_id, e.other_player_id, e.team, e.value, e.confidence, S.rosterVersion, S.clock.length > 0].join("|");
      if (!entry) {
        entry = { el: h("li"), sig: "" };
        feedNodes.set(e.event_id, entry);
      }
      if (entry.sig !== sig) {
        fillEventRow(entry.el, e);
        entry.sig = sig;
      }
      if (entry.el !== ref) list.insertBefore(entry.el, ref);
      else ref = ref.nextSibling;
    }
    for (const [id, entry] of feedNodes) {
      if (!keep.has(id)) {
        entry.el.remove();
        feedNodes.delete(id);
      }
    }
    if (!evs.length) list.append(h("li", { class: "empty", text: mode === "review" ? "Nothing needs review." : "No events yet." }));
  };

  // =========================================================================== event correction dialog

  let dialogEvent = null;

  function fillPlayerSelect(sel, current) {
    const opts = [h("option", { value: "", text: "— none —" })];
    for (const team of ["away", "home"]) {
      const group = h("optgroup", { label: S.teams[team] });
      for (const e of S.roster.values()) {
        if (e.team === team) group.append(h("option", { value: e.player_id, text: `#${e.jersey} ${e.name || ""}`.trim() }));
      }
      if (group.children.length) opts.push(group);
    }
    if (current && !S.roster.has(current)) opts.push(h("option", { value: current, text: pname(current) }));
    sel.replaceChildren(...opts);
    sel.value = current || "";
  }

  function openEventDialog(e) {
    dialogEvent = e || null;
    const kindSel = byId("ev-kind");
    if (!kindSel.options.length) kindSel.replaceChildren(...KINDS.map((k) => h("option", { value: k, text: humanize(k) })));
    setText(byId("ev-title"), e ? "Correct event" : "Add event");
    setText(byId("ev-sub"), e ? `${e.event_id} · rev ${num(e.revision, 0)} · ${e.source || "cv"} · ${describe(e)}` : "Operator entry (source manual, confirmed)");
    kindSel.value = e ? e.kind : "shot_made";
    byId("ev-team").value = (e && e.team) || "";
    fillPlayerSelect(byId("ev-player"), e && e.player_id);
    fillPlayerSelect(byId("ev-other"), e && e.other_player_id);
    byId("ev-value").value = String(e ? num(e.value, 0) : 2);
    byId("ev-status").value = "confirmed";
    byId("ev-status").disabled = !e;
    const t = e ? e.t : latestT();
    byId("ev-t").value = isNum(t) ? String(Math.round(t * 10) / 10) : "";
    byId("ev-reason").value = "";
    setText(byId("ev-error"), "");
    byId("event-dialog").showModal();
  }

  function eventBody() {
    const e = dialogEvent;
    const kind = byId("ev-kind").value;
    const status = byId("ev-status").value;
    const t = Number(byId("ev-t").value);
    const rawValue = byId("ev-value").value.trim();
    const body = {
      kind,
      t: isNum(t) && byId("ev-t").value !== "" ? t : num(latestT(), 0),
      team: byId("ev-team").value || null,
      player_id: byId("ev-player").value || null,
      other_player_id: byId("ev-other").value || null,
      value: rawValue === "" ? 0 : Number(rawValue),
      status,
      attrs: {},
    };
    const reason = byId("ev-reason").value.trim();
    if (e) {
      Object.assign(body, {
        event_id: e.event_id, frame_idx: Math.max(0, num(e.frame_idx, 0)), x: isNum(e.x) ? e.x : null,
        y: isNum(e.y) ? e.y : null, shot_id: e.shot_id || null, revision: num(e.revision, 0) + 1,
      });
      body.attrs = Object.assign({}, e.attrs || {}, { corrected_from_revision: num(e.revision, 0) });
    }
    if (reason) body.attrs.reason = reason;
    return body;
  }

  function valueProblem(b) {
    const v = b.value, st = b.attrs && b.attrs.shot_type;
    if (!Number.isInteger(v)) return "Value must be a whole number.";
    if (b.kind === "shot_attempt" || b.kind === "shot_made") {
      if (![1, 2, 3].includes(v)) return "Shots need a value of 1, 2 or 3.";
      if (st === "free_throw" && v !== 1) return "A free throw is worth 1 point.";
      if (FG_SHOT_TYPES.has(st) && v === 1) return `A ${st} field goal is worth 2 or 3 points.`;
      return "";
    }
    if (b.kind === "score_correction") return v === 0 ? "A score correction needs a signed point change (e.g. -2)." : "";
    if (b.kind === "period_start" || b.kind === "period_end") return v >= 0 && v <= 20 ? "" : "Value is the period number (0-20).";
    return v >= 0 && v <= 3 ? "" : `${humanize(b.kind)} takes a value of 0-3.`;
  }

  function validateEventBody(b) {
    if (b.status === "retracted") return "";
    const vp = valueProblem(b);
    if (vp) return vp;
    if (NEEDS_PLAYER.has(b.kind) && !b.player_id) return `${humanize(b.kind)} needs a player.`;
    if (b.player_id && b.player_id === b.other_player_id) return "Player and other player must differ.";
    const team = b.team || teamOfPlayer(b.player_id);
    const actual = teamOfPlayer(b.player_id);
    if (b.team && actual && actual !== b.team) return `${pname(b.player_id)} is not on ${S.teams[b.team]}.`;
    if (!team && !TEAMLESS_OK.has(b.kind)) return `${humanize(b.kind)} needs a team.`;
    const other = teamOfPlayer(b.other_player_id);
    if (team && other) {
      if (SAME_TEAM_PAIR.has(b.kind) && other !== team) return "The other player must be a teammate.";
      if (OPPONENT_PAIR.has(b.kind) && other === team) return "The other player must be an opponent.";
    }
    return "";
  }

  async function submitEvent(ev) {
    ev.preventDefault();
    const body = eventBody();
    const problem = validateEventBody(body);
    if (problem) {
      setText(byId("ev-error"), problem);
      return;
    }
    const btn = byId("btn-ev-submit");
    btn.disabled = true;
    try {
      await api("api/events", body);
      byId("event-dialog").close();
      toast(body.status === "retracted" ? "Event retracted" : dialogEvent ? "Correction sent" : "Event added");
    } catch (err) {
      setText(byId("ev-error"), err.message || String(err));
    } finally {
      btn.disabled = false;
    }
  }

  // =========================================================================== box score

  const BOX_COLS = [
    ["MIN", (l) => fmtMin(l.seconds)],
    ["PTS", (l) => fmtInt(l.pts)],
    ["FG", (l) => fmtMade(l.fgm, l.fga)],
    ["3PT", (l) => fmtMade(l.fg3m, l.fg3a)],
    ["FT", (l) => fmtMade(l.ftm, l.fta)],
    ["OREB", (l) => fmtInt(l.oreb)],
    ["DREB", (l) => fmtInt(l.dreb)],
    ["REB", (l) => fmtInt(rebOf(l))],
    ["AST", (l) => fmtInt(l.ast)],
    ["STL", (l) => fmtInt(l.stl)],
    ["BLK", (l) => fmtInt(l.blk)],
    ["TO", (l) => fmtInt(l.tov)],
    ["PF", (l) => fmtInt(l.pf)],
    ["+/-", (l) => (num(l.seconds, 0) > 0 ? fmtSigned(l.plus_minus) : "–")],
    ["xPTS", (l) => fmtNum(l.xpts, 1)],
  ];
  const boxRows = { away: new Map(), home: new Map() };

  function ensureBoxHead(table, team) {
    if (table.tHead) return;
    table.append(
      h("caption", { class: "sr-only", text: `${S.teams[team]} box score` }),
      h("thead", null, h("tr", null, h("th", { scope: "col", text: "Player" }), BOX_COLS.map(([label]) => h("th", { scope: "col", text: label })))),
      h("tbody"),
    );
  }

  function boxRow(rows, key) {
    let tr = rows.get(key);
    if (!tr) {
      tr = h("tr", null, h("th", { scope: "row" }), BOX_COLS.map(() => h("td")));
      rows.set(key, tr);
    }
    return tr;
  }

  function fillBoxRow(tr, label, jersey, line, provisional) {
    const head = tr.firstChild;
    const sig = `${label}|${jersey}|${provisional}`;
    if (head.dataset.sig !== sig) {
      head.dataset.sig = sig;
      fill(head, h("span", { class: "jersey", text: jersey ? `${jersey}` : "" }), ` ${label}`,
        provisional ? h("span", { class: "prov-mark", title: "Provisional", text: "P" }) : null);
      head.title = label;
    }
    tr.classList.toggle("dnp", !line);
    const cells = tr.children;
    BOX_COLS.forEach(([, fn], i) => setText(cells[i + 1], line ? fn(line) : "–"));
  }

  function renderBoxTeam(team) {
    const table = byId(`box-${team}-table`);
    ensureBoxHead(table, team);
    const tbody = table.tBodies[0];
    const players = (S.stats && S.stats.players) || {};
    const ids = [...S.roster.values()].filter((e) => e.team === team).map((e) => e.player_id);
    const extra = Object.keys(players).filter((id) => !S.roster.has(id) && players[id] && players[id].team === team).sort();
    const rows = boxRows[team];
    const order = [];
    for (const id of [...ids, ...extra]) {
      const line = players[id];
      const tr = boxRow(rows, id);
      fillBoxRow(tr, pname(id), pjersey(id), line, !!(line && line.provisional));
      order.push(tr);
    }
    const teamLine = S.stats && S.stats.teams && S.stats.teams[team];
    const totals = boxRow(rows, "__totals");
    totals.className = "totals";
    fillBoxRow(totals, "Team totals", "", teamLine ? Object.assign({}, teamLine, { seconds: 0 }) : null, false);
    order.push(totals);
    const live = new Set(order);
    for (const [key, tr] of rows) if (!live.has(tr)) { tr.remove(); rows.delete(key); }
    order.forEach((tr, i) => { if (tbody.children[i] !== tr) tbody.insertBefore(tr, tbody.children[i] || null); });
    const t = teamLine || {};
    setText(byId(`box-${team}-summary`), teamLine
      ? `eFG ${fmtPct(efgOf(t))} · TS ${fmtPct(tsOf(t))} · PPP ${fmtNum(pppOf(t), 2)} · Pace ${fmtNum(t.pace, 1)} · xPTS ${fmtNum(t.xpts, 1)}`
      : "No stats yet");
  }

  renderers.box = () => {
    renderBoxTeam("away");
    renderBoxTeam("home");
  };

  // =========================================================================== shot chart (SVG)

  const shotView = { base: false, marks: null, shots: [], sig: "" };

  function ptsAttr(pts) {
    return pts.map(([x, y]) => `${x.toFixed(2)},${y.toFixed(2)}`).join(" ");
  }

  function buildShotBase() {
    const el = byId("shot-svg");
    const geo = S.geo || courtGeometry({});
    const pad = 2;
    el.setAttribute("viewBox", `${-pad} ${-pad} ${geo.L + 2 * pad} ${geo.W + 2 * pad}`);
    const g = svg("g", { transform: `translate(0 ${geo.W}) scale(1 -1)` });
    g.append(svg("rect", { x: 0, y: 0, width: geo.L, height: geo.W, class: "court-svg-floor" }));
    for (const poly of geo.paint) g.append(svg("polygon", { points: ptsAttr(poly), class: "court-svg-paint" }));
    for (const pl of geo.lines) g.append(svg("polyline", { points: ptsAttr(pl), class: "court-svg-line" }));
    for (const pl of geo.dashes) g.append(svg("polyline", { points: ptsAttr(pl), class: "court-svg-dash" }));
    for (const pl of geo.rims) g.append(svg("polyline", { points: ptsAttr(pl), class: "court-svg-rim" }));
    shotView.marks = svg("g", { "aria-hidden": "true" });
    g.append(shotView.marks);
    el.replaceChildren(g);
    shotView.base = true;
    shotView.sig = "";
  }

  renderers.shots = () => {
    if (!shotView.base) buildShotBase();
    const all = ((S.stats && S.stats.shots) || []).filter((sh) => isNum(sh.x) && isNum(sh.y) && sh.shot_type !== "free_throw");
    const shots = S.shotFilter === "all" ? all : all.filter((sh) => sh.team === S.shotFilter);
    const sig = `${S.shotFilter}|${shots.length}|${shots.map((sh) => `${sh.shot_id}:${sh.made}:${sh.xfg}`).join(",")}|${C.seq.join()}`;
    if (sig === shotView.sig) return;
    shotView.sig = sig;
    shotView.shots = shots;
    const nodes = shots.map((sh, i) => {
      const color = xfgColor(sh.xfg);
      const g = svg("g", { class: "shot", "data-i": i });
      g.append(svg("circle", { cx: sh.x, cy: sh.y, r: 1.7, class: "hit" }));
      if (sh.made === true) g.append(svg("circle", { cx: sh.x, cy: sh.y, r: 0.85, class: "made", fill: color }));
      else g.append(svg("circle", { cx: sh.x, cy: sh.y, r: 0.72, class: "miss", stroke: color, "stroke-dasharray": sh.made == null ? "0.4 0.3" : null }));
      return g;
    });
    shotView.marks.replaceChildren(...nodes);
    const made = shots.filter((sh) => sh.made === true).length;
    const decided = shots.filter((sh) => sh.made != null).length;
    const xs = shots.map((sh) => sh.xfg).filter(isNum);
    const threes = shots.filter((sh) => sh.value === 3 && sh.made === true).length;
    const who = S.shotFilter === "all" ? "Both teams" : S.teams[S.shotFilter];
    setText(byId("shot-summary"), shots.length
      ? `${who}: FG ${made}/${decided} (${fmtPct(decided ? made / decided : null)}) · eFG ${fmtPct(decided ? (made + 0.5 * threes) / decided : null)} · avg xFG ${fmtPct(xs.length ? xs.reduce((s, v) => s + v, 0) / xs.length : null)}`
      : `${who}: no field-goal attempts yet`);
  };

  function showShotTip(ev) {
    const g = ev.target.closest && ev.target.closest(".shot");
    const tip = byId("shot-tooltip");
    if (!g) { tip.hidden = true; return; }
    const sh = shotView.shots[Number(g.getAttribute("data-i"))];
    if (!sh) return;
    const row = (k, v) => h("div", { class: "tt-row" }, h("span", { text: k }), h("span", { text: v }));
    fill(tip,
      h("strong", { text: `${pname(sh.player_id)} · ${sh.value || "?"}PT ${sh.made === true ? "made" : sh.made === false ? "missed" : "pending"}` }),
      row("xFG", fmtPct(sh.xfg)),
      row("Distance", isNum(sh.distance_ft) ? `${sh.distance_ft.toFixed(1)} ft` : "–"),
      row("Zone", humanize(sh.zone) || "–"),
      row("Type", humanize(sh.shot_type) || "–"),
      row("Closest defender", isNum(sh.closest_defender_ft) ? `${sh.closest_defender_ft.toFixed(1)} ft` : "–"),
      sh.assisted_by ? row("Assist", pname(sh.assisted_by)) : null,
      row("Time", timeLabel(sh.t)),
    );
    tip.hidden = false;
    const wrap = byId("shot-wrap").getBoundingClientRect();
    const x = ev.clientX - wrap.left, y = ev.clientY - wrap.top;
    const w = tip.offsetWidth, ht = tip.offsetHeight;
    tip.style.left = `${clamp(x + 12, 0, Math.max(0, wrap.width - w))}px`;
    tip.style.top = `${y + 12 + ht > wrap.height ? Math.max(0, y - ht - 12) : y + 12}px`;
  }

  // =========================================================================== decision quality

  function optionText(o) {
    if (!o) return "–";
    const what = o.kind === "pass" ? `pass to ${pname(o.target_id)}` : o.kind === "shoot" ? "shoot" : humanize(o.kind);
    return `${what} (${fmtNum(o.value, 2)})`;
  }

  renderers.decisions = () => {
    const list = byId("decision-list");
    const decs = ((S.stats && S.stats.decisions) || []).slice(-40).reverse();
    if (!decs.length) {
      list.replaceChildren(h("li", { class: "empty", text: "Graded passes and shots appear here." }));
      return;
    }
    list.replaceChildren(...decs.map((d) => {
      const label = ["good", "neutral", "poor"].includes(d.label) ? d.label : "neutral";
      const same = d.chosen && d.best && d.chosen.kind === d.best.kind && d.chosen.target_id === d.best.target_id;
      return h("li", { class: "dec" },
        h("span", { class: "dec-time", text: timeLabel(d.t) }),
        h("span", { class: "dec-who", text: pname(d.player_id) }),
        h("span", { class: "dec-what", text: `Chose ${optionText(d.chosen)}${same ? " · best option" : ` · best ${optionText(d.best)}`}` }),
        h("span", { class: "dec-grade" },
          h("span", { class: `badge badge-${label}`, text: `${label === "good" ? "▲" : label === "poor" ? "▼" : "●"} ${label}` }),
          h("span", { class: "dec-delta", text: `${fmtSigned(d.delta, 2)} pts` })));
    }));
  };

  // =========================================================================== operator: score override

  function parseClock(text) {
    const s = String(text || "").trim();
    if (!s) return null;
    const m = /^(\d{1,2}):(\d{2}(?:\.\d)?)$/.exec(s);
    if (m) return Number(m[1]) * 60 + Number(m[2]);
    const v = Number(s);
    return isNum(v) && v >= 0 ? v : NaN;
  }

  function prefillScore() {
    const score = (S.stats && S.stats.score) || {};
    const clk = { period: heldClock("period"), game: heldClock("game"), shot: heldClock("shot") };
    byId("op-away-score").value = isNum(score.away) ? score.away : "";
    byId("op-home-score").value = isNum(score.home) ? score.home : "";
    byId("op-period").value = isNum(clk.period) ? clk.period : "";
    byId("op-game-clock").value = isNum(clk.game) ? (clk.game >= 60 ? fmtClock(clk.game) : clk.game.toFixed(1)) : "";
    byId("op-shot-clock").value = isNum(clk.shot) ? clk.shot.toFixed(1) : "";
  }

  async function submitScore(ev) {
    ev.preventDefault();
    const intOf = (id) => (byId(id).value === "" ? null : Number(byId(id).value));
    const body = {
      away_score: intOf("op-away-score"),
      home_score: intOf("op-home-score"),
      period: intOf("op-period"),
      game_clock: parseClock(byId("op-game-clock").value),
      shot_clock: intOf("op-shot-clock"),
    };
    if (Number.isNaN(body.game_clock)) return toast("Game clock must be mm:ss or seconds", "error");
    for (const k of Object.keys(body)) if (body[k] == null) delete body[k];
    if (!Object.keys(body).length) return toast("Enter at least one value", "error");
    const t = latestT();
    if (isNum(t)) body.t = t;
    try { await send("api/scoreboard", body, "Scoreboard override applied"); } catch { /* toast shown */ }
  }

  // =========================================================================== operator: tap-to-sub

  const SUB = { team: "away", mode: "sub", out: null, in: null, five: new Set() };
  let lineups = { home: [], away: [] };

  function lineupKey() {
    return `cv-lineup:${S.teams.away}|${S.teams.home}`;
  }

  function loadLineups() {
    try {
      const raw = store.get("localStorage", lineupKey());
      const parsed = raw ? JSON.parse(raw) : null;
      lineups = parsed && Array.isArray(parsed.home) && Array.isArray(parsed.away) ? parsed : { home: [], away: [] };
    } catch {
      lineups = { home: [], away: [] };
    }
    lineups.home = lineups.home.filter((id) => S.roster.has(id));
    lineups.away = lineups.away.filter((id) => S.roster.has(id));
  }

  function saveLineups() {
    store.set("localStorage", lineupKey(), JSON.stringify(lineups));
  }

  function playerButton(e, role) {
    const pressed = SUB.mode === "five" ? SUB.five.has(e.player_id) : role === "out" ? SUB.out === e.player_id : SUB.in === e.player_id;
    return h("button", {
      type: "button", class: `pbtn ${role}`, "aria-pressed": String(pressed),
      "aria-label": `${e.name || ""} number ${e.jersey}${role === "out" ? ", on floor" : ", bench"}`,
      onclick: () => tapPlayer(e.player_id, role),
    }, h("span", { class: "num", text: e.jersey }), h("span", { class: "nm", text: e.name || "" }));
  }

  renderers.operator = () => {
    const team = SUB.team;
    byId("sub-team-away").setAttribute("aria-pressed", String(team === "away"));
    byId("sub-team-home").setAttribute("aria-pressed", String(team === "home"));
    const players = [...S.roster.values()].filter((e) => e.team === team);
    const floorIds = new Set(lineups[team]);
    const five = SUB.mode === "five" || floorIds.size === 0;
    if (five && SUB.mode !== "five") SUB.mode = "five";
    byId("btn-lineup-set").setAttribute("aria-pressed", String(SUB.mode === "five"));
    const floor = five ? [] : players.filter((e) => floorIds.has(e.player_id));
    const bench = five ? players : players.filter((e) => !floorIds.has(e.player_id));
    byId("sub-floor").replaceChildren(...(floor.length ? floor.map((e) => playerButton(e, "out")) : [h("p", { class: "muted", text: five ? "Pick the five starters from the list." : "Nobody on the floor." })]));
    byId("sub-bench").replaceChildren(...(bench.length ? bench.map((e) => playerButton(e, five ? "five" : "in")) : [h("p", { class: "muted", text: S.roster.size ? "No bench players." : "Waiting for the roster." })]));
    let summary = " ";
    let ready = false;
    if (SUB.mode === "five") {
      summary = `Starting five: ${SUB.five.size}/5 selected`;
      ready = SUB.five.size >= 1 && SUB.five.size <= 5;
      setText(byId("sub-hint"), "Select up to five players for the floor, then confirm.");
    } else {
      summary = SUB.out || SUB.in ? `${SUB.out ? `${pname(SUB.out)} out` : "pick player out"} · ${SUB.in ? `${pname(SUB.in)} in` : "pick player in"}` : " ";
      ready = !!(SUB.out && SUB.in);
      setText(byId("sub-hint"), "Tap a player on the floor, then the player coming in.");
    }
    setText(byId("sub-summary"), summary);
    byId("btn-sub-confirm").disabled = !ready;
  };

  function tapPlayer(id, role) {
    if (SUB.mode === "five") {
      if (SUB.five.has(id)) SUB.five.delete(id);
      else if (SUB.five.size < 5) SUB.five.add(id);
    } else if (role === "out") SUB.out = SUB.out === id ? null : id;
    else SUB.in = SUB.in === id ? null : id;
    mark("operator");
  }

  function clearSub() {
    SUB.out = null;
    SUB.in = null;
    SUB.five.clear();
    mark("operator");
  }

  async function confirmSub() {
    const team = SUB.team;
    const t = latestT();
    const body = SUB.mode === "five" ? { team, players: [...SUB.five] } : { team, out: SUB.out, in: SUB.in };
    if (isNum(t)) body.t = t;
    try {
      await send("api/lineup", body, SUB.mode === "five" ? "Lineup set" : `Sub: ${pname(SUB.in)} for ${pname(SUB.out)}`);
    } catch {
      return;
    }
    if (SUB.mode === "five") lineups[team] = [...SUB.five];
    else lineups[team] = lineups[team].filter((id) => id !== SUB.out).concat([SUB.in]);
    saveLineups();
    SUB.mode = "sub";
    clearSub();
  }

  // =========================================================================== operator: playback

  async function control(action, value) {
    const body = value === undefined ? { action } : { action, value };
    try {
      await send("api/control", body);
    } catch {
      return false;
    }
    if (action === "pause") S.paused = true;
    if (action === "resume") S.paused = false;
    if (action === "mute") S.muted = true;
    if (action === "unmute") S.muted = false;
    if (action === "speed") S.speed = value;
    if (action === "dead_ball") toast("Dead ball marked");
    mark("status");
    return true;
  }

  function onHotkey(ev) {
    if (ev.defaultPrevented || ev.ctrlKey || ev.metaKey || ev.altKey) return;
    const tag = (ev.target && ev.target.tagName) || "";
    if (["INPUT", "SELECT", "TEXTAREA"].includes(tag) || document.querySelector("dialog[open]")) return;
    const key = ev.key.toLowerCase();
    if (key === "m") control(S.muted ? "unmute" : "mute");
    else if (key === "p") control(S.paused ? "resume" : "pause");
    else if (key === "d") control("dead_ball");
    else return;
    ev.preventDefault();
  }

  // =========================================================================== frame tools (calibration, ROI)

  class FrameTool {
    constructor(prefix, opts) {
      this.canvas = byId(`${prefix}-canvas`);
      this.wrap = byId(`${prefix}-wrap`);
      this.empty = byId(`${prefix}-empty`);
      this.ctx = this.canvas.getContext("2d");
      this.opts = opts;
      this.img = null;
      this.cross = null;
      this.drag = null;
      this.focused = false;
      this.dpr = 1;
      if (window.ResizeObserver) new ResizeObserver(() => this.resize()).observe(this.wrap);
      this.canvas.addEventListener("pointerdown", (e) => this.down(e));
      this.canvas.addEventListener("pointermove", (e) => this.move(e));
      this.canvas.addEventListener("pointerup", (e) => this.up(e));
      this.canvas.addEventListener("keydown", (e) => this.key(e));
      this.canvas.addEventListener("focus", () => { this.focused = true; this.draw(); });
      this.canvas.addEventListener("blur", () => { this.focused = false; this.draw(); });
    }

    get size() { return this.img ? [this.img.width, this.img.height] : null; }
    get k() { return this.img ? this.canvas.width / this.img.width : 1; }

    async load() {
      try {
        this.img = await fetchFrameImage();
      } catch (err) {
        toast(err.message || String(err), "error");
        return;
      }
      this.wrap.style.aspectRatio = `${this.img.width} / ${this.img.height}`;
      this.empty.hidden = true;
      this.cross = [this.img.width / 2, this.img.height / 2];
      this.resize();
    }

    resize() {
      const r = this.wrap.getBoundingClientRect();
      this.dpr = Math.min(window.devicePixelRatio || 1, 2);
      this.canvas.width = Math.max(1, Math.round(r.width * this.dpr));
      this.canvas.height = Math.max(1, Math.round(r.height * this.dpr));
      this.draw();
    }

    toImage(ev) {
      const r = this.canvas.getBoundingClientRect();
      const [w, ht] = this.size;
      return [clamp(((ev.clientX - r.left) / r.width) * w, 0, w), clamp(((ev.clientY - r.top) / r.height) * ht, 0, ht)];
    }

    toCanvas(u, v) { return [u * this.k, v * (this.canvas.height / this.img.height)]; }

    down(ev) {
      if (!this.img) return;
      const [u, v] = this.toImage(ev);
      this.cross = [u, v];
      if (this.opts.dragMode && this.opts.dragMode()) {
        this.drag = [u, v, u, v];
        this.canvas.setPointerCapture(ev.pointerId);
      } else {
        this.opts.onPoint(u, v);
      }
      this.draw();
    }

    move(ev) {
      if (!this.drag) return;
      const [u, v] = this.toImage(ev);
      this.drag[2] = u;
      this.drag[3] = v;
      this.draw();
    }

    up() {
      if (!this.drag) return;
      const [a, b, c, d] = this.drag;
      this.drag = null;
      if (Math.abs(c - a) >= 3 && Math.abs(d - b) >= 3) this.opts.onRect([Math.min(a, c), Math.min(b, d), Math.max(a, c), Math.max(b, d)]);
      this.draw();
    }

    key(ev) {
      if (!this.img) return;
      const step = ev.shiftKey ? 10 : 1;
      const moves = { ArrowLeft: [-step, 0], ArrowRight: [step, 0], ArrowUp: [0, -step], ArrowDown: [0, step] };
      if (moves[ev.key]) {
        const [w, ht] = this.size;
        this.cross = [clamp(this.cross[0] + moves[ev.key][0], 0, w), clamp(this.cross[1] + moves[ev.key][1], 0, ht)];
        if (this.drag) { this.drag[2] = this.cross[0]; this.drag[3] = this.cross[1]; }
      } else if (ev.key === "Enter" || ev.key === " ") {
        const [u, v] = this.cross;
        if (this.opts.dragMode && this.opts.dragMode()) {
          if (!this.drag) this.drag = [u, v, u, v];
          else this.up();
        } else {
          this.opts.onPoint(u, v);
        }
      } else if (ev.key === "Escape" && this.drag) {
        this.drag = null;
      } else {
        return;
      }
      ev.preventDefault();
      this.draw();
    }

    draw() {
      const g = this.ctx;
      g.clearRect(0, 0, this.canvas.width, this.canvas.height);
      if (!this.img) return;
      g.drawImage(this.img, 0, 0, this.canvas.width, this.canvas.height);
      this.opts.onDraw(g, this);
      const dpr = this.dpr;
      if (this.drag) {
        const [a, b] = this.toCanvas(this.drag[0], this.drag[1]);
        const [c, d] = this.toCanvas(this.drag[2], this.drag[3]);
        g.setLineDash([5 * dpr, 4 * dpr]);
        g.strokeStyle = "#ffffff";
        g.lineWidth = 1.5 * dpr;
        g.strokeRect(a, b, c - a, d - b);
        g.setLineDash([]);
      }
      if (this.focused && this.cross) {
        const [x, y] = this.toCanvas(this.cross[0], this.cross[1]);
        g.strokeStyle = "#ffffff";
        g.lineWidth = 1 * dpr;
        g.beginPath();
        g.moveTo(x - 12 * dpr, y); g.lineTo(x + 12 * dpr, y);
        g.moveTo(x, y - 12 * dpr); g.lineTo(x, y + 12 * dpr);
        g.stroke();
      }
    }
  }

  function marker(g, tool, u, v, label, color) {
    const [x, y] = tool.toCanvas(u, v);
    const dpr = tool.dpr;
    g.beginPath();
    g.arc(x, y, 5 * dpr, 0, 2 * Math.PI);
    g.fillStyle = color;
    g.fill();
    g.lineWidth = 2 * dpr;
    g.strokeStyle = "#ffffff";
    g.stroke();
    g.font = `600 ${12 * dpr}px ${FONT}`;
    g.textAlign = "left";
    g.textBaseline = "bottom";
    g.lineWidth = 3 * dpr;
    g.strokeStyle = "rgba(0,0,0,0.75)";
    g.strokeText(label, x + 7 * dpr, y - 4 * dpr);
    g.fillStyle = "#ffffff";
    g.fillText(label, x + 7 * dpr, y - 4 * dpr);
  }

  // ----------------------------------------------------------------------- calibration

  const CAL = { points: [], tool: null };
  const landmarkNames = () => {
    const all = Object.keys(S.landmarks || {}), preset = byId("cal-preset").value;
    const side = preset === "right" ? "R" : "L";
    const preferred = preset === "all" ? [] : preset === "full"
      ? ["L_corner_bottom", "L_corner_top", "R_corner_top", "R_corner_bottom"]
      : [`${side}_lane_baseline_bottom`, `${side}_lane_baseline_top`, `${side}_lane_ft_top`, `${side}_lane_ft_bottom`];
    return [...preferred.filter(n => all.includes(n)), ...all.filter(n => !preferred.includes(n))];
  };

  function nextLandmark() {
    const used = new Set(CAL.points.map((p) => p.landmark));
    return landmarkNames().find((n) => !used.has(n)) || "";
  }

  renderers.calibration = () => {
    const names = landmarkNames();
    const sel = byId("cal-landmark");
    byId("cal-landmark-row").hidden = !names.length;
    byId("cal-manual-row").hidden = !!names.length;
    const used = new Set(CAL.points.map((p) => p.landmark));
    const current = sel.value;
    sel.replaceChildren(...names.map((n) => h("option", { value: n, text: `${used.has(n) ? "✓ " : ""}${humanize(n)}` })));
    sel.value = names.includes(current) && !used.has(current) ? current : nextLandmark() || current || names[0] || "";
    renderCalibrationReference();
    const list = byId("cal-points");
    list.replaceChildren(...CAL.points.map((p, i) => h("li", null,
      h("span", { class: "pidx", text: i + 1 }),
      h("span", { text: `${p.landmark ? humanize(p.landmark) : `(${fmtNum(p.x, 2)}, ${fmtNum(p.y, 2)}) ft`} ← (${p.u.toFixed(0)}, ${p.v.toFixed(0)}) px` }),
      h("button", { type: "button", class: "link-btn", text: "Remove", "aria-label": `Remove point ${i + 1}`, onclick: () => { CAL.points.splice(i, 1); calChanged(); } }))));
    setText(byId("cal-count"), `${CAL.points.length} / 4 minimum`);
    byId("btn-cal-submit").disabled = CAL.points.length < 4;
    if (CAL.tool) CAL.tool.draw();
  };

  function calChanged() {
    mark("calibration");
  }

  function renderCalibrationReference() {
    const map = byId("cal-reference");
    if (!map) return;
    map.replaceChildren();
    const L = num(S.court && S.court.length, 94), W = num(S.court && S.court.width, 50);
    map.setAttribute("viewBox", `-5 -5 ${L + 10} ${W + 10}`);
    const line = { fill:"none", stroke:"currentColor", "stroke-width":.45 };
    map.append(svg("rect", {...line, x:0, y:0, width:L, height:W}),
      svg("path", {...line, d:`M${L/2} 0v${W} M0 17h19v16H0 M${L} 17h-19v16h19`}),
      svg("circle", {...line, cx:L/2, cy:W/2, r:6}));
    const name = byId("cal-landmark").value, pt = S.landmarks[name];
    if (pt) map.append(svg("circle", { cx:pt[0], cy:W - pt[1], r:1.5, fill:"#3765dd" }),
      svg("circle", { cx:pt[0], cy:W - pt[1], r:3, fill:"none", stroke:"#3765dd", "stroke-width":.5 }));
  }

  function addCalPoint(u, v) {
    if (landmarkNames().length) {
      const name = byId("cal-landmark").value;
      if (!name) return toast("Pick a landmark first", "error");
      CAL.points = CAL.points.filter((p) => p.landmark !== name);
      CAL.points.push({ u, v, landmark: name });
      byId("cal-landmark").value = nextLandmark();
    } else {
      const x = Number(byId("cal-manual-x").value), y = Number(byId("cal-manual-y").value);
      if (byId("cal-manual-x").value === "" || byId("cal-manual-y").value === "" || !isNum(x) || !isNum(y)) {
        return toast("Enter the court x / y (ft) of this point first", "error");
      }
      CAL.points.push({ u, v, x, y });
    }
    calChanged();
  }

  async function submitCalibration() {
    const pts = CAL.points;
    const body = { image_points: pts.map((p) => [p.u, p.v]) };
    // The JPEG may be scaled down from the source; image_size lets the controller rescale.
    if (CAL.tool && CAL.tool.size) body.image_size = CAL.tool.size.map((v) => Math.round(v));
    if (pts.every((p) => p.landmark)) body.landmarks = pts.map((p) => p.landmark);
    else body.court_points = pts.map((p) => [p.x, p.y]);
    const out = byId("cal-result");
    try {
      const res = await send("api/calibration", body);
      const r = (res && res.result) || {};
      const rmse = isNum(r.rmse_ft) ? r.rmse_ft : r.rmse;
      const inliers = Array.isArray(r.inliers) ? r.inliers.filter(Boolean).length : r.inliers;
      const good = isNum(rmse) && rmse <= 1.0; // 1.0 ft homography gate (RESEARCH D1)
      fill(out,
        h("div", { class: "big", text: isNum(rmse) ? `${rmse.toFixed(2)} ft` : "fitted" }),
        h("div", { class: good ? "status-good" : "status-bad", text: isNum(rmse) ? (good ? "✓ RMSE within the 1.0 ft gate" : "✕ RMSE above the 1.0 ft gate: check points") : "" }),
        isNum(inliers) ? h("div", { class: "muted", text: `${inliers} of ${pts.length} points used as inliers` }) : null,
      );
    } catch (err) {
      out.replaceChildren(h("div", { class: "status-bad", text: err.message || String(err) }));
    }
  }

  function drawCalibration(g, tool) {
    CAL.points.forEach((p, i) => marker(g, tool, p.u, p.v, `${i + 1} ${p.landmark ? humanize(p.landmark) : ""}`.trim(), "#2a78d6"));
  }

  // ----------------------------------------------------------------------- scoreboard ROI

  const ROI = { corners: [], cells: new Map(), history: [], mode: "corners", tool: null };
  const CORNER_NAMES = ["TL", "TR", "BR", "BL"];

  function setRoiMode(mode) {
    ROI.mode = mode;
    byId("roi-mode-corners").setAttribute("aria-pressed", String(mode === "corners"));
    byId("roi-mode-cells").setAttribute("aria-pressed", String(mode === "cells"));
    setText(byId("roi-hint"), mode === "corners"
      ? "Click the corners in order: top-left, top-right, bottom-right, bottom-left."
      : "Drag a box around each digit cell (keyboard: Enter to start and end a box).");
  }

  function nextCellName() {
    const names = [...byId("roi-cell-names").options].map((o) => o.value);
    return names.find((n) => !ROI.cells.has(n)) || "";
  }

  function addCell(box) {
    const name = byId("roi-cell-name").value.trim();
    if (!/^[A-Za-z0-9_.-]{1,32}$/.test(name)) return toast("Cell names use letters, digits, _ . - (max 32)", "error");
    ROI.cells.set(name, box.map((v) => Math.round(v * 10) / 10));
    ROI.history.push(["cell", name]);
    byId("roi-cell-name").value = nextCellName();
    roiChanged();
  }

  function roiPoint(u, v) {
    if (ROI.mode !== "corners") return;
    if (ROI.corners.length >= 4) return toast("Four corners set. Undo or clear to redo them.", "error");
    ROI.corners.push([Math.round(u * 10) / 10, Math.round(v * 10) / 10]);
    ROI.history.push(["corner"]);
    if (ROI.corners.length === 4) setRoiMode("cells");
    roiChanged();
  }

  function roiChanged() {
    byId("roi-corners").replaceChildren(...ROI.corners.map(([u, v], i) => h("li", null,
      h("span", { class: "pidx", text: CORNER_NAMES[i] }), h("span", { text: `(${u.toFixed(0)}, ${v.toFixed(0)}) px` }), h("span"))));
    byId("roi-cells").replaceChildren(...[...ROI.cells].map(([name, b]) => h("li", null,
      h("span", { class: "pidx", text: "▭" }),
      h("span", { text: `${name}: ${b.map((v) => v.toFixed(0)).join(", ")}` }),
      h("button", { type: "button", class: "link-btn", text: "Remove", "aria-label": `Remove cell ${name}`, onclick: () => { ROI.cells.delete(name); roiChanged(); } }))));
    byId("btn-roi-submit").disabled = !(ROI.corners.length === 4 && ROI.cells.size > 0);
    if (ROI.tool) ROI.tool.draw();
  }

  function drawRoi(g, tool) {
    const dpr = tool.dpr;
    if (ROI.corners.length) {
      g.beginPath();
      ROI.corners.forEach(([u, v], i) => { const [x, y] = tool.toCanvas(u, v); if (i) g.lineTo(x, y); else g.moveTo(x, y); });
      if (ROI.corners.length === 4) g.closePath();
      g.strokeStyle = "#eda100";
      g.lineWidth = 2 * dpr;
      g.stroke();
      ROI.corners.forEach(([u, v], i) => marker(g, tool, u, v, CORNER_NAMES[i], "#eda100"));
    }
    for (const [name, [x1, y1, x2, y2]] of ROI.cells) {
      const [a, b] = tool.toCanvas(x1, y1), [c, d] = tool.toCanvas(x2, y2);
      g.strokeStyle = "#3987e5";
      g.lineWidth = 1.5 * dpr;
      g.strokeRect(a, b, c - a, d - b);
      g.font = `600 ${10 * dpr}px ${FONT}`;
      g.textAlign = "left";
      g.textBaseline = "bottom";
      g.fillStyle = "#ffffff";
      g.strokeStyle = "rgba(0,0,0,0.75)";
      g.lineWidth = 3 * dpr;
      g.strokeText(name, a, b - 2 * dpr);
      g.fillText(name, a, b - 2 * dpr);
    }
  }

  function undoRoi() {
    const last = ROI.history.pop();
    if (!last) return;
    if (last[0] === "corner") {
      ROI.corners.pop();
      setRoiMode("corners");
    } else {
      ROI.cells.delete(last[1]);
    }
    roiChanged();
  }

  async function submitRoi() {
    const body = { corners: ROI.corners, cells: Object.fromEntries(ROI.cells), space: "image" };
    if (ROI.tool && ROI.tool.size) body.image_size = ROI.tool.size;
    const out = byId("roi-result");
    try {
      await send("api/scoreboard_roi", body, "Scoreboard ROI saved");
      out.replaceChildren(h("div", { class: "status-good", text: `✓ Saved ${ROI.cells.size} cell${ROI.cells.size === 1 ? "" : "s"}` }));
    } catch (err) {
      out.replaceChildren(h("div", { class: "status-bad", text: err.message || String(err) }));
    }
  }

  // =========================================================================== tabs, settings, wiring

  function selectTab(tab, focus) {
    const tabs = [...document.querySelectorAll('[role="tab"]')];
    for (const t of tabs) {
      const on = t === tab;
      t.setAttribute("aria-selected", String(on));
      t.tabIndex = on ? 0 : -1;
      byId(t.getAttribute("aria-controls")).hidden = !on;
    }
    if (focus) tab.focus();
    store.set("localStorage", "cv-tab", tab.id);
    if (tab.id === "tab-live") courtResize();
    if (tab.id === "tab-calibration" && CAL.tool) CAL.tool.resize();
    if (tab.id === "tab-roi" && ROI.tool) ROI.tool.resize();
  }

  function setupTabs() {
    const tabs = [...document.querySelectorAll('[role="tab"]')];
    for (const t of tabs) t.addEventListener("click", () => selectTab(t, false));
    byId("tab-live").parentElement.addEventListener("keydown", (ev) => {
      const i = tabs.indexOf(document.activeElement);
      if (i < 0) return;
      const next = { ArrowRight: i + 1, ArrowLeft: i - 1, Home: 0, End: tabs.length - 1 }[ev.key];
      if (next === undefined) return;
      ev.preventDefault();
      selectTab(tabs[(next + tabs.length) % tabs.length], true);
    });
    const saved = byId(store.get("localStorage", "cv-tab") || "");
    if (saved && saved.getAttribute("role") === "tab") selectTab(saved, false);
  }

  function openSettings() {
    byId("set-token").value = getToken();
    byId("set-theme").value = store.get("localStorage", "cv-theme") || "auto";
    const dlg = byId("settings-dialog");
    if (!dlg.open) dlg.showModal();
  }

  function saveSettings(ev) {
    ev.preventDefault();
    store.set("sessionStorage", "cv-token", byId("set-token").value.trim());
    const theme = byId("set-theme").value;
    store.set("localStorage", "cv-theme", theme === "auto" ? null : theme);
    applyTheme(theme);
    byId("settings-dialog").close();
    WS.retry = 0;
    connect();
  }

  function takeTokenFromHash() {
    const m = /(?:^#|&)token=([^&]+)/.exec(location.hash);
    if (!m) return;
    store.set("sessionStorage", "cv-token", decodeURIComponent(m[1]));
    history.replaceState(null, "", location.pathname + location.search);
  }

  function wire() {
    byId("btn-settings").addEventListener("click", openSettings);
    byId("settings-form").addEventListener("submit", saveSettings);
    byId("btn-settings-close").addEventListener("click", () => byId("settings-dialog").close());

    byId("feed-filter").addEventListener("change", () => mark("feed"));
    byId("btn-add-event").addEventListener("click", () => openEventDialog(null));
    byId("btn-op-add-event").addEventListener("click", () => openEventDialog(null));
    byId("event-form").addEventListener("submit", submitEvent);
    byId("ev-player").addEventListener("change", () => {
      const team = teamOfPlayer(byId("ev-player").value); // Event.team is the actor's team
      if (team) byId("ev-team").value = team;
    });
    byId("btn-ev-cancel").addEventListener("click", () => byId("event-dialog").close());

    for (const f of ["all", "away", "home"]) {
      byId(`shot-filter-${f}`).addEventListener("click", () => {
        S.shotFilter = f;
        for (const g of ["all", "away", "home"]) byId(`shot-filter-${g}`).setAttribute("aria-pressed", String(g === f));
        mark("shots");
      });
    }
    const shotSvg = byId("shot-svg");
    shotSvg.addEventListener("pointermove", showShotTip);
    shotSvg.addEventListener("pointerleave", () => { byId("shot-tooltip").hidden = true; });

    byId("form-score").addEventListener("submit", submitScore);
    byId("btn-score-prefill").addEventListener("click", prefillScore);

    for (const team of ["away", "home"]) {
      byId(`sub-team-${team}`).addEventListener("click", () => {
        SUB.team = team;
        SUB.mode = lineups[team].length ? "sub" : "five";
        clearSub();
      });
    }
    byId("btn-lineup-set").addEventListener("click", () => {
      SUB.mode = SUB.mode === "five" ? (lineups[SUB.team].length ? "sub" : "five") : "five";
      SUB.five = new Set(SUB.mode === "five" ? lineups[SUB.team] : []);
      SUB.out = null;
      SUB.in = null;
      mark("operator");
    });
    byId("btn-sub-clear").addEventListener("click", clearSub);
    byId("btn-sub-confirm").addEventListener("click", confirmSub);

    byId("btn-pause").addEventListener("click", () => control("pause"));
    byId("btn-resume").addEventListener("click", () => control("resume"));
    byId("btn-mute").addEventListener("click", () => control(S.muted ? "unmute" : "mute"));
    byId("btn-deadball").addEventListener("click", () => control("dead_ball"));
    for (const b of byId("speed-group").querySelectorAll("[data-speed]")) {
      b.addEventListener("click", () => control("speed", Number(b.dataset.speed)));
    }
    document.addEventListener("keydown", onHotkey);

    CAL.tool = new FrameTool("cal", { onPoint: addCalPoint, onDraw: drawCalibration });
    byId("btn-cal-load").addEventListener("click", () => CAL.tool.load());
    byId("btn-cal-undo").addEventListener("click", () => { CAL.points.pop(); calChanged(); });
    byId("btn-cal-clear").addEventListener("click", () => { CAL.points = []; byId("cal-result").replaceChildren(); calChanged(); });
    byId("btn-cal-submit").addEventListener("click", submitCalibration);
    byId("cal-landmark").addEventListener("change", renderCalibrationReference);
    byId("cal-preset").addEventListener("change", () => {
      CAL.points = []; byId("cal-landmark").value = landmarkNames()[0] || "";
      byId("cal-result").replaceChildren(); calChanged();
    });

    ROI.tool = new FrameTool("roi", { onPoint: roiPoint, onRect: addCell, onDraw: drawRoi, dragMode: () => ROI.mode === "cells" });
    byId("btn-roi-load").addEventListener("click", () => ROI.tool.load());
    byId("roi-mode-corners").addEventListener("click", () => setRoiMode("corners"));
    byId("roi-mode-cells").addEventListener("click", () => setRoiMode("cells"));
    byId("btn-roi-undo").addEventListener("click", undoRoi);
    byId("btn-roi-clear").addEventListener("click", () => {
      ROI.corners = [];
      ROI.cells.clear();
      ROI.history = [];
      setRoiMode("corners");
      byId("roi-result").replaceChildren();
      roiChanged();
    });
    byId("btn-roi-submit").addEventListener("click", submitRoi);
    byId("btn-roi-add-manual").addEventListener("click", () => {
      const vals = ["roi-mx1", "roi-my1", "roi-mx2", "roi-my2"].map((id) => Number(byId(id).value));
      const raw = ["roi-mx1", "roi-my1", "roi-mx2", "roi-my2"].map((id) => byId(id).value);
      if (raw.some((v) => v === "") || !vals.every(isNum) || vals[2] <= vals[0] || vals[3] <= vals[1]) {
        return toast("Enter x1 < x2 and y1 < y2 in pixels", "error");
      }
      addCell(vals);
    });

    const media = window.matchMedia("(prefers-color-scheme: dark)");
    if (media.addEventListener) media.addEventListener("change", themeChanged);
  }

  function staleCheck() {
    const overlay = byId("court-overlay");
    if (!S.frame) return;
    const age = (performance.now() - S.frameAt) / 1000;
    if (age > 5) {
      overlay.hidden = false;
      setText(overlay, S.finished ? "Analysis complete" : S.paused ? "Paused" : `No tracking data for ${Math.round(age)} s`);
    } else {
      overlay.hidden = true;
    }
  }

  function boot() {
    takeTokenFromHash();
    const theme = store.get("localStorage", "cv-theme");
    if (theme === "light" || theme === "dark") document.documentElement.dataset.theme = theme;
    readColors();
    courtView.canvas = byId("court-canvas");
    courtView.ctx = courtView.canvas.getContext("2d");
    S.geo = courtGeometry({});
    wire();
    setupTabs();
    if (window.ResizeObserver) new ResizeObserver(courtResize).observe(byId("court-wrap"));
    courtResize();
    setRoiMode("corners");
    mark("teams", "court", "shots", "box", "compare", "insights", "feed", "decisions", "operator", "calibration", "status", "scorebar");
    connect();
    setInterval(heartbeat, 15000);
    setInterval(staleCheck, 1000);
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot);
  else boot();
})();
