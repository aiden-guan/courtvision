/* The open video's analysis on the feed: who has the ball and each decided shot, from its game's analysed run
   (tools/nba_analyze.py <game> timeline, found by courtvision/nba/results.py), drawn for the time of the frame on
   screen, or for the time playing in Watch and Replay (static/playback.js), where the band under the feed seeks.
   When the game has a play-by-play it is shown beside them as the reference. */
(() => {
  "use strict";
  const $ = id => document.getElementById(id);
  const SHOW_S = 3.5, PLAY_S = 4;  // how long a decided play, and a play-by-play line, stay on the feed
  const CARD_KINDS = new Set(["shot_made", "shot_missed", "rebound", "steal", "turnover"]);
  let A = null, loaded = "", fetching = "", failed = false, frameT = NaN, cardsKey = "";
  let hinted = "";
  let segT = [], backT = [], evT = [], madeT = [], points = [], refT = [], playT = [];
  const token = () => { try { return sessionStorage.getItem("cv-token") || ""; } catch { return ""; } };
  const headers = () => (token() ? { Authorization: `Bearer ${token()}` } : {});
  const playing = () => (window.CVPlayback?.mode || "live") !== "live";  // Watch or Replay: the <video> sets the time
  const duration = () => window.CVStudio?.workspace?.active?.duration_s;
  const text = (id, value) => { if ($(id).textContent !== String(value)) $(id).textContent = value; };
  const pct = x => `${Math.round(x * 100)}%`;
  const step = () => A.dt_s || 0.25;
  const abbr = side => A?.teams?.[side]?.abbr || (side === "home" ? "Home" : "Away");
  // the index of the last of the sorted times at or before t (-1 when none is)
  function upTo(times, t) {
    let lo = 0, hi = times.length - 1, k = -1;
    while (lo <= hi) { const mid = (lo + hi) >> 1; if (times[mid] <= t) { k = mid; lo = mid + 1; } else hi = mid - 1; }
    return k;
  }
  // a segment runs to the next grid step, so the gap between two belongs to the first
  function teamAt(t) { const k = upTo(segT, t); return k >= 0 && t < A.segments[k][1] + step() ? A.segments[k][2] : null; }
  function bringingUp(t) { const k = upTo(backT, t); return k >= 0 && t < A.back_court[k][1] + step(); }
  function refAt(t) { const k = upTo(refT, t); return k >= 0 ? A.reference.marks[k][1] : undefined; }
  function playAt(t) { const k = upTo(playT, t); return k >= 0 && t - playT[k] <= PLAY_S ? A.reference.plays[k][1] : null; }
  function scoreAt(t) { const k = upTo(madeT, t); return k >= 0 ? points[k] : { home: 0, away: 0 }; }

  function index() {
    segT = A.segments.map(s => s[0]);
    backT = A.back_court.map(s => s[0]);
    evT = A.events.map(e => e.t);
    madeT = []; points = [];
    const run = { home: 0, away: 0 };
    for (const e of A.events) if (e.kind === "shot_made" && e.team in run) {
      run[e.team] += e.value; madeT.push(e.t); points.push({ ...run });
    }
    refT = A.reference ? A.reference.marks.map(m => m[0]) : [];
    playT = A.reference ? A.reference.plays.map(p => p[0]) : [];
  }

  function card(e) {
    const node = document.createElement("div"), head = document.createElement("strong");
    node.className = "poss-card"; node.dataset.kind = e.kind; node.dataset.team = e.team || "";
    const shot = e.kind === "shot_made" || e.kind === "shot_missed";
    head.textContent = shot ? `${abbr(e.team)} ${e.free_throw ? "free throw" : `${e.value}-pointer`}` :
      e.kind === "rebound" ? `${abbr(e.team)} ${e.offensive ? "offensive" : "defensive"} rebound` : `${abbr(e.team)} ${e.kind}`;
    node.append(head);
    if (shot) {
      const result = document.createElement("span");
      result.className = "poss-result"; result.textContent = e.kind === "shot_made" ? "Made" : "Missed";
      head.append(result);
      const notes = [e.found_at_rim ? "found at the rim: the engine had missed it" : "",
        Number.isFinite(e.p_made) ? `P(made) ${e.p_made.toFixed(2)}` : ""].filter(Boolean);
      if (notes.length) { const note = document.createElement("small"); note.textContent = notes.join(" · "); node.append(note); }
    }
    return node;
  }

  function render(t) {
    frameT = t;
    const on = !!A && $("poss-toggle").checked && Number.isFinite(t);
    $("poss-hud").hidden = $("poss-feed").hidden = !on;
    if (!on) return;
    const team = teamAt(t);
    $("poss-team").dataset.team = team || "";
    text("poss-team", team ? `${abbr(team)} ball` : "No team");
    text("poss-phase", team && bringingUp(t) ? "bringing it up" : "");
    const s = scoreAt(t);
    text("poss-score", `${abbr("away")} ${s.away} – ${s.home} ${abbr("home")}`);
    $("poss-ref-line").hidden = !A.reference;
    if (A.reference) {
      const real = refAt(t), agree = real != null && real === team;
      text("poss-ref", real === undefined ? "—" : real === null ? "loose ball" : `${abbr(real)} ball · ${agree ? "agrees" : "differs"}`);
      $("poss-ref").dataset.tone = real == null ? "" : agree ? "good" : "bad";
    }
    const play = A.reference ? playAt(t) : null;
    $("poss-play").hidden = !play;
    if (play) text("poss-play", play);
    const recent = A.events.slice(upTo(evT, t - SHOW_S) + 1, upTo(evT, t) + 1).filter(e => CARD_KINDS.has(e.kind)).slice(-3);
    const key = recent.map(e => `${e.t}:${e.kind}`).join("|");
    if (key !== cardsKey) { cardsKey = key; $("poss-feed").replaceChildren(...recent.map(card)); }
  }

  // the whole game under the transport: who had the ball (top: the analysis, below: the play-by-play) and each
  // decided shot (a tall tick: made, a short one: missed)
  function ribbon() {
    const span = duration(), svg = $("poss-ribbon-svg");
    $("poss-ribbon").hidden = !(A && span > 0);
    svg.replaceChildren();
    if (!A || !(span > 0)) return;
    const X = t => Math.min(1000, Math.max(0, t / span * 1000)).toFixed(2);
    const d = { "rib-home": "", "rib-away": "", "ref-home": "", "ref-away": "", "tick-home": "", "tick-away": "", "tick-miss": "" };
    for (const [a, b, team] of A.segments) d[`rib-${team}`] += `M${X(a)} 6H${X(b + step())}V10H${X(a)}Z`;
    if (A.reference) {
      const marks = A.reference.marks, end = A.segments.length ? A.segments[A.segments.length - 1][1] : 0;
      marks.forEach(([a, team], k) => { if (team) d[`ref-${team}`] += `M${X(a)} 11H${X(k + 1 < marks.length ? marks[k + 1][0] : end)}V13H${X(a)}Z`; });
    }
    for (const e of A.events) {
      if (e.kind === "shot_made") d[`tick-${e.team}`] += `M${X(e.t)} 0V5`;
      else if (e.kind === "shot_missed") d["tick-miss"] += `M${X(e.t)} 2.5V5`;
    }
    const ns = "http://www.w3.org/2000/svg";
    for (const [cls, path] of Object.entries(d)) {
      if (!path) continue;
      const el = document.createElementNS(ns, "path");
      el.setAttribute("d", path); el.setAttribute("class", cls.startsWith("tick") ? `tick ${cls}` : cls);
      svg.append(el);
    }
  }

  function status() {
    const ws = window.CVStudio?.workspace, video = ws?.active && !ws.active.demo;
    const cell = $("vision-poss");
    const watching = window.CVPlayback?.mode === "watch";
    $("poss-switch").hidden = $("poss-watch").hidden = !A;
    text("poss-watch", watching ? "Live view" : "Watch");
    $("poss-watch").title = watching ? "Back to the live analysis (W)" : "Play the recording at full speed with the analysis on it (W)";
    if (A) {
      const m = A.measures || {}, short = [], long = [];
      if (m.possession != null) { short.push(`ball ${pct(m.possession)}`); long.push(`the team with the ball ${pct(m.possession)} of settled play`); }
      if (m.made != null) { short.push(`made/missed ${pct(m.made)}`); long.push(`made or missed ${pct(m.made)} of real shots`); }
      text("vision-poss", playing() ? "On the recording" : $("poss-toggle").checked ? "On the feed" : "Hidden");
      cell.dataset.tone = "good";
      $("poss-cell").title = `${A.game}, run ${A.tag}${A.jev ? " (with Jev)" : ""}. ` +
        (long.length ? `Right on held-out halves against the play-by-play: ${long.join(", ")}.` : "No play-by-play to score it against.");
      text("poss-measure", short.length ? `Held out: ${short.join(" · ")}` : "Not scored: no play-by-play");
    } else {
      text("vision-poss", fetching ? "Loading…" : failed ? "Could not load" : video ? "Not analysed" : "—");
      cell.dataset.tone = failed ? "bad" : fetching ? "busy" : "idle";
      $("poss-cell").title = video && !fetching ? "No game folder under runs/nba analyses this video. Add it as a game " +
        "(tools/nba_new_game.py) and run the analysis through nba_analyze.py <game> timeline (docs/NBA.md)." : "";
    }
  }

  async function load(want) {
    fetching = want; failed = false; status();
    try {
      const response = await fetch("api/studio/analysis", { headers: headers(), cache: "no-store" });
      if (!response.ok) throw new Error(`The analysis did not load (${response.status}).`);
      const doc = await response.json();
      if (fetching !== want) return;
      A = doc; loaded = want; cardsKey = ""; index(); ribbon(); render(frameT);
      const id = window.CVStudio?.workspace?.active?.id;
      if (id && hinted !== id && !playing()) { hinted = id; window.CV?.toast?.("This game has a possession analysis. Press W to watch it at full speed."); }
    } catch {
      if (fetching === want) { A = null; loaded = want; failed = true; ribbon(); render(NaN); }
    } finally { if (fetching === want) fetching = ""; status(); }
  }

  function sync() {
    const run = window.CVStudio?.workspace?.analysis;
    const want = run ? `${run.game}/${run.tag}/${run.version}` : "";
    if (!want) {
      if (A || loaded) { A = null; loaded = ""; failed = false; ribbon(); render(NaN); }
    } else if (want !== loaded && want !== fetching) load(want);
    status();
  }

  function toggle() {
    if (!A) return;
    $("poss-toggle").checked = !$("poss-toggle").checked;
    render(frameT); status();
  }

  function head(t) {
    const span = duration();
    $("poss-head").hidden = !(span > 0);
    if (span > 0) $("poss-head").style.left = `${Math.min(100, Math.max(0, t / span * 100))}%`;
  }

  window.CVPossession = { get data() { return A; }, toggle };
  $("poss-toggle").addEventListener("change", () => { render(frameT); status(); });
  $("poss-watch").addEventListener("click", () => window.CVPlayback?.toggle("watch"));
  $("poss-ribbon").addEventListener("click", ev => {  // in Watch and Replay the band seeks
    const span = duration(), box = $("poss-ribbon").getBoundingClientRect();
    if (!playing() || !(span > 0)) return;
    window.CVPlayback.seek(Math.min(span, Math.max(0, (ev.clientX - box.left) / box.width * span)));
  });
  window.addEventListener("cv-studio", sync);
  window.addEventListener("cv-frame", ev => { if (!playing()) render(ev.detail.t); });
  window.addEventListener("cv-playback", ev => { render(ev.detail.t); head(ev.detail.t); });
  window.addEventListener("cv-playback-mode", ev => {
    $("poss-head").hidden = ev.detail.mode === "live";
    if (ev.detail.mode === "live") render(window.CVStudio?.processedT ?? NaN);
    status();
  });
  window.addEventListener("courtvision-message", ev => { if (ev.detail.type === "reset" && !playing()) render(NaN); });
})();
