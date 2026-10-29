/* CourtVision generative UI: answers, briefings and board cards composed from typed blocks.
 *
 * A view spec is {title, summary, tone, icon, blocks}. Data blocks name a live source (a stat key, a
 * team, a player, a filter) instead of carrying numbers, so a pinned card rebuilds from the newest game
 * state, and a spec composed by Claude cannot put a number on screen that the game data does not hold.
 * Specs come from three composers: the briefing (what matters now, from workspace and game state), the
 * on-device planner (questions matched to intents; works offline) and Claude through POST api/ask when
 * the server has a key. Every spec is cleaned against the block schema before rendering, and all text
 * is set with textContent.
 */
(() => {
  "use strict";

  const cv = () => window.CV;
  const isNum = (v) => typeof v === "number" && Number.isFinite(v);
  const str = (v, n) => String(v == null ? "" : v).slice(0, n);
  const SVG = "http://www.w3.org/2000/svg";

  function el(tag, props, ...kids) {
    const e = document.createElement(tag);
    for (const [k, v] of Object.entries(props || {})) {
      if (v == null || v === false) continue;
      if (k === "class") e.className = v;
      else if (k === "text") e.textContent = v;
      else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
      else e.setAttribute(k, v === true ? "" : String(v));
    }
    for (const kid of kids.flat()) if (kid != null && kid !== false) e.append(kid instanceof Node ? kid : document.createTextNode(String(kid)));
    return e;
  }
  function sv(tag, attrs) {
    const e = document.createElementNS(SVG, tag);
    for (const [k, v] of Object.entries(attrs || {})) if (v != null) e.setAttribute(k, String(v));
    return e;
  }

  // 16 px icons, stroke only
  const ICONS = {
    spark: "M8 1.8l1.35 3.5 3.5 1.35-3.5 1.35L8 11.5 6.65 8 3.15 6.65 6.65 5.3zM12.6 10.6l.55 1.4 1.4.55-1.4.55-.55 1.4-.55-1.4-1.4-.55 1.4-.55z",
    alert: "M8 2.2 14.3 13H1.7zM8 6.4v3.2M8 11.3v.01",
    check: "M3 8.4l3.1 3.1L13 4.6",
    info: "M8 2a6 6 0 1 0 0 12A6 6 0 0 0 8 2zM8 7.2v3.6M8 5.1v.01",
    court: "M1.5 3.5h13v9h-13zM8 3.5v9M1.5 6h2.2v4H1.5M14.5 6h-2.2v4h2.2",
    ball: "M8 2a6 6 0 1 0 0 12A6 6 0 0 0 8 2zM2 8h12M8 2v12M3.9 3.8c2.2 2.3 2.2 6.1 0 8.4M12.1 3.8c-2.2 2.3-2.2 6.1 0 8.4",
    people: "M6 2.8a2.2 2.2 0 1 0 0 4.4 2.2 2.2 0 0 0 0-4.4zM2 13.5c.4-2.4 2-3.7 4-3.7s3.6 1.3 4 3.7M10.6 3a2.1 2.1 0 0 1 0 4.1M12 9.9c1.2.5 1.9 1.7 2.1 3.6",
    chart: "M3 13.5v-4M6.3 13.5v-7M9.7 13.5V8M13 13.5v-11",
    flame: "M8 14c2.6 0 4.3-1.8 4.3-4.2 0-2.6-2-4.2-2.6-6.8-1 1-1.6 2.2-1.7 3.4C7.2 5.7 6.6 4.8 6.4 3.6 4.7 5 3.7 7.3 3.7 9.8 3.7 12.2 5.4 14 8 14z",
    review: "M2.5 3.5h11v7h-6l-3 2.5v-2.5h-2zM5.5 6.5h5M5.5 8.5h3",
    clock: "M8 2a6 6 0 1 0 0 12A6 6 0 0 0 8 2zM8 4.8V8l2.2 1.4",
    lab: "M6.2 2h3.6M6.7 2v4.2L3.1 12.4A1.1 1.1 0 0 0 4.1 14h7.8a1.1 1.1 0 0 0 1-1.6L9.3 6.2V2M4.6 10h6.8",
    film: "M2 3.5h12v9H2zM2 6h12M2 10h12M5 3.5v9M11 3.5v9",
    trophy: "M5 2.5h6v3.2a3 3 0 0 1-6 0zM5 3.5H2.8c0 2 1 3 2.4 3.2M11 3.5h2.2c0 2-1 3-2.4 3.2M8 8.7v2.8M5.5 13.5h5",
    bolt: "M9.2 1.8 4.4 9h3.4l-1 5.2L11.6 7H8.2z",
    pin: "M10.2 2.2l3.6 3.6-2.2.9-2.4 2.4.4 2.8-1.4 1.4L5.7 10.8 2.6 13.9M5.7 10.8 2.6 7.7 4 6.3l2.8.4 2.4-2.4z",
    close: "M4 4l8 8M12 4l-8 8",
    copy: "M5.5 5.5h7v8h-7zM3.5 10.5v-8h7",
    arrow: "M3 8h10M9 4l4 4-4 4",
  };
  function icon(name) {
    const s = sv("svg", { viewBox: "0 0 16 16", fill: "none", stroke: "currentColor", "stroke-width": 1.5, "stroke-linecap": "round", "stroke-linejoin": "round", "aria-hidden": "true" });
    s.append(sv("path", { d: ICONS[name] || ICONS.spark }));
    return s;
  }

  // ------------------------------------------------------------------ data access

  const S = () => cv().S;
  const stats = () => S().stats || null;
  const teamName = (t) => (t === "home" || t === "away" ? S().teams[t] : "");
  const teamLine = (t) => ((stats() && stats().teams) || {})[t] || {};
  function playerLines() {
    const players = (stats() && stats().players) || {};
    return Object.entries(players).filter(([, l]) => l).map(([id, line]) => ({ id, line, team: line.team || cv().teamOfPlayer(id) }));
  }
  const pct = (v, d = 1) => cv().fmtPct(v, d);
  const int = (v) => cv().fmtInt(v);
  const made = (m, a) => cv().fmtMade(m, a);
  const ratio = (m, a) => (isNum(a) && a > 0 ? (m || 0) / a : null);

  // stat key -> label, team value, player value, format, sub-line; lowerBetter flips "leading"
  const STATS = {
    pts: { label: "Points", team: (t) => t.pts, player: (l) => l.pts, fmt: int },
    fg_pct: { label: "FG%", team: (t) => ratio(t.fgm, t.fga), player: (l) => ratio(l.fgm, l.fga), fmt: (v) => pct(v), sub: (t) => made(t.fgm, t.fga) },
    fg3_pct: { label: "3P%", team: (t) => ratio(t.fg3m, t.fg3a), player: (l) => ratio(l.fg3m, l.fg3a), fmt: (v) => pct(v), sub: (t) => made(t.fg3m, t.fg3a) },
    fg3m: { label: "Threes made", team: (t) => t.fg3m, player: (l) => l.fg3m, fmt: int },
    ft_pct: { label: "FT%", team: (t) => ratio(t.ftm, t.fta), player: (l) => ratio(l.ftm, l.fta), fmt: (v) => pct(v), sub: (t) => made(t.ftm, t.fta) },
    efg: { label: "eFG%", team: (t) => cv().efgOf(t), fmt: (v) => pct(v) },
    ts: { label: "TS%", team: (t) => cv().tsOf(t), fmt: (v) => pct(v) },
    reb: { label: "Rebounds", team: (t) => cv().rebOf(t), player: (l) => cv().rebOf(l), fmt: int },
    oreb: { label: "Offensive rebounds", team: (t) => t.oreb, player: (l) => l.oreb, fmt: int },
    ast: { label: "Assists", team: (t) => t.ast, player: (l) => l.ast, fmt: int },
    stl: { label: "Steals", team: (t) => t.stl, player: (l) => l.stl, fmt: int },
    blk: { label: "Blocks", team: (t) => t.blk, player: (l) => l.blk, fmt: int },
    tov: { label: "Turnovers", team: (t) => t.tov, player: (l) => l.tov, fmt: int, lowerBetter: true },
    pf: { label: "Fouls", team: (t) => t.pf, player: (l) => l.pf, fmt: int, lowerBetter: true },
    poss: { label: "Possessions", team: (t) => cv().possOf(t), fmt: int },
    ppp: { label: "Points per possession", team: (t) => cv().pppOf(t), fmt: (v) => cv().fmtNum(v, 2) },
    pace: { label: "Pace", team: (t) => t.pace, fmt: (v) => cv().fmtNum(v, 1) },
    xpts: { label: "Expected points", team: (t) => t.xpts, player: (l) => l.xpts, fmt: (v) => cv().fmtNum(v, 1) },
    avg_xfg: { label: "Average xFG", team: (t) => t.avg_xfg, fmt: (v) => pct(v, 0) },
    plus_minus: { label: "+/-", player: (l) => (cv().num(l.seconds, 0) > 0 ? l.plus_minus : null), fmt: (v) => cv().fmtSigned(v) },
  };
  const GAME = {
    score_home: { label: () => teamName("home"), value: () => int(((stats() || {}).score || {}).home), team: "home" },
    score_away: { label: () => teamName("away"), value: () => int(((stats() || {}).score || {}).away), team: "away" },
    period: { label: () => "Period", value: () => cv().periodLabel((stats() || {}).period) },
    game_clock: { label: () => "Game clock", value: () => cv().fmtClock((stats() || {}).game_clock) },
    margin: { label: () => "Margin", value: () => { const sc = (stats() || {}).score || {}; return isNum(sc.home) && isNum(sc.away) ? `${sc.home === sc.away ? "Tied" : `${teamName(sc.home > sc.away ? "home" : "away")} +${Math.abs(sc.home - sc.away)}`}` : "–"; } },
    events: { label: () => "Plays logged", value: () => String(S().events.size) },
    review: { label: () => "Need review", value: () => String(reviewQueue().length) },
  };

  const reviewQueue = () => [...S().events.values()].filter((e) => cv().feedFilter(e, "review") && e.status !== "retracted").sort((a, b) => (b.t || 0) - (a.t || 0));
  const playerTeam = (id) => { const r = S().roster.get(id); return (r && r.team) || (playerLines().find((p) => p.id === id) || {}).team || cv().teamOfPlayer(id); };

  // ------------------------------------------------------------------ schema

  const TONES = new Set(["neutral", "good", "warn", "bad", "ember", "info", "live"]);
  const TEAMS = new Set(["home", "away"]);
  const FILTERS = new Set(["key", "all", "review", "scoring"]);
  // commands a card may offer as buttons (static/shell.js runs them)
  const COMMAND = /^(nav:(live|stats|board|operator|calibration|roi|lab)|footage:(load|demo|export)|court:map|plays:(review|add)|playback:(pause|resume)|lab:(refresh|run|copy-next)|player:[\w.:-]{1,64}|ask:[^\n]{1,200})$/;
  const KINDS = new Set(["text", "note", "metrics", "duel", "leaders", "spark", "shots", "plays", "player", "kv", "steps", "actions", "code", "tracking", "decisions", "insights", "narration", "lab"]);

  function cleanBlock(b) {
    if (!b || typeof b !== "object" || !KINDS.has(b.type)) return null;
    const team = TEAMS.has(b.team) ? b.team : null;
    const limit = Math.max(1, Math.min(12, Math.round(Number(b.limit) || 5)));
    switch (b.type) {
      case "text": return b.text ? { type: "text", text: str(b.text, 600) } : null;
      case "note": return b.text ? { type: "note", tone: TONES.has(b.tone) ? b.tone : "neutral", text: str(b.text, 400) } : null;
      case "metrics": {
        const items = (Array.isArray(b.items) ? b.items : []).slice(0, 6).map((it) => {
          if (!it || typeof it !== "object") return null;
          if (it.stat && STATS[it.stat] && STATS[it.stat].team && TEAMS.has(it.team)) return { stat: it.stat, team: it.team };
          if (it.stat && STATS[it.stat] && STATS[it.stat].player && typeof it.player === "string") return { stat: it.stat, player: str(it.player, 64) };
          if (it.game && GAME[it.game]) return { game: it.game };
          if (it.label && it.value != null) return { label: str(it.label, 40), value: str(it.value, 24), sub: it.sub ? str(it.sub, 40) : null };
          return null;
        }).filter(Boolean);
        return items.length ? { type: "metrics", items } : null;
      }
      case "duel": {
        const keys = (Array.isArray(b.stats) ? b.stats : []).filter((k) => STATS[k] && STATS[k].team).slice(0, 10);
        return { type: "duel", stats: keys.length ? keys : ["pts", "fg_pct", "fg3_pct", "reb", "ast", "tov"] };
      }
      case "leaders": return { type: "leaders", stat: STATS[b.stat] && STATS[b.stat].player ? b.stat : "pts", team, limit };
      case "spark": return { type: "spark", series: "margin" };
      case "shots": return { type: "shots", team, player: typeof b.player === "string" ? str(b.player, 64) : null };
      case "plays": return { type: "plays", filter: FILTERS.has(b.filter) ? b.filter : "key", team, player: typeof b.player === "string" ? str(b.player, 64) : null, limit };
      case "player": return typeof b.id === "string" ? { type: "player", id: str(b.id, 64) } : null;
      case "kv": {
        const items = (Array.isArray(b.items) ? b.items : []).slice(0, 10).filter((it) => it && it.label != null && it.value != null).map((it) => ({ label: str(it.label, 40), value: str(it.value, 60) }));
        return items.length ? { type: "kv", items } : null;
      }
      case "steps": {
        const items = (Array.isArray(b.items) ? b.items : []).slice(0, 6).filter((it) => it && it.title).map((it) => ({ title: str(it.title, 80), text: it.text ? str(it.text, 200) : "", state: ["done", "now", "todo"].includes(it.state) ? it.state : "todo" }));
        return items.length ? { type: "steps", items } : null;
      }
      case "actions": {
        const items = (Array.isArray(b.items) ? b.items : []).slice(0, 4).filter((it) => it && it.label && typeof it.command === "string" && COMMAND.test(it.command))
          .map((it) => ({ label: str(it.label, 32), command: str(it.command, 120), primary: !!it.primary }));
        return items.length ? { type: "actions", items } : null;
      }
      case "code": return b.text ? { type: "code", text: str(b.text, 1200) } : null;
      case "tracking": case "lab": case "spark2": return { type: b.type };
      case "decisions": case "insights": case "narration": return { type: b.type, limit };
      default: return null;
    }
  }

  function clean(spec) {
    if (!spec || typeof spec !== "object") return null;
    const blocks = (Array.isArray(spec.blocks) ? spec.blocks : []).slice(0, 8).map(cleanBlock).filter(Boolean);
    return {
      title: str(spec.title || "Answer", 90), summary: spec.summary ? str(spec.summary, 400) : "",
      icon: ICONS[spec.icon] ? spec.icon : "spark", tone: TONES.has(spec.tone) ? spec.tone : "neutral", blocks,
    };
  }

  // ------------------------------------------------------------------ blocks

  function metricItem(it) {
    let label, value, sub = "", team = null;
    if (it.stat && it.team) {
      const def = STATS[it.stat], line = teamLine(it.team);
      label = `${teamName(it.team)} ${def.label}`; value = def.fmt(def.team(line)); sub = def.sub ? def.sub(line) : ""; team = it.team;
    } else if (it.stat && it.player) {
      const def = STATS[it.stat], p = playerLines().find((q) => q.id === it.player);
      label = `${cv().pname(it.player)} ${def.label}`; value = def.fmt(p ? def.player(p.line) : null); team = playerTeam(it.player);
    } else if (it.game) {
      const g = GAME[it.game]; label = g.label(); value = g.value(); team = g.team || null;
    } else { label = it.label; value = it.value; sub = it.sub || ""; }
    return el("div", { class: "metric" }, el("span", { class: "metric-label", text: label }),
      el("span", { class: "metric-value", text: value, "data-team": team }), sub ? el("span", { class: "metric-sub", text: sub }) : null);
  }

  function duel(b) {
    const a = teamLine("away"), h = teamLine("home");
    const rows = b.stats.map((k) => {
      const def = STATS[k], va = def.team(a), vh = def.team(h);
      const max = Math.max(Math.abs(va || 0), Math.abs(vh || 0)) || 1;
      const leadA = isNum(va) && isNum(vh) && (def.lowerBetter ? va < vh : va > vh);
      const leadH = isNum(va) && isNum(vh) && (def.lowerBetter ? vh < va : vh > va);
      const bar = (side, v, lead) => { const d = el("div", { class: `duel-bar ${side}${lead ? "" : " dim"}` }, el("i")); d.firstChild.style.setProperty("--w", isNum(v) ? (Math.abs(v) / max).toFixed(3) : "0"); return d; };
      return el("div", { class: "duel-row" },
        el("span", { class: `v away${leadA ? "" : " dim"}`, text: def.fmt(va) }), bar("away", va, leadA),
        el("span", { class: "k", text: def.label }), bar("home", vh, leadH), el("span", { class: `v${leadH ? "" : " dim"}`, text: def.fmt(vh) }));
    });
    return el("div", { class: "blk-duel" }, el("div", { class: "duel-head" }, el("span", { text: teamName("away") }), el("span", { text: teamName("home") })), rows);
  }

  function leaders(b) {
    const def = STATS[b.stat];
    const rows = playerLines().filter((p) => !b.team || p.team === b.team)
      .map((p) => ({ ...p, v: def.player(p.line) })).filter((p) => isNum(p.v))
      .sort((x, y) => (def.lowerBetter ? x.v - y.v : y.v - x.v)).slice(0, b.limit);
    if (!rows.length) return el("p", { class: "blk-text", text: `No ${def.label.toLowerCase()} recorded${b.team ? ` for ${teamName(b.team)}` : ""} yet.` });
    const max = Math.max(...rows.map((r) => Math.abs(r.v))) || 1;
    return el("div", { class: "blk-bars" }, rows.map((r) => {
      const track = el("span", { class: "bar-track" }, el("i"));
      track.firstChild.style.setProperty("--w", (Math.abs(r.v) / max).toFixed(3));
      const j = cv().pjersey(r.id);
      return el("div", { class: "bar-row", "data-team": r.team },
        el("span", { class: "bar-name", title: cv().pname(r.id) }, j ? el("b", { text: `#${j}` }) : null, cv().pname(r.id)), track, el("span", { class: "bar-val", text: def.fmt(r.v) }));
    }));
  }

  function spark() {
    const series = S().margin;
    const wrap = el("div", { class: "blk-spark" });
    if (series.length < 2) { wrap.append(el("p", { class: "blk-text", text: "The score margin draws here once both teams have scored." })); return wrap; }
    const t0 = series[0][0], t1 = Math.max(series[series.length - 1][0], cv().latestT() || 0, t0 + 1);
    const peak = Math.max(4, ...series.map(([, m]) => Math.abs(m)));
    const X = (t) => ((t - t0) / (t1 - t0)) * 300, Y = (m) => 40 - (m / peak) * 36;
    let d = `M0 40`;
    series.forEach(([t, m], i) => { d += ` L${X(t).toFixed(1)} ${Y(m).toFixed(1)}`; const n = series[i + 1]; if (n) d += ` L${X(n[0]).toFixed(1)} ${Y(m).toFixed(1)}`; });
    d += ` L300 ${Y(series[series.length - 1][1]).toFixed(1)}`;
    const s = sv("svg", { viewBox: "0 0 300 80", preserveAspectRatio: "none", "aria-hidden": "true" });
    const id = `sp${Math.random().toString(36).slice(2, 8)}`;
    const defs = sv("defs");
    const ct = sv("clipPath", { id: `${id}t` }), cb = sv("clipPath", { id: `${id}b` });
    ct.append(sv("rect", { x: 0, y: 0, width: 300, height: 40 })); cb.append(sv("rect", { x: 0, y: 40, width: 300, height: 40 }));
    defs.append(ct, cb);
    s.append(defs, sv("path", { d: `${d} L300 40 Z`, class: "spark-home", "clip-path": `url(#${id}t)` }),
      sv("path", { d: `${d} L300 40 Z`, class: "spark-away", "clip-path": `url(#${id}b)` }),
      sv("line", { x1: 0, x2: 300, y1: 40, y2: 40, class: "spark-zero" }), sv("path", { d, class: "spark-line" }));
    const last = series[series.length - 1][1];
    wrap.append(s, el("div", { class: "spark-legend" }, el("span", { text: `${teamName("home")} above the line` }),
      el("span", { text: last === 0 ? "Tied now" : `${teamName(last > 0 ? "home" : "away")} by ${Math.abs(last)} now` })));
    return wrap;
  }

  function shots(b) {
    const geo = S().geo || cv().courtGeometry({});
    const all = ((stats() && stats().shots) || []).filter((s) => isNum(s.x) && isNum(s.y) && s.shot_type !== "free_throw"
      && (!b.team || s.team === b.team) && (!b.player || s.player_id === b.player));
    const pad = 2;
    const s = sv("svg", { viewBox: `${-pad} ${-pad} ${geo.L + 2 * pad} ${geo.W + 2 * pad}`, role: "img", "aria-label": `${all.length} field-goal attempts` });
    const g = sv("g", { transform: `translate(0 ${geo.W}) scale(1 -1)` });
    const pts = (pl) => pl.map(([x, y]) => `${x.toFixed(2)},${y.toFixed(2)}`).join(" ");
    for (const poly of geo.paint) g.append(sv("polygon", { points: pts(poly), class: "court-svg-paint" }));
    for (const pl of geo.lines) g.append(sv("polyline", { points: pts(pl), class: "court-svg-line" }));
    for (const pl of geo.rims) g.append(sv("polyline", { points: pts(pl), class: "court-svg-rim" }));
    for (const sh of all) {
      const color = cv().xfgColor(sh.xfg);
      if (sh.made === true) g.append(sv("circle", { cx: sh.x, cy: sh.y, r: 0.95, fill: color, stroke: "rgba(0,0,0,.4)", "stroke-width": 0.12 }));
      else g.append(sv("circle", { cx: sh.x, cy: sh.y, r: 0.8, fill: "none", stroke: color, "stroke-width": 0.24 }));
    }
    s.append(g);
    const wrap = el("div", { class: "blk-shots device" }, s);
    const madeN = all.filter((x) => x.made === true).length, decided = all.filter((x) => x.made != null).length;
    return el("div", { class: "blk-spark" }, wrap, el("div", { class: "spark-legend" },
      el("span", { text: all.length ? `FG ${madeN}/${decided} · ${pct(decided ? madeN / decided : null)}` : "No field-goal attempts yet" }),
      el("span", { text: "Filled: made · ring: missed · colour: xFG" })));
  }

  function playsList(b) {
    let evs = [...S().events.values()].filter((e) => e.status !== "retracted");
    if (b.filter === "scoring") evs = evs.filter((e) => e.kind === "shot_made" || e.kind === "score_correction");
    else evs = evs.filter((e) => cv().feedFilter(e, b.filter));
    if (b.team) evs = evs.filter((e) => e.team === b.team);
    if (b.player) evs = evs.filter((e) => e.player_id === b.player || e.other_player_id === b.player);
    evs.sort((x, y) => (y.t || 0) - (x.t || 0));
    evs = evs.slice(0, b.limit);
    if (!evs.length) return el("p", { class: "blk-text", text: b.filter === "review" ? "Nothing needs review right now." : "No matching plays yet." });
    return el("div", { class: "blk-list" }, evs.map((e) => el("div", { class: "blk-item" },
      el("span", { class: "mark", "data-team": e.team, "data-tone": e.attrs && e.attrs.needs_review ? "warn" : null }),
      el("span", { class: "what" }, cv().describe(e), el("small", { text: `${cv().timeLabel(e.t)} · ${e.status || "provisional"}${isNum(e.confidence) && e.confidence < 1 ? ` · ${Math.round(e.confidence * 100)}%` : ""}` })),
      el("button", { type: "button", class: "link-btn", text: "Correct", onclick: () => cv().openEventDialog(S().events.get(e.event_id) || e) }))));
  }

  function playerCard(b) {
    const id = b.id, p = playerLines().find((q) => q.id === id), line = (p && p.line) || {};
    const team = playerTeam(id), r = S().roster.get(id);
    const jersey = cv().pjersey(id);
    const frame = S().frame && (S().frame.players || []).find((q) => q.id === id);
    const card = el("div", { class: "blk-player" },
      el("span", { class: "jersey-big", "data-team": team, text: jersey ? String(jersey) : "?" }),
      el("div", null, el("div", { class: "player-name", text: cv().pname(id) }),
        el("div", { class: "player-team", text: `${team ? teamName(team) : "Team not assigned"}${r && r.position ? ` · ${r.position}` : ""}${frame && isNum(frame.speed) ? ` · moving ${frame.speed.toFixed(1)} ft/s` : ""}` })));
    const m = el("div", { class: "blk-metrics" }, [
      ["Points", int(line.pts)], ["Rebounds", int(cv().rebOf(line))], ["Assists", int(line.ast)],
      ["FG", made(line.fgm, line.fga)], ["3PT", made(line.fg3m, line.fg3a)], ["xPTS", cv().fmtNum(line.xpts, 1)],
    ].map(([k, v]) => metricItem({ label: k, value: v })));
    return el("div", { class: "gen-blocks" }, card, m);
  }

  function kv(items) {
    return el("dl", { class: "blk-kv" }, items.flatMap((it) => [el("dt", { text: it.label }), el("dd", { text: it.value, title: it.value })]));
  }

  function tracking() {
    const st = window.CVStudio || {}, v = st.vision || {}, status = st.status || {}, act = (st.workspace || {}).active || {};
    const items = [
      { label: "People tracked", value: isNum(v.people) ? String(v.people) : S().frame ? String((S().frame.players || []).length) : "–" },
      { label: "Ball", value: v.ball_visible ? "Visible" : S().frame && S().frame.ball && S().frame.ball.visible ? "Visible" : "Out of view" },
      { label: "Court mapping", value: v.calibrated ? (v.gate && v.gate !== "live" ? "Out of view" : "Mapped") : act.demo ? "Simulator map" : "Needs setup" },
      { label: "Throughput", value: isNum(v.fps) ? `${v.fps.toFixed(1)} fps · ${Math.round(v.inference_ms)} ms per frame` : "–" },
      { label: "Source", value: act.demo ? "Simulator" : isNum(act.fps) ? `${act.width}×${act.height} at ${Number(act.fps).toFixed(0)} fps` : "–" },
      { label: "Frames processed", value: isNum(status.frames) ? String(status.frames) : "–" },
      { label: "Behind real time", value: isNum(status.lag_s) ? `${status.lag_s.toFixed(1)} s` : "–" },
    ];
    return kv(items);
  }

  function listOf(kind, limit) {
    if (kind === "decisions") {
      const decs = ((stats() && stats().decisions) || []).slice(-limit).reverse();
      if (!decs.length) return el("p", { class: "blk-text", text: "Graded passes and shots appear here once the court is mapped." });
      return el("div", { class: "blk-list" }, decs.map((d) => el("div", { class: "blk-item" },
        el("span", { class: "mark", "data-tone": d.label === "good" ? "good" : d.label === "poor" ? "bad" : null }),
        el("span", { class: "what" }, cv().pname(d.player_id), el("small", { text: `${cv().timeLabel(d.t)} · ${d.label || "neutral"}` })),
        el("span", { class: "when", text: `${cv().fmtSigned(d.delta, 2)} pts` }))));
    }
    if (kind === "insights") {
      const ins = [...S().insights.values()].sort((a, b) => (b.priority || 0) - (a.priority || 0)).slice(0, limit);
      if (!ins.length) return el("p", { class: "blk-text", text: "No coach insights yet. They appear when the numbers support them." });
      return el("div", { class: "blk-list" }, ins.map((i) => el("div", { class: "blk-item" },
        el("span", { class: "mark", "data-team": i.team }), el("span", { class: "what" }, i.text || "", el("small", { text: cv().humanize(i.kind) })), el("span"))));
    }
    const lines = S().utter.slice(-limit).reverse();
    if (!lines.length) return el("p", { class: "blk-text", text: "No narration yet." });
    return el("div", { class: "blk-list" }, lines.map((u) => el("div", { class: "blk-item" },
      el("span", { class: "mark" }), el("span", { class: "what" }, u.text || "", el("small", { text: `${cv().timeLabel(u.t)} · ${cv().humanize(u.channel)}` })), el("span"))));
  }

  function labBlock() {
    const lab = window.CVLab && window.CVLab.data;
    if (!lab || !lab.available) return el("p", { class: "blk-text", text: lab && lab.reason ? lab.reason : "Open the Model lab to load the learning loop." });
    const st = lab.status || {}, cur = st.current || {}, next = lab.next;
    const items = [{ label: "Student detector", value: cur.student || "none" }, { label: "Cue models", value: cur.cue || "none" }, { label: "Fusion settings", value: cur.fusion || "none" },
      { label: "Games", value: String((st.games || []).filter((g) => g.primary).length) }, { label: "Leaderboard rows", value: String((st.leaderboard || {}).rows || 0) }];
    return el("div", { class: "gen-blocks" }, kv(items), next ? el("div", { class: "blk-note", "data-tone": "ember" }, icon("arrow"), `Next: ${next.label} (${next.heavy ? "heavy" : "light"}). ${next.why}`) : null);
  }

  function block(b) {
    switch (b.type) {
      case "text": return el("p", { class: "blk-text", text: b.text });
      case "note": return el("div", { class: "blk-note", "data-tone": b.tone }, icon(b.tone === "warn" || b.tone === "bad" ? "alert" : b.tone === "good" ? "check" : "info"), el("span", { text: b.text }));
      case "metrics": return el("div", { class: "blk-metrics" }, b.items.map(metricItem));
      case "duel": return duel(b);
      case "leaders": return leaders(b);
      case "spark": return spark();
      case "shots": return shots(b);
      case "plays": return playsList(b);
      case "player": return playerCard(b);
      case "kv": return kv(b.items);
      case "steps": return el("ol", { class: "blk-steps" }, b.items.map((it) => el("li", { class: "step", "data-state": it.state },
        el("span", { class: "step-dot" }, it.state === "done" ? icon("check") : null), el("div", null, el("div", { class: "step-title", text: it.title }), it.text ? el("div", { class: "step-text", text: it.text }) : null))));
      case "actions": return el("div", { class: "blk-actions" }, b.items.map((it) => el("button", { type: "button", class: `btn btn-sm${it.primary ? " btn-primary" : ""}`, text: it.label, onclick: () => window.CVShell && window.CVShell.run(it.command) })));
      case "code": return el("div", { class: "blk-code" }, el("code", { text: b.text }), el("button", { type: "button", class: "icon-btn", "aria-label": "Copy", title: "Copy", onclick: () => copy(b.text) }, icon("copy")));
      case "tracking": return tracking();
      case "lab": return labBlock();
      case "decisions": case "insights": case "narration": return listOf(b.type, b.limit);
      default: return null;
    }
  }

  async function copy(text) {
    try { await navigator.clipboard.writeText(text); cv().toast("Copied"); }
    catch { cv().toast("Copy is unavailable here: select the text instead.", "error"); }
  }

  // ------------------------------------------------------------------ cards

  const mounted = new Set();

  /** Render a spec as a card. opts: {by, onClose, closeLabel, tools: [nodes], static, live}. */
  function renderCard(spec, opts = {}) {
    const s = clean(spec);
    const card = el("article", { class: `gen-card${opts.static ? " static" : ""}`, "data-tone": s.tone });
    const blocks = el("div", { class: "gen-blocks" });
    const draw = () => {
      const nodes = s.blocks.map(block).filter(Boolean);
      nodes.forEach((n, i) => n.style.setProperty("--i", String(i)));
      blocks.replaceChildren(...nodes);
    };
    draw();
    card.append(el("div", { class: "gen-head" }, el("span", { class: "gen-icon" }, icon(s.icon)),
      el("div", { class: "gen-titles" }, el("h3", { class: "gen-title", text: s.title }), s.summary ? el("p", { class: "gen-sub", text: s.summary }) : null),
      opts.onClose ? el("button", { type: "button", class: "icon-btn gen-close", "aria-label": opts.closeLabel || "Remove", title: opts.closeLabel || "Remove", onclick: opts.onClose }, icon("close")) : null));
    if (s.blocks.length) card.append(blocks);
    if (opts.by || (opts.tools && opts.tools.length)) {
      card.append(el("div", { class: "gen-foot" }, opts.by ? composer(opts.by) : null, el("span", { class: "spacer" }), ...(opts.tools || [])));
    }
    if (opts.live !== false) {
      const entry = { card, update: () => { card.classList.add("settled"); draw(); } };
      card._live = entry;
      mounted.add(entry);
    }
    card.addEventListener("animationend", (ev) => { if (ev.target === card) card.classList.add("settled"); });
    return card;
  }

  function composer(by) {
    return el("span", { class: "composer", "data-by": by }, icon("spark"), by === "claude" ? "Composed with Claude" : "Composed on this Mac");
  }

  let refreshTimer = 0;
  function scheduleRefresh() {
    if (refreshTimer) return;
    refreshTimer = setTimeout(() => {
      refreshTimer = 0;
      for (const entry of [...mounted]) {
        if (!entry.card.isConnected) { mounted.delete(entry); continue; }
        try { entry.update(); } catch (err) { console.error("card update failed", err); }
      }
      renderBriefing();
      renderSuggestions();
    }, 600);
  }

  // ------------------------------------------------------------------ the on-device planner

  const STAT_WORDS = [
    [/\b(points?|scor(e|ing|er)s?|buckets?)\b/, "pts"], [/\b(rebounds?|boards?|glass)\b/, "reb"], [/\bassists?\b|\bdimes?\b/, "ast"],
    [/\bsteals?\b/, "stl"], [/\bblocks?\b|\brejections?\b/, "blk"], [/\bturnovers?\b|\bgiveaways?\b/, "tov"],
    [/\bthrees?\b|\b3s\b|\b3-?pointers?\b|three-?pointers?|\b3pt\b/, "fg3m"], [/\bfouls?\b/, "pf"], [/\bexpected\b|\bxpts\b/, "xpts"],
  ];

  function findTeam(q) {
    for (const t of ["home", "away"]) {
      const name = String(S().teams[t] || "").toLowerCase();
      const words = name.split(/\s+/).filter((w) => w.length > 2);
      if ((name && q.includes(name)) || words.some((w) => new RegExp(`\\b${w.replace(/[^a-z0-9]/g, "")}\\b`).test(q))) return t;
    }
    if (/\bhome\b/.test(q)) return "home";
    if (/\b(away|visitors?|road)\b/.test(q)) return "away";
    return null;
  }

  function findPlayer(q) {
    const m = /(?:#|\bnumber\s*|\bno\.?\s*)(\d{1,2})\b/.exec(q);
    const ids = new Map();
    for (const [id, r] of S().roster) ids.set(id, r);
    for (const p of playerLines()) if (!ids.has(p.id)) ids.set(p.id, null);
    for (const fp of (S().frame && S().frame.players) || []) if (!ids.has(fp.id)) ids.set(fp.id, null);
    if (m) {
      const team = findTeam(q);
      for (const id of ids.keys()) if (String(cv().pjersey(id)) === m[1] && (!team || playerTeam(id) === team)) return id;
    }
    for (const [id, r] of ids) {
      if (q.includes(id.toLowerCase())) return id;
      const name = r && r.name ? r.name.toLowerCase() : "";
      if (name && (q.includes(name) || name.split(/\s+/).some((w) => w.length > 2 && new RegExp(`\\b${w}\\b`).test(q)))) return id;
    }
    return null;
  }

  const statOf = (q) => { for (const [re, k] of STAT_WORDS) if (re.test(q)) return k; return null; };
  const has = (q, re) => re.test(q);

  function spec(title, summary, blocks, extra) { return Object.assign({ title, summary, blocks, icon: "spark", tone: "neutral" }, extra || {}); }

  function summarySpec() {
    const sc = (stats() || {}).score || {};
    const top = playerLines().filter((p) => isNum(p.line.pts)).sort((a, b) => b.line.pts - a.line.pts)[0];
    const run = (stats() || {}).run;
    return spec("Game so far", isNum(sc.home) && isNum(sc.away)
      ? `${teamName("away")} ${sc.away}, ${teamName("home")} ${sc.home}${run && run.team && run.points >= 6 ? `. ${teamName(run.team)} are on a ${run.points}-${run.opp_points || 0} run` : ""}${top ? `. Top scorer so far: ${cv().pname(top.id)} with ${top.line.pts}` : ""}.`
      : "No score yet. Start the demo or load footage and map the court to log plays.",
    [{ type: "metrics", items: [{ game: "score_away" }, { game: "score_home" }, { game: "period" }, { game: "game_clock" }] },
      { type: "spark", series: "margin" }, { type: "leaders", stat: "pts", limit: 4 }, { type: "plays", filter: "scoring", limit: 4 }], { icon: "trophy" });
  }

  function plan(raw) {
    const q = ` ${String(raw || "").toLowerCase().trim()} `;
    const team = findTeam(q), player = findPlayer(q), stat = statOf(q);
    const T = team ? teamName(team) : null;

    if (has(q, /\b(help|what can (you|i)|how do i use|commands?)\b/)) {
      return { confident: true, spec: spec("What you can ask", "Questions are answered from the live game on this Mac. Answers are live cards you can pin to the board.", [
        { type: "kv", items: [{ label: "Compare the teams", value: "Team comparison" }, { label: "Who leads in rebounds?", value: "Player leaders" },
          { label: "Shot chart for the home team", value: "Shot map" }, { label: "How is #30 playing?", value: "Player card" },
          { label: "What needs review?", value: "Review queue" }, { label: "How is tracking doing?", value: "Tracking health" },
          { label: "What should I train next?", value: "Model lab" }] }], { icon: "info" }) };
    }
    if (has(q, /\b(review|wrong|mistakes?|errors?|verify|check|uncertain|low confidence|correct(ion)?s?)\b/)) {
      const n = reviewQueue().length;
      return { confident: true, spec: spec(n ? `${n} play${n === 1 ? "" : "s"} need review` : "Nothing needs review", n ? "Plays the engine flagged, newest first. Correct or confirm each one." : "Every logged play is confident or confirmed. Score corrections also appear here.",
        [{ type: "plays", filter: "review", limit: 8 }, { type: "actions", items: [{ label: "Open the review list", command: "plays:review", primary: true }, { label: "Add a play", command: "plays:add" }] }], { icon: "review", tone: n ? "warn" : "good" }) };
    }
    if (has(q, /\b(track(ing|er)?|detector|fps|frame ?rate|latency|throughput|vision|lag|behind)\b/)) {
      return { confident: true, spec: spec("Tracking health", "Live detector and tracker status for the open footage.", [{ type: "tracking" },
        { type: "actions", items: [{ label: "Map the court", command: "court:map" }, { label: "Open Live", command: "nav:live" }] }], { icon: "people" }) };
    }
    if (has(q, /\b(train(ing)?|model|loop|leaderboard|student|next step|fusion|cue models?|box error|promote|lab)\b/)) {
      const lab = window.CVLab && window.CVLab.data;
      const blocks = [{ type: "lab" }];
      if (lab && lab.next && lab.next.command) blocks.push({ type: "code", text: lab.next.command });
      blocks.push({ type: "actions", items: [{ label: "Open the model lab", command: "nav:lab", primary: true }, { label: "Refresh", command: "lab:refresh" }] });
      return { confident: true, spec: spec("Learning loop", "Current model versions and the next step the loop would run.", blocks, { icon: "lab" }) };
    }
    if (has(q, /\b(calibrat\w*|map(ping)? the court|court (setup|mapping|map)|landmarks?|homography)\b/)) {
      const v = (window.CVStudio && window.CVStudio.vision) || {};
      return { confident: true, spec: spec("Map the court", "Mapping turns image tracks into floor positions, which unlocks plays, the shot chart and the box score.", [
        { type: "steps", items: [{ title: "Load footage", state: (window.CVStudio && window.CVStudio.workspace && window.CVStudio.workspace.active) ? "done" : "now" },
          { title: "Pause on a clear wide shot", text: "Mapping pauses the video for you.", state: v.calibrated ? "done" : "todo" },
          { title: "Click four or more landmarks", text: "Spread them across the visible floor; lane corners work well.", state: v.calibrated ? "done" : "todo" },
          { title: "Check the fit", text: "RMSE under 1 ft passes, but also check the map by eye.", state: v.calibrated ? "done" : "todo" }] },
        { type: "actions", items: [{ label: "Map the court", command: "court:map", primary: true }] }], { icon: "court" }) };
    }
    if (has(q, /\b(scorebug|scoreboard setup|roi|digit cells?)\b/)) {
      return { confident: true, spec: spec("Read the scoreboard", "Mark the venue scoreboard corners, then a box per digit, so scores and clocks are read from the video.", [
        { type: "steps", items: [{ title: "Load the latest frame", state: "now" }, { title: "Click the four corners", state: "todo" }, { title: "Drag a box per digit cell", state: "todo" }, { title: "Save the regions", state: "todo" }] },
        { type: "actions", items: [{ label: "Open scoreboard setup", command: "nav:roi", primary: true }] }], { icon: "film" }) };
    }
    if (has(q, /\b(export|download|json|save the analysis)\b/)) {
      return { confident: true, spec: spec("Export the analysis", "Downloads plays, stats, the transcript and tracking status as JSON.", [{ type: "actions", items: [{ label: "Export JSON", command: "footage:export", primary: true }] }], { icon: "film" }) };
    }
    if (player && !has(q, /\bcompare\b/)) {
      return { confident: true, spec: spec(cv().pname(player), "Live line and shots for this player.", [{ type: "player", id: player }, { type: "shots", player }, { type: "plays", filter: "key", player, limit: 4 }], { icon: "people" }) };
    }
    if (has(q, /\b(momentum|runs?|trend|margin|swing|lead changes?|flow)\b/)) {
      const run = (stats() || {}).run;
      return { confident: true, spec: spec("Momentum", run && run.team && run.points >= 6 ? `${teamName(run.team)} are on a ${run.points}-${run.opp_points || 0} run.` : "No run of six points or more right now.",
        [{ type: "spark", series: "margin" }, { type: "metrics", items: [{ game: "margin" }, { game: "score_away" }, { game: "score_home" }] }, { type: "plays", filter: "scoring", limit: 5 }], { icon: "flame", tone: run && run.points >= 6 ? "ember" : "neutral" }) };
    }
    if (has(q, /\b(shot ?chart|shots?|shooting|zones?|xfg|shot quality|where)\b/) && !stat) {
      return { confident: true, spec: spec(T ? `${T} shot chart` : "Shot chart", "Made shots are filled, misses are rings; colour is expected FG% (xFG).",
        [{ type: "shots", team }, team ? { type: "metrics", items: [{ stat: "fg_pct", team }, { stat: "efg", team }, { stat: "avg_xfg", team }] }
          : { type: "duel", stats: ["fg_pct", "fg3_pct", "efg", "avg_xfg"] }], { icon: "ball" }) };
    }
    if (has(q, /\b(decisions?|epv|grades?|good passes|bad passes|shot selection)\b/)) {
      return { confident: true, spec: spec("Decision quality", "EPV-lite grades: the chosen option against the best one available.", [{ type: "decisions", limit: 8 }], { icon: "chart" }) };
    }
    if (has(q, /\b(narration|commentary|said|transcript|voice|caption)\b/)) {
      return { confident: true, spec: spec("Narration", "The latest lines from the narrator.", [{ type: "narration", limit: 6 }], { icon: "review" }) };
    }
    if (has(q, /\b(insights?|coach(ing)?|advice|tactics?|adjust)\b/)) {
      return { confident: true, spec: spec("Coach insights", "Insights appear only when their sample supports them.", [{ type: "insights", limit: 5 }], { icon: "bolt" }) };
    }
    if (stat || has(q, /\b(leaders?|leading|top|best|most|who('s| is)? (hot|best|leading))\b/)) {
      const k = stat || "pts", def = STATS[k];
      return { confident: true, spec: spec(`${def.label}${T ? ` · ${T}` : ""}`, `${def.lowerBetter ? "Fewest" : "Most"} ${def.label.toLowerCase()} by player${T ? ` for ${T}` : ""}.`,
        [{ type: "leaders", stat: k, team, limit: 6 }, def.team ? { type: "duel", stats: [k] } : null].filter(Boolean), { icon: "trophy" }) };
    }
    if (has(q, /\b(compare|vs\.?|versus|matchup|teams?|winning|who('s| is) (up|ahead|winning)|score|efficiency|pace)\b/)) {
      return { confident: true, spec: spec(`${teamName("away")} vs ${teamName("home")}`, "Live team comparison; the brighter bar leads.", [
        { type: "metrics", items: [{ game: "score_away" }, { game: "score_home" }, { game: "margin" }] },
        { type: "duel", stats: ["fg_pct", "fg3_pct", "efg", "ts", "reb", "ast", "tov", "ppp"] }], { icon: "chart" }) };
    }
    if (has(q, /\b(summary|summari[sz]e|recap|what happened|overview|status|how('s| is) the game|so far)\b/)) return { confident: true, spec: summarySpec() };
    return { confident: false, spec: spec("I couldn't place that question", claudeReady ? "Press ⌘↵ to ask Claude, or try one of these." : "Answers are built on this Mac from the live game. Try one of these:", [
      { type: "kv", items: suggestions().slice(0, 5).map((s2) => ({ label: s2, value: "↵" })) }], { icon: "info" }) };
  }

  // ------------------------------------------------------------------ Claude

  let claudeReady = false, claudeModel = null;

  async function probeClaude() {
    try {
      const r = await cv().api("api/ask");
      claudeReady = !!(r && r.available);
      claudeModel = r && r.model;
    } catch { claudeReady = false; }
    window.dispatchEvent(new CustomEvent("cv-claude", { detail: { available: claudeReady, model: claudeModel } }));
  }

  async function askClaude(question) {
    const r = await cv().api("api/ask", { question: str(question, 300) });
    const s = clean(r && r.spec);
    if (!s) throw new Error("Claude returned no usable card.");
    return { spec: s, by: "claude", model: r.model };
  }

  /** Answer a question: on-device first; Claude when asked for, or when the planner cannot place it. */
  async function answer(question, opts = {}) {
    const local = plan(question);
    if (opts.claude || (!local.confident && claudeReady)) {
      if (!claudeReady) return { spec: clean(local.spec), by: "device", note: "Claude is not configured on this server." };
      try { return await askClaude(question); }
      catch (err) { return { spec: clean(local.spec), by: "device", note: err.message || String(err) }; }
    }
    return { spec: clean(local.spec), by: "device" };
  }

  function suggestions() {
    const out = [];
    const review = reviewQueue().length;
    if (review) out.push(`What needs review? (${review})`);
    const run = (stats() || {}).run;
    if (run && run.team && run.points >= 6) out.push(`Show the ${teamName(run.team)} run`);
    const top = playerLines().filter((p) => isNum(p.line.pts) && p.line.pts > 0).sort((a, b) => b.line.pts - a.line.pts)[0];
    if (top) out.push(`How is ${cv().pname(top.id)} playing?`);
    if (stats()) out.push("Compare the teams", `${teamName("home")} shot chart`, "Who leads in rebounds?", "Show the momentum");
    else out.push("How is tracking doing?", "How do I map the court?");
    out.push("What should I train next?", "Summarise the game");
    return [...new Set(out)].slice(0, 7);
  }

  // ------------------------------------------------------------------ the briefing

  function briefingSpecs() {
    const st = window.CVStudio || {}, ws = st.workspace, status = st.status || {}, vision = st.vision || {};
    const active = ws && ws.active, out = [];
    const err = (status && status.error) || (ws && ws.error);
    if (err) out.push(["error", spec("Analysis stopped", err, [{ type: "actions", items: [{ label: "Load footage", command: "footage:load", primary: true }] }], { icon: "alert", tone: "bad" })]);
    if (ws && ws.model && ws.model.state === "loading") out.push(["model", spec("Preparing the detector", "The first run downloads about 350 MB. Your video opens when it is ready.", [], { icon: "bolt", tone: "ember" })]);
    if (st.isStudio !== false && ws && !active) {
      out.push(["welcome", spec("Start with footage", "Everything runs on this Mac. The demo game shows the whole workspace in a few seconds.", [
        { type: "steps", items: [{ title: "Load footage", text: "MP4, MOV, MKV or AVI, up to 8 GB.", state: "now" }, { title: "Track people and the ball", state: "todo" }, { title: "Map the court", text: "Unlocks plays, the shot chart and the box score.", state: "todo" }] },
        { type: "actions", items: [{ label: "Load footage", command: "footage:load", primary: true }, { label: "Run the demo game", command: "footage:demo" }] }], { icon: "film", tone: "ember" })]);
    }
    if (active && active.demo) out.push(["demo", spec("Simulator demo", "Synthetic tracking and plays that exercise every view. Nothing here is camera evidence.", [], { icon: "info", tone: "info" })]);
    else if (active && status.needs_calibration) {
      out.push(["calibrate", spec("Map the court to unlock plays", "People are tracked in the image. Four or more court landmarks turn tracks into floor positions, plays and stats.", [
        { type: "steps", items: [{ title: "Tracking people and the ball", state: "done" }, { title: "Map the court", state: "now" }, { title: "Plays, shot chart and box score", state: "todo" }] },
        { type: "actions", items: [{ label: "Map the court", command: "court:map", primary: true }] }], { icon: "court", tone: "warn" })]);
    } else if (active && vision.calibrated && vision.gate && vision.gate !== "live") {
      out.push(["cut", spec("Camera cut away", "Spatial plays pause until the game camera returns.", [], { icon: "film", tone: "warn" })]);
    }
    if (status.finished && active) {
      out.push(["done", spec("Analysis complete", "Review the flagged plays, then export the result.", [
        { type: "metrics", items: [{ game: "score_away" }, { game: "score_home" }, { game: "review" }] },
        { type: "actions", items: [{ label: "Export JSON", command: "footage:export", primary: true }, { label: "Open box score", command: "nav:stats" }] }], { icon: "check", tone: "good" })]);
    }
    const review = reviewQueue();
    if (review.length) out.push([`review`, spec(`${review.length} play${review.length === 1 ? "" : "s"} need review`, "Flagged by the engine; newest first.", [{ type: "plays", filter: "review", limit: 3 }, { type: "actions", items: [{ label: "Show all", command: "plays:review" }] }], { icon: "review", tone: "warn" })]);
    const run = (stats() || {}).run;
    if (run && run.team && run.points >= 6) out.push(["run", spec(`${teamName(run.team)} on a ${run.points}-${run.opp_points || 0} run`, "Score margin over the analysed video.", [{ type: "spark", series: "margin" }], { icon: "flame", tone: "ember" })]);
    const top = playerLines().filter((p) => isNum(p.line.pts) && p.line.pts >= 6).sort((a, b) => b.line.pts - a.line.pts)[0];
    if (top) {
      const l = top.line;
      out.push(["leader", spec(`Leading scorer: ${cv().pname(top.id)}`, `${l.pts} points on ${made(l.fgm, l.fga)} shooting${isNum(l.fg3m) && l.fg3m ? `, ${l.fg3m} from three` : ""}.`, [
        { type: "actions", items: [{ label: "Open player", command: `player:${top.id}` }] }], { icon: "trophy" })]);
    }
    if (active && !active.demo && vision.calibrated && (!vision.gate || vision.gate === "live")) {
      out.push(["estimates", spec("Estimated plays", "Generic detector: verify teams, player identities and the scoreboard. Replays need review.", [], { icon: "info" })]);
    }
    const fps = vision.fps, src = active && Number(active.fps);
    if (active && !active.demo && isNum(fps) && src && fps < src * 0.5 && !status.paused && !status.finished) {
      out.push(["slow", spec("Running behind real time", `Analysis runs at ${fps.toFixed(1)} fps against a ${src.toFixed(0)} fps source, so it falls behind the clock. Every frame is still processed.`, [], { icon: "clock", tone: "warn" })]);
    }
    return out.slice(0, 4);
  }

  const briefKeys = new Set();
  // Rebuilt on every refresh so live numbers stay current; only cards new to the briefing animate in.
  function renderBriefing() {
    const root = document.getElementById("briefing");
    if (!root || !window.CV) return;
    const nodes = [], seen = new Set();
    for (const [key, sp] of briefingSpecs()) {
      const node = renderCard(sp, { live: false });
      if (briefKeys.has(key)) node.classList.add("settled");
      seen.add(key);
      nodes.push(node);
    }
    briefKeys.clear();
    for (const k of seen) briefKeys.add(k);
    root.replaceChildren(...nodes);
  }

  // ------------------------------------------------------------------ the board

  const BOARD_KEY = "cv-board";
  let board = [];
  function loadBoard() {
    try {
      const raw = window.localStorage.getItem(BOARD_KEY);
      const parsed = raw ? JSON.parse(raw) : [];
      board = Array.isArray(parsed) ? parsed.filter((c) => c && c.spec).slice(0, 40) : [];
    } catch { board = []; }
  }
  function saveBoard() {
    try { window.localStorage.setItem(BOARD_KEY, JSON.stringify(board)); } catch { /* storage blocked: the board lasts for this tab */ }
    const n = document.getElementById("board-count");
    if (n) n.textContent = board.length ? String(board.length) : "";
  }

  function pin(sp, by, question) {
    const s = clean(sp);
    if (!s) return;
    board.unshift({ id: `c${Date.now().toString(36)}${Math.random().toString(36).slice(2, 6)}`, spec: s, by: by || "device", q: str(question, 300), at: Date.now() });
    board = board.slice(0, 40);
    saveBoard();
    renderBoard();
    cv().toast("Pinned to the board");
  }

  function renderBoard() {
    const grid = document.getElementById("board-grid");
    if (!grid) return;
    grid.replaceChildren(...board.map((c) => renderCard(c.spec, {
      by: c.by, onClose: () => { board = board.filter((x) => x.id !== c.id); saveBoard(); renderBoard(); }, closeLabel: "Remove from the board",
      tools: c.q ? [el("span", { class: "muted", title: c.q, text: `“${str(c.q, 48)}${c.q.length > 48 ? "…" : ""}”` })] : [],
    })));
    document.getElementById("board-empty").hidden = board.length > 0;
  }

  function renderSuggestions() {
    const root = document.getElementById("board-suggest");
    if (!root || !window.CV) return;
    const list = suggestions();
    const sig = list.join("|");
    if (root.dataset.sig === sig) return;
    root.dataset.sig = sig;
    root.replaceChildren(...list.map((s2) => el("button", { type: "button", class: "chip", onclick: () => boardAsk(s2.replace(/\s*\(\d+\)$/, "")) }, icon("spark"), s2)));
  }

  async function boardAsk(question, opts = {}) {
    const q = str(question, 300).trim();
    if (!q) return;
    const grid = document.getElementById("board-grid");
    const ghost = el("article", { class: "gen-card" }, el("div", { class: "skel w60" }), el("div", { class: "skel w40" }), el("div", { class: "skel tall" }));
    grid.prepend(ghost);
    document.getElementById("board-empty").hidden = true;
    const res = await answer(q, opts);
    ghost.remove();
    pin(res.spec, res.by, q);
    if (res.note) cv().toast(res.note, "error");
  }

  function openPlayer(id) {
    if (!id) return;
    const sp = spec(cv().pname(id), "Live line and shots for this player.", [{ type: "player", id }, { type: "shots", player: id }, { type: "plays", filter: "key", player: id, limit: 4 }], { icon: "people" });
    if (window.CVShell) window.CVShell.showAnswer(clean(sp), "device", cv().pname(id));
  }

  // ------------------------------------------------------------------ boot

  window.CVGen = { renderCard, clean, plan, answer, suggestions, pin, openPlayer, boardAsk, icon, el, get claude() { return { available: claudeReady, model: claudeModel }; } };

  function boot() {
    if (!window.CV) return;
    loadBoard();
    saveBoard();
    renderBoard();
    renderSuggestions();
    renderBriefing();
    const form = document.getElementById("board-ask");
    form.addEventListener("submit", (ev) => {
      ev.preventDefault();
      const input = document.getElementById("board-input");
      const q = input.value;
      input.value = "";
      boardAsk(q);
    });
    for (const name of ["cv-update", "cv-studio", "cv-lab"]) window.addEventListener(name, scheduleRefresh);
    window.addEventListener("cv-view", (ev) => { if (ev.detail.id === "tab-board") { renderBoard(); renderSuggestions(); } });
    probeClaude();
    window.addEventListener("cv-studio", () => {
      const caps = window.CVStudio && window.CVStudio.workspace && window.CVStudio.workspace.capabilities;
      if (caps && !!caps.claude !== claudeReady) probeClaude();
    });
  }

  // after every deferred script (the shell runs the card buttons)
  if (document.readyState === "complete") boot();
  else document.addEventListener("DOMContentLoaded", boot);
})();
