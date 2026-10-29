/* CourtVision model lab: the NBA learning loop (api/lab) as a view, with one runner and its log. */
(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const cv = () => window.CV;
  const g = () => window.CVGen;
  const isNum = (v) => typeof v === "number" && Number.isFinite(v);
  const STAGE_LABELS = {
    espn: "ESPN", scorebug: "Scorebug", camera: "Camera", teacher: "Teacher", ball_labels: "Ball labels", detect: "Detect",
    ball: "Ball", events: "Events", fuse: "Fuse", eval_fused: "Eval", record: "Record", eval_events: "Eval events",
    cue_rows: "Cue rows", pseudo: "Pseudo", hard_teacher: "Hard", hard_times: "Hard times",
  };
  const KIND_LABELS = { student: "Student detector", cue: "Cue models", fusion: "Fusion settings" };

  let data = null, logText = "", timer = 0, busy = false, visible = false, missing = false;

  const el = (...a) => g().el(...a);

  function stageState(s) {
    const v = String(s || "");
    if (v === "done") return "done";
    if (v === "ready") return "ready";
    if (v.startsWith("failed")) return "failed";
    if (v.startsWith("blocked")) return "blocked";
    if (v === "-" || v === "n/a" || v === "skip") return "na";
    return "todo";
  }

  async function refresh(quiet) {
    if (busy) return;
    busy = true;
    try {
      data = await cv().api("api/lab");
      missing = false;
      if (data.available && (data.runner.running || visible)) {
        const lg = await cv().api("api/lab/log");
        logText = lg.text || "";
        data.logPath = lg.path;
      }
    } catch (err) {
      if (err.status === 404) { missing = true; data = { available: false, reason: "This server has no model lab: open the Studio with courtvision studio." }; }
      else if (!quiet) cv().toast(err.message || String(err), "error");
    } finally {
      busy = false;
    }
    paintBadge();
    if (visible) render();
    window.dispatchEvent(new CustomEvent("cv-lab"));
    schedule();
  }

  function schedule() {
    clearTimeout(timer);
    if (missing) return;
    const running = data && data.runner && data.runner.running;
    timer = setTimeout(() => refresh(true), running ? 2500 : visible ? 8000 : 60000);
  }

  function paintBadge() {
    const badge = $("lab-badge");
    const running = data && data.runner && data.runner.running;
    const ready = data && data.next ? data.next.ready : 0;
    badge.hidden = !(running || ready);
    badge.textContent = running ? "Running" : String(ready);
    badge.title = running ? "A loop run is going" : `${ready} step${ready === 1 ? "" : "s"} ready`;
  }

  async function post(path, body, ok) {
    try {
      const out = await cv().api(path, body || {});
      if (ok) cv().toast(ok);
      await refresh(true);
      return out;
    } catch (err) {
      cv().toast(err.message || String(err), "error");
      return null;
    }
  }

  const runNext = () => post("api/lab/run", { max_steps: 1 }, "Started the next step");
  const runAll = () => post("api/lab/run", {}, "Started the loop");
  const stop = () => post("api/lab/stop", {}, "Stopping the loop run");
  async function copyNext() {
    const cmd = data && data.next && data.next.command;
    if (!cmd) return;
    try { await navigator.clipboard.writeText(cmd); cv().toast("Command copied"); }
    catch { cv().toast("Copy is unavailable here: select the command instead.", "error"); }
  }

  function card(title, meta, body, extra) {
    return el("article", { class: `card ${extra || ""}` }, el("header", { class: "card-head" }, el("h2", { text: title }), meta || null), body);
  }

  function render() {
    const root = $("lab-root");
    if (!data) return;
    if (!data.available) {
      root.replaceChildren(el("article", { class: "card lab-empty" },
        el("h2", { text: "The model lab is not available here" }), el("p", { class: "hint", text: data.reason || "" })));
      return;
    }
    const st = data.status || {}, cur = st.current || {}, cands = st.candidates || {}, run = data.runner || {}, m = data.machine || {};
    // models
    const tiles = el("div", { class: "model-tiles" }, Object.keys(KIND_LABELS).map((k) => {
      const vid = cur[k], ver = ((st.versions || {})[k] || {})[vid] || {};
      return el("div", { class: "model-tile" },
        el("span", { class: "model-kind", text: KIND_LABELS[k] }), el("span", { class: "model-id", text: vid || "none" }),
        el("span", { class: "model-src", text: vid ? `${ver.source || "unknown"} · fitted on ${(ver.fit_games || []).length} game${(ver.fit_games || []).length === 1 ? "" : "s"}` : "no version yet" }),
        (cands[k] || []).length ? el("div", { class: "model-cands" }, cands[k].map((c) => el("span", { class: "badge badge-llm", text: `candidate ${c}` }))) : null);
    }));
    const models = card("Current models", el("span", { class: "muted", text: st.current_set ? `set ${st.current_set}` : "" }), tiles, "span-7");

    // next step
    const nx = data.next;
    const nextBody = el("div", { class: "card-body" });
    if (nx) {
      nextBody.append(
        el("div", null, el("div", { class: "next-label", text: nx.label }), el("p", { class: "next-why", text: nx.why })),
        el("div", { class: "blk-code" }, el("code", { text: nx.command }), el("button", { type: "button", class: "icon-btn", "aria-label": "Copy the command", title: "Copy", onclick: copyNext }, g().icon("copy"))),
        el("div", { class: "lab-run" },
          el("button", { type: "button", class: "btn btn-ember", disabled: run.running, onclick: runNext }, "Run next step"),
          el("button", { type: "button", class: "btn", disabled: run.running, onclick: runAll }, "Run every ready step"),
          el("span", { class: "muted", text: `${nx.ready} ready · ${nx.heavy_ready} heavy · ${nx.heavy ? "heavy step" : "light step"}` })),
      );
    } else {
      const notes = Object.values(data.notes || {});
      nextBody.append(el("div", { class: "next-label", text: "Nothing to run now" }),
        el("p", { class: "next-why", text: notes.length ? notes.join(" ") : "Every stage is done for the current models. Add a game folder with a game.json to keep learning." }));
    }
    for (const a of data.assessments || []) {
      nextBody.append(el("div", { class: "blk-note", "data-tone": a.outcome === "promoted" ? "good" : "neutral" }, g().icon(a.outcome === "promoted" ? "check" : "info"),
        el("span", { text: `${a.candidate}: promotion would be ${a.outcome}. ${a.reason || ""}` }),
        a.outcome === "promoted" ? el("button", { type: "button", class: "btn btn-sm", onclick: () => post("api/lab/promote", { version: a.candidate }, `Promoted ${a.candidate}`) }, "Promote") : null));
    }
    const next = card("Next step", null, nextBody, "span-5 lab-next");

    // machine and runner
    const free = m.free_pct;
    const gauge = el("div", { class: "gauge" },
      el("div", { class: "gauge-top" }, el("span", { text: "Free memory" }), el("b", { text: isNum(free) ? `${free.toFixed(0)}%` : "unknown" })),
      (() => {
        const bar = el("div", { class: "gauge-bar" }, el("i", { "data-tone": !isNum(free) ? null : free < m.watch_free ? "bad" : free < m.min_free ? "warn" : null }), el("span", { class: "mark" }), el("span", { class: "mark" }));
        bar.children[0].style.setProperty("--w", isNum(free) ? String(Math.min(1, free / 100)) : "0");
        bar.children[1].style.setProperty("--at", String((m.min_free || 25) / 100));
        bar.children[2].style.setProperty("--at", String((m.watch_free || 8) / 100));
        return bar;
      })(),
      el("p", { class: "hint", text: `A step starts only above ${m.min_free}% free; the watchdog stops it under ${m.watch_free}%.` }));
    const runnerRows = [
      { label: "Loop run", value: run.running ? `running for ${Math.round(run.elapsed_s || 0)} s (pid ${run.pid})` : run.exit != null ? `last run: ${run.exit_meaning}` : "idle" },
      { label: "Other jobs", value: (m.jobs || []).length ? (m.jobs || []).map((j) => `pid ${j.pid}`).join(", ") : "none" },
    ];
    const machineBody = el("div", { class: "card-body machine" }, gauge, el("dl", { class: "blk-kv" }, runnerRows.flatMap((r) => [el("dt", { text: r.label }), el("dd", { text: r.value, title: r.value })])),
      run.running ? el("button", { type: "button", class: "btn btn-danger", onclick: stop }, "Stop the loop run") : null);
    const machine = card("Machine", null, machineBody, "span-4");

    // games x stages
    const stages = data.stages || Object.keys(STAGE_LABELS);
    const table = el("table", { class: "stage-table" },
      el("thead", null, el("tr", null, el("th", { scope: "col", text: "Game" }), stages.map((s) => el("th", { scope: "col", text: STAGE_LABELS[s] || s, title: s })), el("th", { scope: "col", text: "Box error" }))),
      el("tbody", null, (st.games || []).map((gm) => {
        if (!gm.primary) {
          return el("tr", null, el("td", null, el("div", { class: "gname", text: gm.game }), el("div", { class: "gsub", text: gm.error || (gm.excluded ? "left out of the loop" : `variant of ${gm.variant_of}`) })),
            el("td", { colspan: String(stages.length + 1), class: "muted", text: "Left alone unless named in a run" }));
        }
        const tag = ((gm.inventory || {}).tags || {})[st.tag] || {};
        return el("tr", null,
          el("td", null, el("div", { class: "gname", text: gm.game }), el("div", { class: "gsub", text: `ESPN ${gm.game_id}` })),
          stages.map((s) => {
            let state = stageState((gm.stages || {})[s]);
            if (run.running && nx && nx.game === gm.game && nx.stage === s) state = "running";
            return el("td", null, el("span", { class: "stage-cell", "data-state": state, title: `${STAGE_LABELS[s] || s}: ${(gm.stages || {})[s] || "-"}` }));
          }),
          el("td", { class: "tnum", text: isNum(tag.box_error) ? tag.box_error.toFixed(3) : "–" }));
      })));
    const legend = el("div", { class: "stage-legend" }, [["done", "Done"], ["ready", "Ready to run"], ["running", "Running"], ["todo", "Waiting"], ["failed", "Failed"], ["blocked", "Blocked"]]
      .map(([s, l]) => el("span", null, el("span", { class: "stage-cell", "data-state": s }), l)));
    const games = card("Games", el("span", { class: "muted", text: `tag ${st.tag || "–"}` }), el("div", null, el("div", { class: "table-scroll" }, table), legend), "span-8");

    // leaderboard
    const lb = st.leaderboard || {};
    const entries = Object.entries(lb.mean_box_error || {});
    const maxErr = Math.max(0.0001, ...entries.map(([, v]) => v));
    const lbBody = entries.length ? el("div", { class: "lb-rows" }, entries.sort((a, b) => a[1] - b[1]).map(([set, v]) => {
      const track = el("span", { class: "bar-track" }, el("i"));
      track.firstChild.style.setProperty("--w", String(v / maxErr));
      return el("div", { class: `lb-row${set === st.current_set ? " current" : ""}` }, el("span", { class: "lb-set", text: set }), track, el("span", { class: "lb-val", text: v.toFixed(3) }));
    })) : el("p", { class: "card-body hint", text: "No evaluations yet. Each finished game run adds a row: lower mean box error is better." });
    const decisions = (st.decisions || []).slice().reverse().map((d) => el("div", { class: "blk-item" }, el("span", { class: "mark", "data-tone": d.promoted ? "good" : null }),
      el("span", { class: "what" }, `${d.candidate}: ${d.outcome}`, el("small", { text: d.reason || "" })), el("span", { class: "when", text: String(d.time || "").slice(0, 10) })));
    const board = card("Leaderboard", el("span", { class: "muted", text: `${lb.rows || 0} rows` }), el("div", null, lbBody, decisions.length ? el("div", { class: "card-body blk-list" }, decisions) : null), "span-4");

    // log
    const pre = el("pre", { class: "log-view", text: logText || "No loop log yet. Runs started here, and the latest step log, appear here." });
    const logCard = card("Log", el("span", { class: "muted", text: data.logPath || "" }), pre, "span-12");

    const grid = el("div", { class: "lab-grid" }, models, next, games, machine, board, logCard);
    root.replaceChildren(grid);
    pre.scrollTop = pre.scrollHeight;
  }

  window.CVLab = {
    get data() { return data; },
    refresh: () => refresh(false), runNext, copyNext,
    canRun: () => !!(data && data.available && data.next && !(data.runner && data.runner.running)),
  };

  function boot() {
    if (!window.CV || !window.CVGen) return;
    $("lab-refresh").addEventListener("click", () => refresh(false));
    window.addEventListener("cv-view", (ev) => {
      visible = ev.detail.id === "tab-lab";
      if (visible) refresh(true);
    });
    visible = !$("view-lab").hidden;
    refresh(true);
  }

  if (document.readyState === "complete") boot();
  else document.addEventListener("DOMContentLoaded", boot);
})();
