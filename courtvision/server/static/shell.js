/* CourtVision shell: the Ask palette (commands and generated answers), keyboard navigation, the
 * sliding nav indicator, the side-panel tabs, the theme switch and the mobile drawer. */
(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const cv = () => window.CV;
  const gen = () => window.CVGen;
  const studio = () => window.CVStudio || {};
  const isMac = /Mac|iPhone|iPad/.test(navigator.platform || navigator.userAgent);
  const MOD = isMac ? "⌘" : "Ctrl";

  const VIEWS = [
    ["live", "tab-live", "Live", "L"], ["stats", "tab-stats", "Box score", "S"], ["board", "tab-board", "Board", "B"],
    ["operator", "tab-operator", "Game controls", "C"], ["calibration", "tab-calibration", "Court mapping", "M"],
    ["roi", "tab-roi", "Scoreboard", "R"], ["lab", "tab-lab", "Model lab", "T"],
  ];

  function go(view) {
    const v = VIEWS.find((x) => x[0] === view);
    if (v) cv().selectView(v[1]);
    closeNav();
  }

  // ------------------------------------------------------------------ commands

  function commands() {
    const st = studio(), ws = st.workspace, status = st.status || {}, S = cv().S;
    const active = !!(ws && ws.active) || st.isStudio === false;
    const list = [];
    for (const [id, , label, key] of VIEWS) list.push({ id: `nav:${id}`, group: "Go to", label, icon: "arrow", keys: ["G", key] });
    list.push(
      { id: "footage:load", group: "Footage", label: "Load footage", icon: "film", when: () => st.isStudio !== false },
      { id: "footage:demo", group: "Footage", label: "Run the demo game", icon: "spark", when: () => st.isStudio !== false },
      { id: "footage:export", group: "Footage", label: "Export the analysis as JSON", icon: "copy", when: () => active && st.isStudio !== false },
      { id: "court:map", group: "Footage", label: "Map the court", icon: "court", when: () => active },
      { id: status.paused ? "playback:resume" : "playback:pause", group: "Playback", label: status.paused ? "Resume analysis" : "Pause analysis", icon: "clock", keys: ["P"], when: () => active },
      { id: "playback:mute", group: "Playback", label: S.muted ? "Unmute narration" : "Mute narration", icon: "review", keys: ["M"], when: () => active },
      { id: "playback:deadball", group: "Playback", label: "Mark dead ball", icon: "bolt", keys: ["D"], when: () => active },
      { id: "plays:add", group: "Plays", label: "Add a play", icon: "review", keys: ["A"] },
      { id: "plays:review", group: "Plays", label: "Show the plays that need review", icon: "alert" },
      { id: "view:overlay", group: "View", label: "Toggle the tracking overlay", icon: "people", keys: ["O"], when: () => active },
      { id: "view:possession", group: "View", label: "Toggle the possession analysis", icon: "chart", keys: ["B"], when: () => !!(window.CVPossession && window.CVPossession.data) },
      { id: "view:watch", group: "View", label: window.CVPossession && window.CVPossession.watching ? "Back to the live analysis" : "Watch the analysed game at full speed", icon: "film", keys: ["W"], when: () => !!(window.CVPossession && window.CVPossession.data) },
      { id: "view:fullscreen", group: "View", label: "Expand the game feed", icon: "film", keys: ["F"], when: () => active },
      { id: "view:rail", group: "View", label: "Toggle the side panel", icon: "chart", keys: ["]"] },
      { id: "lens:broadcast", group: "View", label: "Court lens: broadcast camera", icon: "court" },
      { id: "lens:overhead", group: "View", label: "Court lens: overhead", icon: "court" },
      { id: "lens:baseline", group: "View", label: "Court lens: baseline", icon: "court" },
      { id: "view:theme", group: "View", label: "Switch light / dark theme", icon: "spark" },
      { id: "app:settings", group: "View", label: "Settings", icon: "info" },
      { id: "app:shortcuts", group: "View", label: "Keyboard shortcuts", icon: "info", keys: ["?"] },
      { id: "lab:refresh", group: "Model lab", label: "Refresh the learning loop", icon: "lab" },
      { id: "lab:run", group: "Model lab", label: "Run the next training step", icon: "lab", when: () => !!(window.CVLab && window.CVLab.canRun()) },
      { id: "lab:copy-next", group: "Model lab", label: "Copy the next step's command", icon: "copy", when: () => !!(window.CVLab && window.CVLab.data && window.CVLab.data.next) },
    );
    for (const v of ((ws && ws.videos) || []).slice(0, 6)) list.push({ id: `footage:open:${v.id}`, group: "Recent footage", label: `Open ${v.name}`, icon: "film" });
    return list.filter((c) => !c.when || c.when());
  }

  function run(id) {
    const [kind, arg, extra] = String(id).split(":");
    switch (kind) {
      case "nav": go(arg); return;
      case "footage":
        if (arg === "load") studio().openDialog && studio().openDialog();
        else if (arg === "demo") studio().startDemo && studio().startDemo();
        else if (arg === "export") studio().exportJson && studio().exportJson();
        else if (arg === "open" && extra) studio().queueStart && studio().queueStart(extra);
        return;
      case "court": studio().setupCourt && studio().setupCourt(); return;
      case "playback":
        if (arg === "mute") cv().control(cv().S.muted ? "unmute" : "mute");
        else if (arg === "deadball") cv().control("dead_ball");
        else cv().control(arg);
        return;
      case "plays":
        if (arg === "add") cv().openEventDialog(null);
        else { go("live"); cv().setFeedMode("review"); selectRail("rtab-plays"); }
        return;
      case "view":
        if (arg === "overlay") studio().toggleOverlay && studio().toggleOverlay();
        else if (arg === "possession") window.CVPossession && window.CVPossession.toggle();
        else if (arg === "watch") window.CVPossession && window.CVPossession.watch();
        else if (arg === "fullscreen") studio().fullscreen && studio().fullscreen();
        else if (arg === "rail") toggleRail();
        else if (arg === "theme") toggleTheme();
        return;
      case "lens": { const b = document.querySelector(`[data-lens="${arg}"]`); if (b) b.click(); go("live"); return; }
      case "app":
        if (arg === "settings") cv().openSettings(); else openShortcuts();
        return;
      case "lab":
        if (!window.CVLab) return;
        if (arg === "refresh") { window.CVLab.refresh(); go("lab"); }
        else if (arg === "run") window.CVLab.runNext();
        else if (arg === "copy-next") window.CVLab.copyNext();
        return;
      case "player": gen().openPlayer(String(id).slice(7)); return;
      case "ask": openPalette(String(id).slice(4), true); return;
      default: break;
    }
  }

  // ------------------------------------------------------------------ the palette

  const P = { items: [], index: 0, mode: "list", answer: null, busy: false };
  const dlg = () => $("ask-dialog");

  function openPalette(text, submit) {
    if (!dlg().open) dlg().showModal();
    $("ask-input").value = text || "";
    showList();
    $("ask-input").focus();
    if (submit && text) ask(text, false);
  }

  function closePalette() { if (dlg().open) dlg().close(); }

  function score(label, q) {
    const l = label.toLowerCase();
    if (!q) return 1;
    if (l.startsWith(q)) return 3;
    if (l.includes(q)) return 2;
    const words = q.split(/\s+/).filter(Boolean);
    return words.length && words.every((w) => l.includes(w)) ? 1 : 0;
  }

  function looksLikeQuestion(q) {
    return /\?$|^(who|what|how|where|when|why|which|is|are|does|did|show|compare|tell|summar|explain|give)\b/i.test(q) || q.split(/\s+/).length >= 3;
  }

  function showList() {
    P.mode = "list";
    $("ask-answer").hidden = true;
    $("ask-list").hidden = false;
    $("ask-scope").hidden = true;
    const q = $("ask-input").value.trim();
    const ql = q.toLowerCase();
    const cmds = commands().map((c) => ({ ...c, s: score(c.label, ql) })).filter((c) => c.s > 0).sort((a, b) => b.s - a.s);
    const items = [];
    const askItem = q ? { id: "__ask", group: "Ask", label: q, icon: "spark", ask: true } : null;
    if (askItem && (looksLikeQuestion(q) || !cmds.length)) items.push(askItem);
    if (!q) for (const s of gen().suggestions().slice(0, 5)) items.push({ id: "__ask", group: "Suggested questions", label: s.replace(/\s*\(\d+\)$/, ""), icon: "spark", ask: true });
    const limit = q ? 9 : 12;
    for (const c of cmds.slice(0, limit)) items.push(c);
    if (askItem && !items.includes(askItem)) items.push(askItem);
    P.items = items;
    P.index = 0;
    const ul = $("ask-list");
    ul.replaceChildren();
    let group = "";
    items.forEach((it, i) => {
      if (it.group !== group) {
        group = it.group;
        ul.append(gen().el("li", { class: "pal-group", role: "presentation", text: group }));
      }
      const label = gen().el("span", { class: "pal-label" });
      if (it.ask) label.append(q && it.label === q ? "Ask: " : "", gen().el("mark", { text: it.label }));
      else label.append(it.label);
      const hint = gen().el("span", { class: "pal-hint" }, ...(it.keys || []).map((k) => gen().el("kbd", { text: k })));
      if (it.ask && cv() && gen().claude.available && q && it.label === q) hint.append(gen().el("span", { text: "↵ on this Mac · " }), gen().el("kbd", { text: MOD }), gen().el("kbd", { text: "↵" }), gen().el("span", { text: " Claude" }));
      const li = gen().el("li", { class: `pal-item${it.ask ? " ask" : ""}`, role: "option", id: `pal-${i}`, "aria-selected": String(i === 0),
        onclick: () => choose(i, false), onmousemove: () => highlight(i) }, gen().icon(it.icon), label, hint);
      ul.append(li);
    });
    if (!items.length) ul.append(gen().el("li", { class: "pal-group", text: "No matching commands. Press Enter to ask." }));
    $("ask-input").setAttribute("aria-activedescendant", items.length ? "pal-0" : "");
  }

  function highlight(i) {
    if (i === P.index) return;
    const prev = $(`pal-${P.index}`), next = $(`pal-${i}`);
    if (prev) prev.setAttribute("aria-selected", "false");
    if (next) { next.setAttribute("aria-selected", "true"); next.scrollIntoView({ block: "nearest" }); }
    P.index = i;
    $("ask-input").setAttribute("aria-activedescendant", `pal-${i}`);
  }

  function choose(i, claude) {
    const it = P.items[i];
    const q = $("ask-input").value.trim();
    if (!it) { if (q) ask(q, claude); return; }
    if (it.ask) { ask(it.label, claude); return; }
    closePalette();
    run(it.id);
  }

  async function ask(question, claude) {
    if (P.busy) return;
    P.busy = true;
    P.mode = "answer";
    $("ask-input").value = question;
    $("ask-list").hidden = true;
    const box = $("ask-answer");
    box.hidden = false;
    $("ask-scope").hidden = false;
    $("ask-scope").textContent = claude ? "Asking Claude" : "Answer";
    box.replaceChildren(gen().el("div", { class: "skel w60" }), gen().el("div", { class: "skel w40" }), gen().el("div", { class: "skel tall" }), gen().el("div", { class: "skel tall" }));
    try {
      const res = await gen().answer(question, { claude });
      showAnswer(res.spec, res.by, question, res.note);
    } finally {
      P.busy = false;
      $("ask-scope").textContent = "Answer";
    }
  }

  function showAnswer(spec, by, question, note) {
    if (!dlg().open) dlg().showModal();
    P.mode = "answer";
    P.answer = { spec, by, question };
    $("ask-list").hidden = true;
    $("ask-scope").hidden = false;
    if (question && !$("ask-input").value) $("ask-input").value = question;
    const box = $("ask-answer");
    box.hidden = false;
    const card = gen().renderCard(spec, { by, tools: [
      gen().el("button", { type: "button", class: "btn btn-sm", onclick: () => { gen().pin(spec, by, question); } }, gen().icon("pin"), "Pin to board"),
      gen().el("button", { type: "button", class: "btn btn-sm btn-ghost", onclick: () => { $("ask-input").value = ""; showList(); $("ask-input").focus(); } }, "New question"),
    ] });
    box.replaceChildren(card);
    if (note) box.append(gen().el("p", { class: "hint", text: note }));
    $("ask-input").select();
  }

  function onPaletteKey(ev) {
    if (ev.key === "ArrowDown" || ev.key === "ArrowUp") {
      if (P.mode !== "list" || !P.items.length) return;
      ev.preventDefault();
      highlight((P.index + (ev.key === "ArrowDown" ? 1 : -1) + P.items.length) % P.items.length);
    } else if (ev.key === "Enter") {
      ev.preventDefault();
      const claude = ev.metaKey || ev.ctrlKey;
      const q = $("ask-input").value.trim();
      if (P.mode === "answer") { if (q) ask(q, claude); return; }
      if (claude && q) { ask(q, true); return; }
      choose(P.index, claude);
    } else if (ev.key === "Backspace" && P.mode === "answer" && !$("ask-input").value) {
      showList();
    }
  }

  // ------------------------------------------------------------------ nav, rail, theme, drawer

  function moveIndicator() {
    const tab = document.querySelector('#side-nav [aria-selected="true"]');
    const ind = $("nav-indicator");
    if (!tab || !ind) return;
    ind.style.transform = `translateY(${tab.offsetTop}px)`;
    ind.style.height = `${tab.offsetHeight}px`;
  }

  function selectRail(id) {
    for (const t of document.querySelectorAll(".rail-tab")) {
      const on = t.id === id;
      t.setAttribute("aria-selected", String(on));
      t.tabIndex = on ? 0 : -1;
      $(t.getAttribute("aria-controls")).hidden = !on;
    }
    cv().store.set("localStorage", "cv-rail", id);
  }

  function toggleRail() {
    const hidden = document.body.classList.toggle("rail-hidden");
    $("rail-toggle").setAttribute("aria-pressed", String(!hidden));
    cv().store.set("localStorage", "cv-rail-hidden", hidden ? "1" : null);
    setTimeout(() => cv().lens && cv().lens.resize(), 30);
  }

  function effectiveTheme() {
    const t = document.documentElement.dataset.theme;
    if (t) return t;
    return window.matchMedia && window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark";
  }

  function toggleTheme() {
    const next = effectiveTheme() === "dark" ? "light" : "dark";
    cv().store.set("localStorage", "cv-theme", next);
    cv().applyTheme(next);
    paintThemeIcon();
  }

  function paintThemeIcon() {
    const btn = $("theme-toggle");
    const light = effectiveTheme() === "light";
    btn.setAttribute("aria-label", light ? "Switch to dark theme" : "Switch to light theme");
    btn.title = btn.getAttribute("aria-label");
    const path = btn.querySelector("path");
    path.setAttribute("d", light ? "M13.2 9.6A5.6 5.6 0 0 1 6.4 2.8a5.6 5.6 0 1 0 6.8 6.8z"
      : "M8 5.2a2.8 2.8 0 1 0 0 5.6 2.8 2.8 0 0 0 0-5.6zM8 1.5v1.3M8 13.2v1.3M14.5 8h-1.3M2.8 8H1.5M12.6 3.4l-.9.9M4.3 11.7l-.9.9M12.6 12.6l-.9-.9M4.3 4.3l-.9-.9");
  }

  function openNav() { document.body.classList.add("nav-open"); }
  function closeNav() { document.body.classList.remove("nav-open"); }

  function openShortcuts() {
    const rows = [
      [`${MOD} K`, "Ask or jump to anything"], ["/", "Ask"], ["G then L, S, B, C", "Live, box score, board, game controls"],
      ["G then M, R, T", "Court mapping, scoreboard, model lab"], ["P", "Pause or resume"], ["M", "Mute narration"],
      ["D", "Mark dead ball"], ["A", "Add a play"], ["O", "Tracking overlay"], ["B", "Possession analysis"], ["W", "Watch the analysed game"], ["F", "Expand the game feed"], ["]", "Side panel"], ["?", "This list"],
    ];
    $("shortcut-list").replaceChildren(...rows.flatMap(([k, v]) => [gen().el("dt", { text: v }), gen().el("dd", null, ...k.split(" ").map((x) => (x === "then" ? gen().el("span", { class: "muted", text: " then " }) : gen().el("kbd", { text: x }))))]));
    if (!$("shortcuts-dialog").open) $("shortcuts-dialog").showModal();
  }

  // ------------------------------------------------------------------ keys

  let gArmed = 0;
  function typing(ev) {
    const t = ev.target;
    return t && (["INPUT", "SELECT", "TEXTAREA"].includes(t.tagName) || t.isContentEditable);
  }

  function onKey(ev) {
    const k = ev.key;
    if ((ev.metaKey || ev.ctrlKey) && k.toLowerCase() === "k") {
      ev.preventDefault();
      if (dlg().open) closePalette(); else openPalette("");
      return;
    }
    if (ev.metaKey || ev.ctrlKey || ev.altKey || typing(ev) || document.querySelector("dialog[open]")) return;
    if (gArmed && performance.now() - gArmed < 1200) {
      gArmed = 0;
      const v = VIEWS.find((x) => x[3].toLowerCase() === k.toLowerCase());
      if (v) { ev.preventDefault(); ev.stopImmediatePropagation(); go(v[0]); }
      return;
    }
    if (k === "g" || k === "G") { gArmed = performance.now(); ev.preventDefault(); ev.stopImmediatePropagation(); return; }
    if (k === "/") { ev.preventDefault(); openPalette(""); return; }
    if (k === "?") { ev.preventDefault(); openShortcuts(); return; }
    if (k === "]") { ev.preventDefault(); toggleRail(); return; }
    if (k === "o" || k === "O") { ev.preventDefault(); studio().toggleOverlay && studio().toggleOverlay(); return; }
    if ((k === "b" || k === "B") && window.CVPossession && window.CVPossession.data) { ev.preventDefault(); window.CVPossession.toggle(); return; }
    if ((k === "w" || k === "W") && window.CVPossession && window.CVPossession.data) { ev.preventDefault(); window.CVPossession.watch(); return; }
    if (k === "f" || k === "F") { if (!document.body.classList.contains("has-session")) return; ev.preventDefault(); studio().fullscreen && studio().fullscreen(); }
  }

  function boot() {
    $("ask-mod").textContent = MOD;
    $("ask-mod-2").textContent = MOD;
    $("ask-open").addEventListener("click", () => openPalette(""));
    $("ask-input").addEventListener("input", () => { if (P.mode === "answer") $("ask-scope").hidden = true; showList(); });
    $("ask-input").addEventListener("keydown", onPaletteKey);
    dlg().addEventListener("click", (ev) => { if (ev.target === dlg()) closePalette(); });
    window.addEventListener("cv-claude", (ev) => { $("ask-claude-hint").hidden = !ev.detail.available; $("ask-composer-note").textContent = ev.detail.available ? "On this Mac, or with Claude for open questions" : "Answers are built on this Mac from the live game"; });

    document.addEventListener("keydown", onKey, true);
    window.addEventListener("cv-view", () => { requestAnimationFrame(moveIndicator); });
    window.addEventListener("resize", moveIndicator);
    if (window.ResizeObserver) new ResizeObserver(moveIndicator).observe($("side-nav"));
    moveIndicator();

    for (const t of document.querySelectorAll(".rail-tab")) t.addEventListener("click", () => selectRail(t.id));
    document.querySelector(".rail-tabs").addEventListener("keydown", (ev) => {
      const tabs = [...document.querySelectorAll(".rail-tab")];
      const i = tabs.indexOf(document.activeElement);
      const next = { ArrowRight: i + 1, ArrowLeft: i - 1 }[ev.key];
      if (i < 0 || next === undefined) return;
      ev.preventDefault();
      const t = tabs[(next + tabs.length) % tabs.length];
      selectRail(t.id); t.focus();
    });
    const savedRail = cv().store.get("localStorage", "cv-rail");
    if (savedRail && $(savedRail)) selectRail(savedRail);
    if (cv().store.get("localStorage", "cv-rail-hidden") === "1" && window.innerWidth > 1180) toggleRail();
    $("rail-toggle").addEventListener("click", toggleRail);

    $("theme-toggle").addEventListener("click", toggleTheme);
    $("settings-form").addEventListener("submit", () => setTimeout(paintThemeIcon, 0));
    if (window.matchMedia) window.matchMedia("(prefers-color-scheme: light)").addEventListener("change", paintThemeIcon);
    paintThemeIcon();

    $("menu-open").addEventListener("click", openNav);
    $("side-close").addEventListener("click", closeNav);
    $("scrim").addEventListener("click", closeNav);
    $("shortcuts-open").addEventListener("click", openShortcuts);
    $("shortcuts-close").addEventListener("click", () => $("shortcuts-dialog").close());
    for (const d of document.querySelectorAll("dialog.sheet")) d.addEventListener("click", (ev) => { if (ev.target === d) d.close(); });
  }

  window.CVShell = { run, showAnswer, openPalette, go, allowed: () => true };
  if (document.readyState === "complete") boot();
  else document.addEventListener("DOMContentLoaded", boot);
})();
