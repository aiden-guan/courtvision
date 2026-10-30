/* How the feed plays. Live: the pipeline analyses the video as it plays, at its own pace (slower than real time with
   the NBA detector). Watch: the recording itself at full speed with the game's possession analysis
   (static/possession.js). Replay: the recording at full speed with the live analysis done ahead of time
   (static/replay.js). Watch and Replay play the file in a <video> over the live frame, pause the live analysis
   meanwhile, and send `cv-playback` with the time playing; `cv-playback-mode` says when the mode changes. */
(() => {
  "use strict";
  const $ = id => document.getElementById(id);
  const MODES = {
    live: { key: "", on: "Analyse the video as it plays: every frame, at the analysis's own pace" },
    watch: { key: "W", on: "The recording at full speed with the game's possession analysis",
      off: "No possession analysis for this video: it has to be analysed as an NBA game first (docs/NBA.md)" },
    replay: { key: "R", on: "The recording at full speed with the analysis prepared ahead of time",
      off: "No replay yet: prepare one under the feed" },
  };
  let mode = "live", resumeLive = false, raf = 0, beat = 0, lastTick = 0;
  const token = () => { try { return sessionStorage.getItem("cv-token") || ""; } catch { return ""; } };
  const video = () => $("watch-video");
  const uploaded = () => { const a = window.CVStudio?.workspace?.active; return a && !a.demo ? a : null; };
  const can = m => m === "live" || (m === "watch" ? !!window.CVPossession?.data : !!window.CVReplay?.available);
  const emit = (name, detail) => window.dispatchEvent(new CustomEvent(name, { detail }));

  function setMode(next) {
    const active = uploaded();
    if (next === mode || (next !== "live" && !(active && can(next)))) return;
    const v = video(), was = mode;
    mode = next;
    document.body.classList.toggle("watching", next !== "live");
    document.body.dataset.playback = next;
    cancelAnimationFrame(raf);
    clearInterval(beat);
    if (next === "live") {
      v.pause(); v.removeAttribute("src"); v.load(); v.hidden = true; delete v.dataset.video;
      if (resumeLive) window.CV?.control?.("resume");
      resumeLive = false;
    } else {
      if (was === "live") {  // pick the recording up where the live analysis is
        const live = window.CVStudio.status;
        resumeLive = !!live && !live.paused && !live.finished;
        if (resumeLive) window.CV?.control?.("pause");
        const start = window.CVStudio?.processedT || 0;
        v.dataset.video = active.id;
        v.src = `api/studio/video/${active.id}${token() ? `?token=${encodeURIComponent(token())}` : ""}`;
        v.addEventListener("loadedmetadata", () => { v.currentTime = start; v.play().catch(() => { /* the controls start it */ }); }, { once: true });
        v.hidden = false;
      }
      const tick = () => { lastTick = performance.now(); emit("cv-playback", { mode, t: v.currentTime }); };
      const frame = () => { if (mode === "live") return; tick(); raf = requestAnimationFrame(frame); };
      raf = requestAnimationFrame(frame);
      // animation frames stop in a background or hidden page while the video plays on: a slow timer keeps up
      clearInterval(beat);
      beat = setInterval(() => {
        if (mode === "live") clearInterval(beat);
        else if (performance.now() - lastTick > 300) tick();
      }, 250);
    }
    emit("cv-playback-mode", { mode: next, was });
    render();
  }

  function toggle(m) { setMode(mode === m ? "live" : m); }

  function render() {
    const active = uploaded();
    $("mode-switch").hidden = !active;
    for (const b of $("mode-switch").querySelectorAll("[data-mode]")) {
      const m = b.dataset.mode, ok = can(m), info = MODES[m];
      b.setAttribute("aria-pressed", String(mode === m));
      b.disabled = !ok && mode !== m;
      b.title = (ok ? info.on : info.off) + (info.key ? ` (${info.key})` : "");
      if (m === "replay") b.textContent = window.CVReplay?.label || "Replay";
    }
  }

  window.CVPlayback = {
    get mode() { return mode; }, setMode, toggle, can, render,
    seek(t) { if (mode !== "live") video().currentTime = Math.max(0, t); },
    get time() { return mode === "live" ? window.CVStudio?.processedT : video().currentTime; },
  };
  for (const b of $("mode-switch").querySelectorAll("[data-mode]")) b.addEventListener("click", () => setMode(b.dataset.mode));
  video().addEventListener("seeked", () => { if (mode !== "live") emit("cv-playback", { mode, t: video().currentTime, seeked: true }); });
  video().addEventListener("error", () => {
    if (mode === "live") return;
    setMode("live");
    window.CV?.toast?.("This browser cannot play this recording. The live view still works.", "error");
  });
  window.addEventListener("cv-studio", () => {
    // a mode whose source went away (another video opened, the analysis removed) falls back to live
    const active = uploaded();
    if (mode !== "live" && (!active || active.id !== video().dataset.video || !can(mode))) setMode("live");
    render();
  });
})();
