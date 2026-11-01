const assert = require("node:assert/strict");
const { readFileSync } = require("node:fs");
const { resolve } = require("node:path");
const { test } = require("node:test");
const vm = require("node:vm");
const source = readFileSync(resolve(__dirname, "../../courtvision/server/static/visitor.js"), "utf8");
const settle = () => new Promise(resolve => setImmediate(resolve));

function harness(detector, duration = 30) {
  let now = 0, serial = 0, rendered, detectCalls = 0, analysis;
  const elements = new Map(), timers = new Map(), requests = [], messages = [], revoked = [], blobs = new Map();
  const element = id => {
    if (!elements.has(id)) {
      const callbacks = new Map();
      const el = { textContent: "", dataset: {}, children: [], hidden: false, style: {}, currentTime: 0,
        paused: true, ended: false, readyState: 2, seeking: false, videoWidth: 1280, videoHeight: 720, duration,
        playbackRate: 1, checked: true, open: false,
        addEventListener(name, fn) { if (!callbacks.has(name)) callbacks.set(name, []); callbacks.get(name).push(fn); },
        emit(name) { for (const fn of callbacks.get(name) || []) fn({}); },
        pause() { this.paused = true; }, play() { this.paused = false; this.emit("play"); return Promise.resolve(); },
        load() { if (id.startsWith("new-video") && this.src) queueMicrotask(() => this.emit("loadedmetadata")); },
        removeAttribute(name) { delete this[name]; }, setAttribute() {}, closest() { return null; }, close() { this.open = false; },
        classList: { add() {}, remove() {} },
        focus() {}, getContext: () => ({ drawImage() {}, strokeRect() {}, fillRect() {}, fillText() {}, measureText: () => ({ width: 30 }),
          beginPath() {}, arc() {}, ellipse() {}, stroke() {}, moveTo() {}, lineTo() {}, putImageData() {},
          getImageData: () => ({ data: new Uint8ClampedArray(960 * 544 * 4) }) }),
        click() {},
      };
      elements.set(id, el);
    }
    return elements.get(id);
  };
  const apiData = { public_demo: true, active: { id: "demo", run_id: "run", demo: true, duration_s: 180 },
    status: { t: 1, paused: false }, frame: null, model: { state: "idle" }, videos: [] };
  const window = {
    addEventListener() {},
    CVNBA: { load: async () => ({ checkpoint: "original-nba-weights", dispose() {} }) },
    CVNBAClip: { create: (video, model, callbacks) => {
      analysis = { paused: false, records: [],
        pause() { this.paused = true; callbacks.state(); }, resume() { this.paused = false; callbacks.state(); }, close() {},
        at: () => analysis.records[0],
        async run() {
          detectCalls++;
          const result = await (detector ? detector() : new Promise(() => {}));
          this.records.push(result);
          callbacks.frame(result, { width: 1280, height: 720 }, result.t);
        },
        complete() { callbacks.complete(this.records); },
      };
      return analysis;
    } },
    CourtLens: class { clear() {} setGeometry() {} resize() {} setFrame() {} readColors() {} },
    CV: { dispatch: msg => messages.push(msg), selectView() {}, reconnect() {}, toast() {}, courtGeometry: () => ({}),
      S: { court: {}, events: new Map() }, api: async (path, body) => { requests.push({ path, body }); return apiData; } },
    CVStudio: { renderWorkspace: data => { rendered = data; }, receiveStatus() {}, receiveVision() {}, openDialog() {} },
  };
  vm.runInNewContext(source, { window,
    document: { hidden: false, body: { dataset: {} }, getElementById: element, createElement: tag => element(`new-${tag}-${++serial}`),
      querySelector: element, querySelectorAll: () => [] },
    URL: { createObjectURL: obj => { const url = `blob:${++serial}`; blobs.set(url, obj); return url; }, revokeObjectURL: url => revoked.push(url) },
    Blob, AbortController, performance: { now: () => now },
    setTimeout: (fn, ms) => { const id = ++serial; timers.set(id, { fn, at: now + ms }); return id; }, clearTimeout: id => timers.delete(id),
  });
  const visitor = window.CVVisitor;
  visitor.workspaceUpdate(apiData);
  return { visitor, element, requests, messages, revoked, blobs,
    get rendered() { return rendered; }, get detectCalls() { return detectCalls; },
    get analysis() { return analysis; },
    async advance(ms) { now += ms; for (const [id, task] of [...timers]) if (task.at <= now) { timers.delete(id); task.fn(); } await settle(); },
  };
}

test("visitors start empty and never autojoin the shared game", () => {
  const h = harness();
  assert.equal(h.visitor.mode, "idle");
  assert.equal(h.rendered.active, null);
  assert.equal(h.visitor.receive({ type: "frame", t: 99 }), true);
  assert.equal(h.messages.filter(m => m.type === "frame").length, 0);
  assert.equal(h.requests.length, 0);
});

test("demo pause buffers frames and resume preserves their playback timing", async () => {
  const h = harness();
  await h.visitor.startDemo();
  assert.equal(h.requests[0].path, "api/studio/start");
  h.visitor.receive({ type: "frame", t: 1 });
  await h.visitor.control("pause");
  await h.advance(1000);
  h.visitor.receive({ type: "frame", t: 2 });
  await h.advance(2000);
  assert.equal(h.messages.filter(m => m.type === "frame").at(-1).t, 1);
  await h.visitor.control("resume");
  assert.equal(h.messages.filter(m => m.type === "frame").at(-1).t, 1);
  await h.advance(1000);
  assert.equal(h.messages.filter(m => m.type === "frame").at(-1).t, 2);
  assert.equal(h.requests.length, 1); // controls never change the shared server's playback
});

test("exiting a demo discards pending playback and ignores subsequent shared messages", async () => {
  const h = harness(); await h.visitor.startDemo(); await h.visitor.control("pause");
  h.visitor.receive({ type: "caption", text: "old demo" });
  await h.visitor.exit();
  h.visitor.receive({ type: "frame", t: 100 });
  await h.advance(10000);
  assert.equal(h.rendered.active, null);
  assert.equal(h.messages.some(m => m.text === "old demo" || m.t === 100), false);
});

test("local files use blob URLs and switching sessions releases the previous file", async () => {
  const h = harness(), file = { name: "My game.mp4" };
  await h.visitor.openFile(file);
  const firstUrl = h.element("local-video").src;
  assert.equal(h.blobs.get(firstUrl), file);
  assert.equal(h.rendered.active.browser, true);
  h.visitor.receive({ type: "event", event: { event_id: "synthetic" } });
  h.visitor.workspaceUpdate({ public_demo: true, active: { demo: true } });
  assert.equal(h.rendered.active.name, file.name);
  assert.equal(h.messages.some(m => m.event?.event_id === "synthetic"), false);
  await h.visitor.openFile({ name: "Another.mp4" });
  assert.ok(h.revoked.includes(firstUrl));
  assert.equal(h.requests.length, 0); // no upload, prepare, or analysis request
});

test("clips over 30 seconds are rejected before opening a session or loading a model", async () => {
  const h = harness(null, 30.01);
  await h.visitor.openFile({ name: "Too long.mp4" });
  assert.equal(h.visitor.mode, "idle");
  assert.equal(h.rendered.active, null);
  assert.match(h.element("studio-upload-status").textContent, /30 seconds or shorter/);
  assert.equal(h.requests.length, 0);
  assert.equal(h.blobs.size, h.revoked.length);
});

test("a clip exactly 30 seconds long is accepted", async () => {
  const h = harness(null, 30);
  await h.visitor.openFile({ name: "Valid.mp4" });
  assert.equal(h.visitor.mode, "video");
});

test("in-flight detections from a closed video cannot enter the next video's results", async () => {
  let resolveDetection;
  const h = harness(() => new Promise(resolve => { resolveDetection = resolve; }));
  await h.visitor.openFile({ name: "First.mp4" });
  h.element("local-video").emit("loadedmetadata"); await settle();
  await h.advance(200);
  assert.equal(h.detectCalls, 1);
  await h.visitor.openFile({ name: "Second.mp4" });
  resolveDetection({ t: 0, detections: [{ id: "old-track", class: "person", confidence: .9, bbox: [10, 10, 30, 40] }] }); await settle();
  assert.equal(h.element("local-samples").textContent, 0);
  assert.equal(h.rendered.vision, null);
});

test("analysis pause controls the ordered processor; replay seeking uses recorded tracks", async () => {
  let resolveDetection;
  const h = harness(() => new Promise(resolve => { resolveDetection = resolve; }));
  await h.visitor.openFile({ name: "Game.mp4" }); h.element("local-video").emit("loadedmetadata"); await settle();
  await h.visitor.togglePlay();
  assert.equal(h.analysis.paused, true);
  await h.visitor.togglePlay();
  assert.equal(h.analysis.paused, false);
  resolveDetection({ t: 5, detections: [{ id: "home-t1", class: "person", confidence: .9, bbox: [10, 10, 30, 40] }] }); await settle();
  assert.equal(h.element("local-samples").textContent, 1);
  assert.equal(h.rendered.vision.people, 1);
  h.analysis.complete();
  h.element("local-video").currentTime = 5;
  h.element("local-video").emit("seeked");
  assert.equal(h.rendered.vision.tracks[0].id, "home-t1");
  assert.equal(h.detectCalls, 1);
  h.element("local-video").currentTime = 10;
  h.element("local-video").emit("seeked");
  assert.equal(h.rendered.vision.people, 0); // no stale boxes on a different source frame
});
