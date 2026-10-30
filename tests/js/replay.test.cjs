const assert = require("node:assert/strict");
const { readFileSync } = require("node:fs");
const { resolve } = require("node:path");
const { test } = require("node:test");
const vm = require("node:vm");

const source = readFileSync(process.env.REPLAY_SOURCE || resolve(__dirname, "../../courtvision/server/static/replay.js"), "utf8");
const settle = () => new Promise(resolve => setImmediate(resolve));

function harness() {
  const listeners = new Map(), elements = new Map(), requests = [], messages = [];
  const element = id => {
    if (!elements.has(id)) elements.set(id, {
      textContent: "", dataset: {}, style: {}, currentTime: 0, paused: true, checked: false,
      addEventListener() {}, getContext: () => ({ setTransform() {}, clearRect() {} }),
    });
    return elements.get(id);
  };
  const window = {
    devicePixelRatio: 1,
    CV: { dispatch: m => messages.push(m), setReplay() {} },
    CVStudio: { workspace: null },
    addEventListener(name, callback) {
      if (!listeners.has(name)) listeners.set(name, []);
      listeners.get(name).push(callback);
    },
  };
  vm.runInNewContext(source, {
    window, document: { getElementById: element }, sessionStorage: { getItem: () => "" }, setTimeout,
    fetch: url => new Promise(resolve => requests.push({ url, respond: body => resolve({
      ok: true, json: async () => body, text: async () => body,
    }) })),
  });
  const emit = (name, detail) => { for (const fn of listeners.get(name) || []) fn({ detail }); };
  return {
    requests, messages,
    select(id, run = "run1") {
      window.CVStudio.workspace = { active: { id, duration_s: 100 }, replay: { run, state: "done", t_done: 100 } };
      emit("cv-studio");
    },
    enter: () => emit("cv-playback-mode", { mode: "replay", was: "live" }),
    leave: () => emit("cv-playback-mode", { mode: "live", was: "replay" }),
    tick: t => emit("cv-playback", { mode: "replay", t }),
  };
}

const state = label => ({ meta: { type: "meta", label }, events: [], insights: [], utterances: [], captions: [] });
const chunk = label => JSON.stringify({ vt: 1, m: { type: "caption", label } }) + "\n";

test("a delayed state from a previous video cannot replace the current game", async () => {
  const h = harness();
  h.select("first"); h.enter();
  h.select("second"); h.enter();
  h.requests[0].respond(state("first"));
  await settle();
  assert.equal(h.messages.length, 0);
  h.requests[1].respond(state("second"));
  await settle();
  assert.deepEqual(h.messages.map(m => m.label).filter(Boolean), ["second"]);
});

test("leaving and reentering replay does not reuse an in-flight state request", async () => {
  const h = harness();
  h.select("first"); h.enter(); h.leave(); h.enter();
  h.requests[0].respond(state("old"));
  await settle();
  assert.equal(h.messages.length, 0);
  h.requests[1].respond(state("current"));
  await settle();
  assert.deepEqual(h.messages.map(m => m.label).filter(Boolean), ["current"]);
});

for (const sameVideo of [false, true]) {
  test(`old chunks cannot overwrite ${sameVideo ? "a newly prepared replay" : "another video's replay"}`, async () => {
    const h = harness();
    h.select("first"); h.enter();
    h.requests[0].respond(state("first"));
    await settle();
    const oldChunk = h.requests.find(r => r.url.endsWith("/chunk/0"));
    assert.ok(oldChunk);
    h.leave(); h.select(sameVideo ? "first" : "second", "run2"); h.enter();
    h.requests.filter(r => r.url.includes("/state?")).at(-1).respond(state("current"));
    await settle();
    const newChunk = h.requests.filter(r => r.url.endsWith("/chunk/0")).at(-1);
    assert.notEqual(newChunk, oldChunk);
    newChunk.respond(chunk("current"));
    await settle();
    oldChunk.respond(chunk("old"));
    await settle();
    h.messages.length = 0;
    h.tick(1);
    assert.deepEqual(h.messages.map(m => m.label).filter(Boolean), ["current"]);
  });
}
