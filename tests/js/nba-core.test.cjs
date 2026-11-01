const assert = require("node:assert/strict");
const { test } = require("node:test");
const { normalize, history, decode } = require("../../courtvision/server/static/nba-core.js");

test("NBA preprocessing preserves RGB channel order, oldest-first history, and ImageNet normalization", () => {
  const planes = [0, 1, 2].map(i => {
    const rgba = new Uint8ClampedArray(960 * 544 * 4);
    rgba.set([20 + i, 100 + i, 200 + i, 255]); return rgba;
  });
  const result = normalize(planes), size = 960 * 544;
  for (let i = 0; i < 3; i++) for (let c = 0; c < 3; c++) {
    const expected = Math.fround(Math.fround(Math.fround(planes[i][c] / 255) - Math.fround([.485, .456, .406][c])) / Math.fround([.229, .224, .225][c]));
    assert.equal(result[(i * 3 + c) * size], expected);
  }
});

test("temporal input uses 33ms spacing on 60fps footage and repeats the first frame at startup", () => {
  const frames = Array.from({ length: 7 }, (_, i) => ({ t: i / 60, id: i }));
  assert.deepEqual(history(frames, .1).map(f => f.id), [2, 4, 6]);
  assert.deepEqual(history(frames.slice(0, 1), 0).map(f => f.id), [0, 0, 0]);
});

test("the NBA decoder uses trained heatmaps, offsets, log box sizes, and original thresholds", () => {
  const cells = 240 * 136;
  const out = { ball_hm: new Float32Array(cells).fill(-20), person_hm: new Float32Array(cells).fill(-20),
    ball_off: new Float32Array(cells * 2), person_off: new Float32Array(cells * 2), person_wh: new Float32Array(cells * 2) };
  const person = 10 * 240 + 20, ball = 5 * 240 + 6;
  out.person_hm[person] = 2; out.person_off[person] = .5; out.person_off[cells + person] = .25;
  out.person_wh[person] = Math.log(10); out.person_wh[cells + person] = Math.log(20);
  out.ball_hm[ball] = -2; // sigmoid .119 is above the desktop .08 threshold
  out.ball_off[ball] = .5; out.ball_off[cells + ball] = .5;
  const predictions = decode(out);
  assert.equal(predictions.length, 2);
  assert.equal(predictions[0].class, "person");
  for (const [actual, expected] of predictions[0].bbox.map((n, i) => [n, [62, 1, 40, 80][i]])) assert.ok(Math.abs(actual - expected) < 1e-4);
  assert.equal(predictions[1].class, "sports ball");
  assert.deepEqual(predictions[1].bbox, [20, 16, 12, 12]);
});
