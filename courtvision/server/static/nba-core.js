/* The preprocessing and CenterNet decoder from perception/student.py, in JavaScript.
 * No learned parameters are changed. frames are oldest first, RGB, ImageNet-normalised. */
((root) => {
  "use strict";
  const W = 960, H = 544, PLANE = W * H, GRID_W = W / 4, GRID_H = H / 4, CELLS = GRID_W * GRID_H;
  const MEAN = [0.485, 0.456, 0.406], STD = [0.229, 0.224, 0.225];
  const lut = MEAN.map((mean, c) => Float32Array.from({ length: 256 }, (_, v) =>
    Math.fround(Math.fround(Math.fround(v / 255) - Math.fround(mean)) / Math.fround(STD[c]))));

  function normalize(frames) {
    const input = new Float32Array(9 * PLANE);
    frames.forEach((frame, i) => {
      const pixels = frame instanceof Uint8ClampedArray ? frame : new Uint8ClampedArray(frame);
      if (pixels.length !== PLANE * 4) throw new Error("Invalid NBA model frame dimensions");
      for (let c = 0; c < 3; c++) {
        const offset = (i * 3 + c) * PLANE, table = lut[c];
        for (let j = 0; j < PLANE; j++) input[offset + j] = table[pixels[j * 4 + c]];
      }
    });
    return input;
  }

  function history(frames, t) {
    return [2, 1, 0].map(back => {
      const target = t - back / 29.97;
      let chosen = frames[0];
      for (const frame of frames) if (Math.abs(frame.t - target) < Math.abs(chosen.t - target)) chosen = frame;
      return chosen;
    });
  }

  function peaks(logits, limit, threshold) {
    const cutoff = Math.log(threshold / (1 - threshold)), candidates = [];
    for (let y = 0; y < GRID_H; y++) for (let x = 0; x < GRID_W; x++) {
      const at = y * GRID_W + x, value = logits[at];
      if (!Number.isFinite(value) || value < cutoff) continue;
      let max = true;
      for (let yy = Math.max(0, y - 1); yy <= Math.min(GRID_H - 1, y + 1) && max; yy++) {
        for (let xx = Math.max(0, x - 1); xx <= Math.min(GRID_W - 1, x + 1); xx++) {
          if (logits[yy * GRID_W + xx] > value) { max = false; break; }
        }
      }
      if (max) candidates.push({ at, x, y, score: 1 / (1 + Math.exp(-value)) });
    }
    return candidates.sort((a, b) => b.score - a.score).slice(0, limit);
  }

  function decode(outputs, contentHeight = 540) {
    const data = name => outputs[name].data || outputs[name], predictions = [];
    const off = data("person_off"), wh = data("person_wh");
    for (const p of peaks(data("person_hm"), 60, .3)) {
      const cx = (p.x + off[p.at]) * 4, cy = (p.y + off[CELLS + p.at]) * 4;
      const w = Math.exp(wh[p.at]) * 4, h = Math.exp(wh[CELLS + p.at]) * 4;
      if ([cx, cy, w, h].every(Number.isFinite)) predictions.push({ class: "person", score: p.score, bbox: [cx - w / 2, cy - h / 2, w, h] });
    }
    const ballOff = data("ball_off"), radius = 12 * contentHeight / 1080;
    for (const p of peaks(data("ball_hm"), 8, .08)) {
      const x = (p.x + ballOff[p.at]) * 4, y = (p.y + ballOff[CELLS + p.at]) * 4;
      if ([x, y].every(Number.isFinite)) predictions.push({ class: "sports ball", score: p.score, bbox: [x - radius, y - radius, radius * 2, radius * 2] });
    }
    return predictions;
  }
  const core = { W, H, normalize, history, decode };
  if (typeof module !== "undefined") module.exports = core;
  else root.CVNBACore = core;
})(typeof self !== "undefined" ? self : typeof window !== "undefined" ? window : null);
