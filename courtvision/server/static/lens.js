/* CourtVision court lens: the live floor as a lit night arena, seen through a perspective camera.
 *
 * Canvas 2D, no dependencies. Court lines come from the court geometry built in app.js (feet: x along
 * the length, y across, z up). Players stand on the floor as pins carrying their jersey number; the ball
 * is drawn at its tracked height with a floor shadow and a short trail, so a 3-D ball flight reads as an
 * arc. Frames arrive at up to 15 Hz and positions are eased between them at the display rate. The camera
 * orbits on drag and settles on presets with a damped spring. The empty-state hero runs the same lens
 * on a scripted possession.
 */
(() => {
  "use strict";

  const DEG = Math.PI / 180;
  const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
  const lerp = (a, b, t) => a + (b - a) * t;
  const smooth = (t) => t * t * (3 - 2 * t);
  const isNum = (v) => typeof v === "number" && Number.isFinite(v);

  const PRESETS = {
    broadcast: { az: -90, el: 33, r: 112 },
    overhead: { az: -90, el: 89, r: 210 },
    baseline: { az: -166, el: 19, r: 74 },
  };
  const TRAIL_MS = 1300;
  const BALL_TRAIL_MS = 1000;
  const HEAD_Z = 5.4;

  const reducedMotion = () => !!(window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches);

  function hexRgb(hex) {
    const m = /^#?([0-9a-f]{6})$/i.exec(String(hex || "").trim());
    if (!m) return null;
    const n = parseInt(m[1], 16);
    return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
  }

  function tint(hex, amount) {
    const c = hexRgb(hex);
    if (!c) return hex;
    const t = c.map((v) => Math.round(v + (255 - v) * amount));
    return `rgb(${t[0]}, ${t[1]}, ${t[2]})`;
  }

  // ------------------------------------------------------------------ the demo possession

  // A pick-and-roll kicked out to the far corner for a three: offence attacks the right basket.
  const DEMO_T = 8.4;
  const DEMO_PLAYERS = [
    ["o1", "home", "7", [[0, 63, 22], [1.0, 65.5, 24], [1.6, 68.5, 29.5], [2.4, 75, 33], [2.75, 77, 33], [3.6, 79, 30], [5.4, 80, 28], [7.2, 66, 23], [8.4, 63, 22]]],
    ["o2", "home", "32", [[0, 74, 35], [0.9, 69.5, 31.5], [1.6, 69.8, 31], [2.6, 78, 28], [3.4, 84, 26], [5.4, 85, 25.5], [7.2, 75, 34], [8.4, 74, 35]]],
    ["o3", "home", "11", [[0, 76, 6], [2.6, 77, 5.5], [4.5, 76, 8], [7.2, 76.5, 6.5], [8.4, 76, 6]]],
    ["o4", "home", "24", [[0, 89.2, 46.6], [3.3, 89.2, 46.6], [4.6, 88.4, 45.6], [7.2, 88.8, 46.2], [8.4, 89.2, 46.6]]],
    ["o5", "home", "8", [[0, 86.5, 11], [2.8, 86, 13], [3.6, 85, 15.5], [5.4, 87, 18], [7.2, 86.8, 11.5], [8.4, 86.5, 11]]],
    ["d1", "away", "14", [[0, 66.5, 22.5], [1.0, 68.5, 24.5], [1.6, 70.5, 27.8], [2.4, 73.5, 30.5], [3.0, 76, 31], [5.4, 78, 29], [7.2, 69, 23.5], [8.4, 66.5, 22.5]]],
    ["d2", "away", "21", [[0, 76.5, 33], [0.9, 72, 31.5], [1.7, 72.5, 33], [2.5, 79, 30.5], [3.3, 83, 27.5], [5.4, 83.5, 26.5], [7.2, 77, 32.5], [8.4, 76.5, 33]]],
    ["d3", "away", "5", [[0, 78.5, 9], [2.6, 79, 10.5], [4.5, 78.5, 9.5], [8.4, 78.5, 9]]],
    ["d4", "away", "33", [[0, 86, 42], [2.2, 83.5, 38], [2.9, 84.5, 40], [3.5, 87.5, 44.2], [4.4, 87.6, 43.8], [7.2, 86.2, 42.2], [8.4, 86, 42]]],
    ["d5", "away", "10", [[0, 85, 15], [2.6, 85, 19], [3.4, 85.5, 22], [5.4, 86, 21], [7.2, 85.2, 15.4], [8.4, 85, 15]]],
  ];
  const RIM = [88.75, 25, 10];

  function track(keys, t) {
    if (t <= keys[0][0]) return [keys[0][1], keys[0][2]];
    for (let i = 1; i < keys.length; i++) {
      const [t1, x1, y1] = keys[i];
      if (t <= t1) {
        const [t0, x0, y0] = keys[i - 1];
        const s = smooth((t - t0) / Math.max(1e-6, t1 - t0));
        return [lerp(x0, x1, s), lerp(y0, y1, s)];
      }
    }
    const last = keys[keys.length - 1];
    return [last[1], last[2]];
  }

  function arc(a, b, s, apex) {
    return [lerp(a[0], b[0], s), lerp(a[1], b[1], s), lerp(a[2], b[2], s) + 4 * apex * s * (1 - s)];
  }

  function demoState(t) {
    const pos = {};
    const players = DEMO_PLAYERS.map(([id, team, jersey, keys]) => {
      const [x, y] = track(keys, t);
      pos[id] = [x, y];
      return { id, team, jersey, name: null, x, y };
    });
    const hand = (id, z) => [pos[id][0] + 0.6, pos[id][1] - 0.4, z];
    const at = (id, tt) => { const [x, y] = track(DEMO_PLAYERS.find((p) => p[0] === id)[3], tt); return [x + 0.6, y - 0.4]; };
    let ball = null, holder = null, ripple = null;
    if (t < 2.7) {
      holder = "o1";
      ball = hand("o1", 0.45 + 2.6 * Math.abs(Math.sin(t * Math.PI * 2.6)));
    } else if (t < 3.25) {
      const a = [...at("o1", 2.7), 4.2], b = [...at("o4", 3.25), 4.6];
      ball = arc(a, b, (t - 2.7) / 0.55, 2.2);
    } else if (t < 3.45) {
      holder = "o4";
      ball = hand("o4", lerp(4.6, 8.4, smooth((t - 3.25) / 0.2)));
    } else if (t < 4.55) {
      const a = [...at("o4", 3.45), 8.4];
      ball = arc(a, RIM, (t - 3.45) / 1.1, 6.6);
    } else if (t < 4.85) {
      ball = [RIM[0], RIM[1], lerp(10, 6.6, (t - 4.55) / 0.3)];
      ripple = (t - 4.55) / 0.9;
    } else if (t < 5.6) {
      const s = (t - 4.85) / 0.75;
      const z = s < 0.4 ? lerp(6.6, 0.4, (s / 0.4) ** 1.6) : 0.4 + 1.9 * Math.sin(((s - 0.4) / 0.6) * Math.PI);
      ball = [lerp(RIM[0], 87.4, s), lerp(RIM[1], 23.6, s), z];
      ripple = (t - 4.55) / 0.9;
    } else if (t >= 7.7) {
      holder = "o1";
      ball = hand("o1", 0.45 + 2.6 * Math.abs(Math.sin(t * Math.PI * 2.6)));
    }
    return {
      players,
      ball: ball ? { x: ball[0], y: ball[1], z: ball[2], visible: true, holder } : { visible: false, holder: null },
      ripple: ripple != null && ripple < 1 ? ripple : null,
    };
  }

  // ------------------------------------------------------------------ the lens

  class CourtLens {
    constructor(canvas, opts = {}) {
      this.canvas = canvas;
      this.g = canvas.getContext("2d");
      this.wrap = opts.wrap || canvas.parentElement;
      this.demo = !!opts.demo;
      this.fitBox = opts.fitBox || null; // [x0, y0, x1, y1] in court feet, framed instead of the whole floor
      this.frameRect = opts.frameRect || null; // [left, top, right, bottom] as fractions of the canvas
      this.padTop = opts.padTop || 0; // css px kept clear for an overlaid header
      this.orbitCenter = opts.center || null;
      this.onHover = opts.onHover || null;
      this.onPick = opts.onPick || null;
      this.presetName = opts.preset || "broadcast";
      const p = opts.camera || PRESETS[this.presetName];
      this.base = { ...p };
      this.cam = { ...p };
      this.tgt = { ...p };
      this.geo = null;
      this.colors = {};
      this.colorKey = "";
      this.w = 1; this.h = 1; this.dpr = 1;
      this.prev = new Map(); this.curr = new Map(); this.tPrev = 0; this.tCurr = 0;
      this.ballPrev = null; this.ballCurr = null; this.holder = null;
      this.trails = new Map(); this.ballTrail = [];
      this.focus = null; this.hoverId = null; this.heads = [];
      this.layer = document.createElement("canvas");
      this.layerKey = "";
      this.running = false; this.visible = true; this.last = 0; this.t0 = performance.now();
      this.drag = null;
      this.readColors();
      if (opts.interactive !== false) this.bindPointer();
      if (window.ResizeObserver) { this.ro = new ResizeObserver(() => this.resize()); this.ro.observe(this.wrap); }
      if (window.IntersectionObserver) {
        this.io = new IntersectionObserver((entries) => {
          this.visible = entries.some((e) => e.isIntersecting);
          if (this.visible) this.wake();
        });
        this.io.observe(canvas);
      }
      if (document.fonts && document.fonts.ready) document.fonts.ready.then(() => { this.layerKey = ""; this.wake(); });
      this.resize();
    }

    // ---------------------------------------------------------------- inputs

    readColors() {
      const cs = getComputedStyle(document.documentElement);
      const v = (n, d) => cs.getPropertyValue(n).trim() || d;
      this.colors = {
        home: v("--home", "#4c8dff"), homeInk: v("--home-ink", "#ffffff"),
        away: v("--away", "#b57bff"), awayInk: v("--away-ink", "#14061f"),
        neutral: v("--neutral-team", "#7d828c"), ball: v("--ball", "#ff8a2a"), rim: v("--rim", "#ff8a2a"),
        floorHi: v("--floor-hi", "#2c1d12"), floorLo: v("--floor-lo", "#0c0907"),
        line: v("--floor-line", "rgba(255,236,214,.62)"), paint: v("--floor-paint", "rgba(255,118,46,.075)"),
        grain: v("--floor-grain", "rgba(255,220,180,.028)"),
        display: v("--display", "sans-serif"), mono: v("--mono", "monospace"),
      };
      this.colorKey = JSON.stringify(this.colors);
      this.layerKey = "";
      this.wake();
    }

    setGeometry(geo) {
      this.geo = geo;
      this.layerKey = "";
      this.wake();
    }

    setFocus(id) {
      this.focus = id || null;
      this.wake();
    }

    setPreset(name) {
      if (!PRESETS[name]) return;
      this.presetName = name;
      this.base = { ...PRESETS[name] };
      Object.assign(this.tgt, this.base);
      this.wake();
    }

    clear() {
      this.prev.clear(); this.curr.clear(); this.trails.clear();
      this.ballPrev = this.ballCurr = null; this.ballTrail = []; this.holder = null;
      this.tPrev = this.tCurr = 0;
      this.wake();
    }

    alpha(now) {
      const span = clamp(this.tCurr - this.tPrev, 33, 260);
      return clamp((now - this.tCurr) / span, 0, 1);
    }

    setFrame(f) {
      if (!f) { this.clear(); return; }
      const now = performance.now();
      const k = this.alpha(now);
      const shown = new Map();
      for (const [id, c] of this.curr) {
        const pv = this.prev.get(id);
        shown.set(id, pv ? { ...c, x: lerp(pv.x, c.x, k), y: lerp(pv.y, c.y, k) } : c);
      }
      const next = new Map();
      for (const p of f.players || []) {
        if (!p || !isNum(p.x) || !isNum(p.y)) continue;
        next.set(p.id, { id: p.id, x: p.x, y: p.y, team: p.team, jersey: p.jersey, name: p.name, speed: p.speed });
        let tr = this.trails.get(p.id);
        const last = tr && tr[tr.length - 1];
        if (!tr || (last && Math.hypot(p.x - last[0], p.y - last[1]) > 10)) { tr = []; this.trails.set(p.id, tr); }
        tr.push([p.x, p.y, now]);
      }
      for (const id of [...this.trails.keys()]) if (!next.has(id)) this.trails.delete(id);
      this.prev = shown;
      this.curr = next;

      const b = f.ball;
      const shownBall = this.ballShown(now, k);
      if (b && b.visible && isNum(b.x) && isNum(b.y)) {
        const z = isNum(b.z) ? Math.max(0, b.z) : 0;
        const lastB = this.ballTrail[this.ballTrail.length - 1];
        if (lastB && Math.hypot(b.x - lastB[0], b.y - lastB[1]) > 25) this.ballTrail = [];
        this.ballTrail.push([b.x, b.y, z, now]);
        this.ballPrev = shownBall && shownBall.visible ? shownBall : null;
        this.ballCurr = { x: b.x, y: b.y, z, visible: true };
      } else {
        this.ballPrev = null;
        this.ballCurr = this.ballCurr ? { ...this.ballCurr, visible: false } : null;
      }
      this.holder = (b && b.holder) || null;
      this.tPrev = this.tCurr || now - 66;
      this.tCurr = now;
      this.wake();
    }

    ballShown(now, k) {
      const c = this.ballCurr;
      if (!c) return null;
      const p = this.ballPrev;
      if (!p || !c.visible) return c;
      const a = k === undefined ? this.alpha(now) : k;
      return { x: lerp(p.x, c.x, a), y: lerp(p.y, c.y, a), z: lerp(p.z, c.z, a), visible: true };
    }

    // ---------------------------------------------------------------- pointer

    bindPointer() {
      const el = this.wrap;
      el.addEventListener("pointerdown", (ev) => {
        if (ev.button !== 0) return;
        this.drag = { x: ev.clientX, y: ev.clientY, az: this.tgt.az, el: this.tgt.el, moved: false, id: ev.pointerId };
      });
      el.addEventListener("pointermove", (ev) => {
        if (this.drag && this.drag.id === ev.pointerId) {
          const dx = ev.clientX - this.drag.x, dy = ev.clientY - this.drag.y;
          if (!this.drag.moved && Math.hypot(dx, dy) > 4) {
            this.drag.moved = true;
            try { el.setPointerCapture(ev.pointerId); } catch { /* released already */ }
            el.classList.add("dragging");
            this.hover(null);
          }
          if (this.drag.moved) {
            this.tgt.az = this.drag.az - dx * 0.32;
            this.tgt.el = clamp(this.drag.el + dy * 0.24, 8, 89);
            this.wake();
          }
          return;
        }
        this.hoverAt(ev);
      });
      const end = (ev) => {
        if (!this.drag || this.drag.id !== ev.pointerId) return;
        const clicked = !this.drag.moved;
        this.drag = null;
        el.classList.remove("dragging");
        if (clicked && this.onPick) {
          const hit = this.hitTest(ev);
          if (hit) this.onPick(hit.id);
        }
      };
      el.addEventListener("pointerup", end);
      el.addEventListener("pointercancel", end);
      el.addEventListener("pointerleave", () => { if (!this.drag) this.hover(null); });
      el.addEventListener("dblclick", () => { Object.assign(this.tgt, this.base); this.wake(); });
    }

    hitTest(ev) {
      const r = this.canvas.getBoundingClientRect();
      const x = (ev.clientX - r.left) * this.dpr, y = (ev.clientY - r.top) * this.dpr;
      let best = null, bestD = Infinity;
      for (const hd of this.heads) {
        const d = Math.hypot(hd.x - x, hd.y - y);
        if (d < hd.r + 5 * this.dpr && d < bestD) { best = hd; bestD = d; }
      }
      return best;
    }

    hoverAt(ev) {
      const hit = this.hitTest(ev);
      this.hover(hit ? hit.id : null, ev);
    }

    hover(id, ev) {
      if (id !== this.hoverId) { this.hoverId = id; this.wake(); }
      this.wrap.style.cursor = id ? "pointer" : "";
      if (!this.onHover) return;
      if (!id) { this.onHover(null); return; }
      const p = this.curr.get(id);
      const hd = this.heads.find((q) => q.id === id);
      if (p && hd) this.onHover(p, hd.x / this.dpr, hd.y / this.dpr - hd.r / this.dpr, ev);
    }

    // ---------------------------------------------------------------- loop

    resize() {
      const r = this.wrap.getBoundingClientRect();
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const w = Math.max(1, Math.round(r.width * dpr)), h = Math.max(1, Math.round(r.height * dpr));
      if (w === this.w && h === this.h && dpr === this.dpr) return;
      this.dpr = dpr; this.w = w; this.h = h;
      this.canvas.width = w; this.canvas.height = h;
      this.layer.width = w; this.layer.height = h;
      this.layerKey = "";
      this.draw(performance.now());
    }

    wake() {
      if (this.running || !this.visible) return;
      this.running = true;
      this.last = performance.now();
      requestAnimationFrame((t) => this.frame(t));
    }

    frame(now) {
      if (!this.visible) { this.running = false; return; }
      const dt = Math.min(0.1, Math.max(0, (now - this.last) / 1000));
      this.last = now;
      const moving = this.stepCamera(dt);
      this.draw(now);
      const settling = now - this.tCurr < 320 || (this.ballTrail.length && now - this.ballTrail[this.ballTrail.length - 1][3] < BALL_TRAIL_MS)
        || [...this.trails.values()].some((tr) => tr.length > 1 && now - tr[tr.length - 1][2] < TRAIL_MS);
      if ((this.demo && !reducedMotion()) || moving || this.drag || settling) requestAnimationFrame((t) => this.frame(t));
      else this.running = false;
    }

    stepCamera(dt) {
      if (this.demo && !reducedMotion()) {
        const t = (performance.now() - this.t0) / 1000;
        this.tgt.az = this.base.az + 9 * Math.sin((t / 26) * 2 * Math.PI);
        this.tgt.el = this.base.el + 3 * Math.sin((t / 33) * 2 * Math.PI);
      }
      const k = reducedMotion() ? 1 : 1 - Math.exp(-dt * 7.5);
      let d = 0;
      for (const key of ["az", "el", "r"]) {
        const diff = this.tgt[key] - this.cam[key];
        this.cam[key] += diff * k;
        d = Math.max(d, Math.abs(diff));
      }
      return d > 0.02;
    }

    // ---------------------------------------------------------------- camera

    camera() {
      const geo = this.geo || { L: 94, W: 50 };
      const c = this.orbitCenter || [geo.L / 2, geo.W / 2, 2.5];
      const az = this.cam.az * DEG, el = this.cam.el * DEG;
      const ca = Math.cos(az), sa = Math.sin(az), ce = Math.cos(el), se = Math.sin(el);
      const back = [ce * ca, ce * sa, se];
      const f = [-back[0], -back[1], -back[2]];
      const r = [-sa, ca, 0];
      const u = [r[1] * f[2] - r[2] * f[1], r[2] * f[0] - r[0] * f[2], r[0] * f[1] - r[1] * f[0]];
      const eye = [c[0] + back[0] * this.cam.r, c[1] + back[1] * this.cam.r, c[2] + back[2] * this.cam.r];
      const cam = { eye, f, r, u, F: 1, cx: 0, cy: 0, se };
      // frame the floor (or the fit box) inside the canvas
      const m = 3.5;
      const box = this.fitBox || [-m, -m, geo.L + m, geo.W + m];
      const pts = [[box[0], box[1], 0], [box[2], box[1], 0], [box[2], box[3], 0], [box[0], box[3], 0]];
      for (const hp of (geo.hoops || [])) {
        if (hp.bbx >= box[0] - 6 && hp.bbx <= box[2] + 6) pts.push([hp.bbx, hp.cy - 3, 13.2], [hp.bbx, hp.cy + 3, 13.2]);
      }
      let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
      for (const p of pts) {
        const q = this.projectWith(cam, p[0], p[1], p[2]);
        x0 = Math.min(x0, q[0]); x1 = Math.max(x1, q[0]); y0 = Math.min(y0, q[1]); y1 = Math.max(y1, q[1]);
      }
      const fr = this.frameRect || [0.03, 0, 0.97, 0.97];
      const top = Math.max(fr[1] * this.h, this.padTop * this.dpr);
      const L = fr[0] * this.w, R = fr[2] * this.w, B = fr[3] * this.h;
      const aw = Math.max(1, R - L), ah = Math.max(1, B - top);
      const F = Math.min(aw / Math.max(1e-6, x1 - x0), ah / Math.max(1e-6, y1 - y0));
      cam.F = F;
      cam.cx = L + aw / 2 - F * (x0 + x1) / 2;
      cam.cy = top + ah / 2 - F * (y0 + y1) / 2;
      return cam;
    }

    projectWith(c, x, y, z) {
      const dx = x - c.eye[0], dy = y - c.eye[1], dz = z - c.eye[2];
      const zc = Math.max(0.5, dx * c.f[0] + dy * c.f[1] + dz * c.f[2]);
      const xc = dx * c.r[0] + dy * c.r[1] + dz * c.r[2];
      const yc = dx * c.u[0] + dy * c.u[1] + dz * c.u[2];
      const k = c.F / zc;
      return [c.cx + xc * k, c.cy - yc * k, zc];
    }

    // ---------------------------------------------------------------- drawing

    path(g, c, pts, z = 0, close = false) {
      g.beginPath();
      pts.forEach(([x, y], i) => {
        const q = this.projectWith(c, x, y, z);
        if (i) g.lineTo(q[0], q[1]); else g.moveTo(q[0], q[1]);
      });
      if (close) g.closePath();
    }

    drawFloor(c) {
      const g = this.layer.getContext("2d");
      const geo = this.geo;
      const C = this.colors, dpr = this.dpr;
      g.clearRect(0, 0, this.w, this.h);
      if (!geo) return;
      const { L, W } = geo;
      const ctr = this.projectWith(c, L / 2, W / 2, 0);
      const corner = this.projectWith(c, 0, 0, 0), corner2 = this.projectWith(c, L, W, 0);
      const radius = Math.max(Math.hypot(corner[0] - ctr[0], corner[1] - ctr[1]), Math.hypot(corner2[0] - ctr[0], corner2[1] - ctr[1]));

      // apron, then the lit floor
      const m = 4.5;
      this.path(g, c, [[-m, -m], [L + m, -m], [L + m, W + m], [-m, W + m]], 0, true);
      g.fillStyle = C.floorLo;
      g.fill();
      const lit = g.createRadialGradient(ctr[0], ctr[1] - radius * 0.05, radius * 0.04, ctr[0], ctr[1], radius * 1.05);
      lit.addColorStop(0, tint(C.floorHi, 0.06));
      lit.addColorStop(0.5, C.floorHi);
      lit.addColorStop(1, C.floorLo);
      this.path(g, c, [[0, 0], [L, 0], [L, W], [0, W]], 0, true);
      g.fillStyle = lit;
      g.fill();

      // maple boards run the length of the floor
      g.save();
      this.path(g, c, [[0, 0], [L, 0], [L, W], [0, W]], 0, true);
      g.clip();
      g.lineWidth = Math.max(1, dpr * 0.8);
      for (let y = 0.9, i = 0; y < W; y += 1.35, i++) {
        g.strokeStyle = C.grain;
        g.globalAlpha = i % 3 === 0 ? 1 : 0.55;
        this.path(g, c, [[0, y], [L, y]]);
        g.stroke();
      }
      g.globalAlpha = 1;
      g.restore();

      // paint and the centre circle
      g.fillStyle = C.paint;
      for (const poly of geo.paint) { this.path(g, c, poly, 0, true); g.fill(); }
      const cc = [];
      for (let i = 0; i <= 48; i++) { const a = (i / 48) * 2 * Math.PI; cc.push([L / 2 + 6 * Math.cos(a), W / 2 + 6 * Math.sin(a)]); }
      this.path(g, c, cc, 0, true);
      g.fill();

      // lines
      g.lineJoin = "round";
      g.lineCap = "round";
      g.strokeStyle = C.line;
      g.lineWidth = Math.max(1, dpr * 1.15);
      for (const pl of geo.lines) { this.path(g, c, pl); g.stroke(); }
      g.setLineDash([3 * dpr, 4 * dpr]);
      for (const pl of geo.dashes) { this.path(g, c, pl); g.stroke(); }
      g.setLineDash([]);

      // arena falloff
      const fall = g.createRadialGradient(ctr[0], ctr[1], radius * 0.45, ctr[0], ctr[1], radius * 1.35);
      fall.addColorStop(0, "rgba(5,6,7,0)");
      fall.addColorStop(1, "rgba(5,6,7,0.62)");
      g.fillStyle = fall;
      g.fillRect(0, 0, this.w, this.h);

      for (const hp of geo.hoops || []) this.drawHoop(g, c, hp);
    }

    drawHoop(g, c, hp) {
      const C = this.colors, dpr = this.dpr;
      const P = (x, y, z) => this.projectWith(c, x, y, z);
      const line = (pts) => { g.beginPath(); pts.forEach((p, i) => (i ? g.lineTo(p[0], p[1]) : g.moveTo(p[0], p[1]))); };
      const sx = hp.bbx - hp.dir * 3.2;
      // stanchion and arm
      g.strokeStyle = "rgba(255,255,255,0.16)";
      g.lineWidth = 2 * dpr;
      line([P(sx, hp.cy, 0), P(sx, hp.cy, 11.2), P(hp.bbx, hp.cy, 11.2)]);
      g.stroke();
      // backboard and its square
      const half = hp.bbHalf;
      const board = [P(hp.bbx, hp.cy - half, 9), P(hp.bbx, hp.cy + half, 9), P(hp.bbx, hp.cy + half, 12.5), P(hp.bbx, hp.cy - half, 12.5)];
      line(board);
      g.closePath();
      g.fillStyle = "rgba(220,235,255,0.07)";
      g.fill();
      g.strokeStyle = "rgba(255,255,255,0.55)";
      g.lineWidth = Math.max(1, 1.2 * dpr);
      g.stroke();
      line([P(hp.bbx, hp.cy - 1, 10), P(hp.bbx, hp.cy + 1, 10), P(hp.bbx, hp.cy + 1, 11.5), P(hp.bbx, hp.cy - 1, 11.5)]);
      g.closePath();
      g.stroke();
      // net, then the rim on top
      const ring = [], low = [];
      for (let i = 0; i <= 24; i++) {
        const a = (i / 24) * 2 * Math.PI;
        ring.push(P(hp.bx + hp.rimR * Math.cos(a), hp.cy + hp.rimR * Math.sin(a), 10));
        low.push(P(hp.bx + hp.rimR * 0.55 * Math.cos(a), hp.cy + hp.rimR * 0.55 * Math.sin(a), 8.5));
      }
      g.strokeStyle = "rgba(255,255,255,0.28)";
      g.lineWidth = Math.max(0.75, 0.8 * dpr);
      for (let i = 0; i < 24; i += 2) { line([ring[i], low[(i + 2) % 24]]); g.stroke(); line([ring[i], low[(i + 22) % 24]]); g.stroke(); }
      line(low);
      g.stroke();
      line([P(hp.bbx, hp.cy, 10), P(hp.bx - hp.dir * hp.rimR, hp.cy, 10)]);
      g.strokeStyle = C.rim;
      g.lineWidth = Math.max(1.5, 1.8 * dpr);
      g.stroke();
      line(ring);
      g.shadowColor = C.rim;
      g.shadowBlur = 8 * dpr;
      g.stroke();
      g.shadowBlur = 0;
    }

    draw(now) {
      const g = this.g, dpr = this.dpr;
      const c = this.camera();
      const key = `${this.w}x${this.h}|${this.cam.az.toFixed(2)}|${this.cam.el.toFixed(2)}|${this.cam.r.toFixed(1)}|${this.colorKey}|${this.geo ? this.geo.L : 0}`;
      if (key !== this.layerKey) { this.drawFloor(c); this.layerKey = key; }
      g.clearRect(0, 0, this.w, this.h);
      g.drawImage(this.layer, 0, 0);
      if (!this.geo) return;

      let players, ball, holder, ripple = null, trailFor, ballTrail;
      if (this.demo) {
        const t = reducedMotion() ? 4.0 : ((now - this.t0) / 1000) % DEMO_T;
        const st = demoState(t);
        players = st.players; ball = st.ball; holder = st.ball.holder; ripple = st.ripple;
        const past = [];
        for (let i = 9; i >= 1; i--) { const tt = t - i * 0.13; if (tt >= 0) past.push([demoState(tt), i * 130]); }
        trailFor = (p) => past.map(([s, age]) => { const q = s.players.find((x) => x.id === p.id); return [q.x, q.y, age]; });
        ballTrail = [];
        if (ball.visible) for (let i = 12; i >= 1; i--) { const tt = t - i * 0.07; if (tt < 0) continue; const b = demoState(tt).ball; if (b.visible) ballTrail.push([b.x, b.y, b.z, i * 70]); }
      } else {
        const k = this.alpha(now);
        players = [];
        for (const [id, cur] of this.curr) {
          const pv = this.prev.get(id);
          players.push(pv ? { ...cur, x: lerp(pv.x, cur.x, k), y: lerp(pv.y, cur.y, k) } : cur);
        }
        ball = this.ballShown(now, k);
        holder = this.holder;
        trailFor = (p) => (this.trails.get(p.id) || []).filter((q) => now - q[2] < TRAIL_MS).map((q) => [q[0], q[1], now - q[2]]);
        ballTrail = this.ballTrail.filter((q) => now - q[3] < BALL_TRAIL_MS).map((q) => [q[0], q[1], q[2], now - q[3]]);
        this.ballTrail = this.ballTrail.filter((q) => now - q[3] < BALL_TRAIL_MS);
      }

      const C = this.colors;
      const teamColor = (t) => (t === "home" ? C.home : t === "away" ? C.away : C.neutral);
      const dim = (id) => this.focus && id !== this.focus;

      // trails on the floor
      g.lineCap = "round";
      for (const p of players) {
        const tr = trailFor(p);
        if (tr.length < 1) continue;
        const pts = [...tr, [p.x, p.y, 0]];
        g.strokeStyle = teamColor(p.team);
        g.lineWidth = 2 * dpr;
        for (let i = 1; i < pts.length; i++) {
          const a = this.projectWith(c, pts[i - 1][0], pts[i - 1][1], 0), b = this.projectWith(c, pts[i][0], pts[i][1], 0);
          g.globalAlpha = (1 - pts[i - 1][2] / TRAIL_MS) * (dim(p.id) ? 0.12 : 0.42);
          g.beginPath(); g.moveTo(a[0], a[1]); g.lineTo(b[0], b[1]); g.stroke();
        }
      }
      g.globalAlpha = 1;

      // shadows
      for (const p of players) {
        const q = this.projectWith(c, p.x, p.y, 0);
        const s = (c.F * 1.5) / q[2];
        g.save();
        g.translate(q[0], q[1]);
        g.scale(1, Math.max(0.22, c.se));
        const sh = g.createRadialGradient(0, 0, 0, 0, 0, s);
        sh.addColorStop(0, "rgba(0,0,0,0.55)");
        sh.addColorStop(1, "rgba(0,0,0,0)");
        g.fillStyle = sh;
        g.beginPath(); g.arc(0, 0, s, 0, 2 * Math.PI); g.fill();
        g.restore();
      }
      if (ball && ball.visible) {
        const q = this.projectWith(c, ball.x, ball.y, 0);
        const lift = clamp(ball.z / 16, 0, 1);
        const s = ((c.F * 0.85) / q[2]) * (1 + lift * 1.4);
        g.save();
        g.translate(q[0], q[1]);
        g.scale(1, Math.max(0.22, c.se));
        const sh = g.createRadialGradient(0, 0, 0, 0, 0, s);
        sh.addColorStop(0, `rgba(0,0,0,${0.6 * (1 - lift * 0.7)})`);
        sh.addColorStop(1, "rgba(0,0,0,0)");
        g.fillStyle = sh;
        g.beginPath(); g.arc(0, 0, s, 0, 2 * Math.PI); g.fill();
        g.restore();
      }

      // players and the ball, far to near
      const items = players.map((p) => ({ kind: "p", p, depth: this.projectWith(c, p.x, p.y, 0)[2] }));
      if (ball && ball.visible) items.push({ kind: "b", depth: this.projectWith(c, ball.x, ball.y, ball.z)[2] - 0.5 });
      items.sort((a, b) => b.depth - a.depth);
      this.heads = [];
      for (const it of items) {
        if (it.kind === "b") this.drawBall(g, c, ball, ballTrail);
        else this.drawPlayer(g, c, it.p, teamColor(it.p.team), holder === it.p.id, dim(it.p.id));
      }

      if (ripple != null) {
        const q = this.projectWith(c, RIM[0], RIM[1], RIM[2]);
        const rr = (c.F * (0.8 + ripple * 3.4)) / q[2];
        g.save();
        g.translate(q[0], q[1]);
        g.scale(1, Math.max(0.22, c.se));
        g.strokeStyle = C.ball;
        g.globalAlpha = 1 - ripple;
        g.lineWidth = 2 * dpr;
        g.shadowColor = C.ball; g.shadowBlur = 14 * dpr;
        g.beginPath(); g.arc(0, 0, rr, 0, 2 * Math.PI); g.stroke();
        g.restore();
      }
    }

    drawPlayer(g, c, p, color, holder, dimmed) {
      const dpr = this.dpr, C = this.colors;
      const foot = this.projectWith(c, p.x, p.y, 0);
      const head = this.projectWith(c, p.x, p.y, HEAD_Z);
      const s = (c.F * 1.25) / foot[2];
      const R = clamp((c.F * 1.15) / head[2], 7 * dpr, 17 * dpr);
      const hovered = this.hoverId === p.id;
      g.save();
      g.globalAlpha = dimmed ? 0.28 : 1;
      // floor ring
      g.save();
      g.translate(foot[0], foot[1]);
      g.scale(1, Math.max(0.22, c.se));
      g.strokeStyle = color;
      g.lineWidth = 1.6 * dpr;
      g.shadowColor = color; g.shadowBlur = 10 * dpr;
      g.beginPath(); g.arc(0, 0, s, 0, 2 * Math.PI); g.stroke();
      g.restore();
      // stem
      if (Math.hypot(head[0] - foot[0], head[1] - foot[1]) > R * 0.6) {
        const grad = g.createLinearGradient(foot[0], foot[1], head[0], head[1]);
        grad.addColorStop(0, "rgba(255,255,255,0)");
        grad.addColorStop(1, color);
        g.strokeStyle = grad;
        g.lineWidth = 1.4 * dpr;
        g.beginPath(); g.moveTo(foot[0], foot[1]); g.lineTo(head[0], head[1]); g.stroke();
      }
      // holder ring
      if (holder) {
        g.strokeStyle = C.ball;
        g.lineWidth = 2 * dpr;
        g.shadowColor = C.ball; g.shadowBlur = 12 * dpr;
        g.beginPath(); g.arc(head[0], head[1], R + 4 * dpr, 0, 2 * Math.PI); g.stroke();
        g.shadowBlur = 0;
      }
      // head badge
      const fill = g.createRadialGradient(head[0] - R * 0.35, head[1] - R * 0.45, R * 0.1, head[0], head[1], R);
      fill.addColorStop(0, tint(color, 0.35));
      fill.addColorStop(1, color);
      g.shadowColor = color; g.shadowBlur = (hovered ? 18 : 10) * dpr;
      g.fillStyle = fill;
      g.beginPath(); g.arc(head[0], head[1], R, 0, 2 * Math.PI); g.fill();
      g.shadowBlur = 0;
      g.strokeStyle = hovered ? "#ffffff" : "rgba(255,255,255,0.75)";
      g.lineWidth = (hovered ? 2 : 1.25) * dpr;
      g.stroke();
      const label = p.jersey != null && p.jersey !== "" ? String(p.jersey) : "";
      if (label) {
        g.fillStyle = p.team === "home" ? C.homeInk : p.team === "away" ? C.awayInk : "#ffffff";
        g.font = `700 ${Math.round(R * (label.length > 2 ? 0.8 : 1.02))}px ${C.display}`;
        g.textAlign = "center";
        g.textBaseline = "middle";
        g.fillText(label, head[0], head[1] + R * 0.06);
      } else {
        g.fillStyle = "rgba(255,255,255,0.9)";
        g.beginPath(); g.arc(head[0], head[1], R * 0.22, 0, 2 * Math.PI); g.fill();
      }
      g.restore();
      this.heads.push({ id: p.id, x: head[0], y: head[1], r: R });
    }

    drawBall(g, c, b, trail) {
      const dpr = this.dpr, C = this.colors;
      const q = this.projectWith(c, b.x, b.y, b.z);
      const rb = clamp((c.F * 0.62) / q[2], 3.4 * dpr, 9 * dpr);
      // drop line to the shadow
      if (b.z > 1.5) {
        const f = this.projectWith(c, b.x, b.y, 0);
        g.save();
        g.setLineDash([2 * dpr, 3 * dpr]);
        g.strokeStyle = "rgba(255,138,42,0.32)";
        g.lineWidth = dpr;
        g.beginPath(); g.moveTo(f[0], f[1]); g.lineTo(q[0], q[1]); g.stroke();
        g.restore();
      }
      // flight trail
      if (trail && trail.length) {
        const pts = [...trail, [b.x, b.y, b.z, 0]];
        g.save();
        g.globalCompositeOperation = "lighter";
        g.lineCap = "round";
        g.strokeStyle = C.ball;
        for (let i = 1; i < pts.length; i++) {
          const a = this.projectWith(c, pts[i - 1][0], pts[i - 1][1], pts[i - 1][2]);
          const d = this.projectWith(c, pts[i][0], pts[i][1], pts[i][2]);
          const life = 1 - pts[i - 1][3] / BALL_TRAIL_MS;
          g.globalAlpha = clamp(life, 0, 1) * 0.75;
          g.lineWidth = (0.6 + life * 2.8) * dpr;
          g.beginPath(); g.moveTo(a[0], a[1]); g.lineTo(d[0], d[1]); g.stroke();
        }
        g.restore();
      }
      // glow and the ball
      const glow = g.createRadialGradient(q[0], q[1], 0, q[0], q[1], rb * 3.6);
      glow.addColorStop(0, "rgba(255,138,42,0.5)");
      glow.addColorStop(1, "rgba(255,138,42,0)");
      g.fillStyle = glow;
      g.beginPath(); g.arc(q[0], q[1], rb * 3.6, 0, 2 * Math.PI); g.fill();
      const core = g.createRadialGradient(q[0] - rb * 0.35, q[1] - rb * 0.4, rb * 0.05, q[0], q[1], rb);
      core.addColorStop(0, "#fff6ea");
      core.addColorStop(0.35, "#ffb066");
      core.addColorStop(0.8, C.ball);
      core.addColorStop(1, "#b9440f");
      g.fillStyle = core;
      g.beginPath(); g.arc(q[0], q[1], rb, 0, 2 * Math.PI); g.fill();
      if (b.z >= 4) {
        g.font = `500 ${Math.round(10.5 * dpr)}px ${C.mono}`;
        g.textAlign = "left";
        g.textBaseline = "middle";
        g.fillStyle = "rgba(255,255,255,0.78)";
        g.fillText(`${b.z.toFixed(1)} ft`, q[0] + rb + 6 * dpr, q[1] - rb * 0.6);
      }
    }
  }

  CourtLens.PRESETS = PRESETS;
  window.CourtLens = CourtLens;
})();
