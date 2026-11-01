/* Ordered clip analysis. The decoder is paused and advanced one analysis frame at a time;
 * a result is committed with the exact pixels sent to inference. Replay uses recorded results. */
(() => {
  "use strict";
  class Clip {
    constructor(video, model, callbacks) {
      this.video = video; this.model = model; this.callbacks = callbacks;
      this.decoder = document.createElement("video");
      this.decoder.muted = true; this.decoder.playsInline = true; this.decoder.preload = "auto";
      this.canvas = document.createElement("canvas");
      this.records = []; this.paused = false; this.closed = false; this.done = false; this.wake = null;
    }
    pause() { this.paused = true; this.callbacks.state(); }
    resume() { this.paused = false; this.wake?.(); this.wake = null; this.callbacks.state(); }
    close() {
      this.closed = true; this.wake?.(); this.wake = null;
      this.cancelSeek?.();
      this.decoder.pause(); this.decoder.removeAttribute("src"); this.decoder.load();
    }
    async ready() {
      if (this.paused && !this.closed) await new Promise(resolve => { this.wake = resolve; });
      if (this.closed) throw new Error("Analysis closed");
    }
    async seek(t) {
      const v = this.decoder;
      if (!(v.readyState >= 2 && Math.abs(v.currentTime - t) < 1e-5 && !v.seeking)) {
        await new Promise((resolve, reject) => {
          const finish = error => {
            this.cancelSeek = null;
            clearTimeout(timer); v.removeEventListener("seeked", seeked); v.removeEventListener("loadeddata", loaded); v.removeEventListener("error", failed);
            if (error) reject(error); else resolve();
          };
          const seeked = () => { if (v.readyState >= 2 && !v.seeking) finish(); };
          const loaded = () => { if (Math.abs(v.currentTime-t) < 1e-5) seeked(); else v.currentTime = t; };
          const failed = () => finish(new Error("This browser could not decode the next video frame."));
          const timer = setTimeout(() => finish(new Error("The video decoder stalled. Try MP4 (H.264).")), 15000);
          this.cancelSeek = () => finish(new Error("Analysis closed"));
          v.addEventListener("seeked", seeked); v.addEventListener("loadeddata", loaded); v.addEventListener("error", failed);
          if (v.readyState >= 1) v.currentTime = t;
        });
      }
      if (this.closed) throw new Error("Analysis closed");
      this.canvas.width = v.videoWidth; this.canvas.height = v.videoHeight;
      const ctx = this.canvas.getContext("2d", { willReadFrequently: true });
      ctx.drawImage(v, 0, 0);
      return ctx.getImageData(0, 0, v.videoWidth, v.videoHeight);
    }
    async run() {
      try {
        await this.model.reset();
        this.decoder.src = this.video.currentSrc || this.video.src; this.decoder.load();
        this.callbacks.phase("Finding the court camera using the local pipeline…");
        const calibration = [];
        for (let i = 0; i < 6; i++) {
          await this.ready();
          const pixels = await this.seek(i * Math.min(1, Math.max(0, this.video.duration - .001) / 5));
          calibration.push(pixels);
          if (i === 0) this.callbacks.preview?.(pixels);
        }
        await this.ready();
        await this.model.calibrate(calibration, this.video.videoWidth, this.video.videoHeight);
        this.callbacks.phase("Analysing at 30 frames per second with the local NBA pipeline…");
        const fps = 29.97, count = Math.ceil(this.video.duration * fps);
        for (let index = 0; index < count; index++) {
          await this.ready();
          const t = Math.min(index / fps, Math.max(0, this.video.duration - .001)), raw = await this.seek(t);
          const pixels = new ImageData(raw.data.slice(), raw.width, raw.height);
          const result = await this.model.process(raw, index, t);
          if (this.closed) return;
          this.records.push(result);
          this.callbacks.frame(result, pixels, Math.min(this.video.duration, (index+1)/fps));
        }
        this.done = true;
        this.callbacks.complete(this.records);
      } catch (error) { if (!this.closed) this.callbacks.error(error); }
    }
    at(t) {
      if (!this.records.length) return null;
      let lo = 0, hi = this.records.length-1;
      while (lo < hi) { const mid = Math.ceil((lo+hi)/2); if (this.records[mid].t <= t + .001) lo = mid; else hi = mid-1; }
      return this.records[lo];
    }
  }
  window.CVNBAClip = { create: (video, model, callbacks) => new Clip(video, model, callbacks) };
})();
