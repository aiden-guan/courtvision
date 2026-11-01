/* Only RPC IDs and local pixel buffers cross this worker boundary; nothing is posted to a server. */
(() => {
  "use strict";
  async function load() {
    const worker = new Worker("static/nba-worker.js");
    const pending = new Map();
    let serial = 0;
    worker.onmessage = ({ data }) => {
      const request = pending.get(data.id);
      if (!request) return;
      pending.delete(data.id);
      if (data.error) request.reject(new Error(data.error)); else request.resolve(data.result);
    };
    worker.onerror = () => {
      for (const request of pending.values()) request.reject(new Error("The NBA detector could not start on this device. Reload and retry."));
      pending.clear(); worker.terminate();
    };
    const call = (action, data = {}, transfer = []) => new Promise((resolve, reject) => {
      const id = ++serial; pending.set(id, { resolve, reject });
      worker.postMessage({ id, action, ...data }, transfer);
    });
    let info;
    try { info = await call("load"); }
    catch (error) { worker.terminate(); throw error; }
    window.addEventListener("pagehide", () => worker.terminate(), { once: true });
    return { ...info, detect: (frames, contentHeight) => {
      const buffers = frames.map(frame => frame.rgba.slice().buffer);
      return call("detect", { frames: buffers, contentHeight }, buffers);
    } };
  }
  window.CVNBA = { load };
})();
