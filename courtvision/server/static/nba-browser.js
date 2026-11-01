/* Only RPC IDs and local pixel buffers cross this worker boundary; nothing is posted to a server. */
(() => {
  "use strict";
  async function load({ signal, progress = () => {} } = {}) {
    const worker = new Worker("static/nba-worker.js");
    const pending = new Map();
    let serial = 0;
    worker.onmessage = ({ data }) => {
      if (data.loading) { progress(data.loading); return; }
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
    const dispose = () => {
      for (const request of pending.values()) request.reject(new Error("Analysis closed"));
      pending.clear(); worker.terminate();
      signal?.removeEventListener("abort", dispose);
    };
    signal?.addEventListener("abort", dispose, { once: true });
    if (signal?.aborted) { dispose(); throw new Error("Analysis closed"); }
    let info;
    try { info = await call("load"); }
    catch (error) { dispose(); throw error; }
    window.addEventListener("pagehide", () => worker.terminate(), { once: true });
    return { ...info,
      reset: () => call("reset"),
      calibrate: (frames, width, height) => {
        const buffers = frames.map(frame => frame.data.buffer);
        return call("calibrate", { frames: buffers, width, height }, buffers);
      },
      process: (image, frame_idx, t) => {
        const rgba = image.data.buffer;
        return call("process", { rgba, width: image.width, height: image.height, frame_idx, t }, [rgba]);
      },
      dispose,
    };
  }
  window.CVNBA = { load };
})();
