/* Run the existing NBA-trained BallPersonNet in a worker, entirely on the visitor's device. */
"use strict";
importScripts("ort.webgpu.min.js", "nba-core.js");
let session = null, manifest = null, backend = "wasm";
ort.env.wasm.numThreads = 1;
ort.env.wasm.wasmPaths = new URL(".", self.location.href).href;

self.onmessage = async ({ data }) => {
  const { id, action } = data;
  try {
    if (action === "load") {
      const metadata = await fetch("nba-student.json");
      if (!metadata.ok) throw new Error("The NBA model is unavailable. Please retry.");
      manifest = await metadata.json();
      const response = await fetch("nba-student.onnx");
      if (!response.ok) throw new Error("The NBA model could not download. Check your connection and retry.");
      const bytes = await response.arrayBuffer();
      const digest = Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256", bytes)), b => b.toString(16).padStart(2, "0")).join("");
      if (digest !== manifest.onnx_sha256) throw new Error("The NBA model download is incomplete. Reload and retry.");
      if (self.navigator.gpu) {
        try { session = await ort.InferenceSession.create(bytes, { executionProviders: ["webgpu"] }); backend = "webgpu"; }
        catch { /* CPU fallback uses the exact same exported weights. */ }
      }
      if (!session) session = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] });
      self.postMessage({ id, result: { backend, checkpoint: manifest.checkpoint_sha256 } });
    } else if (action === "detect") {
      if (!session) throw new Error("The NBA model is not loaded yet.");
      const tensor = new ort.Tensor("float32", CVNBACore.normalize(data.frames), [1, 9, 544, 960]);
      const outputs = await session.run({ frames: tensor });
      const predictions = CVNBACore.decode(outputs, data.contentHeight);
      for (const output of Object.values(outputs)) output.dispose();
      tensor.dispose();
      self.postMessage({ id, result: predictions });
    }
  } catch (error) { self.postMessage({ id, error: error.message || String(error) }); }
};
