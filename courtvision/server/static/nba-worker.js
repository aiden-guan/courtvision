/* Run the existing NBA-trained BallPersonNet in a worker, entirely on the visitor's device. */
"use strict";
importScripts("ort.min.js");
let session = null, manifest = null, backend = "wasm", python = null, pipeline = null;
let queue = Promise.resolve();
ort.env.wasm.numThreads = 1;
ort.env.wasm.wasmPaths = new URL(".", self.location.href).href;

async function verify(candidate, reference) {
  const input = new Float32Array(9 * 544 * 960);
  for (let i = 0; i < input.length; i++) input[i] = ((i * 13 + (i >>> 10) * 7) % 251 - 125) / 128;
  const tensor = new ort.Tensor("float32", input, [1, 9, 544, 960]);
  let outputs;
  try {
    outputs = await candidate.run({ frames: tensor });
    for (const [name, probes] of Object.entries(reference.outputs)) {
      for (let i = 0; i < probes.indices.length; i++) {
        const value = outputs[name].data[probes.indices[i]];
        if (!Number.isFinite(value) || Math.abs(value - probes.values[i]) > reference.tolerance) {
          throw new Error("This inference backend does not match the original NBA model.");
        }
      }
    }
  } finally {
    if (outputs) for (const output of Object.values(outputs)) output.dispose();
    tensor.dispose();
  }
}

async function handle(data) {
  const { id, action } = data;
  const progress = (stage, label, completed = null, total = null) =>
    self.postMessage({ id, progress: { stage, label, completed, total } });
  try {
    if (action === "load") {
      const metadata = await fetch("nba-student.json");
      if (!metadata.ok) throw new Error("The NBA model is unavailable. Please retry.");
      manifest = await metadata.json();
      progress("model", "Downloading the NBA model…", 0, null);
      const response = await fetch("nba-student.onnx");
      if (!response.ok) throw new Error("The NBA model could not download. Check your connection and retry.");
      let bytes;
      if (response.body) {
        const reader = response.body.getReader(), chunks = [];
        const total = response.headers.get("content-encoding") ? null : Number(response.headers.get("content-length")) || null;
        let completed = 0, reported = 0;
        for (;;) {
          const { value, done } = await reader.read();
          if (done) break;
          chunks.push(value); completed += value.byteLength;
          if (performance.now() - reported > 100) {
            progress("model", "Downloading the NBA model…", completed, total); reported = performance.now();
          }
        }
        const joined = new Uint8Array(completed);
        let offset = 0;
        for (const chunk of chunks) { joined.set(chunk, offset); offset += chunk.byteLength; }
        bytes = joined.buffer;
      } else bytes = await response.arrayBuffer();
      progress("model", "Checking the model on this device…");
      const digest = Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256", bytes)), b => b.toString(16).padStart(2, "0")).join("");
      if (digest !== manifest.onnx_sha256) throw new Error("The NBA model download is incomplete. Reload and retry.");
      const referenceResponse = await fetch("nba-validation.json");
      if (!referenceResponse.ok) throw new Error("The NBA model reference check is unavailable. Please retry.");
      const reference = await referenceResponse.json();
      if (reference.onnx_sha256 !== digest) throw new Error("The NBA model reference is out of date. Reload and retry.");
      if (self.navigator.gpu) {
        try {
          session = await ort.InferenceSession.create(bytes, { executionProviders: ["webgpu"] });
          await verify(session, reference); backend = "webgpu";
        } catch {
          await session?.release(); session = null;
          progress("model", "Checking the CPU backend on this device…");
        }
      }
      if (!session) {
        session = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] });
        await verify(session, reference);
      }
      progress("runtime", "Starting local analysis tools…", 0, 3);
      const { loadPyodide } = await import("https://cdn.jsdelivr.net/pyodide/v314.0.7/full/pyodide.mjs");
      python = await loadPyodide({ indexURL: "https://cdn.jsdelivr.net/pyodide/v314.0.7/full/" });
      progress("runtime", "Loading vision libraries…", 1, 3);
      await python.loadPackage(["numpy", "scipy", "opencv-python"]);
      progress("runtime", "Preparing local tracking…", 2, 3);
      const packageResponse = await fetch("browser-python.json");
      if (!packageResponse.ok) throw new Error("The local vision pipeline is unavailable. Please retry.");
      const packageInfo = await packageResponse.json();
      const bundleResponse = await fetch("browser-python.zip");
      if (!bundleResponse.ok) throw new Error("The local vision pipeline could not download. Please retry.");
      const bundle = await bundleResponse.arrayBuffer();
      const packageHash = Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256", bundle)), b => b.toString(16).padStart(2, "0")).join("");
      if (packageHash !== packageInfo.sha256) throw new Error("The local pipeline download is incomplete. Reload and retry.");
      python.unpackArchive(bundle, "zip", { extractDir: "/home/pyodide" });
      await python.runPythonAsync("from courtvision.browser.runtime import BrowserPipeline");
      pipeline = python.runPython("BrowserPipeline()");
      progress("runtime", "Local analysis tools ready", 3, 3);
      self.postMessage({ id, result: { backend, checkpoint: manifest.checkpoint_sha256, pipeline: "original-python", source: packageHash, validated: true } });
    } else if (action === "reset") {
      pipeline?.destroy(); pipeline = python.runPython("BrowserPipeline()");
      self.postMessage({ id, result: true });
    } else if (action === "calibrate") {
      const frames = python.toPy(data.frames.map(buffer => new Uint8Array(buffer)));
      try {
        const report = (completed, total) => progress("court", completed < total - 1 ?
          `Checking court view ${completed + 1} of ${total - 1}…` : completed < total ?
          "Refining the court map…" : "Court setup complete", completed, total);
        self.postMessage({ id, result: JSON.parse(pipeline.calibrate(frames, data.width, data.height, report)) });
      }
      finally { frames.destroy(); }
    } else if (action === "process") {
      if (!session || !pipeline) throw new Error("The NBA vision pipeline is not loaded yet.");
      const normalized = pipeline.prepare(new Uint8Array(data.rgba), data.width, data.height, data.frame_idx, data.t);
      const buffer = normalized.getBuffer("f32");
      const tensor = new ort.Tensor("float32", new Float32Array(buffer.data), [1, 9, 544, 960]);
      buffer.release(); normalized.destroy();
      let outputs = null, raw = null;
      try {
        outputs = await session.run({ frames: tensor });
        raw = python.toPy(Object.fromEntries(Object.entries(outputs).map(([name, value]) => [name, value.data])));
        self.postMessage({ id, result: JSON.parse(pipeline.finish(raw)) });
      } finally {
        raw?.destroy();
        if (outputs) for (const output of Object.values(outputs)) output.dispose();
        tensor.dispose();
      }
    }
  } catch (error) { self.postMessage({ id, error: error.message || String(error) }); }
}
self.onmessage = ({ data }) => { queue = queue.then(() => handle(data)); };
