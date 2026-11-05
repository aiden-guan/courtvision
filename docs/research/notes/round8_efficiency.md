> Research notes compiled on 2026-11-05 by a research agent of the cloud session (round 8; docs/research/cross_game_detector.md §13), lightly reviewed. Figures marked as estimates or *unverified* were not checked; the sources are cited in place. Paths to that session's scratchpad (crop sheets, drawings) are not in the repository.

# A faster BallPersonNet in the browser and on CPUs, with the same detections

Research note, 2026-11-05. It covers the browser app (`courtvision/server/static/nba-worker.js` with Pyodide's
`courtvision/browser/runtime.py`) and the CPU path (`courtvision/perception/student_onnx.py`).

**I ran no benchmarks for this note.** The container's 4 cores were busy with training, and the container has no GPU.
Each number below comes from one of four places:

1. read from this repository;
2. computed analytically (the per-layer multiply-adds);
3. measured cheaply here (file sizes, gzip ratios, HTTP headers);
4. cited from a source.

Estimates are marked **estimate**. Claims I could not confirm are marked **unverified**.

---

## 0. Summary

### The five changes with the best expected speedup per unit of effort

| # | Change | Where it helps | Expected speedup | Effort | Accuracy risk |
|---|---|---|---|---|---|
| 1 | Cross-origin isolation (COOP/COEP headers) plus ORT's default thread count, instead of the hard-coded `numThreads = 1` | WASM fallback | about 2.5-3.5x on the model at ORT's default of 4 threads (**estimate**; sources report 1.8-2.9x from threads alone on small models and 7.5x with 16 threads) | under a day | none |
| 2 | An fp16 copy of the model for WebGPU when the adapter has `shader-f16`, with the fp32 model as fallback | WebGPU | 1.2x reported for a convolutional net on an M3 Max; up to about 2x (**estimate**). The download halves (47.3 to about 23.7 MB) and session start-up roughly halves | 1-2 days | low (Core ML fp16 left held-out F1 unchanged), but WebGPU must be checked separately |
| 3 | One onnxruntime-web bundle per backend: the native WebGPU EP (`ort.webgpu.min.js`) for the GPU, and the plain CPU build (`ort.wasm.min.js`, no Asyncify) for the fallback. Today the app ships the JSEP build, deprecated since ORT 1.29 | both | **unverified**: I found no public JSEP-against-native benchmark. Moving off JSEP is required eventually anyway | under a day, plus an A/B test | none (the existing reference check gates it) |
| 4 | The native CPU fast path ported to the WASM fallback: the int8 QDQ model, with the offset and size heads evaluated at peaks only | WASM fallback | sparse heads 1.23x (measured natively in this repo); int8 on WASM **unverified** (2.1x natively with VNNI/AMX). Combined 1.6-2.5x (**estimate**). The download drops 47.3 to 12.7 MB | 3-5 days | sparse heads none; int8 low (ball hit 89.25 % against 89.25 %, p 1, for round 4b) |
| 5 | A leaner stride-4 neck and heads: depthwise-separable 3x3 convolutions in `smooth`, `ball_hm` and `person_hm`, distilled from main model v2 | all backends, including native CPU | 31.5 to 23.0 GMACs, 27 % fewer than today's CPU sparse path. About 1.3x on CPU and WASM (**estimate**), less on WebGPU | 1-2 weeks | moderate (retraining; the ball's local context lives in these layers) |

Measure first (step 0, §7.1). Instrument one browser frame end to end before optimising the model for WebGPU users.
Video seeking, the 18.8 MB copies and the Python court tracker may cost as much as the network on a GPU. Nobody has
measured that split yet.

### Key findings

- **The app runs JSEP for both backends.** It ships onnxruntime-web 1.31.1's `ort.min.js`, which is the JSEP build
  (`ort-wasm-simd-threaded.jsep.wasm`, 29.5 MB), for WebGPU and for the WASM fallback alike.
  - ORT 1.29.0's release notes say: "onnxruntime-web has announced the deprecation of WebGL and JSEP" and "The
    native WebGPU EP is the recommended path going forward"
    ([v1.29.0](https://github.com/microsoft/onnxruntime/releases/tag/v1.29.0)).
  - ORT builds JSEP and the native WebGPU EP with `-s ASYNCIFY=1`
    ([onnxruntime_webassembly.cmake](https://github.com/microsoft/onnxruntime/blob/main/cmake/onnxruntime_webassembly.cmake)).
    So today the CPU fallback also runs inside an Asyncify-instrumented binary.
- **The WASM fallback is single-threaded twice over.**
  - `nba-worker.js` sets `ort.env.wasm.numThreads = 1`.
  - Even without that line, the server sends no COOP/COEP headers, so `crossOriginIsolated` is false. ORT 1.31.1 then
    forces one thread (checked in the shipped `ort.min.js`).
- **The only third-party origin will survive `require-corp`.** It is Pyodide on `cdn.jsdelivr.net`, which answers with
  `access-control-allow-origin: *` and `cross-origin-resource-policy: cross-origin` (checked with curl on
  2026-11-05). So `Cross-Origin-Embedder-Policy: require-corp` does not break the app.
- **int8 helps only on the CPU and WASM.**
  - JSEP's WebGPU kernels include `DequantizeLinear` but no `QuantizeLinear`, `QLinearConv` or `ConvInteger`.
  - The native WebGPU EP registers Q/DQ kernels but no quantized convolution.
  - Microsoft's Windows ML page says that "Quantized non-LLM INT8 models are not supported on the WebGPU path".
- **fp16 helps only on WebGPU.**
  - ORT's CPU advice is: "Avoid float16 models if possible, as float16 is not natively supported by CPU and it is
    going to be slow."
  - `shader-f16` is available in Chrome 120+ (macOS, Windows, ChromeOS), in Chrome 144+ on Linux (Intel Gen12+ only),
    and in Safari 26. Firefox does not have it.
- **Caching stem features saves no multiply-adds.** The 9-channel stem does split exactly into three 3-channel
  convolutions. But each frame meets a different weight slice at each of its three positions in the window, so the
  three partial convolutions have to be computed anyway (§4.2). The split saves input bandwidth only.
- **Most of the compute is at stride 4.** 57 % of the 38.7 GMACs are there: `layer1`, `smooth` and the five heads.
  ResNet layers 2-4 are 33 %. Even a backbone that cost nothing would leave 17.4 GMACs in the dense graph, at most 2.2x
  faster. A realistic lighter backbone gives about 1.7-2x (**estimate**). The neck and heads are the bigger lever.

---

## 1. Where the compute and time go

### 1.1 Multiply-adds per stage

Computed from `build_net` in `courtvision/perception/student.py` for a 9×544×960 input. The total matches the
38.70 GMACs of `docs/research/cross_game_detector.md` §3.1.

| Stage | Resolution | GMACs | Share |
|---|---|---:|---:|
| `stem` 7×7, 9→64 (stride 2) | 272×480 | 3.68 | 9.5 % |
| `layer1` (four 3×3, 64→64) | 136×240 (s4) | 4.81 | 12.4 % |
| `layer2` | 68×120 (s8) | 4.28 | 11.1 % |
| `layer3` | 34×60 (s16) | 4.28 | 11.1 % |
| `layer4` | 17×30 (s32) | 4.28 | 11.1 % |
| FPN laterals (1×1) | all | 0.50 | 1.3 % |
| FPN `smooth` 3×3, 128→128 | s4 | 4.81 | 12.4 % |
| `ball_hm` head (3×3 128→64, 1×1) | s4 | 2.41 | 6.2 % |
| `person_hm` head | s4 | 2.41 | 6.2 % |
| `ball_off`, `person_wh`, `person_off` heads (sparse on the CPU path) | s4 | 3 × 2.41 | 18.7 % |
| **Total** | | **38.70** | |

The CPU path already evaluates the three offset and size heads only at the decoded peaks, which leaves 31.48 GMACs.
The browser still runs the whole dense graph.

### 1.2 What the browser does per frame

All of these steps run one after another. The next frame is not even decoded until this one's result comes back
(`nba-clip.js` `run()`).

1. `nba-clip.js` seeks a hidden `<video>` to `index / 29.97`, calls `drawImage`, then `getImageData` (an 8.3 MB RGBA
   frame at 1080p), then copies it with `.slice()` for display.
2. It posts the frame to the worker (transferred).
3. `runtime.py` `prepare` (Pyodide) converts RGBA to BGR, resizes with `INTER_AREA`, normalises through the LUT, and
   concatenates three frames into 18.8 MB of float32.
4. `nba-worker.js` copies that again with `new Float32Array(buffer.data)`.
5. `session.run` either uploads the input to the GPU (WebGPU) or copies it into ORT's heap (WASM).
6. The five outputs (1.04 MB) cross back into Python through `python.toPy`.
7. `finish` runs `decode`, `referee_calls`, the court tracker, ByteTrack and the ball tracker.

How a frame's time divides between these steps is **not known**. Step 0 (§7.1) measures it. On the WASM path the
network will dominate: the model takes seconds per frame single-threaded (**estimate**, from §2.3's instruction
counts). On WebGPU the network may be only part of the frame time.

### 1.3 What is already measured

From `runs/bench/speed_4t.json` and cross_game_detector.md §3.3. Intel Xeon at 2.10 GHz, 4 threads, ONNX Runtime
1.31; median of 40 detections, network plus decoding.

| Path | ms |
|---|---:|
| PyTorch fp32 | 222.0 |
| ORT fp32, whole graph | 129.8 |
| ORT fp32, sparse heads | 105.9 |
| ORT int8 QDQ, sparse heads | 49.6 |

On the Mac, Core ML on the Neural Engine runs the network in 7.7 ms, and a live detection takes about 10-12 ms
(`docs/research/analysis_speed.md`).

### 1.4 Serving and deployment

- **Vercel serves only `showcase/`.** It is a static video page whose CSP allows no scripts (`vercel.json`;
  `.vercelignore` is an allowlist). No inference runs there, so COOP/COEP do not matter for Vercel today.
- **The browser app is served by the FastAPI Studio** (`courtvision studio [--public-demo]`, or the Dockerfile).
  `_GuardMiddleware` in `courtvision/server/app.py` (around lines 434-452) adds CSP, `nosniff`, `Referrer-Policy` and
  `X-Frame-Options` to every response, static files included.
- **That server sends no COOP/COEP, no Cache-Control on static files and no compression.** Starlette `StaticFiles`
  sends only ETag and Last-Modified.

---

## 2. onnxruntime-web: findings for the version in the repo (1.31.1)

### 2.1 Which runtime the app runs, and the alternatives

The 1.31.1 package
([file list](https://data.jsdelivr.com/v1/packages/npm/onnxruntime-web@1.31.1),
[exports](https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.1/package.json)) and ORT's
[build script](https://github.com/microsoft/onnxruntime/blob/main/js/web/script/build.ts) give this mapping:

| Bundle | WebGPU implementation | WASM binary | Size | Asyncify |
|---|---|---|---:|---|
| `ort.min.js` (what the app ships) | JSEP | `ort-wasm-simd-threaded.jsep.wasm` | 29.5 MB | yes |
| `ort.webgpu.min.js` (`onnxruntime-web/webgpu`) | native WebGPU EP | `ort-wasm-simd-threaded.asyncify.wasm` | 28.7 MB | yes |
| `ort.jspi.min.js` | native WebGPU EP | `ort-wasm-simd-threaded.jspi.wasm` | 17.9 MB | no (JSPI) |
| `ort.wasm.min.js` | none (CPU only) | `ort-wasm-simd-threaded.wasm` | 14.8 MB | no |

- **JSEP is on its way out.** ORT 1.29.0 deprecated JSEP and WebGL and called the native WebGPU EP "the recommended
  path going forward" ([release notes](https://github.com/microsoft/onnxruntime/releases/tag/v1.29.0)).
- **The native EP is where convolution work goes now.**
  - ORT 1.30 extended convolution fusion with eight more activations, fused activations into the im2col path, and
    reused subgroup-matrix MatMul for pointwise convolution
    ([v1.30.0](https://github.com/microsoft/onnxruntime/releases/tag/v1.30.0)).
  - WebGPU plugin EP v0.5.0 vectorised NHWC depthwise convolution
    ([notes](https://github.com/microsoft/onnxruntime/releases/tag/plugin-ep-webgpu/v0.5.0)).
  - JSEP's operator table still lists Conv as "need perf optimization"
    ([webgpu-operators.md](https://github.com/microsoft/onnxruntime/blob/main/js/web/docs/webgpu-operators.md)).
- **Microsoft calls the native EP early-stage** on its Windows ML page: "an early-stage provider and not yet
  recommended for performance-critical or ultra-optimized workloads"
  ([Microsoft Learn](https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/webgpu-ep), dated 2026-08-05). That
  page is about Windows ML, not the browser. **No public JSEP-against-native benchmark was found; this needs an A/B
  test.**
- **Asyncify costs speed and size.** Emscripten: "Asyncify adds overhead, both code size and slowness … something like
  50% or so" for the code it instruments ([Emscripten docs](https://emscripten.org/docs/porting/asyncify.html)). Both
  JSEP and native-EP builds use it; the plain CPU build does not. Whether ORT's MLAS kernels are among the instrumented
  code is **unverified**. Loading `ort.wasm.min.js` for WASM-only visitors is worth an A/B test, and it also halves
  the runtime download.
- **JSPI avoids Asyncify** but needs Chrome 137, Firefox 153 or Safari 27
  ([MDN compat data](https://github.com/mdn/browser-compat-data/blob/main/webassembly/jspi.json)).
- **Switching backends needs a fresh worker.** ORT initialises its WASM module once per worker ("multiple calls to
  'initializeWebAssembly()' detected"). A WebGPU session that fails `verify()` therefore needs a new worker loaded with
  the CPU bundle; it cannot simply create a second session in the same worker.

### 2.2 Multi-threaded WASM

**What it needs**

- **ORT's threading rule.** From the
  [env flags doc](https://onnxruntime.ai/docs/tutorials/web/env-flags-and-session-options.html): `numThreads` defaults
  to 0, "half of navigator.hardwareConcurrency or 4, whichever is smaller". "Only when the browser supports WebAssembly
  multi-threading and crossOriginIsolated mode is enabled, multi-threading will be enabled."
  - The shipped 1.31.1 bundle does exactly this. Without isolation it sets `numThreads=1`; with it,
    `Math.min(4, Math.ceil(hardwareConcurrency/2))`. It warns when more than one thread is requested but unavailable.
  - The count includes the calling thread, and it must be set before the first session is created.
- **Cross-origin isolation** means two headers on the top-level document:
  - `Cross-Origin-Opener-Policy: same-origin`;
  - `Cross-Origin-Embedder-Policy: require-corp` (or `credentialless`).
  - web.dev's guide adds: "set the Cross-Origin-Embedder-Policy: require-corp header on iframes and worker scripts
    (This is required regardless of same-origin or cross-origin)"
    ([web.dev](https://web.dev/articles/cross-origin-isolation-guide)).
  - Because `_GuardMiddleware` wraps every response, two added headers cover the page, `nba-worker.js` and ORT's
    `.mjs` files.
- **SharedArrayBuffer and WASM threads are widely supported.** SharedArrayBuffer: Chrome 68, Firefox 79, Safari 15.2.
  WASM threads and atomics: Chrome 74, Firefox 79, Safari 15.2
  ([MDN compat data](https://github.com/mdn/browser-compat-data)).
- **ORT's threads are nested workers.** ORT starts its threads from inside our worker as module workers,
  `new Worker(new URL(import.meta.url), {type:"module",…})`, as seen in the shipped
  `ort-wasm-simd-threaded.jsep.mjs`.
  - Safari has supported nested workers since 15.5, and script loading in nested workers since 16.4 (MDN compat data
    for `Worker`).
  - The thread workers are same-origin, so the existing `worker-src 'self'` CSP allows them. ORT's
    [deploy doc](https://onnxruntime.ai/docs/tutorials/web/deploy.html) says CSP-restricted environments have been
    supported since 1.19.

**What it does to third-party resources**

- **What `require-corp` blocks.** Under it, a cross-origin `no-cors` subresource without a CORP header is blocked.
  CORS-mode requests need CORS approval and are not otherwise blocked by COEP
  ([MDN COEP](https://developer.mozilla.org/en-US/docs/Web/HTTP/Reference/Headers/Cross-Origin-Embedder-Policy)).
- **The app has one third-party origin.** The CSP and a grep of `static/*.js` show only
  `https://cdn.jsdelivr.net/pyodide/v314.0.7/full/` (the worker's `import()` and Pyodide's `loadPackage` fetches).
  `pyodide.mjs`, `pyodide.asm.wasm` and `pyodide-lock.json` all return
  `access-control-allow-origin: *` and `cross-origin-resource-policy: cross-origin` (curl, 2026-11-05). So
  `require-corp` works without `credentialless`.
- **`credentialless` would not suit anyway**: Safari does not support it (Chrome 96, Firefox 119; MDN compat data).
- **COOP `same-origin` cuts the link to cross-origin popups.** The static app opens no popups or iframes (checked by
  grep).
- **A side benefit**: isolated pages get high-resolution `performance.now()`, which helps step 0.

**Typical speedups**

| Source | Workload | Result |
|---|---|---|
| [TensorFlow.js blog, 2020](https://blog.tensorflow.org/2020/09/supercharging-tensorflowjs-webassembly.html) | small models | "SIMD brings a 1.7-4.5X performance improvement to plain Wasm, and multithreading brings another 1.8-2.9X speedup on top of that" |
| [ORT Web launch, 2021](https://opensource.microsoft.com/blog/2021/09/02/onnx-runtime-web-running-your-machine-learning-model-in-browser/) | MobileNetV2, 4-core Xeon | 3.4x with two threads plus SIMD, against plain WASM |
| [img.ly, ORT-web](https://img.ly/blog/browser-background-removal-using-onnx-runtime-webgpu/) | convolutional background-removal net (ISNet, 1024×1024), Apple M3 Max | SIMD on 1 thread ~15,000 ms; SIMD on 16 threads ~2,000 ms (7.5x); neither SIMD nor threads ~53,000 ms |

For this network I expect **about 2.5-3.5x at 4 threads (estimate)**. Its convolutions are large GEMMs, with 32,640
rows at stride 4, which split well across threads. On machines with many cores, `numThreads` can go above ORT's
default of 4, but Pyodide shares the worker, so measure 4 against 8.

### 2.3 WASM SIMD and relaxed SIMD

- **Fixed-width SIMD is already in use.** ORT's build doc says it "is dropping support for non-SIMD and non-threaded
  builds in future versions since v1.19.0" ([build doc](https://onnxruntime.ai/docs/build/web.html)), and the 1.31.1
  package ships only `-simd-threaded` binaries (file list). Browser support: Chrome 91, Firefox 89, Safari 16.4.
- **fp32 GEMM on WASM has no fused multiply-add.** MLAS's WASM path computes
  `wasm_f32x4_add(wasm_f32x4_mul(…))`
  ([mlasi.h](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/mlas/lib/mlasi.h)). That is 4
  multiply-adds per 2 SIMD instructions. Relaxed SIMD does not change this: there is no relaxed-SIMD SGEMM kernel in
  MLAS.
- **int8 GEMM gets about twice the work per instruction.** The fixed-SIMD kernel
  ([qgemm_kernel_wasmsimd.cpp](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/mlas/lib/qgemm_kernel_wasmsimd.cpp))
  is built on `wasm_i32x4_dot_i16x8`: 8 multiply-adds per dot instruction after widening to int16, about twice the
  fp32 rate per instruction. That is an **estimate from the source code**, not a measurement.
- **Relaxed SIMD helps int8 only, and needs a custom build.**
  - [qgemm_kernel_wasmrelaxedsimd.cpp](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/mlas/lib/qgemm_kernel_wasmrelaxedsimd.cpp)
    uses `wasm_i32x4_relaxed_dot_i8x16_i7x16_add`: 16 multiply-adds per instruction.
  - It is compiled only with `--enable_wasm_relaxed_simd`
    ([build_args.py](https://github.com/microsoft/onnxruntime/blob/main/tools/ci_build/build_args.py),
    [onnxruntime_mlas.cmake](https://github.com/microsoft/onnxruntime/blob/main/cmake/onnxruntime_mlas.cmake)), which
    produces an `ort-wasm-relaxedsimd-threaded…` binary.
  - `env.wasm.simd = "relaxed"` only checks for support: "Setting this property does not make ONNX Runtime to switch to
    the corresponding runtime automatically. User need to set `wasmPaths` or `wasmBinary` property to load the
    corresponding runtime" ([env.ts](https://github.com/microsoft/onnxruntime/blob/main/js/common/lib/env.ts)).
  - Browser support: Chrome 114, Firefox 146, Safari only in Technology Preview. A relaxed build would therefore need a
    fixed-SIMD fallback.
  - Gain for this model: **unverified**.

### 2.4 The WebGPU EP

**fp16**

- **Browser support.** ORT: "Float16 support requires Chrome v121 or later, and Edge v122 or later"
  ([get started](https://onnxruntime.ai/docs/get-started/with-javascript/web.html)). MDN compat data for
  `shader-f16`:
  - Chrome 120+ on ChromeOS, macOS and Windows;
  - Chrome on Linux from 144, Intel Gen12+ only;
  - Safari 26;
  - Firefox: no.
- **The JSEP bundle in the repo already asks for f16.** The 1.31.1 bundle requests `shader-f16` and `subgroups` when
  the adapter offers them.
- **Converting the model.** Use
  `onnxconverter_common.float16.convert_float_to_float16(model, keep_io_types=True, node_block_list=[…])`
  ([ORT float16 doc](https://onnxruntime.ai/docs/performance/model-optimizations/float16.html)).
  - `keep_io_types=True` keeps the graph's inputs and outputs float32 with Casts inside, so the JS side does not change.
  - With float16 inputs instead, the 18.8 MB upload halves. The page then feeds `'float16'` tensors; JS `Float16Array`
    exists in Chrome 135, Firefox 129 and Safari 18.2.
  - **I found no onnxruntime-web option called `keepIo16`.** The flag that does this is the converter's
    `keep_io_types`.
- **Never on the CPU.** ORT's advice is to avoid fp16 there
  ([performance diagnosis](https://onnxruntime.ai/docs/tutorials/web/performance-diagnosis.html)). img.ly measured
  fp16 slower than fp32 on WASM: 2,300 against 2,000 ms on 16 threads.
- **Speed.** img.ly, WebGPU on an M3 Max: 120 ms fp32 to 100 ms fp16 (1.2x), and session start-up 400 to 200 ms.
  Larger gains are reported only in weaker sources (for example 3.5x on Real-ESRGAN on Apple, a DEV Community post).
- **Accuracy evidence for this network.** Core ML fp16 on the Neural Engine left held-out ball F1 unchanged (0.9608)
  and person recall at 0.8039 against 0.8037 (`analysis_speed.md` §2).
  - The caveat: WebGPU convolution shaders accumulate in the tensor's own type.
  - `enableMatmulFp32Accumulation` applies only to MatMulNBits, and only in the native EP
    ([API doc](https://onnxruntime.ai/docs/api/js/interfaces/InferenceSession.WebGpuExecutionProviderOption.html)).
  - So fp16 must be validated again in the browser.
- **The reference check would reject fp16 as written.** `verify()` requires |Δ| ≤ 0.003 on raw outputs. fp16's step
  between representable values is 0.0039 in [4, 8) and 0.0078 in [8, 16). Heatmap logits of that size can therefore
  fail on representation alone, and the backend would fall back to WASM. An fp16 model needs:
  - its own reference and tolerance (for example, compare the heatmaps after the sigmoid, with tolerances taken from
    measured fp16-against-fp32 differences on real frames);
  - the five 1×1 output convolutions left in fp32 through `node_block_list`. They cost 0.02 % of compute, and
    `tools/quantize_student.py` keeps them fp32 for the same reason.

**IO binding**

- By default inputs and outputs live in CPU memory. `preferredOutputLocation: 'gpu-buffer'` (globally or per output)
  and `Tensor.fromGpuBuffer` keep data on the GPU
  ([WebGPU EP doc](https://onnxruntime.ai/docs/tutorials/web/ep-webgpu.html)).
- Two uses here:
  - keep `feat` on the GPU for sparse heads (§3.2);
  - keep the last three frames in a GPU ring buffer and upload only the newest (6.3 MB in fp32) instead of 18.8 MB per
    frame.

**Graph capture**

- `enableGraphCapture: true` works for static shapes when every kernel is on the EP. It helps "when CPU sometimes is
  the bottleneck to prepare for the commands"
  ([session options](https://onnxruntime.ai/docs/tutorials/web/env-flags-and-session-options.html)).
- In 1.31.1 every input and output must then be a GPU buffer. The bundle throws "Only 'gpu-buffer' location is
  supported when enableGraphCapture is true" and "External buffer must be provided for input/output index … when
  enableGraphCapture is true". So the app must upload and read back through its own GPUBuffers.
- The gain is only CPU-side dispatch for roughly 45 kernels: probably under 10 % (**estimate**).
- **The export computes its Resize sizes at run time.** The ONNX graph has 94 nodes, including 3 `Shape`, 3 `Slice`,
  3 `Concat` and 12 `Constant` nodes feeding 3 `Resize` nodes. JSEP's `Shape` has "no GPU kernel".
  - ORT should constant-fold these for a static graph (**unverified for JSEP**).
  - Exporting with `scale_factor=2` removes them, and is exact here, since every level is exactly twice the one below:
    17×30, 34×60, 68×120, 136×240.

**`freeDimensionOverrides`**

Not needed: the export has fixed shapes, [1, 9, 544, 960]. It matters only if you export dynamic height and width for
ROI crops (§4.3).

**Other options**

- `preferredLayout`: both the JSEP and native-EP sources default to NHWC
  ([js_execution_provider.h](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/providers/js/js_execution_provider.h),
  [webgpu_execution_provider.h](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/providers/webgpu/webgpu_execution_provider.h)).
  The JS API page lists `'NCHW'` as the default. Keep the default, or A/B the two layouts.
- Two more options are documented on the main-branch JS API page. Which bundles read them (JSEP, the native EP, or
  both) is **unverified**:
  - `validationMode: 'wgpuOnly'` for production (the default is `'basic'`);
  - `storageBufferCacheMode: 'simple'`, which "may reduce GPU memory usage" for static shapes.

### 2.5 int8/QDQ: WASM against WebGPU

- **WASM (the CPU EP): yes.**
  - The QDQ model fuses into QLinearConv and runs MLAS's U8X8 QGEMM with WASM SIMD (§2.3).
  - ORT-web says "Prefer uint8 quantized models", and `tools/quantize_student.py` already writes uint8 activations
    with int8 weights.
  - ORT-web's default `graphOptimizationLevel` is `"all"` (checked in the bundle).
- **WebGPU: no.**
  - JSEP lists `DequantizeLinear` but no `QuantizeLinear`, `QLinearConv` or `ConvInteger`, so a QDQ model would be
    split between GPU and CPU.
  - The native EP registers Q/DQ kernels but has no quantized convolution
    ([webgpu_execution_provider.cc](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/providers/webgpu/webgpu_execution_provider.cc)).
  - Windows ML WebGPU EP page: "Quantized non-LLM INT8 models are not supported on the WebGPU path: such models will run
    in higher precision or may fall back to CPU, negating performance benefits."
  - Use fp16 on WebGPU instead.
- **int8 accuracy, as measured in this repo.**
  - Published model: ball hit 81.3 % against 81.7 % (p 0.23); person boxes 98.4 % in agreement.
  - Round 4b: ball hit 89.25 % against 89.25 % (2 frames only one hits, 2 the other, p 1;
    `runs/bench/ball_check12_r4b_int8.log`); person boxes 98.0 % in agreement; ball candidates 92.2 % in agreement.
  - The ball candidates that disagree are mostly extra low-score candidates near the threshold (§3.2 of the research
    note).

### 2.6 Model size and caching

| File | Size |
|---|---|
| `nba-student.onnx` (fp32) | 47.3 MB; gzip -6 only reaches 43.9 MB (−7.2 %, measured here) |
| int8 QDQ | 12.7 MB |
| fp16 | about 23.7 MB (**estimate**: half the fp32 weights) |
| `ort-wasm-simd-threaded.jsep.wasm` | 29.5 MB; gzip 7.0 MB (−76 %, measured here) |
| plain CPU `.wasm` | 14.8 MB |
| Pyodide `pyodide.asm.wasm` (jsDelivr, cached for a year) | 9.6 MB |

- **Nothing is cached or compressed today.** The worker re-downloads the model on every load (unless the HTTP cache
  revalidates it) and checks its SHA-256. The server sends no Cache-Control and no compression.
- **ORT's caching advice**: the Cache API or the Origin Private File System
  ([large models](https://onnxruntime.ai/docs/tutorials/web/large-models.html)), or IndexedDB (deploy doc).
  - The Cache API works in workers (Chrome 43+, Firefox 41, Safari 11.1).
  - OPFS: Chrome 86, Firefox 111, Safari 15.2.
  - `navigator.storage.persist()` protects either from eviction.
- **Plan**:
  1. Look the model up with `caches.open(...)` / `cache.match()`, keyed by the manifest's `onnx_sha256`, before
     fetching; keep the digest check.
  2. Give model and runtime files versioned names with `Cache-Control: public, max-age=31536000, immutable`.
  3. Gzip or brotli `.wasm`, `.mjs` and `.js` (Starlette's `GZipMiddleware`, or precompressed files). That alone saves
     about 22.5 MB of the first load.
- **This changes load time, not time per frame.**

---

## 3. Model-level options

### 3.1 What a tiny ball needs

- **The ball is a few cells.** At 960 px wide it is about 10-20 px, so 2.5-5 cells at stride 4. Its box radius is
  12 px at 1080p (`ball_radius_px`), about 3 cells at 960 px.
- **Basketball broadcasts move it fast.** In WASB's broadcast basketball set (1920×1080), the ball moves 33.7 ± 21.8 px
  per frame on average ([WASB, BMVC 2023](https://arxiv.org/abs/2311.05237), Table 1). That is about 17 px, or 4
  cells, per frame at this input scale.
- **Higher internal resolution helps small balls.** WASB's ablation removes the stem's strides:
  - basketball F1 rises from 77.4 to 80.4 and 80.6;
  - soccer F1 rises from 81.7 to 88.3;
  - throughput falls (soccer 85.7 to 55.7 FPS), and WASB still downscales every frame to 288×512.
- **Slicing helps small objects.** SAHI raises small-object AP by 6.8 points for FCOS on aerial footage by running the
  detector on higher-resolution slices ([SAHI](https://arxiv.org/abs/2202.06934)).
- **This repo's own probe.** Shrinking the frame to 0.7 inside the network input lowered player recall from 0.625 to
  0.568 (cross_game_detector.md §2).
- **Conclusion: keep the ball at stride 4 and 960 px**, or at a higher resolution in a crop. Look for savings in how
  the stride-4 maps are computed, not in coarser maps.

### 3.2 Sparse heads in the browser (no retraining)

- **What it is.** The same split as `OnnxStudent`: the trunk plus `ball_hm` and `person_hm` run densely, and
  `ball_off`, `person_wh` and `person_off` run at ≤ 68 peak cells.
- **Gain.** 7.22 of 38.70 GMACs; 1.23x measured natively (129.8 to 105.9 ms), with outputs within 1.4e-6.
- **WASM is easy.**
  - `browser-python.zip` already contains `perception/student_onnx.py` (`build.py` packages all of `perception/`), so
    `decode_sparse` and `SparseHead` work in Pyodide.
  - The trunk graph can be cut offline. The heads' weights take about 0.9 MB as an `.npz`.
  - Avoid moving the 16.7 MB `feat` through `toPy`: gather the k×1152 patch floats in JS.
- **WebGPU is harder.** Either read `feat` back (16.7 MB per frame) or keep it on the GPU and run a small second
  session that gathers the 3×3 neighbourhoods and multiplies them by the head weights. JSEP supports Pad, Gather,
  GatherND, MatMul and Concat. The net gain on WebGPU is **unverified**.

### 3.3 A leaner stride-4 neck and heads, distilled from main model v2

- **Design.** Replace the 3×3 convolutions in `smooth` (128→128) and in the `ball_hm` and `person_hm` heads (128→64)
  with depthwise 3×3 followed by pointwise 1×1. Keep the three sparse heads as they are, since they run only at peaks.
- **Multiply-adds (computed).** 31.48 to 23.03 GMACs: 27 % fewer than today's CPU path, 40 % fewer than the dense
  graph. Moving `person_hm` to stride 8 would save a further 0.2-1.8 GMACs, depending on its design.
- **Training.**
  - Teacher: the exact PyTorch copy of the export (`training/student_from_onnx.py`).
  - Student: the same trunk, frozen at first, with the new neck and heads.
  - Losses: the existing focal and L1 losses, plus distillation from the teacher's heatmap logits, offsets and sizes on
    the BARD clips.
  - Distillation methods for detectors: [Localization Distillation](https://arxiv.org/abs/2102.12252) says "logit
    mimicking can outperform feature imitation"; [FGD](https://arxiv.org/abs/2111.11837) reports +3.3 mAP for a
    RetinaNet-R50 student; [fine-grained imitation](https://arxiv.org/abs/1906.03609).
  - The repo already trains this way. `student_finetune.py` with bf16 and channels-last takes 0.37 s a step on the
    4-core Xeon, and the round-1 to round-5 fine-tunes self-trained from this network's own linked detections.
- **Speed.**
  - Compute-bound CPUs gain about the multiply-add ratio, about 1.3x over the sparse path (**estimate**).
    MobileNetV4's roofline analysis: on low ridge-point hardware such as CPUs, "to improve latency, you minimize the
    total number of MACs".
  - GPUs gain less, because depthwise layers are memory-bound. MobileNetV4: "Data movement is the bottleneck on high-RP
    hardware". Group Fisher pruning found that "inference speedup on GPUs is more correlated with the reduction of
    memory rather than FLOPs" ([arXiv](https://arxiv.org/abs/2108.00708)).

### 3.4 Structured channel pruning of ResNet-18 and the FPN

- **Evidence.**
  - [DepGraph (CVPR 2023)](https://arxiv.org/abs/2301.12900): ResNet-50 on ImageNet goes from 4.13 to 1.99 GMACs
    (2.07x) for −0.32 % top-1. Its "speed up" is a ratio of multiply-adds, not of latency.
  - [Group Fisher pruning (ICML 2021)](https://arxiv.org/abs/2108.00708) on COCO detectors:
    - RetinaNet at half the FLOPs: 36.5 to 36.3-36.8 AP, 1.32-1.57x faster on a GPU;
    - Faster R-CNN at a quarter of the FLOPs: 37.4 to 36.6 AP, 3.07x faster.
  - Earlier work: [Li et al.](https://arxiv.org/abs/1608.08710) (ResNet-110 −38 % FLOPs on CIFAR-10);
    [network slimming](https://arxiv.org/abs/1708.06519).
- **For this net.** Prune `layer2`-`layer4` and the laterals first: 13.3 GMACs, 34 %. Leave the stem and `layer1`,
  which carry the high-resolution evidence for the ball.
- **Folded batch norms.** `student_from_onnx` turns each folded BN into identity plus bias. BN-scale criteria such as
  network slimming would need BN back; L1 or Taylor criteria (Torch-Pruning/DepGraph) work as is.
- **Expected.** 1.2-1.4x for the whole network after fine-tuning (**estimate**). High effort; moderate risk for the
  ball.

### 3.5 Distilling into a lighter backbone

- **The ceiling.** The backbone (stem plus `layer1`-`layer4`) is 21.3 of 38.7 GMACs. If it cost nothing:
  - the dense graph would keep 17.4 GMACs, so at most 2.2x faster;
  - the sparse path would keep 10.1 GMACs (laterals, `smooth`, two dense heads), so at most 3.1x faster.

  A realistic replacement at about a quarter of ResNet-18's cost gives about 1.7x (dense) and 2.0x (sparse)
  (**estimate**), and less on GPUs, where depthwise layers are memory-bound. Beyond that, the neck and heads have to
  shrink too (§3.3).
- **Candidates.**

  | Backbone | ImageNet figure | Note |
  |---|---|---|
  | [RepVGG-A0](https://arxiv.org/abs/2101.03697) | 72.41 % top-1 at 1.4 GFLOPs, 3,256 img/s on a 1080Ti, against ResNet-18's 71.16 %, 1.8 GFLOPs, 2,442 img/s ("1.25% and 33% better") | plain 3×3 stacks suit GPUs and im2col CPUs; backbone saving here about 1.1-1.15x overall (**estimate**) |
  | [MobileNetV4-Conv-S](https://arxiv.org/abs/2404.10518) | 73.8 % at 0.2 GMACs (ResNet-18 is 1.8) | depthwise-heavy: big CPU gains, smaller on WebGPU |
  | [MobileNetV3](https://arxiv.org/abs/1905.02244) | "MobileNetV3-Large detection is 25% faster at roughly the same accuracy as MobileNetV2 on COCO" | depthwise-heavy |
  | [EfficientNet-Lite](https://blog.tensorflow.org/2020/03/higher-accuracy-on-vision-models-with-efficientnet-lite.html) | SE removed, ReLU6 instead of swish for int8; post-training quantisation "reduced the model size by 4x and improved inference speed by 2x" | its first quantisation attempt dropped top-1 from 75 % to 46 %: an int8 warning |
  | ResNet-18 at reduced width (for example 0.75x) | | multiply-adds scale by about the width squared in `layer2`-`layer4` |

- **Risks.**
  - The 9-channel motion stem has to be learnt again.
  - FLOPs are not latency ([ShuffleNet V2](https://arxiv.org/abs/1807.11164)).
  - Every option needs the full benchmark suite again.
  - Effort is the highest of all the options here.

### 3.6 Lower input resolution with a high-resolution ball crop

- **People** stay large at lower resolution (median 175 px tall at 720p in SportsMOT). A 640×352 pass has 0.43x the
  pixels, so about 2.3x fewer multiply-adds (computed). The ball cannot come from that pass (§3.1).
- **The ball would come from a crop** at the current scale around the tracked ball, for example 256×256 at 4.9 GMACs,
  with a full-resolution search whenever the ball is lost.
- **Risks.** Two models or two passes, crop scheduling, and retraining the person pass at the new scale: the repo's
  0.7x probe lost recall without retraining.
- **Expected.** 1.5-2.3x (**estimate**); very high effort; high risk.

### 3.7 fp16 weights

- **On WebGPU**: compute in fp16 (§2.4).
- **On WASM**: no fp16 compute.
  - Storing fp16 weights and casting them to fp32 at session start would halve the download with fp32 compute. The
    weights change slightly; the cast at load is **unverified**.
  - int8 is smaller still, and faster.

---

## 4. Temporal and system tricks

### 4.1 Running the person head less often than the ball

- **CPU path.** `person_wh` and `person_off` are already sparse. The only dense person compute left is `person_hm`'s
  3×3 convolution: 2.41 GMACs, 6.2 %. Running it every other frame saves about 3 % on average.
- **Dense browser graph.** The three person heads are 7.22 GMACs (18.7 %), so halving their rate saves about 9 %. Sparse
  heads (§3.2) save more, with no change in detections and nothing to coast through the tracker.
- **Not worth it** unless people move to a separate, cheaper path (§3.6 or §4.3).

### 4.2 Reusing stem features across overlapping three-frame windows

**The stem does split linearly into three 3-channel convolutions.** The stem is
`Conv2d(9, 64, 7, stride 2, pad 3)` with its batch norm folded into a bias, followed by a ReLU. With
`W_k = W[:, 3k:3k+3]` and frames `x`:

```
y_t = W_0 * x_{t-2} + W_1 * x_{t-1} + W_2 * x_t + b        (then ReLU)
```

**Caching the partial convolutions saves nothing.** Each new frame `x_t` is used three times, each time with a
different slice:

- `W_2 * x_t` at time t;
- `W_1 * x_t` at t+1;
- `W_0 * x_t` at t+2.

So every frame still needs three 3-channel convolutions; equivalently, one 3→192 convolution of the new frame. That is
64 × 9 × 49 multiply-adds per output pixel: **exactly the cost of the 9-channel convolution**.

Caching adds costs instead:

- three partial-sum tensors of 64×272×480, 100 MB in fp32;
- two extra full-resolution additions per frame.

On a CPU or WASM that extra memory traffic probably outweighs any gain. After the ReLU the frames are mixed, so no
later layer is reusable either.

**What the split does save is input bandwidth.**

- One new frame per step: 6.3 MB of float32, or 1.6 MB of uint8, instead of 18.8 MB.
- That is better done without changing the arithmetic: keep three normalised frames in a GPU ring buffer and either
  concatenate in-graph, or give the graph three inputs.
- An in-graph uint8 input needs the native EP. JSEP's `Cast` supports only fp16, fp32, int32, uint32 and bool
  ([cast.cc](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/providers/js/operators/cast.cc));
  the native EP's `Cast` handles uint8
  ([cast.cc](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/providers/webgpu/tensor/cast.cc)).

**A shared per-frame stem (late fusion) saves little.** Sharing would need retraining. With one 3-channel stem shared
across slots (1.23 GMACs) and three 1×1 mixes per slot, 2.83 GMACs replace 3.68: −2.2 % of the network. Fusing after
`layer1` saves about 5 %, because `layer1` has to run on every frame in any design. **Not worth it.**

### 4.3 Re-detecting the ball in a region of interest

- **The design.** The network is fully convolutional. Between full frames, run only a crop (for example 256×256, the
  three frames cropped at the same place) centred on the ball tracker's prediction.
  - Precedent: SiamFC uses "a search image centred at the previous position of the target", 255×255
    ([SiamFC](https://arxiv.org/abs/1606.09549)).
  - WASB's mean ball motion is about 17 px per frame at this scale, so ±128 px covers normal motion.
  - Run the whole frame every k frames, after every cut, and whenever the ball is lost.
- **Cost (computed).** A 256×256 crop is about 4.9 GMACs.
  - k = 2: 21.8 GMACs per frame on average (1.8x);
  - k = 3: 16.1 GMACs (2.4x).
- **Risks: high.**
  - Crop outputs differ from full-frame outputs: the FPN's top-down path from stride 32 sees only the crop.
  - Re-acquiring a lost ball adds latency.
  - People must coast on ByteTrack's predictions, or be detected every frame anyway.
  - The pipeline expects detections for every frame (`runtime.py`, `perception/pipeline.py`).
  - A cheap offline test comes first: compare ball heatmaps from crops against full frames on BARD clips.
- **Effort: very high.** Ranked low.

### 4.4 Other video methods, and why they fit poorly here

- **[Deep Feature Flow](https://arxiv.org/abs/1611.07715)**: ImageNet VID 73.9 mAP at 4.05 fps becomes 73.1 at
  20.25 fps (5.0x). It runs key frames plus a flow network that warps their features.
- **[Skip-Convolutions](https://arxiv.org/abs/2104.11487)**: "reduce their computational cost consistently by a factor
  of 3~4x … without any accuracy drop", counted in multiply-adds.
- **[DeltaCNN](https://arxiv.org/abs/2203.03996)**: "speedups of up to 7x" on GPUs.
- **[Looking Fast and Slow](https://arxiv.org/abs/1903.10172)**: interleaves cheap and expensive feature extractors.

All of these rely on features or pixels that change little between frames. Here the camera pans and zooms constantly,
the ball is a few fast pixels whose blur is the very cue the network uses, and ORT-web has no sparse or delta kernels.
Not recommended.

**Many-in, many-out (TrackNetV2-style).**

- Predict heatmaps for all three input frames in one pass, so a clip needs a third of the passes.
- TrackNetV2 reports 2.6 to 31.8 FPS, but from a smaller input and the many-in-many-out change together. I saw those
  numbers only in search-engine listings of the paper (**unverified**).
- TrackNetV5 counts 114 FPS as "38.12×3" ([arXiv 2512.02789](https://arxiv.org/abs/2512.02789)).
- The browser's clip analysis is offline, so outputs that use later frames are allowed. Very high effort; retraining;
  accuracy unknown.

### 4.5 System tricks in the browser (they need step 0's numbers)

- **Overlap the next frame's work with this frame's GPU run.** While `session.run` waits on the GPU, the worker's
  thread is free: seek and prepare frame t+1 then. On WebGPU devices whose CPU and GPU times are similar, frame time
  could improve by up to 1.5-2x (**estimate**). Results must still be committed in order. The desktop pipeline already
  overlaps decoding and preprocessing with the network (`analysis_speed.md` §2-3, `StudentDetector.prefetch`).
- **Remove one 18.8 MB copy.** The worker copies the Pyodide buffer (`new Float32Array(buffer.data)`) and then releases
  it. Wrapping the view directly and releasing it after `session.run` saves a copy, provided ORT copies or uploads the
  input itself.
- **Stop seeking for every frame.** One forum report measured 5-163 ms per `currentTime` seek (anecdotal). WebCodecs'
  `VideoDecoder` (Chrome 94, Firefox 130, Safari 16.4) or `requestVideoFrameCallback` (Chrome 83, Firefox 132,
  Safari 15.4) decode in order instead
  ([Chrome WebCodecs guide](https://developer.chrome.com/docs/web-platform/best-practices/webcodecs)).
- **Preprocess on the GPU.** Resize and normalise on the GPU (native EP, uint8 input). This is not bit-identical to
  `cv2.resize(INTER_AREA)` plus the LUT, so it needs the same parity checks.

---

## 5. Ranking of every idea, by expected speedup per unit of effort

| Rank | Idea | Backend | Expected speedup | Effort | Accuracy risk |
|---:|---|---|---|---|---|
| 1 | COOP/COEP plus ORT's default threads | WASM | 2.5-3.5x model (**estimate**) | XS | none |
| 2 | fp16 model when `shader-f16` exists | WebGPU | 1.2x reported, up to ~2x (**estimate**); half the download | S | low |
| 3 | Bundle per backend: native WebGPU EP; plain CPU wasm | both | **unverified** | XS-S | none |
| 4 | int8 plus sparse heads on the WASM fallback | WASM | 1.6-2.5x (**estimate**; sparse 1.23x measured natively) | M | low |
| 5 | Lean stride-4 neck and heads, distilled | all | ~1.3x CPU/WASM over the sparse path (**estimate**); multiply-adds 1.68x fewer than the dense graph | L | moderate |
| 6 | Overlap next-frame decoding and preparation with GPU inference | WebGPU end to end | up to 1.5-2x frame time if CPU ≈ GPU (**estimate**; needs step 0) | M | none if order is kept |
| 7 | Sparse heads on WebGPU (`feat` stays on GPU) | WebGPU | ≤ 1.23x | M | none |
| 8 | IO binding, frame ring buffer, graph capture | WebGPU | ≤ 1.1x (**estimate**) | M | none |
| 9 | Cache API, immutable caching, gzip for wasm | load time only | first load −22.5 MB; repeat visits skip 47 MB | XS-S | none |
| 10 | Structured pruning of `layer2`-`layer4` and the laterals | all | 1.2-1.4x (**estimate**) | L | moderate |
| 11 | Ball ROI crops, whole frame every k frames | all | 1.8x (k=2) to 2.4x (k=3) in multiply-adds | XL | high |
| 12 | Lighter backbone (RepVGG, MobileNetV4, …), distilled | all | ~1.7-2x realistic (**estimate**); at most 2.2x dense or 3.1x sparse even if the backbone were free | XL | moderate-high |
| 13 | Lower input resolution plus ball crop | all | 1.5-2.3x | XL | high |
| 14 | Many-in, many-out (3 frames per pass) | offline clips | up to 3x fewer passes | XL | unknown |
| 15 | Relaxed-SIMD custom ORT build (int8) | WASM | **unverified** (int8 GEMM ≤ ~2x more in theory) | M-L | none |
| — | Person head at a lower rate | | ≤ 3 % on the CPU path | S | low |
| — | Cached stem partial convolutions | | 0 % multiply-adds | S | none |
| — | Deep Feature Flow, Skip-Conv, DeltaCNN | | needs sparse kernels or flow | XL | high |

Effort: XS under a day; S 1-2 days; M 3-5 days; L 1-2 weeks; XL more.

---

## 6. The top five in detail

### #1. Cross-origin isolation, and threads for the WASM fallback

**Expected speedup.**

- WASM network time falls about 2.5-3.5x at ORT's default `min(4, ceil(hardwareConcurrency/2))` threads
  (**estimate**, from §2.2's sources). Machines with 2-3 logical cores get 2 threads.
- The `verify()` run at load gets the same speedup.
- WebGPU users gain nothing.

**Accuracy risk: none.** The kernels are the same; thread partitioning can only reorder floating-point sums, and
`verify()` still gates the backend.

**Files.**

- `courtvision/server/app.py`: in `_GuardMiddleware.__call__` → `send_with_headers`, append
  `(b"cross-origin-opener-policy", b"same-origin")` and `(b"cross-origin-embedder-policy", b"require-corp")`. They then
  apply to the page, the worker and ORT's pthread `.mjs`.
- `courtvision/server/static/nba-worker.js`:
  - drop `ort.env.wasm.numThreads = 1`, or set the thread count from step 0's measurements when
    `self.crossOriginIsolated` is true;
  - report `crossOriginIsolated` and `ort.env.wasm.numThreads` in the load result.
- `courtvision/server/static/nba-browser.js` and `index.html`: bump the `?v=20261101-progress1` cache-busting strings.
- `tests/test_server_app.py`: assert both headers beside the CSP assertions in
  `test_dashboard_index_and_assets_are_served`.
- `vercel.json`: no change today, since the showcase runs no inference. It would need the same headers if the app were
  ever hosted there.

**Measurement.**

1. Run the model-only harness (§7.2) with `ep=wasm`, threads ∈ {1, 2, 4, 8}, with and without the two headers.
   - Playwright can inject the headers with `context.route`, so you can A/B without touching the server.
   - Record the median, p10 and p90 of 20 `session.run` calls after 3 warm-ups, and confirm that
     `self.crossOriginIsolated === true`.
2. Run step 0 (§7.1) on a 10-second clip.
3. Check for regressions: `.venv/bin/python -m pytest tests/test_server_app.py tests/test_browser_pipeline.py -q` and
   `node --test tests/js/`.
4. Check the browser matrix by hand on the Mac (Chrome, Safari, Firefox), including that Pyodide still loads under
   COEP.

Time WASM in the container only after the training finishes. On 4 shared cores the numbers mean nothing.

### #2. An fp16 model for WebGPU

**Expected speedup.**

- Network time: 1.2x reported for a convolutional net on an M3 Max (img.ly); up to ~2x on GPUs with fast fp16
  (**estimate**).
- Download: 47.3 → ~23.7 MB.
- Session start-up: roughly halved (img.ly: 400 to 200 ms).
- With fp16 inputs, per-frame upload halves as well (optional).

**Accuracy risk: low to moderate.**

- Core ML fp16 left held-out ball F1 and person recall unchanged.
- WebGPU accumulates convolutions in f16 and rounds differently, so measure it.
- Keep the five 1×1 output convolutions in fp32. If needed, also keep the stem or other sensitive nodes in fp32
  (ORT's `auto_mixed_precision` can find them).

**Files.**

- `tools/export_student_browser.py`, or a new `tools/fp16_student.py`:
  - `convert_float_to_float16(model, keep_io_types=True, node_block_list=[f"/student/{h}/{h}.2/Conv" for h in OUTPUTS])`;
  - write `nba-student-fp16.onnx`, its sha256 in the manifest, and an fp16 reference (heatmaps compared after the
    sigmoid; tolerances from measurement).
- `courtvision/server/static/nba-worker.js`:
  - call `navigator.gpu.requestAdapter()` first, set `ort.env.webgpu.adapter`, and pick the fp16 file when
    `adapter.features.has('shader-f16')`, otherwise the fp32 file;
  - run `verify()` with that variant's reference.
- `courtvision/server/static/nba-student.json`: a `variants` list (file, sha256, precision).
- `courtvision/server/static/nba-validation*.json`: one reference per variant.
- `pyproject.toml`: add `onnxconverter-common` to the `onnx` extra.
- `tests/test_browser_pipeline.py`: manifest, variants and reference consistency.

**Measurement.**

1. **Speed.** WebGPU is not available in this container: no GPU, and Chrome on Linux supports only Intel Gen12+ GPUs.
   Run the model-only harness on the Mac with Playwright (`channel: 'chrome'`, headed). Test {fp32, fp16} ×
   {JSEP, native EP} (#3), 20 timed runs each, plus session-creation time.
2. **Accuracy.** Follow `tools/quantize_student.py`'s int8 protocol.
   - Dump 120 held-out inputs (`sample_inputs(Path('data/bard/split/check'), 120, seed)`, the check set
     `tools/quantize_student.py` used) as raw `.bin` files.
   - Run them through the browser harness and save the outputs.
   - Compare them with the fp32 CPU outputs using `quantize_student.compare`'s matching: ball within 15 px of 1080p,
     person boxes at IoU ≥ 0.5.
   - Gate at no worse than int8: person agreement ≥ 98 %, ball candidates ≥ 92 %, and the top ball candidate the same
     in ~95 % of frames.
3. **If agreement is borderline,** score the BARD ball check frames through the harness as well. That costs 1,153
   inputs, roughly 22 GB as float32, so stream them.

### #3. One ORT bundle per backend

**Expected speedup: unverified.**

- I found no published JSEP-against-native comparison.
- In favour of the change:
  - JSEP is deprecated (1.29);
  - the native EP gets the convolution work (§2.1);
  - the plain CPU binary has no Asyncify instrumentation (Emscripten estimates "~50%" overhead on instrumented code)
    and is half the size (14.8 against 29.5 MB).
- The move is needed eventually anyway.

**Accuracy risk: none.** `verify()` against `nba-validation.json` gates each backend, as it does today.

**Files.**

- `courtvision/server/static/nba-worker.js`:
  - probe for a WebGPU adapter, then `importScripts("ort.webgpu.min.js")`, or `ort.jspi.min.js` when
    `"Suspending" in WebAssembly`; otherwise `importScripts("ort.wasm.min.js")`;
  - `wasmPaths` as today.
- `courtvision/server/static/nba-browser.js`: when WebGPU fails `verify()`, restart the worker in WASM mode (ORT
  initialises once per worker).
- `courtvision/server/static/`: add `ort.webgpu.min.js` with `ort-wasm-simd-threaded.asyncify.{mjs,wasm}` (and/or the
  JSPI pair), and `ort.wasm.min.js` with `ort-wasm-simd-threaded.{mjs,wasm}`. Use the same 1.31.1 build: a bundle and
  its wasm must come from the same build. Keep the JSEP files until the A/B test is decided.
- `tests/js/nba-browser.test.cjs`: the restart path.

**Measurement.**

- WebGPU: model-only harness, {JSEP, native-asyncify, native-JSPI} × {fp32, fp16} on the Mac, and on a Windows machine
  if one is available.
- WASM: {jsep.wasm, plain wasm} × threads {1, 4}, in the container after training and on the Mac.
- Record session-creation time and steady-state time.
- Then run step 0 end to end.

### #4. The native CPU fast path on the WASM fallback (int8 and sparse heads)

**Expected speedup.**

- Sparse heads: 1.23x, measured natively (129.8 → 105.9 ms).
- int8 on WASM: **unverified**. Natively it gives 2.1x (105.9 → 49.6 ms), but on VNNI/AMX hardware. The fixed-SIMD
  kernel's instruction mix suggests at most about 2x.
- Combined: 1.6-2.5x on WASM (**estimate**). Together with #1: about 4-8x over today's single-threaded fp32 fallback
  (**estimate**).
- The WASM visitor's download falls 47.3 → 12.7 MB.

**Accuracy risk.**

- Sparse heads: none (≤ 1.4e-6).
- int8: low. Round 4b: ball hit unchanged (89.25 %, p 1); person agreement 98.0 %; extra low-score ball candidates
  near the threshold.
- Re-run both checks for main model v2.

**Files.**

- `tools/quantize_student.py`: quantize the current `courtvision/server/static/nba-student.onnx` (main model v2) and
  write its browser reference with `export_student_browser.write_browser_reference` on the int8 session.
- `tools/export_student_browser.py`: a `--trunk` option. It cuts the graph as `student_onnx.OnnxStudent._split` does
  (outputs `ball_hm`, `person_hm`, `feat`) for fp32 and int8, and saves the three `SparseHead` weight sets to
  `nba-student-heads.npz`.
- `courtvision/browser/runtime.py`: with trunk outputs, `finish()` finds the peaks with `student_onnx._peaks` and calls
  `student_onnx.decode_sparse`, which already ships in `browser-python.zip`, with the head weights loaded once.
- `courtvision/server/static/nba-worker.js`:
  - on WASM, load the int8 trunk;
  - after `run`, Python returns the peak cells, JS gathers the k×1152 patches from `outputs.feat.data`, and Python
    finishes. This avoids `toPy` on 16.7 MB.
- `courtvision/browser/build.py`: package or serve the `.npz`.
- Manifests and references for the variants.
- `tests/test_browser_pipeline.py`: sparse `finish` equals dense `finish` on a fixture.
- `tests/test_perception_student_onnx.py`.

**Measurement.**

1. **Native, first.** These tools exist today:
   - `.venv/bin/python tools/quantize_student.py --src courtvision/server/static/nba-student.onnx --calib data/calib_mix --check data/bard/split/check --method percentile --out runs/models/nba-student-v2-int8.onnx --report runs/bench/int8_v2.json`
     (the calibration and check folders that `runs/bench/int8_r4b.json` used)
   - `.venv/bin/python tools/student_speed.py --frames data/bard/split/check --onnx courtvision/server/static/nba-student.onnx --int8 runs/models/nba-student-v2-int8.onnx --threads 4 --json runs/bench/speed_v2.json`
   - `.venv/bin/python tools/detector_bench.py --models courtvision/server/static/nba-student.onnx,runs/models/nba-student-v2-int8.onnx --sparse --ball-folder data/ft/bard_check_teacher,data/ft/bard_check2_teacher --json runs/bench/ball_v2_int8.json`
     (top-candidate hit rate with a McNemar test; add `--link` for the README's linked-path protocol)
   - `.venv/bin/python tools/detector_bench.py --models … --sparse --coco data/hfcoco/heldout_ann.json --json runs/bench/coco_v2_int8.json`
     (the 5 held-out Roboflow games)
2. **Browser.** Model-only harness, `ep=wasm`, threads {1, 4}, models {fp32 dense, fp32 trunk, int8 trunk}. Include
   the JS gather and the Pyodide `decode_sparse` in the timing. Then run step 0.

### #5. A leaner stride-4 neck and heads, distilled from main model v2

**Expected speedup.**

- Multiply-adds: 31.48 → 23.03 GMACs, 1.37x fewer than today's CPU sparse path and 1.68x fewer than the dense graph.
- Compute-bound CPU and WASM: about 1.3x (**estimate**).
- WebGPU: less (depthwise layers are memory-bound, §3.3).
- This is the main remaining lever for the native CPU path, which already runs sparse heads and int8. It stacks with
  int8.

**Accuracy risk: moderate.** These layers carry the ball's local 3×3 context at stride 4. Mitigations:

- distil from v2's heatmap logits, offsets and sizes;
- freeze the trunk at first;
- keep the sparse heads' design;
- if the ball suffers, keep `ball_hm` a full 3×3 and slim only `smooth` and `person_hm`.

**Files.**

- `courtvision/perception/student.py`: `build_net(..., neck="dw")`.
- `courtvision/training/train_student.py`: a distillation term.
- `tools/student_finetune.py`: flags.
- `courtvision/training/student_from_onnx.py`: load the teacher.
- `courtvision/perception/student_onnx.py`: sparse-head discovery uses the `/{head}.0/Conv` and `/{head}.2/Conv`
  names. Unchanged if the sparse heads keep their structure, but test it.
- `tools/quantize_student.py`: `HEAD_OUTPUT_CONVS` and `SPARSE_HEAD_CONVS` node names.
- `tools/export_student_browser.py`.
- `coreml_model` (generic trace): re-check it on the Mac.

**Measurement.**

- Speed: `student_speed.py` (fp32 and int8, sparse).
- Accuracy against v2 with `detector_bench.py`:
  - `--ball-folder data/ft/bard_check_teacher,data/ft/bard_check2_teacher` (McNemar), with and without `--link`;
  - `--coco data/hfcoco/heldout_ann.json`;
  - `--split val --stride 10` for agreement on players.
- On the Mac: the README's two-game protocol (`tools/nba_ball_bench.py`) and `train_student.evaluate`'s held-out
  frames, as for every earlier fine-tune.
- Gate: no McNemar-significant ball loss, and players' AP50 within noise on the held-out Roboflow games.

---

## 7. The shared measurement harness

### 7.1 Step 0: where the browser's frame time goes

Add `performance.now()` stamps:

- in `nba-worker.js` `process`: before and after `pipeline.prepare`, tensor creation, `session.run`, `python.toPy` and
  `pipeline.finish`;
- in `nba-clip.js`: around `seek()` and `getImageData`.

Return them as `result.timing`. Drive the Studio's public demo with Playwright on a 10-second H.264 clip, uploaded with
`setInputFiles`. Collect the timings through a `page.addInitScript` wrapper of `window.CVNBA.load`, or through the
console.

Report median per-frame milliseconds per stage for WebGPU (on the Mac) and for WASM. This shows whether the model or
the pipeline (seeks, copies, Python court tracking) limits each backend.

In the container, the Pyodide load needs jsDelivr through the agent proxy:
`chromium.launch({proxy: {server: process.env.HTTPS_PROXY}})`. The proxy CA is already in the NSS store
(`/root/.ccr/README.md`).

### 7.2 A model-only harness (not part of the app)

A scratch folder holds:

- copies of the ORT bundles and wasm files;
- the model variants;
- `bench.html` and `bench.js`. The CSP needs external same-origin scripts; there is no inline JS.

It is served by a 15-line `http.server` subclass that can add COOP/COEP. `bench.js` starts a Worker that:

1. imports the chosen bundle;
2. sets `ort.env.wasm.numThreads` and the EP;
3. creates the session (timed);
4. builds `nba-worker.js`'s deterministic input `((i*13+(i>>>10)*7)%251-125)/128`;
5. runs 3 warm-ups and 20 timed `session.run` calls;
6. posts {`crossOriginIsolated`, `numThreads`, EP, `adapter.info`, `has_f16`, create ms, median/p10/p90 ms}.

The driver uses the preinstalled Playwright (Node 1.63, `PLAYWRIGHT_BROWSERS_PATH=/opt/pw-browsers`, Chrome for Testing
153):

```js
const { chromium } = require('playwright');
const browser = await chromium.launch();              // headless Chromium 153 in the container
const context = await browser.newContext();
// optional A/B without touching the server: add cross-origin isolation to every response
await context.route('**/*', async route => {
  const response = await route.fetch();
  await route.fulfill({ response, headers: { ...response.headers(),
    'cross-origin-opener-policy': 'same-origin', 'cross-origin-embedder-policy': 'require-corp' } });
});
const page = await context.newPage();
await page.goto('http://127.0.0.1:8765/bench.html?bundle=ort.wasm.min.js&ep=wasm&threads=4&model=nba-student.onnx');
const result = await (await page.waitForFunction(() => window.benchResult, null, { timeout: 0 })).jsonValue();
console.log(JSON.stringify(result));
```

On the Mac, use `chromium.launch({ channel: 'chrome', headless: false })` for real WebGPU. Whether WebGPU works in
headless mode there is **unverified**.

Accuracy runs reuse the same page. They fetch the `.bin` inputs dumped by `quantize_student.sample_inputs` and return
the raw outputs, which are scored in Python with `quantize_student.compare`'s matching.

### 7.3 Native CPU tools (when the 4 cores are free)

- Speed: `tools/student_speed.py` (median of N, 4 threads; on an otherwise idle machine).
- Accuracy:
  - `tools/quantize_student.py` (agreement report);
  - `tools/detector_bench.py` with `--ball-folder` (McNemar), `--link`, `--coco data/hfcoco/heldout_ann.json` and
    `--split val`.
- Store the outputs under `runs/bench/` as earlier rounds did.

---

## 8. Not verified, and open questions

- No browser timings exist yet for this model, on any backend. Every browser speedup above is reported for other models
  or estimated.
- JSEP against the native WebGPU EP for this convolutional network: no published comparison found.
- int8 against fp32 speed on ORT-web's WASM backend: no published benchmark found. The ~2x per instruction comes from
  reading the kernel source.
- Whether Asyncify slows MLAS's kernels in the JSEP and asyncify builds: not checked.
- fp16 accuracy with WebGPU's f16 accumulation for this network: not measured. Core ML fp16 is supporting evidence, not
  proof.
- Whether ORT constant-folds the export's `Shape→Slice→Concat→Resize` chains under JSEP: not checked.
- Relaxed-SIMD gains: not measured; a custom ORT build is required.
- TrackNetV2's 2.6 → 31.8 FPS comes from search-result listings; I did not open the paper.
- The anecdotal per-seek cost of 5-163 ms comes from a forum post.
- `onnxruntime.ai`, `learn.microsoft.com` and `github.com` release pages were read with curl or WebFetch. Some hosts
  (`hyper.ai`, Papers with Code mirrors) could not be reached.

---

## 9. Sources

### onnxruntime and onnxruntime-web

- Docs:
  [env flags and session options](https://onnxruntime.ai/docs/tutorials/web/env-flags-and-session-options.html) ·
  [WebGPU EP (web)](https://onnxruntime.ai/docs/tutorials/web/ep-webgpu.html) ·
  [performance diagnosis](https://onnxruntime.ai/docs/tutorials/web/performance-diagnosis.html) ·
  [deploy](https://onnxruntime.ai/docs/tutorials/web/deploy.html) ·
  [large models](https://onnxruntime.ai/docs/tutorials/web/large-models.html) ·
  [WebGPU EP (native)](https://onnxruntime.ai/docs/execution-providers/WebGPU-ExecutionProvider.html) ·
  [get started, web](https://onnxruntime.ai/docs/get-started/with-javascript/web.html) ·
  [float16](https://onnxruntime.ai/docs/performance/model-optimizations/float16.html) ·
  [WebGpuExecutionProviderOption](https://onnxruntime.ai/docs/api/js/interfaces/InferenceSession.WebGpuExecutionProviderOption.html) ·
  [build for web](https://onnxruntime.ai/docs/build/web.html)
- Release notes:
  [v1.29.0](https://github.com/microsoft/onnxruntime/releases/tag/v1.29.0) ·
  [v1.30.0](https://github.com/microsoft/onnxruntime/releases/tag/v1.30.0) ·
  [v1.31.0](https://github.com/microsoft/onnxruntime/releases/tag/v1.31.0) ·
  [v1.31.1](https://github.com/microsoft/onnxruntime/releases/tag/v1.31.1) ·
  [WebGPU plugin EP v0.5.0](https://github.com/microsoft/onnxruntime/releases/tag/plugin-ep-webgpu/v0.5.0)
- Source code (main branch, read 2026-11-05):
  [js/web/script/build.ts](https://github.com/microsoft/onnxruntime/blob/main/js/web/script/build.ts) ·
  [js/common/lib/env.ts](https://github.com/microsoft/onnxruntime/blob/main/js/common/lib/env.ts) ·
  [webgpu-operators.md (JSEP)](https://github.com/microsoft/onnxruntime/blob/main/js/web/docs/webgpu-operators.md) ·
  [webgpu_execution_provider.cc](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/providers/webgpu/webgpu_execution_provider.cc) and
  [.h](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/providers/webgpu/webgpu_execution_provider.h) ·
  [js_execution_provider.h](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/providers/js/js_execution_provider.h) ·
  [JSEP cast.cc](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/providers/js/operators/cast.cc) ·
  [WebGPU cast.cc](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/providers/webgpu/tensor/cast.cc) ·
  [mlasi.h](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/mlas/lib/mlasi.h) ·
  [qgemm_kernel_wasmsimd.cpp](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/mlas/lib/qgemm_kernel_wasmsimd.cpp) ·
  [qgemm_kernel_wasmrelaxedsimd.cpp](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/mlas/lib/qgemm_kernel_wasmrelaxedsimd.cpp) ·
  [onnxruntime_mlas.cmake](https://github.com/microsoft/onnxruntime/blob/main/cmake/onnxruntime_mlas.cmake) ·
  [onnxruntime_webassembly.cmake](https://github.com/microsoft/onnxruntime/blob/main/cmake/onnxruntime_webassembly.cmake) ·
  [build_args.py](https://github.com/microsoft/onnxruntime/blob/main/tools/ci_build/build_args.py)
- Package:
  [onnxruntime-web 1.31.1 file list](https://data.jsdelivr.com/v1/packages/npm/onnxruntime-web@1.31.1) ·
  [package.json exports](https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.1/package.json)
- Microsoft Learn: [Windows ML WebGPU EP](https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/webgpu-ep)
  (dated 2026-08-05)
- Blogs:
  [ORT Web launch, 2021](https://opensource.microsoft.com/blog/2021/09/02/onnx-runtime-web-running-your-machine-learning-model-in-browser/) ·
  [ORT Web WebGPU, 2024](https://opensource.microsoft.com/blog/2024/02/29/onnx-runtime-web-unleashes-generative-ai-in-the-browser-using-webgpu/) ·
  [TF.js WASM SIMD and threads, 2020](https://blog.tensorflow.org/2020/09/supercharging-tensorflowjs-webassembly.html) ·
  [img.ly background removal with ORT WebGPU](https://img.ly/blog/browser-background-removal-using-onnx-runtime-webgpu/) ·
  [EfficientNet-Lite](https://blog.tensorflow.org/2020/03/higher-accuracy-on-vision-models-with-efficientnet-lite.html)

### Web platform

- [web.dev cross-origin isolation guide](https://web.dev/articles/cross-origin-isolation-guide) ·
  [MDN COEP](https://developer.mozilla.org/en-US/docs/Web/HTTP/Reference/Headers/Cross-Origin-Embedder-Policy)
- [MDN browser-compat-data](https://github.com/mdn/browser-compat-data): COEP, COOP, SharedArrayBuffer,
  fixed-width SIMD, relaxed SIMD, threads and atomics, JSPI, GPU, GPUSupportedFeatures, Worker, Float16Array, Cache,
  StorageManager, VideoDecoder, `requestVideoFrameCallback`, WebNN (still behind a flag in Chrome)
- [Emscripten Asyncify](https://emscripten.org/docs/porting/asyncify.html) ·
  [Chrome WebCodecs guide](https://developer.chrome.com/docs/web-platform/best-practices/webcodecs)
- jsDelivr Pyodide response headers: checked with `curl -I` on 2026-11-05 (`pyodide.mjs`, `pyodide.asm.wasm`,
  `pyodide-lock.json`)

### Papers

- Detection and sports balls:
  [CenterNet](https://arxiv.org/abs/1904.07850) ·
  [TrackNet](https://arxiv.org/abs/1907.03698) ·
  [WASB](https://arxiv.org/abs/2311.05237) ·
  [TrackNetV5 (describes TrackNetV2's many-in, many-out)](https://arxiv.org/abs/2512.02789) ·
  [SAHI](https://arxiv.org/abs/2202.06934) ·
  [SiamFC](https://arxiv.org/abs/1606.09549)
- Pruning: [DepGraph](https://arxiv.org/abs/2301.12900) ·
  [Pruning filters](https://arxiv.org/abs/1608.08710) ·
  [Network slimming](https://arxiv.org/abs/1708.06519) ·
  [Group Fisher pruning](https://arxiv.org/abs/2108.00708)
- Architectures: [RepVGG](https://arxiv.org/abs/2101.03697) ·
  [MobileNetV4](https://arxiv.org/abs/2404.10518) ·
  [MobileNetV3](https://arxiv.org/abs/1905.02244) ·
  [ShuffleNet V2](https://arxiv.org/abs/1807.11164)
- Distillation: [FGD](https://arxiv.org/abs/2111.11837) ·
  [Fine-grained feature imitation](https://arxiv.org/abs/1906.03609) ·
  [Localization Distillation](https://arxiv.org/abs/2102.12252)
- Video: [Deep Feature Flow](https://arxiv.org/abs/1611.07715) ·
  [Skip-Convolutions](https://arxiv.org/abs/2104.11487) ·
  [DeltaCNN](https://arxiv.org/abs/2203.03996) ·
  [Looking Fast and Slow](https://arxiv.org/abs/1903.10172) ·
  [Mobile VOD with temporally-aware feature maps](https://arxiv.org/abs/1711.06368) ·
  [Video OD for mobiles](https://arxiv.org/abs/1804.05830) ·
  [AdaScale](https://arxiv.org/abs/1902.02910)

### This repository

- `courtvision/perception/student.py`, `student_onnx.py`
- `courtvision/server/static/nba-worker.js`, `nba-browser.js`, `nba-clip.js`
- `courtvision/browser/runtime.py`, `build.py`
- `courtvision/server/app.py`
- `vercel.json`, `.vercelignore`, `docs/DEPLOY_VERCEL.md`
- `tools/student_speed.py`, `tools/detector_bench.py`, `tools/quantize_student.py`, `tools/export_student_browser.py`
- `docs/research/cross_game_detector.md` §2-3, §5, §9-10
- `docs/research/analysis_speed.md`
- `runs/bench/speed_4t.json`, `int8_r4b.json`, `ball_check12_r4b_int8.log`
