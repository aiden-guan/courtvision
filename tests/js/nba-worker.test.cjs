const assert = require('node:assert/strict');
const {readFileSync} = require('node:fs');
const {resolve} = require('node:path');
const {test} = require('node:test');
const vm = require('node:vm');
const source = readFileSync(resolve(__dirname,'../../courtvision/server/static/nba-worker.js'),'utf8');

function threads(crossOriginIsolated, hardwareConcurrency) {
  const context = {URL, self: {crossOriginIsolated, navigator: {hardwareConcurrency}, location: {href: 'http://127.0.0.1/static/nba-worker.js'}}};
  context.importScripts = () => { context.ort = {env: {wasm: {}}}; };
  vm.runInNewContext(source, context);
  return context.ort.env.wasm.numThreads;
}

test('WebAssembly threads only on a cross-origin isolated page, at most four', () => {
  assert.equal(threads(false, 16), 1); // no SharedArrayBuffer: ORT would fall back to one thread anyway
  assert.equal(threads(true, 16), 4);
  assert.equal(threads(true, 2), 2);
  assert.equal(threads(true, undefined), 1);
});
