const assert = require('node:assert/strict');
const {readFileSync} = require('node:fs');
const {resolve} = require('node:path');
const {test} = require('node:test');
const vm = require('node:vm');
const source = readFileSync(resolve(__dirname,'../../courtvision/server/static/nba-browser.js'),'utf8');

test('setup progress preserves pending RPCs and calibration reports to the current clip', async () => {
  let worker;
  class Worker {
    constructor(){worker=this;this.sent=[];}
    postMessage(data){this.sent.push(data);}
    terminate(){}
    reply(data){this.onmessage({data});}
  }
  const window={addEventListener(){}};
  vm.runInNewContext(source,{window,Worker});
  const loading=[],court=[];
  const pending=window.CVNBA.load({progress:event=>loading.push(event)});
  const id=worker.sent[0].id;
  worker.reply({id,progress:{stage:'model',completed:20,total:100}});
  worker.reply({id,progress:{stage:'runtime',completed:1,total:3}});
  worker.reply({id,result:{backend:'webgpu',validated:true}});
  const model=await pending;
  assert.equal(model.validated,true);
  assert.deepEqual(loading.map(p=>p.stage),['model','runtime']);
  const calibration=model.calibrate([{data:new Uint8ClampedArray(4)}],1,1,event=>court.push(event));
  const calibrationId=worker.sent.at(-1).id;
  for(let completed=0;completed<=7;completed++)worker.reply({id:calibrationId,progress:{stage:'court',completed,total:7}});
  worker.reply({id:calibrationId,result:{calibrated:true}});
  assert.equal((await calibration).calibrated,true);
  assert.deepEqual(court.map(p=>p.completed),[0,1,2,3,4,5,6,7]);
  assert.equal(loading.length,2); // the new clip's progress never uses the first load's callback
  model.dispose();
});
