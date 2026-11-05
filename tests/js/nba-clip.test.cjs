const assert = require('node:assert/strict');
const {readFileSync} = require('node:fs');
const {resolve} = require('node:path');
const {test} = require('node:test');
const vm = require('node:vm');
const source = readFileSync(resolve(__dirname,'../../courtvision/server/static/nba-clip.js'),'utf8');
const settle = () => new Promise(resolve => setImmediate(resolve));
function harness() {
  let time=0;
  const listeners=new Map();
  const decoder={readyState:2,seeking:false,videoWidth:2,videoHeight:1,pause(){},load(){},removeAttribute(){},
    addEventListener(name,fn){if(!listeners.has(name))listeners.set(name,new Set());listeners.get(name).add(fn);},
    removeEventListener(name,fn){listeners.get(name)?.delete(fn);},
    get currentTime(){return time;},set currentTime(t){time=t;queueMicrotask(()=>{for(const fn of listeners.get('seeked')||[])fn();});},
  };
  let drawn;
  const canvas={getContext:()=>({drawImage(v){drawn=v.currentTime;},getImageData:()=>({width:2,height:1,data:new Uint8ClampedArray(8).fill(Math.round(drawn*1000))})})};
  const window={};
  vm.runInNewContext(source,{window,setTimeout,clearTimeout,ImageData:class {constructor(data,width,height){Object.assign(this,{data,width,height});}},
    document:{createElement:tag=>tag==='video'?decoder:canvas}});
  const frames=[], calls=[], calibrations=[];let resolveFirst,complete=false,fail;
  const model={reset:async()=>{},calibrate:async views=>{calibrations.push(views.map(v=>v.data[0]));},process:async(raw,index,t)=>{
    calls.push({index,t,pixel:raw.data[0]});
    if(index===0)await new Promise(resolve=>{resolveFirst=resolve;});
    return {frame_idx:index,t,detections:[]};
  }};
  const clip=window.CVNBAClip.create({src:'blob:clip',duration:.11,videoWidth:2,videoHeight:1},model,{
    phase(){},state(){},frame:(result,pixels)=>frames.push({result,pixel:pixels.data[0]}),
    complete(){complete=true;},error(error){fail=error;},
  });
  return {clip,calls,frames,calibrations,get complete(){return complete;},get fail(){return fail;},finishFirst:()=>resolveFirst()};
}

test('analysis preserves frame order and displays the exact pixels sent to inference across a pause',async()=>{
  const h=harness(),running=h.clip.run();await settle();
  assert.equal(h.calls.length,1);h.clip.pause();h.finishFirst();await settle();
  assert.equal(h.calls.length,1);assert.equal(h.frames.length,1);
  h.clip.resume();await running;
  assert.equal(h.fail,undefined);assert.equal(h.complete,true);assert.equal(h.frames.length,4);
  assert.deepEqual(h.calls.map(c=>c.index),[0,1,2,3]);
  assert.deepEqual(h.frames.map(f=>f.pixel),h.calls.map(c=>c.pixel));
  assert.deepEqual(h.frames.map(f=>f.result.t),h.calls.map(c=>c.t));
  assert.equal(h.clip.at(2/29.97+.01).frame_idx,2);
});

test('closing during inference cannot publish its result or start another frame',async()=>{
  const h=harness(),running=h.clip.run();await settle();
  h.clip.close();h.finishFirst();await running;
  assert.equal(h.calls.length,1);assert.equal(h.frames.length,0);assert.equal(h.complete,false);
  assert.equal(h.fail,undefined);
});

test('court setup sends six views a step apart, then the six between them',async()=>{
  const h=harness(),running=h.clip.run();await settle();h.finishFirst();await running;
  assert.equal(h.calibrations.length,1);
  const seen=h.calibrations[0];  // each view's pixel is its seek time in ms (the clip lasts 0.11 s)
  assert.equal(seen.length,12);
  assert.equal(seen[0],0);assert.equal(seen[5],109);  // the last view just before the end
  for(let i=0;i<5;i++){
    assert.ok(seen[i]<seen[i+1]);
    assert.ok(seen[i]<seen[6+i]&&seen[6+i]<seen[i+1]);  // between two views, in the same order
  }
  assert.equal(seen[11],109);  // past the end: the last frame again
});
