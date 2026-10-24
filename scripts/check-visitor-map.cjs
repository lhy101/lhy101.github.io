const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const path = require('node:path');
const source = fs.readFileSync(path.join(__dirname, '../assets/js/visitor-map.js'), 'utf8');

function setup(host = 'lhy101.github.io', siteId = 'map+id/abc=') {
  const images = [], timers = new Map();
  let time = 0, timerId = 0;
  const canvas = {hidden: true, replaceChildren(node) {this.child = node;}};
  const status = {hidden: true};
  const retry = {hidden: true, addEventListener(event, callback) {this.click = callback;}};
  const map = {hidden: false, dataset: {siteId, siteHost: 'lhy101.github.io', statsUrl: 'https://mapmyvisitors.com/web/1c91f'},
    querySelector(selector) {return ({'[data-visitor-map-canvas]': canvas, '[data-visitor-map-status]': status, '[data-visitor-map-retry]': retry})[selector];}};
  const document = {querySelector: () => map, createElement(tag) {
    const element = {appendChild(child) {this.child = child;}, removeAttribute(name) {delete this[name];}};
    if (tag === 'img') images.push(element);
    return element;
  }};
  const window = {location: {hostname: host}, setTimeout(callback, delay) {timers.set(++timerId, {callback, at: time + delay}); return timerId;}, clearTimeout(id) {timers.delete(id);}};
  function advance(ms) {
    const target = time + ms;
    while (true) {
      const next = [...timers].sort((a,b) => a[1].at - b[1].at)[0];
      if (!next || next[1].at > target) break;
      time = next[1].at; timers.delete(next[0]); next[1].callback();
    }
    time = target;
  }
  vm.runInNewContext(source, {document, window, Date});
  return {map, canvas, status, retry, images, timers, advance};
}

for (const host of ['localhost', '127.0.0.1', 'preview.example']) {
  const local = setup(host);
  assert.equal(local.map.hidden, true);
  assert.equal(local.images.length, 0, 'Local previews must not send tracking requests');
}
assert.equal(setup('lhy101.github.io', '').images.length, 0);
const success = setup();
assert.match(success.images[0].src, /d=map%2Bid%2Fabc%3D&cl=ffffff$/);
success.images[0].onload();
success.advance(60000);
assert.equal(success.canvas.hidden, false);
assert.equal(success.canvas.child.href, success.map.dataset.statsUrl);
assert.equal(success.images.length, 1, 'A successful load must not retry and double count');
assert.equal(success.status.hidden, true);

const transient = setup();
transient.images[0].onerror();
transient.advance(1000);
transient.images[1].onload();
assert.equal(transient.canvas.hidden, false);
assert.equal(transient.retry.hidden, true);

const failed = setup();
failed.images[0].onerror();
failed.advance(1000);
failed.images[1].onerror();
failed.advance(60000);
assert.equal(failed.images.length, 2, 'Retries must stop after persistent failures');
assert.equal(failed.map.hidden, false, 'Network failure must not hide Visitors');
assert.equal(failed.retry.hidden, false);
assert.match(failed.status.textContent, /temporarily unavailable/);
failed.retry.click();
assert.match(failed.images[2].src, /&retry=/);
failed.images[2].onload();
assert.equal(failed.canvas.hidden, false);
assert.equal(failed.status.hidden, true);

const stalled = setup();
const lateLoad = stalled.images[0].onload;
stalled.advance(17000);
assert.equal(stalled.retry.hidden, false, 'Stalled requests must reach a retryable state');
lateLoad();
assert.equal(stalled.canvas.hidden, true, 'A stale callback must not replace the failure state');
assert.equal(stalled.timers.size, 0);
console.log('PASS: visitor map success, bounded retry, persistent failure, timeout, late callback, manual recovery, and local tracking guard.');
