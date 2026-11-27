'use strict';
// CPU-only structural/data checks; this is not browser layout verification.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

class Element {
  constructor(tag) { this.tag = tag; this.children = []; this.attributes = {}; this._text = ''; }
  set textContent(value) { this._text = String(value); this.children = []; }
  get textContent() { return this._text + this.children.map((child) => child.textContent).join(''); }
  append(...values) { values.forEach((value) => { if (typeof value === 'string') { const node = new Element('#text'); node.textContent = value; this.children.push(node); } else this.children.push(value); }); }
  replaceChildren(...values) { this.children = []; this._text = ''; this.append(...values); }
  setAttribute(name, value) { this.attributes[name] = String(value); }
}
const ids = new Map(['retry-title', 'retry-copy', 'retry-intro', 'retry-results'].map((name) => [name, new Element('div')]));
const document = {createElement: (tag) => new Element(tag), getElementById: (id) => ids.get(id), createTextNode: (value) => { const node = new Element('#text'); node.textContent = value; return node; }};
const text = (tag, value, className) => { const element = document.createElement(tag); element.textContent = value; if (className) element.className = className; return element; };
// Development copies keep assets beside this helper in dist/. The committed
// repository serves the same files from docs/, alongside the website/ tools.
const localDist = path.join(__dirname, 'dist');
const siteRoot = fs.existsSync(localDist) ? localDist : path.resolve(__dirname, '../docs');
const source = fs.readFileSync(path.join(siteRoot, 'assets/app.js'), 'utf8');
const start = source.indexOf('  function renderRetry(');
const end = source.indexOf('  function explorer(', start);
assert(start >= 0 && end > start);
const context = vm.createContext({document, text, $: document.getElementById});
vm.runInContext(source.slice(start, end) + '\nthis.render = renderRetry;', context);
const fixtureMode = process.argv[2] === '--fixture';
const dataPath = fixtureMode ? process.argv[3] : path.join(siteRoot, 'results.json');
assert(dataPath);
const retry = JSON.parse(fs.readFileSync(dataPath, 'utf8')).retry;
assert(retry.published_results && retry.runs.length);
const walk = (node) => [node, ...node.children.flatMap(walk)];

for (let repeat = 0; repeat < 2; repeat++) {
  context.render(retry);
  const cards = ids.get('retry-results').children;
  assert.equal(cards.length, retry.runs.length);
  cards.forEach((card, index) => {
    const run = retry.runs[index]; const all = walk(card);
    assert.equal(card.tag, 'article');
    assert.equal(all.find((node) => node.tag === 'h4').textContent, run.short_label || run.label);
    assert.deepEqual(all.filter((node) => node.tag === 'dd').slice(0, 2).map((node) => node.textContent), [String(run.sft_updates), String(run.optimizer_updates)]);
    const bodies = all.filter((node) => node.tag === 'tbody');
    const names = all.filter((node) => node.tag === 'ul' && node.className === 'cohort-workloads');
    Object.entries(run.cohorts).forEach(([cohort, methods], cohortIndex) => {
      assert.deepEqual(bodies[cohortIndex].children.map((row) => row.children.map((cell) => cell.textContent)), methods.map((method) => [method.method, `${method.passed} / ${method.total}`, `k=${method.budget}`]));
      assert.deepEqual(names[cohortIndex].children.map((item) => item.textContent), run.cohort_ops[cohort]);
    });
    assert(all.some((node) => node.tag === 'details' && node.className === 'study-provenance'));
    assert(card.textContent.includes(run.source_commit));
  });
  assert.equal(ids.get('retry-copy').textContent, retry.note);
}

const fixture = structuredClone(retry);
fixture.runs[0].cohort_ops.fresh[0] = '<img src=x onerror=alert(1)>';
context.render(fixture);
const nodes = walk(ids.get('retry-results'));
assert(nodes.some((node) => node.tag === 'code' && node.textContent === '<img src=x onerror=alert(1)>'));
assert(!nodes.some((node) => node.tag === 'img'));
const longLabel = structuredClone(retry);
longLabel.runs[longLabel.runs.length - 1].label = 'TEST ONLY long model and placement label '.repeat(20);
context.render(longLabel);
assert(walk(ids.get('retry-results')).some((node) => node.textContent.includes(longLabel.runs[longLabel.runs.length - 1].label)));
console.log(`${fixtureMode ? 'TEST ONLY synthetic' : 'Actual'} ${retry.runs.length}-study cards preserve all method rows, counts, budgets, workload identities and sources; rerender, long-label and safe-text fixtures pass. Browser layout QA remains separate.`);
