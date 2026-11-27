'use strict';
(() => {
  const $ = (id) => document.getElementById(id);
  const text = (tag, value, className) => { const el = document.createElement(tag); el.textContent = value; if (className) el.className = className; return el; };
  const ratio = (value) => Number.isFinite(value) ? `${value.toFixed(3)}×` : 'Unavailable';
  const stages = {
    spec: ['OPERATOR CONTRACT', 'Make the task unambiguous.', 'Every registered workload supplies a callable signature, deterministic inputs, reference math, tolerances, and analytical cost estimates. These define the candidate’s task before any timing is considered.', 'signature + shapes + dtype\n+ canonical and stress inputs\n+ numerical tolerances\n= a reproducible task'],
    generate: ['CURRENT POLICY', 'Produce code, retain the trajectory.', 'A local coding model samples self-contained Triton source. Raw completion tokens, formatted-prompt identities, and old/reference token probabilities accompany the candidate. Search is kept distinct from weight training.', 'contract-derived prompt\n→ raw completion + token IDs\n→ extracted candidate source\n→ source SHA-256 identity'],
    verify: ['EXECUTION FEEDBACK', 'Correctness comes before speed.', 'The worker compares candidates with native PyTorch and quantized-input NumPy references, across independent values and registered stress cases. It checks input mutation and output contracts. Only correct candidates receive performance measurements.', 'source policy + timeout\n→ independent numerical checks\n→ steady-state GPU timing\n→ strict result record'],
    update: ['REVISED TRAINING PROTOCOL', 'Separate updates from improvement.', 'The revised protocol adds training-only supervised warmup and gives every invalid candidate zero training reward. GRPO compares rewards within each sampled group. A saved checkpoint and changed weights prove an update; a separate held-out test is needed to assess quality.', 'train-only SFT bootstrap\n→ correctness-gated GRPO\n→ optimizer + RNG checkpoint\n→ fresh held-out evaluation']
  };
  const tabs = [...document.querySelectorAll('[data-stage]')];
  function selectStage(button) {
    tabs.forEach((tab) => { const selected = tab === button; tab.setAttribute('aria-selected', String(selected)); tab.tabIndex = selected ? 0 : -1; });
    const data = stages[button.dataset.stage];
    ['stage-label', 'stage-title', 'stage-copy', 'stage-code'].forEach((id, i) => { $(id).textContent = data[i]; });
    $('pipeline-panel').setAttribute('aria-labelledby', button.id);
  }
  tabs.forEach((button, index) => {
    button.addEventListener('click', () => selectStage(button));
    button.addEventListener('keydown', (event) => {
      let next = null;
      if (event.key === 'ArrowRight') next = (index + 1) % tabs.length;
      if (event.key === 'ArrowLeft') next = (index - 1 + tabs.length) % tabs.length;
      if (event.key === 'Home') next = 0;
      if (event.key === 'End') next = tabs.length - 1;
      if (next !== null) { event.preventDefault(); selectStage(tabs[next]); tabs[next].focus(); }
    });
  });
  function enableCopy(buttonId, contentId, statusId) {
    const button = $(buttonId); if (!button) return;
    button.addEventListener('click', async () => {
      try { await navigator.clipboard.writeText($(contentId).textContent); $(statusId).textContent = 'Copied to clipboard.'; }
      catch { $(statusId).textContent = 'Select the text to copy it; clipboard access is unavailable in this browser.'; }
    });
  }
  enableCopy('copy-command', 'reproduce-command', 'copy-status');
  enableCopy('copy-citation', 'citation-text', 'citation-status');
  function addBar(parent, value, max, compiled = false) {
    const row = text('div', '', 'bar-row');
    const track = text('div', '', 'bar-track'); const fill = text('div', '', `bar-fill${compiled ? ' compiled' : ''}`);
    fill.style.width = `${Math.max(0, Math.min(100, value / max * 100))}%`; track.append(fill);
    row.append(track, text('span', ratio(value), 'bar-value')); parent.append(row);
  }
  function renderBaseline(data) {
    $('baseline-count').textContent = `${data.passed} / ${data.total}`;
    const chart = $('baseline-chart'); chart.replaceChildren();
    const head = text('div', '', 'chart-head'); head.append(text('h4', 'Geometric mean speedup'));
    const key = text('div', '', 'chart-key'); key.append(text('span', 'vs eager'), text('span', 'vs compile')); head.append(key); chart.append(head);
    const max = Math.max(...Object.values(data.summary).map((item) => item.eager_geomean));
    Object.entries(data.summary).forEach(([dtype, measurement]) => {
      const group = text('div', '', 'bar-group'); group.append(text('p', dtype.toUpperCase()));
      group.setAttribute('aria-label', `${dtype.toUpperCase()}: ${ratio(measurement.eager_geomean)} vs eager across ${measurement.operators} operators; ${ratio(measurement.compile_geomean)} vs compile across ${measurement.compile_operators} operators.`);
      addBar(group, measurement.eager_geomean, max); addBar(group, measurement.compile_geomean, max, true); chart.append(group);
    });
    chart.append(text('p', '1× means equal runtime. Eager: 30 operators. Verified compile: 29 operators. Same hardware and timing mode throughout.', 'chart-foot'));
  }
  function renderHistorical(runs) {
    const body = $('historical-rows'); body.replaceChildren();
    runs.forEach((run) => {
      const row = document.createElement('tr'); const model = document.createElement('td');
      model.append(text('strong', run.mode), text('small', `${run.dtype.toUpperCase()} · ${run.optimizer_updates} verified optimizer update${run.optimizer_updates === 1 ? '' : 's'}`));
      row.append(model);
      ['base_pass1', 'base_best_of_8', 'trained_checkpoint'].forEach((method) => {
        const result = run.methods.find((item) => item.name === method); row.append(text('td', result ? `${result.passed} / ${result.total}` : 'Unavailable'));
      }); body.append(row);
    });
  }
  function renderFailures(runs) {
    const host = $('failure-bars'); host.replaceChildren();
    const run = runs.find((item) => item.model.includes('7B')); if (!run) return;
    const labels = {CHEAT: 'Source-policy rejection', COMPILE_ERROR: 'Compile error', RUNTIME_ERROR: 'Runtime error'};
    Object.entries(run.training_failures).sort((a, b) => b[1] - a[1]).forEach(([status, count]) => {
      const item = text('div', '', 'failure-item'); const heading = text('div', '', 'failure-label'); heading.append(text('span', labels[status] || status), text('span', `${count} / ${run.training_samples}`));
      const track = text('div', '', 'failure-track'); const fill = document.createElement('div'); fill.style.width = `${count / run.training_samples * 100}%`; track.append(fill); item.append(heading, track); host.append(item);
    });
    host.append(text('p', '“CHEAT” is the verifier’s label for a prohibited source construct. It does not imply a correct kernel or a speedup.'));
  }
  function renderRetry(retry) {
    const labels = {pending: 'Results pending.', running: 'Study in progress.', partial: 'Partial corrected-study evidence.', completed: 'Completed corrected studies'};
    $('retry-title').textContent = labels[retry.status] || labels.pending;
    if (retry.note) $('retry-copy').textContent = retry.note;
    $('retry-intro').textContent = 'Passes are reported per operator at the recorded sampling budget. Fresh and inspected development cohorts stay separate; the hybrid checkpoint includes supervised training.';
    if (!retry.published_results || !retry.runs.length) return;
    const host = $('retry-results'); host.hidden = false; host.replaceChildren();
    retry.runs.forEach((run) => {
      const card = text('article', '', 'study-card');
      const heading = text('header', '', 'study-card-heading');
      const topline = text('div', '', 'study-card-topline');
      topline.append(text('p', run.protocol_version ? `${run.protocol_version.toUpperCase()} · recorded study` : 'Recorded study', 'study-version'));
      topline.append(text('span', run.status === 'completed' ? 'Completed' : 'Partial', 'study-status'));
      heading.append(topline, text('h4', run.short_label || run.label));
      if (run.cohort_label) heading.append(text('p', run.cohort_label, 'study-cohort-label'));
      card.append(heading);
      const counts = text('dl', '', 'study-counts');
      [['SFT optimizer updates', run.sft_updates], ['GRPO optimizer updates', run.optimizer_updates]].forEach(([label, value]) => {
        const item = document.createElement('div'); item.append(text('dt', label), text('dd', Number.isInteger(value) ? value : 'Not recorded')); counts.append(item);
      });
      card.append(counts);
      Object.entries(run.cohorts || {}).forEach(([cohort, methods]) => {
        const section = text('section', '', 'study-cohort');
        const title = cohort === 'fresh' ? 'Fresh held-out operators' : 'Inspected development';
        section.append(text('h5', title));
        const workloads = run.cohort_ops?.[cohort];
        if (Array.isArray(workloads) && workloads.length) {
          const names = text('ul', '', 'cohort-workloads'); names.setAttribute('aria-label', title + ' workload names');
          workloads.forEach((name) => { const item = document.createElement('li'); item.append(text('code', name)); names.append(item); });
          section.append(names);
        }
        const scroll = text('div', '', 'study-method-scroll');
        const table = text('table', '', 'study-method-table');
        table.append(text('caption', `${run.short_label || run.label}: ${title}`, 'visually-hidden'));
        const head = document.createElement('thead'); const header = document.createElement('tr');
        ['Method', 'Passes / N', 'Budget'].forEach((label) => { const cell = text('th', label); cell.scope = 'col'; header.append(cell); });
        head.append(header); table.append(head);
        const body = document.createElement('tbody');
        methods.forEach((method) => {
          const row = document.createElement('tr');
          row.append(text('td', method.method), text('td', `${method.passed} / ${method.total}`, method.passed ? 'pass-count' : 'pass-count zero'), text('td', `k=${method.budget}`, 'budget-count'));
          body.append(row);
        });
        table.append(body); scroll.append(table); section.append(scroll);
        if (cohort === 'development') section.append(text('p', 'Pass@1 only. Best-of-eight was not budgeted for this inspected cohort.', 'study-method-note'));
        card.append(section);
      });
      const details = text('details', '', 'study-provenance'); details.append(text('summary', 'Study details and source'));
      const metadata = document.createElement('dl');
      [['Model / placement', run.label], ['Hardware', run.hardware], ['Training mode', run.training_mode], ['Study identity', run.study_key], ['Configuration', run.config_hash], ['Model revision', run.model_revision]].forEach(([key, value]) => {
        if (!value) return;
        const item = document.createElement('div'); item.append(text('dt', key), text('dd', value)); metadata.append(item);
      });
      if (run.source_commit) {
        const item = document.createElement('div'); const value = document.createElement('dd'); value.append(text('code', run.source_commit));
        if (/^[0-9a-f]{40}$/.test(run.source_commit)) {
          const source = text('a', 'View executed source'); source.href = `https://github.com/Eljo-M/kernelbench-rl/commit/${run.source_commit}`; value.append(source);
        }
        item.append(text('dt', 'Executed source'), value); metadata.append(item);
      }
      details.append(metadata);
      if (run.comparison_path) {
        const source = text('a', 'Inspect this study’s strict comparison'); source.href = `https://github.com/Eljo-M/kernelbench-rl/blob/main/${encodeURI(run.comparison_path)}`; details.append(source);
      }
      (run.notes || []).forEach((note) => details.append(text('p', note)));
      card.append(details); host.append(card);
    });
  }
  function explorer(operators) {
    let precision = 'fp32'; let selected = 'rmsnorm_2048';
    function select(name) {
      selected = name; const operator = operators.find((item) => item.name === selected); if (!operator) return;
      $('operator-detail').hidden = false; $('detail-title').textContent = operator.name; $('detail-description').textContent = operator.description;
      const metadata = $('detail-metadata'); metadata.replaceChildren();
      [['Canonical input shapes', operator.input_shapes.map((shape) => `[${shape.join(' × ')}]`).join(' · ')], ['Analytical work', `${operator.analytic_flops.toLocaleString()} FLOPs`], ['FP32 tolerance', `rtol ${operator.rtol} / atol ${operator.atol}`]].forEach(([key, value]) => { const item = document.createElement('div'); item.append(text('dt', key), text('dd', value)); metadata.append(item); });
      $('detail-source').textContent = operator.source; $('detail-hash').textContent = `Recorded source SHA-256: ${operator.source_sha256}`;
      document.querySelectorAll('[data-op]').forEach((button) => { const active = button.dataset.op === name; button.setAttribute('aria-pressed', String(active)); button.closest('tr').setAttribute('aria-selected', String(active)); });
    }
    function render() {
      const query = $('op-search').value.trim().toLowerCase(); const level = $('level-filter').value;
      const filtered = operators.filter((operator) => (level === 'all' || String(operator.level) === level) && `${operator.name} ${operator.description}`.toLowerCase().includes(query));
      const body = $('operator-rows'); body.replaceChildren(); $('result-count').textContent = `${filtered.length} of ${operators.length} operators · ${precision.toUpperCase()} · three seeds per operator`;
      filtered.forEach((operator) => {
        const metrics = operator.precision[precision]; const row = document.createElement('tr'); const name = document.createElement('td'); const button = text('button', operator.name, 'op-button'); button.type = 'button'; button.dataset.op = operator.name; button.setAttribute('aria-controls', 'operator-detail'); button.addEventListener('click', () => select(operator.name)); name.append(button); row.append(name, text('td', `Level ${operator.level}`, 'tier-label'), text('td', `PASS ${metrics.passed}/${metrics.total}`, 'pass-label'), text('td', ratio(metrics.eager_speedup), 'speedup'), text('td', ratio(metrics.compile_speedup), metrics.compile_speedup === null ? 'unavailable' : 'speedup')); body.append(row);
      });
      if (!filtered.length) { const row = document.createElement('tr'); const cell = text('td', 'No matching operators. Try another name or workload tier.'); cell.colSpan = 5; row.append(cell); body.append(row); $('operator-detail').hidden = true; }
      else select(filtered.some((operator) => operator.name === selected) ? selected : filtered[0].name);
    }
    $('op-search').addEventListener('input', render); $('level-filter').addEventListener('change', render);
    document.querySelectorAll('input[name=precision]').forEach((input) => input.addEventListener('change', () => { precision = input.value; render(); }));
    render();
  }
  function configure(config) {
    if (!config.draft) document.querySelectorAll('[data-draft-notice]').forEach((notice) => { notice.hidden = true; });
    if (config.paper_pdf_path && $('paper-download')) { $('paper-download').href = config.paper_pdf_path; $('paper-download').hidden = false; }
    if (config.public_base_url && $('citation-text')) $('citation-text').textContent = $('citation-text').textContent.replace('https://github.com/Eljo-M/kernelbench-rl', `${config.public_base_url.replace(/\/$/, '')}/paper.html`);
    const downloads = $('download-links');
    if (downloads && config.downloads?.length) { downloads.hidden = false; config.downloads.forEach((item) => { const row = document.createElement('li'); const link = text('a', item.label); link.href = item.url; row.append(link); downloads.append(row); }); }
  }
  async function load() {
    try {
      const response = await fetch('project-config.json'); if (!response.ok) throw new Error('configuration unavailable'); configure(await response.json());
      if (!$('operator-rows')) return;
      const records = await fetch('results.json'); if (!records.ok) throw new Error('evidence unavailable'); const data = await records.json();
      if (data.schema_version !== 'kernelbench_website_results_v1') throw new Error('unsupported evidence schema');
      renderBaseline(data.baseline); renderHistorical(data.historical); renderFailures(data.historical); renderRetry(data.retry); explorer(data.baseline.operators);
    } catch (error) {
      if ($('result-count')) $('result-count').textContent = 'Interactive evidence could not load. Open the raw results report linked below.';
      if ($('operator-rows')) { const cell = text('td', 'Evidence unavailable. The repository retains all raw records.'); cell.colSpan = 5; const row = document.createElement('tr'); row.append(cell); $('operator-rows').replaceChildren(row); }
      console.error('KernelBench-RL evidence load:', error.message);
    }
  }
  load();
})();
