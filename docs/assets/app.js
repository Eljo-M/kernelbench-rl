'use strict';
(() => {
  const status = document.getElementById('interaction-status');
  const announce = (message) => { if (status) status.textContent = message; };
  async function copyText(value) {
    try {
      if (navigator.clipboard && window.isSecureContext) {
        await navigator.clipboard.writeText(value);
        return;
      }
    } catch (_) { /* A local selection fallback also works without clipboard permission. */ }
    const field = document.createElement('textarea');
    field.value = value;
    field.setAttribute('readonly', '');
    field.style.position = 'fixed';
    field.style.opacity = '0';
    document.body.append(field);
    field.select();
    const copied = document.execCommand('copy');
    field.remove();
    if (!copied) throw new Error('Select the code and copy it manually.');
  }
  document.querySelectorAll('[data-copy]').forEach((button) => {
    button.addEventListener('click', async () => {
      const target = document.getElementById(button.dataset.copy);
      if (!target) return;
      const original = button.textContent;
      try {
        await copyText(target.textContent);
        button.textContent = 'Copied';
        announce('Copied to clipboard.');
      } catch (error) {
        button.textContent = 'Select to copy';
        announce(error.message);
      }
      window.setTimeout(() => { button.textContent = original; }, 2200);
    });
  });
  const integrity = document.getElementById('verify-sources');
  const integrityStatus = document.getElementById('integrity-status');
  if (integrity) integrity.addEventListener('click', async () => {
    integrity.disabled = true;
    integrityStatus.textContent = 'Checking the two source-file hashes…';
    try {
      if (!window.crypto || !window.crypto.subtle) throw new Error('Use HTTPS or localhost to check hashes in this browser.');
      const response = await fetch('case-studies.json', {cache: 'no-store'});
      if (!response.ok) throw new Error('Could not load the case-study manifest.');
      const manifest = await response.json();
      if (manifest.cases.length !== 2) throw new Error('The selected-case manifest must contain exactly two examples.');
      for (const item of manifest.cases) {
        const source = await fetch(item.provenance.code_path, {cache: 'no-store'});
        if (!source.ok) throw new Error('A source snapshot is unavailable.');
        const bytes = await source.arrayBuffer();
        const hash = await window.crypto.subtle.digest('SHA-256', bytes);
        const actual = Array.from(new Uint8Array(hash), (byte) => byte.toString(16).padStart(2, '0')).join('');
        if (actual !== item.provenance.candidate_sha256) throw new Error('A source hash differs from its saved record.');
      }
      integrityStatus.textContent = 'Both source snapshots match their recorded SHA-256. This checks file integrity; it does not rerun the GPU kernels.';
      announce('Both source hashes match.');
    } catch (error) {
      integrityStatus.textContent = error.message;
      announce(error.message);
    } finally { integrity.disabled = false; }
  });
})();
