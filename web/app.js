import { initViewer, loadModel, onPick, getSelection, clearSelection,
         setSelection, listFaces, lookAtFace, snapshot } from './viewer.js';

const $ = (id) => document.getElementById(id);
const banner = $('banner');
const busy = $('busy');

initViewer($('view'));

let modelLoaded = false;
let canUndo = false;

function setBusy(on, msg = 'working...') {
  $('busy-msg').textContent = msg;
  busy.classList.toggle('on', on);
}

function note(kind, text) {
  banner.className = kind;         // 'ok' | 'err' | ''
  banner.textContent = text;
}

function refreshButtons() {
  const sel = getSelection();
  $('btn-apply').disabled = !modelLoaded || sel.length === 0;
  $('btn-undo').disabled = !canUndo;
  $('btn-step').disabled = !modelLoaded;
  $('btn-stl').disabled = !modelLoaded;
}

onPick((sel, last) => {
  if (sel.length === 0) {
    $('sel-info').textContent = 'no faces selected';
  } else {
    let txt = `<b>${sel.length}</b> face${sel.length > 1 ? 's' : ''}: ${sel.join(', ')}`;
    if (last) {
      txt += `<br>last: #${last.faceId} (${last.kind}, ${last.area.toFixed(0)} mm2)`;
    }
    $('sel-info').innerHTML = txt;
  }
  refreshButtons();
});

function applyModel(data) {
  loadModel(data);
  modelLoaded = true;
  canUndo = !!data.can_undo;
  $('model-info').textContent =
    `${data.filename} - ${data.nfaces} faces, ${(data.volume / 1000).toFixed(1)} cm3`;
  refreshButtons();
}

$('file-input').addEventListener('change', async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  setBusy(true, `loading ${file.name}...`);
  note('', '');
  try {
    const fd = new FormData();
    fd.append('file', file);
    const r = await fetch('/api/load', { method: 'POST', body: fd });
    if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
    applyModel(await r.json());
    note('ok', 'model loaded - click faces to select, then Apply ribs');
  } catch (err) {
    note('err', 'load failed: ' + err.message);
  } finally {
    setBusy(false);
    refreshButtons();
  }
});

function gatherParams() {
  const num = (id) => { const v = $(id).value; return v === '' ? null : Number(v); };
  return {
    pattern: $('pattern').value,
    spacing: num('spacing'),
    spacing_y: num('spacing-y'),
    thickness: num('thickness'),
    height: num('height'),
    orientation_deg: num('orientation'),
    draft_deg: num('draft'),
    margin: num('margin'),
    border: $('border').checked,
    density: num('density'),
    seed: num('seed'),
  };
}

$('btn-apply').addEventListener('click', async () => {
  const face_ids = getSelection();
  if (!face_ids.length) return;
  setBusy(true, `building ribs on ${face_ids.length} face(s)... this can take a minute`);
  note('', '');
  try {
    const r = await fetch('/api/ribs', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ face_ids, params: gatherParams(),
                             engine: $('engine').value }),
    });
    if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
    const data = await r.json();
    applyModel(data);
    const lines = [`engine: ${data.engine}`];
    for (const rep of data.reports) {
      let l = `face ${rep.face_id}: ${rep.lofted}/${rep.segments} ribs built`;
      if (rep.skipped) l += ` (${rep.skipped} skipped)`;
      lines.push(l);
      for (const w of rep.warnings) lines.push('! ' + w);
    }
    if (data.overlay_count > 0) {
      lines.push('note: fast engine active - exports will be faceted (STL + faceted STEP)');
    }
    note('ok', lines.join('\n'));
  } catch (err) {
    note('err', 'ribbing failed: ' + err.message);
  } finally {
    setBusy(false);
    refreshButtons();
  }
});

$('btn-undo').addEventListener('click', async () => {
  setBusy(true, 'undoing...');
  try {
    const r = await fetch('/api/undo', { method: 'POST' });
    if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
    applyModel(await r.json());
    note('ok', 'reverted');
  } catch (err) {
    note('err', 'undo failed: ' + err.message);
  } finally {
    setBusy(false);
    refreshButtons();
  }
});

// Automation/dev hook: load a STEP file by local path (server-side read).
window.__loadPath = async (path) => {
  setBusy(true, `loading ${path}...`);
  try {
    const r = await fetch('/api/load_path', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ path }),
    });
    if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
    applyModel(await r.json());
    note('ok', 'model loaded - click faces to select, then Apply ribs');
    return true;
  } catch (err) {
    note('err', 'load failed: ' + err.message);
    return false;
  } finally {
    setBusy(false);
    refreshButtons();
  }
};
window.__getSelection = () => getSelection();
window.__setSelection = (ids) => setSelection(ids);
window.__faces = () => listFaces();
window.__lookAt = (id) => lookAtFace(id);
window.__snap = (w) => snapshot(w);

$('btn-clear').addEventListener('click', () => clearSelection());
$('btn-step').addEventListener('click', () => { window.location = '/api/export/step'; });
$('btn-stl').addEventListener('click', () => { window.location = '/api/export/stl'; });

refreshButtons();
