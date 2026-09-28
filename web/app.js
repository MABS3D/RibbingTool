import { initViewer, loadModel, clearModel, onPick, getSelection, clearSelection,
         setSelection, listFaces, lookAtFace, snapshot } from './viewer.js';
import { createMappingEditor } from './mapping-editor.js';

const $ = (id) => document.getElementById(id);
const banner = $('banner');
const busy = $('busy');
let mappingEditor = null;
let operationBusy = false;
let updatingModel = false;
let modelToken = null;
let modelRevision = 0;

initViewer($('view'));

function updateGraphControls() {
  const active = ['surface', 'project'].includes($('mapping').value)
    && ['auto', 'graph'].includes($('engine').value);
  $('fillet-junction').disabled = !active;
  $('guide-smoothing').disabled = !active;
}
$('mapping').addEventListener('change', updateGraphControls);
$('engine').addEventListener('change', updateGraphControls);
updateGraphControls();

let modelLoaded = false;
let canUndo = false;

function setBusy(on, msg = 'working...') {
  operationBusy = on;
  $('busy-msg').textContent = msg;
  busy.classList.toggle('on', on);
  $('panel').inert = on;
  mappingEditor?.setBusy(on);
  refreshButtons();
}

function note(kind, text) {
  banner.className = kind;         // 'ok' | 'err' | ''
  banner.textContent = text;
}

// --- live apply progress (poll /api/apply_progress at 2 Hz) -------------
let progressTimer = null;

function renderProgress(p) {
  if (!p.active) return;               // keep the last frame; stopProgress clears
  $('progress').classList.add('on');
  const fill = $('progress-fill');
  const secs = Math.round(p.elapsed);
  if (p.total > 0) {
    fill.classList.remove('indet');
    fill.style.width = `${Math.round(100 * p.done / p.total)}%`;
    $('progress-label').textContent = `${p.stage} ${p.done}/${p.total} · ${secs}s`;
  } else {
    fill.classList.add('indet');       // no total: indeterminate shimmer
    fill.style.width = '40%';
    $('progress-label').textContent = `${p.stage} · ${secs}s`;
  }
}

function startProgress() {
  stopProgress();
  progressTimer = setInterval(async () => {
    try {
      const r = await fetch('/api/apply_progress');
      if (r.ok) renderProgress(await r.json());
    } catch (err) { /* poll failures are non-fatal; keep the last frame */ }
  }, 500);
}

function stopProgress() {
  if (progressTimer) { clearInterval(progressTimer); progressTimer = null; }
  $('progress').classList.remove('on');
  $('progress-fill').classList.remove('indet');
  $('progress-fill').style.width = '0';
  $('progress-label').textContent = '';
}

function refreshButtons() {
  const sel = getSelection();
  const locked = operationBusy || !!mappingEditor?.isPreviewRunning();
  $('btn-apply').disabled = locked || !modelLoaded || sel.length === 0 || (mappingEditor && !mappingEditor.canApply());
  $('btn-undo').disabled = locked || !canUndo;
  $('btn-step').disabled = locked || !modelLoaded;
  $('btn-stl').disabled = locked || !modelLoaded;
  $('file-input').disabled = locked;
  $('btn-unload').disabled = locked || !modelLoaded;
  $('btn-grow').disabled = locked || !sel.length;
  $('btn-clear').disabled = operationBusy;
}

function updateSelectionInfo(sel, last) {
  $('sel-info').title = sel.join(', ');
  if (sel.length === 0) {
    $('sel-info').textContent = 'no faces selected';
  } else {
    const shown = sel.slice(0, 12).join(', ') + (sel.length > 12 ? ', …' : '');
    let txt = `<b>${sel.length}</b> face${sel.length > 1 ? 's' : ''}: ${shown}`;
    if (last) {
      txt += `<br>last: #${last.faceId} (${last.kind}, ${last.area.toFixed(0)} mm2)`;
    }
    $('sel-info').innerHTML = txt;
  }
  refreshButtons();
}

onPick((sel, last) => {
  if (updatingModel) return;
  updateSelectionInfo(sel, last);
  mappingEditor?.selectionChanged();
});

async function applyModel(data, { preserve = false, selection = null } = {}) {
  const revision = ++modelRevision;
  updatingModel = true;
  let loaded;
  try {
    loaded = await loadModel(data, { preserveCamera: preserve });
  } finally {
    if (revision === modelRevision) updatingModel = false;
  }
  if (!loaded || revision !== modelRevision) return;
  updatingModel = true;
  modelLoaded = true;
  modelToken = data.model_token;
  canUndo = !!data.can_undo;
  $('model-info').textContent =
    `${data.filename} - ${data.nfaces} faces, ${(data.volume / 1000).toFixed(1)} cm3`;
  mappingEditor?.modelLoaded(data, { preserve });
  if (selection) {
    const valid = new Set(data.faces.map(f => f.id));
    setSelection(selection.filter(id => valid.has(id)));
  }
  updatingModel = false;
  updateSelectionInfo(getSelection(), null);
  mappingEditor?.refresh();
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
    await applyModel(await r.json());
    note('ok', 'model loaded - click faces to select, then Apply ribs');
  } catch (err) {
    note('err', 'load failed: ' + err.message);
  } finally {
    setBusy(false);
    refreshButtons();
  }
});

function clearLoadedModel() {
  modelRevision += 1;
  updatingModel = true;
  modelLoaded = false;
  modelToken = null;
  canUndo = false;
  mappingEditor?.modelCleared();
  clearModel();
  $('file-input').value = '';
  $('model-info').textContent = 'no model loaded';
  stopProgress();
  updatingModel = false;
  updateSelectionInfo([], null);
  refreshButtons();
}

$('btn-unload').addEventListener('click', async () => {
  if (!modelLoaded || operationBusy || mappingEditor?.isPreviewRunning()) return;
  setBusy(true, 'closing model...');
  try {
    const response = await fetch('/api/unload', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model_token: modelToken }),
    });
    if (!response.ok) throw new Error((await response.json()).detail || response.statusText);
    clearLoadedModel();
    note('', '');
  } catch (err) {
    note('err', 'could not close model: ' + err.message);
  } finally {
    setBusy(false);
  }
});

function gatherParams() {
  const num = (id) => { const v = $(id).value; return v === '' ? null : Number(v); };
  return {
    pattern: $('pattern').value,
    mapping: $('mapping').value,
    spacing: num('spacing'),
    spacing_y: num('spacing-y'),
    thickness: num('thickness'),
    height: num('height'),
    orientation_deg: num('orientation'),
    offset_x: num('offset-x'),
    offset_y: num('offset-y'),
    draft_deg: num('draft'),
    margin: num('margin'),
    border: $('border').checked,
    taper_len: num('taper'),
    fillet_root: num('fillet-root'),
    fillet_top: num('fillet-top'),
    fillet_junction: num('fillet-junction'),
    guide_smoothing: num('guide-smoothing'),
    density: num('density'),
    seed: num('seed'),
    mapping_controls: mappingEditor?.getControls() || [],
  };
}

function restoreParams(params) {
  const fields = { pattern: 'pattern', mapping: 'mapping', spacing: 'spacing',
    spacing_y: 'spacing-y', thickness: 'thickness', height: 'height',
    orientation_deg: 'orientation', offset_x: 'offset-x', offset_y: 'offset-y',
    draft_deg: 'draft', margin: 'margin', taper_len: 'taper', fillet_root: 'fillet-root',
    fillet_top: 'fillet-top', fillet_junction: 'fillet-junction',
    guide_smoothing: 'guide-smoothing', density: 'density', seed: 'seed' };
  for (const [key, id] of Object.entries(fields)) {
    if (!(key in params)) continue;
    const el = $(id), value = params[key];
    if (el.tagName === 'SELECT') {
      if ([...el.options].some(o => o.value === value)) el.value = value;
    } else if (value === null && key === 'spacing_y') el.value = '';
    else if (typeof value === 'number' && Number.isFinite(value)) el.value = value;
  }
  if (typeof params.border === 'boolean') $('border').checked = params.border;
  $('engine').value = 'auto';
  updateGraphControls();
}

$('btn-apply').addEventListener('click', async () => {
  const face_ids = getSelection();
  if (!face_ids.length) return;
  setBusy(true, `building ribs on ${face_ids.length} face(s)... this can take a minute`);
  startProgress();
  note('', '');
  try {
    const r = await fetch('/api/ribs', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ face_ids, params: gatherParams(),
                             engine: $('engine').value, model_token: modelToken }),
    });
    if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
    const data = await r.json();
    setBusy(true, 'preparing 3D view...');
    await applyModel(data, { preserve: data.engine === 'graph',
                       selection: data.engine === 'graph' ? face_ids : null });
    const lines = [`engine: ${data.engine}`];
    for (const rep of data.reports) {
      let l = data.engine === 'graph'
        ? `${rep.lofted} rib paths on ${rep.face_ids.length} selected face(s)`
        : `face ${rep.face_id}: ${rep.lofted}/${rep.segments} ribs built`;
      if (rep.skipped) l += ` (${rep.skipped} skipped)`;
      lines.push(l);
      for (const w of rep.warnings) lines.push('! ' + w);
    }
    if (data.overlay_count > 0) {
      lines.push('exports: STL and faceted STEP (triangulated surfaces)');
    }
    note('ok', lines.join('\n'));
  } catch (err) {
    note('err', 'ribbing failed: ' + err.message);
  } finally {
    stopProgress();
    setBusy(false);
    refreshButtons();
  }
});

$('btn-undo').addEventListener('click', async () => {
  setBusy(true, 'undoing...');
  try {
    const r = await fetch('/api/undo', { method: 'POST' });
    if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
    await applyModel(await r.json(), { preserve: true, selection: getSelection() });
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
    await applyModel(await r.json());
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
window.__lookAt = (id, zoom) => lookAtFace(id, zoom);
window.__snap = (w) => snapshot(w);

$('btn-grow').addEventListener('click', async () => {
  const face_ids = getSelection();
  if (!face_ids.length) return;
  setBusy(true, 'growing selection...');
  try {
    const r = await fetch('/api/grow', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ face_ids,
                             angle_deg: Number($('grow-angle').value) || 20 }),
    });
    if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
    const data = await r.json();
    setSelection(data.face_ids);
    note('ok', `selection grown to ${data.face_ids.length} tangent-connected face(s)`);
  } catch (err) {
    note('err', 'grow failed: ' + err.message);
  } finally {
    setBusy(false);
    refreshButtons();
  }
});

$('btn-clear').addEventListener('click', () => clearSelection());
async function exportModel(extension) {
  if (!modelLoaded || operationBusy || mappingEditor?.isPreviewRunning()) return;
  setBusy(true, `preparing ${extension.toUpperCase()} export...`);
  note('', '');
  try {
    const response = await fetch(`/api/export/${extension}`);
    if (!response.ok) {
      const error = await response.json().catch(() => null);
      throw new Error(error?.detail || response.statusText);
    }
    const blob = await response.blob();
    const disposition = response.headers.get('Content-Disposition') || '';
    const encodedName = disposition.match(/filename\*=UTF-8''([^;]+)/i);
    const quotedName = disposition.match(/filename="([^"]+)"/i);
    const filename = encodedName ? decodeURIComponent(encodedName[1])
      : quotedName?.[1] || `ribbingtool_export.${extension}`;
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.append(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 60000);
    note('ok', `${extension.toUpperCase()} export ready`);
  } catch (err) {
    note('err', 'export failed: ' + err.message);
  } finally {
    setBusy(false);
  }
}
$('btn-step').addEventListener('click', () => exportModel('step'));
$('btn-stl').addEventListener('click', () => exportModel('stl'));

mappingEditor = createMappingEditor({ getSelection, setSelection, getParams: gatherParams,
  setParams: restoreParams, getEngine: () => $('engine').value, refreshButtons });
for (const el of document.querySelectorAll('#panel input, #panel select')) {
  if (el.closest('#mapping-studio') || ['file-input', 'grow-angle'].includes(el.id)) continue;
  el.addEventListener('change', () => mappingEditor.paramsChanged());
}

refreshButtons();

// A browser refresh resumes the server's model and the draft for that exact STEP.
const resumeRevision = modelRevision;
fetch('/api/model').then(async response => {
  if (!response.ok) return;
  const data = await response.json();
  if (resumeRevision === modelRevision && !modelLoaded && !operationBusy) {
    setBusy(true, 'preparing 3D view...');
    try { await applyModel(data); }
    finally { setBusy(false); refreshButtons(); }
  }
}).catch(error => { note('err', 'Could not restore the 3D view: ' + error.message); });
