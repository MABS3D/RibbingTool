import { onSurfacePick, setInteractionMode, setMappingPreview,
         setMappingControls, onMappingControlPick, onMappingControlAnchor,
         onMappingControlRotate } from './viewer.js';

const $ = (id) => document.getElementById(id);
const clone = (value) => JSON.parse(JSON.stringify(value));

/** Surface-anchored design controls. The generator consumes the same data
 * whether it came from individual handles or a future stroke editor. */
export function createMappingEditor({ getSelection, setSelection, getParams,
                                      setParams, getEngine, refreshButtons }) {
  let model = null, controls = [], activeId = null, placement = null;
  let preview = null, visible = false, running = false, busy = false;
  let revision = 0, validRevision = -1, timer = null, sequence = 0;
  let queued = false, history = [], restoring = false;
  let popoverOpen = false, anchor = null, rotationGesture = null;
  const inputs = [
    ['local-radius', 'radius'], ['local-angle', 'angle_deg'],
    ['local-spacing', 'spacing_scale'], ['local-height', 'height_scale'],
  ];

  function supported() {
    return ['surface', 'project'].includes($('mapping').value)
      && ['auto', 'graph'].includes(getEngine());
  }
  function localSupported() {
    return supported() && $('mapping').value === 'surface' && $('pattern').value !== 'stochastic';
  }
  function active() { return controls.find(c => c.id === activeId); }
  function enabledControls() {
    return controls.filter(c => c.enabled !== false).map(({ enabled, normal, ...c }) => clone(c));
  }
  function status(text, state = '') {
    $('mapping-status').textContent = text;
    $('mapping-status').dataset.state = state;
    $('control-status').textContent = text.split('\n')[0];
    $('control-status').dataset.state = state;
  }
  function positionPopover() {
    const popover = $('control-popover'), connector = $('control-connector');
    const shown = popoverOpen && !!active() && !placement && !busy
      && anchor?.id === activeId && anchor.visible;
    popover.hidden = !shown;
    connector.hidden = !shown;
    if (!shown) return;
    const view = $('view'), gap = 156, pad = 10;
    const width = popover.offsetWidth, height = popover.offsetHeight;
    // Keep the menu beside the rotation ring, flipping sides near the edge.
    let left = anchor.x + gap;
    if (left + width > view.clientWidth - pad) left = anchor.x - gap - width;
    left = Math.max(pad, Math.min(left, view.clientWidth - width - pad));
    const top = Math.max(pad, Math.min(anchor.y - 55, view.clientHeight - height - pad));
    popover.style.left = `${Math.round(left)}px`;
    popover.style.top = `${Math.round(top)}px`;
    const line = connector.querySelector('line');
    line.setAttribute('x1', anchor.x);
    line.setAttribute('y1', anchor.y);
    line.setAttribute('x2', Math.max(left, Math.min(anchor.x, left + width)));
    line.setAttribute('y2', Math.max(top + 16, Math.min(anchor.y, top + height - 16)));
  }
  function syncViewerControls() {
    setMappingControls(controls, activeId, {
      editing: popoverOpen && !placement && !busy && localSupported(),
      enabled: !busy,
    });
    positionPopover();
  }
  function openControl(id) {
    if (busy || rotationGesture || !controls.some(c => c.id === id)) return;
    activeId = id;
    popoverOpen = true;
    setPlacement(null);
    render();
    saveDraft();
  }
  function closeControl() {
    if (rotationGesture) return;
    popoverOpen = false;
    syncViewerControls();
  }
  function remember(includeLayout = false) {
    const state = { controls: clone(controls), activeId };
    if (includeLayout) Object.assign(state, { params: getParams(), face_ids: getSelection() });
    history.push(state);
    if (history.length > 50) history.shift();
  }
  function draftKey() { return model?.model_fingerprint ? `ribbing.mapping.v1.${model.model_fingerprint}` : null; }
  function saveDraft() {
    const key = draftKey();
    if (!key || restoring || rotationGesture) return;
    try {
      localStorage.setItem(key, JSON.stringify({ version: 1, controls, activeId,
        face_ids: getSelection(), params: getParams() }));
    } catch { /* Storage may be disabled; Save mapping remains available. */ }
  }
  function setPlacement(mode) {
    placement = mode;
    setInteractionMode(mode ? 'control' : 'select');
    $('btn-local-add').setAttribute('aria-pressed', String(mode === 'add'));
    $('btn-local-move').setAttribute('aria-pressed', String(mode === 'move'));
    $('btn-local-add').textContent = mode === 'add' ? 'Cancel placement' : 'Add control';
    $('mapping-toolbar').hidden = !mode;
    $('mapping-toolbar').textContent = mode === 'move'
      ? 'Click a selected face to move this control. Drag to orbit. Esc to cancel.'
      : 'Click a selected face to place a local control. Drag to orbit. Esc to cancel.';
    syncViewerControls();
  }
  function render() {
    const selection = new Set(getSelection());
    $('local-list').replaceChildren();
    controls.forEach((c, i) => {
      const button = document.createElement('button');
      button.type = 'button';
      button.textContent = `${i + 1} · face ${c.face_id}${c.enabled === false ? ' · off' : ''}`;
      button.setAttribute('aria-pressed', String(c.id === activeId));
      button.title = selection.has(c.face_id) ? 'Edit this control' : 'This control is outside the selection';
      button.addEventListener('click', () => openControl(c.id));
      $('local-list').append(button);
    });
    const c = active();
    $('local-editor').hidden = !c;
    if (c) {
      $('control-title').textContent = `Control ${controls.indexOf(c) + 1}`;
      for (const [id, key] of inputs) $(id).value = c[key];
      $('local-enabled').checked = c.enabled !== false;
      $('local-position').textContent = `Face ${c.face_id} · ${c.position.map(n => n.toFixed(2)).join(', ')} mm`;
    }
    syncViewerControls();
    updateButtons();
  }
  function updateButtons() {
    const editingLocked = busy || !!rotationGesture;
    const available = !!model && getSelection().length > 0 && !editingLocked;
    $('btn-preview').disabled = !available || !supported() || running;
    $('btn-preview').textContent = running ? 'Calculating…' : 'Preview mapping';
    $('btn-hide-preview').disabled = !preview && !visible;
    $('btn-local-add').disabled = !available || !localSupported() || controls.length >= 32;
    $('btn-local-undo').disabled = history.length === 0 || editingLocked;
    $('btn-local-move').disabled = !available || !localSupported() || !active();
    $('btn-local-delete').disabled = editingLocked;
    for (const [id] of inputs) $(id).disabled = editingLocked;
    $('local-enabled').disabled = editingLocked;
    $('btn-control-close').disabled = editingLocked;
    $('btn-mapping-save').disabled = !model || editingLocked;
    $('btn-mapping-open').disabled = !model || editingLocked;
    $('control-popover').dataset.rotating = String(!!rotationGesture);
  }
  function invalidate({ auto = true } = {}) {
    revision += 1;
    validRevision = -1;
    clearTimeout(timer);
    setMappingPreview(null);
    preview = null;
    if (enabledControls().length && !localSupported()) {
      status('Local controls require Surface, a regular pattern, and auto or surface graph engine.', 'error');
      setPlacement(null);
    } else if (!getSelection().length) {
      status('Select faces to preview the mapping.');
      setPlacement(null);
    } else if (!supported()) {
      status('Preview is available with Surface or Project and the surface graph engine.');
      setPlacement(null);
    } else {
      status('Mapping changed. Update the preview before building.', 'dirty');
      if (visible && auto && !busy && !restoring && !rotationGesture) timer = setTimeout(requestPreview, 400);
    }
    saveDraft();
    updateButtons();
    refreshButtons();
  }
  async function requestPreview() {
    clearTimeout(timer);
    if (!model || busy || rotationGesture || !getSelection().length || !supported()) return;
    if (running) { queued = true; return; }
    if (enabledControls().length && !localSupported()) return;
    const outside = enabledControls().filter(c => !getSelection().includes(c.face_id));
    if (outside.length) {
      status('A control is on an unselected face. Select that face, disable the control, or remove it.', 'error');
      return;
    }
    visible = true;
    running = true;
    queued = false;
    const requestRevision = revision;
    const token = model.model_token;
    const started = performance.now();
    status('Tracing the mapping on the selected surface…', 'working');
    const elapsedTimer = setInterval(() => {
      if (!visible || requestRevision !== revision) return;
      status(`Tracing the mapping… ${Math.floor((performance.now() - started) / 1000)}s\n`
        + 'You can keep adjusting controls; the latest settings will be traced next.', 'working');
    }, 1000);
    updateButtons();
    refreshButtons();
    try {
      const response = await fetch('/api/mapping/preview', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ face_ids: getSelection(), params: getParams(),
          engine: getEngine(), model_token: token }),
      });
      const data = await response.json();
      if (requestRevision !== revision || token !== model?.model_token || !visible) return;
      if (!response.ok) throw new Error(data.detail || response.statusText);
      preview = data;
      validRevision = revision;
      setMappingPreview(data, $('preview-height').checked);
      const stats = data.stats || {};
      const elapsed = Number(stats.mapping_seconds ?? 0).toFixed(1);
      status(`${stats.paths ?? data.paths.length} paths · ${elapsed}s · ${enabledControls().length} local controls\n`
        + (data.warnings?.length ? data.warnings.join('\n') : 'Preview ready. Apply builds this layout with thickness and fillets.'), 'ready');
    } catch (error) {
      if (requestRevision === revision) {
        validRevision = -1;
        preview = null;
        setMappingPreview(null);
        status(`Preview unavailable: ${error.message}`, 'error');
      }
    } finally {
      clearInterval(elapsedTimer);
      running = false;
      updateButtons();
      refreshButtons();
      if ((queued || requestRevision !== revision) && visible && !busy && !rotationGesture
          && validRevision !== revision)
        timer = setTimeout(requestPreview, 100);
    }
  }
  function resetForModel(data, { preserve = false, restore = true } = {}) {
    rotationGesture = null;
    popoverOpen = false;
    anchor = null;
    setPlacement(null);
    clearTimeout(timer);
    revision += 1;
    validRevision = -1;
    preview = null;
    visible = false;
    queued = false;
    // Retain identity only; a generated overlay can contain millions of
    // vertices and must be released once the viewer has uploaded its buffers.
    model = { model_token: data.model_token, model_fingerprint: data.model_fingerprint,
      filename: data.filename, faces: data.faces.map(face => ({ id: face.id })) };
    if (!preserve) {
      controls = []; activeId = null; history = [];
      const key = draftKey();
      try {
        const saved = restore && key ? JSON.parse(localStorage.getItem(key)) : null;
        if (saved?.version === 1 && Array.isArray(saved.controls)) {
          const parsed = validateFileControls(saved.controls);
          restoring = true;
          controls = parsed;
          activeId = parsed.some(c => c.id === saved.activeId) ? saved.activeId : parsed[0]?.id;
          setParams(saved.params || {});
          const validFaces = new Set(data.faces.map(f => f.id));
          setSelection((saved.face_ids || []).filter(id => validFaces.has(id)));
          status('Mapping draft restored for this STEP file. Preview to inspect it.', 'dirty');
        } else status('Select faces to preview the mapping.');
      } catch { status('Select faces to preview the mapping.'); }
      finally { restoring = false; }
    } else status('Layout applied. Preview again to continue editing.');
    render();
    refreshButtons();
  }
  function validateFileControls(items) {
    if (!Array.isArray(items) || items.length > 32) throw new Error('Expected up to 32 controls.');
    const ids = new Set();
    return items.map((item, i) => {
      const id = typeof item.id === 'string' && item.id.length < 100 ? item.id : `import-${i}`;
      if (ids.has(id)) throw new Error('Duplicate control IDs.');
      ids.add(id);
      if (!Number.isInteger(item.face_id) || !Array.isArray(item.position)
          || item.position.length !== 3 || !item.position.every(Number.isFinite))
        throw new Error('Invalid surface anchor.');
      const c = { id, face_id: item.face_id, position: [...item.position],
        enabled: item.enabled !== false, normal: item.normal };
      for (const [key, lo, hi] of [['radius', 0.1, Infinity], ['angle_deg', -90, 90],
        ['spacing_scale', .5, 2], ['height_scale', .25, 3]]) {
        if (!Number.isFinite(item[key]) || item[key] < lo || item[key] > hi)
          throw new Error(`Invalid ${key}.`);
        c[key] = item[key];
      }
      if (!Array.isArray(c.normal) || c.normal.length !== 3 || !c.normal.every(Number.isFinite)) c.normal = [0, 0, 1];
      return c;
    });
  }

  onSurfacePick(hit => {
    if (!placement || busy || !localSupported()) return;
    remember();
    if (placement === 'move' && active()) Object.assign(active(), hit);
    else {
      const c = { id: `control-${Date.now()}-${++sequence}`, ...hit,
        radius: Math.max(10, Number($('spacing').value) * 3),
        angle_deg: 0, spacing_scale: 1, height_scale: 1, enabled: true };
      controls.push(c);
      activeId = c.id;
    }
    visible = true;
    popoverOpen = true;
    setPlacement(null);
    render();
    invalidate();
  });
  onMappingControlPick(({ id }) => openControl(id));
  onMappingControlAnchor(value => { anchor = value; positionPopover(); });
  new ResizeObserver(positionPopover).observe($('control-popover'));
  onMappingControlRotate(event => {
    const c = controls.find(item => item.id === event.id);
    if ((!c || busy) && rotationGesture?.id === event.id
        && ['end', 'cancel'].includes(event.phase)) {
      if (c) c.angle_deg = rotationGesture.angle;
      rotationGesture = null;
      render(); invalidate({ auto: false });
      return;
    }
    if (!c || busy) return;
    if (event.phase === 'start') {
      if (rotationGesture) return;
      rotationGesture = { id: c.id, angle: c.angle_deg, controls: clone(controls), activeId,
        preview, valid: validRevision === revision,
        status: $('mapping-status').textContent, state: $('mapping-status').dataset.state };
      clearTimeout(timer);
      queued = false;
      revision += 1; // Ignore any old request that completes during this gesture.
      rotationGesture.revision = revision;
      validRevision = -1;
      status('Drag to rotate. Release to update the mapping; Esc to cancel.', 'dirty');
      updateButtons(); refreshButtons();
      return;
    }
    const gesture = rotationGesture;
    if (!gesture || gesture.id !== c.id) return;
    if (event.phase === 'change') {
      c.angle_deg = event.angle_deg;
      $('local-angle').value = c.angle_deg;
      status(`Rotation ${c.angle_deg}° · release to update the mapping.`, 'dirty');
      syncViewerControls();
      return;
    }
    if (!['end', 'cancel'].includes(event.phase)) return;
    rotationGesture = null;
    c.angle_deg = event.phase === 'cancel' ? gesture.angle : event.angle_deg;
    const changed = c.angle_deg !== gesture.angle;
    if (changed) {
      history.push({ controls: gesture.controls, activeId: gesture.activeId });
      if (history.length > 50) history.shift();
      visible = true;
      render(); invalidate();
    } else if (revision === gesture.revision) {
      preview = gesture.preview;
      validRevision = gesture.valid ? revision : -1;
      status(gesture.status, gesture.state);
      render(); saveDraft(); refreshButtons();
      if (!gesture.valid && visible && !running) timer = setTimeout(requestPreview, 400);
    } else {
      // Other parameters or the selection may have changed during the gesture.
      // Cancelling the angle cannot restore the validity of that older layout.
      render(); invalidate();
    }
  });
  for (const [id, key] of inputs) $(id).addEventListener('change', () => {
    const c = active();
    if (!c) return;
    if (!$(id).checkValidity() || !Number.isFinite($(id).valueAsNumber)) {
      $(id).reportValidity(); $(id).value = c[key]; return;
    }
    remember(); c[key] = $(id).valueAsNumber;
    render(); invalidate();
  });
  $('local-enabled').addEventListener('change', () => {
    if (!active()) return;
    remember(); active().enabled = $('local-enabled').checked;
    render(); invalidate();
  });
  $('btn-local-add').addEventListener('click', () => setPlacement(placement === 'add' ? null : 'add'));
  $('btn-local-move').addEventListener('click', () => setPlacement(placement === 'move' ? null : 'move'));
  $('btn-local-delete').addEventListener('click', () => {
    remember(); controls = controls.filter(c => c.id !== activeId);
    activeId = controls.at(-1)?.id ?? null;
    popoverOpen = false;
    setPlacement(null); render(); invalidate();
  });
  $('btn-local-undo').addEventListener('click', () => {
    const entry = history.pop();
    if (!entry) return;
    controls = entry.controls; activeId = entry.activeId;
    popoverOpen = !!active();
    if (entry.params) {
      restoring = true;
      setParams(entry.params); setSelection(entry.face_ids);
      restoring = false;
    }
    setPlacement(null); render(); invalidate();
  });
  $('btn-preview').addEventListener('click', requestPreview);
  $('btn-hide-preview').addEventListener('click', () => {
    visible = false; clearTimeout(timer); setMappingPreview(null);
    setPlacement(null); status('Preview hidden. Controls will still be used by Apply.'); updateButtons();
  });
  $('preview-height').addEventListener('change', () => {
    if (preview && visible) setMappingPreview(preview, $('preview-height').checked);
  });
  $('btn-control-close').addEventListener('click', closeControl);
  document.addEventListener('keydown', e => {
    if (e.key !== 'Escape' || e.defaultPrevented || rotationGesture) return;
    if (placement) setPlacement(null);
    else closeControl();
  });
  $('btn-mapping-save').addEventListener('click', () => {
    const data = { version: 1, model_fingerprint: model.model_fingerprint,
      filename: model.filename, face_ids: getSelection(), params: getParams(), controls };
    const url = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' }));
    const a = document.createElement('a'); a.href = url;
    a.download = `${model.filename.replace(/\.(step|stp)$/i, '')}.mapping.json`; a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  });
  $('btn-mapping-open').addEventListener('click', () => $('mapping-file').click());
  $('mapping-file').addEventListener('change', async e => {
    const file = e.target.files[0];
    e.target.value = '';
    if (!file) return;
    const token = model?.model_token;
    try {
      if (file.size > 1024 * 1024) throw new Error('Mapping file exceeds 1 MB.');
      const data = JSON.parse(await file.text());
      if (!model || token !== model.model_token)
        throw new Error('The model changed while opening this mapping. Open it again.');
      if (busy || rotationGesture)
        throw new Error('Finish the current operation before opening a mapping.');
      if (data.version !== 1) throw new Error('Unsupported mapping file version.');
      if (!data.params || typeof data.params !== 'object' || Array.isArray(data.params))
        throw new Error('Mapping parameters are missing or invalid.');
      if (!model.model_fingerprint || data.model_fingerprint !== model.model_fingerprint)
        throw new Error('This mapping belongs to a different STEP file. Load its original model first.');
      const parsed = validateFileControls(data.controls);
      const faces = new Set(model.faces.map(f => f.id));
      if (!Array.isArray(data.face_ids) || data.face_ids.some(id => !faces.has(id))
          || parsed.some(c => !faces.has(c.face_id))) throw new Error('Mapping contains unknown face IDs.');
      remember(true); controls = parsed; activeId = controls[0]?.id ?? null;
      popoverOpen = !!active();
      restoring = true;
      setParams(data.params || {}); setSelection(data.face_ids);
      restoring = false;
      setPlacement(null);
      visible = true; render(); invalidate();
    } catch (error) { restoring = false; status(`Cannot open mapping: ${error.message}`, 'error'); }
  });

  return {
    getControls: enabledControls,
    modelLoaded: resetForModel,
    modelCleared() {
      clearTimeout(timer);
      revision += 1;
      validRevision = -1;
      rotationGesture = null;
      model = null;
      controls = []; activeId = null; history = [];
      preview = null; visible = false; queued = false;
      popoverOpen = false; anchor = null;
      setPlacement(null);
      setMappingPreview(null);
      status('Load a STEP file to start.');
      render(); refreshButtons();
    },
    selectionChanged() { if (!restoring) { render(); invalidate(); } },
    paramsChanged() { invalidate(); },
    setBusy(value) {
      busy = value;
      if (value) { clearTimeout(timer); setPlacement(null); }
      else if (visible && validRevision !== revision && !running && supported())
        timer = setTimeout(requestPreview, 400);
      updateButtons();
      syncViewerControls();
    },
    isPreviewRunning: () => running || !!rotationGesture,
    canApply: () => !running && !rotationGesture && (!enabledControls().length || (localSupported() && validRevision === revision)),
    refresh: updateButtons,
  };
}
