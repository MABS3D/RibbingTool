import * as THREE from './vendor/three.module.js';
import { OrbitControls } from './vendor/OrbitControls.js';
import { mergeVertices, toCreasedNormals } from './vendor/BufferGeometryUtils.js';

const BASE = new THREE.Color(0x8a8f98);
const HOVER = new THREE.Color(0xaab2c0);
const SELECTED = new THREE.Color(0xff8c2f);
const MAPPING_SELECTED = new THREE.Color(0x526475);
const ROTATION_RADIUS_PX = 96;

let renderer, scene, camera, controls, group, overlayGroup, mappingGroup, markerGroup, rotationGroup;
let viewerContainer;
let raycaster, pointer, hovered = null;
let pickCb = null, surfacePickCb = null;
let controlPickCb = null, controlAnchorCb = null, controlRotateCb = null;
let interactionMode = 'select', pickDirty = true, modelSize = 100;
let controlItems = [], activeControlId = null, controlEditing = false, controlInputEnabled = true;
let markerKey = '', rotationKey = '', rotationGizmo = null, rotationDrag = null;
let anchorDirty = true, lastAnchor = null, hoveredControl = null, hoveredRing = false;
const controlVisibilityCache = new Map();
const selected = new Set();
const RIB = new THREE.Color(0xb9a184);

export function initViewer(container) {
  viewerContainer = container;
  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  container.appendChild(renderer.domElement);

  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x16181d);

  camera = new THREE.PerspectiveCamera(45, 1, 0.1, 10000);
  camera.position.set(120, 90, 120);

  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.addEventListener('change', () => { pickDirty = true; anchorDirty = true; });

  scene.add(new THREE.HemisphereLight(0xf4f6ff, 0x33363d, 1.0));
  // key light rides the camera: thin rib walls face every which way, and a
  // world-fixed key leaves whole flanks in the dark at grazing view angles
  const key = new THREE.DirectionalLight(0xffffff, 1.6);
  key.position.set(0.6, 1.2, 1.0);
  camera.add(key);
  scene.add(camera);
  const fill = new THREE.DirectionalLight(0xb8c4ff, 0.5);
  fill.position.set(-1.5, -1, -1);
  scene.add(fill);

  group = new THREE.Group();
  scene.add(group);
  overlayGroup = new THREE.Group();
  scene.add(overlayGroup);
  mappingGroup = new THREE.Group();
  markerGroup = new THREE.Group();
  rotationGroup = new THREE.Group();
  scene.add(mappingGroup, markerGroup, rotationGroup);

  raycaster = new THREE.Raycaster();
  pointer = new THREE.Vector2(-2, -2);

  const el = renderer.domElement;
  let downAt = null, dragged = false;
  const updatePointer = (e) => {
    const r = el.getBoundingClientRect();
    pointer.set(((e.clientX - r.left) / r.width) * 2 - 1,
                -((e.clientY - r.top) / r.height) * 2 + 1);
    pickDirty = true;
  };
  // Capture gizmo gestures before OrbitControls sees the pointer event.
  el.addEventListener('pointerdown', (e) => {
    if (e.button !== 0 || !controlInputEnabled) return;
    updatePointer(e);
    if (beginRotation(e)) {
      downAt = null;
      e.preventDefault();
      e.stopImmediatePropagation();
    }
  }, true);
  el.addEventListener('pointermove', (e) => {
    if (!rotationDrag || e.pointerId !== rotationDrag.pointerId) return;
    updatePointer(e);
    moveRotation(e);
    e.preventDefault();
    e.stopImmediatePropagation();
  }, true);
  el.addEventListener('pointerup', (e) => {
    if (!rotationDrag || e.pointerId !== rotationDrag.pointerId) return;
    moveRotation(e);
    finishRotation('end');
    downAt = null;
    e.preventDefault();
    e.stopImmediatePropagation();
  }, true);
  el.addEventListener('pointercancel', (e) => {
    if (rotationDrag?.pointerId !== e.pointerId) return;
    finishRotation('cancel');
    downAt = null;
    e.stopImmediatePropagation();
  }, true);
  el.addEventListener('lostpointercapture', (e) => {
    if (rotationDrag?.pointerId === e.pointerId) finishRotation('cancel');
  });
  window.addEventListener('blur', () => { if (rotationDrag) finishRotation('cancel'); });
  document.addEventListener('keydown', (e) => {
    if (e.key !== 'Escape' || !rotationDrag) return;
    finishRotation('cancel');
    e.preventDefault();
    e.stopImmediatePropagation();
  }, true);
  el.addEventListener('pointermove', (e) => {
    updatePointer(e);
    if (downAt && Math.hypot(e.clientX - downAt[0], e.clientY - downAt[1]) > 4) {
      dragged = true;
    }
  });
  el.addEventListener('pointerleave', () => {
    pointer.set(-2, -2);
    pickDirty = true;
  });
  el.addEventListener('pointerdown', (e) => {
    if (e.button !== 0) return;
    updatePointer(e);
    downAt = [e.clientX, e.clientY];
    dragged = false;
  });
  el.addEventListener('pointercancel', () => { downAt = null; });
  el.addEventListener('pointerup', (e) => {
    if (!downAt) return;
    const moved = Math.hypot(e.clientX - downAt[0], e.clientY - downAt[1]);
    downAt = null;
    if (dragged || moved > 4 || e.button !== 0) return;   // it was an orbit drag
    updatePointer(e);
    const controlHit = pickMappingControl();
    if (controlHit) {
      if (controlPickCb) controlPickCb({ id: controlHit });
      return;
    }
    const intersection = pickIntersection();
    const hit = intersection?.object;
    if (hit) {
      const id = hit.userData.faceId;
      if (interactionMode === 'control') {
        if (selected.has(id) && surfacePickCb) {
          surfacePickCb({
            face_id: id,
            position: intersection.point.toArray(),
            normal: surfaceNormal(intersection).toArray(),
          });
        }
        return;
      }
      if (selected.has(id)) selected.delete(id); else selected.add(id);
      refreshColors();
      if (pickCb) pickCb(getSelection(), hit.userData);
    }
  });

  const resize = () => {
    const w = container.clientWidth, h = container.clientHeight;
    renderer.setSize(w, h);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    pickDirty = true;
    anchorDirty = true;
  };
  new ResizeObserver(resize).observe(container);
  resize();

  renderer.setAnimationLoop(() => {
    if (!rotationDrag) controls.update();
    if (anchorDirty) updateControlScreenState();
    if (pickDirty) {
      const hit = pickIntersection()?.object ?? null;
      const controlHit = pickMappingControl();
      const ringHit = !!hitRotationRing(pointerPixel());
      pickDirty = false;
      if (hit !== hovered || controlHit !== hoveredControl || ringHit !== hoveredRing) {
        hovered = hit; hoveredControl = controlHit; hoveredRing = ringHit; refreshColors();
      }
    }
    renderer.render(scene, camera);
  });
}

function pickIntersection() {
  raycaster.setFromCamera(pointer, camera);
  const hits = raycaster.intersectObjects(group.children, false);
  return hits.length ? hits[0] : null;
}

function surfaceNormal(hit) {
  const mesh = hit.object, face = hit.face;
  const positions = mesh.geometry.attributes.position;
  const normals = mesh.geometry.attributes.normal;
  const localPoint = mesh.worldToLocal(hit.point.clone());
  const bary = THREE.Triangle.getBarycoord(localPoint,
    new THREE.Vector3().fromBufferAttribute(positions, face.a),
    new THREE.Vector3().fromBufferAttribute(positions, face.b),
    new THREE.Vector3().fromBufferAttribute(positions, face.c), new THREE.Vector3());
  const normal = face.normal.clone();
  if (bary && normals) {
    normal.set(0, 0, 0)
      .addScaledVector(new THREE.Vector3().fromBufferAttribute(normals, face.a), bary.x)
      .addScaledVector(new THREE.Vector3().fromBufferAttribute(normals, face.b), bary.y)
      .addScaledVector(new THREE.Vector3().fromBufferAttribute(normals, face.c), bary.z);
    if (normal.lengthSq() < 1e-12) normal.copy(face.normal);
  }
  return normal.applyNormalMatrix(new THREE.Matrix3().getNormalMatrix(mesh.matrixWorld));
}

function refreshColors() {
  const inspectingMapping = mappingGroup.children.length > 0;
  for (const mesh of group.children) {
    const id = mesh.userData.faceId;
    const c = selected.has(id) ? (inspectingMapping ? MAPPING_SELECTED : SELECTED)
      : (mesh === hovered ? HOVER : BASE);
    mesh.material.color.copy(c);
    mesh.material.emissive.set(selected.has(id)
      ? (inspectingMapping ? 0x0a1924 : 0x552a00) : 0x000000);
  }
  renderer.domElement.style.cursor = rotationDrag ? 'grabbing'
    : hoveredRing && controlInputEnabled ? 'grab'
      : hoveredControl && controlInputEnabled ? 'pointer'
        : interactionMode === 'control'
          ? (hovered && selected.has(hovered.userData.faceId) ? 'crosshair' : 'default')
          : (hovered ? 'pointer' : 'default');
}

function disposeGroup(target) {
  // Materials can be shared by all paths in one preview. Dispose each GPU
  // resource once, including textures used by control-point labels.
  const geometries = new Set(), materials = new Set(), textures = new Set();
  target.traverse((object) => {
    if (object.geometry) geometries.add(object.geometry);
    for (const material of [].concat(object.material ?? [])) {
      materials.add(material);
      if (material.map) textures.add(material.map);
    }
  });
  geometries.forEach((geometry) => geometry.dispose());
  textures.forEach((texture) => texture.dispose());
  materials.forEach((material) => material.dispose());
  target.clear();
}

export function loadModel(data, { preserveCamera = false } = {}) {
  if (rotationDrag) finishRotation('cancel');
  for (const g of [group, overlayGroup, mappingGroup, markerGroup, rotationGroup]) disposeGroup(g);
  controlItems = [];
  activeControlId = null;
  markerKey = rotationKey = '';
  rotationGizmo = null;
  controlVisibilityCache.clear();
  hoveredControl = null;
  hoveredRing = false;
  anchorDirty = true;
  selected.clear();
  hovered = null;
  pickDirty = true;
  modelSize = Math.max(data.bbox[3] - data.bbox[0], data.bbox[4] - data.bbox[1],
    data.bbox[5] - data.bbox[2], 1);

  if (data.overlay) {
    let geo = new THREE.BufferGeometry();
    geo.setAttribute('position',
      new THREE.Float32BufferAttribute(new Float32Array(data.overlay.positions), 3));
    geo.setIndex(new THREE.Uint32BufferAttribute(new Uint32Array(data.overlay.indices), 1));
    try {
      // weld duplicated facet vertices, then smooth-shade across gentle
      // facets while keeping true edges (rib walls vs tops) crisp
      const welded = mergeVertices(geo, 1e-4);
      if (welded !== geo) geo.dispose();
      geo = welded;
      const creased = toCreasedNormals(geo, THREE.MathUtils.degToRad(38));
      if (creased !== geo) geo.dispose();
      geo = creased;
    } catch (e) {
      geo.computeVertexNormals();
    }
    const mesh = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({
      color: RIB, metalness: 0.1, roughness: 0.62, side: THREE.DoubleSide,
    }));
    overlayGroup.add(mesh);   // not pickable: raycast only targets `group`
  }

  for (const f of data.faces) {
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position',
      new THREE.Float32BufferAttribute(new Float32Array(f.positions), 3));
    geo.setIndex(new THREE.Uint32BufferAttribute(new Uint32Array(f.indices), 1));
    geo.computeVertexNormals();
    const mat = new THREE.MeshStandardMaterial({
      color: BASE, metalness: 0.15, roughness: 0.72, side: THREE.DoubleSide,
    });
    const mesh = new THREE.Mesh(geo, mat);
    mesh.userData = { faceId: f.id, kind: f.kind, planar: f.planar, area: f.area };
    group.add(mesh);
  }
  if (!preserveCamera) fitCamera(data.bbox);
  refreshColors();
  if (pickCb) pickCb(getSelection(), null);
}

export function clearModel() {
  loadModel({ faces: [], overlay: null, bbox: [-50, -50, -50, 50, 50, 50] });
}

function fitCamera(b) {
  const center = new THREE.Vector3((b[0] + b[3]) / 2, (b[1] + b[4]) / 2, (b[2] + b[5]) / 2);
  const size = Math.max(b[3] - b[0], b[4] - b[1], b[5] - b[2], 1);
  controls.target.copy(center);
  const d = size * 1.6;
  camera.position.copy(center).add(new THREE.Vector3(d, d * 0.75, d));
  camera.near = size / 100;
  camera.far = size * 20;
  camera.updateProjectionMatrix();
  controls.update();
}

export function onPick(cb) { pickCb = cb; }
export function onSurfacePick(cb) { surfacePickCb = cb; }
export function onMappingControlPick(cb) { controlPickCb = cb; }
export function onMappingControlAnchor(cb) { controlAnchorCb = cb; lastAnchor = null; anchorDirty = true; }
export function onMappingControlRotate(cb) { controlRotateCb = cb; }
export function setInteractionMode(mode) {
  if (mode !== 'select' && mode !== 'control') throw new Error(`Unknown interaction mode: ${mode}`);
  interactionMode = mode;
  refreshColors();
}

function updateMappingDepthOffset() {
  const visible = mappingGroup.children.length > 0 || markerGroup.children.length > 0;
  for (const mesh of group.children) {
    // A small depth offset makes coincident surface lines legible. The lines
    // still use the depth buffer, so rear faces remain hidden by the body.
    mesh.material.polygonOffset = visible;
    mesh.material.polygonOffsetFactor = visible ? 1 : 0;
    mesh.material.polygonOffsetUnits = visible ? 1 : 0;
  }
  refreshColors();
}

function validCoordinates(values) {
  return values && values.length >= 6 && values.length % 3 === 0
    && Array.prototype.every.call(values, Number.isFinite);
}

function mappingLine(points, material, segments = false) {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(points, 3));
  const line = segments ? new THREE.LineSegments(geometry, material) : new THREE.Line(geometry, material);
  line.renderOrder = 2;
  mappingGroup.add(line);
}

export function setMappingPreview(data, showHeight = true) {
  disposeGroup(mappingGroup);
  if (data?.paths?.length) {
    const baseMaterial = new THREE.LineBasicMaterial({
      color: 0x63e5ee, transparent: true, opacity: 0.95, depthTest: true, depthWrite: false,
    });
    const topMaterial = new THREE.LineBasicMaterial({
      color: 0xf0c27a, transparent: true, opacity: 0.66, depthTest: true, depthWrite: false,
    });
    const sideMaterial = new THREE.LineBasicMaterial({
      color: 0xf0c27a, transparent: true, opacity: 0.32, depthTest: true, depthWrite: false,
    });
    let hasBase = false, hasTop = false;
    for (const path of data.paths) {
      if (!validCoordinates(path.points)) continue;
      mappingLine(path.points, baseMaterial);
      hasBase = true;
      if (!showHeight || !validCoordinates(path.top) || path.top.length !== path.points.length) continue;
      mappingLine(path.top, topMaterial);
      hasTop = true;
      const count = path.points.length / 3, sides = [];
      // Sparse verticals communicate height without obscuring the mapping.
      const stride = Math.max(1, Math.ceil((count - 1) / 5));
      for (let i = 0; i < count; i += stride) {
        sides.push(...path.points.slice(i * 3, i * 3 + 3), ...path.top.slice(i * 3, i * 3 + 3));
      }
      if ((count - 1) % stride !== 0) {
        sides.push(...path.points.slice(-3), ...path.top.slice(-3));
      }
      mappingLine(sides, sideMaterial, true);
    }
    // Unused materials are not attached to the group and need explicit cleanup.
    if (!hasBase) baseMaterial.dispose();
    if (!hasTop) { topMaterial.dispose(); sideMaterial.dispose(); }
  }
  updateMappingDepthOffset();
}

function controlLabel(number, active, size) {
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = 96;
  const context = canvas.getContext('2d');
  context.beginPath();
  context.arc(48, 48, 38, 0, Math.PI * 2);
  context.fillStyle = active ? '#ffce85' : '#80edf2';
  context.fill();
  context.lineWidth = 5;
  context.strokeStyle = '#17212b';
  context.stroke();
  context.font = 'bold 44px system-ui, sans-serif';
  context.textAlign = 'center';
  context.textBaseline = 'middle';
  context.fillStyle = '#17212b';
  context.fillText(String(number), 48, 50);
  const texture = new THREE.CanvasTexture(canvas);
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({
    map: texture, transparent: true, depthTest: true, depthWrite: false,
  }));
  sprite.scale.set(size, size, 1);
  sprite.renderOrder = 3;
  return sprite;
}

function controlNormal(item) {
  const normal = new THREE.Vector3().fromArray(item.normal ?? [0, 0, 1]);
  if (![normal.x, normal.y, normal.z].every(Number.isFinite) || normal.lengthSq() < 1e-12) {
    normal.set(0, 0, 1);
  }
  return normal.normalize();
}

function activeControl() { return controlItems.find(item => item.id === activeControlId); }

function screenPoint(point) {
  const clip = point.clone().project(camera);
  const rect = renderer.domElement.getBoundingClientRect();
  const parent = viewerContainer.getBoundingClientRect();
  return { x: (clip.x + 1) * rect.width / 2 + rect.left - parent.left,
    y: (1 - clip.y) * rect.height / 2 + rect.top - parent.top,
    inView: clip.z >= -1 && clip.z <= 1 && Math.abs(clip.x) <= 1 && Math.abs(clip.y) <= 1 };
}

function pointerPixel() {
  const rect = renderer.domElement.getBoundingClientRect();
  const parent = viewerContainer.getBoundingClientRect();
  return { x: (pointer.x + 1) * rect.width / 2 + rect.left - parent.left,
    y: (1 - pointer.y) * rect.height / 2 + rect.top - parent.top };
}

function eventPixel(event) {
  const parent = viewerContainer.getBoundingClientRect();
  return { x: event.clientX - parent.left, y: event.clientY - parent.top };
}

function unitsPerPixel(point) {
  const depth = -point.clone().applyMatrix4(camera.matrixWorldInverse).z;
  return Math.max(camera.near, depth) * 2 * Math.tan(THREE.MathUtils.degToRad(camera.fov / 2))
    / Math.max(renderer.domElement.clientHeight, 1);
}

function pointVisible(point, tolerance = modelSize * 1e-5) {
  const projection = screenPoint(point);
  if (!projection.inView) return false;
  const direction = point.clone().sub(camera.position);
  const distance = direction.length();
  if (distance < camera.near) return false;
  const sight = new THREE.Raycaster(camera.position, direction.normalize(), 0, Math.max(0, distance - tolerance));
  // Both the CAD body and generated ribs occlude control anchors.
  return sight.intersectObjects([...group.children, ...overlayGroup.children], false).length === 0;
}

function controlVisibilityKey(item) {
  return JSON.stringify([item.position, item.normal, camera.matrixWorld.elements, camera.projectionMatrix.elements]);
}

function controlVisible(item) {
  // Rotation changes the guide, not the anchor's occlusion. In particular,
  // avoid tracing a many-million-triangle rib mesh on every drag update.
  const key = controlVisibilityKey(item);
  const cached = controlVisibilityCache.get(item.id);
  if (cached?.key === key) return cached.visible;
  const point = new THREE.Vector3().fromArray(item.position)
    .addScaledVector(controlNormal(item), modelSize * 1e-5);
  let visible = pointVisible(point);
  // A label can project above an existing rib even though its surface anchor
  // lies under that rib. Keep its editor available when the actual handle is
  // visible; each candidate is still tested against the body and rib meshes.
  if (!visible) {
    for (const marker of markerGroup.children) {
      if (marker.userData.controlId === item.id && pointVisible(marker.getWorldPosition(new THREE.Vector3()))) {
        visible = true;
        break;
      }
    }
  }
  controlVisibilityCache.set(item.id, { key, visible });
  return visible;
}

function pickMappingControl() {
  if (!controlInputEnabled || !markerGroup.children.length || rotationDrag) return null;
  scene.updateMatrixWorld(true);
  raycaster.setFromCamera(pointer, camera);
  const hits = raycaster.intersectObjects(markerGroup.children, false);
  if (!hits.length) return null;
  const body = raycaster.intersectObjects([...group.children, ...overlayGroup.children], false)[0];
  for (const hit of hits) {
    if (body && body.distance + modelSize * 1e-5 < hit.distance) continue;
    const item = controlItems.find(control => control.id === hit.object.userData.controlId);
    if (!item) continue;
    // This intersection already passed the body/rib occlusion test. It may be
    // a visible corner of a label whose center and anchor are both obscured.
    controlVisibilityCache.set(item.id, { key: controlVisibilityKey(item), visible: true });
    return item.id;
  }
  return null;
}

function ringText(text) {
  const canvas = document.createElement('canvas');
  canvas.width = 128; canvas.height = 64;
  const context = canvas.getContext('2d');
  context.font = 'bold 32px system-ui, sans-serif';
  context.textAlign = 'center'; context.textBaseline = 'middle';
  context.lineWidth = 7; context.strokeStyle = '#17212b';
  context.strokeText(text, 64, 32);
  context.fillStyle = '#e5eff3'; context.fillText(text, 64, 32);
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(canvas),
    transparent: true, depthTest: false, depthWrite: false }));
  sprite.scale.set(.48, .24, 1); sprite.renderOrder = 22;
  return sprite;
}

function buildRotationGizmo(item) {
  disposeGroup(rotationGroup);
  rotationGizmo = null;
  if (!item || !controlEditing) return;
  const normal = controlNormal(item);
  const u = new THREE.Vector3(Math.abs(normal.x) > .9 ? 0 : 1, Math.abs(normal.x) > .9 ? 1 : 0, 0);
  u.addScaledVector(normal, -u.dot(normal)).normalize();
  const v = new THREE.Vector3().crossVectors(normal, u).normalize();
  const root = new THREE.Group();
  root.quaternion.setFromRotationMatrix(new THREE.Matrix4().makeBasis(u, v, normal));
  rotationGroup.add(root);
  function line(points, color, opacity = 1, segments = false) {
    const geometry = new THREE.BufferGeometry().setFromPoints(points.map(p => new THREE.Vector3(...p)));
    const material = new THREE.LineBasicMaterial({ color, transparent: true, opacity, depthTest: false, depthWrite: false });
    const object = segments ? new THREE.LineSegments(geometry, material) : new THREE.Line(geometry, material);
    object.renderOrder = 20; root.add(object); return object;
  }
  const circle = [], range = [], ticks = [];
  for (let degree = 0; degree <= 360; degree += 2) {
    const rad = THREE.MathUtils.degToRad(degree);
    circle.push([Math.cos(rad), Math.sin(rad), 0]);
  }
  for (let degree = -90; degree <= 90; degree += 2) {
    const rad = THREE.MathUtils.degToRad(degree);
    range.push([Math.cos(rad), Math.sin(rad), .001]);
  }
  for (let degree = -180; degree < 180; degree += 5) {
    const rad = THREE.MathUtils.degToRad(degree);
    const end = degree % 30 === 0 ? 1.12 : degree % 15 === 0 ? 1.09 : 1.045;
    ticks.push([Math.cos(rad), Math.sin(rad), 0], [Math.cos(rad) * end, Math.sin(rad) * end, 0]);
  }
  line(circle, 0x94adb8, .5); line(range, 0xe3f5fc, .95); line(ticks, 0xd0e2ea, .88, true);
  line([[.18, 0, 0], [.94, 0, 0]], 0xa2b6c0, .55);
  for (const degree of [-90, -60, -30, 0, 30, 60, 90]) {
    const rad = THREE.MathUtils.degToRad(degree);
    const text = ringText(`${degree > 0 ? '+' : ''}${degree}°`);
    text.position.set(Math.cos(rad) * 1.32, Math.sin(rad) * 1.32, 0);
    root.add(text);
  }
  const indicator = new THREE.Group();
  root.add(indicator);
  const spoke = line([[.15, 0, .002], [1, 0, .002]], 0xffce85, .95);
  root.remove(spoke); indicator.add(spoke);
  const handle = new THREE.Mesh(new THREE.SphereGeometry(.066, 20, 12),
    new THREE.MeshBasicMaterial({ color: 0xffce85, depthTest: false, depthWrite: false }));
  handle.position.set(1, 0, .008); handle.renderOrder = 23; indicator.add(handle);
  const arrowPoints = [[.99, -.11, .005], [1.04, -.04, .005], [1.04, -.04, .005], [1.09, -.11, .005],
    [.99, .11, .005], [1.04, .04, .005], [1.04, .04, .005], [1.09, .11, .005]];
  const arrows = line(arrowPoints, 0xffce85, .95, true);
  root.remove(arrows); indicator.add(arrows);
  rotationGizmo = { root, normal, u, v, indicator, handle, radius: 1,
    center: new THREE.Vector3().fromArray(item.position), screen: null };
}

function updateControlScreenState() {
  anchorDirty = false;
  camera.updateMatrixWorld(true);
  scene.updateMatrixWorld(true);
  const item = activeControl();
  let anchor = { id: activeControlId, x: 0, y: 0, visible: false };
  if (item) {
    const position = new THREE.Vector3().fromArray(item.position);
    const normal = controlNormal(item);
    const projected = screenPoint(position);
    anchor = { id: item.id, x: projected.x, y: projected.y,
      visible: controlVisible(item) };
    if (rotationGizmo) {
      const g = rotationGizmo;
      g.radius = unitsPerPixel(position) * ROTATION_RADIUS_PX;
      g.center.copy(position).addScaledVector(normal, unitsPerPixel(position) * .8);
      g.root.position.copy(g.center);
      g.root.scale.setScalar(g.radius);
      g.root.visible = anchor.visible;
      g.indicator.rotation.z = THREE.MathUtils.degToRad(item.angle_deg || 0);
      g.handle.material.color.set(rotationDrag ? 0xffe9af : controlInputEnabled ? 0xffce85 : 0x839099);
      const samples = [];
      for (let degree = -180; degree <= 180; degree += 5) {
        const p = ringWorldPoint(degree);
        samples.push({ ...screenPoint(p), angle_deg: degree });
      }
      const handle = screenPoint(ringWorldPoint(item.angle_deg || 0));
      const center = screenPoint(g.center);
      const sight = camera.position.clone().sub(g.center).normalize();
      g.screen = { center: { x: center.x, y: center.y }, handle: { x: handle.x, y: handle.y }, samples,
        visible: anchor.visible, edgeOn: Math.abs(sight.dot(g.normal)) < .18 };
    }
  } else if (rotationGizmo) rotationGizmo.root.visible = false;
  const changed = !lastAnchor || anchor.id !== lastAnchor.id || anchor.visible !== lastAnchor.visible
    || Math.abs(anchor.x - lastAnchor.x) > .1 || Math.abs(anchor.y - lastAnchor.y) > .1;
  lastAnchor = anchor;
  if (changed && controlAnchorCb) controlAnchorCb({ ...anchor });
}

function ringWorldPoint(degree) {
  const g = rotationGizmo, rad = THREE.MathUtils.degToRad(degree);
  return g.center.clone().addScaledVector(g.u, Math.cos(rad) * g.radius)
    .addScaledVector(g.v, Math.sin(rad) * g.radius);
}

function distanceToSegment(point, a, b) {
  const dx = b.x - a.x, dy = b.y - a.y;
  const t = Math.max(0, Math.min(1, ((point.x - a.x) * dx + (point.y - a.y) * dy) / (dx * dx + dy * dy || 1)));
  return Math.hypot(point.x - a.x - t * dx, point.y - a.y - t * dy);
}

function hitRotationRing(point) {
  const screen = rotationGizmo?.screen;
  if (!controlEditing || !controlInputEnabled || !screen?.visible) return false;
  if (Math.hypot(point.x - screen.handle.x, point.y - screen.handle.y) <= 12) return true;
  for (let i = 1; i < screen.samples.length; i++) {
    if (distanceToSegment(point, screen.samples[i - 1], screen.samples[i]) <= 7) return true;
  }
  return false;
}

function planePointerAngle(event, drag) {
  const rect = renderer.domElement.getBoundingClientRect();
  const xy = new THREE.Vector2((event.clientX - rect.left) / rect.width * 2 - 1,
    1 - (event.clientY - rect.top) / rect.height * 2);
  const ray = new THREE.Raycaster(); ray.setFromCamera(xy, camera);
  const hit = ray.ray.intersectPlane(new THREE.Plane().setFromNormalAndCoplanarPoint(drag.normal, drag.center), new THREE.Vector3());
  if (!hit) return null;
  const offset = hit.sub(drag.center);
  if (offset.lengthSq() < drag.radius * drag.radius * .0025) return null;
  return THREE.MathUtils.radToDeg(Math.atan2(offset.dot(drag.v), offset.dot(drag.u)));
}

function beginRotation(event) {
  if (anchorDirty) updateControlScreenState();
  if (!hitRotationRing(eventPixel(event)) || rotationDrag) return false;
  // A nearby control stays selectable even when the editing ring crosses it.
  if (pickMappingControl()) return false;
  const item = activeControl(), g = rotationGizmo;
  if (!item || !g) return false;
  const startAngle = Math.max(-90, Math.min(90, Number(item.angle_deg) || 0));
  const before = screenPoint(ringWorldPoint(startAngle - 2));
  const after = screenPoint(ringWorldPoint(startAngle + 2));
  let axis = new THREE.Vector2(after.x - before.x, after.y - before.y);
  if (axis.length() < 1) {
    const a = screenPoint(ringWorldPoint(0)), b = screenPoint(ringWorldPoint(90));
    axis = new THREE.Vector2(a.x - g.screen.center.x, a.y - g.screen.center.y);
    const other = new THREE.Vector2(b.x - g.screen.center.x, b.y - g.screen.center.y);
    if (other.lengthSq() > axis.lengthSq()) axis.copy(other);
  }
  if (axis.lengthSq() < .01) axis.set(1, 0);
  axis.normalize();
  rotationDrag = { id: item.id, pointerId: event.pointerId, startAngle, angle: startAngle, accumulated: 0,
    start: eventPixel(event), axis, center: g.center.clone(), normal: g.normal.clone(),
    u: g.u.clone(), v: g.v.clone(), radius: g.radius, screenFallback: g.screen.edgeOn,
    previousPointerAngle: null, orbitEnabled: controls.enabled };
  rotationDrag.previousPointerAngle = planePointerAngle(event, rotationDrag);
  if (rotationDrag.previousPointerAngle === null) rotationDrag.screenFallback = true;
  controls.enabled = false;
  renderer.domElement.setPointerCapture(event.pointerId);
  if (controlRotateCb) controlRotateCb({ id: item.id, phase: 'start', angle_deg: startAngle });
  anchorDirty = true;
  refreshColors();
  return true;
}

function moveRotation(event) {
  const drag = rotationDrag;
  if (!drag) return;
  let raw;
  if (drag.screenFallback) {
    const point = eventPixel(event);
    raw = drag.startAngle + ((point.x - drag.start.x) * drag.axis.x + (point.y - drag.start.y) * drag.axis.y)
      * 180 / (Math.PI * ROTATION_RADIUS_PX);
  } else {
    const angle = planePointerAngle(event, drag);
    if (angle === null) return;
    const delta = ((angle - drag.previousPointerAngle + 540) % 360) - 180;
    drag.accumulated += delta;
    drag.previousPointerAngle = angle;
    raw = drag.startAngle + drag.accumulated;
  }
  const step = event.shiftKey ? 15 : 1;
  const next = Math.max(-90, Math.min(90, Math.round(raw / step) * step));
  if (next === drag.angle) return;
  drag.angle = next;
  const item = controlItems.find(control => control.id === drag.id);
  if (item) item.angle_deg = next;
  anchorDirty = true;
  updateControlScreenState();
  if (controlRotateCb) controlRotateCb({ id: drag.id, phase: 'change', angle_deg: next });
}

function finishRotation(phase) {
  const drag = rotationDrag;
  if (!drag) return;
  rotationDrag = null;
  const angle = phase === 'cancel' ? drag.startAngle : drag.angle;
  const item = controlItems.find(control => control.id === drag.id);
  if (item) item.angle_deg = angle;
  controls.enabled = drag.orbitEnabled;
  if (renderer.domElement.hasPointerCapture(drag.pointerId)) renderer.domElement.releasePointerCapture(drag.pointerId);
  anchorDirty = true; pickDirty = true;
  refreshColors();
  if (controlRotateCb) controlRotateCb({ id: drag.id, phase, angle_deg: angle });
}

export function setMappingControls(items, activeId = null, { editing = false, enabled = true } = {}) {
  if (rotationDrag && (activeId !== rotationDrag.id || !editing || !enabled
      || !(items ?? []).some(item => item.id === rotationDrag.id))) finishRotation('cancel');
  controlItems = (items ?? []).filter(item => item.position?.length === 3 && item.position.every(Number.isFinite))
    .map(item => ({ ...item, position: [...item.position], normal: item.normal ? [...item.normal] : [0, 0, 1] }));
  const retainedIds = new Set(controlItems.map(item => item.id));
  for (const id of controlVisibilityCache.keys()) if (!retainedIds.has(id)) controlVisibilityCache.delete(id);
  activeControlId = activeId; controlEditing = editing; controlInputEnabled = enabled;
  const nextMarkerKey = JSON.stringify([activeId, controlItems.map(({ id, position, normal, enabled: on }) => ({ id, position, normal, on }))]);
  if (nextMarkerKey !== markerKey) {
    markerKey = nextMarkerKey;
    disposeGroup(markerGroup);
    // Handles mark anchors; they are not disks pretending to show surface influence.
    const radius = modelSize * 0.0045;
    for (const [index, item] of controlItems.entries()) {
      const active = item.id === activeId;
      const normal = controlNormal(item), position = new THREE.Vector3().fromArray(item.position);
      const marker = new THREE.Mesh(new THREE.SphereGeometry(radius * (active ? 1.2 : 1), 16, 12),
        new THREE.MeshBasicMaterial({ color: active ? 0xffce85 : 0x63e5ee, depthTest: true }));
      marker.position.copy(position).addScaledVector(normal, radius * .7);
      marker.userData.controlId = item.id;
      markerGroup.add(marker);
      const label = controlLabel(index + 1, active, radius * 3.4);
      label.position.copy(position).addScaledVector(normal, radius * 3);
      label.userData.controlId = item.id;
      markerGroup.add(label);
    }
  }
  const item = activeControl();
  const nextRotationKey = JSON.stringify([editing, activeId, item?.position, item?.normal]);
  if (nextRotationKey !== rotationKey) {
    rotationKey = nextRotationKey;
    buildRotationGizmo(item);
  }
  anchorDirty = true; pickDirty = true;
  updateControlScreenState();
  updateMappingDepthOffset();
}

/** Read-only screen geometry, also useful to exercise real pointer gestures. */
export function getMappingControlScreenInfo() {
  if (anchorDirty) updateControlScreenState();
  return { id: activeControlId, anchor: lastAnchor ? { ...lastAnchor } : null,
    ring: rotationGizmo?.screen ? JSON.parse(JSON.stringify(rotationGizmo.screen)) : null };
}

export function getSelection() { return [...selected].sort((a, b) => a - b); }
export function clearSelection() {
  selected.clear();
  refreshColors();
  if (pickCb) pickCb(getSelection(), null);
}
export function setSelection(ids) {
  selected.clear();
  for (const id of ids) selected.add(id);
  refreshColors();
  if (pickCb) pickCb(getSelection(), null);
}
export function listFaces() {
  return group.children.map((m) => ({ ...m.userData }));
}
export function snapshot(width = 640) {
  // Render synchronously and grab pixels in the same task (no
  // preserveDrawingBuffer needed).
  renderer.render(scene, camera);
  const src = renderer.domElement;
  const c = document.createElement('canvas');
  c.width = width;
  c.height = Math.round(width * src.height / src.width);
  c.getContext('2d').drawImage(src, 0, 0, c.width, c.height);
  return c.toDataURL('image/jpeg', 0.78);
}

export function lookAtFace(faceId, zoom = 1.0) {
  const mesh = group.children.find((m) => m.userData.faceId === faceId);
  if (!mesh) return false;
  mesh.geometry.computeBoundingSphere();
  const bs = mesh.geometry.boundingSphere;
  // average normal of the face mesh
  const n = new THREE.Vector3();
  const na = mesh.geometry.attributes.normal;
  for (let i = 0; i < na.count; i++) n.add(new THREE.Vector3().fromBufferAttribute(na, i));
  n.normalize();
  if (n.lengthSq() < 0.5) n.set(0, 0, 1);
  controls.target.copy(bs.center);
  camera.position.copy(bs.center)
    .addScaledVector(n, Math.max(bs.radius * 2.6, 40) / zoom);
  controls.update();
  return true;
}
