import * as THREE from './vendor/three.module.js';
import { OrbitControls } from './vendor/OrbitControls.js';

const BASE = new THREE.Color(0x8a8f98);
const HOVER = new THREE.Color(0xaab2c0);
const SELECTED = new THREE.Color(0xff8c2f);

let renderer, scene, camera, controls, group, overlayGroup;
let raycaster, pointer, hovered = null;
let pickCb = null;
const selected = new Set();
const RIB = new THREE.Color(0xb9a184);

export function initViewer(container) {
  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  container.appendChild(renderer.domElement);

  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x16181d);

  camera = new THREE.PerspectiveCamera(45, 1, 0.1, 10000);
  camera.position.set(120, 90, 120);

  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;

  scene.add(new THREE.HemisphereLight(0xf4f6ff, 0x33363d, 1.0));
  const key = new THREE.DirectionalLight(0xffffff, 1.6);
  key.position.set(1, 2, 1.5);
  scene.add(key);
  const fill = new THREE.DirectionalLight(0xb8c4ff, 0.5);
  fill.position.set(-1.5, -1, -1);
  scene.add(fill);

  group = new THREE.Group();
  scene.add(group);
  overlayGroup = new THREE.Group();
  scene.add(overlayGroup);

  raycaster = new THREE.Raycaster();
  pointer = new THREE.Vector2(-2, -2);

  const el = renderer.domElement;
  el.addEventListener('pointermove', (e) => {
    const r = el.getBoundingClientRect();
    pointer.set(((e.clientX - r.left) / r.width) * 2 - 1,
                -((e.clientY - r.top) / r.height) * 2 + 1);
  });
  let downAt = null;
  el.addEventListener('pointerdown', (e) => { downAt = [e.clientX, e.clientY]; });
  el.addEventListener('pointerup', (e) => {
    if (!downAt) return;
    const moved = Math.hypot(e.clientX - downAt[0], e.clientY - downAt[1]);
    downAt = null;
    if (moved > 4 || e.button !== 0) return;   // it was an orbit drag
    const hit = pick();
    if (hit) {
      const id = hit.userData.faceId;
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
  };
  new ResizeObserver(resize).observe(container);
  resize();

  renderer.setAnimationLoop(() => {
    controls.update();
    const hit = pick();
    if (hit !== hovered) { hovered = hit; refreshColors(); }
    renderer.render(scene, camera);
  });
}

function pick() {
  raycaster.setFromCamera(pointer, camera);
  const hits = raycaster.intersectObjects(group.children, false);
  return hits.length ? hits[0].object : null;
}

function refreshColors() {
  for (const mesh of group.children) {
    const id = mesh.userData.faceId;
    const c = selected.has(id) ? SELECTED : (mesh === hovered ? HOVER : BASE);
    mesh.material.color.copy(c);
    mesh.material.emissive.set(selected.has(id) ? 0x552a00 : 0x000000);
  }
  document.body.style.cursor = hovered ? 'pointer' : 'default';
}

export function loadModel(data) {
  for (const g of [group, overlayGroup]) {
    for (const m of g.children) {
      m.geometry.dispose();
      m.material.dispose();
    }
    g.clear();
  }
  selected.clear();
  hovered = null;

  if (data.overlay) {
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position',
      new THREE.Float32BufferAttribute(new Float32Array(data.overlay.positions), 3));
    geo.setIndex(new THREE.Uint32BufferAttribute(new Uint32Array(data.overlay.indices), 1));
    geo.computeVertexNormals();
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
  fitCamera(data.bbox);
  refreshColors();
  if (pickCb) pickCb(getSelection(), null);
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

export function lookAtFace(faceId) {
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
  camera.position.copy(bs.center).addScaledVector(n, Math.max(bs.radius * 2.6, 40));
  controls.update();
  return true;
}
