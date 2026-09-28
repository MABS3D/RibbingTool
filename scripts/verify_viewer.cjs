// Browser regressions for display preparation, BVH picking and worker cancellation.
// npm install --no-save playwright; npx playwright install chromium
// node scripts/verify_viewer.cjs (optional CHROME_PATH for an existing Chromium)
const { chromium } = require('playwright');
const assert = require('node:assert/strict');
const http = require('node:http');
const fs = require('node:fs');
const path = require('node:path');
const root = path.resolve(__dirname, '..');
let browser, server;

(async () => {
  server = http.createServer((request, response) => {
    if (request.url === '/') {
      response.setHeader('Content-Type', 'text/html');
      response.end('<div id="view" style="width:640px;height:480px"></div>');
      return;
    }
    const file = path.resolve(root, '.' + request.url);
    if (!file.startsWith(root + path.sep)) { response.writeHead(403).end(); return; }
    response.setHeader('Content-Type', 'text/javascript');
    fs.createReadStream(file).on('error', () => response.destroy()).pipe(response);
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  browser = await chromium.launch({ headless: true,
    ...(process.env.CHROME_PATH ? { executablePath: process.env.CHROME_PATH } : {}),
    args: ['--enable-unsafe-swiftshader'] });
  const page = await browser.newPage();
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.goto(`http://127.0.0.1:${server.address().port}/`);
  const result = await page.evaluate(async () => {
    const THREE = await import('/web/vendor/three.module.js');
    const { mergeGeometries } = await import('/web/vendor/BufferGeometryUtils.js');
    const { MeshBVH, acceleratedRaycast } = await import('/web/vendor/three-mesh-bvh.js');
    const viewer = await import('/web/viewer.js');
    viewer.initViewer(document.getElementById('view'));
    function prepare(geometry) {
      return new Promise((resolve, reject) => {
        const worker = new Worker('/web/rib-display-worker.js', { type: 'module' });
        worker.onmessage = ({ data }) => { worker.terminate(); data.error ? reject(Error(data.error)) : resolve(data.result); };
        worker.onerror = error => { worker.terminate(); reject(Error(error.message)); };
        worker.postMessage({ positions: geometry.attributes.position.array, indices: geometry.index.array });
      });
    }
    function geometryFrom(result) {
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute('position', new THREE.BufferAttribute(result.positions, 3));
      geometry.setAttribute('normal', new THREE.BufferAttribute(result.normals, 3));
      geometry.setIndex(new THREE.BufferAttribute(result.indices, 1));
      return geometry;
    }
    const cube = new THREE.BoxGeometry(2, 2, 2);
    const small = await prepare(cube);
    // Keep the hard corner: there must still be separate normals per cube side.
    const cornerNormals = new Set();
    for (let i = 0; i < small.positions.length; i += 3) {
      if (small.positions[i] === 1 && small.positions[i + 1] === 1 && small.positions[i + 2] === 1)
        cornerNormals.add(Array.from(small.normals.slice(i, i + 3)).join(','));
    }
    // This used to fall between the simplification threshold and target count.
    const middle = await prepare(new THREE.PlaneGeometry(10, 10, 400, 400));
    const torus = new THREE.TorusGeometry(20, 3, 256, 1024);
    const separate = cube.clone().translate(45, 0, 0);
    const big = mergeGeometries([torus, separate]);
    const reduced = await prepare(big);
    const geometry = geometryFrom(reduced);
    geometry.boundsTree = MeshBVH.deserialize(reduced.bvh, geometry, { setIndex: false });
    const material = new THREE.MeshBasicMaterial({ side: THREE.DoubleSide });
    const mesh = new THREE.Mesh(geometry, material);
    mesh.updateMatrixWorld();
    const ray = new THREE.Raycaster(); ray.firstHitOnly = true;
    let hitAgreement = true;
    for (const [x, y] of [[20, 0], [0, 20], [0, 0], [45, 0], [70, 0]]) {
      ray.set(new THREE.Vector3(x, y, 50), new THREE.Vector3(0, 0, -1));
      const plain = []; THREE.Mesh.prototype.raycast.call(mesh, ray, plain);
      const fast = []; acceleratedRaycast.call(mesh, ray, fast);
      plain.sort((a, b) => a.distance - b.distance);
      fast.sort((a, b) => a.distance - b.distance);
      hitAgreement &&= !!plain.length === !!fast.length && (!plain.length || Math.abs(plain[0].distance - fast[0].distance) < 1e-6);
    }
    const bounds = new THREE.Box3().setFromBufferAttribute(geometry.attributes.position);
    const normalsFinite = reduced.normals.every(Number.isFinite);
    // Cancel an in-flight worker and load another model. Its late result must
    // never replace the new model or reappear after Close.
    const overlay = { positions: cube.attributes.position.array, indices: cube.index.array };
    const model = { faces: [], overlay, bbox: [-1, -1, -1, 1, 1, 1] };
    const stale = viewer.loadModel(model);
    viewer.clearModel();
    const cancelled = await stale;
    const loaded = await viewer.loadModel(model);
    const ready = viewer.getDisplayStats();
    viewer.clearModel();
    await new Promise(resolve => setTimeout(resolve, 100));
    let rejected = false;
    const invalid = cube.clone(); invalid.attributes.position.array[0] = NaN;
    try { await prepare(invalid); } catch { rejected = true; }
    return { cubeTriangles: small.stats.displayTriangles, cornerNormals: cornerNormals.size,
      middleTriangles: middle.stats.displayTriangles, reduced: reduced.stats,
      normalsFinite, hitAgreement, retainedSeparateComponent: bounds.max.x === 46,
      cancelled, loaded, ready: !!ready, cleared: viewer.getDisplayStats() === null,
      rejectedInvalid: rejected };
  });
  assert.equal(result.cubeTriangles, 12);
  assert.equal(result.cornerNormals, 3);
  assert.equal(result.middleTriangles, 320000);
  assert.ok(result.reduced.displayTriangles < result.reduced.sourceTriangles);
  assert.ok(result.reduced.errorMm <= .02);
  for (const key of ['normalsFinite', 'hitAgreement', 'retainedSeparateComponent', 'loaded', 'ready', 'cleared', 'rejectedInvalid'])
    assert.equal(result[key], true, key);
  assert.equal(result.cancelled, false);
  assert.deepEqual(errors, []);
  console.log(JSON.stringify({ passed: true, ...result }, null, 2));
})().catch(error => { console.error(error); process.exitCode = 1; })
  .finally(async () => { await browser?.close(); server?.close(); });
setTimeout(() => { console.error('Viewer regression deadline exceeded'); process.exit(2); }, 120000).unref();
