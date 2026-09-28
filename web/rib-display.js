// Display-only geometry. The server's rib solids and export meshes are untouched.
import * as THREE from './vendor/three.module.js';
import { mergeVertices, toCreasedNormals } from './vendor/BufferGeometryUtils.js';
import { MeshoptSimplifier } from './vendor/meshopt-simplifier.js';
import { MeshBVH } from './vendor/three-mesh-bvh.js';

export async function prepareRibDisplay(positions, indices) {
  const start = performance.now();
  if (!positions.length || positions.length % 3 || !indices.length || indices.length % 3
      || !positions.every(Number.isFinite) || indices.some(i => i >= positions.length / 3)) {
    throw new Error('Invalid rib preview mesh');
  }
  const sourceTriangles = indices.length / 3;
  let error = 0;
  await MeshoptSimplifier.ready;
  if (sourceTriangles > 400000) {
    // ErrorAbsolute uses model units (mm). Do not prune disconnected ribs or
    // move open boundaries just to reach the requested triangle count.
    [indices, error] = MeshoptSimplifier.simplify(indices, positions, 3,
      1200000, .02, ['LockBorder', 'ErrorAbsolute']);
  }
  const [remap, count] = MeshoptSimplifier.compactMesh(indices);
  const compact = new Float32Array(count * 3);
  for (let i = 0; i < remap.length; i++) {
    const dest = remap[i];
    if (dest !== 0xffffffff) compact.set(positions.subarray(i * 3, i * 3 + 3), dest * 3);
  }
  let geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(compact, 3));
  geometry.setIndex(new THREE.BufferAttribute(indices, 1));
  const creased = toCreasedNormals(geometry, THREE.MathUtils.degToRad(38));
  if (creased !== geometry) geometry.dispose();
  // toCreasedNormals expands every triangle. Re-index including the normals,
  // preserving sharp edges while avoiding sixfold vertex-buffer expansion.
  geometry = mergeVertices(creased, 1e-4);
  if (creased !== geometry) creased.dispose();
  const tree = new MeshBVH(geometry, { maxLeafSize: 16 });
  const bvh = MeshBVH.serialize(tree, { cloneBuffers: false });
  return { positions: geometry.attributes.position.array,
    normals: geometry.attributes.normal.array, indices: geometry.index.array, bvh,
    stats: { sourceTriangles, displayTriangles: geometry.index.count / 3,
      displayVertices: geometry.attributes.position.count, errorMm: error,
      preparationMs: performance.now() - start } };
}
