import { prepareRibDisplay } from './rib-display.js';

self.onmessage = async ({ data }) => {
  try {
    const result = await prepareRibDisplay(data.positions, data.indices);
    const buffers = new Set([result.positions.buffer, result.normals.buffer,
      result.indices.buffer, ...result.bvh.roots]);
    if (result.bvh.indirectBuffer) buffers.add(result.bvh.indirectBuffer.buffer);
    self.postMessage({ result }, [...buffers]);
  } catch (error) {
    self.postMessage({ error: error.message });
  }
};
