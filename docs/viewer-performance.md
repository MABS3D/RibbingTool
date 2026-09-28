# Viewer performance after Apply

The failing reference case is the dashboard assembly with faces 17, 277, 280,
295 and 428 selected and one local rotation control. Its generated overlay
contains 6,861,338 triangles and 3,429,409 indexed vertices. The model response
is approximately 252 MB of JSON.

Two costs dominated interaction: `toCreasedNormals` expanded the overlay to
20,584,014 vertices, and camera movement triggered unaccelerated occlusion
raycasts through all rib triangles for the mapping-control anchor. The body
itself has only 32,229 display triangles.

## Changes

- A module Web Worker prepares the rib display. Meshoptimizer simplifies large
  overlays with a 0.02 mm absolute **estimated** error limit and locked open
  boundaries; component pruning is disabled. The triangle target is a request,
  not permission to exceed the error limit.
- The creased normals are retained, then vertices are indexed again using both
  position and normal. Sharp corners remain sharp.
- A BVH built in the worker accelerates rib occlusion queries. Picking still
  targets the original CAD faces; IDs, selection and geometry generation are
  unchanged.
- Orbit gestures defer unnecessary hover picking. Rendering occurs when the
  camera or scene changes, rather than continuously while idle.
- Closing or replacing a model cancels its worker. Revision guards prevent a
  delayed result from overwriting the new model. GPU resources are disposed.

These changes affect the browser only. Server solids, recipes, Undo history,
STL/STEP generation and export precision are unchanged. No server restart is
needed; reloading the page resumes the existing model.

## Measurements

Replayed the same captured model and control in Chromium using ANGLE/D3D11
on an NVIDIA GTX 1070 Ti, with the same 1100 × 800 viewport and pointer path.
Raw measurements: [before](benchmarks/viewer-before.json), [after](benchmarks/viewer-after.json).
Times below describe this local browser benchmark, not guaranteed frame rates
on every device or every model.

| Measurement | Before | After |
| --- | ---: | ---: |
| Rib triangles displayed | 6,861,338 | 1,900,160 |
| Rib vertices displayed | 20,584,014 | 1,731,271 |
| Median frame interval during orbit | 888.9 ms | 6.9 ms |
| 95th percentile frame interval during orbit | 972.1 ms | 7.0 ms |
| Renders during a three-second idle sample | 437 | 1 |
| Display preparation elapsed time | 13.71 s | 12.56 s |

Preparation still takes time: the improvement moves that work off the UI
thread and removes repeated work while navigating. The large JSON response
also remains; transport size and preparation latency are separate future work.
The simplifier's error estimate is not a certified surface-distance bound.

## Browser regressions

```powershell
npm install --no-save playwright
npx playwright install chromium
node scripts/verify_viewer.cjs
```

Set `CHROME_PATH` to use an existing Chromium executable. The test serves an
isolated local page and checks sharp cube corners, small and large meshes,
retention of a separate component, BVH hits against ordinary raycasts, invalid
input, worker cancellation, model replacement and Close. It never connects to
the user's running CAD server.

The real app was also checked for control placement, popup reopening, numeric
edits, rotation gestures, Escape, Shift snapping, mapping save/load, resize,
model close/reload and draft restoration. A real Apply followed by STL and STEP
downloads, Undo and Close also passed. The related backend regression selection
passed 13 tests.

Dependencies and license notices are in [web/vendor](../web/vendor/README.md).
