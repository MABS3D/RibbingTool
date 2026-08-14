# RibbingTool — Design Spec

**Date:** 2026-08-14
**Status:** Approved (user approved Approach A: B-rep loft pipeline; test subjects designated)

## Purpose

A local CAD tool that opens STEP files, lets the user select faces in a 3D UI, applies
parametric periodic rib patterns to those faces (including fully freeform curved faces),
and exports the ribbed solid as STEP. Primary use case: stiffening/lightweighting parts
for 3D printing.

## Test subjects

- `testdata/part1.stp` — "180 90 cruscotto_Part 1" (1 solid, 506 faces, ~508 cm³)
- `testdata/part2.stp` — "180 90 cruscotto_Part 2" (1 solid, 463 faces, ~495 cm³)

Target faces: the external curved skin of each part (wraps over a rounded corner —
nearly developable, so flattening distortion should be low). Acceptance: ribbing applied
to those faces produces a valid, volume-increased solid that round-trips through STEP.

## Architecture

Local single-user web app.

- **Backend** (Python 3.13, `.venv` via uv): FastAPI + uvicorn.
  - Geometry kernel: OCCT 7.9.3 via `cadquery-ocp` (verified installed and loading the
    test parts).
  - Flattening: libigl 2.6.2 (LSCM); hand-rolled scipy LSCM as fallback if igl
    misbehaves.
  - 2D pattern math: shapely 2.1 + scipy (Voronoi).
  - Fallback boolean engine: manifold3d 3.5 (mesh booleans, used only if OCCT fuse
    fails; output then marked faceted).
- **Frontend**: static HTML + vendored three.js (no build step). 3D viewer with
  hover-highlight, click multi-select of faces, parameter panel, Apply / Undo / Export.

## Core pipeline (per selected face)

1. **Mesh** the face (OCCT `BRepMesh_IncrementalMesh`), keeping per-vertex UVs.
2. **Flatten** the face mesh to 2D with LSCM (planar faces flatten exactly; freeform
   conformally). Reject faces that are not topological disks with a clear error
   ("split this face in your CAD first").
3. **Generate pattern** in flattened space: segment network per pattern type, rotated
   by orientation angle, clipped to the flattened boundary inset by `margin`.
4. **Map back**: sample each segment footprint polygon, flattened point → containing
   triangle → barycentric → UV → exact surface point and normal
   (`GeomLProp_SLProps`).
5. **Loft rib solids**: per segment, bottom loop at `surface − embed` (default 0.3 mm,
   into material), top loop at `surface + height`; draft shrinks the top footprint by
   `height·tan(draft)`. B-spline wires + `ThruSections` (solid mode).
6. **Fuse**: all rib segments fused together first, then one fuse with the body
   (fuzzy tolerance, parallel mode). Per-segment loft failures skip-and-report.
7. **Export**: AP214 STEP; STL as a bonus for printing.

## Patterns

All defined as 2D segment networks in flattened space, buffered to rib thickness:

| Pattern | Definition | Extra params |
|---|---|---|
| rectangular | two orthogonal line families | spacing_x, spacing_y (one may be off → parallel ribs) |
| quadmesh | square grid preset (equal spacings) | spacing |
| triangular | three line families at base angle + 0°/60°/120° | spacing, base angle |
| isogrid | equilateral triangular preset | spacing |
| hexagonal | honeycomb cell walls | cell size (across flats) |
| stochastic | Voronoi edges of seeded blue-noise points | density, seed |

Optional **border rib** along the face outline (at the inset margin).

**Shared parameters:** rib height, rib thickness, orientation angle, draft angle,
boundary inset margin, embed depth. Units: mm and degrees.

## Known limits (v1)

- Conformal flattening distorts area on strongly curved regions → pattern spacing
  drifts there. Accepted trade-off.
- Closed periodic faces (full cylinders, tori) arrive from OCCT pre-cut at the
  parametric seam and unroll fine — but the rib pattern does not join across the seam
  line. Only genuinely boundary-less/genus>0 meshes are rejected with an explanatory
  error. (Verified: a full cylinder lateral face unrolls near-isometrically.)
- Warn when rib height exceeds the local minimum curvature radius (self-intersection
  risk); proceed anyway.
- No assemblies (first/only solid per file), no variable-density patterns, no fillets
  at rib roots, no session persistence.

## Error handling

- Every geometry operation wrapped; failures surface as UI banner messages, never
  silent.
- Per-segment loft failures: skip, count, report ("N of M ribs failed").
- OCCT fuse failure/timeout → automatic manifold3d mesh-boolean fallback, output
  visibly marked as faceted.

## Testing

- pytest, headless (no UI required): in-code STEP fixtures (box, cylindrical shell,
  freeform loft) + the two cruscotto parts.
- Assertions: output passes `BRepCheck_Analyzer`, volume strictly increases, STEP
  round-trips, rib count sanity per pattern.
- Golden tests for 2D pattern generators; planar flattening must be identity up to
  rigid transform.
- UI verified end-to-end by driving the browser (load → select → apply → export).

## Project layout

```
server/
  main.py            # FastAPI app + routes
  geometry/
    step_io.py       # STEP read/write, shape session store
    meshing.py       # per-face tessellation with UVs + face IDs
    flatten.py       # LSCM flattening (igl + scipy fallback)
    patterns.py      # 2D segment networks for all six patterns
    ribbing.py       # map-back + loft + per-face orchestration
    booleans.py      # OCCT fuse + manifold3d fallback
web/
  index.html, app.js, viewer.js, three vendored
testdata/            # part1.stp, part2.stp
tests/
scripts/
```
