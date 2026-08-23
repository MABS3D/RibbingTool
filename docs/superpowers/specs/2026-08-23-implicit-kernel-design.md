# Implicit Geometry Kernel (v1) — Design

Approved 2026-08-23 (user: build it; scope: implicit replaces the fast engine
as the default for projected applies; fast remains the unfold-mapping engine).

## Why

nTop-class ribbing quality comes from implicit (SDF) modeling, not B-rep:
booleans are `min()` and cannot fail, root fillets are smooth-min blends that
handle every rib-rib and rib-body junction uniformly, and taper/draft/crown
are field modulations. Our measured B-rep/mesh pain (OCCT fuse crashes,
T-vertices, offset self-intersections at junction cusps, snap gates,
degrade paths, bead approximations) all disappears by construction.

Benchmarks on this machine (venv, numpy/scipy/scikit-image):
field eval 0.25 s per 2M samples; marching_cubes 0.33 s per 258x258x66 tile;
EDT on a 6000x3800 grid 0.85 s. Preview apply lands ~10-20 s; export
resolution a few minutes.

## Architecture

New module `server/geometry/implicit.py`. One public entry:

    build_rib_implicit(shape, face_ids, params, quality=1.0,
                       frame_cache=None) -> (clusters, reports)

Same contract as `build_rib_meshes`: clusters is a list of watertight
`(verts float64 (n,3), tris int64 (m,3))` bodies (here: usually ONE lattice
solid), reports carry lofted/segments/warnings. Everything downstream
(overlay display, undo, STL / faceted-STEP export union, recipes) is
untouched.

### Stage 1 — pattern-space grids (once per apply, ~1-3 s)

Reuses the existing pipeline verbatim: selection -> region_meshes ->
projection frame (+frame_cache reuse, sign conventions as in ribbing.py) ->
kept-triangle filter (normal . n > 0.30) -> closed domain (slit closing) ->
`generate_segments` over the origin-symmetric lattice window (all six
patterns, offsets, border ring included as segments).

Rasterized onto one 2D grid in frame (u,v) space, cell ~= voxel size:

- **depth map** `D(u,v)`: front-most surface depth (position . n) from
  scanline-rasterizing the kept projected triangles (numpy per-triangle
  fill over its bbox; front-most = max).
- **pattern distance** `P(u,v)`: EDT of the rasterized segment network
  (scipy distance_transform_edt), so `P - thickness/2` is the 2D rib SDF.
- **domain mask + boundary distance** `B(u,v)`: rings rasterized with
  skimage.draw.polygon; EDT gives distance-to-boundary for taper; the mask
  kills the field outside the (margin-inset) domain.

### Stage 2 — the field (pure numpy, vectorized)

For sample points (u, v, d) in frame space, with bilinear grid lookups:

    half   = thickness/2 + draft_slope * clip(top - d, 0, None)   # draft
    height = H * clip(B / taper_len, 0, 1)                        # taper
    prism  = max(P - half, d - (D + height), (D - embed) - d)     # rib
    shell  = |d - D| - embed                                      # body skin
    rib    = smooth_min(prism, shell, k = fillet_root)            # root blend
    field  = rib + fillet_top-rounding on the top combination     # crown

Outside the domain mask the field is +inf (no material). The shell term is
clipped to a band near ribs so the lattice, not a full coating, is emitted:
shell participates only where P < thickness/2 + fillet_root + spacing/4.

### Stage 3 — meshing (tiled, seam-exact)

One global voxel grid in frame space (preview ~0.30 mm / quality, export
~0.15 mm). Tiles ~ (256)^2 x depth-band cells, each evaluated only within
[D_min - embed - k, D_max + height + k] of its local surface; tile sample
coordinates come from the shared global grid so shared faces see identical
field values -> marching_cubes per tile, vertices on tile boundaries weld
exactly (round to 1e-6); weld, drop degenerates, orient by signed volume,
validate with manifold3d (non-manifold tile -> report warning, still
export-unionable). Vertices transform to world via the rigid frame.

## Integration

- Engine routing (`server/main.py`): `auto` = **implicit** when
  params.mapping == "project"; unfold keeps today's rule (exact on planar,
  fast on curved). Explicit engine choices: auto | implicit | fast | exact.
  implicit + mapping=unfold -> HTTP 400 (clear message).
- UI: engine dropdown gains "implicit (field)" and **Mapping defaults to
  project**; RibParams default stays "unfold" for API compatibility (the
  byte-identity test keeps guarding it).
- Recipes store the engine; export rebuilds implicit applies at export
  resolution.
- Fillet params map directly: fillet_root -> smooth-min k; fillet_top ->
  crown rounding radius (still clipped to 0.35 * thickness).

## Non-goals (v1)

- Unfold mapping on the implicit kernel (needs an invertible chart; the
  fast engine keeps that role).
- Undercuts inside the rib band (depth map is a heightfield per frame —
  the same limitation projected mapping already has).
- Parametric B-rep output (export stays faceted STEP / STL; nTop makes the
  same trade).
- Stochastic determinism across window changes beyond what the fast engine
  provides today.

## Testing (TDD)

1. Kernel unit tests on a synthetic flat depth map (no OCCT): single rib
   volume within 3% of analytic; watertight; smooth-min k>0 adds volume
   monotonically; crown reduces volume; taper zone max height follows the
   ramp; draft widens the base.
2. Box/cylinder parity vs the fast engine: volume within 10%, watertight,
   full coverage.
3. Junction case: two crossing ribs -> ONE watertight body, no internal
   shells (manifold genus sanity via manifold3d).
4. Performance budget: box apply at preview resolution < 10 s.
5. Slow acceptance: cruscotto band pair [519,580], isogrid 12/1.2/1.5,
   taper 10, fillet_root 1.2, fillet_top 0.4 -> watertight, coverage, and
   the fillet actually present (volume vs fillet-less run).

## Risks / mitigations

- Memory: never materialize the full dense volume; tiles only, float32.
- Thin features at preview res (1.2 mm rib / 0.3 mm voxel = 4 cells):
  acceptable for preview; export res doubles it; marching cubes on the BCC
  -like snapped grid keeps ribs closed (validated by watertight tests).
- Depth-map aliasing at steep flanks: kept-filter already excludes >72 deg;
  cells straddling excluded zones fall outside the domain mask.
