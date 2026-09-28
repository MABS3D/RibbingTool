# RibbingTool

Local CAD tool for adding periodic rib patterns to STEP models — including
fully freeform curved faces. Open a `.step/.stp`, click faces in 3D (or grow a
whole tangent-connected skin from one click), pick a pattern and parameters,
apply, export STEP/STL. Built for 3D-printing stiffening/lightweighting.

![patterns](img/hero.jpg)

## Setup (Windows, pip-only)

```bash
uv venv --python 3.13 .venv
uv pip install --python .venv -r requirements.txt
```

## Run

```bash
.venv\Scripts\python.exe -m uvicorn server.main:app --port 8317
```

Open http://localhost:8317 and load a STEP file.

## Workflow

1. **Load** a STEP file (file picker, or `POST /api/load_path {"path": ...}`).
2. **Select faces** — click to toggle. **Grow tangent** expands the selection
   across smoothly-connected faces (fillets, blended skins); faces with
   curvature radius under ~2.5 mm act as barriers so growth doesn't flood
   through rounded wall ends onto the far side of a shell.
3. **Pattern + parameters** — rectangular, quadmesh, triangular, isogrid,
   hexagonal, stochastic (seeded Voronoi); spacing, thickness, height,
   orientation, draft, boundary margin, optional border rib (welds open rib
   ends into one frame), edge taper (without a border, rib height ramps to
   zero over `taper_len` mm at open boundaries — stiffener run-outs).
4. **Apply.** Surface mapping follows the selected faces with connected rib
   curves, balancing cell stretch across folds. Project uses a frontal grid;
   Unfold retains the legacy flattening engines.
5. **Export** STEP or STL. Undo steps back one apply.

### Edit the mapping before Apply

**Mapping studio** previews attachment paths and height guides without building
solids. Add controls directly on selected faces to blend a local rotation,
spacing multiplier and height multiplier over a chosen surface radius. Disable,
move or undo controls, and save/load the layout as JSON. The same mapping is
used by preview, Apply and fine export. These are manually prescribed design
fields. See [the workflow and current limits](mapping-controls.md).

## Engines

`auto` uses the **surface graph** engine for **Surface** and **Project** mapping.
Surface is the UI default: it balances cell stretch along the selected faces,
including lateral walls and folds. Project preserves a frontal lattice and
stretches cells on steep faces. The graph construction separates
wall thickness, top rounding, root blending, junction blending, and guide
smoothing. The geometry is a sampled mesh; STEP/STL exports rebuild it at a
finer resolution and unite it with the body. See [surface graph details](surface-graph.md).

The previous surface-field construction remains available as `implicit`
("legacy surface field") for comparison. For unfold mapping, the other engines
remain available:

| | fast (default on curved) | exact |
|---|---|---|
| Apply | instant-ish: ribs are exact B-rep solids kept as an overlay; no boolean | OCCT fuse into the body |
| Export STEP | faceted (triangulated) STEP via mesh union | smooth B-rep STEP |
| Export STL | mesh union (watertight) | tessellated fused body |
| Re-ribbing | fine — the body B-rep is never modified | fine |
| Scale | thousands of ribs on real parts | small jobs; planar faces |

With unfold mapping, `auto` uses exact when every selected face is planar,
fast otherwise.

Why: OCCT mass-fuses of many rib solids against real curved bodies are
unreliable at scale — measured on the test part (142 ribs, 506-face body):
direct fuse crashes the process, chunked/OBB variants ran 26–69 minutes and
returned corrupted results. The fast engine sidesteps booleans at apply time
entirely and unions in mesh space (manifold3d) only at export, in seconds.
When the exact engine's fuse fails or produces a corrupt result, it
automatically falls back to the mesh path and marks the output faceted.

## How unfold mapping works

Per connected region of selected faces: weld the face triangulations along
OCCT's shared-edge node polygons → LSCM-flatten the region (seam-safe;
non-disk topology is cut open with `igl.cut_to_disk`) → generate the 2D
segment network → map each rib footprint back through barycentric → UV →
exact surface evaluation → build each rib as a solid between `surface − embed`
and `surface + height` along true surface normals (drafted top). Ribs whose
mapping shows folds, slit-jumps, or extreme conformal stretch are skipped and
counted in the per-region report.

## Known limits

- Surface graph exports are faceted, not analytic CAD fillets. Complete dense
  models can take minutes to generate, and fine exports require substantial
  memory. The `quadmesh` pattern is a square lattice; it is not a curvature-
  or stress-optimized quad network.
- In unfold mapping, conformal (LSCM) flattening preserves angles, not areas: on strongly curved
  or pinched regions the pattern spacing drifts, and ribs in extreme zones
  are skipped (reported). A curvature-aligned quad parametrization (nTop-style
  "quad mesh" flow) is out of scope for v1.
- Region borders can look ragged: clipped rib ends at the boundary, and
  mildly folded rib tops over tight concave transitions. Increasing the
  boundary margin and enabling the border rib cleans most of it.
- Fast-engine STEP output is faceted (fine for slicers; heavy for downstream
  CAD). Exact smooth STEP works best on planar faces or small curved jobs.
- Closed periodic faces (full cylinders) unroll at their CAD seam; the
  pattern does not join across the seam line.
- Assemblies retain their solids and original face IDs. Mesh export unions
  closed solids; CAD faces that cannot be completely tessellated stop
  generation with their IDs instead of silently leaving holes.

## Roadmap

V2 direction (see `docs/superpowers/specs/2026-08-14-v2-stress-aligned-roadmap.md`):
load-driven rib layouts per Li et al. 2017 (*Rib-reinforced Shell
Structure*: FEA → principal-stress-aligned quad mesh → rib network →
contribution-based simplification → T-sections) and Ding & Yamazaki 2005
(adaptive growth stiffeners).

## Tests

```bash
.venv\Scripts\python.exe -m pytest            # fast suite
.venv\Scripts\python.exe -m pytest -m slow    # cruscotto acceptance (minutes)
```

The bounded geometry benchmark separates preview, Apply, fine rebuild and
export, with closed-mesh and attachment checks. See
[generation debugging and performance](generation-performance.md) for
measured results and commands.

## Layout

- `server/geometry/` — step_io, meshing (+ region welding), flatten (LSCM +
  cuts), patterns (2D networks), ribbing (mapping + solids + fuse),
  booleans (OCCT + manifold3d), selection (grow tangent)
- `server/main.py` — FastAPI app
- `web/` — three.js viewer (face picking, overlay display), panel UI
- `testdata/` — cruscotto Part 1/2 test models
