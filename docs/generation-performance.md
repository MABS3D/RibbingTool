# Generation debugging and performance

The September 2026 changes preserve the sampling resolution, attachment tests,
mapping Jacobian guards and manifold validation. They reduce repeated work in
the graph engine and correct dependence on the order of earlier edits.

## Correctness changes

- Local controls start from a canonical support mesh. A previously refined
  angle/spacing layout is cached separately, so a later edit cannot inherit its
  refinement history. Returning to an accepted layout reuses its exact intrinsic
  distances. Height-only edits can reuse that same layout.
- Intrinsic distance patches use conservative triangle bounds and connectivity
  across shared edges. A triangle fan touching another fan at only one vertex
  must not receive zero distance from the native geodesic solver. Results are
  checked against the Euclidean lower bound; an unsuitable tight crop is retried
  with a larger collar under the same validity checks.
- Graph CAD meshes and selection frames are deterministic. Earlier display or
  export tessellation, and a frame left by another selection, cannot silently
  change a recipe. The legacy projected engine retains its existing frame rules.
- Clearance crossings are deduplicated in arc length to 1e-9 mm. Coinciding
  samples previously created zero-length ribbon edges and degenerate triangles.
  The endpoints and physical clearance are retained within that tolerance.

## Reuse and sampling

Prepared graphs, curves and spatial indices are shared by mapping preview,
Apply and fine export. Parameter, shape, location, orientation and selection
changes invalidate the relevant cached data. Arrays are read-only and mutable
reports are copied, preventing warnings or caller edits from contaminating the
next operation. Cache entry counts and estimated memory have finite bounds.

Triangle coordinate matrices are computed once per spatial index. Parallel
guide directions use an equivalent affine crown field; nearly degenerate
triangles retain the general interpolation semantics. Fine export still rebuilds
at quality 3; it does not substitute the coarser interactive mesh.

Display payloads retain one CAD-body snapshot and one current overlay snapshot.
Reload, exact body changes, Undo and Close model invalidate them as needed.
JSON responses avoid an additional recursive copy of the coordinate lists.
Apply progress remains active until its display response has been serialized.

Binary STL writing uses packed little-endian records in chunks of 65,536
triangles. The compatibility benchmark produced a byte-identical STL, including
the normals, triangle ordering and attribute bytes.

STL export without ribs also rebuilds and validates the body at export
resolution; it must not serialize the coarser display cache. The validated
body mesh is retained in that history entry for repeated exports. CAD faces
missing from export tessellation now stop both bare-body STL and body/rib mesh
exports. Export errors stay in the browser workspace, preserving the selected
faces and controls, instead of navigating to an error response. These checks
prevent an incomplete file from being downloaded; they do not repair a CAD
tessellation failure.

## Measured display and serialization costs

Local Windows measurements used the same STEP and mesh inputs in separate
processes, with no concurrent heavy jobs. Values below are medians of three
calls. These timings exclude solid construction.

| Operation | Before | After |
| --- | ---: | ---: |
| GET model after loading part1 | 1.426 s | 0.0127 s |
| GET model with an 87,394-triangle overlay | 1.960 s | 0.1345 s |
| Binary STL encoding of those triangles | 0.267 s | 0.0058 s |

The first CAD load still takes about 1.5 seconds in this test: caching benefits
subsequent requests. Browser parsing and rendering are not included in these
server measurements. Large models and complete dashboard exports remain more
expensive than the bounded cases.

## Geometry measurements

Unprofiled, sequential runs in the same environment, at unchanged quality:

| Case and operation | Before | After |
| --- | ---: | ---: |
| Plate, Apply at quality 1 | 3.287 s | 2.331 s |
| Plate, fine rebuild at quality 3 | 62.802 s | 39.856 s |
| Curved dashboard face 97, Apply | 6.957 s | 7.085 s |
| Curved dashboard face 97, fine rebuild | 46.289 s | 42.690 s |
| Curved dashboard face 97, repeated preview | 0.627 s | 0.00030 s |

The largest measured solid-build improvement is on the plate: 36.5% less time
for its fine rebuild. The curved case improves by 7.8%; its Apply time is
essentially unchanged. Repeated preview times are internal curve preparation
and serialization, not browser round trips. Peak process private memory remains
about 2.44 GB on the fine plate case; this is not a lower-memory extractor.

Both cases pass manifold, body-attachment and STL checks. For face 97, preview
paths, Apply/fine/union arrays and the STL bytes are exactly unchanged. The plate
has different triangulation after the ribbon-field evaluation changes: its fine
rib volume changes by 0.000323%, and the final body-union volume by 0.00593%.
Bidirectional sampled surface distances are at most 0.001963 mm for the fine
ribs and 0.02783 mm for the body union. These are sampled comparisons, not a
certified maximum Hausdorff bound. The export voxel size in this case is
approximately 0.0833 mm.

Local-field diagnostics also use both complete dashboard selections with a
70 mm control radius, 25-degree rotation and 0.85 spacing multiplier. On part2,
the first 25-degree solve drops from 25.21 to 11.64 seconds, and a 20-to-25-degree
edit from 18.32 to 8.19 seconds. Cold layouts have exactly the same arrays as the
baseline. Returning to 20 degrees now restores the original support instead of
retaining the mesh refined for 25 degrees. Repeated layouts and height edits
take approximately 0.13 seconds without another geodesic propagation. These
numbers cover the local-field stage, excluding initial CAD mapping and solid
construction.

On part1 the formerly rejected 25-degree case now passes in 21.72 seconds.
It needs one more exact refinement pass than the old six-refinement limit
allowed. Its cold result and the result after a 20-degree edit have identical
arrays; repeating the layout takes about 0.14 seconds. The two real dashboard
regressions are retained as slow tests in `tests/test_mapping_controls.py`.

## Incomplete CAD tessellation

Generation also checks that every selected CAD face survives region meshing,
and that every body face has usable triangles before constructing the distance
field. Missing faces are reported with their IDs; incomplete source meshes are
not cached or used to generate ribs.

An additional audit of `180 90 cruscotto_Assieme.step` found 966 CAD faces but
only 963 meshed at the graph resolution: faces 355, 738 and 787 were omitted.
The complete shape and those three faces pass `BRepCheck_Analyzer`; this is an
observed tessellation limitation, not evidence that the STEP is corrupt.
Returning triangles alone was insufficient: four more conical faces had
partial coverage, and two narrow B-spline faces lost part of their contour.
`cad_tessellation.py` now checks triangle boundaries against CAD edge polygons.
Small B-spline slivers receive a local native remesh with their neighboring
faces; supported regular faces can be recovered by constrained triangulation
in UV using the exact shared 3D contour. Refinement, area, normal and contour
checks reject incomplete or folded recoveries. Surfaces, edges, tolerances,
face IDs and CAD volume are unchanged. Singular parameterizations that cannot
be recovered still fail explicitly.

The assembly now supplies all 966 faces to generation. Export converts each
closed CAD solid separately, welds shared CAD edge nodes, then unions solids;
this avoids making assembly contact faces non-manifold by raw concatenation.
Binary STL is checked at its actual float32 precision. Collapsed triangles
are reconstructed as a solid, with component, volume and closed-edge checks,
rather than dropped individually. Irrecoverable precision loss aborts export.
The separate part1 and part2 reference files remain covered by regression
tests (506/506 and 463/463 faces).

## Reproducing the geometry benchmark

From the repository in PowerShell:

```powershell
.venv/Scripts/python.exe scripts/benchmark_generation.py --code-root . --case plate --output output/benchmark-plate.json --save-meshes --save-stl
.venv/Scripts/python.exe scripts/benchmark_generation.py --code-root . --case dashboard --faces 97 --output output/benchmark-dashboard.json --save-meshes --save-stl
```

The plate is 60 x 40 x 8 mm. The dashboard case selects the curved face 97 of
`testdata/part1.stp` on its complete original CAD body; it does not benchmark a
full dashboard selection. Use `--model` for another STEP and its original face
IDs. For a before/after comparison, use separate source directories and the
same input, parameters and dependency environment. Run them sequentially.

The harness separates cold/warm preview, Apply, fine rebuild, body union and STL
encoding. It records source/input hashes, geometry metrics and memory, and can
save meshes for a geometric comparison. `--profile` adds cProfile output for
diagnosis; compare unprofiled runs for timing claims.

A completed full run must pass closed-manifold, attachment, preview consistency
and STL validity checks. A failed quality gate returns a nonzero exit code.
The default timeout is 180 seconds. The 8 GiB private-memory guard is supported
on Windows; other platforms enforce the timeout only. Partial results survive
a timeout or a worker failure, and only that benchmark's process is terminated.
