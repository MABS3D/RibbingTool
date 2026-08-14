# V2 roadmap: load-driven, stress-aligned ribbing

Inspired by the two papers the user supplied (Downloads):

- **li2017.pdf** — Li et al., *Rib-reinforced Shell Structure* (Pacific
  Graphics 2017). The full pipeline this tool should grow toward:
  1. FEA on the shell (CST membrane + DKT bending; ribs as beam elements
     stiffness-superimposed on the shell).
  2. Principal-stress field → global parametrization (MIQ) → **quad mesh
     whose edges align with principal stress directions**.
  3. Extract rib network (curved / circular / tree-like elements) from the
     quad mesh.
  4. Simplify: drop ribs by strain-energy-decrement contribution.
  5. Rib-flow optimization (ribs swing on the surface) + cross-section
     optimization (hyperelliptic T-sections avoid corner stress
     concentration).
- **ding2005.pdf** — Ding & Yamazaki, *Adaptive growth technique of
  stiffener layout* (Eng. Opt. 37:3). Alternative generator: stiffeners grow
  and branch from seed points along maximum strain-energy-sensitivity
  directions under a volume budget — organic tree-like layouts.

## Incremental plan (each step useful on its own)

1. **Curvature-aligned orientation (no FEM)** — set the pattern orientation
   per region from the dominant principal-curvature direction instead of
   PCA; cheap proxy for stress alignment on shells.
2. **FEA integration** — scikit-fem or a small DKT shell implementation on
   the region mesh; user picks anchors (supports) and load faces in the UI.
   Output: principal stress directions + magnitudes per triangle.
3. **Stress-aligned quadmesh pattern** — replace the uniform lattice in flat
   space with a cross-field-guided quad pattern (libigl has `miq` /
   cross-field tooling); "quadmesh" pattern gains an `align: stress` mode.
4. **Contribution-based simplification** — build the dense network, drop
   ribs whose removal changes compliance least (greedy, re-solve or use
   sensitivity from ding2005 eq. 3–5).
5. **T-section profiles** — capsule footprint extrudes a T (or hyperelliptic
   T) instead of a rectangle; parameters: flange width/thickness.
6. **Adaptive growth mode** — ding2005's seeded growth as a separate
   "organic" pattern type.

Steps 1 and 5 are days; 2–4 are the real project (weeks).
