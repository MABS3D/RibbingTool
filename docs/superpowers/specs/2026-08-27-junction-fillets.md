# SPEC: Rib-rib junction fillets + draft parity (author: orchestrator, from LEAP71/nTop source study)

## approach

nTop 5.10 lists "rib-rib fillets" as a first-class field-driven knob. LEAP71/PicoGK get clean
junctions for free because their atomic primitive is `AddBeam(A, radA, B, radB, bRoundCap=true)` —
capped cone-beams whose smooth voxel union rounds every meeting. Our junctions instead come from a
HARD `min` over pattern segments in `_exact_pattern_distance`: the 2D distance field creases on the
junction bisectors, the rib prism inherits each crease as a sharp vertical seam running up the rib
walls at every isogrid node, and junction crowns concentrate the pinch/micro-hole repairs (the
"crater" class in the user's rim screenshot). Fix at the 2D source: replace the hard min with a
compact-support polynomial smooth-min of the TWO nearest segment distances, radius `k_j` =
`params.fillet_junction` (new, default 0.0 = bit-identical off state).

## algorithm

1. `_exact_pattern_distance` today keeps one running `d = minimum(d, d_seg)` per raster cell
   (vectorized per segment, Liang-Barsky clipped). Change to a running TWO-min: per segment update,
   `m1 = d_seg < d1` shifts d1→d2 (`d2 = where(m1, d1, minimum(d2, d_seg))`; `d1 = where(m1, d_seg, d1)`).
   One extra compare+select per (segment, cell) — same asymptotics.
2. Output `P = smin_poly(d1, d2, k_j)` with the codebase's mix form
   (`h = clip(0.5 + 0.5*(d2-d1)/k_j, 0, 1); P = d2*h + d1*(1-h) - k_j*h*(1-h)`), else `P = d1`
   when `k_j == 0`. Compact support ⇒ anywhere the second segment is ≥ k_j farther, P == d1 exactly
   (mid-segment field untouched; only junction neighborhoods round).
3. Clamp `k_j <= spacing/4` (junction lobes must not swallow pattern windows) and floor the effect
   at `k_j >= 2*cell` when nonzero (sub-raster rounding just aliases).
4. Segments sharing an ENDPOINT (the isogrid node case) both go to ~0 at the node, so smin there
   deepens P by ~k_j/4 — that is the FILLET (material widens by design at junctions, exactly what
   a molded rib node looks like). This is intended behavior, not the welt defect class: here the
   two branches are DIFFERENT segments by construction; no idempotency contract exists (contrast
   round-8 lesson (c), which is about blending a value with itself).

## draft parity (bundled, small)

nTop's knob set: thickness, height, draft, extrusion direction. Extrusion direction = our frame-z
(mold draw axis) — already parity. Draft exists in the UI but has NO kernel test. Add one:
kernel fixture, flat substrate, draft_deg=2: fit both rib flank planes over s in [0.5, 3.5],
assert flank-vs-draw-axis angle = 2.0° ± 0.2° on BOTH sides, and that the cap nose inherits the
widening toward the root (q_end nose section at low s wider than at high s). If the current
implementation fails the molded convention (angle measured from draw axis, applied symmetrically),
fix the `half` draft term to match — the convention is the spec.

## integration

- `_exact_pattern_distance(segs, cell, origin, shape2d, reach)` gains `k_j=0.0` param; only the P
  build call in `build_rib_implicit` passes it (B/Bo/OB calls untouched — boundary distances must
  stay exact, no rounding).
- `RibParams.fillet_junction: float = 0.0`; UI slider "Junction fillet (mm)" under Root/Top fillet;
  API passthrough + recipe serialization (params dict already round-trips).
- Kernel (`_rib_field`) untouched — it consumes P and inherits the rounding; `r_top` crown rounding
  rides the rounded ridge automatically. Stretch correction uses Pgu/Pgv = np.gradient(P): smin is
  C1 so gradients stay clean (hard-min P was only C0 — this IMPROVES the stretch field at nodes).

## tests (feature-TDD: write failing first)

- New kernel test `test_junction_fillet_radius`: grid pattern with one perpendicular crossing,
  k_j=2.0. Slice the extracted mesh at mid-height (s ∈ [1.4, 1.6] thin slab), take the inner
  corner contour at the junction quadrant, fit a circle to the corner arc: assert fitted radius
  ≥ 0.7·k_j (today, k_j absent → inner corner radius ≈ extraction rounding ≈ res ≪ 1.4 → fails).
- Byte-identity guard: k_j=0 build hash == pre-change hash on the same fixture.
- Existing suite must stay green untouched (default 0).

## risks / perf

- Triple junctions (isogrid 6-way stars): top-2 smin rounds each sector pairwise; the fillet is
  slightly asymmetric per sector — visually fine (sectors are 60°); exact top-3 variant is a
  documented follow-up if star centers look faceted.
- Two-min accumulation: +1 compare/select per segment-cell — P build is a minor cost next to igl;
  expect <1% apply delta.
- Crown pinch repairs at junctions should DROP (wider crown ridge at nodes = fewer sub-voxel
  pinches); count `_repair_pinches` events on the assieme before/after as evidence.

## fallback

`fillet_junction=0` (default) is provably byte-identical (compact support + explicit k_j==0
branch). Ship dark, expose the slider once the kernel test is green.
