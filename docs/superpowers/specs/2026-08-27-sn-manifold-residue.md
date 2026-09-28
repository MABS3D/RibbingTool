# SPEC: SurfaceNets manifold-residue diagnostic (probe-first; author: orchestrator)

## context

SN on the assieme leaves 66 boundary + 3 non-manifold edges after the repair stack (0.003% of
2.47M edges; MC passes the same apply) — one of the two blockers keeping SN off default (the other
is 1.71x cost). SN-inherent: identical at R6/G7 and R4/G5. NO CODE until the residue is classified;
each candidate class has a different, cheap fix — guessing risks fixing none.

## probe plan (one script, scratchpad, env-flagged SN build)

1. Build assieme under RIBBING_EXTRACTOR=surface_nets with a post-repair hook (monkeypatch around
   the manifold census in mesh_field) capturing: bad-edge endpoint coordinates (world), edge kind
   (boundary vs >2-incidence), and the component/patch each belongs to.
2. Transform to frame `(w − ctr) @ M` and classify EVERY bad edge against five hypotheses:
   (a) GATE CLIFFS: nz of nearest substrate point in [0.09, 0.15] (the binary facing gate's
       cut plane slicing ribs → open sheets the hole-filler can't walk);
   (b) CREASE CELLS: inside the blend's flagged-cell mask (exact_ok=False cells relax on the
       trilinear surrogate; a seam between exact-snapped and surrogate-snapped vertices could
       leave hairline gaps) — dump the cfl mask alongside;
   (c) BAND-LID RIMS: columns where the leaning-crown lid changed dhi (Dhid ≠ Dhi) — restored
       material meeting unrestored columns at a quantized lid step;
   (d) TILE PLANES: within 1 cell of an ownership plane (would contradict the bitwise-weld proof —
       treat as a real bug with highest priority if ANY edge lands here);
   (e) SPLIT LEFTOVERS: vertices touched by _nonmanifold_edge_split (log its rewrites) — the
       deferred-pass fix landed, but multi-bad-edge triangles degrade to downstream repairs.
3. Histogram the classes. Expect concentration; the fix follows the winner:
   (a) → the zone-A gate-cap spec (in flight from the Plan agent) likely CURES these as a side
       effect — sequence zone-A first, re-probe after;
   (b) → dilate the exact_ok=False mask by 1 cell so the snap-mode seam sits fully inside the
       blend band (one-line change, re-prove bitwise welds);
   (c) → smooth/pad the dilated lid by +1 sample where it steps (maximum_filter footprint +1);
   (d) → bitwise repro at the offending plane (tile=48 vs 192 on a cropped window), fix the
       ownership arithmetic;
   (e) → iterate the split pass to fixpoint (the deferred loop already recomputes; raise the pass
       cap if it's bailing early) or hand the leftovers to a targeted stitch (collapse the
       incidence-1 edge pairs by weld radius res/10).

## acceptance

- Assieme under SN: manifold census clean (m3d_empty True, 0 boundary / 0 non-manifold), geometry
  visually unchanged elsewhere (render pair at the former residue sites).
- Full fast suite green under RIBBING_EXTRACTOR=surface_nets (stitch file reports its SN numbers).
- THEN re-open the SN-default decision: remaining blocker is cost alone (1.71x assieme; wrap
  1.51x post-levers). Note the staged-scatter fix (round 8) may have moved SN's assieme number —
  re-measure before judging.

## the by-construction fix for the ambiguity classes: Manifold Dual Contouring (user-proposed)

Schaefer, Ju, Warren 2007 ("Manifold dual contouring") replaces one-vertex-per-cell with one
vertex per SURFACE COMPONENT within a cell, plus quad wiring that references, in each of the 4
cells around a sign-changing edge, the vertex instance whose component owns that edge's crossing.
This eliminates BOTH classic dual pathologies by construction (ambiguous cells → pinches;
ambiguous faces → 4-quad edges) and retires `_repair_pinches` + `_nonmanifold_edge_split` as
no-ops on the SN path. libfive's "watertight, manifold" mesher uses the same mechanism — the
study target and this proposal converge.

Composition with the 2604.00157 placement (if the crease-cell borrow lands): topology and
placement are orthogonal layers — MDC wiring + sample-consistent placement stack, with two
adaptations: (1) sample assignment becomes per-COMPONENT (nearest component's crossings);
(2) DROP their vertex-escape and keep our cell clamp — MDC's manifoldness is combinatorial, so
escaped vertices still self-intersect geometrically (manifold ≠ embedded), and escape only buys
sharp features our fillet floors deliberately removed.

Vectorization sketch (tile-world, no per-cell python): precomputed 256-entry corner-sign table →
per-edge component labels; vertices keyed by (cell, component); init/relax/exact-snap lift per
component; L-step neighbor table needs cross-face component adjacency (also table-derivable);
quad emission picks the component-labeled vertex per adjacent cell. Integer ownership and the
bitwise-weld argument carry over unchanged ((cell, component) ids are deterministic).

SCOPE HONESTY: MDC addresses the 3 non-manifold edges and the repair-surgery debt — NOT the 66
boundary edges (holes = missing quads: dual extraction is closed by construction unless a cell
lost its vertex or a quad was dropped → suspect band-rim truncation or historical surgery, per
the classification above). The diagnostic still gates; MDC is the designated fix for classes
(e)/ambiguity and a prerequisite-quality step for making SN the default.

## sequencing note

Runs AFTER: (1) the in-flight fix-round verification completes (done for fast+assieme; slow suite
pending), (2) the zone-A gate-cap round lands (class (a) is plausibly dominant — the visual
agent's "combed crest" residual and the user's shard zone both sit at gate cliffs). MDC
implementation follows the diagnostic's classification, alongside the libfive ambiguity study.
