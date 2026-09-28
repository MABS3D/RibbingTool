# SPEC: Field-driven parameter modulation — nTop parity headline (author: orchestrator, from LEAP71 SurfaceModulation study)

## approach

nTop's rib pitch: "simulation data and other spatially varying fields control rib height, thickness,
orientation, draft." LEAP71's implementation idiom (read from source): `SurfaceModulation` = one
normalized 2D function per parameter, constructed from a constant | a 2D lambda | AN IMAGE
(grayscale → value mapping) | a 1D LineModulation projected along an axis, composable with +, −,
scalar·. Our engine is already architected for this and doesn't know it: every field sample reads
2D rasters (P/B/D/OB/Bv) at the projection-frame foot — a modulation is just TWO MORE RASTERS.
Scope v1: HEIGHT and THICKNESS modulation (nTop's two lead knobs). Default absent → byte-identical.

## algorithm

1. `RibParams.modulation: dict | None = None`, shape:
   `{"height": SPEC, "thickness": SPEC}` (either key optional), where SPEC is one of
   - `{"type": "linear", "deg": 30, "lo": 0.5, "hi": 1.0}` — normalized gradient across the
     selection bbox along direction `deg` in the frame plane;
   - `{"type": "radial", "center": [u,v] | "auto", "r0": 0, "r1": 80, "lo": 1.0, "hi": 0.4}`;
   - `{"type": "boundary", "d0": 0, "d1": 15, "lo": 0.4, "hi": 1.0}` — ramp on distance from the
     domain boundary, REUSES the existing B raster (zero new geometry queries; this is the
     stress-concentration preset: tall ribs mid-panel, low near rims);
   - `{"type": "image", "path": "...png", "lo": 0.5, "hi": 1.5}` — grayscale bilinear over the
     selection bbox (the LEAP71 image idiom verbatim; a user can paint a stiffness map in any
     editor).
2. `build_rib_implicit` materializes `Mh`, `Mt` float32 rasters on the SAME grid as P (cell,
   origin, shape2d) — trivially composable later (types sum/multiply like LEAP71 operators; v1
   ships single-source per param).
3. Sampling: in `_sample_at_foot`, add two bilinear reads → per-sample `mh`, `mt` in [lo, hi].
4. Kernel consumption (`_rib_field` — already per-sample-array capable for height via taper):
   - `H0 = params.height * mh` (array). Taper/cap machinery already treats height per-sample;
     `q_end`'s `t_eff * h_end / H0` term becomes per-sample (it already broadcasts).
   - `half = (params.thickness / 2) * mt` (+ existing draft term). The projected-width floor
     stands: clamp `half >= 2 * cell * stretch_guard` (the round-6 lesson — sub-cell walls ring);
     clamp `mh` so `H0 >= max(2.2*res, ...)` floor family stays satisfiable.
5. KEY SYNERGY (why this is safe NOW and wasn't before round 8): where a modulation drives height
   toward its floor, the CAP machinery terminates the rib with the plateau+nose instead of a
   feather — modulation-to-thin produces capped ends by construction. Before the caps landed this
   feature would have shipped torn lace at every modulation low.

## integration

- server/geometry/patterns.py: RibParams field + validation (reject unknown type, clamp lo/hi to
  [0.1, 3.0]).
- build_rib_implicit: raster build (~30 lines incl. image path via PIL/imageio — PIL ships with
  the venv? verify; else `imageio`); pass Mh/Mt into SurfaceField (two optional fields, default
  None — same pattern as OB/Bv).
- mesh_field `_sample_at_foot`: two `_bilinear` reads when rasters present.
- `_rib_field`: accept per-sample H0/half arrays (audit every use of params.height/thickness in
  the kernel body — the caps block, floors, r_top cap `0.35*thickness`, slab embed — and decide
  scalar-vs-array per site; r_top cap and slab depth stay SCALAR (crown rounding and slab are
  global properties; varying them per-sample invites new artifact classes — document this).
- UI: "Modulation" section, two dropdowns (None/Linear/Radial/Boundary/Image) + inline params;
  recipes serialize the dict (params dict already round-trips through /api/dev/last_recipe).

## tests (feature-TDD)

- Kernel `test_height_modulation_tracks_map`: linear 0.5→1.0 across the fixture, height 4 →
  crowns must measure ~2.0mm at the lo edge, ~4.0mm at the hi edge, monotone between (foot-window
  the measurement, round-5 lesson). Fails before (no modulation plumbing), passes after.
- Kernel `test_thickness_modulation_respects_floor`: mt driving half below the projected floor
  must clamp (assert min measured width == floor width, not below) — the guard test.
- Byte-identity: modulation=None build hash == pre-change hash.

## risks / perf

- +2 bilinear reads per sample when active (0 when None): ~1-2% apply.
- Image type: resolution mismatch is fine (bilinear), but sRGB gamma — decode as luminance,
  document "linear grayscale" and move on.
- Floors interacting with lo values: clamps make the map lie locally (measured height ≠ map at
  the floor) — INTENDED; the test encodes it.
- Modulated H0 in the stitch/runout quality tests: untouched (they don't pass modulation).

## fallback

`modulation=None` default; the dict is additive and versioned by key presence. Ship the plumbing +
linear/boundary first; image type can trail by a round if PIL/imageio availability stalls.
