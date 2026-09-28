# Implementation Specs: Zone A (facing-gate shards) and Zone B (cap-to-body fade)

**Target file:** `server/geometry/implicit.py`
**Line refs are from the round-8 shipped state read on 2026-08-27** (`_CAP_RUNOUT` at L35, `_rib_field` at L230-369, facing gate at L348, root blend at L327-333). A fix round (crease-blend welt, band-lid, orphan repair) edited this file after the read; **every edit below is anchored to exact code text, not line numbers** — search for the anchor strings before editing. If an anchor is missing or duplicated, stop and re-read the region.

**Iron contracts both specs obey** (from the project memory):
- `max(f, g)` gates are BINARY at g's zero set. There is no soft fade band. Never build attenuation as a max() value ramp.
- Feature floors track the MARCHING res (`res`), never the raster `cell`.
- Sub-voxel free rib-top sheets ("feathers") alias into torn lace. The only legal path for a rib surface to reach the body is the root fillet's body blend (solid-backed, cannot tear).
- Raw interpolated `nzc` rings at facet pitch; smoothed rasters (`NZr`) exist for anything that must be spatially stable.
- No Newton mesh-normal reprojection polish.
- Smooth-min mix order is fixed: `h` from `(shell - prism)/k`, `f = prism*h + shell*(1-h) - k*h*(1-h)`. Swapping deletes every filleted rib.

---

## ZONE A — capped termination at the facing gate

### Approach

The facing gate `f = max(f, 25*(0.09 - nzc))` (anchor: `np.float32(25.0) * (np.float32(0.09) - nzc)`) is binary at raw `nzc = 0.09`: ribs stand at full height right up to a 3D cut surface that wanders with the facet-pitch wobble of the interpolated pseudo-normal (±~0.045 in nz at 84°, since substrate ang_defl 0.09 rad gives ~5° dihedrals), slicing ribs mid-body and leaving partial-height tatters and gate-orphaned crown fragments in the 81-85° band. The fix reuses the round-8 cap machinery verbatim: build a **signed 2D raster `Bg` of exact projected distance to the smoothed-normal contour `{NZr = 0.15}`** (positive on the ribbable side), and add a second hypot trim term to the wall SDF so every rib **ends in the same in-plane rounded nose the taper cap produces**, completing strictly before the smoothed 0.15 contour — so the raw binary gate at 0.09 (kept unchanged as the true back-face backstop) can never touch surviving material. This makes the code finally match its own round-4 comment ("full material by nz~0.15, gone below 0.09"), which the binary gate never actually delivered.

**Design decisions resolved:**
1. *"Effective gate height attenuation from nzc":* deliberately **no height ramp**. A height ramp keyed on nzc (even smoothed) still needs a termination station, and without one it degenerates to a plateau-stub running into the binary cut — the defect class the caps round eliminated. The 81-85° band is projection-compressed to sub-mm anyway. Attenuation is expressed *only* as the spatial trim: full height → rounded capped end.
2. *Trim station source:* the **raster distance-to-contour proxy**, not a local nzc formulation. A purely local form keyed on `nzc`/`nzs` cannot produce an in-plane hypot nose (no distance coordinate; converting the nz deficit through an assumed |grad nz| slope is wrong on any curvature). The raster reuses `_exact_pattern_distance` so it inherits the anti-scallop discipline — a pixel EDT of the gate boolean is forbidden (half-cell scallops stretch maximally on exactly these slopes).
3. *Wobble:* the contour is extracted from the gaussian-smoothed `NZr` (σ ≈ 1.5 mm) → facet-pitch stable; the raw-nzc backstop stays raw (bridges depend on it).
4. *Solid webs (nTop-consistent, must survive):* webs live where the projected pattern period compresses below rib thickness (up to ~75°). The trim is a uniform spatial station, not a degeneracy classifier: web material above the 0.15 contour is untouched; only its terminal 81-85° fringe becomes a clean capped edge.
5. *Slab:* the trim applies to the rib **wall only**. The sub-surface slab keeps its current extent down to the 0.09 backstop (cluster connectivity + body coverage preserved).

### Algorithm

Raster stage (once per apply, in `build_rib_implicit`):
1. After `NZr` is normalized (anchor: `NZr = (NZr / nl).astype(np.float32)`): if `_GATE_CAP` and `float(NZr.min()) < _GATE_HI` (0.15), extract contour chains `skimage.measure.find_contours(NZr, _GATE_HI)`; convert index coords to mm (`origin + coords*cell`; find_contours returns (row=u_idx, col=v_idx), matching `[u,v]` raster indexing). Optionally drop closed chains enclosing area < `(3*cell)**2` (specks). Optional refinement (only if curved-rim cap-line scallops are observed): `_smooth_rings` on closed chains — normally unnecessary.
2. `d = _exact_pattern_distance(chains, cell, origin, shape2d, reach=2.5)` (unsigned; 1e3 beyond reach). `Bg = np.where(NZr >= _GATE_HI, d, -d).astype(np.float32)`. The 1e3 saturation is self-correct on both sides. If no chains: `Bg = None` (zero cost, exact no-op).
3. Store on `SurfaceField` as a new defaulted field appended at the END of the dataclass (`Bg: np.ndarray = None`) — constructors use keyword args, appending is backward-compatible.

Sampling stage (`mesh_field`):
4. `_sample_at_foot` additionally returns `bg = _bilinear(g.Bg, cu, cv, ...) if g.Bg is not None else None` (6-tuple; anchor: `return P, B, Bo, obv, stretch`). Update BOTH call sites: `_core` and the crease-blend partner branch — the partner branch trims on its own sheet's foot value, exactly as it does for `P2/B2/Bo2/ob2/st2`.

Field stage (`_rib_field`, new parameter `bg=None`):
5. Hoist `nz_c = np.maximum(nzc, np.float32(0.2))` out of the taper branch (same value; taper branch reuses it).
6. After the existing wall/trim block (anchor: `wall = np.hypot(lat, m) - half`), add the gate trim with clearance `q_g = np.float32(max(res, 0.3))` (projected mm past the contour; the nose forward semi-axis `half*nz_c/_CAP_STEEP ≤ ~0.08 mm` projected always fits inside it, so the binary backstop can never slice the tip):
   - `mg = clip(q_g - bg, 0, None) * (_CAP_STEEP / nz_c)`
   - taper-only path unchanged (bit-identical when `bg is None`); when both trims active, compose as nested hypot `hypot(hypot(lat, m), mg) - half` = the exact SDF of the doubly-trimmed centerline with a rounded corner where an open rim meets a steep band (the desired molded look).
7. Crown rounding, embed clamp, B setback, root smooth-min, slab caps, facing gate 0.09, OB cut, Bv lean cap remain verbatim. The gate trim applies regardless of `taper_len`/`border`.

### Integration anchors

| Change | Anchor (search string) |
|---|---|
| Constants `_GATE_CAP = True`, `_GATE_HI = 0.15` | below `_CAP_STEEP = 3.0` block |
| `SurfaceField.Bg` field (append last, default None) | `M: np.ndarray = None` |
| `_rib_field(..., bg=None)` signature | `def _rib_field(s, P, B, inside, Bo, params, cell, nzc, ob=None, res=None,` |
| Hoist `nz_c` | `nz_c = np.maximum(nzc, np.float32(0.2))` |
| Gate trim | `wall = np.hypot(lat, m) - half` |
| Foot sampling | `return P, B, Bo, obv, stretch` |
| `_core` plumb-through | `vals = _rib_field(s, P, B, ins, Bo, params, g.cell, nzc,` |
| Crease-partner plumb-through | `f2 = _rib_field(s2.astype(np.float32), P2, B2, ins2, Bo2,` |
| Raster build | `NZr = (NZr / nl).astype(np.float32)` |
| SurfaceField ctor `Bg=Bg` | `surface = SurfaceField(cell=cell, origin=origin, ...` |

The welt/band-lid/orphan fixes may have moved these regions — re-read `_sample_at_foot` and the crease-partner `_rib_field` call immediately before editing and rebase the 6-tuple change onto whatever the welt fix left there.

### Pseudocode

```python
# --- build_rib_implicit, after NZr normalization ---
Bg = None
if _GATE_CAP and float(NZr.min()) < _GATE_HI:
    from skimage import measure                      # local import, codebase style
    chains = []
    for c in measure.find_contours(NZr, _GATE_HI):   # (row,col) = (u_idx, v_idx)
        pts = c * cell + np.asarray(origin)          # -> (u, v) mm
        if len(pts) >= 3:
            chains.append(pts)
    if chains:
        d = _exact_pattern_distance(chains, cell, origin, shape2d, reach=2.5)
        Bg = np.where(NZr >= np.float32(_GATE_HI), d, -d).astype(np.float32)

# --- _rib_field ---
nz_c = np.maximum(nzc, np.float32(0.2))              # hoisted; taper branch reuses
...                                                  # taper block unchanged (q_end)
lat  = P * stretch
wall = lat - half
if q_end is not None:
    m = np.clip(q_end - Bo, np.float32(0.0), None) * (np.float32(_CAP_STEEP) / nz_c)
    wall = np.hypot(lat, m) - half                   # bit-identical taper-only path
if bg is not None:
    # in-plane gate cap: same trimmed-centerline SDF, station = the smoothed
    # NZr=0.15 contour + q_g clearance; the raw 0.09 backstop below stays a
    # pure back-face guard and can no longer slice standing ribs
    q_g = np.float32(max(res, 0.3))
    mg = np.clip(q_g - bg, np.float32(0.0), None) * (np.float32(_CAP_STEEP) / nz_c)
    wall = np.hypot(wall + half, mg) - half          # nested hypot; = hypot(lat,m,mg)-half
# crown / embed / B setback / root blend / slab / gate(0.09) / OB / Bv: verbatim
```
(`wall + half` recovers `hypot(lat, m)` or `lat` without a second temporary; both operands are nonnegative.)

### Reproducing metric (new test, `tests/test_gate_termination.py`)

Fixture: `wrap_cyl_step` (260° cylinder, R=30, h=50 — natural 81-85° band), `biggest_face_id(kind="cylinder")`, stitch-test params (isogrid, spacing 12, thickness 1.6, height 3, taper_len 0, mapping="project", fillets 0). Engine res 0.30. Rebuild the substrate with the ENGINE's tessellation (`region_meshes(s, [fid], 0.25, ang_defl=0.09)`, merge, `_projection_frame`) and classify output vertices by `igl.point_mesh_squared_distance` + barycentric **interpolated pseudo-normal** (idiom in `test_implicit_engine.py::test_steep_wall_gets_ribs`). Bilinear-sample the smoothed normal at each vertex's foot for `nzs_foot` (reconstruct `NZr` the way `build_rib_implicit` does, or monkeypatch-capture the SurfaceField).

Assertions (runout-test structure):
- **(a) band emptiness:** < max(20, 0.001·len(v)) off-surface vertices (`sqrt(sqrD) > 0.4`) with `nzs_foot < 0.13`. Today: fails by hundreds-to-thousands. After: material ends at `nzs ≥ ~0.155`; margin 0.025 covers bilinear/query noise. Do NOT assert raw interpolated nz tighter than 0.095 — post-fix raw values can dip to ~0.105 via facet wobble; print raw-nz min as diagnostic only.
- **(b) shard counter:** face-subgraph of faces whose three vertices all have `h > 0.4` and `nzs_foot < 0.18`; per connected component compute max h. Assert zero components with `max h < 0.7*height` (partial-height tatter) — full ribs/webs crossing the band carry max h ≈ height and pass. Anti-vacuity guard: the ≥55° region must still hold rib material (> 100 off-surface verts at interpolated nz ∈ [0.2, 0.5]) — web/steep-rib preservation in executable form.
- **(c) debris + watertight:** transplant `test_runout_quality.py::test_runout_band_has_no_debris` verbatim, band = components touching `nzs_foot < 0.2`: every component ≥ 10 voxels volume and ≥ 24 faces; every cluster manifold.

Calibration: run against the current engine first and record measured failure numbers in the docstring.

### Risks (test-keyed)

- `test_stitch_quality.py` (wrap_cyl, Zone A active): crest runs slope-gated 47-65° (nz 0.42-0.68); trim on R=30 removes material only above ~78°. No run, control set, run count, crest-radius assertion, or honesty guard touched. Verify by running the file.
- `test_no_ribs_on_foldover_backside` (wrap_cyl): trim strictly removes near-silhouette material → margin improves.
- `test_steep_wall_gets_ribs` (cyl_patch, max 60°, NZr ≥ ~0.5): no contour → Bg=None → byte-identical. Same for box parity, fillet-root volume, boundary-inset, curved-watertight, wall-smoothness, groove/sphere crown, sn_repairs, api routing.
- Slow `test_band_acceptance` / `test_no_floating_material` (cruscotto): real silhouettes → trims fire; watertight/1-cluster hold (slab persists, wall-only trim). Run the slow suite once before calling it done.
- Kernel/runout/orphan/braid tests: construct SurfaceField with kwargs, no Bg → default None → byte-identical.
- Sealed-slit bridges: bridge cps pin to front-facing strip edges (raw backstop inert, unchanged); feet land on strip columns with NZr ≈ 1 → Bg large positive → no trim.
- Behavior change to accept: coverage on tight rolls now ends ~78-81° instead of raw 85° — matches the round-4 documented intent; the flag makes A/B one flip.
- Multi-sheet folds: Bg (like NZr) is single-valued per column; a front sheet over another sheet's steep band could inherit a wrong trim (same compromise stretch already accepts). Escape = per-sheet classification, out of scope.

### Perf
One find_contours + one `_exact_pattern_distance` at reach 2.5 (same class as the OB raster) — sub-second; +1 float32 raster. Field: one extra bilinear + ~5 ops/sample only when Bg exists; < ~2-3% on steep-band parts, 0% elsewhere. Mesh size shrinks (tatter micro-triangles disappear).

### Fallback
`_GATE_CAP = True` next to `_CAP_RUNOUT`. False → Bg=None everywhere → bit-identical. Tunables in order: `q_g` clearance, `_GATE_HI` (0.15; never below 0.09 + observed wobble), speck-area prefilter.

---

## ZONE B — seamless cap-to-body fade (end gusset via ramped root blend)

### Approach

Today's cap ends the run-out as: taper ramp → plateau at `h_plat = min(4.4*res, 0.55*height)` (~1.3 mm interactive) → 3x-steep hypot nose → whatever `fillet_root` the user set. With fillet_root 0 (or small vs h_plat) the nose is a raw ~1.3 mm cliff standing proud of the body — the "side bumps". The melt cannot come from ramping the top surface to zero (sub-voxel feather = torn lace; the stub-band contract forbids it): the legal path to zero is the root fillet's body blend, solid-backed. So: **locally ramp the root smooth-min k from `fillet_root` up to `k_end ≈ 1.2*h_plat` over the last ~one thickness of surface run before the trim station, and hold it beyond.** When the blend radius reaches the plateau height, the fillet arc departs directly from the crown roll: plateau-crest → crown radius → fillet radius → tangent into the body — one continuous rounded descent (the injection-molding "end gusset"), no free wall segment left to read as a shoulder. Where the user's fillet_root already exceeds k_end, the field is untouched (max composition) — which keeps the entire runout-quality module (fillets 2/2) byte-identical.

**Candidates:**
- **(i) local k-ramp — CHOSEN.** Body junction is the smooth-min's tangent (G1) approach to shell; along-rib onset made kink-free by smootherstep (a C0 kink in k(x) prints a faint crease). No new top-surface features below the r_top floor → no feather risk by construction.
- **(ii) plateau tapered into a root dome — REJECTED.** height→0 under smin leaves a free pad of height ~k/4 riding to the trim (smin(s, s) = s − k/4): sub-resolvable unless k is already ramped, i.e. needs (i) anyway, plus re-adds the "descending top re-enters the stub window" failure the round-8 sweep documented.
- **(iii) crown dropped to r_top at the end — REJECTED.** End crest ≈ 1.2*res free height; crown roll + root skirt land the crest in the sub-resolvable band — the verified one-floor failure mode that forced the two-floor plateau.

### Algorithm

All inside `_rib_field`, active only when `_CAP_GUSSET and q_end is not None and k_end > k_base`:
1. Hoist the plateau: `h_plat = min(4.4*res, 0.55*params.height)` (anchor: `np.float32(min(4.4 * res, 0.55 * params.height))` — coordinate with the shipped two-floor expression, do not duplicate the constant).
2. `k_base = float(max(params.fillet_root, 0.0))`. `k_end = float(min(1.2*h_plat, 0.6*params.height, 2.0))` — ≥ h_plat so the fillet reaches the crown roll; ≤ 2.0 so the lateral skirt stays inside the corridor reach margin (`+ 2.0`) and band pads.
3. Ramp coordinate: surface-metric length `L_k = max(params.thickness, 3.0*res)`, projected `L_p = L_k * nz_c`. `t = clip((q_end + L_p - Bo)/L_p, 0, 1)` (0 interior, 1 at/beyond the trim; beyond, k holds at k_end and the skirt wraps the nose ring; the skirt self-terminates within ~k because the trimmed prism grows at 3x there). Smootherstep: `t = t*t*(3 - 2*t)`.
4. `k_arr = maximum(float32(k_base), float32(k_end)*t)`.
5. Blend with array k, epsilon-guarded divisor: `kk = maximum(k_arr, 1e-6)`; `h = clip(0.5 + 0.5*(shell - prism)/kk, 0, 1)`; `f = prism*h + shell*(1-h) - k_arr*h*(1-h)`. Limits exact: k_arr→0 gives min(prism, shell) in value; k_arr ≡ k_base gives today's formula. Mix order is the memorized one — do not touch.
6. Everything after the blend unchanged. B-setback truncates the skirt at open rims exactly as it truncates the slab — flush, consistent.

Scope: v1 applies the gusset to the **taper** trim only. Zone A gate noses keep plain fillet_root (mid-wall ends at 78-81°; a full-height gusset there would be a monster). If screenshots later ask, the k_arr machinery accepts a second ramp keyed on bg with k_end clipped to ~thickness (separate flag then).

### Integration anchors

| Change | Anchor |
|---|---|
| `_CAP_GUSSET = True` | below `_CAP_STEEP` block |
| Hoist `h_plat` | `height, np.float32(min(4.4 * res, 0.55 * params.height)))` |
| Gusset ramp + array blend | `k = float(max(params.fillet_root, 0.0))` through `f = prism * h + shell * (1 - h) - k * h * (1 - h)` |

Same function the fix round touched — re-read the taper and blend blocks immediately before editing; if cap constants changed, inherit the shipped values (specs are parameterized on them, not on 3.0/4.4).

### Pseudocode

```python
# in the taper block (q_end computed), reuse hoisted h_plat and nz_c
shell = s + _SINK
k_base = float(max(params.fillet_root, 0.0))
k_end = float(min(1.2 * h_plat, 0.6 * params.height, 2.0)) \
    if (_CAP_GUSSET and q_end is not None) else 0.0
if k_end > k_base + 1e-6:
    # end gusset: the root blend radius swells to swallow the plateau-high
    # nose over the last ~thickness of surface run, and holds past the trim
    # so the skirt ring melts the nose into the body.  The top surface never
    # fades (feather-proof); the fillet surface is the path to zero.
    L_p = np.float32(max(params.thickness, 3.0 * res)) * nz_c   # projected span
    t = np.clip((q_end + L_p - Bo) / L_p, np.float32(0.0), np.float32(1.0))
    t = t * t * (np.float32(3.0) - np.float32(2.0) * t)         # C1 onset
    k_arr = np.maximum(np.float32(k_base), np.float32(k_end) * t)
    kk = np.maximum(k_arr, np.float32(1e-6))
    hm = np.clip(0.5 + 0.5 * (shell - prism) / kk, 0.0, 1.0)
    f = prism * hm + shell * (1.0 - hm) - k_arr * hm * (1.0 - hm)
elif k_base > 0:
    ...                                   # today's scalar smooth-min, verbatim
else:
    f = np.minimum(prism, shell)          # today's hard min, verbatim
```

### Reproducing metric (new test, `tests/test_cap_fade.py`)

Fixture: runout-quality harness verbatim (flat plane z=10, one rib along u at v=15, B = Bo = u ramp, CELL 0.25, RES 0.30) but **fillets 0/0** — thickness 1.6, height 4, taper 5. Derived: FLOOR 0.66, plateau 1.32, q_end 0.867, B-setback at u = 0.3.
- **(m1) shoulder height — primary:** `_profile` idiom (|v−15| < 0.4, 0.3 mm bins, h > 0.2); S = h_top of the terminal occupied bin. Today S ≈ plateau ≈ 1.32 (FAIL). Assert `S < 0.55*plateau` (≈0.73): the gusset skirt fills bins down to the B-setback with h_top ≈ 0.3-0.45 (PASS). Also assert material EXISTS in bin [0.3, 0.6) (today: empty — the melt must bridge nose to body).
- **(m2) junction slope discontinuity — diagnostic print**, optional assertion after calibration: p95 dihedral across seam-zone mesh edges (h < 0.6, u ∈ [0.2, 1.2]); today ≳ 60-90°, gusset ≲ 40° (Taubin/MC noise — calibrate before asserting).
- **(m3) feather honesty guard:** for every upward-facing (nz > 0.5) vertex with h ∈ [0.2, 0.9], assert no downward-facing (nz < −0.5) vertex within 0.45 mm (a free sheet has two nearby sides; a fillet skirt is backed by slab). Must be 0 both today and after — pins the mechanism and guards future edits.
- **(m4)** debris + watertight, transplanted from `test_runout_band_has_no_debris`.
- Do NOT copy the stub-window fraction test into the fillets-0 fixture: the gusset skirt legitimately sweeps upward-facing material through [0.3, 1.35]*FLOOR near the nose — the root fillet's body blend is the explicitly legal path. The window contract stays guarded by the untouched fillets-2/2 module.

Calibration: record S_before (≈1.32) and S_after in the docstring.

### Risks (test-keyed)

- `tests/test_runout_quality.py` (fillets 2/2): k_end = min(1.58, 2.4, 2.0) = 1.58 < fillet_root 2 → guard keeps the scalar path → **byte-identical**, including the stub-window fraction. This bit-identity is the reason for the guard's exact form; do not "simplify" into an always-array path.
- `test_taper_follows_boundary_ramp` (fillets 0, taper 8, RES 0.25 → plateau 1.1, k_end 1.32, L_p 2.0): gusset tops out ≈ 1.32 ≤ window allowance 1.85 in u ∈ [1,3); other windows and 14.0±0.3 untouched (t=0 beyond Bo ≈ 3.1). Verify.
- `test_taper_runout_is_surface_length_on_slope`: asserts at foot-u 4.3-5.3; gusset active only Bo < ~1.8 → inert.
- `test_retaining_rim_rib_survives_setback_and_taper`: Bo = 1e6 → t=0 → k_arr ≡ 0 → value-identical min path.
- All taper_len=0 tests: q_end None → scalar path, byte-identical.
- Slow `test_band_acceptance` (taper 10, height 1.5 → plateau 0.825, k_end 0.9): plain build gains gusset volume (~10² mm³); filleted build (root 1.2 > 0.9) unchanged and gains root fillets (~10³ mm³) → `filleted > plain` holds. Run the slow pair once.
- `test_api::test_engine_routing` (implicit with DEFAULT taper_len 5, fillets 0 → gusset fires on box rims): asserts only status + engine name → safe. Audit any future implicit test without taper_len=0 — the 5.0 default silently activates this path.
- Attachment/reach invariants: gusset material lies between rib and body within height of the surface → inside reach. Blend bulge ≤ plateau + k/4 < height (inside highpad); lateral skirt ≤ k_end ≤ 2.0 (inside the corridor `+2.0` — the reason for the clip).
- Crease-blend interaction: k-ramp is per-branch (partner call computes its own k_arr from Bo2) — symmetric; the crease smin k_c is a different constant, untouched.

### Perf
Scalar-path guard → zero cost for taper-off and filleted builds. Gusset path: ~6 element-wise ops + one array divisor, taper-branch samples only → end-to-end noise (< 1%).

### Fallback
`_CAP_GUSSET = True` next to `_CAP_RUNOUT`/`_GATE_CAP`. False → scalar path → bit-identical. Tunables in order: k_end factor (1.2 → 1.0 if gussets read fat on thin ribs; never below 1.0*h_plat or the shoulder returns), ramp length L_k (thickness → 1.5*thickness for softer swell), smootherstep (keep; removing it prints the onset crease).

---

### Sequencing and shared notes
1. Land Zone B first (smaller, single-function, no raster/schema change), with its test; then Zone A (schema + raster + plumbing), with its test. Both flags default ON, each independently revertible to a bit-identical engine.
2. Both live inside `_rib_field`/`mesh_field` where the fix round was active: before editing, re-read `_rib_field`, `_sample_at_foot`, `_core`, `_crease_blend`, and the NZr/SurfaceField block, and re-anchor. Inherit shipped constants.
3. After landing: hot-reload discipline before UI verification; `GET /api/dev/last_recipe` is the first move on any "still broken" report.
