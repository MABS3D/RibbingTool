"""Braided-rib defect (user screenshots, defect 2): field discontinuity
across a concave channel's medial plane — asserted on the OUTCOME.

The surface-field engine samples the 2D pattern rasters at the foot
x - s*n(cp) of each evaluation point (implicit.mesh_field).  Inside a
concave recessed channel the closest-point map is DISCONTINUOUS:
crossing the channel's medial plane, cp flips from one wall sheet to the
opposing one, so the foot (u, v) jumps by ~the sheet separation.  Every
rib whose pattern gradient has a component along that jump gets its wall
cut laterally dislocated mid-channel — and because the medial sheet cuts
OBLIQUELY through the marching grid, adjacent grid columns along the
seam flip sheet assignment at voxel pitch: the extracted walls shred
into the braided / ropey strands the user sees crossing the recessed
channel of the dashboard panel.

RE-TARGET RATIONALE (2026-08-25, sanctioned): the original primary
assertions here measured _surface_eval internals — closest-point foot
continuity (jump 3.35mm vs 0.4mm bound) and the s directional
derivative (step 1.58 vs 0.6).  The shipped fix, the medial-crease
two-branch blend (RIBTOOL_CREASE_BLEND), deliberately does NOT fake a
continuous closest-point map: near a channel's medial sheet there are
genuinely TWO walls, and a continuous single foot could only be
manufactured by inventing a fictitious surface between them, moving the
zero set by O(|f_left - f_right|) ~ millimeters and violating the 0.1mm
thickness budget.  The geometrically correct outcome is a MITER seam:
each wall's rib family stays exact on its own side and the COMPOSED
field hands over at the medial sheet.  So the primary assertions now
measure the marched grid field itself — captured from the very array
mesh_field hands the extractor — along the SEAM-PARALLEL grid axes,
where the braid actually lives (voxel-pitch sheet-assignment flips).
The seam-NORMAL axis is excluded by design and printed as a diagnostic:
the blend's distance-excess penalty (beta=8) intentionally compresses
the miter handover into ~1 cell (a hard miter crease, as a real two-
prism miter joint has), which is a stationary crease surface, not the
voxel-pitch inconsistency of the defect.  The foot-jump and gradient
numbers remain as printed diagnostics (they explain WHY the raw map
cannot be sampled naively); the C0-of-s check is unchanged (true before
and after the fix).  Honesty guard: test_braid_metric_detects_raw_map
rebuilds the same fixture with the blend disabled and asserts the
metric FAILS by a wide margin — the primary assertion cannot pass
vacuously, and running this file with RIBTOOL_CREASE_BLEND=0 makes the
primary test fail.

Verified numbers (this file):
  raw map (blend off): seam-parallel per-step field jumps 8.3 / 8.9
  (v / d axis, in units of res) — the full |f_left - f_right| sheet
  flip.  Blend on: 2.97 / 1.32.  Legitimate motion stack: foot
  advection on these 56deg walls <= ~1.6*res per step (the engine's own
  steep-slide model m), s-term <= 1*res, miter-ramp leak through the
  seam obliquity ~2*res: < 4.6*res combined, and the terms do not
  co-locate.  Threshold 5*res sits above every legitimate stack and
  1.7x under the raw flip.
  Foot diagnostics (unchanged mechanism): max foot jump between 0.1mm
  samples 2.06-3.35mm; s gradient step 1.58; s itself C0 (0.083).
"""
import numpy as np
import pytest

from server.geometry.implicit import (SurfaceField, _surface_eval,
                                      _vertex_normals, mesh_field)
from server.geometry.patterns import RibParams


# --- geometry: V-channel from two planar sheets meeting at a concave
# dihedral (valley along v at u = 20), flat flanks beyond the rim ---
SLOPE = 1.5          # wall slope dz/du: 56.3 deg walls, 33.7 deg from
                     # vertical -> cp facing nz = 0.5547 (the engine's
                     # facing gate keeps full material above nz ~ 0.15,
                     # so these walls are ribbed on the real part)
VALLEY_Z = 5.0
WALL_H = 4.5         # rim at z = 9.5, half-width 3 -> ~6 mm wide channel
DX = 0.1             # sweep sample spacing (mm)

# sweep lines: constant-z traverses of the channel crossing the medial
# plane u = 20.  z0 = 7.2 / 7.9 / 8.6 put the medial-plane sample at
# s = 1.22 / 1.60 / 1.99 mm above the sheets — the band ribs live in.
SWEEP_Z = (7.2, 7.9, 8.6)
SWEEP_V = (8.0, 15.0, 22.0)      # away from the open v-boundaries
US = np.arange(19.005, 21.005 + 1e-9, DX)   # offset: no sample sits
                                            # exactly on the medial plane

# s is C0 (min of continuous sheet distances): 1-Lipschitz along the
# sweep, with slack for the hybrid height blend
C0_TOL = 1.5 * DX

# --- marched-grid fixture: the same channel, tilted so the medial
# sheet cuts the grid obliquely (the braid needs the seam to wander
# through grid columns; a grid-aligned seam degenerates to one clean
# crease plane and hides the voxel-pitch alternation) ---
CELL = 0.25
RES = 0.25
TILT = 0.15          # valley line u = 20 - TILT*(v-15)
LIP_TOL = 5.0        # seam-parallel per-step bound, units of RES (see
                     # module docstring for the justification)


@pytest.fixture(scope="module")
def channel():
    """Triangulated V-channel substrate + angle-weighted pseudo-normals."""
    step = 1.0
    xs = np.arange(0.0, 40.0 + 1e-9, step)
    ys = np.arange(0.0, 30.0 + 1e-9, step)
    uu, vv = np.meshgrid(xs, ys, indexing="ij")
    zz = VALLEY_Z + np.minimum(SLOPE * np.abs(uu - 20.0), WALL_H)
    V = np.ascontiguousarray(
        np.column_stack([uu.ravel(), vv.ravel(), zz.ravel()]), np.float64)
    nu, nv = len(xs), len(ys)
    quads = []
    for i in range(nu - 1):
        for j in range(nv - 1):
            a = i * nv + j
            b = (i + 1) * nv + j
            c = (i + 1) * nv + j + 1
            d = i * nv + j + 1
            quads.append((a, b, c))
            quads.append((a, c, d))
    F = np.asarray(quads, np.int64)
    return V, F, _vertex_normals(V, F)


def _sweep(channel, z0, y0):
    """Evaluate the surface field on one line crossing the medial plane.

    Returns s and the raster-sampling foot x - s*n(cp) — exactly what
    mesh_field's field closure feeds to the P/B/mask bilinear lookups.
    """
    V, F, N = channel
    X = np.column_stack([US, np.full_like(US, y0), np.full_like(US, z0)])
    s, C, nrm = _surface_eval(X, V, F, N)
    s = s.astype(np.float64)
    foot = X - s[:, None] * nrm.astype(np.float64)
    # fixture sanity: the whole sweep is in air above the walls, inside
    # the rib band, and every foot lies on a wall sheet interior — the
    # measurements below are about the MAP, not about geometry edges
    assert s.min() > 0.2 and s.max() < 3.0, "sweep left the rib band"
    assert np.all(np.abs(foot[:, 0] - 20.0) < 2.9), "foot off the walls"
    return s, foot


def test_signed_height_is_c0_across_channel(channel):
    # the min-of-sheets distance is continuous — this must hold BEFORE
    # and after any fix, and proves the fixture itself is sound
    for z0 in SWEEP_Z:
        for y0 in SWEEP_V:
            s, _ = _sweep(channel, z0, y0)
            step = np.abs(np.diff(s)).max()
            assert step < C0_TOL, \
                f"s not C0 at z0={z0} y0={y0}: step {step:.3f}mm"


def test_closest_point_map_diagnostics(channel):
    # DIAGNOSTICS (demoted from primary assertions by the sanctioned
    # re-target — see module docstring): the raw closest-point foot
    # jumps by ~the sheet separation and the s directional derivative
    # flips +-sin(56deg) crossing the medial plane.  These numbers are
    # the MECHANISM of the braid; the composed-field fix leaves them in
    # place by design (a miter seam keeps both sheets' true maps), so
    # they are printed, not asserted.  The outcome tests below carry
    # the contract.
    for z0 in SWEEP_Z:
        for y0 in SWEEP_V:
            s, foot = _sweep(channel, z0, y0)
            jump = np.linalg.norm(np.diff(foot, axis=0), axis=1)
            g = np.diff(s) / DX
            print(f"z0={z0} y0={y0}: max foot jump {jump.max():.3f}mm "
                  f"(median step {np.median(jump):.3f}mm), "
                  f"max s-gradient step {np.abs(np.diff(g)).max():.3f}")


# --- marched-grid outcome tests -------------------------------------


def _uc(vv):
    return 20.0 - TILT * (vv - 15.0)


def _tilted_channel_surface():
    """SurfaceField on the tilted V-channel with an oblique rib family
    crossing it (synthetic rasters, test_implicit_kernel conventions)."""
    from scipy.ndimage import gaussian_filter
    step = 1.0
    xs = np.arange(0.0, 40.0 + 1e-9, step)
    ys = np.arange(0.0, 30.0 + 1e-9, step)
    uu, vv = np.meshgrid(xs, ys, indexing="ij")
    zz = VALLEY_Z + np.minimum(SLOPE * np.abs(uu - _uc(vv)), WALL_H)
    V = np.ascontiguousarray(
        np.column_stack([uu.ravel(), vv.ravel(), zz.ravel()]), np.float64)
    nu, nv = len(xs), len(ys)
    quads = []
    for i in range(nu - 1):
        for j in range(nv - 1):
            a = i * nv + j
            b = (i + 1) * nv + j
            c = (i + 1) * nv + j + 1
            d = i * nv + j + 1
            quads.append((a, b, c))
            quads.append((a, c, d))
    F = np.asarray(quads, np.int64)
    N = _vertex_normals(V, F)
    ru = int(40.0 / CELL) + 1
    rv = int(30.0 / CELL) + 1
    gu, gv = np.meshgrid(np.arange(ru) * CELL, np.arange(rv) * CELL,
                         indexing="ij")
    # diagonal rib family: every rib crosses the channel obliquely, so
    # its wall cut has a pattern-gradient component along the foot jump
    P = np.minimum.reduce([np.abs((gu - 20.0) - (gv - 15.0) - c)
                           for c in (-8.0, 0.0, 8.0)]) / np.sqrt(2.0)
    P = P.astype(np.float32)
    B = np.full((ru, rv), 1e6, np.float32)
    Pgu, Pgv = np.gradient(P, CELL)
    # analytic channel normal per raster column, engine-style smoothing
    dd = gu - _uc(gv)
    sgn = np.where(np.abs(dd) < 3.0, np.sign(dd), 0.0)
    nx = -SLOPE * sgn
    ny = -SLOPE * TILT * sgn
    nz = np.ones_like(nx)
    nl = np.sqrt(nx * nx + ny * ny + nz * nz)
    sig = max(1.5 / CELL, 2.0)
    NXr = gaussian_filter(nx / nl, sig)
    NYr = gaussian_filter(ny / nl, sig)
    NZr = gaussian_filter(nz / nl, sig)
    nl = np.clip(np.sqrt(NXr * NXr + NYr * NYr + NZr * NZr), 1e-6, None)
    return SurfaceField(cell=CELL, origin=(0.0, 0.0), P=P, B=B,
                        mask=np.ones((ru, rv), bool), Bo=B.copy(),
                        V=V, F=F, N=N,
                        dlo=np.full((ru, rv), 2.0, np.float32),
                        dhi=np.full((ru, rv), 17.5, np.float32),
                        Pgu=Pgu.astype(np.float32),
                        Pgv=Pgv.astype(np.float32),
                        NXr=(NXr / nl).astype(np.float32),
                        NYr=(NYr / nl).astype(np.float32),
                        NZr=(NZr / nl).astype(np.float32))


def _params():
    return RibParams(pattern="rectangular", spacing=10, thickness=2.0,
                     height=4.0, margin=0, taper_len=0, fillet_root=0.0,
                     fillet_top=0.0)


def _build_marched_grid(monkeypatch):
    """Run mesh_field exactly as shipped and capture the marched grid —
    the very array handed to the extractor — plus its absolute
    coordinates (halo inferred from the captured shape, so the test
    does not hardcode the engine's halo policy)."""
    from skimage import measure
    rec = []
    orig = measure.marching_cubes

    def spy(vol, level, **kw):
        rec.append(np.array(vol, copy=True))
        return orig(vol, level, **kw)

    monkeypatch.setattr(measure, "marching_cubes", spy)
    clusters = mesh_field(_tilted_channel_surface(), _params(),
                          resolution=RES, extractor="marching_cubes")
    monkeypatch.setattr(measure, "marching_cubes", orig)
    assert len(rec) == 1, "fixture no longer fits one tile-slab"
    vol = rec[0]
    iu0 = int(np.floor(0.0 / RES)) - 1
    iu1 = int(np.ceil(40.0 / RES)) + 1
    iv0 = int(np.floor(0.0 / RES)) - 1
    iv1 = int(np.ceil(30.0 / RES)) + 1
    G = (vol.shape[0] - (iu1 - iu0 + 1)) // 2
    assert vol.shape[0] == iu1 - iu0 + 1 + 2 * G, vol.shape
    assert vol.shape[1] == iv1 - iv0 + 1 + 2 * G, vol.shape
    d0 = int(np.floor(2.0 / RES))
    d1 = int(np.ceil(17.5 / RES))
    assert vol.shape[2] == d1 - d0 + 1 + 2 * G, vol.shape
    us = np.arange(iu0 - G, iu1 + G + 1) * RES
    vs = np.arange(iv0 - G, iv1 + G + 1) * RES
    dsv = np.arange(d0 - G, d1 + G + 1) * RES
    return vol, us, vs, dsv, clusters


def _seam_lipschitz(vol, us, vs, dsv):
    """Per-axis max |df|/RES over sample pairs inside the seam-tracking
    region |u - uc(v)| <= 3 (the channel), v in [5, 25] (off the open
    rims), z in [6.6, 9.2] (the rib band the original sweeps measured;
    below it the foot flips are sub-threshold by the blend's design —
    <= ~1mm at the rib root — and above the rim the sheets end)."""
    U = us[:, None, None]
    Vv = vs[None, :, None]
    D = dsv[None, None, :]
    reg = np.broadcast_to(((np.abs(U - _uc(Vv)) <= 3.0)
                           & (Vv >= 5.0) & (Vv <= 25.0)
                           & (D >= 6.6) & (D <= 9.2)), vol.shape)
    out = []
    for ax in range(3):
        sl0 = tuple(slice(None, -1) if a == ax else slice(None)
                    for a in range(3))
        sl1 = tuple(slice(1, None) if a == ax else slice(None)
                    for a in range(3))
        d = np.abs(vol[sl1] - vol[sl0])[reg[sl0] & reg[sl1]]
        out.append(float(d.max()) / RES if d.size else 0.0)
    return out


def test_marched_field_continuous_across_channel(monkeypatch):
    # THE braid outcome, on the engine as configured (no env forcing:
    # running this file with RIBTOOL_CREASE_BLEND=0 must fail here).
    # Seam-parallel axes (v, d) carry the defect: the raw map flips the
    # full |f_left - f_right| between adjacent columns wherever the
    # oblique medial sheet crosses them.  The seam-normal axis (u) is
    # the designed hard-miter handover and is diagnostic only.
    vol, us, vs, dsv, clusters = _build_marched_grid(monkeypatch)
    Lu, Lv, Ld = _seam_lipschitz(vol, us, vs, dsv)
    print(f"seam-parallel steps: v {Lv:.2f}, d {Ld:.2f} x res "
          f"(bound {LIP_TOL}); miter-normal u {Lu:.2f} (diagnostic)")
    worst = max(Lv, Ld)
    assert worst < LIP_TOL, (
        f"marched field steps {worst:.2f}x res between adjacent seam-"
        f"parallel samples in the channel rib band (bound {LIP_TOL}): "
        f"the composed field still flips wall-sheet branches at voxel "
        f"pitch along the medial seam — ribs crossing the channel "
        f"extract as braided strands")
    # the same build must still deliver one watertight solid
    import manifold3d as m3d
    assert clusters and all(
        not m3d.Manifold(m3d.Mesh(np.ascontiguousarray(v, np.float32),
                                  np.ascontiguousarray(t, np.uint32)))
        .is_empty() for v, t in clusters)


def test_braid_metric_detects_raw_map(monkeypatch):
    # HONESTY GUARD for the re-target: with the blend disabled the same
    # metric must fail by a wide margin, or the primary assertion above
    # has gone vacuous (fixture drift, metric rot, dead kill-switch).
    monkeypatch.setenv("RIBTOOL_CREASE_BLEND", "0")
    vol, us, vs, dsv, _ = _build_marched_grid(monkeypatch)
    Lu, Lv, Ld = _seam_lipschitz(vol, us, vs, dsv)
    print(f"raw-map seam-parallel steps: v {Lv:.2f}, d {Ld:.2f} x res; "
          f"u {Lu:.2f}")
    assert max(Lv, Ld) > 1.3 * LIP_TOL, (
        f"raw closest-point map no longer shows the braid "
        f"({max(Lv, Ld):.2f}x res <= {1.3 * LIP_TOL}): the fixture or "
        f"metric has drifted and the primary test proves nothing")


def test_blend_exactly_inert_without_foot_jump(monkeypatch):
    # WELT regression (same-sheet smin non-idempotency): both crease
    # tiers require a foot jump > tau1 across some incident grid face,
    # so any sample with NO such face has no genuine second sheet and
    # the blend's contract is EXACT inertness there (F == f1, bitwise).
    # An unfloored partner argmax used to hand every dilation-ring
    # sample a same-sheet partner (f2 ~ f1), and smin(a, a) = a - k_c/4
    # stamped a ~0.075mm proud welt band beside every detected seam
    # (25,644 sub-threshold samples on this fixture, median -0.075).
    from server.geometry.implicit import _CREASE_TAU_LO
    monkeypatch.delenv("RIBTOOL_CREASE_BLEND", raising=False)
    von, us_on, _, _, _ = _build_marched_grid(monkeypatch)
    monkeypatch.setenv("RIBTOOL_CREASE_BLEND", "0")
    voff, us, vs, dsv, _ = _build_marched_grid(monkeypatch)
    monkeypatch.delenv("RIBTOOL_CREASE_BLEND", raising=False)
    off = int(round((us[0] - us_on[0]) / RES))
    assert off >= 0 and all(von.shape[i] - voff.shape[i] == 2 * off
                            for i in range(3)), (von.shape, voff.shape)
    if off:
        von = von[off:-off, off:-off, off:-off]
    # feet on the common grid, engine-style: x - s*n(cp) in float32
    g = _tilted_channel_surface()
    UU, VV, DD = np.meshgrid(us, vs, dsv, indexing="ij")
    X = np.column_stack([UU.ravel(), VV.ravel(), DD.ravel()])
    s, _, nrm = _surface_eval(X, g.V, g.F, g.N)
    FU = (X[:, 0].astype(np.float32) - s * nrm[:, 0]).reshape(voff.shape)
    FV = (X[:, 1].astype(np.float32) - s * nrm[:, 1]).reshape(voff.shape)
    tau1 = max(_CREASE_TAU_LO, 4.0 * RES)
    big = np.zeros(voff.shape, bool)     # sample has a face jump > tau1
    for ax in range(3):
        sl0 = tuple(slice(None, -1) if a == ax else slice(None)
                    for a in range(3))
        sl1 = tuple(slice(1, None) if a == ax else slice(None)
                    for a in range(3))
        hit = np.hypot(FU[sl1] - FU[sl0], FV[sl1] - FV[sl0]) > tau1
        big[sl0] |= hit
        big[sl1] |= hit
    diff = von != voff
    # anti-vacuity: the blend must have genuinely acted on jump samples
    assert int((diff & big).sum()) > 1000, \
        "blend inactive on the crease fixture — the assertion below is vacuous"
    bad = diff & ~big
    n_bad = int(bad.sum())
    assert n_bad == 0, (
        f"{n_bad} samples with no incident foot-jump > tau1 differ "
        f"blend-on vs blend-off (median offset "
        f"{float(np.median((von - voff)[bad])):.4f}mm): the crease blend "
        f"is not bit-inert off the seam — same-sheet smin welt")


def test_braid_on_groove_engine_build():
    # Engine-level reproduction on the conftest groove_step fixture
    # (top + groove faces) was attempted and does NOT reproduce the
    # braid — for a geometric reason, not a resolution one, so it is
    # skipped rather than asserted:
    #   * groove_step's plane/cylinder junctions are CONVEX edges: in
    #     the air wedge above them the closest point CLAMPS to the edge
    #     curve and the foot stays continuous.  Measured on a 2.5s
    #     build (isogrid 12/1.6/h4, faces top+groove): vertical rib
    #     walls cross the junction pinned at x = 15.20/16.80 with zero
    #     dislocation.
    #   * the groove itself is smoothly concave with its medial axis at
    #     the cylinder axis, 18 mm off the surface — far outside the
    #     <=4 mm rib band, so no evaluation point ever crosses it.
    #   The braid needs OPPOSING walls a few mm apart (a recessed
    #   channel), whose medial sheet lies INSIDE the rib band.
    # The marched-grid tests above reproduce exactly that geometry at
    # the kernel level and assert the outcome on the engine's own grid.
    pytest.skip("braid cannot reproduce on groove_step: its junctions "
                "are convex (edge-clamped feet) and its concave medial "
                "axis lies 18mm off-surface, outside the rib band; "
                "the marched-grid tests above carry the reproduction")
