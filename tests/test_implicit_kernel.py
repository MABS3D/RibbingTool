"""Implicit kernel unit tests on synthetic surface fields (no OCCT).

The kernel meshes a field of the form
    smooth_min(rib_prism, body_half_space, k=fillet_root)
capped by a slab floor, so every output solid is ribs + a thin sub-surface
slab in ONE watertight body.  Rib-only volumes are measured by differencing
against a slab-only run (empty pattern).  The substrate is a synthetic
triangulated plane (optionally sloped): the kernel ribs the exact
closest-point distance field of that mesh, so height is true surface-
normal distance everywhere.
"""
import numpy as np
import pytest

from server.geometry.implicit import (SurfaceField, _exact_pattern_distance,
                                      _vertex_normals, mesh_field)
from server.geometry.patterns import RibParams


CELL = 0.25


def _surface(pattern_dist, size=(40.0, 30.0), depth=10.0, slope=0.0,
             bdist=1e6):
    """Triangulated plane substrate (optionally sloped along u) + rasters."""
    step = 1.0
    xs = np.arange(0.0, size[0] + 1e-9, step)
    ys = np.arange(0.0, size[1] + 1e-9, step)
    uu, vv = np.meshgrid(xs, ys, indexing="ij")
    zz = depth + slope * uu
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
    # rasters live on the RASTER pitch (CELL), not the mesh pitch
    ru = int(size[0] / CELL) + 1
    rv = int(size[1] / CELL) + 1
    gu, gv = np.meshgrid(np.arange(ru) * CELL, np.arange(rv) * CELL,
                         indexing="ij")
    P = pattern_dist(gu, gv).astype(np.float32)
    B = np.full((ru, rv), bdist, np.float32)
    mask = np.ones((ru, rv), bool)
    dlo = np.full((ru, rv), float(zz.min()) - 3.0, np.float32)
    dhi = np.full((ru, rv), float(zz.max()) + 8.0, np.float32)
    Pgu, Pgv = np.gradient(P, CELL)
    # analytic plane normal, constant across the raster
    npl = np.array([-slope, 0.0, 1.0]) / np.sqrt(1.0 + slope * slope)
    return SurfaceField(cell=CELL, origin=(0.0, 0.0), P=P, B=B, mask=mask,
                        Bo=B.copy(), V=V, F=F, N=N, dlo=dlo, dhi=dhi,
                        Pgu=Pgu.astype(np.float32),
                        Pgv=Pgv.astype(np.float32),
                        NXr=np.full((ru, rv), npl[0], np.float32),
                        NYr=np.full((ru, rv), npl[1], np.float32),
                        NZr=np.full((ru, rv), npl[2], np.float32))


def _volume(clusters):
    return sum(np.einsum("ij,ij->i", v[t[:, 0]],
                         np.cross(v[t[:, 1]], v[t[:, 2]])).sum() / 6.0
               for v, t in clusters)


def _watertight(clusters):
    import manifold3d as m3d
    for v, t in clusters:
        man = m3d.Manifold(m3d.Mesh(np.ascontiguousarray(v, np.float32),
                                    np.ascontiguousarray(t, np.uint32)))
        if man.is_empty():
            return False
    return bool(clusters)


def _params(**kw):
    base = dict(pattern="rectangular", spacing=10, thickness=2.0, height=4.0,
                margin=0, taper_len=0, fillet_root=0.0, fillet_top=0.0)
    base.update(kw)
    return RibParams(**base)


SLAB = lambda uu, vv: np.full_like(uu, 1e6)          # no ribs anywhere
ONE_RIB = lambda uu, vv: np.abs(uu - 20.0)           # rib along v at u=20
CROSS = lambda uu, vv: np.minimum(np.abs(uu - 20.0), np.abs(vv - 15.0))


def test_single_rib_volume_analytic():
    p = _params()
    slab = mesh_field(_surface(SLAB), p, resolution=CELL)
    rib = mesh_field(_surface(ONE_RIB), p, resolution=CELL)
    assert _watertight(slab) and _watertight(rib)
    v_rib = _volume(rib) - _volume(slab)
    expected = 30.0 * 2.0 * 4.0          # length x thickness x height
    assert v_rib == pytest.approx(expected, rel=0.06)


def test_smooth_min_adds_material_monotonically():
    vols = []
    for k in (0.0, 0.6, 1.2):
        clusters = mesh_field(_surface(ONE_RIB), _params(fillet_root=k),
                              resolution=CELL)
        assert _watertight(clusters)
        vols.append(_volume(clusters))
    assert vols[0] < vols[1] < vols[2]


def test_crown_rounding_removes_material():
    v0 = _volume(mesh_field(_surface(ONE_RIB), _params(), resolution=CELL))
    v1 = _volume(mesh_field(_surface(ONE_RIB), _params(fillet_top=0.6),
                            resolution=CELL))
    assert v1 < v0 - 1.0


def test_taper_follows_boundary_ramp():
    # boundary distance grows with u: height must ramp 0 -> full over 8mm
    def surface():
        g = _surface(lambda uu, vv: np.abs(vv - 15.0))   # rib along u
        nu = g.B.shape[0]
        ramp = (np.arange(nu, dtype=np.float32) * CELL)[:, None] \
            * np.ones_like(g.B)
        g.B = ramp
        g.Bo = ramp.copy()
        return g
    p = _params(taper_len=8.0)
    clusters = mesh_field(surface(), p, resolution=CELL)
    assert _watertight(clusters)
    v = np.vstack([c[0] for c in clusters])
    for u0, u1, frac in ((1.0, 3.0, 3.0 / 8), (5.0, 7.0, 7.0 / 8),
                         (12.0, 20.0, 1.0)):
        m = (v[:, 0] >= u0) & (v[:, 0] < u1)
        top = v[m, 2].max() - 10.0                      # above the surface
        allowed = 4.0 * frac
        assert top <= allowed + 0.35, f"u [{u0},{u1}): {top} > {allowed}"
    assert v[:, 2].max() == pytest.approx(14.0, abs=0.3)   # full height hit


def test_taper_runout_is_surface_length_on_slope():
    # taper_len is a SURFACE run-out length.  B is measured in PROJECTED
    # mm, and a slope stretches every projected mm by 1/cos on the
    # surface: uncorrected, a 60-degree slope turns an 8mm run-out into a
    # 16mm fan of pointed stubs (the user's fern band).  With the facing
    # compensation the ramp must complete within ~taper_len of PROJECTED
    # distance scaled by cos(slope).
    slope = 1.7320508                       # 60 degrees: cos = 0.5
    def surface():
        g = _surface(lambda uu, vv: np.abs(vv - 15.0), slope=slope)
        nu = g.B.shape[0]
        ramp = (np.arange(nu, dtype=np.float32) * CELL)[:, None] \
            * np.ones_like(g.B)
        g.B = ramp
        g.Bo = ramp.copy()
        return g
    p = _params(taper_len=8.0)
    clusters = mesh_field(surface(), p, resolution=CELL)
    assert _watertight(clusters)
    v = np.vstack([c[0] for c in clusters])
    import igl
    g = surface()
    sq, _, Cp = igl.point_mesh_squared_distance(
        np.ascontiguousarray(v), g.V, g.F)
    h = np.sqrt(sq)
    # window on the FOOT (closest surface point) u — rib tops lean along
    # the tilted normal, so windowing on the vertex u lets tops rooted
    # farther up the ramp leak in.  Ramp must COMPLETE by foot-u ~
    # taper*cos(60)=4mm (+slack); uncorrected it is only at
    # 4*(4.3..5.3)/8 = 2.2..2.7mm there.
    m = (Cp[:, 0] >= 4.3) & (Cp[:, 0] < 5.3) & (h > 0.3)
    assert m.any()
    assert h[m].max() > 0.85 * 4.0, \
        f"run-out still ramping at foot 4.3-5.3mm: max h {h[m].max():.2f}"


def test_rib_width_is_surface_metric_on_slope():
    # the wall cut P < thickness/2 is a PROJECTED band: a rib crossing a
    # 60-degree slope stretches to thickness/cos(60) = 2x width on the
    # surface (wide flat leaves with staircase tops — the user's fern).
    # With the metric correction the SURFACE width is thickness, so the
    # PROJECTED width must be thickness*cos(60) = 0.8mm.
    slope = 1.7320508                       # 60 degrees along u
    def surface():
        # one rib along v at u=20: its perpendicular IS the fall line
        return _surface(lambda uu, vv: np.abs(uu - 20.0), slope=slope)
    p = _params()
    clusters = mesh_field(surface(), p, resolution=CELL)
    assert _watertight(clusters)
    v = np.vstack([c[0] for c in clusters])
    import igl
    g = surface()
    sq, _, _ = igl.point_mesh_squared_distance(
        np.ascontiguousarray(v), g.V, g.F)
    h = np.sqrt(sq)
    # wall band: a THIN height slice (the rib leans along the tilted
    # normal — a tall slice conflates lean with width), mid-domain in v
    m = (h > 1.35) & (h < 1.65) & (v[:, 1] > 8) & (v[:, 1] < 22)
    assert m.sum() > 50
    width = np.percentile(v[m, 0], 98) - np.percentile(v[m, 0], 2)
    # projected width must be ~thickness*cos(60)=0.8 (+lean of the slice
    # 0.26 + voxel slack), NOT the uncorrected 1.6+lean=1.9
    assert width < 1.35, \
        f"rib projected width {width:.2f}mm — stretched on the slope"


def test_lean_cap_over_mask_holes():
    # blades on a slope lean sideways over openings by height*sin(slope);
    # the Bv lean cap trims them ~1.6mm past the domain edge so windows
    # stay clear.  Self-validating: without Bv the same build leans deep
    # into the hole.
    from scipy.ndimage import distance_transform_edt
    slope = 1.0                              # 45 degrees along u

    def surface(with_bv):
        g = _surface(lambda uu, vv: np.abs(vv - 15.0), slope=slope)
        # punch a mask hole DOWNSLOPE of the rib's crossing
        i0 = int(22.0 / CELL)
        i1 = int(34.0 / CELL)
        j0 = int(9.0 / CELL)
        j1 = int(21.0 / CELL)
        g.mask[i0:i1, j0:j1] = False
        g.B[i0:i1, j0:j1] = 0.0
        if with_bv:
            g.Bv = (distance_transform_edt(~g.mask) * CELL).astype(
                np.float32)
        return g

    def penetration(g):
        # depth into the hole = distance to the NEAREST rim (material may
        # legitimately fringe ~1.6mm along every rim)
        clusters = mesh_field(g, _params(), resolution=CELL)
        v = np.vstack([c[0] for c in clusters])
        m = (v[:, 0] > 22.0) & (v[:, 0] < 34.0) & (v[:, 1] > 9.0) \
            & (v[:, 1] < 21.0)
        if not m.any():
            return 0.0
        din = np.minimum.reduce([v[m, 0] - 22.0, 34.0 - v[m, 0],
                                 v[m, 1] - 9.0, 21.0 - v[m, 1]])
        return float(din.max())

    deep = penetration(surface(False))
    capped = penetration(surface(True))
    assert deep > 2.4, f"fixture does not lean ({deep:.2f}mm) — dead test"
    assert capped < 2.3, \
        f"lean cap failed: {capped:.2f}mm into the mask hole"


def test_slope_ribs_extrude_along_surface_normal():
    # on a 30-degree slope the rib top plane is parallel to the surface,
    # offset along the normal: measured VERTICALLY it sits height/cos(30)
    # above the surface.  This is now exact by construction (s IS the
    # normal distance) — no gradient correction involved.
    g = _surface(lambda uu, vv: np.abs(vv - 15.0),       # rib along u
                 slope=np.tan(np.radians(30.0)))
    p = _params()
    clusters = mesh_field(g, p, resolution=CELL)
    assert _watertight(clusters)
    v = np.vstack([c[0] for c in clusters])
    m = (np.abs(v[:, 1] - 15.0) < 0.8) & (v[:, 0] > 8) & (v[:, 0] < 32)
    rise = v[m, 2] - (10.0 + v[m, 0] * np.tan(np.radians(30.0)))
    expected = 4.0 / np.cos(np.radians(30.0))          # 4.62
    assert rise.max() == pytest.approx(expected, abs=0.25)


def test_exact_segment_distance_subvoxel():
    # EDT of a rasterized staircase scallops by ~cell/2 — walls ripple and
    # border ridges crenellate. Exact band distance must be sub-voxel true.
    seg = [[(3.0, 2.0), (36.0, 27.0)]]                 # diagonal polyline
    P = _exact_pattern_distance(seg, cell=0.25, origin=(0.0, 0.0),
                                shape2d=(161, 121), reach=4.0)
    a = np.array([3.0, 2.0]); b = np.array([36.0, 27.0])
    d = (b - a) / np.linalg.norm(b - a)
    rng = np.random.default_rng(3)
    for _ in range(60):
        t = rng.uniform(0.1, 0.9)
        off = rng.uniform(-3.0, 3.0)
        q = a + t * (b - a) + off * np.array([-d[1], d[0]])
        i, j = int(round(q[0] / 0.25)), int(round(q[1] / 0.25))
        grid_err = abs(P[i, j] - abs(off))
        assert grid_err < 0.19, f"{grid_err} at offset {off}"


def test_micro_hole_filling_restores_manifold():
    import manifold3d as m3d
    from server.geometry.implicit import _fill_microholes
    # unit cube as 12 triangles with ONE removed -> open 3-edge hole
    v = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                  [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], np.float64)
    f = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
                  [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
                  [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]], np.int64)
    f_holed = f[:-1]
    assert m3d.Manifold(m3d.Mesh(v.astype(np.float32),
                                 f_holed.astype(np.uint32))).is_empty()
    v2, f2 = _fill_microholes(v, f_holed)
    assert not m3d.Manifold(m3d.Mesh(v2.astype(np.float32),
                                     f2.astype(np.uint32))).is_empty()


def test_pinch_repair_splits_shared_vertex():
    # two tetrahedra sharing exactly one vertex: manifold3d rejects the
    # pinch, and a rejected lattice silently drops out of the export union
    import manifold3d as m3d
    from server.geometry.implicit import _repair_pinches
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
                  [-1, 0, 0], [0, -1, 0], [0, 0, -1]], np.float64)
    f = np.array([[0, 2, 1], [0, 3, 2], [0, 1, 3], [1, 2, 3],
                  [0, 4, 5], [0, 6, 4], [0, 5, 6], [4, 6, 5]], np.int64)
    v2, f2 = _repair_pinches(v, f)
    assert len(v2) == 8                      # pinch vertex duplicated
    assert not m3d.Manifold(m3d.Mesh(v2.astype(np.float32),
                                     f2.astype(np.uint32))).is_empty()


def test_retaining_rim_rib_survives_setback_and_taper():
    # a rib along a domain rim line must (a) survive the 1.2-cell setback —
    # the final mask is dilated past the rim, so B>0 under its centerline —
    # and (b) stand at FULL height even with taper, because the ramp
    # follows distance to OPEN boundary (Bo), not to the rim.  Fading or
    # amputating here reads as melted/chewed stubs.
    cell = CELL
    nu, nv = 161, 121                       # 40 x 30 mm
    steep = np.zeros((nu, nv), bool)
    steep[119:122, :] = True                # rim line along u = 30
    mask = ~steep
    final = mask.copy()
    final[116:121, :] = True
    rim_chain = [(30.0, float(j * cell)) for j in range(nv)]
    P = _exact_pattern_distance([rim_chain], cell, (0.0, 0.0), (nu, nv),
                                reach=4.0)
    from scipy.ndimage import distance_transform_edt
    g = _surface(SLAB)
    g.P = P.astype(np.float32)
    g.B = (distance_transform_edt(final) * cell).astype(np.float32)
    g.Bo = np.full((nu, nv), 1e6, np.float32)      # open boundary: far
    g.mask = final
    p = _params(thickness=1.2, height=3.0, taper_len=10.0)
    clusters = mesh_field(g, p, resolution=CELL)
    assert _watertight(clusters)
    v = np.vstack([c[0] for c in clusters])
    m = np.abs(v[:, 0] - 30.0) < 1.5
    assert m.any(), "no material within 1.5mm of the rim line"
    rise = v[m, 2].max() - 10.0
    assert rise > 0.8 * 3.0, f"rim rib melted to {rise:.2f}mm"


def test_crossing_ribs_single_watertight_body():
    clusters = mesh_field(_surface(CROSS), _params(fillet_root=1.0),
                          resolution=CELL)
    assert _watertight(clusters)
    assert len(clusters) == 1            # one welded solid, not fragments
    v, t = clusters[0]
    # junction region contains material ABOVE the surface
    m = (np.abs(v[:, 0] - 20.0) < 2) & (np.abs(v[:, 1] - 15.0) < 2)
    assert (v[m, 2] > 12.0).any()
