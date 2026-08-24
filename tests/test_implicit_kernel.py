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
    return SurfaceField(cell=CELL, origin=(0.0, 0.0), P=P, B=B, mask=mask,
                        Bo=B.copy(), V=V, F=F, N=N, dlo=dlo, dhi=dhi)


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
