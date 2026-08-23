"""Implicit kernel unit tests on synthetic frame-space grids (no OCCT).

The kernel meshes a field of the form
    smooth_min(rib_prism, body_half_space, k=fillet_root)
capped by a slab floor, so every output solid is ribs + a thin sub-surface
slab in ONE watertight body.  Rib-only volumes are measured by differencing
against a slab-only run (empty pattern).
"""
import numpy as np
import pytest

from server.geometry.implicit import FrameGrids, mesh_field
from server.geometry.patterns import RibParams


CELL = 0.25


def _grids(pattern_dist, size=(40.0, 30.0), depth=10.0, bdist=1e6):
    nu = int(size[0] / CELL) + 1
    nv = int(size[1] / CELL) + 1
    uu, vv = np.meshgrid(np.arange(nu) * CELL, np.arange(nv) * CELL,
                         indexing="ij")
    D = np.full((nu, nv), depth, np.float32)
    P = pattern_dist(uu, vv).astype(np.float32)
    B = np.full((nu, nv), bdist, np.float32)
    mask = np.ones((nu, nv), bool)
    return FrameGrids(cell=CELL, origin=(0.0, 0.0), D=D, P=P, B=B, mask=mask)


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
    slab = mesh_field(_grids(SLAB), p, resolution=CELL)
    rib = mesh_field(_grids(ONE_RIB), p, resolution=CELL)
    assert _watertight(slab) and _watertight(rib)
    v_rib = _volume(rib) - _volume(slab)
    expected = 30.0 * 2.0 * 4.0          # length x thickness x height
    assert v_rib == pytest.approx(expected, rel=0.06)


def test_smooth_min_adds_material_monotonically():
    vols = []
    for k in (0.0, 0.6, 1.2):
        clusters = mesh_field(_grids(ONE_RIB), _params(fillet_root=k),
                              resolution=CELL)
        assert _watertight(clusters)
        vols.append(_volume(clusters))
    assert vols[0] < vols[1] < vols[2]


def test_crown_rounding_removes_material():
    v0 = _volume(mesh_field(_grids(ONE_RIB), _params(), resolution=CELL))
    v1 = _volume(mesh_field(_grids(ONE_RIB), _params(fillet_top=0.6),
                            resolution=CELL))
    assert v1 < v0 - 1.0


def test_taper_follows_boundary_ramp():
    # boundary distance grows with u: height must ramp 0 -> full over 8mm
    def grids():
        g = _grids(lambda uu, vv: np.abs(vv - 15.0))   # rib along u
        nu = g.B.shape[0]
        g.B = (np.arange(nu, dtype=np.float32) * CELL)[:, None] \
            * np.ones_like(g.B)
        return g
    p = _params(taper_len=8.0)
    clusters = mesh_field(grids(), p, resolution=CELL)
    assert _watertight(clusters)
    v = np.vstack([c[0] for c in clusters])
    for u0, u1, frac in ((1.0, 3.0, 3.0 / 8), (5.0, 7.0, 7.0 / 8),
                         (12.0, 20.0, 1.0)):
        m = (v[:, 0] >= u0) & (v[:, 0] < u1)
        top = v[m, 2].max() - 10.0                      # above the surface
        allowed = 4.0 * frac
        assert top <= allowed + 0.35, f"u [{u0},{u1}): {top} > {allowed}"
    assert v[:, 2].max() == pytest.approx(14.0, abs=0.3)   # full height hit


def test_segment_clipping_preserves_direction():
    # clamping instead of clipping used to redirect out-of-window diagonal
    # family lines into false chords across the raster
    from server.geometry.implicit import _clip_to_box
    a, b = _clip_to_box(np.array([-50.0, 30.0]), np.array([150.0, 90.0]),
                        100.0, 100.0)
    d = (b - a) / np.linalg.norm(b - a)
    assert d[1] / d[0] == pytest.approx(60.0 / 200.0, abs=1e-9)
    assert a[0] == pytest.approx(0.0) and b[0] == pytest.approx(100.0)
    assert _clip_to_box(np.array([-10.0, -10.0]), np.array([-1.0, 50.0]),
                        100.0, 100.0) is None


def test_crossing_ribs_single_watertight_body():
    clusters = mesh_field(_grids(CROSS), _params(fillet_root=1.0),
                          resolution=CELL)
    assert _watertight(clusters)
    assert len(clusters) == 1            # one welded solid, not fragments
    v, t = clusters[0]
    # junction region contains material ABOVE the surface
    m = (np.abs(v[:, 0] - 20.0) < 2) & (np.abs(v[:, 1] - 15.0) < 2)
    assert (v[m, 2] > 12.0).any()
