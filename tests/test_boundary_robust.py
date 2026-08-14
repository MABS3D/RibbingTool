"""Regression tests for boundary-poke robustness and taper-aware fillets.

Root cause (found on 180 90 cruscotto.stp): capsule caps and offset rings
legitimately poke past the flattened domain edge; the per-cluster snap gate
(max_snap=0.5) treated ONE such point as fatal for the WHOLE cluster, so big
merged lattice clusters silently degraded to per-segment capsules — killing
cluster fillets (fillet_top no-op), skipping boundary ribs outright, and at
margin 0 building nothing at all.  Separately, fillet geometry (integrated
arcs and additive beads) ignored the local tapered rib height, rising to full
r_root even where the rib fades to zero (club-shaped run-out tips).
"""
import numpy as np
import pytest

from server.geometry.patterns import RibParams
from server.geometry.ribbing import build_rib_meshes
from server.geometry.step_io import load_step
from tests.test_ribbing import biggest_face_id


def _watertight(v, t):
    import manifold3d as m3d
    man = m3d.Manifold(m3d.Mesh(np.ascontiguousarray(v, np.float32),
                                np.ascontiguousarray(t, np.uint32)))
    return not man.is_empty()


def _face_plane(shape, fid):
    from server.geometry.meshing import mesh_shape
    m = next(x for x in mesh_shape(shape) if x.face_id == fid)
    z0 = float(m.vertices[:, 2].mean())
    return z0, (1.0 if z0 > 4 else -1.0)


def test_margin_zero_builds_full_coverage(box_step):
    # segments clipped AT the boundary poke their caps past it; the mapper
    # must clamp those points onto the edge instead of refusing the cluster
    s = load_step(box_step)
    fid = biggest_face_id(s)
    p = RibParams(pattern="rectangular", spacing=10, spacing_y=0,
                  thickness=1.6, height=4, margin=0, taper_len=0)
    clusters, reports = build_rib_meshes(s, [fid], p)
    assert sum(r.lofted for r in reports) == sum(r.segments for r in reports)
    assert sum(r.skipped for r in reports) == 0
    av = np.vstack([v for v, t in clusters])
    # ribs must actually reach the face edges (60 x 40 box face)
    assert av[:, 0].min() < 1.0 and av[:, 0].max() > 59.0
    assert all(_watertight(v, t) for v, t in clusters)


def test_merged_cluster_survives_boundary_pokes(box_step):
    # a quadmesh lattice touching the boundary must stay ONE welded cluster,
    # not shatter into per-segment capsules (or lose boundary cells)
    s = load_step(box_step)
    fid = biggest_face_id(s)
    p = RibParams(pattern="quadmesh", spacing=12, thickness=1.6, height=4,
                  margin=0, taper_len=0)
    clusters, reports = build_rib_meshes(s, [fid], p)
    assert sum(r.lofted for r in reports) == sum(r.segments for r in reports)
    assert sum(r.skipped for r in reports) == 0
    assert len(clusters) <= 5          # merged lattice, not 22+ fragments
    assert all(_watertight(v, t) for v, t in clusters)


def test_fillet_top_applies_on_merged_lattice(box_step):
    # fillet_top must materialize on the merged cluster itself — not vanish
    # via silent whole-cluster degrade
    s = load_step(box_step)
    fid = biggest_face_id(s)
    base = dict(pattern="quadmesh", spacing=12, thickness=1.6, height=4,
                margin=0, taper_len=0)
    plain, _ = build_rib_meshes(s, [fid], RibParams(**base))
    ftop, reports = build_rib_meshes(
        s, [fid], RibParams(**base, fillet_top=0.5))
    assert not any("fillets skipped" in w
                   for r in reports for w in r.warnings)
    vol = lambda cs: sum(abs(np.einsum("ij,ij->i", v[t[:, 0]],
                         np.cross(v[t[:, 1]], v[t[:, 2]])).sum() / 6.0)
                         for v, t in cs)
    # the crown rounds material off the top edges
    assert vol(ftop) < vol(plain) * 0.999
    assert all(_watertight(v, t) for v, t in ftop)


def test_root_fillet_respects_taper(box_step):
    # in the taper run-out the fillet must fade with the rib: mesh height in
    # the outer ramp zone stays near the ramp, never at full r_root
    s = load_step(box_step)
    fid = biggest_face_id(s)
    z0, sign = _face_plane(s, fid)
    p = RibParams(pattern="rectangular", spacing=10, spacing_y=0, thickness=2,
                  height=5, margin=3, taper_len=8, fillet_root=1.5)
    clusters, _ = build_rib_meshes(s, [fid], p)
    av = np.vstack([v for v, t in clusters])
    h = sign * (av[:, 2] - z0)
    d = np.minimum.reduce([av[:, 0], 60 - av[:, 0], av[:, 1], 40 - av[:, 1]])
    zone1 = (d >= 3.0) & (d < 4.0)     # ramp allows 0.62 here
    zone2 = (d >= 4.0) & (d < 5.0)     # ramp allows 1.25 here
    assert zone1.any() and zone2.any()
    assert h[zone1].max() <= 0.9, f"run-out bulge: {h[zone1].max():.2f}"
    assert h[zone2].max() <= 1.6, f"run-out bulge: {h[zone2].max():.2f}"
    assert h.max() == pytest.approx(5.0, abs=0.05)   # full height mid-face
    assert all(_watertight(v, t) for v, t in clusters)


@pytest.mark.slow
def test_whole_front_single_apply(cruscotto_full_path):
    # the user's real workflow: ONE apply across the whole front at
    # spacing 12 must fit under the segment cap and keep cluster coverage
    from server.geometry.selection import grow_tangent
    s = load_step(cruscotto_full_path)
    sel = grow_tangent(s, [287], angle_deg=20.0)
    assert len(sel) > 200
    p = RibParams(pattern="isogrid", spacing=12, thickness=1.2, height=1.5,
                  margin=2, taper_len=10)
    clusters, reports = build_rib_meshes(s, sel, p)
    lofted = sum(r.lofted for r in reports)
    segments = sum(r.segments for r in reports)
    assert segments > 4000             # today this raises before building
    assert lofted / segments > 0.97
