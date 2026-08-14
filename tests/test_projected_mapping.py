"""Projected pattern mapping: lattice generated in a global plane and
projected onto the surface, so patterns stay phase-continuous across grooves,
recesses, and even between disconnected coplanar panels (the cruscotto seam).

Surface-metric (unfold) mapping cannot do this: a recess's walls unfold to
their true width, consuming lattice periods, so the two flanks arrive out of
phase and rim bands fill with compressed ribs.
"""
import math
import pathlib

import numpy as np
import pytest

from server.geometry.patterns import RibParams
from server.geometry.ribbing import build_rib_meshes
from server.geometry.step_io import load_step


@pytest.fixture(scope="session")
def groove_box_step(tmp_path_factory):
    # 80x40x10 box with an off-center 4mm-wide, 6mm-deep groove across the
    # top: two coplanar top panels of DIFFERENT widths (30 and 46), so
    # unfold mapping provably cannot phase-align them by accident
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
    from OCP.gp import gp_Pnt
    from tests.conftest import write_step
    box = BRepPrimAPI_MakeBox(80.0, 40.0, 10.0).Shape()
    slot = BRepPrimAPI_MakeBox(gp_Pnt(30.0, -1.0, 4.0), 4.0, 42.0, 7.0).Shape()
    shape = BRepAlgoAPI_Cut(box, slot).Shape()
    p = tmp_path_factory.mktemp("fx") / "groovebox.step"
    write_step(shape, p)
    return p


def _top_panel_faces(shape):
    # the two z=10 planar panels on either side of the groove
    from server.geometry.meshing import mesh_shape
    out = []
    for m in mesh_shape(shape):
        v = m.vertices
        if np.allclose(v[:, 2], 10.0, atol=1e-6) and len(v) >= 3:
            out.append(m.face_id)
    assert len(out) == 2, f"expected 2 top panels, got {out}"
    return out


def _rib_xs_at(clusters, y_probe, z_min):
    """X positions where rib material crosses the line y=y_probe on top."""
    xs = []
    for v, t in clusters:
        m = (np.abs(v[:, 1] - y_probe) < 1.5) & (v[:, 2] > z_min)
        xs.extend(v[m, 0].tolist())
    return np.sort(np.asarray(xs))


def _centers(a):
    """Cluster raw vertex x's into rib centers."""
    cs, cur = [], [a[0]]
    for x in a[1:]:
        if x - cur[-1] > 2.0:
            cs.append(np.mean(cur)); cur = []
        cur.append(x)
    cs.append(np.mean(cur))
    return np.asarray(cs)


def _assert_one_grid(cl, cr, spacing):
    # for every rib on side A there is a rib on side B whose x differs by
    # an integer multiple of spacing (+-0.05 periods)
    for x in cl:
        k = (cr - x) / spacing
        assert (np.abs(k - np.round(k)) < 0.05).any(), \
            f"left rib at x={x:.2f} has no phase-aligned right rib (right={cr})"


def test_projected_phase_continuous_across_groove(groove_box_step):
    s = load_step(groove_box_step)
    panels = _top_panel_faces(s)
    p = RibParams(pattern="rectangular", spacing=8, spacing_y=0,
                  thickness=1.6, height=3, margin=1, taper_len=0,
                  orientation_deg=90.0, mapping="project")
    clusters, reports = build_rib_meshes(s, panels, p)
    assert sum(r.lofted for r in reports) > 0
    # ribs run along y (orientation 90): their x positions form the lattice.
    # Sample rib x-centers on each panel and check both sides sit on ONE
    # global 8mm grid: for every rib on side A there is a rib on side B
    # whose x differs by an integer multiple of spacing (+-0.4mm)
    xs = _rib_xs_at(clusters, 20.0, 10.5)
    assert len(xs) > 4
    left = xs[xs < 30.0]
    right = xs[xs > 34.0]
    assert len(left) and len(right)
    _assert_one_grid(_centers(left), _centers(right), 8.0)


def test_projected_mapping_on_curved(cyl_patch_step):
    from tests.test_ribbing import biggest_face_id
    s = load_step(cyl_patch_step)
    fid = biggest_face_id(s, kind="cylinder")
    p = RibParams(pattern="isogrid", spacing=12, thickness=1.6, height=3,
                  taper_len=0, mapping="project")
    clusters, reports = build_rib_meshes(s, [fid], p)
    assert reports[0].lofted > 5
    import manifold3d as m3d
    for v, t in clusters:
        man = m3d.Manifold(m3d.Mesh(np.ascontiguousarray(v, np.float32),
                                    np.ascontiguousarray(t, np.uint32)))
        assert not man.is_empty()


def test_projected_bridges_hairline_groove(groove_box_step):
    # the 4mm groove projects to a slit that splits the kept domain and
    # margin carves a bare channel along it; morphological closing must
    # seal it so ribs CROSS the groove in one welded, watertight cluster
    # (points over the sealed strip clamp onto the flanks — the bridge)
    from tests.test_boundary_robust import _watertight
    s = load_step(groove_box_step)
    panels = _top_panel_faces(s)
    p = RibParams(pattern="rectangular", spacing=8, spacing_y=0,
                  thickness=1.6, height=3, margin=1, taper_len=0,
                  mapping="project")
    clusters, reports = build_rib_meshes(s, panels, p)
    lofted = sum(r.lofted for r in reports)
    segments = sum(r.segments for r in reports)
    assert segments and lofted / segments > 0.9
    bridging = False
    for v, t in clusters:
        if not ((v[:, 0] < 29.0).any() and (v[:, 0] > 35.0).any()):
            continue                   # cluster does not reach both flanks
        span = ((v[t, 0].min(axis=1) < 30.2) & (v[t, 0].max(axis=1) > 33.8)
                & (v[t, 2].min(axis=1) > 3.0))
        bridging = bridging or bool(span.any())
    assert bridging, "no rib crosses the groove"
    assert all(_watertight(v, t) for v, t in clusters)


def test_frame_cache_aligns_separate_applies(groove_box_step):
    # the real workflow is one apply per panel: a shared frame_cache must
    # land both on ONE global lattice even though the panels' own PCA axes
    # differ (30x40 vs 46x40 swaps the in-plane principal directions)
    from server.geometry.meshing import mesh_shape
    s = load_step(groove_box_step)
    panels = _top_panel_faces(s)
    cx = {m.face_id: float(m.vertices[:, 0].mean())
          for m in mesh_shape(s) if m.face_id in panels}
    left_id = min(panels, key=lambda f: cx[f])
    right_id = max(panels, key=lambda f: cx[f])
    p = RibParams(pattern="rectangular", spacing=8, spacing_y=0,
                  thickness=1.6, height=3, margin=1, taper_len=0,
                  mapping="project")
    cache = {}
    ca, _ = build_rib_meshes(s, [left_id], p, frame_cache=cache)
    cb, _ = build_rib_meshes(s, [right_id], p, frame_cache=cache)
    xs = _rib_xs_at(ca + cb, 20.0, 10.5)
    assert len(xs) > 4
    left = xs[xs < 30.0]
    right = xs[xs > 34.0]
    assert len(left) and len(right)
    _assert_one_grid(_centers(left), _centers(right), 8.0)


def test_projected_band_clusters_outward(cruscotto_full_path):
    # a dim-rendered half band = flipped shading normals = inward winding:
    # every cluster mesh must enclose POSITIVE signed volume
    p = RibParams(pattern="isogrid", spacing=12, thickness=1.2, height=1.5,
                  margin=2, taper_len=10, mapping="project")
    s = load_step(cruscotto_full_path)
    clusters, reports = build_rib_meshes(s, [519, 580], p)
    assert sum(r.lofted for r in reports) > 500
    vols = [np.einsum("ij,ij->i", v[t[:, 0]],
                      np.cross(v[t[:, 1]], v[t[:, 2]])).sum() / 6.0
            for v, t in clusters]
    bad = [f"{x:.1f}" for x in vols if x <= 0]
    assert not bad, f"inward-wound clusters (signed volume <= 0): {bad}"


def test_unfold_remains_default(box_step):
    # absence of the parameter (and legacy payloads) must keep today's
    # behavior byte-for-byte
    from tests.test_ribbing import biggest_face_id
    s = load_step(box_step)
    fid = biggest_face_id(s)
    base = dict(pattern="rectangular", spacing=10, spacing_y=0,
                thickness=1.6, height=4, margin=2, taper_len=0)
    a, _ = build_rib_meshes(s, [fid], RibParams(**base))
    b, _ = build_rib_meshes(s, [fid], RibParams(**base, mapping="unfold"))
    va = np.vstack([v for v, t in a])
    vb = np.vstack([v for v, t in b])
    assert va.shape == vb.shape and np.allclose(va, vb)
