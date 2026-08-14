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
    # cluster raw vertex x's into rib centers
    def centers(a):
        cs, cur = [], [a[0]]
        for x in a[1:]:
            if x - cur[-1] > 2.0:
                cs.append(np.mean(cur)); cur = []
            cur.append(x)
        cs.append(np.mean(cur))
        return np.asarray(cs)
    cl, cr = centers(left), centers(right)
    for x in cl:
        k = (cr - x) / 8.0
        assert (np.abs(k - np.round(k)) < 0.05).any(), \
            f"left rib at x={x:.2f} has no phase-aligned right rib (right={cr})"


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
