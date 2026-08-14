import math

import numpy as np
import pytest

from server.geometry.booleans import mesh_union
from server.geometry.patterns import RibParams
from server.geometry.ribbing import build_rib_meshes
from server.geometry.step_io import load_step, shape_volume
from tests.test_ribbing import biggest_face_id


def _signed_volume(v, t):
    return np.einsum("ij,ij->i", v[t[:, 0]],
                     np.cross(v[t[:, 1]], v[t[:, 2]])).sum() / 6.0


def _watertight(v, t):
    import manifold3d as m3d
    man = m3d.Manifold(m3d.Mesh(np.ascontiguousarray(v, np.float32),
                                np.ascontiguousarray(t, np.uint32)))
    return not man.is_empty()


@pytest.mark.slow
def test_part1_export_single_shell(part1_path):
    # boundary repair must make the body manifold: union -> ONE shell
    s = load_step(part1_path)
    from server.geometry.selection import grow_tangent
    grown = grow_tangent(s, [65], angle_deg=20.0)
    clusters, _ = build_rib_meshes(
        s, grown, RibParams(pattern="isogrid", spacing=12, height=4))
    shells = mesh_union(s, clusters, lin_defl=0.2)
    assert len(shells) == 1


def _face_plane_z(shape, fid):
    from server.geometry.meshing import mesh_shape
    m = next(x for x in mesh_shape(shape) if x.face_id == fid)
    return float(m.vertices[:, 2].mean())


def test_taper_reaches_zero(box_step):
    s = load_step(box_step)
    fid = biggest_face_id(s)          # one of the 60x40 faces (top OR bottom)
    z0 = _face_plane_z(s, fid)
    base = dict(pattern="rectangular", spacing=10, spacing_y=0,
                thickness=2, height=5, margin=2)
    clusters, _ = build_rib_meshes(s, [fid], RibParams(**base, taper_len=8))
    dist = np.abs(np.vstack([v for v, t in clusters])[:, 2] - z0)
    assert dist.max() > 4.5                  # full height mid-rib
    top_half = np.vstack([v[len(v) // 2:] for v, t in clusters])
    # knife edge: some top-side vertices sit essentially ON the face plane
    assert (np.abs(top_half[:, 2] - z0) < 0.02).any()


def test_fillets_watertight_and_add_volume(box_step):
    s = load_step(box_step)
    fid = biggest_face_id(s)
    z0 = _face_plane_z(s, fid)
    base = dict(pattern="quadmesh", spacing=12, thickness=2, height=5,
                margin=3, taper_len=0)
    plain, _ = build_rib_meshes(s, [fid], RibParams(**base))
    filleted, _ = build_rib_meshes(
        s, [fid], RibParams(**base, fillet_root=1.5, fillet_top=0.6))
    v0 = sum(abs(_signed_volume(v, t)) for v, t in plain)
    v1 = sum(abs(_signed_volume(v, t)) for v, t in filleted)
    assert all(_watertight(v, t) for v, t in filleted)
    assert v1 > v0 * 1.02                    # root fillet adds material
    vv = np.vstack([v for v, t in filleted])
    dist = np.abs(vv[:, 2] - z0)
    assert dist.max() == pytest.approx(5.0, abs=0.05)   # full height reached


def test_fillets_on_curved(cyl_patch_step):
    s = load_step(cyl_patch_step)
    fid = biggest_face_id(s, kind="cylinder")
    p = RibParams(pattern="isogrid", spacing=12, thickness=1.6, height=3,
                  fillet_root=1.0, fillet_top=0.5, taper_len=0)
    clusters, reports = build_rib_meshes(s, [fid], p)
    assert reports[0].lofted > 10
    assert all(_watertight(v, t) for v, t in clusters)
