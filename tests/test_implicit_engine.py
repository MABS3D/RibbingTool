"""Implicit engine on real geometry: parity, coverage, invariants."""
import math
import time

import numpy as np
import pytest

from server.geometry.implicit import build_rib_implicit
from server.geometry.patterns import RibParams
from server.geometry.ribbing import RibbingError, build_rib_meshes
from server.geometry.step_io import load_step
from tests.test_ribbing import biggest_face_id


def _volume(clusters):
    return sum(np.einsum("ij,ij->i", v[t[:, 0]],
                         np.cross(v[t[:, 1]], v[t[:, 2]])).sum() / 6.0
               for v, t in clusters)


def _watertight(clusters):
    import manifold3d as m3d
    return bool(clusters) and all(
        not m3d.Manifold(m3d.Mesh(np.ascontiguousarray(v, np.float32),
                                  np.ascontiguousarray(t, np.uint32)))
        .is_empty() for v, t in clusters)


def test_box_parity_with_fast_engine(box_step):
    s = load_step(box_step)
    fid = biggest_face_id(s)
    p = RibParams(pattern="quadmesh", spacing=12, thickness=2.0, height=4,
                  margin=2, taper_len=0, mapping="project")
    imp, reports = build_rib_implicit(s, [fid], p)
    assert _watertight(imp)
    fast, _ = build_rib_meshes(s, [fid], p)
    # the implicit solid includes its sub-surface slab: subtract the
    # analytic slab volume (inset area x slab thickness above the sink)
    slab = (60 - 4.0) * (40 - 4.0) * (0.3 + 0.6 - 0.05)
    v_ribs = _volume(imp) - slab
    assert v_ribs == pytest.approx(_volume(fast), rel=0.15)
    t0 = time.time()
    build_rib_implicit(s, [fid], p)
    assert time.time() - t0 < 10.0          # preview budget on the box


def test_curved_watertight(cyl_patch_step):
    s = load_step(cyl_patch_step)
    fid = biggest_face_id(s, kind="cylinder")
    p = RibParams(pattern="isogrid", spacing=12, thickness=1.6, height=3,
                  taper_len=0, mapping="project", fillet_root=1.0,
                  fillet_top=0.4)
    clusters, reports = build_rib_implicit(s, [fid], p)
    assert _watertight(clusters)
    assert len(clusters) == 1
    # ribs must rise above the largest cylinder radius (30) minus sag
    allv = np.vstack([v for v, t in clusters])
    r = np.hypot(allv[:, 0], allv[:, 1])
    assert r.max() > 32.0


def test_fillet_root_adds_volume(box_step):
    s = load_step(box_step)
    fid = biggest_face_id(s)
    base = dict(pattern="quadmesh", spacing=12, thickness=2.0, height=4,
                margin=2, taper_len=0, mapping="project")
    v0 = _volume(build_rib_implicit(s, [fid], RibParams(**base))[0])
    v1 = _volume(build_rib_implicit(
        s, [fid], RibParams(**base, fillet_root=1.5))[0])
    assert v1 > v0 + 5.0


def test_ribs_end_on_inset_boundary(box_step):
    # ribs must terminate ON the true inset boundary, not on the pixelated
    # mask edge (which reads as a serrated fringe at every rim)
    s = load_step(box_step)
    fid = biggest_face_id(s)
    p = RibParams(pattern="isogrid", spacing=10, thickness=1.6, height=4,
                  margin=3, taper_len=0, border=True, mapping="project")
    clusters, _ = build_rib_implicit(s, [fid], p)
    assert _watertight(clusters)
    v = np.vstack([c[0] for c in clusters])
    above = v[np.abs(v[:, 2]) > 0.4]           # rib material off the surface
    # box face 60x40, margin 3: inset is [3,57]x[3,37]
    assert above[:, 0].min() > 3.0 - 0.3 and above[:, 0].max() < 57.0 + 0.3
    assert above[:, 1].min() > 3.0 - 0.3 and above[:, 1].max() < 37.0 + 0.3


def test_steep_wall_gets_ribs(cyl_patch_step):
    # the surface field ribs EVERY selected face: walls the heightfield
    # engine culled past 55 degrees (the user's unribbed center face) now
    # get ribs exactly like flat faces, standing along the local normal
    s = load_step(cyl_patch_step)
    fid = biggest_face_id(s, kind="cylinder")
    p = RibParams(pattern="isogrid", spacing=12, thickness=1.6, height=3,
                  taper_len=0, mapping="project", fillet_root=1.0,
                  fillet_top=0.4)
    clusters, _ = build_rib_implicit(s, [fid], p)
    assert _watertight(clusters)

    import igl
    from server.geometry.implicit import _vertex_normals
    from server.geometry.meshing import region_meshes
    from server.geometry.ribbing import _merge_regions, _projection_frame
    regions = region_meshes(s, [fid], 0.4)
    if len(regions) > 1:
        regions = [_merge_regions(regions)]
    region = regions[0]
    ctr, axes = _projection_frame(regions)
    n_axis = np.cross(axes[:, 0], axes[:, 1])
    V = np.ascontiguousarray(
        np.column_stack([(region.vertices - ctr) @ axes,
                         (region.vertices - ctr) @ n_axis]), np.float64)
    F = np.asarray(region.triangles, np.int64)
    N = _vertex_normals(V, F)

    w = np.vstack([c[0] for c in clusters])
    X = np.ascontiguousarray(
        np.column_stack([(w - ctr) @ axes[:, 0],
                         (w - ctr) @ axes[:, 1],
                         (w - ctr) @ n_axis]))
    sqrD, I, _ = igl.point_mesh_squared_distance(X, V, F)
    # attachment invariant: every rib vertex hugs the selected surface
    reach = p.height + p.embed + 0.6 + p.fillet_root + 1.0
    assert math.sqrt(sqrD.max()) < reach, "rib material floats off the surface"
    # ...and ribbing actually happens on STEEP surface (normals > 55 deg
    # off the view axis) — the band the heightfield engine refused
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    fn = fn / np.linalg.norm(fn, axis=1)[:, None]
    steep_face = (fn @ n_axis) < math.cos(math.radians(55.0))
    on_steep = steep_face[I] & (np.sqrt(sqrD) > 0.8)
    assert int(on_steep.sum()) > 100, \
        "no rib material on the steep band — cull behavior snuck back"


@pytest.mark.slow
def test_band_acceptance(cruscotto_full_path):
    # the user's real workflow on the band pair, fillets on: one watertight
    # solid, ribs crossing the center, root blends actually adding material
    s = load_step(cruscotto_full_path)
    base = dict(pattern="isogrid", spacing=12, thickness=1.2, height=1.5,
                margin=2, taper_len=10, mapping="project")
    plain, _ = build_rib_implicit(s, [519, 580], RibParams(**base))
    filleted, reports = build_rib_implicit(
        s, [519, 580], RibParams(**base, fillet_root=1.2, fillet_top=0.4))
    assert _watertight(plain) and _watertight(filleted)
    assert len(filleted) == 1
    v = filleted[0][0]
    assert (np.abs(v[:, 2]) < 1.5).any()          # material at the z=0 seam
    assert _volume(filleted) > _volume(plain)     # root blends add material


@pytest.mark.slow
def test_no_floating_material(cruscotto_full_path):
    # structural invariant of the surface field: material only exists
    # within [slab, height] of the BODY surface.  Over window recesses and
    # past silhouettes nothing can float because s IS the distance to the
    # selected surface — the old gate existed to patch a heightfield flaw.
    s = load_step(cruscotto_full_path)
    from server.geometry.selection import grow_tangent
    from server.geometry.meshing import mesh_shape
    curved = [m for m in mesh_shape(s) if not m.is_planar]
    seed = max(curved, key=lambda m: m.area).face_id
    grown = grow_tangent(s, [seed], angle_deg=20.0)
    p = RibParams(pattern="isogrid", spacing=12, thickness=1.2, height=1.5,
                  margin=2, taper_len=10, fillet_root=1.2, fillet_top=0.4,
                  mapping="project")
    clusters, _ = build_rib_implicit(s, grown, p)
    assert len(clusters) == 1
    assert _watertight(clusters)

    import igl
    bv, bf, off = [], [], 0
    for m in mesh_shape(s, 0.3, 0.3):
        bv.append(np.asarray(m.vertices, np.float64))
        bf.append(np.asarray(m.triangles, np.int64) + off)
        off += len(m.vertices)
    BV = np.vstack(bv)
    BF = np.vstack(bf)
    v = clusters[0][0]
    sqrD, _, _ = igl.point_mesh_squared_distance(
        np.ascontiguousarray(v), BV, BF)
    reach = p.height + p.embed + 0.6 + p.fillet_root + 0.8
    assert math.sqrt(sqrD.max()) < reach, \
        f"floating material {math.sqrt(sqrD.max()):.2f}mm off the body"


def test_unfold_rejected():
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from server.geometry.step_io import save_step
    import os, tempfile
    p = os.path.join(tempfile.gettempdir(), "imp_box.step")
    save_step(BRepPrimAPI_MakeBox(20.0, 20.0, 5.0).Shape(), p)
    s = load_step(p)
    with pytest.raises(RibbingError):
        build_rib_implicit(s, [1], RibParams(mapping="unfold"))
