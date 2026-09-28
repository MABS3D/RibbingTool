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


def test_box_parity_in_rotated_frame(rotated_box_step):
    # the projection frame of a skew-rotated box is a genuine rotation
    # (M != M.T).  The body gate queries the body SDF in WORLD coordinates;
    # a frame/world transform mixup sends those queries to ghost locations
    # and carves ~90% of the slab and every rib root away.  Axis-aligned
    # fixtures cannot see this (their frame is ~identity).
    s = load_step(rotated_box_step)
    fid = biggest_face_id(s)
    p = RibParams(pattern="quadmesh", spacing=12, thickness=2.0, height=4,
                  margin=2, taper_len=0, mapping="project")
    imp, reports = build_rib_implicit(s, [fid], p)
    assert _watertight(imp)
    fast, _ = build_rib_meshes(s, [fid], p)
    slab = (60 - 4.0) * (40 - 4.0) * (0.3 + 0.6 - 0.05)
    v_ribs = _volume(imp) - slab
    assert v_ribs == pytest.approx(_volume(fast), rel=0.15)


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
    sqrD, I, Cp = igl.point_mesh_squared_distance(X, V, F)
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
    # ...but NOT below the facing gate's cut (interpolated cp normal —
    # the same quantity the gate tests): the degenerate silhouette band
    # squashes the pattern into torn flaps and must stay empty
    a, b, c = V[F[I, 0]], V[F[I, 1]], V[F[I, 2]]
    v0, v1, v2 = b - a, c - a, Cp - a
    d00 = (v0 * v0).sum(1)
    d01 = (v0 * v1).sum(1)
    d11 = (v1 * v1).sum(1)
    d20 = (v2 * v0).sum(1)
    d21 = (v2 * v1).sum(1)
    den = np.clip(d00 * d11 - d01 * d01, 1e-20, None)
    w1 = (d11 * d20 - d01 * d21) / den
    w2 = (d00 * d21 - d01 * d20) / den
    w0 = 1.0 - w1 - w2
    ni = (N[F[I, 0]] * w0[:, None] + N[F[I, 1]] * w1[:, None]
          + N[F[I, 2]] * w2[:, None])
    ni /= np.clip(np.linalg.norm(ni, axis=1), 1e-14, None)[:, None]
    on_sil = (ni[:, 2] < 0.05) & (np.sqrt(sqrD) > 0.8)
    assert int(on_sil.sum()) < max(20, 0.005 * len(X)), \
        f"{int(on_sil.sum())} verts on sub-gate silhouette surface"


def test_rib_walls_smooth_on_curved(cyl_patch_step):
    # pattern sampled at the closest point of a FACETED substrate jumps
    # by ~height*dihedral at every facet edge (0.6mm at 17deg angular
    # deflection): rib walls come out wavy on curved faces.  Local plane
    # fits over 2mm patches must be sub-voxel flat (cylinder sag over a
    # 2mm chord is 0.02mm — planarity is a valid proxy here).
    from scipy.spatial import cKDTree
    s = load_step(cyl_patch_step)
    fid = biggest_face_id(s, kind="cylinder")
    p = RibParams(pattern="isogrid", spacing=12, thickness=1.6, height=3,
                  taper_len=0, mapping="project")
    clusters, _ = build_rib_implicit(s, [fid], p)
    v = np.vstack([c[0] for c in clusters])
    f = np.vstack([c[1] + off for c, off in
                   zip(clusters, np.cumsum([0] + [len(c[0]) for c in
                                                  clusters[:-1]]))])
    r = np.hypot(v[:, 0], v[:, 1])
    tn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    tl = np.clip(np.linalg.norm(tn, axis=1), 1e-14, None)
    tnu = tn / tl[:, None]
    cen = v[f].mean(1)
    cr = np.hypot(cen[:, 0], cen[:, 1])
    radial = np.abs((tnu[:, 0] * cen[:, 0] + tnu[:, 1] * cen[:, 1]) / cr)
    # wall tris: mid-height above the R=30 cylinder, normal NOT radial
    wall = (cr > 30.8) & (cr < 32.2) & (radial < 0.4)
    idx = np.nonzero(wall)[0]
    assert len(idx) > 200, "no wall population found"
    # vertex normals: restrict each plane fit to SAME-FACING vertices so
    # crown and root curvature cannot pollute the wall measurement
    vn = np.zeros_like(v)
    for k in range(3):
        np.add.at(vn, f[:, k], tn)
    vn /= np.clip(np.linalg.norm(vn, axis=1), 1e-14, None)[:, None]
    rng = np.random.default_rng(3)
    tree = cKDTree(v)
    rms = []
    for ti in rng.choice(idx, 120, replace=False):
        ids = tree.query_ball_point(cen[ti], 2.0)
        pts = v[ids]
        dev = (pts - cen[ti]) @ tnu[ti]
        m = (np.abs(dev) < 0.5) & (vn[ids] @ tnu[ti] > 0.9)
        if m.sum() < 20:
            continue
        Q = pts[m] - pts[m].mean(0)
        _, _, Vt = np.linalg.svd(Q, full_matrices=False)
        rms.append(np.sqrt(((Q @ Vt[2]) ** 2).mean()))
    med = float(np.median(rms))
    assert med < 0.03, f"rib walls wavy on curved face: median RMS {med:.3f}mm"


def test_rib_crowns_smooth_on_concave(groove_step):
    # rib tops are the s=height offset surface.  On a CONCAVE face the
    # Euclidean-distance offset of a tessellated substrate is a
    # min-of-planes — creased at every facet Voronoi wall (corduroy
    # crowns; convex faces round the creases into C1 arcs and hide the
    # defect).  The projection height coordinate must leave them smooth:
    # plane-fit sag of the R=22 crown arc over 2mm patches is 0.023mm.
    from scipy.spatial import cKDTree
    s = load_step(groove_step)
    fid = biggest_face_id(s, kind="cylinder")
    p = RibParams(pattern="isogrid", spacing=12, thickness=1.6, height=3,
                  taper_len=0, mapping="project")
    clusters, _ = build_rib_implicit(s, [fid], p)
    v = np.vstack([c[0] for c in clusters])
    f = np.vstack([c[1] + off for c, off in
                   zip(clusters, np.cumsum([0] + [len(c[0]) for c in
                                                  clusters[:-1]]))])
    tn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    tl = np.clip(np.linalg.norm(tn, axis=1), 1e-14, None)
    tnu = tn / tl[:, None]
    cen = v[f].mean(1)
    # groove cylinder axis: x-parallel line through (y=25, z=38), R=25;
    # crowns sit at ~R-height=22 from it, facing the axis
    ay = 25.0 - cen[:, 1]
    az = 38.0 - cen[:, 2]
    ad = np.hypot(ay, az)
    rad = np.column_stack([np.zeros(len(cen)), ay / ad, az / ad])
    crown = (np.abs(ad - 22.0) < 0.7) \
        & (np.einsum("ij,ij->i", tnu, rad) > 0.9)
    idx = np.nonzero(crown)[0]
    assert len(idx) > 100, "no crown population found"
    vn = np.zeros_like(v)
    for k in range(3):
        np.add.at(vn, f[:, k], tn)
    vn /= np.clip(np.linalg.norm(vn, axis=1), 1e-14, None)[:, None]
    rng = np.random.default_rng(4)
    tree = cKDTree(v)
    rms = []
    for ti in rng.choice(idx, 120, replace=False):
        ids = tree.query_ball_point(cen[ti], 2.0)
        pts = v[ids]
        dev = (pts - cen[ti]) @ tnu[ti]
        m = (np.abs(dev) < 0.5) & (vn[ids] @ tnu[ti] > 0.9)
        if m.sum() < 15:
            continue
        Q = pts[m] - pts[m].mean(0)
        _, _, Vt = np.linalg.svd(Q, full_matrices=False)
        rms.append(np.sqrt(((Q @ Vt[2]) ** 2).mean()))
    med = float(np.median(rms))
    assert med < 0.04, f"rib crowns corduroy on concave face: median RMS {med:.3f}mm"


def test_crowns_follow_smooth_surface_on_sphere(sphere_patch_step):
    # rib tops are the s=height offset: with EUCLIDEAN distance to a
    # tessellated sphere the crowns arc over every facet (flat-arc-flat
    # undulation that reads as candle wax under grazing light).  The
    # projection height h = (x-cp)·n̂ reconstructs the smooth surface to
    # first order: crown vertices must sit at a consistent radius.
    s = load_step(sphere_patch_step)
    from server.geometry.meshing import mesh_shape
    curved = [m for m in mesh_shape(s, 0.5, 0.5) if not m.is_planar]
    fid = max(curved, key=lambda m: m.area).face_id
    p = RibParams(pattern="isogrid", spacing=10, thickness=1.6, height=3,
                  taper_len=0, mapping="project")
    clusters, _ = build_rib_implicit(s, [fid], p)
    v = np.vstack([c[0] for c in clusters])
    f = np.vstack([c[1] + off for c, off in
                   zip(clusters, np.cumsum([0] + [len(c[0]) for c in
                                                  clusters[:-1]]))])
    tn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    tl = np.clip(np.linalg.norm(tn, axis=1), 1e-14, None)
    vn = np.zeros_like(v)
    for k in range(3):
        np.add.at(vn, f[:, k], tn)
    vn /= np.clip(np.linalg.norm(vn, axis=1), 1e-14, None)[:, None]
    r = np.linalg.norm(v, axis=1)
    radial = np.einsum("ij,ij->i", vn, v / r[:, None])
    crown = (r > 32.4) & (radial > 0.95)
    assert crown.sum() > 300, "no crown population found"
    rc = r[crown]
    dev = np.abs(rc - np.median(rc))
    spread = float(np.percentile(dev, 90))
    assert spread < 0.05, \
        f"crowns undulate on the sphere: p90 radius deviation {spread:.3f}mm"


def test_no_ribs_on_foldover_backside(wrap_cyl_step):
    # a selection that wraps past the silhouette (260-degree cylinder)
    # has bands facing AWAY from the projection.  The single-valued
    # pattern raster cannot parameterize them (it would paint a mirrored,
    # crest-compressed lattice); material there must fade out at the fold
    # crest instead of shredding against the evaluation window
    s = load_step(wrap_cyl_step)
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
        np.column_stack([(w - ctr) @ axes, (w - ctr) @ n_axis]))
    sqrD, I, Cp = igl.point_mesh_squared_distance(X, V, F)
    # cp facing via the face normal of the hit triangle (robust enough
    # for a band test); off-surface material must not stand on clearly
    # BACK-facing surface
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    fn = fn / np.clip(np.linalg.norm(fn, axis=1), 1e-14, None)[:, None]
    off = np.sqrt(sqrD) > 0.8
    back_area = (fn[I[off], 2] < -0.25).mean()
    assert back_area < 0.01, \
        f"{100 * back_area:.1f}% of rib material stands on back-facing surface"


def test_no_slab_wedge_in_sealed_strip(split_top_step):
    # beyond an OPEN substrate boundary the closest point snaps to the
    # edge and the side test goes negative under the edge's tangent
    # plane: a thin slab wedge floats off every panel edge into sealed
    # separator strips (bare smears on the real part's valleys).  Ribs
    # may bridge the strip; bare wedge sheets may not exist in its void.
    s = load_step(split_top_step)
    from server.geometry.meshing import mesh_shape
    top = [m.face_id for m in mesh_shape(s, 0.4, 0.3)
           if m.is_planar and len(m.vertices)
           and np.allclose(np.asarray(m.vertices)[:, 2], 10.0, atol=1e-6)]
    assert len(top) == 2, f"expected 2 split top faces, got {len(top)}"
    p = RibParams(pattern="isogrid", spacing=10, thickness=1.6, height=3,
                  margin=1, taper_len=0, mapping="project")
    clusters, _ = build_rib_implicit(s, top, p)
    v = np.vstack([c[0] for c in clusters])
    # the sealed strip spans x in [38.75, 41.25]: its void lies BELOW the
    # top plane (z=10) over the slot.  Bridge ribs cross ABOVE z~9.7
    # (embed 0.3); wedge smears hang deeper.  Allow the first 0.45mm for
    # rib embedment; anything deeper inside the slot void is leakage.
    strip = (v[:, 0] > 39.2) & (v[:, 0] < 40.8) & (v[:, 2] < 9.55) \
        & (v[:, 2] > 4.5) & (v[:, 1] > 1.0) & (v[:, 1] < 39.0)
    assert strip.sum() == 0, \
        f"{strip.sum()} wedge verts inside the sealed-strip void"


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
