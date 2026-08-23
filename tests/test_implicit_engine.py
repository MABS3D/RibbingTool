"""Implicit engine on real geometry: parity, coverage, performance."""
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


def test_ribs_stay_out_of_cull_band(cyl_patch_step):
    # cells culled as steep must stay rib-free beyond sub-voxel rounding:
    # ribs built over the plunging wall shear into curled tabs at every
    # rib end, and rim-strip dilation must give room on the KEPT side only
    s = load_step(cyl_patch_step)
    fid = biggest_face_id(s, kind="cylinder")
    p = RibParams(pattern="isogrid", spacing=12, thickness=1.6, height=3,
                  taper_len=0, mapping="project", fillet_root=1.0,
                  fillet_top=0.4)
    clusters, _ = build_rib_implicit(s, [fid], p)
    assert _watertight(clusters)

    from server.geometry.meshing import region_meshes
    from server.geometry.ribbing import _projection_frame, _kept_domain
    from scipy.ndimage import binary_opening, distance_transform_edt
    from server.geometry.implicit import _rasterize_depth, make_gradients
    regions = region_meshes(s, [fid], 0.4)
    ctr, axes = _projection_frame(regions)
    n_axis = np.cross(axes[:, 0], axes[:, 1])
    flat = (regions[0].vertices - ctr) @ axes
    depth = (regions[0].vertices - ctr) @ n_axis
    tris = np.asarray(regions[0].triangles, np.int64)
    e1 = flat[tris[:, 1]] - flat[tris[:, 0]]
    e2 = flat[tris[:, 2]] - flat[tris[:, 0]]
    signed2 = 0.5 * (e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0])
    v3 = regions[0].vertices
    area3 = 0.5 * np.linalg.norm(
        np.cross(v3[tris[:, 1]] - v3[tris[:, 0]],
                 v3[tris[:, 2]] - v3[tris[:, 0]]), axis=1)
    kept = np.nonzero(signed2 > 0.30 * np.clip(area3, 1e-12, None))[0]
    domain = _kept_domain(flat, tris, kept, p)
    res = float(np.clip(min(0.30, p.thickness / 4.5), 0.10, 0.5))
    cell = max(res / 2.0, 0.15)
    minx, miny, maxx, maxy = domain.bounds
    origin = (minx - 2 * cell, miny - 2 * cell)
    shape2d = (int((maxx - origin[0]) / cell) + 3,
               int((maxy - origin[1]) / cell) + 3)
    D = _rasterize_depth(flat, depth, tris, kept, cell, origin, shape2d)
    unc = D <= -1e8
    if unc.any():
        _, (ri, ci) = distance_transform_edt(unc, return_indices=True)
        D[unc] = D[ri[unc], ci[unc]]
    Gu, Gv = make_gradients(D, cell)
    steep = binary_opening(
        np.hypot(Gu, Gv) > np.tan(np.radians(55.0)), iterations=2)
    if not steep.any():
        pytest.skip("fixture has no cull band")
    into = distance_transform_edt(~steep) * cell

    w = np.vstack([c[0] for c in clusters])
    uv = np.column_stack([(w - ctr) @ axes[:, 0],
                          (w - ctr) @ axes[:, 1]])
    gi = np.clip(((uv[:, 0] - origin[0]) / cell).astype(int),
                 0, shape2d[0] - 1)
    gj = np.clip(((uv[:, 1] - origin[1]) / cell).astype(int),
                 0, shape2d[1] - 1)
    on_steep = steep[gi, gj] & (into[gi, gj] > 0.3)
    assert not on_steep.any(), \
        f"{int(on_steep.sum())} rib vertices sit inside the culled band"


@pytest.mark.slow
def test_no_material_over_unbacked_cells(cruscotto_full_path):
    # the depth map is a heightfield of the FRONT surface: over window
    # recesses (lip in front, floor far behind) and open shadow the slab
    # and ribs used to float as free-standing plates with frayed edges.
    # Cells whose backmost kept surface is far behind the frontmost (or
    # absent) must stay rib-free.
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

    from server.geometry.meshing import region_meshes
    from server.geometry.ribbing import _merge_regions, _projection_frame, \
        _kept_domain
    from scipy.ndimage import distance_transform_edt
    from server.geometry.implicit import _rasterize_depth
    regions = region_meshes(s, grown, 0.7)
    region = [_merge_regions(regions)] if len(regions) > 1 else regions
    ctr, axes = _projection_frame(region)
    n_axis = np.cross(axes[:, 0], axes[:, 1])
    flat = (region[0].vertices - ctr) @ axes
    depth = (region[0].vertices - ctr) @ n_axis
    tris = np.asarray(region[0].triangles, np.int64)
    e1 = flat[tris[:, 1]] - flat[tris[:, 0]]
    e2 = flat[tris[:, 2]] - flat[tris[:, 0]]
    signed2 = 0.5 * (e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0])
    v3 = region[0].vertices
    area3 = 0.5 * np.linalg.norm(
        np.cross(v3[tris[:, 1]] - v3[tris[:, 0]],
                 v3[tris[:, 2]] - v3[tris[:, 0]]), axis=1)
    kept = np.nonzero(signed2 > 0.30 * np.clip(area3, 1e-12, None))[0]
    domain = _kept_domain(flat, tris, kept, p)
    cell = 0.15
    minx, miny, maxx, maxy = domain.bounds
    origin = (minx - 2 * cell, miny - 2 * cell)
    shape2d = (int((maxx - origin[0]) / cell) + 3,
               int((maxy - origin[1]) / cell) + 3)
    D = _rasterize_depth(flat, depth, tris, kept, cell, origin, shape2d)
    Dback = _rasterize_depth(flat, depth, tris, kept, cell, origin,
                             shape2d, backmost=True)
    unc = D <= -1e8
    if unc.any():
        _, (ri, ci) = distance_transform_edt(unc, return_indices=True)
        D[unc] = D[ri[unc], ci[unc]]

    w = np.vstack([c[0] for c in clusters])
    uv = np.column_stack([(w - ctr) @ axes[:, 0],
                          (w - ctr) @ axes[:, 1]])
    gi = np.clip(((uv[:, 0] - origin[0]) / cell).astype(int),
                 0, shape2d[0] - 1)
    gj = np.clip(((uv[:, 1] - origin[1]) / cell).astype(int),
                 0, shape2d[1] - 1)
    unbacked = ((D - Dback) > max(2.0 * p.thickness, 3.0)) | (Dback > 1e8)
    from scipy.ndimage import binary_dilation as _bd
    into = distance_transform_edt(~unbacked) * cell
    # ribs terminate on the unbacked boundary: only sub-voxel rounding
    # spill may cross it
    deep = unbacked[gi, gj] & (into[gi, gj] > 0.3)
    assert not deep.any(), \
        f"{int(deep.sum())} rib vertices sit over unbacked cells"
    spill = unbacked[gi, gj] & (into[gi, gj] <= 0.3)
    assert spill.mean() < 0.02, \
        f"{spill.mean():.1%} of rib vertices over unbacked cells"


def test_unfold_rejected():
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from server.geometry.step_io import save_step
    import os, tempfile
    p = os.path.join(tempfile.gettempdir(), "imp_box.step")
    save_step(BRepPrimAPI_MakeBox(20.0, 20.0, 5.0).Shape(), p)
    s = load_step(p)
    with pytest.raises(RibbingError):
        build_rib_implicit(s, [1], RibParams(mapping="unfold"))
