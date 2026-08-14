import numpy as np
import pytest

from server.geometry.meshing import mesh_shape, region_meshes
from server.geometry.patterns import RibParams
from server.geometry.ribbing import build_rib_solids
from server.geometry.step_io import load_step, shape_volume


def _ids_by_area(shape, n=2, kind=None):
    ms = mesh_shape(shape)
    if kind:
        ms = [m for m in ms if m.surface_kind == kind]
    return [m.face_id for m in sorted(ms, key=lambda m: -m.area)[:n]]


def test_adjacent_faces_weld_into_one_region(box_step):
    s = load_step(box_step)
    ms = mesh_shape(s)
    big = max(ms, key=lambda m: m.area).face_id
    sides = [m.face_id for m in ms if m.face_id != big]
    # top + one adjacent side share an edge -> one region
    regions = region_meshes(s, [big, sides[0]])
    assert len(regions) == 1
    assert sorted(regions[0].face_ids) == sorted([big, sides[0]])


def test_disjoint_faces_split_regions(box_step):
    s = load_step(box_step)
    areas = sorted(mesh_shape(s), key=lambda m: -m.area)
    top, bottom = areas[0].face_id, areas[1].face_id  # opposite faces
    regions = region_meshes(s, [top, bottom])
    assert len(regions) == 2


def test_region_pattern_crosses_face_boundary(cyl_patch_step):
    # lateral cylinder face + adjacent planar side: coherent region ribs
    s = load_step(cyl_patch_step)
    ms = mesh_shape(s)
    cyl = next(m.face_id for m in ms if m.surface_kind == "cylinder")
    # find a planar face adjacent to the cylinder (shares welded verts)
    planar = [m.face_id for m in ms if m.is_planar]
    target = None
    for fid in planar:
        if len(region_meshes(s, [cyl, fid])) == 1:
            target = fid
            break
    assert target is not None
    solids, reports = build_rib_solids(
        s, [cyl, target], RibParams(pattern="isogrid", spacing=12, height=2.5))
    assert len(reports) == 1              # one welded region
    assert sorted(reports[0].face_ids) == sorted([cyl, target])
    assert reports[0].lofted > 10
    # ribs exist (positive volume solids)
    assert all(shape_volume(x) > 0 for x in solids[:5])


def test_full_cylinder_region_keeps_seam(full_cyl_step):
    # welding must not close the parametric seam: the lateral face of a full
    # cylinder still unrolls and takes ribs through the region path
    s = load_step(full_cyl_step)
    from server.geometry.meshing import mesh_shape as _ms
    cyl = next(m.face_id for m in _ms(s) if m.surface_kind == "cylinder")
    solids, reports = build_rib_solids(
        s, [cyl], RibParams(pattern="rectangular", spacing=15, height=2.5))
    assert reports[0].lofted > 5


def test_flatten_orientation_stable(box_step):
    # same face flattened twice gives identical coords (PCA-aligned)
    from server.geometry.flatten import flatten
    m = max(mesh_shape(load_step(box_step)), key=lambda x: x.area)
    a, b = flatten(m), flatten(m)
    assert np.allclose(a, b)
