"""Prepared graph reuse must preserve geometry and remain independent of edit history."""
from dataclasses import replace

import manifold3d as m3d
import numpy as np
import pytest
from OCP.BRep import BRep_Builder
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCP.TopoDS import TopoDS_Compound
from OCP.gp import gp_Pnt

from server.geometry import graph_ribs as gr, meshing
from server.geometry.patterns import RibParams


def params(**changes):
    return replace(RibParams(mapping='surface', spacing=6, thickness=1.6,
        height=3, margin=1, taper_len=0, fillet_root=.6, fillet_top=.3,
        fillet_junction=.5), **changes)


def plate():
    return BRepPrimAPI_MakeBox(18., 12., 4.).Shape()


def same_paths(a, b):
    assert len(a['paths']) == len(b['paths'])
    for pa, pb in zip(a['paths'], b['paths']):
        for key in ('points', 'normals', 'height'):
            np.testing.assert_array_equal(pa[key], pb[key])


def test_preview_apply_and_fine_share_curves_and_spatial_indices(monkeypatch):
    shape, cache, p = plate(), {}, params()
    prepared = gr.prepare_rib_graph(shape, [6, 6], p, frame_cache=cache)
    clusters, reports = gr.build_rib_graph(shape, [6], p, frame_cache=cache)
    assert reports[0].warnings

    def unexpected(*args, **kwargs):
        raise AssertionError('An unchanged recipe must reuse its prepared geometry and BVHs')

    monkeypatch.setattr(meshing, 'region_meshes', unexpected)
    monkeypatch.setattr(meshing, 'mesh_shape', unexpected)
    monkeypatch.setattr(gr.MeshDistance, '__init__', unexpected)
    fine, fine_reports = gr.build_rib_graph(shape, [6], p, quality=3, frame_cache=cache)
    again = gr.prepare_rib_graph(shape, [6], p, frame_cache=cache)
    same_paths(prepared, again)
    assert again['reports'][0].warnings == []
    assert len(fine_reports[0].warnings) == 1
    for output in (clusters, fine):
        v, f = output[0]
        solid = m3d.Manifold(m3d.Mesh64(np.ascontiguousarray(v), np.ascontiguousarray(f, np.uint64)))
        assert solid.status() == m3d.Error.NoError
        assert len(solid.decompose()) == 1
    assert len(fine[0][1]) > len(clusters[0][1])


def test_parameter_edits_reuse_cad_but_not_stale_curves(monkeypatch):
    shape, cache = plate(), {}
    first = gr.prepare_rib_graph(shape, [6], params(), frame_cache=cache)

    def unexpected(*args, **kwargs):
        raise AssertionError('Parameter edits should not remesh the same CAD selection')

    monkeypatch.setattr(meshing, 'region_meshes', unexpected)
    height = gr.prepare_rib_graph(shape, [6], params(height=6), frame_cache=cache)
    assert len(height['paths']) == len(first['paths'])
    for a, b in zip(first['paths'], height['paths']):
        np.testing.assert_array_equal(a['points'], b['points'])
        np.testing.assert_allclose(b['height'], a['height']*2)
    spacing = gr.prepare_rib_graph(shape, [6], params(spacing=4), frame_cache=cache)
    assert len(spacing['paths']) > len(first['paths'])


def test_graph_recipe_is_independent_of_previous_selection_and_legacy_frame():
    builder = BRep_Builder()
    shape = TopoDS_Compound(); builder.MakeCompound(shape)
    builder.Add(shape, BRepPrimAPI_MakeBox(18., 12., 4.).Shape())
    builder.Add(shape, BRepPrimAPI_MakeBox(gp_Pnt(103., 7., 0.), 18., 12., 4.).Shape())
    p, cache = params(mapping='project'), {}
    first = gr.prepare_rib_graph(shape, [6], p, frame_cache=cache)
    # Legacy projected engines may still retain a model-wide lattice frame.
    cache['frame'] = (first['center'], first['matrix'][:, :2])
    second = gr.prepare_rib_graph(shape, [12], p, frame_cache=cache)
    fresh = gr.prepare_rib_graph(shape, [12], p, frame_cache={})
    np.testing.assert_array_equal(second['center'], fresh['center'])
    np.testing.assert_array_equal(second['matrix'], fresh['matrix'])
    same_paths(second, fresh)


def test_shape_identity_selection_and_mutable_reports_do_not_alias():
    cache, p = {}, params()
    shape = plate()
    first = gr.prepare_rib_graph(shape, [6], p, frame_cache=cache)
    first['reports'][0].warnings.append('not a cached warning')
    first['paths'][0]['height'] = np.full(1, 999.)
    first['warnings'].append('not a cached warning')
    again = gr.prepare_rib_graph(shape, [6], p, frame_cache=cache)
    assert not again['reports'][0].warnings and not again['warnings']
    assert np.max(again['paths'][0]['height']) == p.height
    with pytest.raises(ValueError):
        again['vertices'][0, 0] += 100
    other = BRepPrimAPI_MakeBox(gp_Pnt(100., 0., 0.), 18., 12., 4.).Shape()
    translated = gr.prepare_rib_graph(other, [6], p, frame_cache=cache)
    np.testing.assert_allclose(translated['center']-again['center'], [100, 0, 0])
    bottom = gr.prepare_rib_graph(shape, [5], p, frame_cache=cache)
    assert not np.array_equal(bottom['matrix'], again['matrix'])


def test_bounded_cache_eviction_recomputes_identical_geometry(monkeypatch):
    shape, cache = plate(), {}
    monkeypatch.setattr(gr, '_GRAPH_CACHE_ENTRIES', 2)
    first = gr.prepare_rib_graph(shape, [6], params(), frame_cache=cache)
    for angle in (13., 27., 41., 53.):
        gr.prepare_rib_graph(shape, [6], params(orientation_deg=angle), frame_cache=cache)
    assert len(cache['_graph_prepared']) <= 2
    restored = gr.prepare_rib_graph(shape, [6], params(), frame_cache=cache)
    same_paths(first, restored)


def test_affine_parallel_ribbon_field_matches_general_interpolation():
    from scipy.spatial.transform import Rotation
    rot = Rotation.from_euler('xyz', [21, 37, -13], degrees=True).as_matrix()
    v = np.array([[-12., -9, 0], [12, -9, 0], [12, 9, 0], [-12, 9, 0]])
    f = np.array([[0, 1, 2], [0, 2, 3]])
    n = np.tile([0., 0, 1], (4, 1))
    p = params(taper_len=4, draft_deg=7)
    paths = gr.trace_rib_paths(v, f, n, [((-15., 0), (15., 0))], p)
    for path in paths:
        path['points'] = path['points']@rot.T
        path['normals'] = path['normals']@rot.T
    ribbons = gr.ribbons_from_paths(paths, p)
    q = np.random.default_rng(42).uniform([-14, -2, -3], [14, 2, 6], (4000, 3))@rot.T
    for ribbon in ribbons:
        assert ribbon.parallel_normal is not None
        affine = ribbon.field(q, .8, .3, .12, height=3)
        ribbon.parallel_normal = None
        general = ribbon.field(q, .8, .3, .12, height=3)
        np.testing.assert_allclose(affine, general, rtol=0, atol=2e-10)


def test_cad_source_is_not_inherited_from_previous_tessellation():
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeCylinder
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    shape = BRepPrimAPI_MakeCylinder(8., 12., np.deg2rad(120.)).Shape()
    reference = gr.prepare_rib_graph(shape, [1], params(), frame_cache={})
    BRepMesh_IncrementalMesh(shape, .02, False, .02, True)
    warm = gr.prepare_rib_graph(shape, [1], params(), frame_cache={})
    np.testing.assert_array_equal(reference['vertices'], warm['vertices'])
    same_paths(reference, warm)


def test_moving_or_reversing_the_same_shape_wrapper_invalidates_its_recipe():
    from OCP.gp import gp_Trsf, gp_Vec
    from OCP.TopLoc import TopLoc_Location
    shape, cache, p = plate(), {}, params(mapping='project')
    first = gr.prepare_rib_graph(shape, [6], p, frame_cache=cache)
    transform = gp_Trsf(); transform.SetTranslation(gp_Vec(100., 0., 0.))
    shape.Move(TopLoc_Location(transform))
    moved = gr.prepare_rib_graph(shape, [6], p, frame_cache=cache)
    np.testing.assert_allclose(moved['center']-first['center'], [100, 0, 0])
    shape.Reverse()
    reversed_graph = gr.prepare_rib_graph(shape, [6], p, frame_cache=cache)
    assert reversed_graph['matrix'][:, 2] @ moved['matrix'][:, 2] == pytest.approx(-1.)


@pytest.mark.parametrize('margin', [1., 1.-4e-10, 1.+4e-10])
def test_clearance_crossing_does_not_create_degenerate_ribbon_segments(margin):
    v = np.array([[-12., -9, 0], [12, -9, 0], [12, 9, 0], [-12, 9, 0]])
    f = np.array([[0, 1, 2], [0, 2, 3]])
    n = np.tile([0., 0, 1], (4, 1))
    p = params(margin=margin, taper_len=4)
    [path] = gr.trace_rib_paths(v, f, n, [((-15., 0), (15., 0))], p, step=.3)
    lengths = np.linalg.norm(np.diff(path['points'], axis=0), axis=1)
    assert np.all(lengths > 1e-9)
    clearance = p.margin+p.thickness/2
    np.testing.assert_allclose(path['points'][[0, -1], 0], [-12+clearance, 12-clearance], atol=1e-8, rtol=0)
    [ribbon] = gr.ribbons_from_paths([path], p)
    assert np.all(ribbon.mesh.nondegenerate)


def test_selected_faces_cannot_disappear_during_tessellation_or_cleanup(monkeypatch):
    from server.geometry.ribbing import RibbingError
    original = meshing.region_meshes

    def omit_top(*args, **kwargs):
        regions = original(*args, **kwargs)
        for region in regions:
            keep = region.tri_face != 6
            region.triangles = region.triangles[keep]
            region.tri_face = region.tri_face[keep]
            region.wedge_uvs = region.wedge_uvs[keep]
        # Leave face_ids untouched, as region metadata is not proof that
        # any usable triangles survived for every selected CAD face.
        return regions

    monkeypatch.setattr(meshing, 'region_meshes', omit_top)
    cache = {}
    with pytest.raises(RibbingError, match=r'selected faces 6 have no usable triangles'):
        gr.prepare_rib_graph(plate(), [5, 6], params(), frame_cache=cache)
    assert not cache['_graph_sources'] and not cache['_graph_prepared']


@pytest.mark.parametrize('empty', [False, True])
def test_body_tessellation_cannot_omit_an_excluded_face(monkeypatch, empty):
    from server.geometry.ribbing import RibbingError
    original = meshing.mesh_shape

    def omit_bottom(*args, **kwargs):
        meshes = original(*args, **kwargs)
        if empty:
            for mesh in meshes:
                if mesh.face_id == 5:
                    mesh.triangles = np.empty((0, 3), dtype=np.int32)
            return meshes
        return [mesh for mesh in meshes if mesh.face_id != 5]

    monkeypatch.setattr(meshing, 'mesh_shape', omit_bottom)
    shape, cache = plate(), {}
    # The selected top is healthy; the missing bottom must still stop the
    # body distance field before any mesh extraction or attachment runs.
    with pytest.raises(RibbingError, match=r'body faces 5 have no usable triangles'):
        gr.build_rib_graph(shape, [6], params(), frame_cache=cache)
    assert not cache['_graph_body_sources']
    monkeypatch.setattr(meshing, 'mesh_shape', original)
    _, _, face_ids = gr._body_source(shape, .15, cache)
    assert set(face_ids) == set(range(1, 7))


@pytest.mark.slow
def test_real_dashboard_assembly_recovers_complete_body_without_changing_cad():
    import os
    from pathlib import Path
    from server.geometry.step_io import load_step, face_map, shape_volume
    from server.geometry.booleans import _to_manifold
    from OCP.BRepCheck import BRepCheck_Analyzer
    path = Path(os.environ.get('RIBBING_ASSIEME_STEP',
                Path.home()/'Downloads'/'180 90 cruscotto_Assieme.step'))
    if not path.exists():
        pytest.skip('Set RIBBING_ASSIEME_STEP to the dashboard assembly STEP')
    shape, cache = load_step(path), {}
    before, nfaces = shape_volume(shape), face_map(shape).Size()
    vertices, triangles, ids = gr._body_source(shape, .15, cache)
    assert set(ids) == set(range(1, nfaces+1))
    assert np.isfinite(vertices).all() and len(triangles)
    # Export must be a valid union of the two closed original solids.
    body = _to_manifold(shape, .2)
    assert body.status() == m3d.Error.NoError and len(body.decompose()) == 1
    assert body.volume() == pytest.approx(before, rel=.005)
    assert face_map(shape).Size() == nfaces
    assert shape_volume(shape) == pytest.approx(before, rel=1e-12)
    assert BRepCheck_Analyzer(shape).IsValid()
