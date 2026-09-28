"""Never publish an incomplete display mesh as an apparently successful STL."""
import struct

import manifold3d as m3d
import numpy as np
import pytest
from fastapi.testclient import TestClient
from OCP.BRep import BRep_Tool
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox

import server.main as api
from server.geometry import booleans
from server.geometry.step_io import face_map


def read_stl(blob):
    count = struct.unpack_from('<I', blob, 80)[0]
    assert len(blob) == 84+50*count
    record = np.dtype([('normal', '<f4', (3,)), ('vertices', '<f4', (3, 3)),
                       ('attribute', '<u2')])
    points = np.frombuffer(blob, dtype=record, offset=84)['vertices'].reshape(-1, 3)
    vertices, inverse = np.unique(points, axis=0, return_inverse=True)
    return m3d.Manifold(m3d.Mesh64(np.ascontiguousarray(vertices, np.float64),
                                  np.ascontiguousarray(inverse.reshape(-1, 3), np.uint64)))


def test_bare_stl_uses_closed_export_mesh_instead_of_incomplete_display_cache():
    loaded = api._load_shape(BRepPrimAPI_MakeBox(18., 12., 4.).Shape(), 'plate.step')
    # The viewer can still show a partial mesh. It is not export geometry.
    api.STATE['meshes'] = api.STATE['meshes'][:-1]
    response = TestClient(api.app).get('/api/export/stl')
    assert response.status_code == 200, response.text
    solid = read_stl(response.content)
    assert solid.status() == m3d.Error.NoError
    assert len(solid.decompose()) == 1
    assert solid.volume() == pytest.approx(18*12*4)
    assert TestClient(api.app).get('/api/model').json() == loaded


@pytest.mark.parametrize('overlay,extension', [(False, 'stl'), (True, 'stl'), (True, 'step')])
def test_missing_cad_face_aborts_export_without_modifying_model(monkeypatch, overlay, extension):
    shape = BRepPrimAPI_MakeBox(18., 12., 4.).Shape()
    loaded = api._load_shape(shape, 'plate.step')
    if overlay:
        # A healthy rib solid must not mask a missing body surface.
        api._push(shape, [BRepPrimAPI_MakeBox(2., 2., 6.).Shape()], [])
    current = TestClient(api.app).get('/api/model').json()
    missing = face_map(shape).FindKey(5)
    original = BRep_Tool.Triangulation_s

    def omit_face(face, *args, **kwargs):
        return None if face.IsSame(missing) else original(face, *args, **kwargs)

    monkeypatch.setattr(BRep_Tool, 'Triangulation_s', omit_face)
    response = TestClient(api.app).get('/api/export/'+extension)
    assert response.status_code == 400, response.text[:200]
    assert 'tessellation' in response.json()['detail']
    assert '5' in response.json()['detail']
    assert 'Content-Disposition' not in response.headers
    assert TestClient(api.app).get('/api/model').json() == current
    assert api.STATE['model_token'] == loaded['model_token']
    assert 'fine_shells' not in api._entry() and 'body_stl_mesh' not in api._entry()


def test_nonmanifold_bare_export_reports_error_and_releases_lock(monkeypatch):
    shape = BRepPrimAPI_MakeBox(18., 12., 4.).Shape()
    loaded = api._load_shape(shape, 'plate.step')
    monkeypatch.setattr(booleans, '_shape_to_mesh', lambda *a: (
        np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]), np.array([[0, 1, 2]])))
    client = TestClient(api.app)
    response = client.get('/api/export/stl')
    assert response.status_code == 400
    assert 'manifold' in response.json()['detail']
    assert client.get('/api/model').json() == loaded
    assert client.post('/api/unload').status_code == 200


def test_repeated_bare_export_reuses_validated_mesh_but_new_body_does_not(monkeypatch):
    api._load_shape(BRepPrimAPI_MakeBox(18., 12., 4.).Shape(), 'plate.step')
    client = TestClient(api.app)
    first = client.get('/api/export/stl')
    assert first.status_code == 200
    original = booleans._shape_to_mesh

    def unexpected(*args, **kwargs):
        raise AssertionError('unchanged body must not be remeshed')

    monkeypatch.setattr(booleans, '_shape_to_mesh', unexpected)
    assert client.get('/api/export/stl').content == first.content
    monkeypatch.setattr(booleans, '_shape_to_mesh', original)
    api._load_shape(BRepPrimAPI_MakeBox(18., 12., 8.).Shape(), 'taller.step')
    second = client.get('/api/export/stl')
    assert second.status_code == 200
    assert read_stl(second.content).volume() == pytest.approx(18*12*8)


def test_stl_retriangulates_precision_collapses_without_punching_holes():
    from server.geometry.export_mesh import stl_export_mesh, _closed_coordinates
    profile = m3d.CrossSection([np.array([[0, 0], [10, 0], [10, 10],
                                        [1e-6, 10], [0, 10-1e-6]])])
    original = profile.extrude(2).translate([1000, 1000, 0])
    v, f = booleans._man_to_arrays(original, None)
    before = v.copy()
    assert not _closed_coordinates(v.astype(np.float32).astype(float), f)
    q, g = stl_export_mesh(v, f)
    assert _closed_coordinates(q, g)
    assert np.array_equal(before, v)
    result = read_stl(api._stl_bytes(q, g))
    assert result.status() == m3d.Error.NoError
    assert len(result.decompose()) == 1
    assert result.volume() == pytest.approx(original.volume(), rel=1e-6)


def test_stl_rejects_loss_of_entire_component_and_open_sources():
    from server.geometry.export_mesh import stl_export_mesh
    tiny = m3d.Manifold.cube([1, 1, 1]).translate([1e8, 1e8, 1e8])
    with pytest.raises(booleans.BooleanError, match='precision'):
        stl_export_mesh(*booleans._man_to_arrays(tiny, None))
    with pytest.raises(booleans.BooleanError, match='closed manifold'):
        stl_export_mesh(np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]),
                        np.array([[0, 1, 2]]))
