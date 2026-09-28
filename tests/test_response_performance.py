"""Cached display snapshots and binary export must preserve model semantics."""
import struct

import numpy as np
import pytest
from fastapi.testclient import TestClient
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox

import server.main as api


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_binary_stl_preserves_records_across_chunk_boundary(dtype):
    # Non-contiguous inputs, reversed winding, zero-area triangles and a
    # partial final chunk exercise the real binary interchange contract.
    vertices = np.array([[0, 0, 0], [1.25, 0, 0], [0, 2.5, 0],
                         [0, 0, 3.75], [1e-8, 0, 0]], dtype=dtype)
    faces = np.tile([[0, 1, 2], [2, 1, 0], [1, 2, 3], [0, 0, 0], [0, 4, 2]],
                    (13108, 1))[::-1][:65539]
    blob = api._stl_bytes(vertices, faces)
    assert blob[:80] == bytes(80)
    assert struct.unpack_from("<I", blob, 80)[0] == len(faces)
    assert len(blob) == 84 + 50 * len(faces)
    for index in [0, 1, 2, 3, 4, 65534, 65535, 65536, 65537, 65538]:
        points = vertices[faces[index]]
        normal = np.cross(points[1] - points[0], points[2] - points[0])
        length = np.linalg.norm(normal)
        normal /= length if length >= 1e-12 else 1
        record = struct.unpack_from("<12fH", blob, 84 + 50 * index)
        np.testing.assert_array_equal(record[:3], normal.astype(np.float32))
        np.testing.assert_array_equal(record[3:12], points.astype(np.float32).ravel())
        assert record[-1] == 0
    assert api._stl_bytes(vertices, np.empty((0, 3), dtype=int)) == bytes(84)


def test_display_reuses_body_and_invalidates_overlay_after_apply_and_undo(monkeypatch):
    shape = BRepPrimAPI_MakeBox(60, 40, 8).Shape()
    first = api._load_shape(shape, "plate.step")
    counts = {"body": 0, "overlay": 0}
    mesh_shape, overlay_mesh = api.mesh_shape, api._overlay_mesh

    def mesh(*args, **kwargs):
        counts["body"] += 1
        return mesh_shape(*args, **kwargs)

    def overlay(*args, **kwargs):
        counts["overlay"] += 1
        return overlay_mesh(*args, **kwargs)

    monkeypatch.setattr(api, "mesh_shape", mesh)
    monkeypatch.setattr(api, "_overlay_mesh", overlay)
    client = TestClient(api.app)
    assert client.get("/api/model").json() == first
    vertices = np.array([[1., 1., 8.], [2., 1., 8.], [1., 2., 9.]])
    triangles = np.array([[0, 1, 2]])
    api._push(shape, [(vertices, triangles)], [])
    applied = client.get("/api/model").json()
    assert applied["faces"] == first["faces"]
    assert applied["volume"] == first["volume"]
    assert applied["overlay"] == {"positions": vertices.ravel().tolist(),
                                   "indices": [0, 1, 2]}
    assert applied["can_undo"] and applied["overlay_count"] == 1
    assert client.get("/api/model").json() == applied
    assert counts == {"body": 0, "overlay": 1}
    assert client.post("/api/undo").json() == first
    assert counts == {"body": 0, "overlay": 2}


def test_display_invalidates_body_on_exact_change_reload_and_unload():
    client = TestClient(api.app)
    shape = BRepPrimAPI_MakeBox(60, 40, 8).Shape()
    first = api._load_shape(shape, "plate.step")
    changed = BRepPrimAPI_MakeBox(60, 40, 12).Shape()
    api._push(changed, [], [])
    current = client.get("/api/model").json()
    assert current["volume"] == pytest.approx(60 * 40 * 12)
    assert current["bbox"][-1] == 12
    assert current["faces"] != first["faces"]
    assert client.post("/api/undo").json() == first
    # Even a reload of the very same CAD object must replace session metadata.
    reloaded = api._load_shape(shape, "new-name.step", fingerprint="new-fingerprint")
    assert reloaded["model_token"] != first["model_token"]
    assert reloaded["filename"] == "new-name.step"
    assert reloaded["model_fingerprint"] == "new-fingerprint"
    assert client.get("/api/model").json() == reloaded
    assert client.post("/api/unload").status_code == 200
    assert not api.STATE["frame_cache"]
    assert client.get("/api/model").status_code == 400


def test_apply_progress_stays_active_until_display_is_ready(monkeypatch):
    from server.geometry.ribbing import RibReport

    api._load_shape(BRepPrimAPI_MakeBox(60, 40, 8).Shape(), "plate.step")
    monkeypatch.setattr(api, "build_rib_graph",
                        lambda *a, **kw: ([], [RibReport(face_id=6)]))
    payload = api._mesh_payload

    def display():
        assert api.PROGRESS["active"]
        assert api.PROGRESS["stage"] == "preparing display"
        return payload()

    monkeypatch.setattr(api, "_mesh_payload", display)
    response = TestClient(api.app).post("/api/ribs", json={
        "face_ids": [6], "params": {"mapping": "surface"}})
    assert response.status_code == 200, response.text
    assert response.json()["reports"][0]["face_id"] == 6
    assert not api.PROGRESS["active"]
    assert api.PROGRESS["stage"] == "complete"
