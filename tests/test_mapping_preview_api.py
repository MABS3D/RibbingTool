"""API guarantees for editing mapping before generating solid ribs.

The small plate exercises the real preview pipeline.  Expensive application
and fine export are replaced only in the recipe/locking contract tests.
"""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import hashlib
from threading import Event

import numpy as np
import pytest
from fastapi.testclient import TestClient

import server.main as api
from server.geometry.ribbing import RibReport


@pytest.fixture
def client():
    return TestClient(api.app)


def _load(client, path):
    with open(path, "rb") as stream:
        response = client.post(
            "/api/load", files={"file": ("box.step", stream, "application/step")})
    assert response.status_code == 200, response.text
    return response.json()


@pytest.fixture
def loaded(client, box_step):
    return _load(client, box_step)


def _top_face(loaded):
    horizontal = [face for face in loaded["faces"]
                  if np.ptp(np.asarray(face["positions"]).reshape(-1, 3)[:, 2]) < 1e-6]
    return max(horizontal, key=lambda face: np.mean(
        np.asarray(face["positions"]).reshape(-1, 3)[:, 2]))


def _control(face):
    vertices = np.asarray(face["positions"]).reshape(-1, 3)
    center = (vertices.min(axis=0) + vertices.max(axis=0)) / 2
    return {"id": "guide-1", "face_id": face["id"],
            "position": center.tolist(), "radius": 18.0,
            "angle_deg": 25.0, "spacing_scale": 0.8, "height_scale": 1.4}


def _request(loaded, controls=None, **params):
    body = {"face_ids": [_top_face(loaded)["id"]], "engine": "auto",
            "params": {"mapping": "surface", "pattern": "quadmesh",
                       "spacing": 12.0, "height": 4.0, **params}}
    if controls is not None:
        body["params"]["mapping_controls"] = controls
    return body


def _assert_error(response, status=400):
    assert response.status_code == status, response.text
    assert isinstance(response.json().get("detail"), str)
    assert response.json()["detail"].strip()


def _fake_preview(*args, **kwargs):
    return {"paths": [{"points": [0., 0., 8., 10., 0., 8.],
                       "top": [0., 0., 12., 10., 0., 12.]}],
            "controls": [], "stats": {"paths": 1, "points": 2,
                                       "controls": 0, "mapping_seconds": 0.0},
            "warnings": []}


def test_preview_is_repeatable_and_does_not_apply_geometry(client, loaded):
    stack = api.STATE["stack"]
    entry = stack[-1]
    overlay, recipes = entry["overlay"], entry["recipes"]
    body = _request(loaded)
    first = client.post("/api/mapping/preview", json=body)
    second = client.post("/api/mapping/preview", json=body)
    assert first.status_code == 200, first.text
    assert second.status_code == 200, second.text
    a, b = first.json(), second.json()
    assert a["paths"] and a["paths"] == b["paths"]
    assert a["controls"] == b["controls"] == []
    assert a["stats"]["paths"] == len(a["paths"])
    count = 0
    for path in a["paths"]:
        points = np.asarray(path["points"]).reshape(-1, 3)
        top = np.asarray(path["top"]).reshape(-1, 3)
        assert len(points) >= 2
        assert points.shape == top.shape
        assert np.isfinite(points).all() and np.isfinite(top).all()
        count += len(points)
    assert a["stats"]["points"] == count
    assert a["stats"]["controls"] == 0
    assert a["stats"]["mapping_seconds"] >= 0
    assert api.STATE["stack"] is stack and len(stack) == 1
    assert stack[-1] is entry
    assert entry["overlay"] is overlay and overlay == []
    assert entry["recipes"] is recipes and recipes == []
    assert client.get("/api/dev/last_recipe").json() == {"recipes": []}


def test_preview_controls_change_real_paths_without_creating_ribs(client, loaded):
    control = _control(_top_face(loaded))
    baseline = client.post("/api/mapping/preview", json=_request(loaded))
    changed = client.post("/api/mapping/preview", json=_request(loaded, [control]))
    repeated = client.post("/api/mapping/preview", json=_request(loaded, [control]))
    assert baseline.status_code == 200, baseline.text
    assert changed.status_code == repeated.status_code == 200, changed.text
    out = changed.json()
    assert out["paths"] != baseline.json()["paths"]
    assert out["paths"] == repeated.json()["paths"]
    assert len(out["controls"]) == out["stats"]["controls"] == 1
    assert len(api.STATE["stack"]) == 1
    assert api.STATE["stack"][-1]["overlay"] == []


def test_load_fingerprint_matches_file_contents_across_api_routes(client, loaded, box_step):
    fingerprint = hashlib.sha256(box_step.read_bytes()).hexdigest()
    assert loaded["model_fingerprint"] == fingerprint
    first_token = loaded["model_token"]
    current = client.get("/api/model")
    assert current.status_code == 200, current.text
    assert current.json()["model_fingerprint"] == fingerprint
    assert current.json()["model_token"] == first_token

    reloaded = client.post("/api/load_path", json={"path": str(box_step)})
    assert reloaded.status_code == 200, reloaded.text
    second = reloaded.json()
    assert second["model_fingerprint"] == fingerprint
    assert second["model_token"] != first_token
    current = client.get("/api/model")
    assert current.status_code == 200, current.text
    assert current.json()["model_fingerprint"] == fingerprint
    assert current.json()["model_token"] == second["model_token"]

    uploaded_again = _load(client, box_step)
    assert uploaded_again["model_fingerprint"] == fingerprint
    assert uploaded_again["model_token"] not in {first_token, second["model_token"]}


def test_preview_rejects_margin_that_removes_every_path_without_mutation(client, loaded):
    stack = api.STATE["stack"]
    entry = stack[-1]
    overlay, recipes = entry["overlay"], entry["recipes"]
    response = client.post("/api/mapping/preview", json=_request(loaded, margin=1000.0))
    _assert_error(response)
    assert api.STATE["stack"] is stack and len(stack) == 1
    assert stack[-1] is entry
    assert entry["overlay"] is overlay and overlay == []
    assert entry["recipes"] is recipes and recipes == []
    current = client.get("/api/model")
    assert current.status_code == 200, current.text
    assert current.json()["model_token"] == loaded["model_token"]
    assert current.json()["volume"] == loaded["volume"]
    assert current.json()["overlay_count"] == 0


@pytest.mark.parametrize("face_ids", [[], [999], [0, 999]])
def test_preview_rejects_empty_or_invalid_selection(client, loaded, face_ids):
    body = _request(loaded)
    body["face_ids"] = face_ids
    _assert_error(client.post("/api/mapping/preview", json=body))
    assert len(api.STATE["stack"]) == 1


def test_preview_requires_loaded_model(client, loaded):
    stack = api.STATE["stack"]
    try:
        api.STATE["stack"] = []
        _assert_error(client.post("/api/mapping/preview", json=_request(loaded)))
    finally:
        api.STATE["stack"] = stack


@pytest.mark.parametrize("mapping,engine", [
    ("unfold", "auto"), ("invalid", "auto"), ("surface", "exact"),
    ("surface", "fast"), ("surface", "implicit"), ("project", "fast"),
    ("surface", "invalid"),
])
def test_preview_rejects_unsupported_mapping_or_engine(client, loaded, mapping, engine):
    body = _request(loaded, mapping=mapping)
    body["engine"] = engine
    _assert_error(client.post("/api/mapping/preview", json=body))


@pytest.mark.parametrize("engine", ["auto", "graph"])
def test_project_preview_remains_available_without_local_controls(client, loaded, engine):
    body = _request(loaded, mapping="project")
    body["engine"] = engine
    response = client.post("/api/mapping/preview", json=body)
    assert response.status_code == 200, response.text
    assert response.json()["paths"]


def test_project_preview_does_not_silently_ignore_local_controls(client, loaded):
    body = _request(loaded, [_control(_top_face(loaded))], mapping="project")
    _assert_error(client.post("/api/mapping/preview", json=body))


@pytest.mark.parametrize("field,value", [
    ("radius", 0), ("radius", -1), ("radius", "large"),
    ("angle_deg", 91), ("angle_deg", -91),
    ("spacing_scale", 0.49), ("spacing_scale", 2.01),
    ("height_scale", 0.24), ("height_scale", 3.01),
    ("position", [1, 2]), ("position", ["a", 2, 8]), ("face_id", 999),
])
def test_preview_rejects_malformed_control(client, loaded, field, value):
    control = _control(_top_face(loaded))
    control[field] = value
    _assert_error(client.post("/api/mapping/preview", json=_request(loaded, [control])))
    assert len(api.STATE["stack"]) == 1


def test_preview_rejects_control_on_excluded_face(client, loaded):
    selected = _top_face(loaded)
    excluded = next(face for face in loaded["faces"] if face["id"] != selected["id"])
    control = _control(excluded)
    _assert_error(client.post("/api/mapping/preview", json=_request(loaded, [control])))


@pytest.mark.parametrize("operation", ["preview", "apply", "undo", "load", "stl", "step"])
def test_busy_preview_blocks_conflicting_operations_and_recovers(
        client, loaded, box_step, monkeypatch, operation):
    entered, release = Event(), Event()

    def blocking_preview(*args, **kwargs):
        entered.set()
        assert release.wait(10), "test did not release the preview worker"
        return _fake_preview()

    monkeypatch.setattr(api, "preview_rib_graph", blocking_preview)
    body = _request(loaded)
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(client.post, "/api/mapping/preview", json=body)
        try:
            assert entered.wait(5), "preview never reached the geometry worker"
            if operation == "preview":
                response = client.post("/api/mapping/preview", json=body)
            elif operation == "apply":
                response = client.post("/api/ribs", json=body)
            elif operation == "undo":
                response = client.post("/api/undo")
            elif operation == "load":
                response = client.post("/api/load_path", json={"path": str(box_step)})
            else:
                response = client.get(f"/api/export/{operation}")
            _assert_error(response, 409)
        finally:
            release.set()
        assert pending.result(timeout=5).status_code == 200
    recovered = client.post("/api/mapping/preview", json=body)
    assert recovered.status_code == 200, recovered.text
    assert len(api.STATE["stack"]) == 1


def test_preview_failure_releases_lock_and_preserves_model(client, loaded, monkeypatch):
    def broken_preview(*args, **kwargs):
        raise ValueError("control cannot be mapped onto this selected surface")

    monkeypatch.setattr(api, "preview_rib_graph", broken_preview)
    body = _request(loaded)
    _assert_error(client.post("/api/mapping/preview", json=body))
    monkeypatch.setattr(api, "preview_rib_graph", _fake_preview)
    response = client.post("/api/mapping/preview", json=body)
    assert response.status_code == 200, response.text
    assert len(api.STATE["stack"]) == 1


def test_new_upload_invalidates_old_preview_and_apply_token(client, loaded, box_step, monkeypatch):
    previous_token = loaded["model_token"]
    assert isinstance(previous_token, str) and previous_token
    latest = _load(client, box_step)
    assert latest["model_token"] != previous_token
    body = _request(loaded)
    body["model_token"] = previous_token
    for endpoint in ("/api/mapping/preview", "/api/ribs"):
        _assert_error(client.post(endpoint, json=body), 409)
    assert len(api.STATE["stack"]) == 1
    monkeypatch.setattr(api, "preview_rib_graph", _fake_preview)
    body["model_token"] = latest["model_token"]
    response = client.post("/api/mapping/preview", json=body)
    assert response.status_code == 200, response.text
    del body["model_token"]
    legacy = client.post("/api/mapping/preview", json=body)
    assert legacy.status_code == 200, legacy.text


def test_apply_and_fine_rebuild_preserve_mapping_control_recipe(client, loaded, monkeypatch):
    control = _control(_top_face(loaded))
    body = _request(loaded, [control])
    saved_params = deepcopy(body["params"])
    vertices = np.array([[0., 0., 8.], [1., 0., 8.],
                         [0., 1., 8.], [0., 0., 9.]])
    triangles = np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]])
    clusters = [(vertices, triangles)]
    calls = []

    def build(shape, face_ids, params, **kwargs):
        calls.append({"face_ids": list(face_ids),
                      "controls": deepcopy(params.mapping_controls),
                      "quality": kwargs.get("quality"),
                      "frame_cache": kwargs.get("frame_cache")})
        return clusters, [RibReport(face_id=face_ids[0], lofted=1)]

    monkeypatch.setattr(api, "build_rib_graph", build)
    response = client.post("/api/ribs", json=body)
    assert response.status_code == 200, response.text
    assert response.json()["engine"] == "graph"
    assert len(api.STATE["stack"]) == 2
    recipe = api.STATE["stack"][-1]["recipes"][-1]
    assert recipe["params"] == saved_params
    assert recipe["face_ids"] == body["face_ids"]
    assert calls[0]["controls"] == [control]
    entry = api.STATE["stack"][-1]
    overlay, recipes = entry["overlay"], entry["recipes"]
    preview = client.post("/api/mapping/preview", json=body)
    assert preview.status_code == 200, preview.text
    assert len(api.STATE["stack"]) == 2
    assert api.STATE["stack"][-1] is entry
    assert entry["overlay"] is overlay and entry["recipes"] is recipes
    assert recipe["params"] == saved_params
    assert len(calls) == 1
    monkeypatch.setattr(api, "mesh_union", lambda *args, **kwargs: clusters)
    assert api._union_shells() is clusters
    assert len(calls) == 2
    assert calls[1]["quality"] == 3.0
    assert calls[1]["controls"] == calls[0]["controls"]
    assert calls[1]["face_ids"] == calls[0]["face_ids"]
    assert calls[1]["frame_cache"] is calls[0]["frame_cache"]
    assert api._union_shells() is clusters  # cached export must not rebuild again
    assert len(calls) == 2
