"""Closing a model releases its state without bypassing operation ownership."""
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest
from fastapi.testclient import TestClient

import server.main as api


@pytest.fixture
def client():
    return TestClient(api.app)


def _load(client, box_step):
    with box_step.open("rb") as stream:
        response = client.post(
            "/api/load", files={"file": ("box.step", stream, "application/step")})
    assert response.status_code == 200, response.text
    return response.json()


@pytest.fixture
def loaded(client, box_step):
    return _load(client, box_step)


def _assert_empty():
    assert api.STATE == {
        "stack": [], "filename": None, "meshes": None, "frame_cache": {},
        "model_token": None, "model_fingerprint": None,
    }
    assert api.PROGRESS == {
        "stage": "", "done": 0, "total": 0, "t0": 0.0, "active": False,
    }


def _assert_conflict(response):
    assert response.status_code == 409, response.text
    assert isinstance(response.json().get("detail"), str)
    assert response.json()["detail"].strip()


def _snapshot():
    # Object identity matters here: rejection must retain the loaded geometry
    # and its caches, not reconstruct a superficially equivalent model.
    return (dict(api.STATE), dict(api.PROGRESS),
            tuple(api.STATE["stack"]), dict(api.STATE["frame_cache"]))


def _assert_unchanged(snapshot):
    state, progress, entries, cache = snapshot
    assert api.STATE.keys() == state.keys()
    for key, value in state.items():
        assert api.STATE[key] is value, f"rejected unload replaced STATE[{key!r}]"
    assert len(api.STATE["stack"]) == len(entries)
    assert all(actual is expected for actual, expected in zip(api.STATE["stack"], entries))
    assert api.STATE["frame_cache"].keys() == cache.keys()
    assert all(api.STATE["frame_cache"][key] is value for key, value in cache.items())
    assert api.PROGRESS == progress


def test_unload_releases_history_meshes_recipes_caches_and_progress(client, loaded):
    entry = api.STATE["stack"][-1]
    # Represent a built model with fine export and mapping caches without
    # paying for another solid build just to verify reference cleanup.
    api.STATE["stack"].append({"shape": entry["shape"], "overlay": [object()],
                               "recipes": [{"face_ids": [loaded["faces"][0]["id"]]}],
                               "fine_shells": [object()]})
    api.STATE["frame_cache"] = {"surface_maps": {"map": object()},
                                "local_mapping": {"field": object()}}
    api.PROGRESS.update(stage="sampling", done=17, total=20, t0=123.0, active=True)
    response = client.post("/api/unload", json={"model_token": loaded["model_token"]})
    assert response.status_code == 200, response.text
    assert response.json() == {"unloaded": True}
    _assert_empty()
    assert client.get("/api/model").status_code == 400
    assert client.get("/api/export/stl").status_code == 400
    assert client.get("/api/dev/last_recipe").json() == {"recipes": []}
    assert client.get("/api/apply_progress").json() == {**api.PROGRESS, "elapsed": 0.0}


@pytest.mark.parametrize("body", [None, {}])
def test_unload_accepts_optional_body_and_is_idempotent(client, loaded, body):
    kwargs = {} if body is None else {"json": body}
    for _ in range(2):
        response = client.post("/api/unload", **kwargs)
        assert response.status_code == 200, response.text
        assert response.json() == {"unloaded": True}
        _assert_empty()


def test_unload_rejects_busy_preview_without_mutation_then_recovers(client, loaded, monkeypatch):
    entered, release = Event(), Event()

    def blocking_preview(*args, **kwargs):
        entered.set()
        assert release.wait(10), "test failed to release the preview worker"
        return {"paths": [], "controls": [], "stats": {}, "warnings": []}

    monkeypatch.setattr(api, "preview_rib_graph", blocking_preview)
    api.STATE["frame_cache"]["retained_test_map"] = object()
    api.PROGRESS.update(stage="previous operation", done=7, total=7, t0=123.0)
    request = {"face_ids": [loaded["faces"][0]["id"]],
               "params": {"mapping": "surface"}, "model_token": loaded["model_token"]}
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(client.post, "/api/mapping/preview", json=request)
        try:
            assert entered.wait(5), "preview never reached the geometry worker"
            snapshot = _snapshot()
            _assert_conflict(client.post("/api/unload", json={"model_token": loaded["model_token"]}))
            _assert_unchanged(snapshot)
            assert len(api.STATE["stack"]) == 1
        finally:
            release.set()
        assert pending.result(timeout=5).status_code == 200
    response = client.post("/api/unload", json={"model_token": loaded["model_token"]})
    assert response.status_code == 200, response.text
    _assert_empty()


def test_old_model_token_cannot_unload_a_new_upload(client, loaded, box_step):
    previous_token = loaded["model_token"]
    current = _load(client, box_step)
    assert current["model_token"] != previous_token
    snapshot = _snapshot()
    _assert_conflict(client.post("/api/unload", json={"model_token": previous_token}))
    _assert_unchanged(snapshot)
    model = client.get("/api/model")
    assert model.status_code == 200, model.text
    assert model.json()["model_token"] == current["model_token"]
    assert model.json()["volume"] == current["volume"]
    response = client.post("/api/unload", json={"model_token": current["model_token"]})
    assert response.status_code == 200, response.text
    _assert_empty()


def test_token_from_unloaded_model_is_stale_but_tokenless_retry_is_idempotent(client, loaded):
    token = loaded["model_token"]
    assert client.post("/api/unload", json={"model_token": token}).status_code == 200
    snapshot = _snapshot()
    _assert_conflict(client.post("/api/unload", json={"model_token": token}))
    _assert_unchanged(snapshot)
    _assert_empty()
    response = client.post("/api/unload")
    assert response.status_code == 200, response.text
    assert response.json() == {"unloaded": True}
    _assert_empty()
