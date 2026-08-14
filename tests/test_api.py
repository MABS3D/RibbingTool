from fastapi.testclient import TestClient

from server.main import app


def _load(client, step_path):
    with open(step_path, "rb") as f:
        r = client.post("/api/load",
                        files={"file": ("box.step", f, "application/step")})
    assert r.status_code == 200, r.text
    return r.json()


def test_load_and_mesh(box_step):
    c = TestClient(app)
    data = _load(c, box_step)
    assert data["nfaces"] == 6
    assert len(data["faces"]) == 6
    f0 = data["faces"][0]
    assert len(f0["positions"]) % 3 == 0
    assert len(f0["indices"]) % 3 == 0
    assert data["volume"] > 0
    assert len(data["bbox"]) == 6


def test_ribs_undo_export(box_step):
    c = TestClient(app)
    d0 = _load(c, box_step)
    big = max(d0["faces"], key=lambda f: f["area"])["id"]
    r = c.post("/api/ribs", json={"face_ids": [big],
                                  "params": {"pattern": "quadmesh", "spacing": 12}})
    assert r.status_code == 200, r.text
    d1 = r.json()
    assert d1["volume"] > d0["volume"]
    assert d1["reports"][0]["lofted"] > 0
    r = c.get("/api/export/step")
    assert r.status_code == 200 and len(r.content) > 1000
    r = c.get("/api/export/stl")
    assert r.status_code == 200 and len(r.content) > 84
    r = c.post("/api/undo")
    assert r.status_code == 200
    assert abs(r.json()["volume"] - d0["volume"]) < 1e-6
    r = c.post("/api/undo")
    assert r.status_code == 400


def test_load_path(box_step):
    c = TestClient(app)
    r = c.post("/api/load_path", json={"path": str(box_step)})
    assert r.status_code == 200 and r.json()["nfaces"] == 6
    r = c.post("/api/load_path", json={"path": "Z:/does/not/exist.step"})
    assert r.status_code == 400


def test_bad_face(box_step):
    c = TestClient(app)
    _load(c, box_step)
    r = c.post("/api/ribs", json={"face_ids": [999], "params": {}})
    assert r.status_code == 400
    assert "detail" in r.json()


def test_bad_params(box_step):
    c = TestClient(app)
    _load(c, box_step)
    r = c.post("/api/ribs", json={"face_ids": [1],
                                  "params": {"pattern": "nope"}})
    assert r.status_code == 400


def test_no_model():
    c = TestClient(app)
    from server.main import STATE
    STATE["stack"] = []
    assert c.post("/api/ribs", json={"face_ids": [1], "params": {}}).status_code == 400
    assert c.get("/api/export/step").status_code == 400
