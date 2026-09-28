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


def test_engine_routing(box_step):
    c = TestClient(app)
    d0 = _load(c, box_step)
    big = max(d0["faces"], key=lambda f: f["area"])["id"]
    # projected mapping routes auto -> explicit surface graph
    r = c.post("/api/ribs", json={
        "face_ids": [big],
        "params": {"pattern": "quadmesh", "spacing": 12,
                   "mapping": "project"}})
    assert r.status_code == 200, r.text
    assert r.json()["engine"] == "graph"
    c.post("/api/undo")
    # implicit + unfold is a clear client error
    r = c.post("/api/ribs", json={"face_ids": [big], "engine": "implicit",
                                  "params": {"mapping": "unfold"}})
    assert r.status_code == 400
    # explicit fast + project still allowed (comparison/debugging)
    r = c.post("/api/ribs", json={
        "face_ids": [big], "engine": "fast",
        "params": {"pattern": "quadmesh", "spacing": 12,
                   "mapping": "project"}})
    assert r.status_code == 200 and r.json()["engine"] == "fast"


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


def test_grow_angle_param(box_step):
    c = TestClient(app)
    d0 = _load(c, box_step)
    seed = d0["faces"][0]["id"]
    absent = c.post("/api/grow", json={"face_ids": [seed]})
    assert absent.status_code == 200
    explicit = c.post("/api/grow", json={"face_ids": [seed], "angle_deg": 20})
    assert explicit.status_code == 200
    # absent = legacy default (20 deg)
    assert absent.json()["face_ids"] == explicit.json()["face_ids"]
    # out-of-range angles clamp to [5, 60]: 200 deg would flood across the
    # box's 90-deg creases, clamped to 60 it must stay on the seed face
    wild = c.post("/api/grow", json={"face_ids": [seed], "angle_deg": 200})
    assert wild.status_code == 200
    sixty = c.post("/api/grow", json={"face_ids": [seed], "angle_deg": 60})
    assert wild.json()["face_ids"] == sixty.json()["face_ids"]
    assert len(wild.json()["face_ids"]) == 1


def test_no_model():
    c = TestClient(app)
    from server.main import STATE
    STATE["stack"] = []
    assert c.post("/api/ribs", json={"face_ids": [1], "params": {}}).status_code == 400
    assert c.get("/api/export/step").status_code == 400
