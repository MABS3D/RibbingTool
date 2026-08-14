import struct
import time

from fastapi.testclient import TestClient

from server.main import app
from server.geometry.step_io import load_step, shape_volume


def _load(client, step_path):
    r = client.post("/api/load_path", json={"path": str(step_path)})
    assert r.status_code == 200, r.text
    return r.json()


def _curved_face(data, kind):
    return next(f["id"] for f in sorted(data["faces"], key=lambda f: -f["area"])
                if f["kind"] == kind)


def test_fast_engine_on_curved(cyl_patch_step, tmp_path):
    c = TestClient(app)
    d0 = _load(c, cyl_patch_step)
    fid = _curved_face(d0, "cylinder")

    t0 = time.perf_counter()
    r = c.post("/api/ribs", json={"face_ids": [fid], "engine": "auto",
                                  "params": {"pattern": "isogrid", "spacing": 12,
                                             "height": 3}})
    dt = time.perf_counter() - t0
    assert r.status_code == 200, r.text
    d1 = r.json()
    assert d1["engine"] == "fast"            # auto routes curved to fast
    # interconnected patterns merge into few (often one) junction-free clusters
    assert d1["overlay_count"] >= 1
    assert d1["overlay"] is not None
    assert dt < 60, f"fast apply took {dt:.1f}s"
    # body B-rep untouched
    assert abs(d1["volume"] - d0["volume"]) < 1e-6

    # STL export includes rib volume (more triangles than plain body export)
    stl = c.get("/api/export/stl")
    assert stl.status_code == 200
    ntri = struct.unpack("<I", stl.content[80:84])[0]
    assert ntri > 1000

    # faceted STEP export loads back in OCCT with volume > body volume
    step = c.get("/api/export/step")
    assert step.status_code == 200
    p = tmp_path / "fast.step"
    p.write_bytes(step.content)
    s = load_step(p)
    assert shape_volume(s) > d0["volume"] * 1.001

    # second application stacks more overlay on the same body
    r2 = c.post("/api/ribs", json={"face_ids": [fid], "engine": "fast",
                                   "params": {"pattern": "rectangular",
                                              "spacing": 15, "height": 2,
                                              "orientation_deg": 45}})
    assert r2.status_code == 200, r2.text
    assert r2.json()["overlay_count"] > d1["overlay_count"]

    # undo removes the second batch
    r3 = c.post("/api/undo")
    assert r3.status_code == 200
    assert r3.json()["overlay_count"] == d1["overlay_count"]


def test_auto_routes_planar_to_exact(box_step):
    c = TestClient(app)
    d0 = _load(c, box_step)
    big = max(d0["faces"], key=lambda f: f["area"])["id"]
    r = c.post("/api/ribs", json={"face_ids": [big], "engine": "auto",
                                  "params": {"pattern": "quadmesh", "spacing": 12}})
    assert r.status_code == 200, r.text
    d1 = r.json()
    assert d1["engine"] == "exact"
    assert d1["overlay_count"] == 0
    assert d1["volume"] > d0["volume"]
