import struct

import pytest
from fastapi.testclient import TestClient

from server.geometry.patterns import RibParams
from server.geometry.ribbing import build_rib_solids
from server.geometry.selection import grow_tangent
from server.geometry.step_io import load_step, shape_volume
from server.main import app

# external curved skin seeds, from UI inspection of the models
SEEDS = {"part1_path": 65, "part2_path": None}


@pytest.mark.slow
@pytest.mark.parametrize("path_fx,seed", [("part1_path", 65), ("part2_path", None)])
def test_cruscotto_external_skin(path_fx, seed, request):
    s = load_step(request.getfixturevalue(path_fx))
    if seed is None:
        # part2: seed from its largest freeform face
        from server.geometry.meshing import mesh_shape
        curved = [m for m in mesh_shape(s) if not m.is_planar]
        seed = max(curved, key=lambda m: m.area).face_id
    grown = grow_tangent(s, [seed], angle_deg=20.0)
    assert len(grown) >= 3
    p = RibParams(pattern="isogrid", spacing=12, thickness=1.6, height=4, margin=2)
    solids, reports = build_rib_solids(s, grown, p)
    total_segments = sum(r.segments for r in reports)
    total_lofted = sum(r.lofted for r in reports)
    assert total_lofted > 100
    # part1's curated skin builds >95%; auto-seeded regions on part2 hit
    # high-distortion zones where the artifact guards reject more ribs
    assert total_lofted >= total_segments * 0.45
    assert all(shape_volume(x) > 0 for x in solids[:20])


@pytest.mark.slow
def test_cruscotto_ui_flow_fast_engine(part1_path, tmp_path):
    """Full API flow on part1: load -> grow -> fast apply -> exports."""
    c = TestClient(app)
    r = c.post("/api/load_path", json={"path": str(part1_path)})
    assert r.status_code == 200
    d0 = r.json()

    r = c.post("/api/grow", json={"face_ids": [65], "angle_deg": 20})
    assert r.status_code == 200
    grown = r.json()["face_ids"]

    r = c.post("/api/ribs", json={"face_ids": grown, "engine": "fast",
                                  "params": {"pattern": "isogrid", "spacing": 12,
                                             "thickness": 1.6, "height": 4}})
    assert r.status_code == 200, r.text
    d1 = r.json()
    assert d1["engine"] == "fast"
    assert d1["overlay_count"] > 100

    stl = c.get("/api/export/stl")
    assert stl.status_code == 200
    ntri = struct.unpack("<I", stl.content[80:84])[0]
    assert ntri > 10000

    step = c.get("/api/export/step")
    assert step.status_code == 200
    p = tmp_path / "part1_ribbed.step"
    p.write_bytes(step.content)
    s2 = load_step(p)
    assert shape_volume(s2) > d0["volume"] * 1.005   # ribs added >0.5% volume
