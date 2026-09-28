"""Apply-progress reporting: engine callback ticks + the poll endpoint.

The callback is reporting-only: a build with a callback must be BYTE-
identical to one without, and a raising callback must never break a
build (it is logged once, then disabled).
"""
import hashlib
from itertools import groupby

import numpy as np
from fastapi.testclient import TestClient

from server.geometry.implicit import mesh_field
from server.main import app
from tests.test_implicit_kernel import CELL, ONE_RIB, _params, _surface


def _digest(clusters):
    h = hashlib.sha256()
    for v, t in clusters:
        h.update(np.ascontiguousarray(v).tobytes())
        h.update(np.ascontiguousarray(t).tobytes())
    return h.hexdigest()


def test_engine_progress_ticks_and_byte_identity():
    # tile=64 splits the 40x30mm fixture into a real multi-tile grid
    calls = []
    base = mesh_field(_surface(ONE_RIB), _params(), resolution=CELL,
                      tile=64)
    got = mesh_field(_surface(ONE_RIB), _params(), resolution=CELL,
                     tile=64,
                     progress=lambda st, d, t: calls.append((st, d, t)))
    assert _digest(got) == _digest(base)      # the callback is inert
    assert calls
    assert all(isinstance(st, str) and st for st, _, _ in calls)
    # done never regresses within a stage
    for st, grp in groupby(calls, key=lambda c: c[0]):
        ds = [d for _, d, _ in grp]
        assert ds == sorted(ds), f"{st}: done regressed: {ds}"
    tiles = [(d, t) for st, d, t in calls if st == "extracting tile"]
    assert len(tiles) > 2                     # 0-tick + one per tile
    assert tiles[0][0] == 0
    assert tiles[-1][0] == tiles[-1][1] > 1   # done == total at the end
    post = [st for st, _, t in calls if t == 0]
    assert "welding" in post and "smoothing" in post


def test_raising_callback_cannot_break_build():
    def boom(stage, done, total):
        raise RuntimeError("misbehaving UI callback")
    clusters = mesh_field(_surface(ONE_RIB), _params(), resolution=CELL,
                          progress=boom)
    assert clusters                           # build survived the callback


def test_progress_endpoint_idle():
    c = TestClient(app)
    r = c.get("/api/apply_progress")
    assert r.status_code == 200
    d = r.json()
    assert d["active"] is False
    assert set(d) >= {"stage", "done", "total", "elapsed"}


def test_progress_endpoint_after_apply(box_step):
    c = TestClient(app)
    with open(box_step, "rb") as f:
        r = c.post("/api/load",
                   files={"file": ("box.step", f, "application/step")})
    assert r.status_code == 200, r.text
    big = max(r.json()["faces"], key=lambda fc: fc["area"])["id"]
    r = c.post("/api/ribs", json={
        "face_ids": [big],
        "params": {"pattern": "quadmesh", "spacing": 12,
                   "mapping": "project"}})
    assert r.status_code == 200, r.text
    assert r.json()["engine"] == "graph"
    d = c.get("/api/apply_progress").json()
    assert d["active"] is False               # cleared in the finally
    assert d["stage"] and d["stage"] != "starting"   # the closure ran
    assert d["elapsed"] == 0.0                # idle again
