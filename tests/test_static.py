from fastapi.testclient import TestClient

from server.main import app


def test_index_served():
    r = TestClient(app).get("/")
    assert r.status_code == 200 and "app.js" in r.text


def test_static_modules_served():
    c = TestClient(app)
    for path in ("/web/app.js", "/web/viewer.js",
                 "/web/vendor/three.module.js", "/web/vendor/OrbitControls.js"):
        assert c.get(path).status_code == 200, path
