import base64
import io
import struct
import tempfile
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .geometry.booleans import BooleanError, _shape_to_mesh, mesh_union
from .geometry.flatten import FlattenError
from .geometry.meshing import mesh_shape
from .geometry.patterns import RibParams
from .geometry.ribbing import (
    RibbingError,
    build_rib_meshes,
    build_rib_solids,
    fuse_into,
)
from .geometry.step_io import (
    StepError,
    load_step,
    save_step,
    shape_volume,
    write_faceted_step,
)

app = FastAPI(title="RibbingTool")


@app.middleware("http")
async def _no_stale_statics(request, call_next):
    # a stale cached app.js/viewer.js silently ignores new features
    response = await call_next(request)
    if request.url.path == "/" or request.url.path.startswith("/web"):
        response.headers["Cache-Control"] = "no-cache"
    return response
WEB = Path(__file__).resolve().parents[1] / "web"

# Stack of model states. Each entry:
#   shape:   B-rep body (never modified by the fast engine)
#   overlay: list of un-fused rib solids sitting on the body (fast engine)
# frame_cache keeps projected applies on one lattice frame per model.
STATE = {"stack": [], "filename": None, "meshes": None, "frame_cache": {}}


class RibsRequest(BaseModel):
    face_ids: list[int]
    params: dict = {}
    engine: str = "auto"          # auto | fast | exact


class LoadPathRequest(BaseModel):
    path: str


class GrowRequest(BaseModel):
    face_ids: list[int]
    angle_deg: float = 20.0


def _entry():
    return STATE["stack"][-1]


def _overlay_mesh(overlay, lin_defl=0.35):
    """Rib overlay as one display mesh.

    Cluster meshes (verts, tris) are disjoint and junction-free by
    construction — plain concatenation. Legacy OCCT solids are meshed.
    """
    if not overlay:
        return None
    vs, ts, off = [], [], 0
    for s in overlay:
        if isinstance(s, tuple):
            v, t = s
        else:
            v, t = _shape_to_mesh(s, lin_defl)
        if len(v) == 0:
            continue
        vs.append(np.asarray(v, np.float64))
        ts.append(np.asarray(t, np.int64) + off)
        off += len(v)
    if not vs:
        return None
    v = np.vstack(vs)
    t = np.vstack(ts)
    return {"positions": np.round(v, 4).ravel().tolist(),
            "indices": t.ravel().tolist()}


def _mesh_payload():
    entry = _entry()
    shape = entry["shape"]
    meshes = mesh_shape(shape, 0.5, 0.5)
    STATE["meshes"] = meshes
    faces = []
    for m in meshes:
        faces.append({
            "id": m.face_id,
            "kind": m.surface_kind,
            "planar": m.is_planar,
            "area": float(m.area),
            "positions": np.round(m.vertices, 4).ravel().tolist(),
            "indices": m.triangles.ravel().tolist(),
        })
    if meshes:
        allv = np.vstack([m.vertices for m in meshes])
        bbox = [*allv.min(0).tolist(), *allv.max(0).tolist()]
    else:
        bbox = [0, 0, 0, 0, 0, 0]
    return {
        "filename": STATE["filename"],
        "nfaces": len(faces),
        "faces": faces,
        "volume": float(shape_volume(shape)),
        "bbox": bbox,
        "can_undo": len(STATE["stack"]) > 1,
        "overlay": _overlay_mesh(entry["overlay"]),
        "overlay_count": len(entry["overlay"]),
    }


def _push(shape, overlay, recipes):
    STATE["stack"].append({"shape": shape, "overlay": overlay,
                           "recipes": recipes})
    if len(STATE["stack"]) > 10:
        STATE["stack"] = STATE["stack"][:1] + STATE["stack"][-9:]


def _load_shape(shape, filename):
    STATE["stack"] = [{"shape": shape, "overlay": [], "recipes": []}]
    STATE["filename"] = filename
    STATE["frame_cache"] = {}
    return _mesh_payload()


@app.post("/api/load")
async def api_load(file: UploadFile = File(...)):
    data = await file.read()
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".step", delete=False) as tf:
            tf.write(data)
            tmp = tf.name
        shape = load_step(tmp)
    except StepError as e:
        raise HTTPException(400, str(e))
    finally:
        if tmp:
            Path(tmp).unlink(missing_ok=True)
    return _load_shape(shape, file.filename)


@app.post("/api/load_path")
def api_load_path(req: LoadPathRequest):
    """Load a STEP file from a local path (local single-user tool)."""
    try:
        shape = load_step(req.path)
    except StepError as e:
        raise HTTPException(400, str(e))
    return _load_shape(shape, Path(req.path).name)


@app.post("/api/grow")
def api_grow(req: GrowRequest):
    if not STATE["stack"]:
        raise HTTPException(400, "no model loaded")
    if not req.face_ids:
        raise HTTPException(400, "no faces selected")
    from .geometry.selection import grow_tangent
    try:
        grown = grow_tangent(_entry()["shape"], req.face_ids,
                             angle_deg=max(5.0, min(60.0, req.angle_deg)))
    except Exception as e:
        raise HTTPException(400, f"selection growth failed: {e}")
    return {"face_ids": grown}


@app.post("/api/ribs")
def api_ribs(req: RibsRequest):
    if not STATE["stack"]:
        raise HTTPException(400, "no model loaded")
    entry = _entry()
    params = RibParams.from_dict(req.params)

    engine = req.engine
    if engine not in ("auto", "fast", "exact"):
        raise HTTPException(400, f"unknown engine: {engine}")
    if engine == "auto":
        by_id = {m.face_id: m for m in (STATE["meshes"] or [])}
        all_planar = all(by_id[f].is_planar for f in req.face_ids if f in by_id)
        # projected mapping only exists in the fast engine
        engine = ("exact" if all_planar and params.mapping != "project"
                  else "fast")

    try:
        if engine == "exact":
            solids, reports = build_rib_solids(entry["shape"], req.face_ids,
                                               params, stagger=True)
            welded = fuse_into(entry["shape"], entry["overlay"] + solids, reports)
            _push(welded, [], [])
        else:
            clusters, reports = build_rib_meshes(entry["shape"], req.face_ids,
                                                 params,
                                                 frame_cache=STATE["frame_cache"])
            _push(entry["shape"], entry["overlay"] + clusters,
                  entry["recipes"] + [{"face_ids": list(req.face_ids),
                                       "params": dict(req.params)}])
    except (RibbingError, FlattenError, BooleanError, StepError, ValueError) as e:
        raise HTTPException(400, str(e))

    payload = _mesh_payload()
    payload["reports"] = [vars(r) for r in reports]
    payload["engine"] = engine
    return payload


@app.post("/api/undo")
def api_undo():
    if len(STATE["stack"]) < 2:
        raise HTTPException(400, "nothing to undo")
    STATE["stack"].pop()
    return _mesh_payload()


def _export_name(ext):
    return (STATE["filename"] or "model").rsplit(".", 1)[0] + "_ribbed." + ext


def _stl_bytes(v, t):
    buf = io.BytesIO()
    buf.write(b"\0" * 80)
    buf.write(struct.pack("<I", len(t)))
    e1 = v[t[:, 1]] - v[t[:, 0]]
    e2 = v[t[:, 2]] - v[t[:, 0]]
    n = np.cross(e1, e2)
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    ln[ln < 1e-12] = 1.0
    n = n / ln
    for k, (a, b, c) in enumerate(t):
        buf.write(struct.pack("<3f", *n[k]))
        buf.write(struct.pack("<3f", *v[a]))
        buf.write(struct.pack("<3f", *v[b]))
        buf.write(struct.pack("<3f", *v[c]))
        buf.write(b"\0\0")
    return buf.getvalue()


def _union_shells():
    """Body + ribs unioned in mesh space at export quality.

    Ribs are rebuilt from their recipes at high sampling quality so facets
    drop below print resolution; the interactive overlay stays coarser.
    """
    entry = _entry()
    if "fine_shells" in entry:
        return entry["fine_shells"]
    solids = entry["overlay"]
    if entry.get("recipes"):
        try:
            fine = []
            for rec in entry["recipes"]:
                # same frame cache as the interactive applies, so projected
                # exports rebuild on the identical lattice
                s, _ = build_rib_meshes(entry["shape"], rec["face_ids"],
                                        RibParams.from_dict(rec["params"]),
                                        quality=3.0,
                                        frame_cache=STATE["frame_cache"])
                fine += s
            solids = fine
        except Exception:
            pass  # fall back to the interactive-quality overlay meshes
    shells = mesh_union(entry["shape"], solids, lin_defl=0.2)
    entry["fine_shells"] = shells
    return shells


def _concat_shells(shells):
    vs, ts, off = [], [], 0
    for v, t in shells:
        vs.append(v)
        ts.append(np.asarray(t, np.int64) + off)
        off += len(v)
    return np.vstack(vs), np.vstack(ts)


@app.get("/api/export/step")
def api_export_step():
    if not STATE["stack"]:
        raise HTTPException(400, "no model loaded")
    entry = _entry()
    out = Path(tempfile.gettempdir()) / "ribbingtool_export.step"
    if entry["overlay"]:
        shells = _union_shells()
        write_faceted_step(out, shells=shells,
                           name=(STATE["filename"] or "model").rsplit(".", 1)[0])
    else:
        save_step(entry["shape"], out)
    return FileResponse(out, filename=_export_name("step"),
                        media_type="application/step")


@app.get("/api/export/stl")
def api_export_stl():
    if not STATE["stack"]:
        raise HTTPException(400, "no model loaded")
    entry = _entry()
    if entry["overlay"]:
        v, t = _concat_shells(_union_shells())
    else:
        meshes = STATE["meshes"] or mesh_shape(entry["shape"], 0.2, 0.3)
        vs, ts, off = [], [], 0
        for m in meshes:
            vs.append(m.vertices)
            ts.append(m.triangles.astype(np.int64) + off)
            off += len(m.vertices)
        v, t = np.vstack(vs), np.vstack(ts)
    return Response(_stl_bytes(v, t), media_type="model/stl",
                    headers={"Content-Disposition":
                             f'attachment; filename="{_export_name("stl")}"'})


@app.post("/api/dev/snapshot")
async def api_dev_snapshot(request: Request):
    """Save a data-URL viewport snapshot to output/ (dev/verification aid)."""
    data = (await request.body()).decode()
    if "base64," not in data:
        raise HTTPException(400, "expected a base64 data URL")
    out = Path(__file__).resolve().parents[1] / "output"
    out.mkdir(exist_ok=True)
    p = out / "ui_snapshot.jpg"
    p.write_bytes(base64.b64decode(data.split("base64,", 1)[1]))
    return {"saved": str(p)}


@app.get("/", response_class=HTMLResponse)
def index():
    return (WEB / "index.html").read_text(encoding="utf-8")


app.mount("/web", StaticFiles(directory=str(WEB)), name="web")
