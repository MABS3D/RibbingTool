import base64
import io
import hashlib
import struct
import tempfile
import time
import threading
import uuid
from contextlib import contextmanager
from functools import wraps
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .geometry.booleans import BooleanError, _shape_to_mesh, _to_manifold, _man_to_arrays, mesh_union
from .geometry.export_mesh import stl_export_mesh
from .geometry.flatten import FlattenError
from .geometry.implicit import build_rib_implicit
from .geometry.graph_ribs import build_rib_graph, preview_rib_graph
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
STATE = {"stack": [], "filename": None, "meshes": None, "frame_cache": {},
         "model_token": None, "model_fingerprint": None}

# FastAPI runs synchronous handlers in a thread pool. Mapping previews and
# model operations share OCCT objects and caches, so never run them together.
MODEL_LOCK = threading.Lock()


@contextmanager
def _model_access():
    if not MODEL_LOCK.acquire(blocking=False):
        raise HTTPException(409, "model busy; wait for the current operation to finish")
    try:
        yield
    finally:
        MODEL_LOCK.release()


def _model_operation(fn):
    @wraps(fn)
    def guarded(*args, **kwargs):
        with _model_access():
            return fn(*args, **kwargs)
    return guarded

# Progress remains readable while the model lock is held by a build.
PROGRESS = {"stage": "", "done": 0, "total": 0, "t0": 0.0, "active": False}


class RibsRequest(BaseModel):
    face_ids: list[int]
    params: dict = {}
    engine: str = "auto"          # auto | graph | implicit | fast | exact
    model_token: str | None = None


class LoadPathRequest(BaseModel):
    path: str


class UnloadRequest(BaseModel):
    model_token: str | None = None


class GrowRequest(BaseModel):
    face_ids: list[int]
    angle_deg: float = 20.0


def _entry():
    return STATE["stack"][-1]


def _request_params(req):
    if not STATE["stack"]:
        raise HTTPException(400, "no model loaded")
    if req.model_token is not None and req.model_token != STATE["model_token"]:
        raise HTTPException(409, "the model changed; reload it before continuing")
    if not req.face_ids:
        raise HTTPException(400, "no faces selected")
    valid = {m.face_id for m in (STATE["meshes"] or [])}
    if set(req.face_ids) - valid:
        raise HTTPException(400, "selection contains faces outside the current model")
    try:
        return RibParams.from_dict(req.params)
    except (ValueError, TypeError) as e:
        raise HTTPException(400, f"invalid parameters: {e}") from e


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
    # Graph applies leave the CAD body unchanged. Keep one display snapshot
    # per body instead of re-tessellating/flattening it after every apply.
    # Holding the object itself avoids identity reuse after a model reload.
    cache = STATE["frame_cache"]
    body = cache.get("display_body")
    if body is None or body["shape"] is not shape:
        meshes = mesh_shape(shape, 0.5, 0.5)
        faces = [{
            "id": m.face_id,
            "kind": m.surface_kind,
            "planar": m.is_planar,
            "area": float(m.area),
            "positions": np.round(m.vertices, 4).ravel().tolist(),
            "indices": m.triangles.ravel().tolist(),
        } for m in meshes]
        if meshes:
            allv = np.vstack([m.vertices for m in meshes])
            bbox = [*allv.min(0).tolist(), *allv.max(0).tolist()]
        else:
            bbox = [0, 0, 0, 0, 0, 0]
        body = {"shape": shape, "meshes": meshes, "faces": faces,
                "bbox": bbox, "volume": float(shape_volume(shape))}
        cache["display_body"] = body
    STATE["meshes"] = body["meshes"]
    # Stack entries are immutable geometry snapshots. Retain only the current
    # overlay payload, so ten undo states do not also retain ten JSON meshes.
    display = cache.get("display_overlay")
    if display is None or display["entry"] is not entry:
        display = {"entry": entry, "overlay": _overlay_mesh(entry["overlay"])}
        cache["display_overlay"] = display
    return {
        "filename": STATE["filename"],
        "model_token": STATE["model_token"],
        "model_fingerprint": STATE["model_fingerprint"],
        "nfaces": len(body["faces"]),
        "faces": body["faces"],
        "volume": body["volume"],
        "bbox": body["bbox"],
        "can_undo": len(STATE["stack"]) > 1,
        "overlay": display["overlay"],
        "overlay_count": len(entry["overlay"]),
    }


def _push(shape, overlay, recipes):
    STATE["stack"].append({"shape": shape, "overlay": overlay,
                           "recipes": recipes})
    if len(STATE["stack"]) > 10:
        STATE["stack"] = STATE["stack"][:1] + STATE["stack"][-9:]


def _load_shape(shape, filename, fingerprint=None):
    STATE["stack"] = [{"shape": shape, "overlay": [], "recipes": []}]
    STATE["filename"] = filename
    STATE["frame_cache"] = {}
    STATE["model_token"] = uuid.uuid4().hex
    STATE["model_fingerprint"] = fingerprint
    return _mesh_payload()


@app.post("/api/load")
async def api_load(file: UploadFile = File(...)):
    data = await file.read()
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".step", delete=False) as tf:
            tf.write(data)
            tmp = tf.name
        with _model_access():
            shape = load_step(tmp)
            payload = _load_shape(shape, file.filename, hashlib.sha256(data).hexdigest())
    except StepError as e:
        raise HTTPException(400, str(e))
    finally:
        if tmp:
            Path(tmp).unlink(missing_ok=True)
    return JSONResponse(payload)


@app.post("/api/load_path")
@_model_operation
def api_load_path(req: LoadPathRequest):
    """Load a STEP file from a local path (local single-user tool)."""
    try:
        shape = load_step(req.path)
    except StepError as e:
        raise HTTPException(400, str(e))
    return JSONResponse(_load_shape(
        shape, Path(req.path).name,
        hashlib.sha256(Path(req.path).read_bytes()).hexdigest()))


@app.get("/api/model")
@_model_operation
def api_model():
    if not STATE["stack"]:
        raise HTTPException(400, "no model loaded")
    return JSONResponse(_mesh_payload())


@app.post("/api/unload")
@_model_operation
def api_unload(req: UnloadRequest | None = None):
    if req is not None and req.model_token is not None and req.model_token != STATE["model_token"]:
        raise HTTPException(409, "the model changed; reload it before closing")
    STATE.update(stack=[], filename=None, meshes=None, frame_cache={},
                 model_token=None, model_fingerprint=None)
    PROGRESS.update(stage="", done=0, total=0, t0=0.0, active=False)
    return {"unloaded": True}


@app.post("/api/grow")
@_model_operation
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


@app.post("/api/mapping/preview")
@_model_operation
def api_mapping_preview(req: RibsRequest):
    params = _request_params(req)
    if req.engine not in ("auto", "graph") or params.mapping not in ("surface", "project"):
        raise HTTPException(400, "mapping preview requires Surface or Project with auto or surface graph engine")
    try:
        payload = preview_rib_graph(_entry()["shape"], req.face_ids, params,
                                    frame_cache=STATE["frame_cache"])
    except (RibbingError, FlattenError, BooleanError, StepError, ValueError, TypeError) as e:
        raise HTTPException(400, str(e)) from e
    payload["model_token"] = STATE["model_token"]
    return JSONResponse(payload)


@app.post("/api/ribs")
@_model_operation
def api_ribs(req: RibsRequest):
    params = _request_params(req)
    entry = _entry()

    engine = req.engine
    if engine not in ("auto", "graph", "implicit", "fast", "exact"):
        raise HTTPException(400, f"unknown engine: {engine}")
    if params.mapping not in ("surface", "project", "unfold"):
        raise HTTPException(400, f"unknown mapping: {params.mapping}")
    if params.mapping == "surface" and engine not in ("auto", "graph"):
        raise HTTPException(400, "surface mapping requires auto or surface graph engine")
    if params.mapping_controls and (params.mapping != "surface" or engine not in ("auto", "graph")):
        raise HTTPException(400, "local mapping controls require Surface with auto or surface graph engine")
    if engine == "auto":
        if params.mapping in ("surface", "project"):
            engine = "graph"
        else:
            by_id = {m.face_id: m for m in (STATE["meshes"] or [])}
            all_planar = all(by_id[f].is_planar
                             for f in req.face_ids if f in by_id)
            engine = "exact" if all_planar else "fast"

    PROGRESS.update(stage="starting", done=0, total=0, t0=time.time(),
                    active=True)

    def _report(stage, done, total):
        PROGRESS["stage"] = stage
        PROGRESS["done"] = done
        PROGRESS["total"] = total

    try:
        if engine == "exact":
            solids, reports = build_rib_solids(entry["shape"], req.face_ids,
                                               params, stagger=True)
            welded = fuse_into(entry["shape"], entry["overlay"] + solids, reports)
            _push(welded, [], [])
        else:
            build = {"graph": build_rib_graph, "implicit": build_rib_implicit,
                     "fast": build_rib_meshes}[engine]
            kwargs = {"frame_cache": STATE["frame_cache"]}
            if engine in ("graph", "implicit"):
                kwargs["progress"] = _report    # only the field engine reports
            clusters, reports = build(entry["shape"], req.face_ids, params,
                                      **kwargs)
            _push(entry["shape"], entry["overlay"] + clusters,
                  entry["recipes"] + [{"face_ids": list(req.face_ids),
                                       "params": dict(req.params),
                                       "engine": engine}])
        _report("preparing display", 0, 1)
        payload = _mesh_payload()
        payload["reports"] = [vars(r) for r in reports]
        payload["engine"] = engine
        # All fields are already JSON types. Avoid FastAPI recursively copying
        # millions of coordinates through jsonable_encoder a second time.
        response = JSONResponse(payload)
        _report("complete", 1, 1)
        return response
    except (RibbingError, FlattenError, BooleanError, StepError, ValueError, TypeError) as e:
        raise HTTPException(400, str(e))
    finally:
        PROGRESS["active"] = False


@app.get("/api/apply_progress")
def api_apply_progress():
    """Progress of the apply in flight (the UI polls this at 2 Hz)."""
    out = dict(PROGRESS)
    out["elapsed"] = (round(time.time() - out["t0"], 1) if out["active"]
                      else 0.0)
    return out


@app.post("/api/undo")
@_model_operation
def api_undo():
    if len(STATE["stack"]) < 2:
        raise HTTPException(400, "nothing to undo")
    STATE["stack"].pop()
    return JSONResponse(_mesh_payload())


def _export_name(ext):
    return (STATE["filename"] or "model").rsplit(".", 1)[0] + "_ribbed." + ext


def _stl_bytes(v, t):
    buf = io.BytesIO()
    buf.write(b"\0" * 80)
    buf.write(struct.pack("<I", len(t)))
    # Binary STL has tightly packed 50-byte little-endian records. Chunking
    # bounds working memory without a Python loop/struct.pack per triangle.
    record_type = np.dtype([("normal", "<f4", (3,)),
                            ("vertices", "<f4", (3, 3)), ("attribute", "<u2")])
    for start in range(0, len(t), 65536):
        vertices = v[t[start:start + 65536]]
        normals = np.cross(vertices[:, 1] - vertices[:, 0],
                           vertices[:, 2] - vertices[:, 0])
        lengths = np.linalg.norm(normals, axis=1, keepdims=True)
        lengths[lengths < 1e-12] = 1.0
        records = np.zeros(len(vertices), dtype=record_type)
        records["normal"] = normals / lengths
        records["vertices"] = vertices
        buf.write(records.tobytes())
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
                build = {"graph": build_rib_graph,
                         "implicit": build_rib_implicit}.get(
                             rec.get("engine"), build_rib_meshes)
                s, _ = build(entry["shape"], rec["face_ids"],
                             RibParams.from_dict(rec["params"]),
                             quality=3.0,
                             frame_cache=STATE["frame_cache"])
                fine += s
            solids = fine
        except Exception as e:
            raise HTTPException(400, f"high-quality rib rebuild failed: {e}") from e
    try:
        shells = mesh_union(entry["shape"], solids, lin_defl=0.2)
    except BooleanError as e:
        raise HTTPException(400, str(e)) from e
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
@_model_operation
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
@_model_operation
def api_export_stl():
    if not STATE["stack"]:
        raise HTTPException(400, "no model loaded")
    entry = _entry()
    if entry["overlay"]:
        v, t = _concat_shells(_union_shells())
    else:
        # Display meshes may omit faces and are intentionally coarser. Build
        # and validate the CAD body at export resolution even without ribs.
        if "body_stl_mesh" not in entry:
            try:
                entry["body_stl_mesh"] = _man_to_arrays(
                    _to_manifold(entry["shape"], 0.2), None)
            except BooleanError as e:
                raise HTTPException(400, str(e)) from e
        v, t = entry["body_stl_mesh"]
    try:
        v, t = stl_export_mesh(v, t)
    except BooleanError as e:
        raise HTTPException(400, str(e)) from e
    return Response(_stl_bytes(v, t), media_type="model/stl",
                    headers={"Content-Disposition":
                             f'attachment; filename="{_export_name("stl")}"'})


@app.get("/api/dev/last_recipe")
def api_dev_last_recipe():
    """Face ids + params of recent applies (dev/verification aid: lets a
    headless repro replay exactly what the UI did)."""
    if not STATE["stack"]:
        return {"recipes": []}
    return {"recipes": _entry()["recipes"][-3:]}


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
