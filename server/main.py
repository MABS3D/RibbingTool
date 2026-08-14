import io
import struct
import tempfile
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .geometry.booleans import BooleanError, _shape_to_mesh, mesh_union
from .geometry.flatten import FlattenError
from .geometry.meshing import mesh_shape
from .geometry.patterns import RibParams
from .geometry.ribbing import (
    RibbingError,
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
WEB = Path(__file__).resolve().parents[1] / "web"

# Stack of model states. Each entry:
#   shape:   B-rep body (never modified by the fast engine)
#   overlay: list of un-fused rib solids sitting on the body (fast engine)
STATE = {"stack": [], "filename": None, "meshes": None}


class RibsRequest(BaseModel):
    face_ids: list[int]
    params: dict = {}
    engine: str = "auto"          # auto | fast | exact


class LoadPathRequest(BaseModel):
    path: str


def _entry():
    return STATE["stack"][-1]


def _overlay_mesh(overlay, lin_defl=0.35):
    """Merge overlay rib solids into one display mesh (positions, indices)."""
    vs, ts, off = [], [], 0
    for s in overlay:
        v, t = _shape_to_mesh(s, lin_defl)
        if len(v) == 0:
            continue
        vs.append(v)
        ts.append(t + off)
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


def _push(shape, overlay):
    STATE["stack"].append({"shape": shape, "overlay": overlay})
    if len(STATE["stack"]) > 10:
        STATE["stack"] = STATE["stack"][:1] + STATE["stack"][-9:]


def _load_shape(shape, filename):
    STATE["stack"] = [{"shape": shape, "overlay": []}]
    STATE["filename"] = filename
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
        engine = "exact" if all_planar else "fast"

    try:
        solids, reports = build_rib_solids(entry["shape"], req.face_ids, params,
                                           stagger=(engine == "exact"))
        if engine == "exact":
            welded = fuse_into(entry["shape"], entry["overlay"] + solids, reports)
            _push(welded, [])
        else:
            _push(entry["shape"], entry["overlay"] + solids)
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


def _union_mesh():
    """Body + overlay unioned in mesh space (fine tessellation for export)."""
    entry = _entry()
    return mesh_union(entry["shape"], entry["overlay"], lin_defl=0.2)


@app.get("/api/export/step")
def api_export_step():
    if not STATE["stack"]:
        raise HTTPException(400, "no model loaded")
    entry = _entry()
    out = Path(tempfile.gettempdir()) / "ribbingtool_export.step"
    if entry["overlay"]:
        v, t = _union_mesh()
        write_faceted_step(out, v, t,
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
        v, t = _union_mesh()
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


@app.get("/", response_class=HTMLResponse)
def index():
    return (WEB / "index.html").read_text(encoding="utf-8")


app.mount("/web", StaticFiles(directory=str(WEB)), name="web")
