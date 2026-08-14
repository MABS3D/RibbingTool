import io
import struct
import tempfile
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .geometry.booleans import BooleanError
from .geometry.flatten import FlattenError
from .geometry.meshing import mesh_shape
from .geometry.patterns import RibParams
from .geometry.ribbing import RibbingError, apply_ribs
from .geometry.step_io import StepError, load_step, save_step, shape_volume

app = FastAPI(title="RibbingTool")
WEB = Path(__file__).resolve().parents[1] / "web"
STATE = {"stack": [], "filename": None, "meshes": None}


class RibsRequest(BaseModel):
    face_ids: list[int]
    params: dict = {}


def _mesh_payload():
    shape = STATE["stack"][-1]
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
    }


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
    STATE["stack"] = [shape]
    STATE["filename"] = file.filename
    return _mesh_payload()


@app.post("/api/ribs")
def api_ribs(req: RibsRequest):
    if not STATE["stack"]:
        raise HTTPException(400, "no model loaded")
    params = RibParams.from_dict(req.params)
    try:
        out, reports = apply_ribs(STATE["stack"][-1], req.face_ids, params)
    except (RibbingError, FlattenError, BooleanError, StepError, ValueError) as e:
        raise HTTPException(400, str(e))
    STATE["stack"].append(out)
    if len(STATE["stack"]) > 10:
        STATE["stack"] = STATE["stack"][:1] + STATE["stack"][-9:]
    payload = _mesh_payload()
    payload["reports"] = [vars(r) for r in reports]
    return payload


@app.post("/api/undo")
def api_undo():
    if len(STATE["stack"]) < 2:
        raise HTTPException(400, "nothing to undo")
    STATE["stack"].pop()
    return _mesh_payload()


@app.get("/api/export/step")
def api_export_step():
    if not STATE["stack"]:
        raise HTTPException(400, "no model loaded")
    out = Path(tempfile.gettempdir()) / "ribbingtool_export.step"
    save_step(STATE["stack"][-1], out)
    name = (STATE["filename"] or "model").rsplit(".", 1)[0] + "_ribbed.step"
    return FileResponse(out, filename=name, media_type="application/step")


@app.get("/api/export/stl")
def api_export_stl():
    if not STATE["stack"]:
        raise HTTPException(400, "no model loaded")
    meshes = STATE["meshes"] or mesh_shape(STATE["stack"][-1], 0.2, 0.3)
    buf = io.BytesIO()
    buf.write(b"\0" * 80)
    ntri = sum(len(m.triangles) for m in meshes)
    buf.write(struct.pack("<I", ntri))
    for m in meshes:
        v, t = m.vertices, m.triangles
        for a, b, c in t:
            n = np.cross(v[b] - v[a], v[c] - v[a])
            ln = np.linalg.norm(n)
            if ln > 0:
                n = n / ln
            buf.write(struct.pack("<3f", *n))
            for p in (v[a], v[b], v[c]):
                buf.write(struct.pack("<3f", *p))
            buf.write(b"\0\0")
    name = (STATE["filename"] or "model").rsplit(".", 1)[0] + "_ribbed.stl"
    return Response(buf.getvalue(), media_type="model/stl",
                    headers={"Content-Disposition":
                             f'attachment; filename="{name}"'})


@app.get("/", response_class=HTMLResponse)
def index():
    return (WEB / "index.html").read_text(encoding="utf-8")


app.mount("/web", StaticFiles(directory=str(WEB)), name="web")
