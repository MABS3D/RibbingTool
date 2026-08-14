from dataclasses import dataclass

import numpy as np

from OCP.BRep import BRep_Tool
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.BRepGProp import BRepGProp
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.GProp import GProp_GProps
from OCP.GeomAbs import (
    GeomAbs_BSplineSurface,
    GeomAbs_BezierSurface,
    GeomAbs_Cone,
    GeomAbs_Cylinder,
    GeomAbs_Plane,
    GeomAbs_Sphere,
    GeomAbs_Torus,
)
from OCP.TopAbs import TopAbs_REVERSED
from OCP.TopLoc import TopLoc_Location
from OCP.TopoDS import TopoDS

from .step_io import face_map

_KIND = {
    GeomAbs_Plane: "plane",
    GeomAbs_Cylinder: "cylinder",
    GeomAbs_Cone: "cone",
    GeomAbs_Sphere: "sphere",
    GeomAbs_Torus: "torus",
    GeomAbs_BSplineSurface: "bspline",
    GeomAbs_BezierSurface: "bezier",
}


@dataclass
class FaceMesh:
    face_id: int
    vertices: np.ndarray   # (n,3) float64, world coords
    triangles: np.ndarray  # (m,3) int32, CCW seen from material-outside
    uvs: np.ndarray        # (n,2) float64, surface UV per vertex
    is_planar: bool
    surface_kind: str
    area: float


def face_mesh(face, face_id):
    loc = TopLoc_Location()
    tri = BRep_Tool.Triangulation_s(face, loc)
    if tri is None:
        return None
    trsf = loc.Transformation()
    n = tri.NbNodes()
    verts = np.empty((n, 3))
    uvs = np.zeros((n, 2))
    has_uv = tri.HasUVNodes()
    for i in range(1, n + 1):
        p = tri.Node(i).Transformed(trsf)
        verts[i - 1] = (p.X(), p.Y(), p.Z())
        if has_uv:
            q = tri.UVNode(i)
            uvs[i - 1] = (q.X(), q.Y())
    m = tri.NbTriangles()
    tris = np.empty((m, 3), dtype=np.int32)
    rev = face.Orientation() == TopAbs_REVERSED
    for i in range(1, m + 1):
        a, b, c = tri.Triangle(i).Get()
        tris[i - 1] = (a - 1, c - 1, b - 1) if rev else (a - 1, b - 1, c - 1)
    ad = BRepAdaptor_Surface(face)
    kind = _KIND.get(ad.GetType(), "other")
    gp = GProp_GProps()
    BRepGProp.SurfaceProperties_s(face, gp)
    return FaceMesh(face_id, verts, tris, uvs, kind == "plane", kind, gp.Mass())


def mesh_shape(shape, lin_defl=0.5, ang_defl=0.5):
    BRepMesh_IncrementalMesh(shape, lin_defl, False, ang_defl, True)
    fm = face_map(shape)
    out = []
    for fid in range(1, fm.Size() + 1):
        r = face_mesh(TopoDS.Face_s(fm.FindKey(fid)), fid)
        if r is not None:
            out.append(r)
    return out
