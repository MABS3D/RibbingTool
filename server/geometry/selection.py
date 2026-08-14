"""Tangent-continuity face selection growth (shell/skin picking)."""

import math

import numpy as np

from OCP.BRep import BRep_Tool
from OCP.BRepAdaptor import BRepAdaptor_Curve
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.ShapeAnalysis import ShapeAnalysis_Surface
from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_REVERSED
from OCP.TopExp import TopExp
from OCP.TopTools import TopTools_IndexedDataMapOfShapeListOfShape
from OCP.TopoDS import TopoDS
from OCP.gp import gp_Pnt

from .step_io import face_map


def _face_normal_at(face, point):
    """Unit outward normal of `face` at (or near) a 3D point."""
    surf = BRep_Tool.Surface_s(face)
    sas = ShapeAnalysis_Surface(surf)
    uv = sas.ValueOfUV(point, 1e-4)
    from OCP.BRepLProp import BRepLProp_SLProps
    props = BRepLProp_SLProps(BRepAdaptor_Surface(face), 2, 1e-6)
    props.SetParameters(uv.X(), uv.Y())
    if not props.IsNormalDefined():
        return None
    n = props.Normal()
    v = np.array([n.X(), n.Y(), n.Z()])
    if face.Orientation() == TopAbs_REVERSED:
        v = -v
    return v


def grow_tangent(shape, seed_ids, angle_deg=20.0, max_faces=300):
    """Expand seed face ids across tangent-continuous shared edges.

    Two faces are considered smoothly connected when their outward normals
    at the shared edge's midpoint differ by less than angle_deg.
    """
    from OCP.TopExp import TopExp_Explorer

    fm = face_map(shape)
    emap = TopTools_IndexedDataMapOfShapeListOfShape()
    TopExp.MapShapesAndAncestors_s(shape, TopAbs_EDGE, TopAbs_FACE, emap)

    def face_id(face_shape):
        return fm.FindIndex(face_shape)

    cos_tol = math.cos(math.radians(angle_deg))
    selected = set(int(i) for i in seed_ids)
    frontier = list(selected)
    while frontier and len(selected) < max_faces:
        fid = frontier.pop()
        face = TopoDS.Face_s(fm.FindKey(fid))
        eex = TopExp_Explorer(face, TopAbs_EDGE)
        while eex.More():
            edge = TopoDS.Edge_s(eex.Current())
            eex.Next()
            idx = emap.FindIndex(edge)
            if idx == 0:
                continue
            it_faces = [TopoDS.Face_s(nb) for nb in emap.FindFromIndex(idx)]
            others = [f for f in it_faces if face_id(f) not in selected]
            if not others:
                continue
            # midpoint of the edge
            try:
                curve = BRepAdaptor_Curve(edge)
                tmid = 0.5 * (curve.FirstParameter() + curve.LastParameter())
                pmid = curve.Value(tmid)
            except Exception:
                continue
            n_here = _face_normal_at(face, pmid)
            if n_here is None:
                continue
            for other in others:
                ofid = face_id(other)
                if ofid in selected:
                    continue
                n_other = _face_normal_at(other, pmid)
                if n_other is None:
                    continue
                if float(np.dot(n_here, n_other)) >= cos_tol:
                    selected.add(ofid)
                    frontier.append(ofid)
    return sorted(selected)
