import numpy as np

from OCP.BRepAlgoAPI import BRepAlgoAPI_Fuse
from OCP.TopTools import TopTools_ListOfShape


class BooleanError(Exception):
    pass


def _fuse_args(args, tools):
    op = BRepAlgoAPI_Fuse()
    la, lt = TopTools_ListOfShape(), TopTools_ListOfShape()
    for s in args:
        la.Append(s)
    for s in tools:
        lt.Append(s)
    op.SetArguments(la)
    op.SetTools(lt)
    op.SetFuzzyValue(1e-4)
    op.SetRunParallel(True)
    op.Build()
    if not op.IsDone():
        raise BooleanError("boolean fuse failed")
    return op.Shape()


def fuse_list(solids):
    if not solids:
        raise BooleanError("nothing to fuse")
    if len(solids) == 1:
        return solids[0]
    return _fuse_args(solids[:1], solids[1:])


def fuse(a, b):
    return _fuse_args([a], [b])


# --- mesh-boolean fallback (manifold3d) -------------------------------------

def _shape_to_mesh(shape, lin_defl):
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.BRep import BRep_Tool
    from OCP.TopLoc import TopLoc_Location
    from OCP.TopExp import TopExp
    from OCP.TopTools import TopTools_IndexedMapOfShape
    from OCP.TopAbs import TopAbs_FACE, TopAbs_REVERSED
    from OCP.TopoDS import TopoDS

    BRepMesh_IncrementalMesh(shape, lin_defl, False, 0.4, True)
    fm = TopTools_IndexedMapOfShape()
    TopExp.MapShapes_s(shape, TopAbs_FACE, fm)
    V, F = [], []
    for i in range(1, fm.Size() + 1):
        face = TopoDS.Face_s(fm.FindKey(i))
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation_s(face, loc)
        if tri is None:
            continue
        base = len(V)
        tr = loc.Transformation()
        for k in range(1, tri.NbNodes() + 1):
            p = tri.Node(k).Transformed(tr)
            V.append((p.X(), p.Y(), p.Z()))
        rev = face.Orientation() == TopAbs_REVERSED
        for k in range(1, tri.NbTriangles() + 1):
            a, b, c = tri.Triangle(k).Get()
            F.append((base + a - 1, base + c - 1, base + b - 1) if rev
                     else (base + a - 1, base + b - 1, base + c - 1))
    return np.array(V, np.float64), np.array(F, np.int64)


def _weld(v, f, tol=1e-3):
    key = np.round(v / tol).astype(np.int64)
    _, idx, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    f2 = inv[f]
    # drop triangles that collapsed during welding
    ok = (f2[:, 0] != f2[:, 1]) & (f2[:, 1] != f2[:, 2]) & (f2[:, 0] != f2[:, 2])
    return v[idx], f2[ok]


def _to_manifold(shape, lin_defl):
    import manifold3d as m3d
    v, f = _weld(*_shape_to_mesh(shape, lin_defl))
    mesh = m3d.Mesh(v.astype(np.float32), f.astype(np.uint32))
    man = m3d.Manifold(mesh)
    if man.is_empty():
        raise BooleanError("mesh is not manifold — fallback impossible")
    return man


def mesh_union(body_shape, rib_solids, lin_defl=0.25, simplify_tol=0.02):
    """Union body + ribs in mesh space. Returns (vertices, triangles) arrays.

    No B-rep reconstruction — orders of magnitude faster than OCCT fuse at
    scale; the result is faceted (for STL and faceted-STEP export).
    """
    import manifold3d as m3d
    mans = [_to_manifold(body_shape, lin_defl)]
    for s in rib_solids:
        mans.append(_to_manifold(s, lin_defl))
    man = m3d.Manifold.batch_boolean(mans, m3d.OpType.Add)
    if man.is_empty():
        raise BooleanError("mesh boolean union produced empty result")
    if simplify_tol:
        try:
            man = man.simplify(simplify_tol)
        except Exception:
            pass
    mesh = man.to_mesh()
    v = np.asarray(mesh.vert_properties, np.float64)[:, :3]
    t = np.asarray(mesh.tri_verts, np.int64)
    return v, t


def mesh_fallback_fuse(body_shape, rib_solids, lin_defl=0.3):
    import manifold3d as m3d
    mans = [_to_manifold(body_shape, lin_defl)]
    for s in rib_solids:
        mans.append(_to_manifold(s, lin_defl))
    man = m3d.Manifold.batch_boolean(mans, m3d.OpType.Add)
    if man.is_empty():
        raise BooleanError("mesh boolean union produced empty result")
    mesh = man.to_mesh()
    return _mesh_to_shape(np.asarray(mesh.vert_properties, float)[:, :3],
                          np.asarray(mesh.tri_verts, np.int64))


def _mesh_to_shape(v, f):
    from OCP.BRepBuilderAPI import (
        BRepBuilderAPI_MakeFace,
        BRepBuilderAPI_MakePolygon,
        BRepBuilderAPI_MakeSolid,
        BRepBuilderAPI_Sewing,
    )
    from OCP.gp import gp_Pnt
    from OCP.TopoDS import TopoDS
    from OCP.TopAbs import TopAbs_SHELL
    from OCP.TopExp import TopExp_Explorer

    sew = BRepBuilderAPI_Sewing(1e-3)
    for a, b, c in f:
        try:
            poly = BRepBuilderAPI_MakePolygon(
                gp_Pnt(*v[a]), gp_Pnt(*v[b]), gp_Pnt(*v[c]), True)
            sew.Add(BRepBuilderAPI_MakeFace(poly.Wire()).Face())
        except Exception:
            continue
    sew.Perform()
    shell_shape = sew.SewedShape()
    ex = TopExp_Explorer(shell_shape, TopAbs_SHELL)
    if not ex.More():
        raise BooleanError("sewing the fallback mesh produced no shell")
    mk = BRepBuilderAPI_MakeSolid()
    while ex.More():
        mk.Add(TopoDS.Shell_s(ex.Current()))
        ex.Next()
    return mk.Solid()
