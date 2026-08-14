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
    from OCP.BRepTools import BRepTools
    from OCP.TopLoc import TopLoc_Location
    from OCP.TopExp import TopExp
    from OCP.TopTools import TopTools_IndexedMapOfShape
    from OCP.TopAbs import TopAbs_FACE, TopAbs_REVERSED
    from OCP.TopoDS import TopoDS

    # stale cached triangulations create T-vertices where one face kept a
    # coarser mesh than its neighbor — remesh everything consistently
    BRepTools.Clean_s(shape)
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


def _weld(v, f, tol=1e-6):
    # OCCT discretizes shared edges once, so coincident nodes are
    # bit-identical: weld exact duplicates only. Coarser tolerances collapse
    # tiny triangles at fine deflections and hole the mesh (non-manifold).
    key = np.round(v / tol).astype(np.int64)
    _, idx, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    f2 = inv[f]
    # drop triangles that collapsed during welding
    ok = (f2[:, 0] != f2[:, 1]) & (f2[:, 1] != f2[:, 2]) & (f2[:, 0] != f2[:, 2])
    return v[idx], f2[ok]


def _repair_boundary(v, f, tol=1e-3, rounds=4):
    """Repair stray boundary edges that make the whole mesh non-manifold.

    Two defect species from real B-reps: near-coincident vertex pairs the
    exact weld missed (snap-welded), and T-vertices where adjacent faces
    disagree on an edge's discretization (the coarse triangle is split at
    the interloping vertex).
    """
    from collections import Counter
    from scipy.spatial import cKDTree
    for _ in range(rounds):
        cnt = Counter()
        for t in f:
            for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
                cnt[(min(a, b), max(a, b))] += 1
        bnd = [e for e, c in cnt.items() if c == 1]
        if not bnd:
            return v, f
        bverts = np.unique(np.array(bnd).ravel())

        tree = cKDTree(v[bverts])
        pairs = tree.query_pairs(tol)
        if pairs:
            remap = np.arange(len(v))
            for i, j in pairs:
                a, b = bverts[i], bverts[j]
                remap[max(a, b)] = min(a, b)
            f = remap[f]
            ok = ((f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2])
                  & (f[:, 0] != f[:, 2]))
            f = f[ok]
            continue

        # T-vertex splits: boundary vertex b lying on boundary edge (a, c).
        # Real defects sit microns off the line — use a print-scale tolerance.
        col_tol = max(tol, 0.05)
        splits = []
        bset = set(map(int, bverts))
        for a, c in bnd:
            ac = v[c] - v[a]
            L2 = float(ac @ ac)
            if L2 < 1e-12:
                continue
            for b in bset - {int(a), int(c)}:
                t_par = float((v[b] - v[a]) @ ac) / L2
                if not 0.01 < t_par < 0.99:
                    continue
                perp = v[b] - (v[a] + t_par * ac)
                if float(perp @ perp) < col_tol * col_tol:
                    splits.append((int(a), int(c), int(b)))
                    break
        if not splits:
            return v, f
        rows = list(map(tuple, f))
        for a, c, b in splits:
            for idx, tri in enumerate(rows):
                if a in tri and c in tri and b not in tri:
                    i = tri.index(a)
                    # directed edge must be a->c or c->a in this triangle
                    if tri[(i + 1) % 3] == c:
                        x = tri[(i + 2) % 3]
                        rows[idx] = (a, b, x)
                        rows.append((b, c, x))
                    elif tri[(i - 1) % 3] == c:
                        x = tri[(i + 1) % 3]
                        rows[idx] = (c, b, x)
                        rows.append((b, a, x))
                    else:
                        continue
                    break
        f = np.asarray(rows, f.dtype)
    return v, f


def _to_manifold(shape, lin_defl):
    import manifold3d as m3d
    v, f = _weld(*_shape_to_mesh(shape, lin_defl))
    v, f = _repair_boundary(v, f)
    mesh = m3d.Mesh(v.astype(np.float32), f.astype(np.uint32))
    man = m3d.Manifold(mesh)
    if man.is_empty():
        raise BooleanError("mesh is not manifold — fallback impossible")
    return man


def _man_to_arrays(man, simplify_tol):
    if simplify_tol:
        try:
            man = man.simplify(simplify_tol)
        except Exception:
            pass
    mesh = man.to_mesh()
    return (np.asarray(mesh.vert_properties, np.float64)[:, :3],
            np.asarray(mesh.tri_verts, np.int64))


def mesh_union(body_shape, rib_solids, lin_defl=0.25, simplify_tol=0.02):
    """Union body + ribs in mesh space. Returns a list of (verts, tris) shells.

    One watertight shell when everything unions; when the body tessellation
    defeats manifold3d (real B-reps can carry non-manifold junctions), fall
    back to two overlapping shells — body + unioned rib lattice — which
    slicers merge natively.
    """
    import manifold3d as m3d
    body_man = None
    try:
        body_man = _to_manifold(body_shape, lin_defl)
    except BooleanError:
        pass
    rib_mans = []
    for s in rib_solids:
        try:
            if isinstance(s, tuple):
                v, f = s   # watertight cluster mesh, indices already shared
                rib_mans.append(m3d.Manifold(
                    m3d.Mesh(np.ascontiguousarray(v, np.float32),
                             np.ascontiguousarray(f, np.uint32))))
            else:
                rib_mans.append(_to_manifold(s, lin_defl))
        except Exception:
            continue
    if not rib_mans and body_man is None:
        raise BooleanError("no meshable geometry to union")

    if body_man is not None:
        man = m3d.Manifold.batch_boolean([body_man] + rib_mans, m3d.OpType.Add)
        if not man.is_empty():
            return [_man_to_arrays(man, simplify_tol)]

    shells = []
    v, f = _weld(*_shape_to_mesh(body_shape, lin_defl))
    shells.append((v.astype(np.float64), f.astype(np.int64)))
    if rib_mans:
        lattice = m3d.Manifold.batch_boolean(rib_mans, m3d.OpType.Add)
        if not lattice.is_empty():
            shells.append(_man_to_arrays(lattice, simplify_tol))
        else:
            for rm in rib_mans:
                shells.append(_man_to_arrays(rm, None))
    return shells


def mesh_fallback_fuse(body_shape, rib_solids, lin_defl=0.5,
                       max_sew_triangles=25000):
    """Mesh-union fallback that rebuilds a (faceted) B-rep via sewing.

    Sewing cost grows steeply with triangle count, so the union is
    simplified first and oversized jobs are refused — those should use the
    fast engine (overlay + faceted export), which never sews.
    """
    import manifold3d as m3d
    mans = [_to_manifold(body_shape, lin_defl)]
    for s in rib_solids:
        mans.append(_to_manifold(s, lin_defl))
    man = m3d.Manifold.batch_boolean(mans, m3d.OpType.Add)
    if man.is_empty():
        raise BooleanError("mesh boolean union produced empty result")
    try:
        man = man.simplify(0.05)
    except Exception:
        pass
    mesh = man.to_mesh()
    tri_verts = np.asarray(mesh.tri_verts, np.int64)
    if len(tri_verts) > max_sew_triangles:
        raise BooleanError(
            f"fallback result too large to rebuild as B-rep "
            f"({len(tri_verts)} triangles) — use the fast engine for this job")
    return _mesh_to_shape(np.asarray(mesh.vert_properties, float)[:, :3],
                          tri_verts)


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
