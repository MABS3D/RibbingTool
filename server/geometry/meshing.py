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


@dataclass
class RegionMesh:
    """Welded mesh of one connected multi-face region.

    UVs live per triangle corner (wedge): a welded vertex on a shared edge
    has different parameters in each adjacent face.
    """
    face_ids: list
    vertices: np.ndarray    # (n,3)
    triangles: np.ndarray   # (m,3) consistently outward-wound
    tri_face: np.ndarray    # (m,) face id per triangle
    wedge_uvs: np.ndarray   # (m,3,2) surface UV per triangle corner
    all_planar: bool


def _edge_node_pairs(shape, fm, selected_faces):
    """Global-index vertex pairs to weld, from shared-edge node polygons.

    Uses OCCT's per-edge PolygonOnTriangulation so welding is exact and a
    face's own seam edges are never welded shut (a full cylinder must stay
    an unrolled sheet).
    """
    from OCP.TopExp import TopExp, TopExp_Explorer
    from OCP.TopTools import TopTools_IndexedDataMapOfShapeListOfShape
    from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE

    emap = TopTools_IndexedDataMapOfShapeListOfShape()
    TopExp.MapShapesAndAncestors_s(shape, TopAbs_EDGE, TopAbs_FACE, emap)

    def polygon_nodes(edge, face_entry):
        face, offset = face_entry
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation_s(face, loc)
        poly = BRep_Tool.PolygonOnTriangulation_s(edge, tri, loc)
        if poly is None:
            return None
        nodes = poly.Nodes()
        return [nodes.Value(i) - 1 + offset
                for i in range(nodes.Lower(), nodes.Upper() + 1)]

    pairs = []
    for idx in range(1, emap.Extent() + 1):
        edge = TopoDS.Edge_s(emap.FindKey(idx))
        adj = []
        for fshape in emap.FindFromIndex(idx):
            fid = fm.FindIndex(fshape)
            if fid in selected_faces:
                adj.append(fid)
        adj = sorted(set(adj))
        if len(adj) != 2:
            # welding across 3+ faces at a junction edge would create a
            # non-manifold mesh — leave such edges unwelded
            continue
        base = polygon_nodes(edge, selected_faces[adj[0]])
        nodes = polygon_nodes(edge, selected_faces[adj[1]])
        if base and nodes and len(nodes) == len(base):
            pairs += list(zip(base, nodes))
    return pairs


def region_meshes(shape, face_ids, lin_defl=0.4, ang_defl=0.3):
    """Weld the selected faces' meshes and split into connected regions.

    Welding follows OCCT's shared-edge node polygons (exact, seam-safe).
    """
    from .step_io import StepError
    BRepMesh_IncrementalMesh(shape, lin_defl, False, ang_defl, True)
    fm = face_map(shape)
    metas, selected_faces = [], {}
    off = 0
    for fid in face_ids:
        if not 1 <= fid <= fm.Size():
            raise StepError(f"face id {fid} out of range 1..{fm.Size()}")
        face = TopoDS.Face_s(fm.FindKey(fid))
        m = face_mesh(face, fid)
        if m is not None and len(m.triangles):
            metas.append(m)
            selected_faces[fid] = (face, off)
            off += len(m.vertices)
    if not metas:
        return []

    allv = np.vstack([m.vertices for m in metas])
    n_all = len(allv)

    # union-find weld over exact shared-edge node pairs
    parent = np.arange(n_all)

    def vfind(a):
        root = a
        while parent[root] != root:
            root = parent[root]
        while parent[a] != root:
            parent[a], a = root, parent[a]
        return root

    for a, b in _edge_node_pairs(shape, fm, selected_faces):
        ra, rb = vfind(a), vfind(b)
        if ra != rb:
            parent[ra] = rb

    def _rebuild():
        roots = np.array([vfind(i) for i in range(n_all)])
        uniq_roots, inverse = np.unique(roots, return_inverse=True)
        verts = allv[uniq_roots]
        tris, tri_face, wedge = [], [], []
        off = 0
        for m in metas:
            t = inverse[m.triangles + off]
            tris.append(t)
            tri_face.append(np.full(len(t), m.face_id))
            wedge.append(m.uvs[m.triangles])
            off += len(m.vertices)
        return (verts, np.vstack(tris), np.concatenate(tri_face),
                np.vstack(wedge))

    verts, tris, tri_face, wedge = _rebuild()

    # weld residual coincident duplicates that appear as zero-length triangle
    # edges (defects, never seams) — they NaN the LSCM cotangents otherwise
    for _ in range(3):
        va = verts[tris]
        zero_pairs = []
        for i, j in ((0, 1), (1, 2), (2, 0)):
            L = np.linalg.norm(va[:, i] - va[:, j], axis=1)
            hit = (L < 1e-7) & (tris[:, i] != tris[:, j])
            for k in np.nonzero(hit)[0]:
                zero_pairs.append((tris[k, i], tris[k, j]))
        if not zero_pairs:
            break
        root_of = {}
        for gi in range(n_all):
            root_of.setdefault(vfind(gi), []).append(gi)
        # map merged-vertex index back to any global index and union
        roots = np.array([vfind(i) for i in range(n_all)])
        uniq_roots = np.unique(roots)
        for a, b in zero_pairs:
            ga = root_of[uniq_roots[a]][0]
            gb = root_of[uniq_roots[b]][0]
            ra, rb = vfind(ga), vfind(gb)
            if ra != rb:
                parent[ra] = rb
        verts, tris, tri_face, wedge = _rebuild()

    # drop true slivers (collinear, distinct vertices)
    va = verts[tris]
    areas = 0.5 * np.linalg.norm(
        np.cross(va[:, 1] - va[:, 0], va[:, 2] - va[:, 0]), axis=1)
    ok = ((areas > 1e-8)
          & (tris[:, 0] != tris[:, 1])
          & (tris[:, 1] != tris[:, 2])
          & (tris[:, 0] != tris[:, 2]))
    tris, tri_face, wedge = tris[ok], tri_face[ok], wedge[ok]
    if not len(tris):
        return []

    # connected components over shared welded edges
    edge_owner = {}
    n_tri = len(tris)
    parent = list(range(n_tri))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i, t in enumerate(tris):
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            k = (min(a, b), max(a, b))
            if k in edge_owner:
                union(i, edge_owner[k])
            else:
                edge_owner[k] = i

    comps = {}
    for i in range(n_tri):
        comps.setdefault(find(i), []).append(i)

    planar_ids = {m.face_id for m in metas if m.is_planar}
    regions = []
    for idxs in comps.values():
        idxs = np.asarray(idxs)
        sub_t = tris[idxs]
        used = np.unique(sub_t)
        remap = np.full(len(verts), -1, dtype=np.int64)
        remap[used] = np.arange(len(used))
        fids = sorted(set(int(f) for f in tri_face[idxs]))
        regions.append(RegionMesh(
            face_ids=fids,
            vertices=verts[used],
            triangles=remap[sub_t].astype(np.int32),
            tri_face=tri_face[idxs],
            wedge_uvs=wedge[idxs],
            all_planar=all(f in planar_ids for f in fids),
        ))
    regions.sort(key=lambda r: r.face_ids[0])
    return regions
