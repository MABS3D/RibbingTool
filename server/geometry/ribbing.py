import math
from dataclasses import dataclass, field

import numpy as np
import shapely
from shapely.geometry import Polygon
from shapely.strtree import STRtree

from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakePolygon
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.BRepClass3d import BRepClass3d_SolidClassifier
from OCP.BRepLProp import BRepLProp_SLProps
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.BRepOffsetAPI import BRepOffsetAPI_ThruSections
from OCP.TopAbs import TopAbs_IN, TopAbs_REVERSED
from OCP.gp import gp_Pnt

from .booleans import BooleanError, _fuse_args, mesh_fallback_fuse
from .flatten import FlattenError, boundary_loops, flatten
from .meshing import face_mesh
from .patterns import capsule, clip_and_border, generate_segments
from .step_io import get_face, shape_volume


class RibbingError(Exception):
    pass


@dataclass
class RibReport:
    face_id: int
    segments: int = 0
    lofted: int = 0
    skipped: int = 0
    warnings: list = field(default_factory=list)


MAX_SEGMENTS = 4000
_FOLD_COS = math.cos(math.radians(85))


def _signed_area(coords):
    x, y = coords[:, 0], coords[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)


def _flat_boundary(mesh, flat):
    """Shapely polygon (outer + holes) of the flattened face."""
    loops = boundary_loops(mesh.triangles)
    rings = [flat[np.asarray(l)] for l in loops if len(l) >= 3]
    if not rings:
        raise FlattenError("face boundary could not be traced")
    areas = [abs(_signed_area(r)) for r in rings]
    outer = rings[int(np.argmax(areas))]
    holes = [r for i, r in enumerate(rings) if i != int(np.argmax(areas))]
    poly = Polygon(outer, holes)
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty:
        raise FlattenError("flattened face boundary is degenerate")
    return poly


class _SurfaceMapper:
    """Maps flattened 2D points back onto the true surface with normal offsets."""

    def __init__(self, mesh, flat, face, body):
        self.mesh = mesh
        self.flat = flat
        self.tri_polys = [Polygon(flat[t]) for t in mesh.triangles]
        self.tree = STRtree(self.tri_polys)
        self.adaptor = BRepAdaptor_Surface(face)
        self.props = BRepLProp_SLProps(self.adaptor, 1, 1e-6)
        self.sign = -1.0 if face.Orientation() == TopAbs_REVERSED else 1.0
        self._calibrate(body)

    def _eval(self, u, v):
        """Surface point and material-outward unit normal at (u, v)."""
        self.props.SetParameters(float(u), float(v))
        if not self.props.IsNormalDefined():
            return None, None
        p = self.props.Value()
        n = self.props.Normal()
        nv = np.array([n.X(), n.Y(), n.Z()]) * self.sign
        return np.array([p.X(), p.Y(), p.Z()]), nv

    def _calibrate(self, body):
        """Probe just below the surface: must be inside the solid."""
        areas = np.array([p.area for p in self.tri_polys])
        tri = self.mesh.triangles[int(np.argmax(areas))]
        uv = self.mesh.uvs[tri].mean(axis=0)
        p, n = self._eval(uv[0], uv[1])
        if p is None:
            return
        probe = p - n * 0.2
        cls = BRepClass3d_SolidClassifier(body)
        cls.Perform(gp_Pnt(*probe), 1e-6)
        if cls.State() != TopAbs_IN:
            self.sign = -self.sign

    def _locate(self, pts):
        """Barycentric UV interpolation for an (k,2) array of flat points."""
        k = len(pts)
        geoms = shapely.points(pts)
        tri_idx = np.full(k, -1, dtype=np.int64)
        pi, ti = self.tree.query(geoms, predicate="intersects")
        for p, t in zip(pi, ti):
            if tri_idx[p] < 0:
                tri_idx[p] = t
        missing = np.nonzero(tri_idx < 0)[0]
        if len(missing):
            near = self.tree.query_nearest(geoms[missing])
            for p, t in zip(near[0], near[1]):
                m = missing[p]
                if tri_idx[m] < 0:
                    tri_idx[m] = t
        tris = self.mesh.triangles[tri_idx]
        a, b, c = (self.flat[tris[:, i]] for i in range(3))
        v0, v1, v2 = b - a, c - a, pts - a
        d00 = np.einsum("ij,ij->i", v0, v0)
        d01 = np.einsum("ij,ij->i", v0, v1)
        d11 = np.einsum("ij,ij->i", v1, v1)
        d20 = np.einsum("ij,ij->i", v2, v0)
        d21 = np.einsum("ij,ij->i", v2, v1)
        den = d00 * d11 - d01 * d01
        den[np.abs(den) < 1e-18] = 1e-18
        bv = (d11 * d20 - d01 * d21) / den
        bw = (d00 * d21 - d01 * d20) / den
        bu = 1.0 - bv - bw
        bary = np.clip(np.stack([bu, bv, bw], axis=1), 0.0, None)
        bary /= bary.sum(axis=1, keepdims=True)
        uvs = self.mesh.uvs[tris]                      # (k,3,2)
        return np.einsum("kj,kjd->kd", bary, uvs)      # (k,2)

    def map_loop(self, pts2d, offset):
        """Map a closed 2D loop to 3D at signed normal offset. None on failure."""
        uvq = self._locate(np.asarray(pts2d, float))
        out = np.empty((len(uvq), 3))
        normals = np.empty((len(uvq), 3))
        for i, (u, v) in enumerate(uvq):
            p, n = self._eval(u, v)
            if p is None:
                return None
            out[i] = p + n * offset
            normals[i] = n
        mean = normals.mean(axis=0)
        ln = np.linalg.norm(mean)
        if ln < 1e-9:
            return None
        if (normals @ (mean / ln)).min() < _FOLD_COS:
            return None  # surface folds >85 deg within one rib — unsafe
        return out

    def min_curvature_radius(self, pts2d):
        """Smallest |1/max curvature| over sample points; inf if undefined."""
        uvq = self._locate(np.asarray(pts2d, float))
        props = BRepLProp_SLProps(self.adaptor, 2, 1e-6)
        rmin = math.inf
        for u, v in uvq:
            try:
                props.SetParameters(float(u), float(v))
                if not props.IsCurvatureDefined():
                    continue
                c = max(abs(props.MaxCurvature()), abs(props.MinCurvature()))
                if c > 1e-9:
                    rmin = min(rmin, 1.0 / c)
            except Exception:
                continue
        return rmin


def _wire(pts):
    mk = BRepBuilderAPI_MakePolygon()
    for p in pts:
        mk.Add(gp_Pnt(float(p[0]), float(p[1]), float(p[2])))
    mk.Close()
    if not mk.IsDone():
        raise RibbingError("failed to build rib wire")
    return mk.Wire()


def _loft(bottom, top):
    """Ruled loft with planar caps — only sound when both loops are planar."""
    ts = BRepOffsetAPI_ThruSections(True, True, 1e-6)
    ts.AddWire(_wire(bottom))
    ts.AddWire(_wire(top))
    ts.CheckCompatibility(False)
    ts.Build()
    if not ts.IsDone():
        raise RibbingError("loft failed")
    return ts.Shape()


def _sew_rib(bottom, top):
    """Closed triangulated solid between two same-count loops.

    ThruSections cannot cap non-planar loops (produces invalid solids that
    corrupt booleans), so on curved surfaces every rib facet is built as an
    explicit planar triangle: side quads split in two, convex caps fanned.
    """
    from OCP.BRepBuilderAPI import (
        BRepBuilderAPI_MakeFace,
        BRepBuilderAPI_MakeSolid,
        BRepBuilderAPI_Sewing,
    )
    from OCP.TopAbs import TopAbs_SHELL
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS

    sew = BRepBuilderAPI_Sewing(1e-6)

    def tri(a, b, c):
        if (np.linalg.norm(np.cross(np.asarray(b) - a, np.asarray(c) - a))
                < 1e-10):
            return  # degenerate sliver
        poly = BRepBuilderAPI_MakePolygon(
            gp_Pnt(*map(float, a)), gp_Pnt(*map(float, b)),
            gp_Pnt(*map(float, c)), True)
        sew.Add(BRepBuilderAPI_MakeFace(poly.Wire()).Face())

    K = len(bottom)
    for k in range(K):
        k2 = (k + 1) % K
        tri(bottom[k], bottom[k2], top[k2])
        tri(bottom[k], top[k2], top[k])
    for k in range(1, K - 1):
        tri(bottom[0], bottom[k + 1], bottom[k])
        tri(top[0], top[k], top[k + 1])
    sew.Perform()
    ex = TopExp_Explorer(sew.SewedShape(), TopAbs_SHELL)
    if not ex.More():
        raise RibbingError("rib sewing produced no shell")
    mk = BRepBuilderAPI_MakeSolid()
    mk.Add(TopoDS.Shell_s(ex.Current()))
    solid = mk.Solid()
    if shape_volume(solid) < 0:
        solid = solid.Reversed()
    return solid


def _face_ribs(shape, fid, params, lin_defl, rep):
    face = get_face(shape, fid)
    BRepMesh_IncrementalMesh(face, lin_defl, False, 0.3, True)
    mesh = face_mesh(face, fid)
    if mesh is None:
        rep.warnings.append("face could not be triangulated")
        return []
    flat = flatten(mesh)
    boundary = _flat_boundary(mesh, flat)
    lines = clip_and_border(generate_segments(params, boundary.bounds),
                            boundary, params)
    subsegs = []
    for ls in lines:
        cs = list(ls.coords)
        subsegs += [(cs[i], cs[i + 1]) for i in range(len(cs) - 1)]
    if not subsegs:
        rep.warnings.append(
            "pattern produced no ribs on this face (margin too large or spacing "
            "larger than the face)")
        return []
    if len(subsegs) > MAX_SEGMENTS:
        raise RibbingError(
            f"pattern produces {len(subsegs)} rib segments (max {MAX_SEGMENTS}) — "
            "increase spacing or lower density")
    rep.segments = len(subsegs)

    mapper = _SurfaceMapper(mesh, flat, face, shape)
    w_bot = params.thickness / 2.0
    w_top = max(w_bot - params.height * math.tan(math.radians(params.draft_deg)),
                w_bot * 0.05, 1e-3)
    build = _loft if mesh.is_planar else _sew_rib
    solids = []
    for i, (p0, p1) in enumerate(subsegs):
        # On curved faces, stagger each rib's offsets by a hair (< 0.13 mm) so
        # no two ribs share an offset surface: tangent cap-cap contacts at
        # pattern junctions otherwise corrupt the boolean fuse.
        if mesh.is_planar:
            d_embed, d_height = params.embed, params.height
        else:
            d_embed = params.embed + (i * 7 % 64) * 0.002
            d_height = params.height + (i * 11 % 64) * 0.002
        try:
            bot2 = capsule(p0, p1, w_bot)
            top2 = capsule(p0, p1, w_top)
            bot3 = mapper.map_loop(bot2, -d_embed)
            top3 = mapper.map_loop(top2, d_height)
            if bot3 is None or top3 is None:
                rep.skipped += 1
                continue
            solid = build(bot3, top3)
            if not BRepCheck_Analyzer(solid).IsValid():
                rep.skipped += 1
                continue
            solids.append(solid)
            rep.lofted += 1
        except Exception:
            rep.skipped += 1
    if solids:
        mids = np.array([[(a[0] + b[0]) / 2, (a[1] + b[1]) / 2]
                         for a, b in subsegs[:40]])
        rmin = mapper.min_curvature_radius(mids)
        if params.height > 0.8 * rmin:
            rep.warnings.append(
                f"rib height {params.height:g} exceeds ~80% of the local "
                f"curvature radius ({rmin:.1f}) — ribs may self-intersect")
    else:
        rep.warnings.append("all rib segments failed to build on this face")
    return solids


def apply_ribs(shape, face_ids, params, lin_defl=0.4, allow_fallback=True):
    """Apply the rib pattern to the given faces. Returns (new_shape, reports)."""
    if not face_ids:
        raise RibbingError("no faces selected")
    all_solids, reports = [], []
    for fid in face_ids:
        rep = RibReport(face_id=fid)
        reports.append(rep)
        try:
            all_solids += _face_ribs(shape, fid, params, lin_defl, rep)
        except FlattenError as e:
            rep.warnings.append(str(e))
        except RibbingError:
            raise
        except Exception as e:
            rep.warnings.append(f"face failed: {e}")
    if not all_solids:
        raise RibbingError("no ribs could be built: "
                           + "; ".join(w for r in reports for w in r.warnings))
    try:
        out = _fuse_args([shape], all_solids)
    except BooleanError:
        if not allow_fallback:
            raise
        out = mesh_fallback_fuse(shape, all_solids)
        for r in reports:
            r.warnings.append(
                "OCCT fuse failed — output is faceted (mesh boolean fallback)")
    v0, v1 = shape_volume(shape), shape_volume(out)
    if v1 <= v0 + 1e-9:
        raise RibbingError("ribbing produced no volume increase — fuse failed")
    if not BRepCheck_Analyzer(out).IsValid():
        for r in reports:
            r.warnings.append("result failed BRepCheck (may still export fine)")
    return out, reports
