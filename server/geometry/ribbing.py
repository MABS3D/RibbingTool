import math
from dataclasses import dataclass, field

import numpy as np
from shapely.geometry import Polygon

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
from .meshing import region_meshes
from .patterns import (
    capsule,
    capsule_cap_triangles,
    clip_and_border,
    generate_segments,
)
from .step_io import get_face, shape_volume


class RibbingError(Exception):
    pass


@dataclass
class RibReport:
    face_id: int
    face_ids: list = field(default_factory=list)
    segments: int = 0
    lofted: int = 0
    skipped: int = 0
    warnings: list = field(default_factory=list)


MAX_SEGMENTS = 4000
_FOLD_COS = math.cos(math.radians(85))


def _signed_area(coords):
    x, y = coords[:, 0], coords[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)


def _flat_boundary(triangles, flat):
    """Shapely polygon (outer + holes) of a flattened region."""
    loops = boundary_loops(triangles)
    rings = [flat[np.asarray(l)] for l in loops if len(l) >= 3]
    if not rings:
        raise FlattenError("region boundary could not be traced")
    areas = [abs(_signed_area(r)) for r in rings]
    imax = int(np.argmax(areas))
    poly = Polygon(rings[imax], [r for i, r in enumerate(rings) if i != imax])
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty:
        raise FlattenError("flattened region boundary is degenerate")
    return poly


def _flatten_with_cuts(region):
    """Flatten a region, cutting non-disk topology open when needed.

    Returns (flat_coords, triangles_for_lookup): the triangle array matches
    region.tri_face / region.wedge_uvs by index in both cases (igl.cut_mesh
    preserves triangle order, only duplicating vertices along cuts).
    """
    from dataclasses import replace
    try:
        return flatten(region), region.triangles
    except FlattenError:
        pass
    import igl
    f = np.ascontiguousarray(region.triangles, np.int64)
    try:
        paths = igl.cut_to_disk(f)
    except Exception:
        paths = []
    if not paths:
        raise FlattenError("region cannot be flattened (non-disk topology)")
    # cut_to_disk's paths include existing boundary loops — only interior
    # edges (two adjacent triangles) may be flagged for cutting
    from collections import Counter
    edge_count = Counter()
    for t in f:
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            edge_count[(min(a, b), max(a, b))] += 1
    cut_edges = set()
    for path in paths:
        for a, b in zip(path[:-1], path[1:]):
            e = (min(a, b), max(a, b))
            if edge_count.get(e, 0) == 2:
                cut_edges.add(e)
    if not cut_edges:
        raise FlattenError("region cannot be flattened (non-disk topology)")
    flags = np.zeros(f.shape, dtype=bool)
    for k, t in enumerate(f):
        for e, (a, b) in enumerate(((t[0], t[1]), (t[1], t[2]), (t[2], t[0]))):
            if (min(a, b), max(a, b)) in cut_edges:
                flags[k, e] = True
    vcut, fcut, _ = igl.cut_mesh(
        np.ascontiguousarray(region.vertices, np.float64), f, flags)
    shadow = replace(region, vertices=vcut,
                     triangles=np.asarray(fcut, np.int32))
    return flatten(shadow), shadow.triangles


class _RegionMapper:
    """Maps flattened 2D points back onto the true (multi-face) surface."""

    def __init__(self, region, flat, body, triangles=None):
        from shapely.strtree import STRtree
        self.region = region
        self.flat = flat
        self.triangles = region.triangles if triangles is None else triangles
        # LSCM can fold near boundaries: flipped (negative-area) flat
        # triangles overlap the valid domain and hijack point location,
        # stretching ribs into spikes. Only sane triangles may locate points.
        t = self.triangles
        e1 = flat[t[:, 1]] - flat[t[:, 0]]
        e2 = flat[t[:, 2]] - flat[t[:, 0]]
        signed = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
        self.valid_idx = np.nonzero(signed > 1e-12)[0]
        self.tree = STRtree([Polygon(flat[t[i]]) for i in self.valid_idx])
        self.props = {}
        self.sign = {}
        self.curv = {}
        for fid in region.face_ids:
            face = get_face(body, fid)
            ad = BRepAdaptor_Surface(face)
            self.props[fid] = BRepLProp_SLProps(ad, 1, 1e-6)
            self.curv[fid] = BRepLProp_SLProps(ad, 2, 1e-6)
            self.sign[fid] = -1.0 if face.Orientation() == TopAbs_REVERSED else 1.0
        self.flip = 1.0
        self._calibrate(body)

    def _eval(self, fid, u, v):
        props = self.props[fid]
        props.SetParameters(float(u), float(v))
        if not props.IsNormalDefined():
            return None, None
        p = props.Value()
        n = props.Normal()
        nv = (np.array([n.X(), n.Y(), n.Z()])
              * self.sign[fid] * self.flip)
        return np.array([p.X(), p.Y(), p.Z()]), nv

    def _calibrate(self, body):
        t = self.triangles
        e1 = self.flat[t[:, 1]] - self.flat[t[:, 0]]
        e2 = self.flat[t[:, 2]] - self.flat[t[:, 0]]
        areas = np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0])
        k = int(np.argmax(areas))
        fid = int(self.region.tri_face[k])
        uv = self.region.wedge_uvs[k].mean(axis=0)
        p, n = self._eval(fid, uv[0], uv[1])
        if p is None:
            return
        cls = BRepClass3d_SolidClassifier(body)
        cls.Perform(gp_Pnt(*(p - n * 0.2)), 1e-6)
        if cls.State() != TopAbs_IN:
            self.flip = -1.0

    def _locate(self, pts):
        import shapely
        k = len(pts)
        geoms = shapely.points(pts)
        tri_idx = np.full(k, -1, dtype=np.int64)
        snap = np.zeros(k)
        pi, ti = self.tree.query(geoms, predicate="intersects")
        for p, t in zip(pi, ti):
            if tri_idx[p] < 0:
                tri_idx[p] = self.valid_idx[t]
        missing = np.nonzero(tri_idx < 0)[0]
        if len(missing):
            near = self.tree.query_nearest(geoms[missing])
            for p, t in zip(near[0], near[1]):
                m = missing[p]
                if tri_idx[m] < 0:
                    tri_idx[m] = self.valid_idx[t]
                    snap[m] = self.tree.geometries[t].distance(geoms[m])
        self.last_snap = snap
        tris = self.triangles[tri_idx]
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
        bary = np.clip(np.stack([1.0 - bv - bw, bv, bw], axis=1), 0.0, None)
        bary /= bary.sum(axis=1, keepdims=True)
        return tri_idx, bary

    def map_loop(self, pts2d, offset, max_snap=0.5):
        """Map a closed 2D loop to 3D at signed normal offset(s).

        offset may be a scalar or a per-point array (rib run-out tapering).
        None on failure. Loops with points snapping further than max_snap
        outside the flat domain are rejected — mapping them stretches ribs
        along the region border into spike artifacts.
        """
        pts2d = np.asarray(pts2d, float)
        offset = np.broadcast_to(np.asarray(offset, float), (len(pts2d),))
        tri_idx, bary = self._locate(pts2d)
        if self.last_snap.max() > max_snap:
            return None
        uvq = np.einsum("kj,kjd->kd", bary, self.region.wedge_uvs[tri_idx])
        out = np.empty((len(uvq), 3))
        normals = np.empty((len(uvq), 3))
        for i in range(len(uvq)):
            fid = int(self.region.tri_face[tri_idx[i]])
            p, n = self._eval(fid, uvq[i, 0], uvq[i, 1])
            if p is None:
                return None
            out[i] = p + n * offset[i]
            normals[i] = n
        mean = normals.mean(axis=0)
        ln = np.linalg.norm(mean)
        if ln < 1e-9:
            return None
        if (normals @ (mean / ln)).min() < _FOLD_COS:
            return None  # crosses a sharp edge / folds >85 deg — unsafe rib
        # conformal maps preserve angles, not lengths: in highly distorted
        # zones a small flat segment maps to a monster 3D rib — reject those
        d2 = np.linalg.norm(np.diff(np.vstack([pts2d, pts2d[:1]]), axis=0),
                            axis=1)
        d3 = np.linalg.norm(np.diff(np.vstack([out, out[:1]]), axis=0),
                            axis=1)
        p2, p3 = d2.sum(), d3.sum()
        # ARAP keeps the flattening near-isometric, so surviving ribs should
        # map close to their flat size — anything beyond this is pathology
        if p2 > 1e-9 and not (0.55 <= p3 / p2 <= 1.8):
            return None
        # a single 3D edge jumping much further than its flat length means
        # the loop crossed an internal slit — a bowtie rib, not a rib
        step = max(p2 / max(len(d2), 1), 1e-6)
        if (d3 - d2).max() > 3.0 * step + 0.8:
            return None
        return out

    def min_curvature_radius(self, pts2d):
        tri_idx, bary = self._locate(np.asarray(pts2d, float))
        uvq = np.einsum("kj,kjd->kd", bary, self.region.wedge_uvs[tri_idx])
        rmin = math.inf
        for i in range(len(uvq)):
            fid = int(self.region.tri_face[tri_idx[i]])
            try:
                props = self.curv[fid]
                props.SetParameters(float(uvq[i, 0]), float(uvq[i, 1]))
                if not props.IsCurvatureDefined():
                    continue
                c = max(abs(props.MaxCurvature()), abs(props.MinCurvature()))
                if c > 1e-9:
                    rmin = min(rmin, 1.0 / c)
            except Exception:
                continue
        return rmin


def _folded(bottom, top):
    """True when the top loop folds/collapses relative to the bottom.

    Offsetting along diverging normals over concave curvature can fold the
    top outline over itself — those ribs come out crumpled/twisted.
    """
    n = (top - bottom).mean(axis=0)
    ln = np.linalg.norm(n)
    if ln < 1e-9:
        return True
    n = n / ln
    # build an in-plane 2D frame and compare signed projected areas
    a = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(n, a)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)

    def signed_area(loop):
        x, y = loop @ u, loop @ v
        return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)

    ab, at = signed_area(bottom), signed_area(top)
    if ab == 0 or at == 0 or (ab > 0) != (at > 0):
        return True
    return not (0.15 <= abs(at) / abs(ab) <= 6.0)


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


def _sew_rib(bottom, top, cap_tris=None):
    """Closed triangulated solid between two same-count loops.

    ThruSections cannot cap non-planar loops (produces invalid solids that
    corrupt booleans), so on curved surfaces every rib facet is built as an
    explicit planar triangle. cap_tris (from capsule_cap_triangles) keeps
    cap chords across the rib width; without it, caps fall back to an
    end-vertex fan (fine only for near-planar loops).
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
            return
        poly = BRepBuilderAPI_MakePolygon(
            gp_Pnt(*map(float, a)), gp_Pnt(*map(float, b)),
            gp_Pnt(*map(float, c)), True)
        sew.Add(BRepBuilderAPI_MakeFace(poly.Wire()).Face())

    K = len(bottom)
    for k in range(K):
        k2 = (k + 1) % K
        tri(bottom[k], bottom[k2], top[k2])
        tri(bottom[k], top[k2], top[k])
    if cap_tris is None:
        cap_tris = [(0, k, k + 1) for k in range(1, K - 1)]
    for a, b, c in cap_tris:
        tri(top[a], top[b], top[c])            # top faces up
        tri(bottom[a], bottom[c], bottom[b])   # bottom faces down
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


def _region_ribs(shape, region, params, rep, stagger, quality=1.0):
    flat, map_tris = _flatten_with_cuts(region)
    boundary = _flat_boundary(map_tris, flat)
    lines = clip_and_border(generate_segments(params, boundary.bounds),
                            boundary, params)
    subsegs = []
    for ls in lines:
        cs = list(ls.coords)
        subsegs += [(cs[i], cs[i + 1]) for i in range(len(cs) - 1)]
    if not subsegs:
        rep.warnings.append(
            "pattern produced no ribs on this region (margin too large or "
            "spacing larger than the region)")
        return []
    if len(subsegs) > MAX_SEGMENTS:
        raise RibbingError(
            f"pattern produces {len(subsegs)} rib segments (max {MAX_SEGMENTS})"
            " — increase spacing or lower density")
    rep.segments = len(subsegs)

    mapper = _RegionMapper(region, flat, shape, triangles=map_tris)
    w_bot = params.thickness / 2.0
    w_top = max(w_bot - params.height * math.tan(math.radians(params.draft_deg)),
                w_bot * 0.05, 1e-3)
    step = float(np.clip(params.spacing / 6.0, 0.7, 1.5)) / max(quality, 0.1)
    cap_pts = 5 if quality <= 1.0 else 11
    build = _loft if region.all_planar else _sew_rib

    # run-out tapering: with no border rib, rib height ramps to ~0 towards
    # the open boundary instead of ending abruptly
    import shapely as _shp
    taper = params.taper_len > 0 and not params.border
    bnd_line = boundary.buffer(-params.margin).boundary if taper else None
    # thin run-out tips are poison for the exact OCCT fuse (stagger=True):
    # keep them a bit taller there
    f_min = 0.3 if stagger else 0.05

    def top_offsets(pts2d, full_height):
        if not taper:
            return full_height
        d = _shp.distance(_shp.points(np.asarray(pts2d)), bnd_line)
        if d.min() >= params.taper_len:
            return full_height
        return full_height * np.clip(d / params.taper_len, f_min, 1.0)

    solids = []
    for i, (p0, p1) in enumerate(subsegs):
        # Staggered offsets keep tangent cap contacts out of the exact OCCT
        # fuse; the mesh-space union does not need them.
        if stagger and not region.all_planar:
            d_embed = params.embed + (i * 7 % 64) * 0.002
            d_height = params.height + (i * 11 % 64) * 0.002
        else:
            d_embed, d_height = params.embed, params.height
        try:
            bot2, ns = capsule(p0, p1, w_bot, step=step, cap_pts=cap_pts,
                               return_meta=True)
            bot3 = mapper.map_loop(bot2, -d_embed)
            top2 = capsule(p0, p1, w_top, step=step, cap_pts=cap_pts)
            off = top_offsets(top2, d_height)
            top3 = mapper.map_loop(top2, off)
            if bot3 is None or top3 is None or _folded(bot3, top3):
                rep.skipped += 1
                continue
            if build is _loft and not isinstance(off, np.ndarray):
                solid = _loft(bot3, top3)
            else:
                # tapered tops are non-planar (ThruSections cannot cap them);
                # curved ribs need width-wise cap chords
                cap_tris = capsule_cap_triangles(len(bot2), ns, cap_pts)
                solid = _sew_rib(bot3, top3, cap_tris)
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
        rep.warnings.append("all rib segments failed to build on this region")
    return solids


def build_rib_solids(shape, face_ids, params, lin_defl=0.4, stagger=False,
                     quality=1.0):
    """Build rib solids for the selected faces without fusing.

    Connected faces are welded into regions that share one coherent flattened
    pattern (ribs run continuously across face boundaries). quality > 1
    refines rib facet sampling (export-grade smoothness).
    Returns (solids, reports). Raises RibbingError if nothing could be built.
    """
    if not face_ids:
        raise RibbingError("no faces selected")
    if len(face_ids) > 25:
        # many-face regions: coarser mapping mesh keeps memory in check
        # (surface evaluation stays exact through UVs regardless)
        lin_defl = max(lin_defl, 0.7)
    regions = region_meshes(shape, face_ids, lin_defl)
    if not regions:
        raise RibbingError("selected faces could not be triangulated")
    all_solids, reports = [], []
    for region in regions:
        rep = RibReport(face_id=region.face_ids[0], face_ids=region.face_ids)
        reports.append(rep)
        try:
            all_solids += _region_ribs(shape, region, params, rep, stagger,
                                       quality)
        except FlattenError as e:
            rep.warnings.append(str(e))
        except RibbingError:
            raise
        except Exception as e:
            rep.warnings.append(f"region failed: {e}")
    if not all_solids:
        raise RibbingError("no ribs could be built: "
                           + "; ".join(w for r in reports for w in r.warnings))
    return all_solids, reports


def fuse_into(shape, solids, reports, allow_fallback=True):
    """Exact-fuse solids into shape with validation; mesh fallback on failure.

    OCCT mass fuses on curved geometry can 'succeed' with corrupted results
    (volume loss, invalid shells) — those are detected and retried through
    the mesh-boolean fallback rather than surfaced as exact output.
    """
    v0 = shape_volume(shape)
    fell_back = False
    try:
        out = _fuse_args([shape], solids)
        # volume is the cheap corruption probe — BRepCheck on garbage
        # geometry can churn for minutes
        if shape_volume(out) <= v0 + 1e-9:
            raise BooleanError("fuse produced no volume increase")
    except BooleanError as e:
        if not allow_fallback:
            raise
        out = mesh_fallback_fuse(shape, solids)
        fell_back = True
        for r in reports:
            r.warnings.append(
                f"OCCT fuse unreliable here ({e}) — output is faceted "
                "(mesh boolean fallback)")
    if shape_volume(out) <= v0 + 1e-9:
        raise RibbingError("ribbing produced no volume increase — fuse failed")
    if not fell_back and not BRepCheck_Analyzer(out).IsValid():
        for r in reports:
            r.warnings.append("result failed BRepCheck (may still export fine)")
    return out


def apply_ribs(shape, face_ids, params, lin_defl=0.4, allow_fallback=True):
    """Build ribs and fuse them into the body (exact engine).

    Returns (new_shape, reports).
    """
    all_solids, reports = build_rib_solids(shape, face_ids, params, lin_defl,
                                           stagger=True)
    out = fuse_into(shape, all_solids, reports, allow_fallback)
    return out, reports
