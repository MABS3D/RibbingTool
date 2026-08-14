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


MAX_SEGMENTS = 9000
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

    def __init__(self, region, flat, body, triangles=None, min_cos=None):
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
        if min_cos is None:
            self.valid_idx = np.nonzero(signed > 1e-12)[0]
        else:
            # projected mapping: a triangle's 2D signed area is its 3D area
            # times (unit normal . projection axis), so this drops steep and
            # back-facing triangles — their compressed shadows must not
            # locate points
            v3 = region.vertices
            a3 = np.linalg.norm(np.cross(v3[t[:, 1]] - v3[t[:, 0]],
                                         v3[t[:, 2]] - v3[t[:, 0]]), axis=1)
            self.valid_idx = np.nonzero(
                signed > np.maximum(min_cos * a3, 1e-12))[0]
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
            # curvature-adaptive sampling: silhouettes kink visibly when the
            # bend angle per facet exceeds a few degrees, so tightly curved
            # ribs sample finer (~4 deg per segment) than flat ones
            r_loc = mapper.min_curvature_radius(np.array(
                [p0, p1, ((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2)]))
            step_i = step if math.isinf(r_loc) else float(
                np.clip(0.07 * r_loc, 0.3 / max(quality, 1.0), step))
            bot2, ns = capsule(p0, p1, w_bot, step=step_i, cap_pts=cap_pts,
                               return_meta=True)
            bot3 = mapper.map_loop(bot2, -d_embed)
            top2 = capsule(p0, p1, w_top, step=step_i, cap_pts=cap_pts)
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


def _map_points(mapper, pts2d, offsets, max_snap=0.5):
    """Raw per-point surface mapping (no loop guards). None on failure.

    max_snap=None skips the out-of-domain gate; the caller can inspect
    mapper.last_snap per point instead.
    """
    pts2d = np.asarray(pts2d, float)
    offsets = np.broadcast_to(np.asarray(offsets, float), (len(pts2d),))
    tri_idx, bary = mapper._locate(pts2d)
    if max_snap is not None and mapper.last_snap.max() > max_snap:
        # capsule caps and offset rings legitimately poke past the flat
        # domain edge — clamping them onto it is correct run-out geometry.
        # The domain polygon is buffer(0)-repaired and can seal invalid
        # hole rings into phantom material, so "inside the polygon" is not
        # proof of a mapping hole: only points sitting on a FLIPPED flat
        # triangle (dropped from the location tree — the fold zones that
        # stretch ribs into giant fans) stay fatal.
        domain = getattr(mapper, "domain", None)
        if domain is None:
            return None
        import shapely as _shp
        far = _shp.points(pts2d[mapper.last_snap > max_snap])
        bad = _shp.contains(domain, far) & (
            _shp.distance(far, domain.boundary) >= 0.05)
        if bad.any():
            flip = np.setdiff1d(np.arange(len(mapper.triangles)),
                                mapper.valid_idx)
            folded = np.zeros(len(far), bool)
            for i in flip:
                folded |= _shp.intersects(
                    Polygon(mapper.flat[mapper.triangles[i]]), far)
            if (bad & folded).any():
                return None
    uvq = np.einsum("kj,kjd->kd", bary, mapper.region.wedge_uvs[tri_idx])
    out = np.empty((len(uvq), 3))
    for i in range(len(uvq)):
        fid = int(mapper.region.tri_face[tri_idx[i]])
        p, n = mapper._eval(fid, uvq[i, 0], uvq[i, 1])
        if p is None:
            return None
        out[i] = p + n * offsets[i]
    return out


def _cluster_mesh(mapper, poly, params, top_offsets, step, quality,
                  strict=False):
    """One watertight triangle mesh for a merged 2D footprint cluster.

    Crossing ribs built as separate solids leave micro-steps and sliver
    scars where their nearly-coplanar tops get unioned; merging footprints
    in 2D first makes every junction a single seamless surface.
    Returns None when the triangulation or the mapping is untrustworthy —
    the caller falls back to per-segment building with strict guards.
    """
    import manifold3d as m3d
    from shapely.geometry.polygon import orient

    poly = orient(poly, 1.0)               # exterior CCW, holes CW
    rings2d, ranges = [], []
    start = 0
    for ring in [poly.exterior, *poly.interiors]:
        L = ring.length
        probe = np.array([ring.interpolate(f * L).coords[0]
                          for f in (0.0, 0.33, 0.66)])
        r_loc = mapper.min_curvature_radius(probe)
        step_r = step if math.isinf(r_loc) else float(
            np.clip(0.07 * r_loc, 0.3 / max(quality, 1.0), step))
        n = max(8, int(math.ceil(L / step_r)))
        pts = np.array([ring.interpolate(i * L / n).coords[0]
                        for i in range(n)])
        rings2d.append(pts)
        ranges.append((start, n))
        start += n
    verts2d = np.vstack(rings2d)
    tris = np.asarray(m3d.triangulate([r.astype(np.float64) for r in rings2d]),
                      np.int64)

    # triangulate() does not fail loudly on eps-invalid rings — it emits
    # overlapping junk (curtains across holes). The signed areas must be
    # non-negative and sum to the RESAMPLED rings' shoelace area (the exact
    # polygon differs legitimately where arcs got corner-cut).
    def shoelace(r):
        x, y = r[:, 0], r[:, 1]
        return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)

    ref_area = sum(shoelace(r) for r in rings2d)
    e1 = verts2d[tris[:, 1]] - verts2d[tris[:, 0]]
    e2 = verts2d[tris[:, 2]] - verts2d[tris[:, 0]]
    signed = 0.5 * (e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0])
    if len(signed) == 0 or signed.min() < -1e-9 or ref_area <= 0:
        return None
    ok_tris = signed > 1e-9                # drop collinear-sample degenerates
    tris = tris[ok_tris]
    if abs(signed.sum() - ref_area) > max(0.5, 0.01 * ref_area):
        return None

    bot3 = _map_points(mapper, verts2d, -params.embed)
    top3 = _map_points(mapper, verts2d, top_offsets(verts2d, params.height))
    if bot3 is None or top3 is None:
        return None

    if strict:
        # per-segment fallback: reject conformal-stretch monsters outright
        edges = np.unique(np.sort(np.vstack([tris[:, [0, 1]], tris[:, [1, 2]],
                                             tris[:, [2, 0]]]), axis=1), axis=0)
        d2 = np.linalg.norm(verts2d[edges[:, 0]] - verts2d[edges[:, 1]], axis=1)
        d3 = np.linalg.norm(top3[edges[:, 0]] - top3[edges[:, 1]], axis=1)
        keep = d2 > 1e-9
        ratio = d3[keep] / d2[keep]
        if len(ratio) and (ratio.max() > 1.8 or ratio.min() < 0.55):
            return None

    r_root = float(max(params.fillet_root, 0.0))
    r_top = float(np.clip(params.fillet_top, 0.0,
                          0.35 * params.thickness))
    if r_root <= 0 and r_top <= 0:
        nb = len(verts2d)
        v = np.vstack([bot3, top3])        # bottom block, then top block
        faces = []
        for a, b, c in tris:
            faces.append((a + nb, b + nb, c + nb))   # top faces up
            faces.append((a, c, b))                  # bottom faces down
        for off, n in ranges:
            for i in range(n):
                a = off + i
                b = off + (i + 1) % n
                faces.append((a, b, b + nb))
                faces.append((a, b + nb, a + nb))
        return v, np.asarray(faces, np.int64)

    return _cluster_mesh_filleted(mapper, params, top_offsets, rings2d,
                                  ranges, verts2d, tris, r_root, r_top)


def _ring_normals(ring2d):
    """Per-vertex outward 2D normals (CCW exterior / CW holes), smoothed."""
    t = np.roll(ring2d, -1, axis=0) - np.roll(ring2d, 1, axis=0)
    ln = np.linalg.norm(t, axis=1, keepdims=True)
    t = t / np.clip(ln, 1e-12, None)
    w = np.stack([t[:, 1], -t[:, 0]], axis=1)
    for _ in range(2):                     # tame reflex-corner spikes
        w = w + np.roll(w, 1, axis=0) + np.roll(w, -1, axis=0)
        w = w / np.clip(np.linalg.norm(w, axis=1, keepdims=True), 1e-12, None)
    return w


def _cluster_mesh_filleted(mapper, params, top_offsets, rings2d, ranges,
                           verts2d, tris, r_root, r_top):
    """Ring-stack cluster mesh with root/crown fillet bands.

    Bottom cap sits on the outward-offset skirt rings; quarter-round arc
    levels blend skirt -> wall -> inset top, so ribs emerge organically from
    the body instead of being plastered onto it.
    """
    import manifold3d as m3d

    def shoelace(r):
        x, y = r[:, 0], r[:, 1]
        return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)

    def valid_triangulation(rings):
        try:
            t = np.asarray(
                m3d.triangulate([r.astype(np.float64) for r in rings]),
                np.int64)
        except Exception:
            return None
        vv = np.vstack(rings)
        e1 = vv[t[:, 1]] - vv[t[:, 0]]
        e2 = vv[t[:, 2]] - vv[t[:, 0]]
        signed = 0.5 * (e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0])
        ref = sum(shoelace(r) for r in rings)
        if (len(signed) == 0 or signed.min() < -1e-9 or ref <= 0
                or abs(signed.sum() - ref) > max(0.5, 0.01 * ref)):
            return None
        return t[signed > 1e-9]

    normals = [_ring_normals(r) for r in rings2d]
    KR, KT = 3, 3
    H = np.asarray(np.broadcast_to(
        np.asarray(top_offsets(verts2d, params.height), float),
        (len(verts2d),)))
    # taper run-outs: radii must fade with the local rib height, or the
    # arcs rise above the ramp (club-shaped tips)
    rr = np.minimum(r_root, np.clip(H, 0.0, None) * 0.85)
    rt = np.minimum(r_top, np.clip(H, 0.0, None) * 0.5)
    rr_r = [rr[o:o + n] for o, n in ranges]
    rt_r = [rt[o:o + n] for o, n in ranges]

    # 2D rings and heights per stack level (bottom cap upward)
    levels = []                            # (ring2d list, h array list)
    skirt = [r + w * s[:, None] for r, w, s in zip(rings2d, normals, rr_r)]
    levels.append((skirt, [np.full(len(r), -params.embed) for r in skirt]))
    if r_root > 0:
        for j in range(KR + 1):
            th = (math.pi / 2) * j / KR
            l2d = [r + w * s[:, None] * (1 - math.sin(th))
                   for r, w, s in zip(rings2d, normals, rr_r)]
            levels.append((l2d, [s * (1 - math.cos(th)) for s in rr_r]))
    inset = ([r - w * s[:, None] for r, w, s in zip(rings2d, normals, rt_r)]
             if r_top > 0 else rings2d)
    if r_top > 0:
        for j in range(KT + 1):
            th = (math.pi / 2) * j / KT
            l2d = [r - w * s[:, None] * (1 - math.cos(th))
                   for r, w, s in zip(rings2d, normals, rt_r)]
            hs = []
            for (o, n), s in zip(ranges, rt_r):
                hv = H[o:o + n] - s * (1 - math.sin(th))
                hs.append(np.clip(hv, rr[o:o + n], None))
            levels.append((l2d, hs))
    else:
        levels.append((rings2d, [H[o:o + n] for o, n in ranges]))

    bot_tris = valid_triangulation(skirt)
    top_tris = valid_triangulation(inset) if r_top > 0 else tris
    if bot_tris is None or top_tris is None:
        return None                        # offset rings self-defeated

    S = len(levels)
    counts = [len(r) for r in rings2d]
    n_ring_total = sum(counts)
    all_v = []
    for l2d, hs in levels:
        # offset skirt/inset rings may poke past the flat domain edge —
        # snapping them back onto it is the correct fillet run-out there
        pts = _map_points(mapper, np.vstack(l2d), np.concatenate(hs),
                          max_snap=max(r_root, r_top) + 0.6)
        if pts is None:
            return None
        all_v.append(pts)
    v = np.vstack(all_v)

    def gv(level, idx):
        return level * n_ring_total + idx

    faces = []
    for a, b, c in top_tris:
        faces.append((gv(S - 1, a), gv(S - 1, b), gv(S - 1, c)))
    for a, b, c in bot_tris:
        faces.append((gv(0, a), gv(0, c), gv(0, b)))
    for lev in range(S - 1):
        for off, n in ranges:
            for i in range(n):
                a = off + i
                b = off + (i + 1) % n
                faces.append((gv(lev, a), gv(lev, b), gv(lev + 1, b)))
                faces.append((gv(lev, a), gv(lev + 1, b), gv(lev + 1, a)))
    return v, np.asarray(faces, np.int64)


def _root_bead(mapper, poly, r_root, params, top_offsets, step, quality):
    """Additive quarter-round bead swept along a cluster's base contour.

    Fillet geometry as its own watertight tube: it unions into the corner
    between rib wall and body at export and can never cost rib coverage.
    """
    from shapely.geometry.polygon import orient

    poly = orient(poly, 1.0)
    K = 4
    theta = [(math.pi / 2) * j / K for j in range(K + 1)]
    # profile in (outward, height): arc surface->wall, then close through
    # the material corner
    prof = ([(r_root * (1 - math.sin(t)), r_root * (1 - math.cos(t)))
             for t in theta]
            + [(-0.15, r_root * 0.7), (-0.15, -0.15),
               (r_root * 0.7, -0.15)])
    m = len(prof)
    meshes = []
    for ring in [poly.exterior, *poly.interiors]:
        L = ring.length
        if L < 2.0:
            continue
        probe = np.array([ring.interpolate(f * L).coords[0]
                          for f in (0.0, 0.33, 0.66)])
        r_loc = mapper.min_curvature_radius(probe)
        step_r = step if math.isinf(r_loc) else float(
            np.clip(0.07 * r_loc, 0.3 / max(quality, 1.0), step))
        n = max(8, int(math.ceil(L / step_r)))
        ring2d = np.array([ring.interpolate(i * L / n).coords[0]
                           for i in range(n)])
        # continuity guard: rings crossing flattening slits jump across
        # openings in 3D and would sweep the bead into giant fans
        base3 = _map_points(mapper, ring2d, 0.0, max_snap=r_root + 0.6)
        if base3 is None:
            continue
        d2 = np.linalg.norm(np.roll(ring2d, -1, axis=0) - ring2d, axis=1)
        d3 = np.linalg.norm(np.roll(base3, -1, axis=0) - base3, axis=1)
        if (d3 > 3.0 * np.clip(d2, 1e-9, None) + 1.0).any():
            continue
        w = _ring_normals(ring2d)
        # the bead must fade with the tapered rib height along the contour
        H_st = np.broadcast_to(np.asarray(
            top_offsets(ring2d, params.height), float), (n,))
        s = np.clip(H_st / r_root, 0.05, 1.0)
        pts2d = np.vstack([ring2d + w * (off * s[:, None]) for off, _ in prof])
        hs = np.concatenate([h * s for _, h in prof])
        pts3d = _map_points(mapper, pts2d, hs, max_snap=r_root + 0.6)
        if pts3d is None:
            continue
        faces = []
        for p in range(m):
            p2 = (p + 1) % m
            for i in range(n):
                i2 = (i + 1) % n
                a = p * n + i
                b = p * n + i2
                c = p2 * n + i2
                d = p2 * n + i
                faces.append((a, b, c))
                faces.append((a, c, d))
        f = np.asarray(faces, np.int64)
        vol = np.einsum("ij,ij->i", pts3d[f[:, 0]],
                        np.cross(pts3d[f[:, 1]], pts3d[f[:, 2]])).sum() / 6.0
        if vol < 0:
            f = f[:, ::-1]
        meshes.append((pts3d, f))
    return meshes


def _projection_frame(regions):
    """Shared projected-mapping frame: (center, (3,2) in-plane axes).

    ONE best-fit plane for ALL regions puts disjoint panels in a single 2D
    space so they can share lattice phase. Axis signs are deterministic, and
    (u, v, n) is right-handed with the outward normal so front-facing
    triangles keep their CCW winding in projection.
    """
    allv = np.vstack([r.vertices for r in regions])
    ctr = allv.mean(axis=0)
    x = allv - ctr
    _, vecs = np.linalg.eigh(x.T @ x)      # ascending variance
    n = vecs[:, 0]                         # least variance = view normal
    outward = np.zeros(3)
    for r in regions:
        v = r.vertices
        outward += np.cross(v[r.triangles[:, 1]] - v[r.triangles[:, 0]],
                            v[r.triangles[:, 2]] - v[r.triangles[:, 0]]
                            ).sum(axis=0)
    if n @ outward < 0:
        n = -n
    u = vecs[:, 1]
    if u[int(np.argmax(np.abs(u)))] < 0:
        u = -u
    return ctr, np.stack([u, np.cross(n, u)], axis=1)


def _lattice_window(params, bounds):
    """Origin-symmetric generation window on whole lattice periods.

    The family generators phase their lines through the window center;
    pinning that center at the projection origin gives every domain in the
    same frame the same global lattice phase — panels separated by grooves,
    recesses, or seams stay period-aligned. (Hex walls are index-anchored
    already; stochastic just gets a window-stable seed cloud.)
    """
    sx = max(params.spacing, 1e-6)
    sy = sx
    if params.pattern == "rectangular" and params.spacing_y:
        sy = max(params.spacing_y, 1e-6)
    minx, miny, maxx, maxy = bounds
    hx = sx * (math.floor(max(abs(minx), abs(maxx)) / sx) + 1)
    hy = sy * (math.floor(max(abs(miny), abs(maxy)) / sy) + 1)
    return (-hx, -hy, hx, hy)


def _kept_domain(flat, triangles, valid_idx):
    """Projected clip boundary: union of the front-facing triangles only.

    Steep walls and undercuts are no rib territory, and without them the
    kept set may be DISCONNECTED (wings + recess floors) — a MultiPolygon
    the downstream clip/margin/taper machinery accepts as-is.
    """
    import shapely
    tri = flat[triangles[valid_idx]]
    if len(tri) == 0:
        return None
    tri = np.concatenate([tri, tri[:, :1]], axis=1)   # close the rings
    geom = shapely.union_all(shapely.polygons(tri))
    if geom is not None and not geom.is_valid:
        geom = geom.buffer(0)
    return geom


def build_rib_meshes(shape, face_ids, params, lin_defl=0.4, quality=1.0):
    """Fast-engine rib builder: watertight cluster meshes, junction-free.

    Merges crossing capsule footprints in 2D per region, builds each
    connected cluster as one triangle mesh (no OCCT solids, no booleans).
    Returns (clusters:[(verts, tris)], reports).
    """
    from shapely.geometry import Polygon as ShpPolygon
    from shapely.ops import unary_union

    if not face_ids:
        raise RibbingError("no faces selected")
    if len(face_ids) > 25:
        lin_defl = max(lin_defl, 0.7)
    regions = region_meshes(shape, face_ids, lin_defl)
    if not regions:
        raise RibbingError("selected faces could not be triangulated")
    project = params.mapping == "project"
    if project:
        p_ctr, p_axes = _projection_frame(regions)
    clusters, reports = [], []
    for region in regions:
        rep = RibReport(face_id=region.face_ids[0], face_ids=region.face_ids)
        reports.append(rep)
        try:
            if project:
                # lattice lives in the shared front-view plane: phase stays
                # continuous across grooves/recesses, where surface-metric
                # unfolding spends whole periods walking their walls
                flat = (region.vertices - p_ctr) @ p_axes
                map_tris = region.triangles
                mapper = _RegionMapper(region, flat, shape,
                                       triangles=map_tris, min_cos=0.30)
                boundary = _kept_domain(flat, map_tris, mapper.valid_idx)
                if boundary is None or boundary.is_empty:
                    rep.warnings.append(
                        "region is edge-on to the projection plane")
                    continue
                gen_bounds = _lattice_window(params, boundary.bounds)
            else:
                flat, map_tris = _flatten_with_cuts(region)
                boundary = _flat_boundary(map_tris, flat)
                gen_bounds = boundary.bounds
            lines = clip_and_border(generate_segments(params, gen_bounds),
                                    boundary, params)
            subsegs = []
            for ls in lines:
                cs = list(ls.coords)
                subsegs += [(cs[i], cs[i + 1]) for i in range(len(cs) - 1)]
            if not subsegs:
                rep.warnings.append("pattern produced no ribs on this region")
                continue
            if len(subsegs) > MAX_SEGMENTS:
                raise RibbingError(
                    f"pattern produces {len(subsegs)} rib segments "
                    f"(max {MAX_SEGMENTS}) — increase spacing or lower density")
            rep.segments = len(subsegs)

            if not project:
                mapper = _RegionMapper(region, flat, shape,
                                       triangles=map_tris)
            mapper.domain = boundary   # lets _map_points clamp boundary pokes
            step = float(np.clip(params.spacing / 6.0, 0.7, 1.5))
            w_bot = params.thickness / 2.0
            import shapely as _shp
            taper = params.taper_len > 0 and not params.border
            inset_poly = boundary.buffer(-params.margin) if taper else None
            bnd_line = inset_poly.boundary if taper else None

            # cluster meshes taper to a true knife edge: distance must be
            # SIGNED — capsule cap tips poke past the clip line, and unsigned
            # distance would ramp them back up instead of to zero
            def top_offsets(pts2d, full_height):
                if not taper:
                    return full_height
                pts = _shp.points(np.asarray(pts2d))
                d = _shp.distance(pts, bnd_line)
                d = np.where(_shp.contains(inset_poly, pts), d, 0.0)
                if d.min() >= params.taper_len:
                    return full_height
                return full_height * np.clip(d / params.taper_len, 0.0, 1.0)

            # prefilter conformal-stretch monsters BEFORE clustering: one
            # pinched segment must not poison a lattice-wide cluster
            ends = np.array([[s[0], s[1]] for s in subsegs], float)
            flat_len = np.linalg.norm(ends[:, 1] - ends[:, 0], axis=1)
            m0 = _map_points(mapper, ends[:, 0], 0.0, max_snap=None)
            snap0 = mapper.last_snap.copy()
            m1 = _map_points(mapper, ends[:, 1], 0.0, max_snap=None)
            snap1 = mapper.last_snap.copy()
            if m0 is None or m1 is None:
                raise RibbingError("region mapping failed")
            ratio = (np.linalg.norm(m1 - m0, axis=1)
                     / np.clip(flat_len, 1e-9, None))
            sane = ((ratio >= 0.55) & (ratio <= 1.8)
                    & (snap0 <= 0.5) & (snap1 <= 0.5))
            rep.skipped += int((~sane).sum())
            subsegs = [s for s, ok in zip(subsegs, sane) if ok]
            if not subsegs:
                rep.warnings.append("all segments in over-distorted zones")
                continue

            caps = [ShpPolygon(capsule(p0, p1, w_bot, step=step))
                    for p0, p1 in subsegs]
            merged = unary_union(caps)
            polys = [p for p in getattr(merged, "geoms", [merged])
                     if p.area > 1e-6]
            reps_pts = [c.representative_point() for c in caps]
            from dataclasses import replace as _dc_replace
            no_fillet = (_dc_replace(params, fillet_root=0.0, fillet_top=0.0)
                         if (params.fillet_root > 0 or params.fillet_top > 0)
                         else None)
            fillet_misses = 0
            for poly in polys:
                members = [i for i, rp in enumerate(reps_pts)
                           if poly.contains(rp)]
                try:
                    mesh = _cluster_mesh(mapper, poly, params, top_offsets,
                                         step, quality)
                except Exception:
                    mesh = None
                if mesh is None and no_fillet is not None:
                    # fillets must never cost coverage: degrade this cluster
                    # to plain geometry and lay an additive root bead along
                    # its base contour instead
                    fillet_misses += len(members)
                    try:
                        mesh = _cluster_mesh(mapper, poly, no_fillet,
                                             top_offsets, step, quality)
                    except Exception:
                        mesh = None
                    if mesh is not None and params.fillet_root > 0:
                        try:
                            clusters.extend(_root_bead(
                                mapper, poly, params.fillet_root, params,
                                top_offsets, step, quality))
                        except Exception:
                            pass
                if mesh is not None:
                    clusters.append(mesh)
                    rep.lofted += len(members)
                    continue
                # cluster untrustworthy (eps-invalid ring, pinched-zone
                # stretch): rebuild its segments individually with strict
                # guards — monsters get skipped, sane ribs survive
                for i in members:
                    try:
                        seg_mesh = _cluster_mesh(
                            mapper, caps[i], params, top_offsets, step,
                            quality, strict=True)
                    except Exception:
                        seg_mesh = None
                    if seg_mesh is None:
                        rep.skipped += 1
                    else:
                        clusters.append(seg_mesh)
                        rep.lofted += 1
            if fillet_misses:
                rep.warnings.append(
                    f"fillets skipped on {fillet_misses} ribs "
                    "(boundary-adjacent clusters built without them)")
        except FlattenError as e:
            rep.warnings.append(str(e))
        except RibbingError:
            raise
        except Exception as e:
            rep.warnings.append(f"region failed: {e}")
    if not clusters:
        raise RibbingError("no ribs could be built: "
                           + "; ".join(w for r in reports for w in r.warnings))
    return clusters, reports


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
