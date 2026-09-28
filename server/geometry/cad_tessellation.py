"""Recover failed CAD faces using their shared, discretized contours.

Only mesh data changes. Original surfaces, edges, face IDs and tolerances are
retained. A failed recovery leaves the face unmeshed for the caller's existing
coverage guard; it never substitutes a partly triangulated surface.
"""
from collections import Counter
import logging

import numpy as np
from shapely import constrained_delaunay_triangles
from shapely.geometry import Polygon

from OCP.BRep import BRep_Builder, BRep_Tool
from OCP.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Curve2d, BRepAdaptor_Surface
from OCP.BRepGProp import BRepGProp
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.BRepTools import BRepTools_WireExplorer
from OCP.GCPnts import GCPnts_TangentialDeflection
from OCP.GeomAbs import (GeomAbs_Cone, GeomAbs_Cylinder, GeomAbs_Plane, GeomAbs_Sphere,
                         GeomAbs_BSplineSurface, GeomAbs_BezierSurface)
from OCP.GProp import GProp_GProps
from OCP.Poly import Poly_PolygonOnTriangulation, Poly_Triangle, Poly_Triangulation
from OCP.ShapeAnalysis import ShapeAnalysis_Surface
from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_FORWARD, TopAbs_REVERSED, TopAbs_WIRE
from OCP.TopExp import TopExp, TopExp_Explorer
from OCP.TopLoc import TopLoc_Location
from OCP.TopTools import TopTools_IndexedDataMapOfShapeListOfShape
from OCP.TopoDS import TopoDS
from OCP.gp import gp_Pnt, gp_Pnt2d, gp_Vec

from .step_io import face_map

log = logging.getLogger(__name__)
_REGULAR = {GeomAbs_Cone, GeomAbs_Cylinder, GeomAbs_Plane, GeomAbs_Sphere,
            GeomAbs_BSplineSurface, GeomAbs_BezierSurface}
_MAX_TRIANGLES = 100000


def _triangles(tri):
    return np.array([tri.Triangle(i).Get() for i in range(1, tri.NbTriangles()+1)], np.int64)


def _complete_boundary(face, tri, loc):
    """A non-null OCCT triangulation can cover only a small part of a face."""
    if tri is None or not tri.NbTriangles():
        return False
    f = _triangles(tri)
    edges = np.sort(np.vstack([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
    stride = tri.NbNodes()+1
    keys, counts = np.unique(edges[:, 0]*stride+edges[:, 1], return_counts=True)
    boundary = set(keys[counts == 1])
    expected = set(); contour_segments = []
    ex = TopExp_Explorer(face, TopAbs_EDGE)
    while ex.More():
        edge = TopoDS.Edge_s(ex.Current())
        poly = BRep_Tool.PolygonOnTriangulation_s(edge, tri, loc)
        if poly is None:
            if not BRep_Tool.Degenerated_s(edge):
                return False
        else:
            nodes = [poly.Node(i) for i in range(1, poly.NbNodes()+1)]
            expected.update(min(a, b)*stride+max(a, b) for a, b in zip(nodes, nodes[1:]) if a != b)
            contour_segments.extend(zip(nodes, nodes[1:]))
        ex.Next()
    if boundary == expected and np.all(counts <= 2):
        return True
    # At analytic poles, several UV vertices represent one 3D point. Test
    # coverage in physical space before declaring a missing surface. Seam
    # segments occur twice and cancel after this exact-coincidence weld.
    vertices = np.array([tri.Node(i).Coord() for i in range(1, tri.NbNodes()+1)])
    _, remap = np.unique(np.round(vertices/1e-9).astype(np.int64), axis=0, return_inverse=True)
    g = remap[f-1]
    g = g[(g[:, 0] != g[:, 1]) & (g[:, 1] != g[:, 2]) & (g[:, 0] != g[:, 2])]
    edges = np.sort(np.vstack([g[:, [0, 1]], g[:, [1, 2]], g[:, [2, 0]]]), axis=1)
    keys, counts = np.unique(edges[:, 0]*stride+edges[:, 1], return_counts=True)
    contour = Counter()
    for a, b in contour_segments:
        a, b = remap[a-1], remap[b-1]
        if a != b:
            contour[min(a, b)*stride+max(a, b)] += 1
    return np.all(counts <= 2) and set(keys[counts == 1]) == {k for k, n in contour.items() if n == 1}


def _edge_samples(edge, edge_id, adjacent, fm, bad, cache, linear, angular):
    if edge_id in cache:
        return cache[edge_id]
    source = None
    for item in adjacent:
        other = TopoDS.Face_s(item)
        if fm.FindIndex(other) in bad:
            continue
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation_s(other, loc)
        if tri is None:
            continue
        poly = BRep_Tool.PolygonOnTriangulation_s(edge, tri, loc)
        if poly is not None and poly.HasParameters() and poly.NbNodes() >= 2:
            parameters = np.array([poly.Parameter(i) for i in range(1, poly.NbNodes()+1)])
            points = np.array([tri.Node(poly.Node(i)).Transformed(loc.Transformation()).Coord()
                               for i in range(1, poly.NbNodes()+1)])
            source = points, parameters
            break
    if source is None:
        # A seam, or an edge whose incident faces both need recovery. Cache
        # one discretization so both occurrences receive exactly the same 3D nodes.
        forward = TopoDS.Edge_s(edge.Oriented(TopAbs_FORWARD))
        curve = BRepAdaptor_Curve(forward)
        sample = GCPnts_TangentialDeflection(curve, angular, linear)
        parameters = np.array([sample.Parameter(i) for i in range(1, sample.NbPoints()+1)])
        points = np.array([sample.Value(i).Coord() for i in range(1, sample.NbPoints()+1)])
        if len(points) < 2:
            raise ValueError('edge cannot be discretized')
        points[0] = BRep_Tool.Pnt_s(TopExp.FirstVertex_s(forward)).Coord()
        points[-1] = BRep_Tool.Pnt_s(TopExp.LastVertex_s(forward)).Coord()
        source = points, parameters
    points, parameters = source
    if parameters[0] > parameters[-1]:
        points, parameters = points[::-1].copy(), parameters[::-1].copy()
    tolerances = np.full(len(points), max(BRep_Tool.Tolerance_s(edge), 1e-7))
    forward = TopoDS.Edge_s(edge.Oriented(TopAbs_FORWARD))
    tolerances[0] = max(tolerances[0], BRep_Tool.Tolerance_s(TopExp.FirstVertex_s(forward)))
    tolerances[-1] = max(tolerances[-1], BRep_Tool.Tolerance_s(TopExp.LastVertex_s(forward)))
    cache[edge_id] = points, parameters, tolerances
    return cache[edge_id]


def _normal(surface, uv):
    du, dv, p = gp_Vec(), gp_Vec(), gp_Pnt()
    surface.D1(float(uv[0]), float(uv[1]), p, du, dv)
    n = np.array(du.Crossed(dv).Coord())
    length = np.linalg.norm(n)
    if not np.isfinite(length) or length < 1e-14:
        raise ValueError('singular surface parameterization')
    return n/length


def _refine(surface, uv, vertices, faces, boundary, linear, angular):
    """Conforming interior bisection with the neighboring CAD mesh fixed."""
    uv, vertices = uv.tolist(), vertices.tolist()
    normals = [_normal(surface, point) for point in uv]
    cosine = np.cos(angular)
    for _ in range(14):
        edges = {tuple(sorted((a, b))) for f in faces for a, b in zip(f, f[1:]+f[:1])}
        if not boundary <= edges:
            raise ValueError('triangulation omitted contour segments')
        splits = {}
        for a, b in sorted(edges-boundary):
            mid = (np.asarray(uv[a])+uv[b])/2
            exact = np.array(surface.Value(*mid).Coord())
            error = np.linalg.norm(exact-(np.asarray(vertices[a])+vertices[b])/2)
            if error > linear or np.dot(normals[a], normals[b]) < cosine:
                splits[a, b] = len(uv)
                uv.append(mid.tolist()); vertices.append(exact.tolist())
                normals.append(_normal(surface, mid))
        if not splits:
            break
        rebuilt, split_nodes = [], set(splits.values())
        for f in faces:
            ring = []
            for a, b in zip(f, f[1:]+f[:1]):
                ring.append(a)
                key = tuple(sorted((a, b)))
                if key in splits:
                    ring.append(splits[key])
            if len(ring) == 3:
                rebuilt.append(f)
            elif len(ring) == 4:
                i = next(i for i, node in enumerate(ring) if node in split_nodes)
                r = ring[i:]+ring[:i]
                rebuilt.extend([[r[0], r[1], r[2]], [r[0], r[2], r[3]]])
            else:
                mid = np.asarray([uv[i] for i in f]).mean(0)
                center = len(uv)
                uv.append(mid.tolist()); vertices.append(list(surface.Value(*mid).Coord()))
                normals.append(_normal(surface, mid))
                rebuilt.extend([[center, a, b] for a, b in zip(ring, ring[1:]+ring[:1])])
        faces = rebuilt
        if len(faces) > _MAX_TRIANGLES:
            raise ValueError('surface recovery exceeded its refinement budget')
    else:
        raise ValueError('surface recovery did not meet the sampling tolerance')
    return np.asarray(uv), np.asarray(vertices), np.asarray(faces, np.int64)


def _recover_face(face, fm, em, bad, edge_cache, linear, angular):
    if BRepAdaptor_Surface(face).GetType() not in _REGULAR:
        raise ValueError('recovery requires a supported regular surface')
    surface = BRep_Tool.Surface_s(face)
    analysis = ShapeAnalysis_Surface(surface)
    vertices, uvs, rings, edge_records = [], [], [], []
    max_tolerance = BRep_Tool.Tolerance_s(face)
    wx = TopExp_Explorer(face, TopAbs_WIRE)
    while wx.More():
        wire = TopoDS.Wire_s(wx.Current())
        ordered = BRepTools_WireExplorer(wire, face)
        ring, first, expected_edges, actual_edges = [], len(vertices), 0, 0
        ex = TopExp_Explorer(wire, TopAbs_EDGE)
        while ex.More():
            expected_edges += 1; ex.Next()
        while ordered.More():
            edge = TopoDS.Edge_s(ordered.Current()); actual_edges += 1
            if BRep_Tool.Degenerated_s(edge):
                raise ValueError('singular boundary requires a different parameterization')
            eid = em.FindIndex(edge)
            points, parameters, tolerance = _edge_samples(
                edge, eid, em.FindFromIndex(eid), fm, bad, edge_cache, linear, angular)
            curve = BRepAdaptor_Curve2d(edge, face)
            hint = np.array([curve.Value(float(t)).Coord() for t in parameters])
            uv = np.array([analysis.NextValueOfUV(gp_Pnt2d(*q), gp_Pnt(*p), 1e-7).Coord()
                           for p, q in zip(points, hint)])
            for axis, periodic in enumerate((surface.IsUPeriodic(), surface.IsVPeriodic())):
                if periodic:
                    period = surface.UPeriod() if axis == 0 else surface.VPeriod()
                    uv[:, axis] += period*np.round((hint[:, axis]-uv[:, axis])/period)
            on_surface = np.array([surface.Value(*p).Coord() for p in uv])
            if np.any(np.linalg.norm(on_surface-points, axis=1) > tolerance*1.01+1e-7):
                raise ValueError('shared contour exceeds its CAD surface tolerance')
            max_tolerance = max(max_tolerance, float(tolerance.max()))
            uv = np.round(uv, 12)
            rev = edge.Orientation() == TopAbs_REVERSED
            if rev:
                points, uv = points[::-1], uv[::-1]
            nodes = []
            for i, (p, q) in enumerate(zip(points, uv)):
                if i == 0 and ring:
                    node = ring[-1]
                    if np.linalg.norm(np.asarray(vertices[node])-p) > 1e-7 or np.linalg.norm(np.asarray(uvs[node])-q) > 1e-7:
                        raise ValueError('shared contour is not a continuous UV wire')
                else:
                    node = len(vertices)
                    vertices.append(p.tolist()); uvs.append(q.tolist()); ring.append(node)
                nodes.append(node)
            edge_records.append((eid, edge, nodes[::-1] if rev else nodes, parameters, rev))
            ordered.Next()
        if actual_edges != expected_edges or len(ring) < 4:
            raise ValueError('incomplete or degenerate boundary wire')
        last = ring.pop()
        if np.linalg.norm(np.asarray(vertices[last])-vertices[first]) > 1e-7 or np.linalg.norm(np.asarray(uvs[last])-uvs[first]) > 1e-7:
            raise ValueError('boundary does not close in its parameter domain')
        # The closing node occurs only in this wire's last edge record.
        nodes = edge_records[-1][2]
        nodes[:] = [first if node == last else node for node in nodes]
        vertices.pop(); uvs.pop(); rings.append(ring)
        wx.Next()
    vertices, uv = np.asarray(vertices), np.asarray(uvs)
    if not len(rings) or not np.isfinite(vertices).all() or not np.isfinite(uv).all():
        raise ValueError('empty or non-finite face boundary')
    center = uv.mean(0); p, du, dv = gp_Pnt(), gp_Vec(), gp_Vec()
    surface.D1(*center, p, du, dv)
    scale = np.array([du.Magnitude(), dv.Magnitude()])
    if np.min(scale) < 1e-12:
        raise ValueError('singular surface metric')
    xy = (uv-center)*scale
    rings.sort(key=lambda ring: Polygon(xy[ring]).area, reverse=True)
    polygon = Polygon(xy[rings[0]], [xy[ring] for ring in rings[1:]])
    if not polygon.is_valid or polygon.area <= 1e-14:
        raise ValueError('self-intersecting or empty parameter domain')
    index = {tuple(point): i for i, point in enumerate(xy)}
    if len(index) != len(xy):
        raise ValueError('distinct boundary nodes occupy the same UV point')
    faces = []
    for triangle in constrained_delaunay_triangles(polygon).geoms:
        f = [index[tuple(point)] for point in list(triangle.exterior.coords)[:3]]
        a, b, c = uv[f]
        if (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]) < 0:
            f = [f[0], f[2], f[1]]
        faces.append(f)
    boundary = {tuple(sorted((a, b))) for ring in rings for a, b in zip(ring, ring[1:]+ring[:1])}
    perimeter = sum(np.linalg.norm(vertices[a]-vertices[b]) for a, b in boundary)
    uv, vertices, faces = _refine(surface, uv, vertices, faces, boundary, linear, angular)
    xyz = vertices[faces]
    cross = np.cross(xyz[:, 1]-xyz[:, 0], xyz[:, 2]-xyz[:, 0])
    areas = np.linalg.norm(cross, axis=1)*.5
    if not len(faces) or not np.isfinite(areas).all() or np.any(areas <= 1e-14):
        raise ValueError('recovered triangulation contains degenerate triangles')
    for normal, point in zip(cross, uv[faces].mean(1)):
        if normal @ _normal(surface, point) <= 0:
            raise ValueError('recovered triangulation folds over the CAD surface')
    incidence = Counter(tuple(sorted((a, b))) for f in faces.tolist() for a, b in zip(f, f[1:]+f[:1]))
    if {edge for edge, count in incidence.items() if count == 1} != boundary or max(incidence.values()) > 2:
        raise ValueError('recovered triangulation does not preserve the complete contour')
    props = GProp_GProps(); BRepGProp.SurfaceProperties_s(face, props)
    # Vertex tolerances can exceed the width of a tiny sliver. Measure that
    # uncertainty in absolute area; do not silently discard the small face.
    area_budget = props.Mass()*max(.005, angular**2/4) + 2*perimeter*max_tolerance
    if abs(areas.sum()-props.Mass()) > area_budget:
        raise ValueError('recovered triangulation does not cover the CAD face area')
    return vertices, uv, faces, edge_records


def _install(face, result, linear):
    vertices, uv, faces, edge_records = result
    tri = Poly_Triangulation(len(vertices), len(faces), True)
    transform = face.Location().Transformation().Inverted()
    for i, (p, q) in enumerate(zip(vertices, uv), 1):
        tri.SetNode(i, gp_Pnt(*p).Transformed(transform))
        tri.SetUVNode(i, gp_Pnt2d(*q))
    for i, (a, b, c) in enumerate(faces, 1):
        tri.SetTriangle(i, Poly_Triangle(int(a)+1, int(b)+1, int(c)+1))
    tri.Deflection(linear)
    builder = BRep_Builder(); builder.UpdateFace(face, tri)
    groups = {}
    for eid, edge, nodes, parameters, reverse in edge_records:
        poly = Poly_PolygonOnTriangulation(len(nodes), True)
        for i, (node, parameter) in enumerate(zip(nodes, parameters), 1):
            poly.SetNode(i, int(node)+1); poly.SetParameter(i, float(parameter))
        poly.Deflection(linear)
        groups.setdefault(eid, []).append((edge, poly, reverse))
    for records in groups.values():
        edge = records[0][0]
        if len(records) == 1:
            builder.UpdateEdge(edge, records[0][1], tri, face.Location())
        elif len(records) == 2 and records[0][2] != records[1][2]:
            forward, reverse = sorted(records, key=lambda item: item[2])
            builder.UpdateEdge(TopoDS.Edge_s(edge.Oriented(TopAbs_FORWARD)), forward[1], reverse[1], tri, face.Location())
        else:
            raise ValueError('ambiguous seam edge in recovered face')


def _incomplete_faces(fm):
    bad = {}
    for fid in range(1, fm.Size()+1):
        face = TopoDS.Face_s(fm.FindKey(fid)); loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation_s(face, loc)
        if not _complete_boundary(face, tri, loc):
            bad[fid] = face
    return bad


def repair_tessellation(shape, linear, angular):
    fm = face_map(shape); bad = _incomplete_faces(fm)
    if not bad:
        return {'recovered': [], 'failed': {}}
    em = TopTools_IndexedDataMapOfShapeListOfShape()
    TopExp.MapShapesAndAncestors_s(shape, TopAbs_EDGE, TopAbs_FACE, em)
    cache, prepared, failed = {}, {}, {}
    for fid, face in bad.items():
        try:
            prepared[fid] = _recover_face(face, fm, em, bad, cache, linear, angular)
        except Exception as error:
            failed[fid] = str(error)
            log.warning('CAD tessellation recovery refused face %s: %s', fid, error)
    recovered = []
    for fid, face in bad.items():
        try:
            if fid not in prepared:
                BRep_Builder().UpdateFace(face, None)
                continue
            _install(face, prepared[fid], linear)
            recovered.append(fid)
        except Exception as error:
            BRep_Builder().UpdateFace(face, None)
            failed[fid] = str(error)
            log.warning('CAD tessellation installation failed for face %s: %s', fid, error)
    return {'recovered': recovered, 'failed': failed}


def tessellate_shape(shape, linear, angular):
    BRepMesh_IncrementalMesh(shape, linear, False, angular, True)
    fm = face_map(shape)
    refined = []
    for fid, face in _incomplete_faces(fm).items():
        if BRepAdaptor_Surface(face).GetType() not in (GeomAbs_BSplineSurface, GeomAbs_BezierSurface):
            continue
        loc = TopLoc_Location(); tri = BRep_Tool.Triangulation_s(face, loc)
        props = GProp_GProps(); BRepGProp.SurfaceProperties_s(face, props)
        # Coarse contour chords can cross on a very thin CAD sliver. Limit
        # the native retry to small, simple faces; all other recovery uses
        # the explicit bounded refinement above.
        if tri is None or tri.NbNodes() > 64 or props.Mass() > 25*linear**2:
            continue
        BRepMesh_IncrementalMesh(face, min(linear/10, .001), False, min(angular/4, .02), False)
        tri = BRep_Tool.Triangulation_s(face, loc)
        if _complete_boundary(face, tri, loc):
            refined.append(fid)
    if refined:
        em = TopTools_IndexedDataMapOfShapeListOfShape()
        TopExp.MapShapesAndAncestors_s(shape, TopAbs_EDGE, TopAbs_FACE, em)
        dirty, refined_set = set(), set(refined)
        for i in range(1, em.Extent()+1):
            adjacent = {fm.FindIndex(face) for face in em.FindFromIndex(i)}
            if adjacent & refined_set:
                dirty.update(adjacent-refined_set)
        for fid in dirty:
            BRep_Builder().UpdateFace(TopoDS.Face_s(fm.FindKey(fid)), None)
        # Keep the finer shared edge polygons and rebuild their neighbors
        # consistently. Cleaning the whole shape here would lose the retry.
        BRepMesh_IncrementalMesh(shape, linear, False, angular, True)
    return dict(repair_tessellation(shape, linear, angular), refined=refined)
