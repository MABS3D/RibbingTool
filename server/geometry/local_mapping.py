"""Compact, manually prescribed fields on the selected surface.

A control is stored in world coordinates with its CAD face id.  Its support
uses intrinsic mesh geodesic distance, never a 3D ball: close opposing walls and
unselected gaps must not receive the same edit.  This is a design field, not
stress analysis.  Future brush samples can use the same scalar field inputs.
"""
import hashlib
import json
import math

import igl
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components

from .flatten import _tri_areas


class MappingControlError(ValueError):
    pass


def validate_controls(controls, face_ids=None):
    if controls is None:
        return []
    if not isinstance(controls, list) or len(controls) > 32:
        raise MappingControlError("mapping controls must be a list of at most 32 points")
    result = []
    seen = set()
    for i, item in enumerate(controls):
        if not isinstance(item, dict):
            raise MappingControlError("each mapping control must be an object")
        try:
            fid = int(item["face_id"])
            if float(item["face_id"]) != fid or fid < 1:
                raise ValueError()
            position = np.asarray(item["position"], dtype=float)
            radius = float(item.get("radius", 20.))
            angle = float(item.get("angle_deg", 0.))
            spacing = float(item.get("spacing_scale", 1.))
            height = float(item.get("height_scale", 1.))
        except (KeyError, ValueError, TypeError, OverflowError) as exc:
            raise MappingControlError("invalid mapping control values") from exc
        if (position.shape != (3,) or not np.isfinite(position).all()
                or not np.isfinite([radius, angle, spacing, height]).all()):
            raise MappingControlError("mapping control values must be finite")
        if not (radius > 0 and -90 <= angle <= 90 and .5 <= spacing <= 2
                and .25 <= height <= 3):
            raise MappingControlError("control limits: radius > 0 mm, angle -90..90 deg, "
                                      "spacing 0.5..2, height 0.25..3")
        if face_ids is not None and fid not in face_ids:
            raise MappingControlError(f"control {i+1} belongs to unselected face {fid}")
        key = str(item.get("id", f"control-{i+1}"))
        if key in seen:
            raise MappingControlError("mapping control ids must be unique")
        seen.add(key)
        result.append(dict(id=key, face_id=fid, position=position.tolist(), radius=radius,
                           angle_deg=angle, spacing_scale=spacing, height_scale=height))
    return result


def refine_control_mesh(v, f, uv, n, tri_face, controls, max_triangles=250000, marked_faces=None):
    """Conforming local midpoint refinement, identical in preview and export.

    CAD planes can consist of just two triangles. Interpolating a field on
    those original vertices would completely miss a control at their centre.
    New vertices interpolate the already computed base map and surface, so
    refinement cannot move the original lattice or change its phase.
    """
    if not controls:
        return v, f, uv, n, tri_face
    for _ in range(24):
        edges, inv = np.unique(np.sort(f[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1),
                               axis=0, return_inverse=True)
        inv = inv.reshape(-1, 3)
        length = np.linalg.norm(v[edges[:, 1]]-v[edges[:, 0]], axis=1)
        mid = (v[edges[:, 0]]+v[edges[:, 1]])/2
        split = np.zeros(len(edges), bool)
        if marked_faces is not None:
            split[inv[marked_faces].ravel()] = True
        else:
            for control in controls:
                radius = control['radius']
                near = np.linalg.norm(mid-control['position'], axis=1) <= radius+length/2
                split |= near & (length > radius/6)
        if not split.any():
            return v, f, uv, n, tri_face
        mids = np.full(len(edges), -1, np.int64)
        mids[split] = np.arange(split.sum())+len(v)
        m = mids[inv]
        mask = (m >= 0)@np.array([1, 2, 4])
        projected_count = int(np.sum(np.choose(mask, [1, 2, 2, 3, 2, 3, 3, 4])))
        if projected_count > max_triangles:
            raise MappingControlError("control radius is too small for this surface; "
                                      "increase it or select a smaller surface patch")
        vv = np.vstack([v, mid[split]])
        uv = np.vstack([uv, (uv[edges[split, 0]]+uv[edges[split, 1]])/2])
        nn = (n[edges[split, 0]]+n[edges[split, 1]])/2
        n = np.vstack([n, nn])
        faces, provenance = [], []
        for code in range(8):
            ids = np.flatnonzero(mask == code)
            if not len(ids):
                continue
            abc, mm = f[ids].T, m[ids].T
            if code in (1, 2, 4):
                rot = {1: 0, 2: 1, 4: 2}[code]
                a, b, c = np.roll(abc, -rot, axis=0)
                ab = np.roll(mm, -rot, axis=0)[0]
                pieces = [(a, ab, c), (ab, b, c)]
            elif code in (3, 6, 5):
                rot = {3: 0, 6: 1, 5: 2}[code]
                a, b, c = np.roll(abc, -rot, axis=0)
                ab, bc, _ = np.roll(mm, -rot, axis=0)
                pieces = [(b, bc, ab), (a, ab, c), (ab, bc, c)]
            elif code == 7:
                a, b, c = abc
                ab, bc, ca = mm
                pieces = [(a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)]
            else:
                pieces = [tuple(abc)]
            for piece in pieces:
                faces.append(np.column_stack(piece))
                provenance.append(tri_face[ids])
        v, f, tri_face = vv, np.vstack(faces), np.concatenate(provenance)
        if marked_faces is not None:
            return v, f, uv, n, tri_face
    raise MappingControlError("control refinement did not converge; increase the radius")


def _anchor(v, f, uv, tri_face, control):
    ids = np.flatnonzero(tri_face == control['face_id'])
    if not len(ids):
        raise MappingControlError("control face is absent from the selected surface")
    point = np.asarray(control['position'], float).reshape(1, 3)
    sq, local, closest = igl.point_mesh_squared_distance(point, v, f[ids])
    if sq[0] > 1.0**2:
        raise MappingControlError("control point is more than 1 mm from its selected face")
    index = ids[local[0]]
    tri = f[index]
    a, b, c = v[tri]
    bary = np.linalg.lstsq(np.column_stack([b-a, c-a]), closest[0]-a, rcond=None)[0]
    bary = np.r_[1-bary.sum(), bary]
    return tri, closest[0], bary@uv[tri], index


def _surface_distances(v, f, source_face, point, radius):
    """Exact compact-support distances, with a validated collar fallback.

    Some cropped CAD fans are unsuitable for the native solver. Retry on a
    larger support before refusing an edit; both attempts retain the same
    edge-connectivity and intrinsic-distance validity checks.
    """
    try:
        return _surface_distances_patch(v, f, source_face, point, radius)
    except MappingControlError:
        return _surface_distances_patch(v, f, source_face, point, radius, collar=True)


def _surface_distances_patch(v, f, source_face, point, radius, collar=False):
    """Exact intrinsic distance on a bounded patch containing the support.

    Edge shortest paths are not sufficient on the very thin CAD triangles:
    their interpolated gradients can be arbitrarily steep across a triangle.
    Insert the picked point in its triangle and solve on the piecewise-linear
    surface itself. A geodesic of length <= radius lies inside that 3D ball;
    conservative triangle bounds retain every triangle crossed by such paths.
    """
    # Every path with length <= radius lies in the Euclidean radius ball.
    # A triangle AABB is a conservative sphere-intersection test, including
    # long triangles whose vertices all lie outside the ball. No arbitrary
    # outer collar or remote geodesic targets are needed for compact support.
    triangles = v[f]
    delta = np.maximum(np.maximum(triangles.min(axis=1)-point,
                                   point-triangles.max(axis=1)), 0.)
    patch_radius = radius*(1.35 if collar else 1.)
    selected = np.flatnonzero(np.einsum('ij,ij->i', delta, delta) <= patch_radius*patch_radius)
    selected = np.unique(np.r_[selected, source_face])
    original_ids, inverse = np.unique(f[selected], return_inverse=True)
    ff = inverse.reshape(-1, 3)
    vv = v[original_ids].copy()
    source = int(np.flatnonzero(selected == source_face)[0])
    tri = ff[source]
    close = np.linalg.norm(vv[tri]-point, axis=1)
    if close.min() < 1e-8:
        source_vertex = int(tri[close.argmin()])
    else:
        source_vertex = len(vv)
        vv = np.vstack([vv, point])
        a, b, c = vv[tri]
        xy = np.linalg.lstsq(np.column_stack([b-a, c-a]), point-a, rcond=None)[0]
        bary = np.r_[1-xy.sum(), xy]
        if bary.min() < 1e-8:
            edge = np.delete(tri, bary.argmin())
            owners = np.flatnonzero(np.isin(ff, edge).sum(axis=1) == 2)
            pieces = []
            for owner in owners:
                t = ff[owner]
                for j in range(3):
                    x, y, z = t[j], t[(j+1)%3], t[(j+2)%3]
                    if x in edge and y in edge:
                        pieces.extend([(x, source_vertex, z), (source_vertex, y, z)])
                        break
            ff = np.vstack([np.delete(ff, owners, axis=0), np.asarray(pieces)])
        else:
            a, b, c = tri
            ff = np.vstack([np.delete(ff, source, axis=0),
                            [[a, b, source_vertex], [b, c, source_vertex], [c, a, source_vertex]]])
    # Cropping can leave triangle fans touching at a vertex outside the
    # support ball. Vertex connectivity incorrectly retains those fans;
    # the native solver may return zero for unreachable targets. Use face
    # connectivity across complete edges, preserving the selected topology.
    edges = np.sort(ff[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1)
    _, inverse = np.unique(edges, axis=0, return_inverse=True)
    order = np.argsort(inverse, kind='stable')
    same = inverse[order[1:]] == inverse[order[:-1]]
    a, b = order[:-1][same]//3, order[1:][same]//3
    graph = sparse.coo_matrix((np.ones(len(a)), (a, b)), shape=(len(ff), len(ff))).tocsr()
    _, labels = connected_components(graph, directed=False)
    source_faces = np.flatnonzero((ff == source_vertex).any(axis=1))
    # A remote bridge can join two fans that touch only at the picked
    # vertex. The source still has no unambiguous surface neighbourhood:
    # require connectivity within its incident triangle star itself.
    source_edge = (edges[order[:-1][same]] == source_vertex).any(axis=1)
    star = sparse.coo_matrix((np.ones(np.count_nonzero(source_edge)),
                              (a[source_edge], b[source_edge])),
                             shape=(len(ff), len(ff))).tocsr()
    source_components, _ = connected_components(star[source_faces][:, source_faces], directed=False)
    source_labels = np.unique(labels[source_faces])
    if source_components != 1 or len(source_labels) != 1:
        raise MappingControlError("the control lies on a pinched surface boundary; "
                                  "move it into a selected face")
    ff = ff[labels == source_labels[0]]
    component = np.unique(ff)
    local = np.full(len(vv), -1, int)
    local[component] = np.arange(len(component))
    ff = local[ff]
    direct = np.linalg.norm(vv[component]-point, axis=1)
    original = component < len(original_ids)
    # Intrinsic distance cannot be shorter than the Euclidean distance.
    # Vertices outside the ball have exactly zero field weight, so asking
    # the solver to reach them only adds unnecessary propagation work.
    targets = np.flatnonzero(original & (direct <= radius))
    result = np.full(len(v), np.inf)
    if not len(targets):
        return result
    distance = igl.exact_geodesic(np.ascontiguousarray(vv[component]), np.ascontiguousarray(ff, np.int64),
                                  np.array([local[source_vertex]], np.int64), np.empty(0, np.int64),
                                  targets.astype(np.int64), np.empty(0, np.int64))
    if (not np.isfinite(distance).all()
            or np.any(distance < direct[targets]-1e-6*np.maximum(direct[targets], 1.))):
        raise MappingControlError("intrinsic distance solver returned an invalid surface path")
    result[original_ids[component[targets]]] = distance
    return result


def _distance_key(control):
    # The caller binds this dictionary to one support mesh. A stable anchor
    # key survives face reordering during conforming refinement.
    return control['face_id'], tuple(control['position']), control['radius']


def control_fields(v, f, uv, tri_face, controls, distance_cache=None):
    """Return locally warped UV, height multipliers and projected controls.

    Edits are accumulated from the original coordinates, independent of list
    order. Overlapping supports blend instead of repeatedly amplifying edits.
    A positive-area guard rejects impossible fields before tracing geometry.
    Chart seams remain boundaries, matching the existing surface mapper.
    """
    if not controls:
        return uv, np.ones(len(v)), []
    delta, logheight = np.zeros_like(uv), np.zeros(len(v))
    mapweights, heightweights = np.zeros(len(v)), np.zeros(len(v))
    projected = []
    for control in controls:
        tri, point, center, source_face = _anchor(v, f, uv, tri_face, control)
        if control['angle_deg'] == 0 and control['spacing_scale'] == 1 and control['height_scale'] == 1:
            projected.append(dict(control, position=point.tolist()))
            continue
        key = _distance_key(control)
        distance = distance_cache.get(key) if distance_cache is not None else None
        if distance is None:
            distance = _surface_distances(v, f, source_face, point, control['radius'])
            if distance_cache is not None:
                distance_cache[key] = distance
        t = np.minimum(distance/control['radius'], 1.)
        weight = np.maximum(1-t*t, 0)**3
        # Cubic in squared radius: flat at the centre, C2 at support boundary.
        # Rotation is applied inversely in pattern space so positive UI angle
        # rotates the physical ribs in the positive local map direction.
        angle = -math.radians(control['angle_deg'])*weight
        scale = np.exp(-math.log(control['spacing_scale'])*weight)
        q = uv-center
        rotated = np.column_stack([np.cos(angle)*q[:, 0]-np.sin(angle)*q[:, 1],
                                    np.sin(angle)*q[:, 0]+np.cos(angle)*q[:, 1]])*scale[:, None]
        delta += rotated-q
        logheight += weight*math.log(control['height_scale'])
        if control['angle_deg'] != 0 or control['spacing_scale'] != 1:
            mapweights += weight
        if control['height_scale'] != 1:
            heightweights += weight
        projected.append(dict(control, position=point.tolist()))
    mapped = uv+delta/np.maximum(1., mapweights)[:, None]
    base_area = _tri_areas(uv, f)
    ratio = _tri_areas(mapped, f)/base_area
    from .surface_mapping import stretch
    relative = stretch(np.column_stack([uv, np.zeros(len(uv))]), f, mapped)
    bad = np.flatnonzero((ratio < .08) | (relative[:, 1] < .12))
    if not np.isfinite(mapped).all() or len(bad):
        error = MappingControlError("local controls fold or excessively compress the mapping; "
                                    "reduce angle/spacing changes or separate overlapping controls")
        error.triangles = bad
        raise error
    return mapped, np.exp(logheight/np.maximum(1., heightweights)), projected


def apply_local_controls(v, f, uv, n, tri_face, controls, cache=None):
    """Deterministic adaptive fields with exact final geodesic verification.

    A cached refined layout must never become the starting mesh for a new
    angle/pitch: that made the same edit succeed or fail depending on prior
    UI actions. Keep a canonical support and memoize layouts separately.
    Native exact geodesics and all Jacobian guards validate each candidate
    mesh before it can be used for ribs.
    """
    projected_controls = []
    for control in controls:
        _, point, _, _ = _anchor(v, f, uv, tri_face, control)
        projected_controls.append(dict(control, position=point.tolist()))
    controls = [c for c in projected_controls
                if c['angle_deg'] != 0 or c['spacing_scale'] != 1 or c['height_scale'] != 1]
    if not controls:
        return v, f, uv, n, np.ones(len(v)), projected_controls
    # A canonical accumulation order also makes overlapping fields invariant
    # to control-list permutations, including the refinement decisions.
    controls.sort(key=lambda c: (*_distance_key(c), c['angle_deg'], c['spacing_scale'], c['height_scale']))
    digest = hashlib.sha256()
    for a in (v, f, uv, n, tri_face):
        digest.update(np.ascontiguousarray(a).tobytes())
    support = sorted(_distance_key(c) for c in controls)
    digest.update(json.dumps(support).encode())
    key = digest.hexdigest()
    meshes = cache.setdefault('local_control_supports', {}) if cache is not None else {}
    held = meshes.get(key)
    if not isinstance(held, dict):
        v, f, uv, n, tri_face = refine_control_mesh(v, f, uv, n, tri_face, controls)
        held = dict(base=(v, f, uv, n, tri_face), distances={}, layouts={})
        if key not in meshes and len(meshes) >= 3:
            meshes.pop(next(iter(meshes)))
        meshes[key] = held
    else:
        v, f, uv, n, tri_face = held['base']
    # Height never changes the surface map or its refinement. It can reuse
    # an accepted layout without repeating intrinsic distance propagation.
    layout_key = tuple((*_distance_key(c), c['angle_deg'], c['spacing_scale']) for c in controls)
    layout = held['layouts'].get(layout_key)
    if layout is not None:
        v, f, uv, n, tri_face, distances = layout
        mapped, height, _ = control_fields(v, f, uv, tri_face, controls, distances)
        return v, f, mapped, n, height, projected_controls
    distances = dict(held['distances'])
    # Finite geometry budget remains the primary resource bound. The former
    # six-level cap rejected valid CAD slivers before they were resolved.
    for attempt in range(25):
        try:
            mapped, height, _ = control_fields(v, f, uv, tri_face, controls, distances)
            if attempt == 0:
                held['distances'].update(distances)
            if len(held['layouts']) >= 3:
                held['layouts'].pop(next(iter(held['layouts'])))
            held['layouts'][layout_key] = (v, f, uv, n, tri_face, distances)
            return v, f, mapped, n, height, projected_controls
        except MappingControlError as error:
            if attempt == 0:
                held['distances'].update(distances)
            bad = getattr(error, 'triangles', None)
            if attempt == 24 or bad is None or not len(bad):
                raise
            v, f, uv, n, tri_face = refine_control_mesh(
                v, f, uv, n, tri_face, controls, marked_faces=bad)
            distances = {}
