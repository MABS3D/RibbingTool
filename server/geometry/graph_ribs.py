"""Ribs from an explicit surface graph, thickened in three dimensions.

Pattern mapping is performed once, on curves. Distance queries never
reassign the pattern to a different closest face at a concave medial seam.
Geometry is expressed in the selection frame; pattern coordinates can
follow surface distances or the frontal projection.
"""
from dataclasses import asdict, dataclass, field as dataclass_field, is_dataclass
from collections import OrderedDict
import copy
import json
import math
import time

import igl
import numpy as np
from scipy.ndimage import gaussian_filter1d, maximum_filter, minimum_filter

from . import implicit as imp
from .booleans import _weld
from .patterns import generate_segments


def _unit(a):
    return a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-15)


def _barycentric(q, v, f, ids):
    a, b, c = (v[f[ids, i]] for i in range(3))
    x, y, z = b - a, c - a, q - a
    xx, xy, yy = (x*x).sum(1), (x*y).sum(1), (y*y).sum(1)
    xz, yz = (x*z).sum(1), (y*z).sum(1)
    den = np.maximum(xx*yy-xy*xy, 1e-25)
    u, w = (yy*xz-xy*yz)/den, (xx*yz-xy*xz)/den
    return np.column_stack([1-u-w, u, w])


class MeshDistance:
    """Reusable BVH; optional angle/edge pseudonormals for a closed body."""
    def __init__(self, v, f, signed=False):
        self.v = np.ascontiguousarray(v, np.float64)
        self.f = np.ascontiguousarray(f, np.int64)
        self.tree = igl.AABB()
        self.tree.init(self.v, self.f)
        if self.f.shape[1] == 3:
            a, b, c = (self.v[self.f[:, i]] for i in range(3))
            self.base = a
            self.ab, self.ac = b-a, c-a
            xx = np.einsum('ij,ij->i', self.ab, self.ab)
            xy = np.einsum('ij,ij->i', self.ab, self.ac)
            yy = np.einsum('ij,ij->i', self.ac, self.ac)
            self.nondegenerate = xx*yy-xy*xy > 1e-25
            den = np.maximum(xx*yy-xy*xy, 1e-25)
            self.inv_gram = np.column_stack([yy/den, -xy/den, xx/den])
        if signed:
            self.fn = _unit(np.cross(v[f[:, 1]]-v[f[:, 0]],
                                     v[f[:, 2]]-v[f[:, 0]]))
            self.vn = imp._vertex_normals(v, f)
            edges = np.concatenate([f[:, [1, 2]], f[:, [2, 0]], f[:, [0, 1]]])
            _, inv = np.unique(np.sort(edges, axis=1), axis=0,
                               return_inverse=True)
            en = np.zeros((inv.max()+1, 3))
            np.add.at(en, inv, np.tile(self.fn, (3, 1)))
            self.en = _unit(en)[inv].reshape(3, len(f), 3).transpose(1, 0, 2)

    def query(self, q):
        return self.tree.squared_distance(self.v, self.f,
                                          np.ascontiguousarray(q, np.float64))

    def barycentric(self, q, ids):
        """Evaluate cached triangle coordinates instead of rebuilding each Gram matrix."""
        delta = q-self.base[ids]
        x = np.einsum('ij,ij->i', self.ab[ids], delta)
        y = np.einsum('ij,ij->i', self.ac[ids], delta)
        gram = self.inv_gram[ids]
        u = gram[:, 0]*x+gram[:, 1]*y
        w = gram[:, 1]*x+gram[:, 2]*y
        return np.column_stack([1-u-w, u, w])

    def signed(self, q, query=None):
        sq, ids, cp = self.query(q) if query is None else query
        b = self.barycentric(cp, ids)
        normal = self.fn[ids].copy()
        on_edge = b.min(1) < 1e-7
        normal[on_edge] = self.en[ids[on_edge], b[on_edge].argmin(1)]
        on_vertex = b.max(1) > 1-1e-7
        normal[on_vertex] = self.vn[self.f[ids[on_vertex],
                                           b[on_vertex].argmax(1)]]
        sign = np.where(np.einsum('ij,ij->i', q-cp, normal) >= 0, 1., -1.)
        return np.sqrt(sq)*sign


def projected_paths(v, f, n, a, b, coordinates=None, values=None):
    """Pull a finite pattern segment back through its own mesh triangles.

    Both sides of a fold belong to the selected surface. Culling on the
    projected triangle orientation breaks curves at silhouette crossings.
    Weld intersection endpoints and follow connectivity before resampling.
    Explicit surface coordinates also support steep walls without a
    singular frontal projection. UV overlaps never require sheet lookup.
    """
    a, b = np.asarray(a), np.asarray(b)
    length = np.linalg.norm(b-a)
    if length < 1e-8:
        return []
    direction = (b-a)/length
    across = np.array([-direction[1], direction[0]])
    coords = v[:, :2] if coordinates is None else coordinates
    val = (coords-a)@across
    dv = val[f]
    # Zero belongs to the positive side, consistently with the edge test
    # below. Otherwise a pattern exactly on a mesh edge disappears.
    tri = f[(dv.min(1) < 0) & (dv.max(1) >= 0)]
    if not len(tri):
        return []
    points = np.zeros((len(tri), 2, 3))
    normals = np.zeros_like(points)
    flat_points = np.zeros((len(tri), 2, 2))
    attributes = np.zeros((len(tri), 2)) if values is not None else None
    count = np.zeros(len(tri), int)
    for i, j in ((0, 1), (1, 2), (2, 0)):
        ia, ib = tri[:, i], tri[:, j]
        good = (val[ia] >= 0) != (val[ib] >= 0)
        ix = np.flatnonzero(good)
        t = val[ia[good]]/(val[ia[good]]-val[ib[good]])
        points[ix, count[good]] = v[ia[good]]*(1-t[:, None])+v[ib[good]]*t[:, None]
        normals[ix, count[good]] = n[ia[good]]*(1-t[:, None])+n[ib[good]]*t[:, None]
        flat_points[ix, count[good]] = coords[ia[good]]*(1-t[:, None])+coords[ib[good]]*t[:, None]
        if attributes is not None:
            attributes[ix, count[good]] = values[ia[good]]*(1-t)+values[ib[good]]*t
        count[ix] += 1
    # Clip along the finite segment, including interpolated surface normals.
    t = (flat_points-a)@direction
    dt = t[:, 1]-t[:, 0]
    safe = np.where(np.abs(dt) > 1e-12, dt, 1.)
    s0, s1 = -t[:, 0]/safe, (length-t[:, 0])/safe
    lo = np.maximum(0, np.minimum(s0, s1))
    hi = np.minimum(1, np.maximum(s0, s1))
    parallel = np.abs(dt) <= 1e-12
    lo[parallel], hi[parallel] = 0., 1.
    ok = ((count == 2) & (hi > lo)
          & (~parallel | ((t[:, 0] >= 0) & (t[:, 0] <= length))))
    p0, dp = points[:, 0].copy(), points[:, 1]-points[:, 0]
    n0, dn = normals[:, 0].copy(), normals[:, 1]-normals[:, 0]
    points = np.stack([p0+dp*lo[:, None], p0+dp*hi[:, None]], axis=1)
    normals = np.stack([n0+dn*lo[:, None], n0+dn*hi[:, None]], axis=1)
    if attributes is not None:
        x, dx = attributes[:, 0], attributes[:, 1]-attributes[:, 0]
        attributes = np.stack([x+dx*lo, x+dx*hi], axis=1)
    ok &= np.linalg.norm(points[:, 1]-points[:, 0], axis=1) > 1e-7
    if not ok.any():
        return []
    points, normals = points[ok].reshape(-1, 3), normals[ok].reshape(-1, 3)
    _, first, inv = np.unique(np.round(points/1e-6).astype(np.int64),
                              axis=0, return_index=True, return_inverse=True)
    pv = points[first]
    nv = np.zeros_like(pv)
    np.add.at(nv, inv, normals)
    nv = _unit(nv)
    av = None
    if attributes is not None:
        av = np.bincount(inv, weights=attributes[ok].ravel())/np.bincount(inv)
    edges = np.unique(np.sort(inv.reshape(-1, 2), axis=1), axis=0)
    adj = [[] for _ in pv]
    for i, (x, y) in enumerate(edges):
        adj[x].append((y, i))
        adj[y].append((x, i))
    used = np.zeros(len(edges), bool)
    paths = []
    for start in [i for i, ns in enumerate(adj) if len(ns) != 2]+list(range(len(pv))):
        for nxt, edge in adj[start]:
            if used[edge]:
                continue
            ids, cur = [start], nxt
            used[edge] = True
            while True:
                ids.append(cur)
                todo = [(j, k) for j, k in adj[cur] if not used[k]]
                if not todo or len(adj[cur]) != 2:
                    break
                cur, edge = todo[0]
                used[edge] = True
                if cur == start:
                    ids.append(cur)
                    break
            paths.append((pv[ids], nv[ids]) if av is None else (pv[ids], nv[ids], av[ids]))
    return paths


def _open_boundary(v, f):
    edges = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
    edges, counts = np.unique(edges, axis=0, return_counts=True)
    edges = edges[counts == 1]
    return MeshDistance(v, edges) if len(edges) else None


def guide_normals(v, f, normals, radius):
    """Diffuse directions on the surface, without moving graph attachment points.

    Area/cotangent weights give the smoothing radius physical units and avoid
    making it depend on the CAD tessellator's very uneven triangle density.
    """
    if radius <= 0:
        return normals
    from scipy.sparse.linalg import spsolve
    mass = igl.massmatrix(v, f, igl.MASSMATRIX_TYPE_BARYCENTRIC)
    laplace = igl.cotmatrix(v, f)
    active = np.flatnonzero(mass.diagonal() > 1e-15)
    system = mass-radius**2*laplace
    result = normals.copy()
    result[active] = spsolve(system[active][:, active], (mass@normals)[active])
    result = _unit(result)
    # A smoothing guide must retain an outward component at sharp returns.
    outward = (result*normals).sum(1)
    result += np.maximum(.2-outward, 0)[:, None]*normals
    return _unit(result)


def _segment_groups(segments, pattern):
    """Few independent fields, with incident non-collinear edges separated."""
    if pattern != "stochastic":
        angles = [math.atan2(b[1]-a[1], b[0]-a[0]) % math.pi for a, b in segments]
        keys = [0. if a > math.pi-1e-5 else round(a, 5) for a in angles]
        unique = {key: i for i, key in enumerate(sorted(set(keys)))}
        if len(unique) <= 8:
            return [unique[k] for k in keys]
    incident, groups = {}, []
    for a, b in segments:
        keys = [tuple(np.round(q, 6)) for q in (a, b)]
        used = set().union(*(incident.get(k, set()) for k in keys))
        color = next(i for i in range(len(used)+1) if i not in used)
        groups.append(color)
        for key in keys:
            incident.setdefault(key, set()).add(color)
    return groups


@dataclass
class Ribbon:
    mesh: MeshDistance
    normals: np.ndarray
    top_distance: np.ndarray
    height_values: np.ndarray | None = None

    def __post_init__(self):
        self.parallel_normal = None
        # Planar/parallel guides, including tapered endpoints, have an affine
        # crown distance on each ribbon triangle. Evaluate that directly;
        # interpolating three identical normals for every voxel adds no data.
        if np.max(np.abs(self.normals-self.normals[0])) < 1e-12:
            self.parallel_normal = self.normals[0].copy()
            f, gram = self.mesh.f, self.mesh.inv_gram
            a = self.top_distance[f[:, 0]]
            x = self.top_distance[f[:, 1]]-a
            y = self.top_distance[f[:, 2]]-a
            u, w = gram[:, 0]*x+gram[:, 1]*y, gram[:, 1]*x+gram[:, 2]*y
            gradient = self.mesh.ab*u[:, None]+self.mesh.ac*w[:, None]
            self.top_constant = a-np.einsum('ij,ij->i', self.mesh.base, gradient)
            self.top_correction = gradient-self.parallel_normal
            levels = np.einsum('ij,j->i', self.mesh.v, self.parallel_normal)-self.top_distance
            flat = (np.ptp(levels[f], axis=1) < 1e-12) & self.mesh.nondegenerate
            # Preserve an exactly planar crown as a plane. Numerically tiny
            # affine gradients otherwise create alternating signs on grid
            # points that lie exactly on the crown (marching-cubes slivers).
            self.top_correction[flat] = 0
            self.top_constant[flat] = -levels[f[flat, 0]]

    def distances(self, q, half, draft, height=None):
        """Wall and crown distances from one closest-point query.

        Keep these separate until the network has been blended. Blending
        already capped ribs rounds their crowns upward and a subsequent
        height cut leaves a sharp edge through the requested top fillet.
        """
        sq, ids, cp = self.mesh.query(q)
        if self.parallel_normal is None or self.height_values is not None:
            b = self.mesh.barycentric(cp, ids)
            tri = self.mesh.f[ids]
        if self.parallel_normal is not None:
            top = (np.einsum('ij,j->i', q, self.parallel_normal)
                   + np.einsum('ij,ij->i', cp, self.top_correction[ids])+self.top_constant[ids])
        else:
            n = _unit(np.einsum('ijk,ij->ik', self.normals[tri], b))
            top = np.einsum('ij,ij->i', q-cp, n)+np.einsum('ij,ij->i', self.top_distance[tri], b)
        # The closest point already accounts for caps at curve endpoints.
        side = np.sqrt(np.maximum(sq-np.maximum(top, 0)**2, 0))-half
        depth = np.maximum(-top, 0)
        if self.height_values is not None:
            height = np.einsum('ij,ij->i', self.height_values[tri], b)
        if height is not None:
            depth = np.minimum(depth, height)
        side -= draft*depth
        return side, top, height

    def field(self, q, half, radius, draft, height=None):
        side, top, height = self.distances(q, half, draft, height)
        if height is not None:
            radius = np.minimum(radius, height)
        return _round_intersection(side, top, radius)


def trace_rib_paths(v, f, n, segments, params, step=0.3, groups=None,
                    coordinates=None, height_values=None):
    """One curve pipeline for the editing preview and generated ribbons."""
    boundary = _open_boundary(*_weld(v, f))
    if groups is None:
        groups = _segment_groups(segments, params.pattern)
    paths = []
    half = params.thickness/2
    clearance = params.margin+half
    for (a, b), group in zip(segments, groups):
        for raw in projected_paths(v, f, n, a, b, coordinates, height_values):
            points, normals = raw[:2]
            arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
            if arc[-1] < 1e-6:
                continue
            s = np.linspace(0, arc[-1], max(2, int(math.ceil(arc[-1]/step))+1))
            pos = np.column_stack([np.interp(s, arc, points[:, i]) for i in range(3)])
            nr = np.column_stack([np.interp(s, arc, normals[:, i]) for i in range(3)])
            hscale = np.interp(s, arc, raw[2]) if len(raw) == 3 else np.ones(len(s))
            closed = np.linalg.norm(points[0]-points[-1]) < 1e-6
            nr = gaussian_filter1d(nr, 0.6/max(s[1]-s[0], .01), axis=0,
                                  mode="wrap" if closed else "nearest")
            nr = _unit(nr)
            distance = (np.sqrt(boundary.query(pos)[0]) if boundary is not None
                        else np.full(len(pos), 1e6))
            cross = np.flatnonzero((distance[:-1] >= clearance)
                                   != (distance[1:] >= clearance))
            if len(cross):
                alpha = (clearance-distance[cross])/(distance[cross+1]-distance[cross])
                extra = s[cross]+alpha*(s[cross+1]-s[cross])
                ss = np.sort(np.r_[s, extra])
                # A crossing at an existing sample can differ by only a few
                # ulps. Keeping both creates almost-zero-length ribbon edges
                # and singular triangles, which corrupt nearby field queries.
                # Keep the full arc endpoints; clipping stays within 1e-9 mm.
                ids = np.flatnonzero(np.r_[True, np.diff(ss) > 1e-9])
                ids[-1] = len(ss)-1
                ss = ss[ids]
                pos = np.column_stack([np.interp(ss, s, pos[:, i]) for i in range(3)])
                nr = _unit(np.column_stack([np.interp(ss, s, nr[:, i]) for i in range(3)]))
                distance = np.interp(ss, s, distance)
                hscale = np.interp(ss, s, hscale)
                s = ss
            keep = distance >= clearance-1e-8
            runs = np.split(np.flatnonzero(keep), np.flatnonzero(np.diff(np.flatnonzero(keep)) > 1)+1)
            for run in runs:
                if len(run) < 2 or s[run[-1]]-s[run[0]] < half:
                    continue
                p, nn, dd = pos[run], nr[run], distance[run]
                ramp = np.ones(len(run))
                if params.taper_len > 0 and not params.border:
                    ramp = np.clip((dd-clearance)/params.taper_len, 0, 1)
                    ramp = ramp*ramp*(3-2*ramp)
                nominal_height = params.height*hscale[run]
                paths.append(dict(points=p, normals=nn, height=nominal_height*ramp,
                                  nominal_height=nominal_height, group=group,
                                  variable_height=height_values is not None))
    return paths


def ribbons_from_paths(paths, params):
    buffers = {}
    for path in paths:
        p, nn, h = path['points'], path['normals'], path['height']
        low = -max(params.embed, .5)-max(params.fillet_root, 0)
        vv = np.stack([p+low*nn, p+h[:, None]*nn], axis=1).reshape(-1, 3)
        vn = np.repeat(nn, 2, axis=0)
        td = np.stack([low-h, np.zeros(len(h))], axis=1).ravel()
        i = np.arange(len(p)-1)*2
        ff = np.vstack([np.column_stack([i, i+2, i+3]),
                        np.column_stack([i, i+3, i+1])])
        buf = buffers.setdefault(path['group'], [[], [], [], [], [], 0, False])
        buf[0].append(vv)
        buf[1].append(ff+buf[5])
        buf[2].append(vn)
        buf[3].append(td)
        buf[4].append(np.repeat(path['nominal_height'], 2))
        buf[5] += len(vv)
        buf[6] |= path['variable_height']
    return [Ribbon(MeshDistance(np.vstack(v), np.vstack(f)), np.vstack(n),
                   np.concatenate(t), np.concatenate(h) if variable else None)
            for v, f, n, t, h, _, variable in buffers.values()]


def make_ribbons(v, f, n, segments, params, step=0.3, groups=None, coordinates=None,
                 height_values=None):
    paths = trace_rib_paths(v, f, n, segments, params, step, groups, coordinates, height_values)
    return ribbons_from_paths(paths, params), len(paths)


def _round_union(fields, radius):
    m = fields.min(axis=0)
    if np.all(np.asarray(radius) <= 0):
        return m
    return np.maximum(radius, m)-np.sqrt((np.maximum(radius-fields, 0)**2).sum(axis=0))


def _round_intersection(a, b, radius):
    """Inset circular fillet: preserve the two planes outside the corner."""
    a, b = a+radius, b+radius
    return (np.minimum(np.maximum(a, b), 0)
            + np.hypot(np.maximum(a, 0), np.maximum(b, 0))-radius)


def _crown_envelope(sides, tops, reach, smoothing):
    """Blend nearby crown directions without raising a constant-height crown.

    A hard minimum switches abruptly between the guide planes at a curved
    junction. The normalized soft minimum is exact for equal heights and a
    single contributor, and never lies below the lowest contributing field.
    Compact lateral weights prevent remote high ribs from lifting a run-out.
    """
    capped = np.maximum(tops, sides-reach)
    crown = capped.min(axis=0)
    if smoothing <= 0 or reach <= 0 or len(tops) == 1:
        return crown
    t = np.clip(np.maximum(sides, 0)/reach, 0, 1)
    weights = (1-t)**4*(4*t+1)
    total = weights.sum(axis=0)
    active = total > 0
    low = np.where(weights[:, active] > 0, tops[:, active], np.inf).min(axis=0)
    exponent = np.minimum(-(tops[:, active]-low)/smoothing, 0)
    average = (weights[:, active]*np.exp(exponent)).sum(axis=0)/total[active]
    crown[active] = low-smoothing*np.log(average)
    return crown


class RibField:
    def __init__(self, ribbons, body_v, body_f, params, resolution=.25,
                 cull=True, substrate=None, excluded=None, height_values=None):
        v, f = _weld(body_v, body_f)
        self.body = MeshDistance(v, f, signed=True)
        self.ribbons = ribbons
        self.params = params
        self.substrate = (MeshDistance(*substrate, signed=True)
                          if substrate is not None else None)
        self.excluded = MeshDistance(*excluded) if excluded is not None else None
        self.cull = cull
        self.height_values = height_values
        self.max_height = (params.height*float(np.max(height_values))
                           if height_values is not None else params.height)
        vertices = np.vstack([r.mesh.v for r in ribbons])
        offsets = np.cumsum([0]+[len(r.mesh.v) for r in ribbons[:-1]])
        faces = np.vstack([r.mesh.f+off for r, off in zip(ribbons, offsets)])
        self.support = MeshDistance(vertices, faces)
        # A blended surface can only involve a ribbon field below max(k).
        # Bound that sublevel set by a distance to the ribbon skeleton,
        # including draft and two voxels for complete marching cells.
        k = max(params.fillet_root, params.fillet_junction, 0)
        draft = abs(math.tan(math.radians(params.draft_deg)))
        width = params.thickness/2 + k + draft*(
            self.max_height+max(params.embed, .5)+max(params.fillet_root, 0))
        self.reach_base = math.hypot(width, k)
        self.reach = self.reach_base+2*resolution

    def __call__(self, x):
        p = self.params
        result = np.empty(len(x), np.float32)
        for start in range(0, len(x), 250000):
            chunk = x[start:start+250000]
            active = (self.support.query(chunk)[0] <= self.reach**2
                      if self.cull else np.ones(len(chunk), bool))
            values = np.full(len(chunk), self.reach, np.float32)
            if not active.any():
                result[start:start+len(chunk)] = values
                continue
            q = chunk[active]
            body = self.body.signed(q)
            distances = [r.distances(q, p.thickness/2,
                                     math.tan(math.radians(p.draft_deg)), p.height)
                         for r in self.ribbons]
            sides = np.stack([d[0] for d in distances])
            tops = np.stack([d[1] for d in distances])
            top_radius = min(p.fillet_top, p.thickness/2, p.height)
            junction = max(p.fillet_junction, 0)
            k = max(p.fillet_root, 0)
            # Extend each wall only far enough to construct the local blend.
            # The crown envelope includes the taper and manual height fields,
            # but is bounded laterally: a tall, distant rib cannot lift a tip.
            extension = max(k, junction)+top_radius
            crowns = _crown_envelope(sides, tops, extension,
                                      min(p.thickness/20, junction/10))
            wall = _round_union(np.maximum(sides, tops-extension), junction)
            value = wall
            if k:
                # A root fillet needs room below the local crown. Let its
                # radius vanish with the run-out instead of trimming a full
                # sized blend into a thin, broad tab at a zero-height tip.
                room = np.clip(body-crowns, 0, 2*k)
                root_radius = room*(1-room/(4*k))
                value = _round_union(np.stack([wall, body+.05]), root_radius)
                # Only a local root footprint is needed. A whole-body skin
                # would obscure the preview and mask disconnected graph defects.
                value = np.maximum(value, wall-root_radius)
            if self.height_values is not None:
                # A locally shortened rib must retain a feasible crown radius.
                heights = np.stack([np.broadcast_to(d[2], len(q)) for d in distances])
                owner = np.maximum(tops, sides-extension).argmin(axis=0)
                top_radius = np.minimum(top_radius, heights[owner, np.arange(len(q))])
            value = _round_intersection(value, crowns, top_radius)
            # Trim after both blends: even a root radius larger than the
            # rib height must not lift the crown above the height offset.
            query = None
            local_height = self.max_height
            if self.height_values is not None and self.substrate is not None:
                query = self.substrate.query(q) if query is None else query
                bary = self.substrate.barycentric(query[2], query[1])
                local_height = p.height*np.einsum('ij,ij->i', self.height_values[self.substrate.f[query[1]]], bary)
            value = np.maximum(value, body-local_height)
            depth = imp._slab_depth(p)
            value = np.maximum(value, -body-depth)
            if self.substrate is not None:
                query = self.substrate.query(q) if query is None else query
                selected = self.substrate.signed(q, query)
                # A closed body has TWO exterior sides. Its signed distance
                # alone cannot stop a root blend growing out of the back of
                # a thin wall. Outside the body only the positive side of the
                # selected surface may contain ribs; the embedded overlap is
                # allowed inside the body and bounded by the selected face.
                value = np.maximum(value, np.minimum(body, -selected))
                value = np.maximum(value, -selected-depth)
                if self.excluded is not None:
                    # Root blends may broaden beyond their centerline. Outside
                    # the body, retain only points owned by a selected face.
                    # Embedded material remains available for a solid union.
                    other_sq = self.excluded.query(q)[0]
                    ownership = np.sqrt(query[0])-np.sqrt(other_sq)
                    # Above a shared sharp edge both closest points can be
                    # the SAME edge over an entire wedge of space. A zero
                    # ownership plateau would itself become a marching-cubes
                    # surface. Resolve that tie by the tangential displacement
                    # from the selected face, which is positive outside it.
                    tied = np.abs(query[0]-other_sq) < 1e-9
                    if tied.any():
                        delta = q[tied]-query[2][tied]
                        normal = self.substrate.fn[query[1][tied]]
                        tangent = delta-(delta*normal).sum(1)[:, None]*normal
                        ownership[tied] = np.linalg.norm(tangent, axis=1)
                    value = np.maximum(value, np.minimum(body, ownership))
            values[active] = value
            result[start:start+len(chunk)] = values
        return result


def extraction_grid(v, f, n, ribbons, p, res, ctr=None, matrix=None):
    """Conservative bands rasterized from the actual extruded ribbons."""
    rv = np.vstack([r.mesh.v for r in ribbons])
    offsets = np.cumsum([0]+[len(r.mesh.v) for r in ribbons[:-1]])
    rf = np.vstack([r.mesh.f+off for r, off in zip(ribbons, offsets)])
    spread = (p.thickness/2 + max(p.fillet_root, p.fillet_junction)
              + abs(math.tan(math.radians(p.draft_deg)))*max(
                  [p.height]+[float(np.max(r.height_values)) for r in ribbons
                              if r.height_values is not None]) + 3*res)
    cell = max(res, .25)
    origin = rv[:, :2].min(0)-spread-2*cell
    hi = rv[:, :2].max(0)+spread+2*cell
    while np.prod(np.ceil((hi-origin)/cell)+1) > 4.5e7:
        cell *= 1.4
    shape = tuple(np.ceil((hi-origin)/cell).astype(int)+1)
    # These are conservative query bounds, not geometry. Rasterizing each
    # triangle's complete depth interval avoids a costly exact pixel clip
    # on the many nearly edge-on triangles of a ribbon wall.
    lo = np.full(shape, 1e10, np.float32)
    high = np.full(shape, -1e10, np.float32)
    tris = rv[rf]
    mins, maxs = tris.min(1), tris.max(1)
    aa = np.maximum(np.floor((mins[:, :2]-origin)/cell).astype(int), 0)
    bb = np.minimum(np.ceil((maxs[:, :2]-origin)/cell).astype(int)+1, shape)
    for a, b, mn, mx in zip(aa, bb, mins[:, 2], maxs[:, 2]):
        sl = (slice(a[0], b[0]), slice(a[1], b[1]))
        np.minimum(lo[sl], mn, out=lo[sl])
        np.maximum(high[sl], mx, out=high[sl])
    radius = 2*int(math.ceil(spread/cell))+1
    lo = minimum_filter(lo, size=radius)-spread
    high = maximum_filter(high, size=radius)+spread
    mask = lo < 1e8
    lo[~mask], high[~mask] = 0, 0
    zero = np.zeros(shape, np.float32)
    return imp.SurfaceField(cell=cell, origin=tuple(origin), P=zero, B=zero,
                            Bo=zero, mask=mask, V=v, F=f, N=n,
                            dlo=lo, dhi=high, ctr=ctr, M=matrix)


def attached_components(clusters, body, warnings):
    """Reject unsupported extraction islands using the export body itself.

    A volume cutoff loses small, legitimate ribs. Conversely, touching one
    vertex is insufficient: a sampled grazing contact can disappear during
    the solid union. Check the union's components for actual body overlap.
    """
    import manifold3d as m3d
    from .ribbing import RibbingError
    parts = []
    for v, f in clusters:
        solid = m3d.Manifold(m3d.Mesh64(np.require(v, np.float64, ["C", "W"]),
                                      np.require(f, np.uint64, ["C", "W"])))
        if solid.is_empty():
            raise RibbingError("cannot check attachment of an invalid rib solid")
        parts.extend(solid.decompose())
    # Topological decomposition returns internal cavities as negative shells.
    # Reassociate those with their enclosing material before boolean union;
    # unioning negative shells as if they were solids would fill the cavities.
    cavities = [p for p in parts if p.volume() < 0]
    parts = [p for p in parts if p.volume() > 0]
    for cavity in cavities:
        mesh = cavity.to_mesh64()
        hole = m3d.Manifold(m3d.Mesh64(
            np.array(mesh.vert_properties[:, :3], copy=True),
            np.array(mesh.tri_verts[:, ::-1], dtype=np.uint64, order="C")))
        parents = [i for i, part in enumerate(parts)
                   if (hole-part).volume() <= 1e-9]
        if not parents:
            raise RibbingError("rib cavity has no enclosing material")
        parent = min(parents, key=lambda i: parts[i].volume())
        parts[parent] = parts[parent]-hole
    kept, removed = [], []
    for part in parts:
        joined = part + body
        if joined.is_empty():
            raise RibbingError("rib attachment union failed")
        unsupported = False
        for piece in joined.decompose():
            if piece.volume() <= 0:
                continue
            overlap = piece ^ body
            if overlap.volume() <= 1e-9:
                unsupported = True
                break
        (removed if unsupported else kept).append(part)
    if not kept:
        raise RibbingError("rib graph has no solid attachment to the body")
    if removed:
        warnings.append(f"removed {len(removed)} unsupported extraction fragment(s), "
                        f"{sum(p.volume() for p in removed):.3f}mm3; no stable body union")
    # Smoothing can make formerly disjoint shells overlap. Concatenating their
    # arrays is not a boolean union and can create detached remnants at export.
    normalized = m3d.Manifold.batch_boolean(kept, m3d.OpType.Add)
    if normalized.is_empty():
        raise RibbingError("rib component union failed")
    mesh = normalized.to_mesh64()
    return [(np.array(mesh.vert_properties[:, :3], dtype=np.float64, copy=True),
             np.array(mesh.tri_verts, dtype=np.int64, copy=True))]


_GRAPH_CACHE_ENTRIES = 4
_GRAPH_CACHE_BYTES = 192 * 1024**2


@dataclass(frozen=True)
class _ShapeIdentity:
    # Retain the wrapper while cached, but do not hash the mutable OCCT
    # object itself: Move/Reverse can change its hash or orientation.
    shape: object = dataclass_field(compare=False, hash=False, repr=False)
    identity: int
    orientation: int
    location: tuple


def _shape_identity(shape):
    transform = shape.Location().Transformation()
    return _ShapeIdentity(shape, id(shape), int(shape.Orientation()),
                          tuple(transform.Value(i, j) for i in range(1, 4) for j in range(1, 5)))


def _cache_size(value, seen=None):
    """Conservative resident estimate, including native BVH storage."""
    if seen is None:
        seen = set()
    if id(value) in seen:
        return 0
    seen.add(id(value))
    if isinstance(value, np.ndarray):
        return value.nbytes
    if isinstance(value, MeshDistance):
        return sum(_cache_size(v, seen) for v in vars(value).values()) + len(value.f)*128
    if isinstance(value, dict):
        return sum(_cache_size(v, seen) for v in value.values())
    if isinstance(value, (tuple, list)):
        return sum(_cache_size(v, seen) for v in value)
    if isinstance(value, RibField) or is_dataclass(value):
        return sum(_cache_size(v, seen) for v in vars(value).values())
    if hasattr(value, 'num_tri') and hasattr(value, 'num_vert'):
        return value.num_tri()*64+value.num_vert()*48
    return 0


def _trim_graph_cache(cache):
    if cache is None:
        return
    while cache and (len(cache) > _GRAPH_CACHE_ENTRIES
                     or sum(_cache_size(entry) for entry in cache.values()) > _GRAPH_CACHE_BYTES):
        cache.popitem(last=False)


def _graph_result(graph, entry=None):
    # Cache geometry is immutable. Callers own all mutable recipe/report
    # containers, particularly warnings, which the solid builder appends to.
    result = dict(graph)
    result['paths'] = [dict(path) for path in graph['paths']]
    result['reports'] = copy.deepcopy(graph['reports'])
    result['controls'] = copy.deepcopy(graph['controls'])
    result['warnings'] = list(graph['warnings'])
    if entry is not None:
        result['_cached_resources'] = entry['resources']
    return result


def _require_face_coverage(expected, present, scope):
    """OCCT can finish successfully while silently dropping entire faces."""
    missing = sorted(set(expected)-set(present))
    if missing:
        from .ribbing import RibbingError
        ids = ', '.join(map(str, missing[:30]))
        if len(missing) > 30:
            ids += f' (and {len(missing)-30} more)'
        raise RibbingError(
            f'CAD tessellation is incomplete: {scope} faces {ids} have no usable triangles. '
            'Generation stopped to avoid using an open or incomplete surface; '
            'the CAD mesher could not preserve these faces at the required resolution.')


def _selection_regions(shape, face_ids, lin_defl, frame_cache):
    """Canonical CAD tessellation, reusable across parameter-only edits."""
    from OCP.BRepTools import BRepTools
    from .meshing import region_meshes
    key = (_shape_identity(shape), tuple(face_ids), float(min(lin_defl, .15)))
    sources = frame_cache.setdefault('_graph_sources', OrderedDict()) if frame_cache is not None else None
    if sources is not None and key in sources:
        sources.move_to_end(key)
        return list(sources[key])
    # OCCT retains finer triangulations from earlier display/export calls.
    # Establish the requested resolution once rather than inheriting that
    # hidden state; source arrays remain valid after later CAD remeshing.
    BRepTools.Clean_s(shape)
    regions = region_meshes(shape, face_ids, min(lin_defl, .15), ang_defl=.075)
    # Region metadata retains requested face IDs even when tessellation or
    # degenerate-triangle cleanup removed their actual geometry.
    _require_face_coverage(face_ids,
                          (int(fid) for region in regions for fid in np.unique(region.tri_face)),
                          'selected')
    if sources is not None:
        for region in regions:
            for value in vars(region).values():
                if isinstance(value, np.ndarray):
                    value.setflags(write=False)
        sources[key] = regions
        while sources and (len(sources) > 3 or sum(_cache_size(regions) for regions in sources.values()) > 64*1024**2):
            sources.popitem(last=False)
    return list(regions)


def _body_source(shape, lin_defl, frame_cache):
    """Canonical whole-body arrays survive OCCT display/export remeshing."""
    from OCP.BRepTools import BRepTools
    from .meshing import mesh_shape
    from .step_io import face_map
    key = (_shape_identity(shape), float(min(lin_defl, .15)))
    sources = frame_cache.setdefault('_graph_body_sources', OrderedDict()) if frame_cache is not None else None
    if sources is not None and key in sources:
        sources.move_to_end(key)
        return sources[key]
    BRepTools.Clean_s(shape)
    meshes = mesh_shape(shape, min(lin_defl, .15), .075)
    # Missing excluded faces are equally unsafe: their holes would corrupt
    # the body's signed distance and attachment/collision constraints.
    _require_face_coverage(range(1, face_map(shape).Size()+1),
                          (mesh.face_id for mesh in meshes if len(mesh.triangles)),
                          'body')
    vertices, triangles, face_ids, offset = [], [], [], 0
    for mesh in meshes:
        vertices.append(np.asarray(mesh.vertices, np.float64))
        triangles.append(np.asarray(mesh.triangles, np.int64)+offset)
        face_ids.append(np.full(len(mesh.triangles), mesh.face_id, np.int64))
        offset += len(mesh.vertices)
    arrays = (np.vstack(vertices), np.vstack(triangles), np.concatenate(face_ids))
    for value in arrays:
        value.setflags(write=False)
    if sources is not None:
        sources[key] = arrays
        while sources and (len(sources) > 2 or sum(_cache_size(value) for value in sources.values()) > 64*1024**2):
            sources.popitem(last=False)
    return arrays


def prepare_rib_graph(shape, face_ids, params, lin_defl=.15, frame_cache=None, progress=None):
    from .local_mapping import validate_controls, apply_local_controls
    from .ribbing import RibbingError, RibReport, _merge_regions, _projection_frame, _lattice_window
    if params.mapping not in ("surface", "project"):
        raise RibbingError("the graph engine requires surface or project mapping")
    if not face_ids:
        raise RibbingError("no faces selected")
    face_ids = sorted(set(face_ids))
    if params.thickness <= 0 or params.height <= 0:
        raise RibbingError("rib thickness and height must be positive")
    controls = validate_controls(params.mapping_controls, face_ids)
    if controls and params.mapping != "surface":
        raise RibbingError("local controls require Surface mapping and the graph engine")
    if controls and params.pattern == "stochastic":
        raise RibbingError("local mapping controls currently require a regular lattice pattern")
    if params.spacing <= 0:
        raise RibbingError("rib spacing must be positive")
    report = imp._safe_progress(progress)
    cache, key = None, None
    if frame_cache is not None:
        try:
            # Holding the OCCT shape in the key prevents Python-id reuse and
            # distinguishes translated/oriented instances of the same TShape.
            key = (_shape_identity(shape), tuple(face_ids), float(min(lin_defl, .15)),
                   json.dumps(asdict(params), sort_keys=True, separators=(',', ':'), allow_nan=False))
            hash(key)
        except (TypeError, ValueError):
            key = None
        if key is not None:
            cache = frame_cache.setdefault('_graph_prepared', OrderedDict())
            entry = cache.get(key)
            if entry is not None:
                cache.move_to_end(key)
                if report:
                    report("reusing prepared rib graph", 1, 1)
                return _graph_result(entry['graph'], entry)
    if report:
        report("meshing selection", 0, 0)
    regions = _selection_regions(shape, face_ids, lin_defl, frame_cache)
    if not regions:
        raise RibbingError("selected faces could not be triangulated")
    if len(regions) > 1 and params.mapping == "project":
        regions = [_merge_regions(regions)]
    ctr, axes = _projection_frame(regions)
    # The frame belongs to this shape/selection. Reusing a nearby frame from
    # an earlier selection made identical recipes depend on editing history.
    matrix = np.column_stack([axes, np.cross(axes[:, 0], axes[:, 1])])
    paths, reports, vertices, triangles, normals, heights = [], [], [], [], [], []
    projected_controls, off = [], 0
    for region in regions:
        v = np.ascontiguousarray((region.vertices-ctr)@matrix)
        f = np.asarray(region.triangles, np.int64)
        # Keep the CAD chart seams for flattening. They can be welded only
        # after curves have been mapped back to their physical triangles.
        if params.mapping == "project":
            v, f = _weld(v, f)
        n = guide_normals(v, f, imp._vertex_normals(v, f), params.guide_smoothing)
        coordinates = v[:, :2]
        if params.mapping == "surface":
            from .surface_mapping import surface_coordinates
            if report:
                report("mapping surface distances", 0, 0)
            coordinates = surface_coordinates(v, f, coordinates, frame_cache)
        local_controls = [dict(c, position=((np.asarray(c['position'])-ctr)@matrix).tolist())
                          for c in controls if c['face_id'] in region.face_ids]
        hscale = None
        if local_controls:
            v, f, coordinates, n, hscale, projected = apply_local_controls(
                v, f, coordinates, n, region.tri_face, local_controls, frame_cache)
            if not any(c['height_scale'] != 1 for c in local_controls):
                hscale = None
            projected_controls.extend([dict(c, position=(np.asarray(c['position'])@matrix.T+ctr).tolist())
                                       for c in projected])
        bounds = (*coordinates.min(0), *coordinates.max(0))
        segments = list(generate_segments(params, _lattice_window(params, bounds)))
        groups = _segment_groups(segments, params.pattern)
        if params.border:
            import shapely
            # Never close narrow unselected strips or holes in the domain.
            domain = shapely.union_all(shapely.polygons(coordinates[f]))
            inset = domain.buffer(-params.margin-params.thickness/2)
            if inset.is_empty:
                raise RibbingError("margin leaves no border rib area")
            border_group = max(groups, default=-1)+1
            for ring in imp._ring_chains(inset, max(params.spacing/6, 1.)):
                segments.extend(zip(ring[:-1], ring[1:]))
                groups.extend([border_group]*(len(ring)-1))
        if report:
            report("tracing rib graph", 0, 0)
        region_paths = trace_rib_paths(v, f, n, segments, params, groups=groups,
                                      coordinates=coordinates, height_values=hscale)
        # Regions must not share a ribbon group: each has its own geometry.
        for path in region_paths:
            path['group'] = (len(reports), path['group'])
        paths.extend(region_paths)
        count = len(region_paths)
        reports.append(RibReport(face_id=region.face_ids[0], face_ids=region.face_ids,
                                  segments=len(segments), lofted=count))
        vertices.append(v)
        triangles.append(f+off)
        normals.append(n)
        heights.append(hscale if hscale is not None else np.ones(len(v)))
        off += len(v)
    if not paths:
        raise RibbingError("pattern and margin leave no ribs on the selection")
    if len(projected_controls) != len(controls):
        raise RibbingError("a control face could not be triangulated; place it again on the selected surface")
    graph = dict(paths=paths, reports=reports, vertices=np.vstack(vertices),
                triangles=np.vstack(triangles), normals=np.vstack(normals),
                height_values=(np.concatenate(heights) if any(c['height_scale'] != 1 for c in controls) else None),
                controls=projected_controls, center=ctr, matrix=matrix,
                warnings=(["Local controls prescribe design fields, not stresses. "
                           "Influence follows intrinsic mesh surface distances; existing chart seams remain boundaries."]
                          if controls else []))
    if cache is not None:
        for value in graph.values():
            if isinstance(value, np.ndarray):
                value.setflags(write=False)
        for path in paths:
            for value in path.values():
                if isinstance(value, np.ndarray):
                    value.setflags(write=False)
        entry = dict(graph=graph, resources={})
        cache[key] = entry
        _trim_graph_cache(cache)
        return _graph_result(graph, entry)
    return graph


def preview_rib_graph(shape, face_ids, params, frame_cache=None):
    """Lightweight base/crown curves; no solid sampling, extraction or union."""
    start = time.perf_counter()
    graph = prepare_rib_graph(shape, face_ids, params, frame_cache=frame_cache)
    ctr, matrix = graph['center'], graph['matrix']
    paths = [dict(points=(path['points']@matrix.T+ctr).ravel().tolist(),
                  top=((path['points']+path['height'][:, None]*path['normals'])@matrix.T+ctr).ravel().tolist())
             for path in graph['paths']]
    return dict(paths=paths, controls=graph['controls'], warnings=graph['warnings'],
                stats=dict(paths=len(paths), points=sum(len(p['points'])//3 for p in paths),
                           controls=len(graph['controls']), triangles=len(graph['triangles']),
                           mapping_seconds=round(time.perf_counter()-start, 3)))


def build_rib_graph(shape, face_ids, params, lin_defl=.15, quality=1.,
                    frame_cache=None, progress=None):
    from .ribbing import RibbingError
    t0 = time.time()
    report = imp._safe_progress(progress)
    graph = prepare_rib_graph(shape, face_ids, params, lin_defl, frame_cache, progress)
    paths, reports = graph['paths'], graph['reports']
    v, f, n = graph['vertices'], graph['triangles'], graph['normals']
    ctr, matrix = graph['center'], graph['matrix']
    resources = graph.get('_cached_resources', {})
    ribbons = resources.get('ribbons')
    if ribbons is None:
        ribbons = ribbons_from_paths(paths, params)
        resources['ribbons'] = ribbons

    if not ribbons:
        raise RibbingError("pattern and margin leave no ribs on the selection")
    res = float(np.clip(min(.25/max(quality, .5), params.thickness/6), .08, .5))
    field = resources.get('field')
    if field is None:
        world_v, body_f, tri_face = _body_source(shape, lin_defl, frame_cache)
        body_v = (world_v-ctr)@matrix
        excluded_faces = body_f[~np.isin(tri_face, face_ids)]
        excluded = (body_v, excluded_faces) if len(excluded_faces) else None
        field = RibField(ribbons, body_v, body_f, copy.deepcopy(params), resolution=res,
                         substrate=(v, f) if graph['height_values'] is not None else _weld(v, f),
                         excluded=excluded, height_values=graph['height_values'])
        resources['field'] = field
    else:
        # Resolution changes only conservative culling padding, never the
        # prepared curves, ribbon geometry, signed body, or spatial indices.
        field = copy.copy(field)
        field.reach = field.reach_base+2*res
    grids = resources.setdefault('grids', OrderedDict())
    grid = grids.get(res)
    if grid is None:
        grid = extraction_grid(v, f, n, ribbons, params, res, ctr, matrix)
        grids[res] = grid
        while len(grids) > 2:
            grids.popitem(last=False)
    else:
        grids.move_to_end(res)
    if frame_cache is not None:
        _trim_graph_cache(frame_cache.get('_graph_prepared'))
    warnings = list(graph["warnings"])
    clusters = imp.mesh_field(grid, params, resolution=res, reports=warnings,
                              field=field, progress=report)
    if not clusters:
        raise RibbingError("rib graph produced no geometry")
    if report:
        report("checking body attachment", 0, 0)
    from .booleans import _to_manifold
    body = resources.get('attachment_body')
    if body is None:
        body = _to_manifold(shape, .2)
        resources['attachment_body'] = body
        if frame_cache is not None:
            _trim_graph_cache(frame_cache.get('_graph_prepared'))
    clusters = attached_components([(vv@matrix.T+ctr, ff) for vv, ff in clusters],
                                   body, warnings)
    reports[0].warnings = warnings+[f"{params.mapping} graph: "
        f"{sum(r.lofted for r in reports)} paths, {res:.2f}mm voxels, {time.time()-t0:.1f}s"]
    return clusters, reports
