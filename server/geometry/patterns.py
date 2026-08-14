import math
from dataclasses import dataclass, fields

import numpy as np
from scipy.spatial import Voronoi
from shapely.geometry import LineString, MultiLineString, Polygon


@dataclass
class RibParams:
    pattern: str = "rectangular"
    spacing: float = 12.0
    spacing_y: float | None = None
    thickness: float = 1.6
    height: float = 4.0
    orientation_deg: float = 0.0
    draft_deg: float = 0.0
    margin: float = 2.0
    embed: float = 0.3
    border: bool = False
    density: float = 0.008
    seed: int = 1
    base_angle_deg: float = 0.0
    taper_len: float = 5.0   # run-out length at open boundaries (0 = off)

    @classmethod
    def from_dict(cls, d):
        keys = {f.name for f in fields(cls)}
        clean = {}
        for k, v in d.items():
            if k not in keys:
                continue
            if v is None or v == "":
                continue
            clean[k] = v
        return cls(**clean)


def _corners(bounds):
    minx, miny, maxx, maxy = bounds
    return np.array([[minx, miny], [maxx, miny], [maxx, maxy], [minx, maxy]])


def _family(bounds, spacing, angle_deg):
    """Parallel lines at angle_deg, minimally covering bounds."""
    if spacing <= 0:
        return []
    c = _corners(bounds)
    center = c.mean(axis=0)
    a = math.radians(angle_deg)
    d = np.array([math.cos(a), math.sin(a)])
    n = np.array([-d[1], d[0]])
    projn = (c - center) @ n
    projd = (c - center) @ d
    lo, hi = projd.min() - spacing, projd.max() + spacing
    segs = []
    for k in range(int(math.floor(projn.min() / spacing)),
                   int(math.ceil(projn.max() / spacing)) + 1):
        o = center + n * (k * spacing)
        segs.append((tuple(o + d * lo), tuple(o + d * hi)))
    return segs


def _rotate_segments(segs, bounds, angle_deg):
    if not angle_deg:
        return segs
    c = _corners(bounds).mean(axis=0)
    a = math.radians(angle_deg)
    R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    out = []
    for p0, p1 in segs:
        q0 = c + R @ (np.asarray(p0) - c)
        q1 = c + R @ (np.asarray(p1) - c)
        out.append((tuple(q0), tuple(q1)))
    return out


def _grow(bounds, f):
    minx, miny, maxx, maxy = bounds
    dx, dy = (maxx - minx) * f, (maxy - miny) * f
    return minx - dx, miny - dy, maxx + dx, maxy + dy


def _hex_walls(bounds, size):
    """Pointy-top honeycomb walls; size = across-flats (wall-to-wall) distance."""
    if size <= 0:
        return []
    minx, miny, maxx, maxy = _grow(bounds, 0.35)
    s = size / math.sqrt(3.0)          # hex edge length / circumradius
    dx, dy = size, 1.5 * s
    walls = {}
    j0, j1 = int(math.floor(miny / dy)) - 1, int(math.ceil(maxy / dy)) + 1
    i0, i1 = int(math.floor(minx / dx)) - 1, int(math.ceil(maxx / dx)) + 1
    for j in range(j0, j1 + 1):
        cy = j * dy
        off = dx / 2 if j % 2 else 0.0
        for i in range(i0, i1 + 1):
            cx = i * dx + off
            pts = [(cx + s * math.cos(math.radians(30 + 60 * k)),
                    cy + s * math.sin(math.radians(30 + 60 * k))) for k in range(6)]
            for k in range(6):
                a, b = pts[k], pts[(k + 1) % 6]
                mid = (round((a[0] + b[0]) / 2, 4), round((a[1] + b[1]) / 2, 4))
                walls.setdefault(mid, (a, b))
    return list(walls.values())


def _stochastic(bounds, density, seed):
    minx, miny, maxx, maxy = _grow(bounds, 0.15)
    area = (maxx - minx) * (maxy - miny)
    n = max(8, int(area * density))
    rng = np.random.default_rng(seed)
    pts = rng.uniform([minx, miny], [maxx, maxy], size=(n, 2))
    vor = Voronoi(pts)
    segs = []
    for a, b in vor.ridge_vertices:
        if a >= 0 and b >= 0:
            pa, pb = vor.vertices[a], vor.vertices[b]
            if np.isfinite(pa).all() and np.isfinite(pb).all():
                segs.append((tuple(pa), tuple(pb)))
    return segs


def generate_segments(params, bounds):
    """Segment network covering bounds, rotated by orientation_deg."""
    p, sp, rot = params.pattern, params.spacing, params.orientation_deg
    base = params.base_angle_deg
    if p in ("rectangular", "quadmesh"):
        sy = sp if p == "quadmesh" else (
            params.spacing_y if params.spacing_y is not None else sp)
        segs = _family(bounds, sp, 90.0 + rot)
        if sy and sy > 0:
            segs += _family(bounds, sy, 0.0 + rot)
        return segs
    if p in ("triangular", "isogrid"):
        a0 = base + rot
        return (_family(bounds, sp, a0) + _family(bounds, sp, a0 + 60)
                + _family(bounds, sp, a0 + 120))
    if p == "hexagonal":
        return _rotate_segments(_hex_walls(bounds, sp), bounds, rot)
    if p == "stochastic":
        return _rotate_segments(_stochastic(bounds, params.density, params.seed),
                                bounds, rot)
    raise ValueError(f"unknown pattern: {p}")


def clip_and_border(segments, boundary, params):
    """Clip segments to boundary inset by margin; add border ring edges."""
    inset = boundary.buffer(-params.margin)
    if inset.is_empty:
        return []
    out = []
    # stubs shorter than ~a thickness read as spiky slivers at region borders
    min_len = max(1.2, params.thickness * 1.2)
    if segments:
        inter = inset.intersection(MultiLineString([LineString(s) for s in segments]))
        stack = [inter]
        while stack:
            g = stack.pop()
            if g.is_empty:
                continue
            if isinstance(g, LineString):
                if g.length > min_len:
                    out.append(g)
            elif hasattr(g, "geoms"):
                stack.extend(g.geoms)
    if params.border:
        # uniform arc-length resampling: raw region boundaries carry
        # thousands of mesh-edge vertices and would explode into thousands
        # of micro-ribs (minutes-long applies)
        step_b = float(min(max(params.spacing / 3.0, 2.0), 6.0))
        budget = 1200
        for poly in getattr(inset, "geoms", [inset]):
            for ring in [poly.exterior, *poly.interiors]:
                L = ring.length
                if L < 4 * params.thickness:
                    continue
                # slit artifacts: long rings enclosing ~no area
                ring_area = abs(Polygon(ring).area)
                if ring_area < params.thickness * L:
                    continue
                n = max(8, int(L / step_b))
                if n > budget:
                    continue
                budget -= n
                pts = [ring.interpolate(i * L / n) for i in range(n + 1)]
                for i in range(n):
                    out.append(LineString([pts[i], pts[i + 1]]))
    return out


def capsule(p0, p1, half_width, step=2.5, cap_pts=5):
    """Closed CCW stadium loop around segment p0-p1 (last point not repeated).

    Same segment + step => same point count for any half_width, so a
    narrower top loop pairs 1:1 with its bottom loop for ruled lofting.
    """
    p0 = np.asarray(p0, float)
    p1 = np.asarray(p1, float)
    w = max(float(half_width), 1e-3)
    d = p1 - p0
    L = float(np.linalg.norm(d))
    d = d / L if L > 1e-9 else np.array([1.0, 0.0])
    n = np.array([-d[1], d[0]])
    ns = max(2, int(math.ceil(L / step)) + 1)
    ts = np.linspace(0.0, L, ns)
    pts = []
    pts += [p0 + d * t - n * w for t in ts]                    # bottom side ->
    ang = np.linspace(-90, 90, cap_pts + 2)[1:-1]
    pts += [p1 + w * (math.cos(math.radians(a)) * d
                      + math.sin(math.radians(a)) * n) for a in ang]   # cap at p1
    pts += [p1 + n * w - d * t for t in ts]                    # top side <-
    ang = np.linspace(90, 270, cap_pts + 2)[1:-1]
    pts += [p0 + w * (math.cos(math.radians(a)) * d
                      + math.sin(math.radians(a)) * n) for a in ang]   # cap at p0
    return np.array(pts)
