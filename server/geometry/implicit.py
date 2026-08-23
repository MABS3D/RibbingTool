"""Implicit (SDF) rib engine: one scalar field, meshed once.

Ribs, root blends, crowns, taper and draft compose as field math in the
projection frame; the boolean with the body is a smooth-min, so every
rib-rib and rib-body junction blends uniformly and cannot fail.  The field
is evaluated tile by tile on ONE global voxel grid (absolute multiples of
the resolution on all three axes), so marching-cubes output from adjacent
tiles welds exactly at shared planes.
"""
import math
import time
from dataclasses import dataclass

import numpy as np

_AIR = np.float32(1e3)          # field value outside the domain
_SINK = 0.05                    # slab top sits this far under the surface


@dataclass
class FrameGrids:
    """Pattern-space rasters, all (nu, nv), indexed [u, v]."""
    cell: float                 # grid pitch (mm)
    origin: tuple               # (u0, v0) of grid[0, 0]
    D: np.ndarray               # front-most surface depth (position . n)
    P: np.ndarray               # 2D distance to rib centerlines
    B: np.ndarray               # distance to the domain boundary (taper)
    mask: np.ndarray            # inside the closed, margin-inset domain


def _bilinear(grid, u, v, cell, origin):
    gu = (u - origin[0]) / cell
    gv = (v - origin[1]) / cell
    i0 = np.clip(gu.astype(np.int32), 0, grid.shape[0] - 2)
    j0 = np.clip(gv.astype(np.int32), 0, grid.shape[1] - 2)
    fu = np.clip(gu - i0, 0.0, 1.0).astype(np.float32)
    fv = np.clip(gv - j0, 0.0, 1.0).astype(np.float32)
    return (grid[i0, j0] * (1 - fu) * (1 - fv)
            + grid[i0 + 1, j0] * fu * (1 - fv)
            + grid[i0, j0 + 1] * (1 - fu) * fv
            + grid[i0 + 1, j0 + 1] * fu * fv)


def _slab_depth(params):
    return params.embed + max(params.fillet_root, 0.0) + 0.6


def field(u, v, d, grids, params):
    """Signed field (negative = material) at frame-space points (u, v, d)."""
    g = grids
    D = _bilinear(g.D, u, v, g.cell, g.origin)
    P = _bilinear(g.P, u, v, g.cell, g.origin)
    inside = _bilinear(g.mask.astype(np.float32), u, v, g.cell,
                       g.origin) > 0.5
    # beyond the raster there is no surface: cap the field as air there,
    # or marching cubes leaves open sheets at the sampling box walls
    inside &= ((u >= g.origin[0]) & (v >= g.origin[1])
               & (u <= g.origin[0] + (g.D.shape[0] - 1) * g.cell)
               & (v <= g.origin[1] + (g.D.shape[1] - 1) * g.cell))

    height = np.float32(params.height)
    if params.taper_len > 0 and not params.border:
        B = _bilinear(g.B, u, v, g.cell, g.origin)
        height = height * np.clip(B / params.taper_len, 0.0, 1.0)
    top = D + height

    half = np.float32(params.thickness / 2.0)
    if params.draft_deg > 0:
        half = half + math.tan(math.radians(params.draft_deg)) \
            * np.clip(top - d, 0.0, None)

    wall = P - half
    r_top = float(np.clip(params.fillet_top, 0.0, 0.35 * params.thickness))
    if r_top > 0:
        # rounded corner between wall and top plane
        a = wall + r_top
        b = (d - top) + r_top
        prism = (np.minimum(np.maximum(a, b), 0.0)
                 + np.hypot(np.clip(a, 0.0, None), np.clip(b, 0.0, None))
                 - r_top)
    else:
        prism = np.maximum(wall, d - top)
    prism = np.maximum(prism, (D - params.embed) - d)

    shell = d - (D - _SINK)             # body half-space, sunk out of view
    k = float(max(params.fillet_root, 0.0))
    if k > 0:
        h = np.clip(0.5 + 0.5 * (shell - prism) / k, 0.0, 1.0)
        f = prism * h + shell * (1 - h) - k * h * (1 - h)
    else:
        f = np.minimum(prism, shell)

    f = np.maximum(f, (D - _slab_depth(params)) - d)     # slab floor cap
    return np.where(inside, f.astype(np.float32), _AIR)


def _taubin(v, f, rounds=4, lam=0.5, mu=-0.53):
    """Taubin lambda|mu smoothing: strips marching-cubes voxel ripple
    without the shrinkage plain Laplacian smoothing causes."""
    import scipy.sparse as sp
    n = len(v)
    e = np.vstack([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    i = np.concatenate([e[:, 0], e[:, 1]])
    j = np.concatenate([e[:, 1], e[:, 0]])
    A = sp.csr_matrix((np.ones(len(i), np.float32), (i, j)), shape=(n, n))
    deg = np.asarray(A.sum(axis=1)).ravel()
    deg[deg == 0] = 1.0
    L = sp.diags((1.0 / deg).astype(np.float32)) @ A
    for _ in range(rounds):
        v = v + lam * (L @ v - v)
        v = v + mu * (L @ v - v)
    return v


def _weld(verts, faces, tol=1e-5):
    key = np.round(verts / tol).astype(np.int64)
    _, idx, inv = np.unique(key, axis=0, return_index=True,
                            return_inverse=True)
    f = inv[faces]
    ok = ((f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2]))
    return verts[idx], f[ok]


def mesh_field(grids, params, resolution, tile=192, reports=None):
    """March the field on the global grid; returns [(verts, tris)] in frame
    coordinates (one welded solid), or [] if the field is empty."""
    from skimage import measure

    g = grids
    res = float(resolution)
    nu, nv = g.D.shape
    u_lo = g.origin[0]
    v_lo = g.origin[1]
    u_hi = u_lo + (nu - 1) * g.cell
    v_hi = v_lo + (nv - 1) * g.cell
    # global integer sample grid (absolute multiples of res)
    iu0 = int(math.floor(u_lo / res)) - 1
    iu1 = int(math.ceil(u_hi / res)) + 1
    iv0 = int(math.floor(v_lo / res)) - 1
    iv1 = int(math.ceil(v_hi / res)) + 1

    pad = _slab_depth(params) + 2 * res
    top_pad = params.height + max(params.fillet_root, 0.0) \
        + max(params.fillet_top, 0.0) + 2 * res

    all_v, all_f, off = [], [], 0
    for ta in range(iu0, iu1, tile):
        tb = min(ta + tile, iu1)
        for tc in range(iv0, iv1, tile):
            td = min(tc + tile, iv1)
            us = np.arange(ta, tb + 1) * res
            vs = np.arange(tc, td + 1) * res
            # depth band from the tile's masked surface cells
            ci0 = np.clip(((us[0] - u_lo) / g.cell).astype(int), 0, nu - 1)
            ci1 = np.clip(int((us[-1] - u_lo) / g.cell) + 2, 1, nu)
            cj0 = np.clip(((vs[0] - v_lo) / g.cell).astype(int), 0, nv - 1)
            cj1 = np.clip(int((vs[-1] - v_lo) / g.cell) + 2, 1, nv)
            m = g.mask[ci0:ci1, cj0:cj1]
            if not m.any():
                continue
            Dt = g.D[ci0:ci1, cj0:cj1][m]
            d0 = int(math.floor((Dt.min() - pad) / res))
            d1 = int(math.ceil((Dt.max() + top_pad) / res))
            ds = np.arange(d0, d1 + 1) * res
            U, V, W = np.meshgrid(us, vs, ds, indexing="ij")
            vol = field(U.ravel(), V.ravel(), W.ravel(), g, params) \
                .reshape(U.shape)
            if vol.min() >= 0 or vol.max() <= 0:
                continue
            verts, faces, _, _ = measure.marching_cubes(
                vol, 0.0, spacing=(res, res, res))
            verts += (us[0], vs[0], ds[0])
            all_v.append(verts)
            all_f.append(faces + off)
            off += len(verts)

    if not all_v:
        return []
    v, f = _weld(np.vstack(all_v).astype(np.float64), np.vstack(all_f))
    v = _taubin(v, f)
    signed = np.einsum("ij,ij->i", v[f[:, 0]],
                       np.cross(v[f[:, 1]], v[f[:, 2]])).sum() / 6.0
    if signed < 0:
        f = f[:, ::-1]
    if reports is not None:
        import manifold3d as m3d
        man = m3d.Manifold(m3d.Mesh(v.astype(np.float32),
                                    f.astype(np.uint32)))
        if man.is_empty():
            reports.append("implicit mesh failed manifold validation")
    return [(v, f.astype(np.int64))]


# --- stage 1: pattern-space rasters from real geometry ----------------------

def _rasterize_depth(flat, depth, tris, kept, cell, origin, shape2d):
    """Front-most surface depth per cell from the kept projected triangles."""
    D = np.full(shape2d, -1e9, np.float32)
    u0, v0 = origin
    for a, b, c in tris[kept]:
        p = (np.array([flat[a], flat[b], flat[c]]) - (u0, v0)) / cell
        dz = np.array([depth[a], depth[b], depth[c]], np.float64)
        i0 = max(int(np.floor(p[:, 0].min())), 0)
        i1 = min(int(np.ceil(p[:, 0].max())) + 1, shape2d[0])
        j0 = max(int(np.floor(p[:, 1].min())), 0)
        j1 = min(int(np.ceil(p[:, 1].max())) + 1, shape2d[1])
        if i1 <= i0 or j1 <= j0:
            continue
        gi, gj = np.meshgrid(np.arange(i0, i1), np.arange(j0, j1),
                             indexing="ij")
        det = ((p[1, 0] - p[0, 0]) * (p[2, 1] - p[0, 1])
               - (p[2, 0] - p[0, 0]) * (p[1, 1] - p[0, 1]))
        if abs(det) < 1e-12:
            continue
        w1 = ((gi - p[0, 0]) * (p[2, 1] - p[0, 1])
              - (p[2, 0] - p[0, 0]) * (gj - p[0, 1])) / det
        w2 = ((p[1, 0] - p[0, 0]) * (gj - p[0, 1])
              - (gi - p[0, 0]) * (p[1, 1] - p[0, 1])) / det
        w0 = 1.0 - w1 - w2
        eps = -0.02
        inside = (w0 >= eps) & (w1 >= eps) & (w2 >= eps)
        if not inside.any():
            continue
        dval = (w0 * dz[0] + w1 * dz[1] + w2 * dz[2]).astype(np.float32)
        sub = D[i0:i1, j0:j1]
        np.copyto(sub, np.maximum(sub, np.where(inside, dval, -1e9)))
    return D


def _rasterize_rings(geom, cell, origin, shape2d):
    """Boolean raster of a (Multi)Polygon with holes."""
    from skimage.draw import polygon as sk_polygon
    mask = np.zeros(shape2d, bool)
    geoms = getattr(geom, "geoms", [geom])
    for g in geoms:
        ext = (np.asarray(g.exterior.coords) - origin) / cell
        rr, cc = sk_polygon(ext[:, 0], ext[:, 1], shape2d)
        mask[rr, cc] = True
        for ring in g.interiors:
            h = (np.asarray(ring.coords) - origin) / cell
            rr, cc = sk_polygon(h[:, 0], h[:, 1], shape2d)
            mask[rr, cc] = False
    return mask


def _clip_to_box(q0, q1, hi0, hi1):
    """Liang-Barsky clip of segment q0-q1 to [0,hi0]x[0,hi1]; None if out."""
    d = q1 - q0
    t0, t1 = 0.0, 1.0
    for axis, hi in ((0, hi0), (1, hi1)):
        for p, q in ((-d[axis], q0[axis]), (d[axis], hi - q0[axis])):
            if abs(p) < 1e-12:
                if q < 0:
                    return None
                continue
            t = q / p
            if p < 0:
                t0 = max(t0, t)
            else:
                t1 = min(t1, t)
            if t0 > t1:
                return None
    return q0 + t0 * d, q0 + t0 * d + (t1 - t0) * d


def _rasterize_segments(segs, cell, origin, shape2d):
    """Boolean raster of the rib centerline network.

    Segments are clipped to the raster box parametrically — clamping their
    endpoints instead would redirect out-of-window family lines into false
    chords across the pattern.
    """
    from skimage.draw import line as sk_line
    mask = np.zeros(shape2d, bool)
    hi0, hi1 = shape2d[0] - 1, shape2d[1] - 1
    for s in segs:
        pts = (np.asarray(s, float) - origin) / cell
        for q0, q1 in zip(pts[:-1], pts[1:]):
            hit = _clip_to_box(q0, q1, hi0, hi1)
            if hit is None:
                continue
            a, b = hit
            rr, cc = sk_line(int(round(a[0])), int(round(a[1])),
                             int(round(b[0])), int(round(b[1])))
            mask[rr, cc] = True
    return mask


def _ring_chains(geom, step):
    """Boundary rings resampled as polyline segments (border rib)."""
    out = []
    geoms = getattr(geom, "geoms", [geom])
    for g in geoms:
        for ring in [g.exterior, *g.interiors]:
            n = max(8, int(math.ceil(ring.length / step)))
            pts = [ring.interpolate(i * ring.length / n).coords[0]
                   for i in range(n + 1)]
            out.append(pts)
    return out


def build_rib_implicit(shape, face_ids, params, lin_defl=0.4, quality=1.0,
                       frame_cache=None):
    """Implicit-engine entry: same contract as build_rib_meshes.

    Projected mapping only: the field lives in the shared front-view frame.
    Returns ([(verts, tris)], reports) — normally ONE watertight solid of
    ribs blended into a thin sub-surface slab.
    """
    from scipy.ndimage import distance_transform_edt
    from .meshing import region_meshes
    from .patterns import generate_segments
    from .ribbing import (RibbingError, RibReport, _kept_domain,
                          _lattice_window, _merge_regions, _projection_frame)

    if params.mapping != "project":
        raise RibbingError(
            "the implicit engine requires mapping=project (unfold stays on "
            "the fast engine)")
    if not face_ids:
        raise RibbingError("no faces selected")
    if len(face_ids) > 25:
        lin_defl = max(lin_defl, 0.7)
    regions = region_meshes(shape, face_ids, lin_defl)
    if not regions:
        raise RibbingError("selected faces could not be triangulated")
    if len(regions) > 1:
        regions = [_merge_regions(regions)]
    region = regions[0]

    ctr, axes = _projection_frame(regions)
    if frame_cache is not None:
        held = frame_cache.get("frame")
        if held is not None and (
                np.cross(held[1][:, 0], held[1][:, 1])
                @ np.cross(axes[:, 0], axes[:, 1])
                >= math.cos(math.radians(25.0))):
            ctr, axes = held
        else:
            frame_cache["frame"] = (ctr, axes)
    n_axis = np.cross(axes[:, 0], axes[:, 1])

    rep = RibReport(face_id=region.face_ids[0], face_ids=region.face_ids)
    t0 = time.time()

    flat = (region.vertices - ctr) @ axes
    depth = (region.vertices - ctr) @ n_axis
    tris = np.asarray(region.triangles, np.int64)
    e1 = flat[tris[:, 1]] - flat[tris[:, 0]]
    e2 = flat[tris[:, 2]] - flat[tris[:, 0]]
    signed2 = 0.5 * (e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0])
    v3 = region.vertices
    area3 = 0.5 * np.linalg.norm(
        np.cross(v3[tris[:, 1]] - v3[tris[:, 0]],
                 v3[tris[:, 2]] - v3[tris[:, 0]]), axis=1)
    kept = np.nonzero(signed2 > 0.30 * np.clip(area3, 1e-12, None))[0]

    domain = _kept_domain(flat, tris, kept, params)
    if domain is None or domain.is_empty:
        raise RibbingError("selection is edge-on to the projection plane")
    inset = domain.buffer(-params.margin) if params.margin > 0 else domain
    if inset.is_empty:
        raise RibbingError("margin leaves no rib area")

    # voxels must resolve the thinnest feature: ~4.5 cells across a rib
    res = float(np.clip(min(0.30 / max(quality, 0.5),
                            params.thickness / 4.5), 0.10, 0.5))
    minx, miny, maxx, maxy = domain.bounds
    cell = max(res / 2.0, 0.15)
    while ((maxx - minx) / cell + 2) * ((maxy - miny) / cell + 2) > 4.5e7:
        cell *= 1.4                     # cap raster memory on huge fronts
    origin = (minx - 2 * cell, miny - 2 * cell)
    shape2d = (int((maxx - origin[0]) / cell) + 3,
               int((maxy - origin[1]) / cell) + 3)

    D = _rasterize_depth(flat, depth, tris, kept, cell, origin, shape2d)
    inset_mask = _rasterize_rings(inset, cell, origin, shape2d)
    covered = D > -1e8
    holes = inset_mask & ~covered
    if holes.any():
        # sealed separator strips have no surface of their own: borrow the
        # nearest flank's depth so ribs bridge them (same jog the fast
        # engine's clamp produces)
        _, (ri, ci) = distance_transform_edt(~covered, return_indices=True)
        D[holes] = D[ri[holes], ci[holes]]
    mask = inset_mask

    segs = list(generate_segments(params, _lattice_window(params,
                                                          domain.bounds)))
    if params.border:
        segs += _ring_chains(inset, max(cell * 2, params.spacing / 6.0))
    center = _rasterize_segments(segs, cell, origin, shape2d)
    if not center.any():
        raise RibbingError("pattern produced no ribs on this region")
    P = (distance_transform_edt(~center) * cell).astype(np.float32)
    B = (distance_transform_edt(inset_mask) * cell).astype(np.float32)

    grids = FrameGrids(cell=cell, origin=origin, D=D, P=P, B=B, mask=mask)
    warn = []
    clusters = mesh_field(grids, params, resolution=res, reports=warn)
    rep.warnings += warn
    rep.segments = len(segs)
    rep.lofted = len(segs) if clusters else 0
    if not clusters:
        raise RibbingError("implicit field produced no geometry")
    world = []
    for v, f in clusters:
        w = (ctr + v[:, :1] * axes[:, 0] + v[:, 1:2] * axes[:, 1]
             + v[:, 2:3] * n_axis)
        world.append((w, f))
    rep.warnings.append(
        f"implicit engine: {res:.2f}mm voxels, {time.time() - t0:.1f}s")
    return world, [rep]
