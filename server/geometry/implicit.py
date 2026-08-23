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
    B: np.ndarray               # distance to the final rib-domain boundary
    mask: np.ndarray            # inside the closed, margin-inset domain
    Gu: np.ndarray = None       # smoothed clipped dD/du (slope correction)
    Gv: np.ndarray = None       # smoothed clipped dD/dv
    Bo: np.ndarray = None       # distance to the OPEN boundary only
                                # (taper ramp: cull rims must not fade it)
    Sc: np.ndarray = None       # distance to the slope-cull band: ribs end
                                # on a smooth offset of the band (ramp in
                                # field()), not on the pixelated mask edge
    gate: np.ndarray = None     # (nlv, nu, nv) body-material lookup:
                                # True where ribs/slab may exist at that
                                # depth level (nTop-style body-field cut)
    gate_d0: float = 0.0        # frame depth of gate level 0
    gate_dlv: float = 1.0       # level spacing


def make_gradients(D, cell, max_slope=3.1):
    """Depth-map gradients for surface-normal rib extrusion.

    Clipped to the kept-filter slope (~72 deg) so infilled separator steps
    cannot spike them, then smoothed so marching cubes sees a stable frame.
    """
    from scipy.ndimage import gaussian_filter
    Gu, Gv = np.gradient(D.astype(np.float32), cell)
    m = np.hypot(Gu, Gv)
    scale = np.where(m > max_slope, max_slope / np.clip(m, 1e-9, None), 1.0)
    return (gaussian_filter(Gu * scale, 2.0).astype(np.float32),
            gaussian_filter(Gv * scale, 2.0).astype(np.float32))


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
    # just deep enough to seal rib bases into the body: the smooth-min
    # blend bulge lives ABOVE the surface, so k must NOT deepen the slab —
    # at fillet_root 2 it punched through 2.5mm walls and freckled the back
    return params.embed + 0.6


def field(u, v, d, grids, params):
    """Signed field (negative = material) at frame-space points (u, v, d)."""
    g = grids
    D = _bilinear(g.D, u, v, g.cell, g.origin)
    # ribs stand on the LOCAL surface normal, not the view axis: without
    # this, slopes grow leaning blades with knife tops and lose height.
    # First-order heightfield correction: s = true normal distance, and
    # laterals are sampled at the normal ray's foot point.
    if g.Gu is not None:
        Du = _bilinear(g.Gu, u, v, g.cell, g.origin)
        Dv = _bilinear(g.Gv, u, v, g.cell, g.origin)
        c = 1.0 / np.sqrt(1.0 + Du * Du + Dv * Dv)
        s = (d - D) * c
        fu = u + s * Du * c
        fv = v + s * Dv * c
    else:
        c = 1.0
        s = d - D
        fu, fv = u, v
    P = _bilinear(g.P, fu, fv, g.cell, g.origin)
    B = _bilinear(g.B, fu, fv, g.cell, g.origin)
    inside = _bilinear(g.mask.astype(np.float32), fu, fv, g.cell,
                       g.origin) > 0.5
    # beyond the raster there is no surface: cap the field as air there,
    # or marching cubes leaves open sheets at the sampling box walls
    inside &= ((u >= g.origin[0]) & (v >= g.origin[1])
               & (u <= g.origin[0] + (g.D.shape[0] - 1) * g.cell)
               & (v <= g.origin[1] + (g.D.shape[1] - 1) * g.cell))

    height = np.float32(params.height)
    if params.taper_len > 0 and not params.border:
        # the run-out ramp follows the OPEN boundary only: slope-cull rims
        # are interior transitions where retaining ribs meet lattice ribs
        # at full height — fading there reads as melted stubs. Sampled at
        # the foot: the shifted point would wobble the ramp on slopes.
        Bo = g.B if g.Bo is None else g.Bo
        open_d = _bilinear(Bo, u, v, g.cell, g.origin)
        height = height * np.clip(open_d / params.taper_len, 0.0, 1.0)

    half = np.float32(params.thickness / 2.0)
    if params.draft_deg > 0:
        half = half + math.tan(math.radians(params.draft_deg)) \
            * np.clip(height - s, 0.0, None)

    wall = P - half
    r_top = float(np.clip(params.fillet_top, 0.0, 0.35 * params.thickness))
    if r_top > 0:
        # rounded corner between wall and top plane
        a = wall + r_top
        b = (s - height) + r_top
        prism = (np.minimum(np.maximum(a, b), 0.0)
                 + np.hypot(np.clip(a, 0.0, None), np.clip(b, 0.0, None))
                 - r_top)
    else:
        prism = np.maximum(wall, s - height)
    prism = np.maximum(prism, -s - params.embed)
    # ribs end ON the true domain boundary (B is its distance field), not
    # on the pixelated mask edge — cut mask cells read as a serrated
    # fringe. The 1.2-cell setback suppresses sub-voxel slivers at cull
    # rims (they surfaced as flakes stuck to steep walls); border rib
    # centerlines sit at B=0, so borders become flush walls.
    prism = np.maximum(prism, 1.2 * np.float32(g.cell) - B)

    shell = s + _SINK                   # body half-space, sunk out of view
    k = float(max(params.fillet_root, 0.0))
    if k > 0:
        h = np.clip(0.5 + 0.5 * (shell - prism) / k, 0.0, 1.0)
        f = prism * h + shell * (1 - h) - k * h * (1 - h)
    else:
        f = np.minimum(prism, shell)

    f = np.maximum(f, -s - _slab_depth(params))          # slab floor cap
    # the slab must end on the domain boundary too: applied after the cap
    # (the prism clamp above cannot reach it), otherwise the sheet follows
    # the pixelated mask/coverage edge and frays at every recess rim
    f = np.maximum(f, 1.2 * np.float32(g.cell) - B)
    # cull-band termination: a linear ramp against the (smooth) distance
    # to the steep band ends every rib on a clean offset line ~1mm before
    # the band — sampled at the foot, so leaning tops stay put
    if g.Sc is not None:
        sc = _bilinear(g.Sc, u, v, g.cell, g.origin)
        f = np.maximum(f, np.float32(max(1.0, 0.75 * params.thickness)) - sc)
    # nTop-style body cut: material may only exist inside the body (or
    # within tolerance of its surface). Looked up at the sample's own 3D
    # position, so rib tops leaning past a lip terminate against the smooth
    # recess wall instead of being amputated at the pixelated mask edge.
    if g.gate is not None:
        gi = np.clip(((u - g.origin[0]) / g.cell).astype(np.int64), 0,
                     g.gate.shape[1] - 1)
        gj = np.clip(((v - g.origin[1]) / g.cell).astype(np.int64), 0,
                     g.gate.shape[2] - 1)
        gl = np.clip(((d - g.gate_d0) / g.gate_dlv).astype(np.int64), 0,
                     g.gate.shape[0] - 1)
        body_ok = g.gate[gl, gi, gj]
    else:
        body_ok = True
    # air must stay within a few voxels of zero: a 1e3 jump puts rim
    # crossings at t~0.001 — micron triangles that collapse into the
    # degenerate/duplicate faces manifold3d rejects
    air = np.float32(max(4.0 * g.cell, 0.6))
    return np.where(inside & body_ok, f.astype(np.float32), air)


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


def _fill_microholes(v, f, max_edges=250):
    """Fan-fill small boundary loops (marching-cubes micro-holes).

    Sub-voxel cracks survive welding when the two sides genuinely differ;
    at 0.02% of edges the standard repair is to patch them. The cap keeps
    a genuinely missing region visible instead of papering it over —
    crack loops are long thin slivers well under it.
    """
    import igl
    e = np.sort(np.vstack([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]),
                axis=1)
    _, counts = np.unique(e, axis=0, return_counts=True)
    if not (counts == 1).any():
        return v, f
    loops = igl.boundary_loop_all(np.asarray(f, np.int64))
    add_v, add_f = [], []
    nid = len(v)
    for loop in loops:
        loop = np.asarray(loop, np.int64)
        if len(loop) < 3 or len(loop) > max_edges:
            continue
        add_v.append(v[loop].mean(axis=0))
        for i in range(len(loop)):
            # fan winding must oppose the loop direction so patch normals
            # agree with the surrounding surface
            add_f.append((nid, loop[(i + 1) % len(loop)], loop[i]))
        nid += 1
    if not add_v:
        return v, f
    return (np.vstack([v, np.asarray(add_v)]),
            np.vstack([f, np.asarray(add_f, f.dtype)]))


def _repair_pinches(v, f):
    """Split marching-cubes pinch vertices (two closed fans sharing one
    vertex). manifold3d strictly rejects them, and a rejected lattice
    silently drops out of the export union."""
    import igl
    from collections import defaultdict
    ok = np.asarray(igl.is_vertex_manifold(f.astype(np.int64))).ravel()
    bad = set(map(int, np.nonzero(~ok)[0]))
    if not bad:
        return v, f
    f = f.copy()
    inc = defaultdict(list)
    for fi, tri in enumerate(f):
        for a in tri:
            if int(a) in bad:
                inc[int(a)].append(fi)
    add = []
    nid = len(v)
    for vid, fis in inc.items():
        parent = {fi: fi for fi in fis}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        by_other = defaultdict(list)
        for fi in fis:
            for o in f[fi]:
                if int(o) != vid:
                    by_other[int(o)].append(fi)
        for fl in by_other.values():
            for a, b in zip(fl[:-1], fl[1:]):
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[ra] = rb
        fans = defaultdict(list)
        for fi in fis:
            fans[find(fi)].append(fi)
        for comp in list(fans.values())[1:]:
            for fi in comp:
                f[fi] = np.where(f[fi] == vid, nid, f[fi])
            add.append(v[vid])
            nid += 1
    if add:
        v = np.vstack([v, np.asarray(add)])
    return v, f


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
            # one-cell overlap: neighbors evaluate the shared cells from
            # identical samples and produce identical triangles; each
            # triangle is kept by the tile owning its centroid, so tile
            # joints cannot crack (naked abutment relied on lewiner's
            # boundary-cell triangulation matching — empirically it does
            # not, and the mesh leaked thousands of micro-cracks)
            us = np.arange(ta - 1, tb + 2) * res
            vs = np.arange(tc - 1, td + 2) * res
            # depth band from the tile's masked surface cells
            ci0 = np.clip(((us[0] - u_lo) / g.cell).astype(int), 0, nu - 1)
            ci1 = np.clip(int((us[-1] - u_lo) / g.cell) + 2, 1, nu)
            cj0 = np.clip(((vs[0] - v_lo) / g.cell).astype(int), 0, nv - 1)
            cj1 = np.clip(int((vs[-1] - v_lo) / g.cell) + 2, 1, nv)
            m = g.mask[ci0:ci1, cj0:cj1]
            if not m.any():
                continue
            Dt = g.D[ci0:ci1, cj0:cj1][m]
            # normal-extruded ribs on slopes reach height/cos(theta) along
            # the view axis: stretch this tile's band by its local slope.
            # Full-resolution max — a subsampled estimate once let a rib
            # poke through the band top, leaving an open sheet.
            scale = 1.0
            if g.Gu is not None:
                gm = float(np.hypot(g.Gu[ci0:ci1, cj0:cj1],
                                    g.Gv[ci0:ci1, cj0:cj1]).max())
                scale = math.sqrt(1.0 + gm * gm)
            d0 = int(math.floor((Dt.min() - pad * scale) / res))
            d1 = int(math.ceil((Dt.max() + top_pad * scale) / res))
            ds = np.arange(d0, d1 + 1) * res
            U, V, W = np.meshgrid(us, vs, ds, indexing="ij")
            vol = field(U.ravel(), V.ravel(), W.ravel(), g, params) \
                .reshape(U.shape)
            if vol.min() >= 0 or vol.max() <= 0:
                continue
            verts, faces, _, _ = measure.marching_cubes(
                vol, 0.0, spacing=(res, res, res))
            verts += (us[0], vs[0], ds[0])
            cen = verts[faces].mean(axis=1)
            lo_u = -np.inf if ta == iu0 else ta * res
            hi_u = np.inf if tb >= iu1 else tb * res
            lo_v = -np.inf if tc == iv0 else tc * res
            hi_v = np.inf if td >= iv1 else td * res
            own = ((cen[:, 0] >= lo_u) & (cen[:, 0] < hi_u)
                   & (cen[:, 1] >= lo_v) & (cen[:, 1] < hi_v))
            faces = faces[own]
            if not len(faces):
                continue
            all_v.append(verts)
            all_f.append(faces + off)
            off += len(verts)

    if not all_v:
        return []
    v, f = _weld(np.vstack(all_v).astype(np.float64), np.vstack(all_f))
    v = _taubin(v, f)
    # per-component: drop marching-cubes crumbs (isolated slivers at steep
    # rims), orient each body outward by its own signed volume
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components
    adj = sp.csr_matrix((np.ones(len(f) * 3, np.int8),
                         (np.concatenate([f[:, 0], f[:, 1], f[:, 2]]),
                          np.concatenate([f[:, 1], f[:, 2], f[:, 0]]))),
                        shape=(len(v), len(v)))
    _, lab = connected_components(adj, directed=False)
    keep = []
    for cid in np.unique(lab[f[:, 0]]):
        fc = f[lab[f[:, 0]] == cid]
        vol = np.einsum("ij,ij->i", v[fc[:, 0]],
                        np.cross(v[fc[:, 1]], v[fc[:, 2]])).sum() / 6.0
        if abs(vol) < 8.0 or len(fc) < 24:
            continue
        keep.append(fc[:, ::-1] if vol < 0 else fc)
    if not keep:
        return []
    f = np.vstack(keep)
    # pinches first: boundary loops through pinched vertices cannot be
    # walked, so holes must be filled on the split mesh
    v, f = _repair_pinches(v, f)
    v, f = _fill_microholes(v, f)
    if reports is not None:
        import manifold3d as m3d
        mesh = m3d.Mesh(v.astype(np.float32), f.astype(np.uint32))
        # marching cubes leaves micro-cracks at tile planes and mask rims;
        # manifold3d's native merge() sews this exact defect class
        try:
            mesh.merge()
        except Exception:
            pass
        if m3d.Manifold(mesh).is_empty():
            reports.append("implicit mesh failed manifold validation")
    return [(v, f.astype(np.int64))]


# --- stage 1: pattern-space rasters from real geometry ----------------------

def _rasterize_depth(flat, depth, tris, kept, cell, origin, shape2d,
                     backmost=False):
    """Front-most (or back-most) surface depth per cell from the kept
    projected triangles."""
    D = np.full(shape2d, -1e9 if not backmost else 1e9, np.float32)
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
        if backmost:
            np.copyto(sub, np.minimum(sub, np.where(inside, dval, 1e9)))
        else:
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


def _exact_pattern_distance(segs, cell, origin, shape2d, reach):
    """Exact 2D distance to the centerline network, within `reach` of it.

    Rasterized-centerline EDT scallops walls by ~cell/2 (stair-step chains),
    which reads as rippled rib walls and crenellated border ridges. Exact
    point-to-segment distance in a band around each piece is sub-voxel true;
    beyond every band the field is just air, so 1e3 is fine there.
    """
    P = np.full(shape2d, 1e3, np.float32)
    pad = reach / cell
    for s in segs:
        pts = (np.asarray(s, float) - origin) / cell
        for q0, q1 in zip(pts[:-1], pts[1:]):
            i0 = max(int(math.floor(min(q0[0], q1[0]) - pad)), 0)
            i1 = min(int(math.ceil(max(q0[0], q1[0]) + pad)) + 1, shape2d[0])
            j0 = max(int(math.floor(min(q0[1], q1[1]) - pad)), 0)
            j1 = min(int(math.ceil(max(q0[1], q1[1]) + pad)) + 1, shape2d[1])
            if i1 <= i0 or j1 <= j0:
                continue
            gi = np.arange(i0, i1, dtype=np.float32)[:, None]
            gj = np.arange(j0, j1, dtype=np.float32)[None, :]
            e = q1 - q0
            L2 = float(e @ e)
            if L2 < 1e-12:
                dx = gi - q0[0]
                dy = gj - q0[1]
            else:
                t = np.clip(((gi - q0[0]) * e[0] + (gj - q0[1]) * e[1]) / L2,
                            0.0, 1.0)
                dx = gi - (q0[0] + t * e[0])
                dy = gj - (q0[1] + t * e[1])
            sub = P[i0:i1, j0:j1]
            np.copyto(sub, np.minimum(sub,
                                      np.sqrt(dx * dx + dy * dy) * cell))
    return P


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
    Dback = _rasterize_depth(flat, depth, tris, kept, cell, origin, shape2d,
                             backmost=True)
    inset_mask = _rasterize_rings(inset, cell, origin, shape2d)
    covered = D > -1e8
    if not covered.any():
        raise RibbingError("selection is edge-on to the projection plane")
    if (~covered).any():
        # sealed separator strips borrow the nearest flank's depth so ribs
        # bridge them; filling EVERY uncovered cell (not just domain holes)
        # keeps the depth gradients finite — a -1e9 cliff at the domain
        # edge turns into inf*0 = NaN and the gaussian smears it inward
        _, (ri, ci) = distance_transform_edt(~covered, return_indices=True)
        D[~covered] = D[ri[~covered], ci[~covered]]
    # a heightfield with first-order normal correction cannot represent
    # near-cull walls: past ~55 deg the ribs shred. Cull steep BANDS from
    # the rib domain — from the SMOOTHED gradients, so the cull contour is
    # smooth instead of stair-stepped; morphological opening spares the
    # one-cell gradient cliffs of infilled separator strips (those must
    # keep bridging).
    from scipy.ndimage import binary_opening
    Gu, Gv = make_gradients(D, cell)
    steep = binary_opening(
        np.hypot(Gu, Gv) > math.tan(math.radians(55.0)), iterations=2)
    # columns whose front surface is backed by material within a few mm
    # along the view ray: the heightfield can be trusted there. Over
    # recess floors and past the silhouette the front-most depth is a lip
    # far in front of the true surface — handled by the body gate below.
    backed = ((D - Dback) <= max(2.0 * params.thickness, 3.0)) \
        & (Dback <= 1e8)
    mask = inset_mask & ~steep

    segs = list(generate_segments(params, _lattice_window(params,
                                                          domain.bounds)))
    if params.border:
        segs += _ring_chains(inset, max(cell * 2, params.spacing / 6.0))
    reach = (params.thickness / 2.0
             + math.tan(math.radians(max(params.draft_deg, 0.0)))
             * (params.height + params.embed)
             + max(params.fillet_root, 0.0)
             + 0.35 * params.thickness + 2.0)
    P = _exact_pattern_distance(segs, cell, origin, shape2d, reach)
    if P.min() > reach:
        raise RibbingError("pattern produced no ribs on this region")
    # rib-termination distance measured from the FINAL domain (inset
    # boundary, slope-cull rims and their dilated rim strips): the -B clamp
    # ends every rib on it
    B = (distance_transform_edt(mask) * cell).astype(np.float32)
    # run-out ramp distance: to the OPEN boundary only (the inset polygon
    # edge) — rims are interior and must not fade the height
    Bo = (distance_transform_edt(inset_mask) * cell).astype(np.float32)
    # cull-band distance: ribs end on a smooth offset of the band. The
    # band edge itself is pixelated and meanders with gradient noise, but
    # its distance field is smooth — the ramp in field() terminates ribs
    # on a clean perpendicular line ~1mm before the band. Gaussian-
    # smoothing the field straightens the offset lines against the
    # band's gradient-noise meander. Rim walls (chain-following
    # extrusions) were tried here and read as chewed strips; the body
    # gate + this ramp is the clean nTop-style answer.
    from scipy.ndimage import gaussian_filter as _gf
    Sc = _gf(distance_transform_edt(~steep), 6.0).astype(np.float32) * cell

    # --- nTop-style body gate -------------------------------------------
    # additive material is cut by the BODY's own field, not the pixelated
    # 2D domain: points in the void behind a lip (over recess floors, past
    # the silhouette) are removed against the smooth recess wall, so ribs
    # die into the body exactly like an implicit-modeler feature. Looked
    # up as a body-interior test (fast-winding SDF <= tol) on depth
    # levels, only over unbacked columns near the void boundary — deep
    # void is cut outright and backed columns are always allowed.
    tol = float(np.clip(0.25 * params.thickness, 0.2, 0.5))
    g_d0 = -(params.embed + _slab_depth(params))
    g_d1 = params.height
    g_nlv = max(int(math.ceil((g_d1 - g_d0) / max(res, 0.4))) + 1, 2)
    g_dlv = (g_d1 - g_d0) / (g_nlv - 1)
    gate = np.zeros((g_nlv,) + shape2d, bool)
    gate[:, backed] = True
    from scipy.ndimage import distance_transform_edt as _edt
    void_d = _edt(backed) * cell          # depth inside unbacked columns
    band = 4.0
    cand = (~backed) & (void_d <= band) & inset_mask
    t_gate = time.time()
    if cand.any():
        import igl
        from .meshing import mesh_shape
        bm = mesh_shape(shape, 0.3, 0.3)
        bv, bf, boff = [], [], 0
        for m in bm:
            bv.append(np.asarray(m.vertices, np.float64))
            bf.append(np.asarray(m.triangles, np.int64) + boff)
            boff += len(m.vertices)
        BV = np.vstack(bv)
        BF = np.vstack(bf)
        ci, cj = np.nonzero(cand)
        cu = ci * cell + origin[0]
        cv = cj * cell + origin[1]
        n_c = len(ci)
        chunk = 1_500_000
        for l in range(g_nlv):
            dd = g_d0 + l * g_dlv
            for s0 in range(0, n_c, chunk):
                sl = slice(s0, min(s0 + chunk, n_c))
                pts = ctr[None, :] + cu[sl, None] * axes[:, 0] \
                    + cv[sl, None] * axes[:, 1] + dd * n_axis[None, :]
                out = igl.signed_distance(np.ascontiguousarray(pts), BV, BF)
                gate[l, ci[sl], cj[sl]] = out[0] <= tol
    rep.warnings.append(
        f"body gate: {int(cand.sum())} cols x {g_nlv} lv, "
        f"{time.time() - t_gate:.0f}s")

    grids = FrameGrids(cell=cell, origin=origin, D=D, P=P, B=B, mask=mask,
                       Gu=Gu, Gv=Gv, Bo=Bo, Sc=Sc, gate=gate, gate_d0=g_d0,
                       gate_dlv=g_dlv)
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
