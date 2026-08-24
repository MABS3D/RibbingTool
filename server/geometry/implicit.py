"""Implicit (SDF) rib engine: surface-field substrate, one scalar field.

The substrate is the exact distance field of the SELECTED surface mesh:
every voxel queries the closest point on the selection (igl BVH), so the
height coordinate s is the TRUE signed distance along the local surface
normal — on flat faces, steep walls and overhangs alike.  The 2D pattern
rasters are sampled at that closest point (a closest-point surface
field, the nTop "thickening" construction), so rib width is constant in
surface coordinates on any slope and no slope cull exists: every
selected face gets ribbed.  Ribs, root blends, crowns, taper and draft
compose as field math; the boolean with the body is a smooth-min, so
every rib-rib and rib-body junction blends uniformly and cannot fail.
The field is evaluated tile by tile on ONE global voxel grid (absolute
multiples of the resolution on all three axes), so marching-cubes
output from adjacent tiles welds exactly at shared planes.  Evaluation
is narrowed to a band around the surface by per-column padded depth
bounds (an OpenVDB-style narrow band, built from conservative
per-triangle projected bounding boxes so wall columns inherit their
wall's full depth extent).

Sign convention: s is signed by dot(x - cp, n(cp)) with angle-weighted
vertex pseudo-normals (Baerentzen) — the exact first-order signed
distance on smooth patches, valid on open meshes without any
watertightness assumption.
"""
import math
import time
from dataclasses import dataclass

import numpy as np

_SINK = 0.05                    # slab top sits this far under the surface


@dataclass
class SurfaceField:
    """Pattern-space rasters + substrate mesh, all in the frame.

    The rasters are pure 2D PATTERN DATA (centerline distance, domain
    boundary distance, domain mask); geometry comes from the substrate
    mesh via exact closest-point queries.  Rasters are indexed [u, v].
    """
    cell: float                 # raster pitch (mm)
    origin: tuple               # (u0, v0) of raster[0, 0]
    P: np.ndarray               # 2D distance to rib centerlines
    B: np.ndarray               # distance to the final rib-domain boundary
    mask: np.ndarray            # inside the closed, margin-inset domain
    Bo: np.ndarray              # distance to the OPEN boundary only
                                # (taper ramp: rims must not fade it)
    V: np.ndarray               # substrate vertices (frame coords)
    F: np.ndarray               # substrate triangles
    N: np.ndarray               # angle-weighted vertex pseudo-normals
    dlo: np.ndarray             # (nu, nv) per-column band lower bound,
                                # padded for slab + embed + blend dip
    dhi: np.ndarray             # (nu, nv) per-column band upper bound,
                                # padded for height + fillets (rib
                                # corridor columns; slab-only elsewhere)
    BV: np.ndarray = None       # closed BODY mesh (whole part, WORLD
    BF: np.ndarray = None       # coords): negative-side material must
    body_tol: float = 0.0       # live inside it (the nTop boolean-with-
                                # the-body; kills the fake walls the
                                # closest-point map grows in the void
                                # between folded sheets)
    ctr: np.ndarray = None      # frame->world transform for body queries
    M: np.ndarray = None        # (3,3) frame axis columns


def _vertex_normals(V, F):
    """Angle-weighted vertex pseudo-normals (Baerentzen & Anaes).

    dot(x - cp, n(cp)) is the exact first-order side test on smooth
    patches, stays correct across triangle edges, and degrades gracefully
    at open boundaries — no watertightness required.
    """
    p0, p1, p2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    fn = np.cross(p1 - p0, p2 - p0)
    ln = np.linalg.norm(fn, axis=1)
    ok = ln > 1e-14
    fnu = np.zeros_like(fn)
    fnu[ok] = fn[ok] / ln[ok, None]
    N = np.zeros_like(V)

    def _ang(a, b, c):          # angle at corner a
        u = b - a
        w = c - a
        cosv = (u * w).sum(axis=1) / np.clip(
            np.linalg.norm(u, axis=1) * np.linalg.norm(w, axis=1),
            1e-14, None)
        return np.arccos(np.clip(cosv, -1.0, 1.0))

    for src, ang in ((0, _ang(p0, p1, p2)), (1, _ang(p1, p2, p0)),
                     (2, _ang(p2, p0, p1))):
        np.add.at(N, F[:, src], fnu * ang[:, None])
    ln = np.linalg.norm(N, axis=1)
    ok = ln > 1e-14
    N[ok] = N[ok] / ln[ok, None]
    return N


def _surface_eval(X, V, F, N, chunk=1_500_000):
    """Exact signed distance to the substrate + closest surface points.

    Unsigned distance from the BVH; sign from the interpolated
    pseudo-normal.  Chunked to bound peak memory on multi-megavoxel
    bands.
    """
    import igl
    s = np.empty(len(X), np.float32)
    C = np.empty((len(X), 3))
    for i0 in range(0, len(X), chunk):
        sl = slice(i0, min(i0 + chunk, len(X)))
        Q = np.ascontiguousarray(X[sl], np.float64)
        sqrD, I, Cp = igl.point_mesh_squared_distance(Q, V, F)
        d = np.sqrt(sqrD)
        # barycentric coords of the closest point in its triangle ->
        # interpolated pseudo-normal -> side test
        a, b, c = V[F[I, 0]], V[F[I, 1]], V[F[I, 2]]
        v0 = b - a
        v1 = c - a
        v2 = Cp - a
        d00 = (v0 * v0).sum(1)
        d01 = (v0 * v1).sum(1)
        d11 = (v1 * v1).sum(1)
        d20 = (v2 * v0).sum(1)
        d21 = (v2 * v1).sum(1)
        den = np.clip(d00 * d11 - d01 * d01, 1e-20, None)
        w1 = (d11 * d20 - d01 * d21) / den
        w2 = (d00 * d21 - d01 * d20) / den
        w0 = 1.0 - w1 - w2
        n = (N[F[I, 0]] * w0[:, None] + N[F[I, 1]] * w1[:, None]
             + N[F[I, 2]] * w2[:, None])
        nn = np.linalg.norm(n, axis=1)
        ok = nn > 1e-14
        n[ok] = n[ok] / nn[ok, None]
        side = np.einsum("ij,ij->i", Q - Cp, n)
        s[sl] = np.where(side >= 0.0, d, -d).astype(np.float32)
        C[sl] = Cp
    return s, C


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


def _rib_field(s, P, B, inside, Bo, params, cell):
    """Rib prism + root blend + slab in TRUE surface coordinates.

    s: exact signed distance to the substrate (negative inside the
    body); P/B/Bo/mask sampled at each point's closest surface point.
    """
    height = np.float32(params.height)
    if params.taper_len > 0 and not params.border:
        # the run-out ramp follows the OPEN boundary only: domain rims
        # are interior transitions and must not fade the height
        height = height * np.clip(Bo / params.taper_len, 0.0, 1.0)

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
    # ribs end ON the true domain boundary (B is its distance field,
    # sampled at the closest surface point), not on the pixelated mask
    # edge — cut mask cells read as a serrated fringe.  The 1.2-cell
    # setback suppresses sub-voxel slivers at rims; border rib
    # centerlines sit at B=0, so borders become flush walls.
    prism = np.maximum(prism, 1.2 * np.float32(cell) - B)

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
    # the pixelated mask edge and frays at every recess rim
    f = np.maximum(f, 1.2 * np.float32(cell) - B)
    air = np.float32(max(4.0 * cell, 0.6))
    return np.where(inside, f.astype(np.float32), air)


def mesh_field(surface, params, resolution, tile=192, reports=None):
    """March the field on the global grid; returns [(verts, tris)] in frame
    coordinates (one welded solid), or [] if the field is empty."""
    from skimage import measure

    g = surface
    res = float(resolution)
    nu, nv = g.dlo.shape
    u_lo = g.origin[0]
    v_lo = g.origin[1]
    u_hi = u_lo + (nu - 1) * g.cell
    v_hi = v_lo + (nv - 1) * g.cell
    # global integer sample grid (absolute multiples of res)
    iu0 = int(math.floor(u_lo / res)) - 1
    iu1 = int(math.ceil(u_hi / res)) + 1
    iv0 = int(math.floor(v_lo / res)) - 1
    iv1 = int(math.ceil(v_hi / res)) + 1

    air = np.float32(max(4.0 * g.cell, 0.6))
    all_v, all_f, off = [], [], 0
    DSLAB = 128                       # max d-slices per marching box
    for ta in range(iu0, iu1, tile):
        tb = min(ta + tile, iu1)
        for tc in range(iv0, iv1, tile):
            td = min(tc + tile, iv1)
            # one-cell overlap: neighbors evaluate the shared cells from
            # identical samples and produce identical triangles; each
            # triangle is kept by the tile owning its centroid, so tile
            # joints cannot crack
            us = np.arange(ta - 1, tb + 2) * res
            vs = np.arange(tc - 1, td + 2) * res
            # nearest raster column per sample
            gi = np.clip(np.round((us - u_lo) / g.cell).astype(np.int64),
                         0, nu - 1)
            gj = np.clip(np.round((vs - v_lo) / g.cell).astype(np.int64),
                         0, nv - 1)
            if not g.mask[gi][:, gj].any():
                continue
            blo = g.dlo[gi][:, gj]
            bhi = g.dhi[gi][:, gj]
            d0 = int(math.floor(float(blo.min()) / res))
            d1 = int(math.ceil(float(bhi.max()) / res))
            if d1 <= d0:
                continue
            # tall folded regions span enormous depth ranges; march in
            # d-slabs so no box explodes (2-slice overlap, ownership by
            # centroid — same contract as the u/v tile joints)
            for e0 in range(d0, d1, DSLAB):
                e1 = min(e0 + DSLAB, d1)
                ds = np.arange(e0 - 1, e1 + 2) * res
                # OpenVDB-style narrow band: evaluate only within the
                # per-column padded depth bounds (no meshgrids — the
                # band mask is built by broadcasting and points are
                # gathered by index)
                band = ((ds[None, None, :] >= blo[:, :, None])
                        & (ds[None, None, :] <= bhi[:, :, None]))
                if not band.any():
                    continue
                iu, iv, iw = np.nonzero(band)
                X = np.column_stack([us[iu], vs[iv], ds[iw]])
                s, C = _surface_eval(X, g.V, g.F, g.N)
                del X
                # pattern/boundary/mask sampled at the closest surface
                # point: the closest-point surface field.  The pattern is
                # carried along the surface, so rib width is constant in
                # surface coordinates on any slope, and ribs on walls
                # continue the lattice of the flanks they connect to.
                cu = C[:, 0]
                cv = C[:, 1]
                P = _bilinear(g.P, cu, cv, g.cell, g.origin)
                B = _bilinear(g.B, cu, cv, g.cell, g.origin)
                Bo = _bilinear(g.Bo, cu, cv, g.cell, g.origin)
                ins = _bilinear(g.mask.astype(np.float32), cu, cv, g.cell,
                                g.origin) > 0.5
                # beyond the raster there is no surface: cap the field as
                # air there (by the SAMPLE's own position), or marching
                # cubes leaves open sheets at the sampling box walls
                ins &= ((us[iu] >= u_lo) & (us[iu] <= u_hi)
                        & (vs[iv] >= v_lo) & (vs[iv] <= v_hi))
                vals = _rib_field(s, P, B, ins, Bo, params, g.cell)
                del P, B, Bo, ins, cu, cv
                # nTop-style boolean with the body: material on the
                # NEGATIVE side of the substrate (the slab) must be
                # inside the BODY.  On a wrapped selection the
                # closest-point map switches between folded sheets, and
                # the slab of one sheet would be truncated by a raw wall
                # deep in the void between them — the body field cuts
                # exactly there.  Rib-side voxels hug the substrate and
                # are exempt (a body cut at rib-top height would eat
                # every rib).  Queried only at actual material
                # candidates, and in WORLD coordinates.
                if g.BV is not None:
                    import igl
                    # material candidates on the NEGATIVE side only: the
                    # slab.  Rib voxels are also vals<0 but stand on the
                    # positive side — a body cut at rib-top height would
                    # eat every rib.
                    cand = (vals < 0.0) & (s < 0.5)
                    del s
                    if cand.any():
                        Xw = np.ascontiguousarray(
                            np.column_stack([us[iu[cand]], vs[iv[cand]],
                                             ds[iw[cand]]]) @ g.M + g.ctr)
                        out = igl.signed_distance(Xw, g.BV, g.BF)
                        # raising the field value above zero cuts the
                        # voxel out of the solid
                        vals[cand] = np.maximum(
                            vals[cand],
                            out[0].astype(np.float32) - g.body_tol)
                        del Xw, out
                vol = np.full(band.shape, air, np.float32)
                vol[band] = vals
                del band, vals, iu, iv, iw, C
                if vol.min() >= 0 or vol.max() <= 0:
                    continue
                verts, faces, _, _ = measure.marching_cubes(
                    vol, 0.0, spacing=(res, res, res))
                del vol
                verts += (us[0], vs[0], ds[0])
                cen = verts[faces].mean(axis=1)
                lo_u = -np.inf if ta == iu0 else ta * res
                hi_u = np.inf if tb >= iu1 else tb * res
                lo_v = -np.inf if tc == iv0 else tc * res
                hi_v = np.inf if td >= iv1 else td * res
                lo_d = -np.inf if e0 == d0 else e0 * res
                hi_d = np.inf if e1 >= d1 else e1 * res
                own = ((cen[:, 0] >= lo_u) & (cen[:, 0] < hi_u)
                       & (cen[:, 1] >= lo_v) & (cen[:, 1] < hi_v)
                       & (cen[:, 2] >= lo_d) & (cen[:, 2] < hi_d))
                faces = faces[own]
                if not len(faces):
                    continue
                all_v.append(verts)
                all_f.append(faces + off)
                off += len(verts)

    if not all_v:
        return []
    v, f = _weld(np.vstack(all_v).astype(np.float64), np.vstack(all_f))
    # per-component FIRST (before smoothing): drop marching-cubes crumbs
    # (isolated slivers at rims and closest-point switch walls) and orient
    # each body outward by its own signed volume.  Smoothing before this
    # lets crumb vertices with long thin triangles get launched tens of
    # millimeters across the void by the uniform graph Laplacian.
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
    # drop vertices orphaned by the crumb filter: they must not ride
    # along in the output (and must not be smoothed)
    used, f = np.unique(f, return_inverse=True)
    v = np.ascontiguousarray(v[used])
    f = np.ascontiguousarray(f.reshape(-1, 3))
    v = _taubin(v, f, rounds=6)
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


# --- stage 1: pattern-space rasters + band bounds from real geometry ----

def _rasterize_depth_range(flat, depth, tris, cell, origin, shape2d):
    """Per-column [min, max] surface depth: accurate inside-fill first,
    conservative per-triangle bounding boxes only for what it misses.

    Inside-fill rasterization drops near-edge-on triangles (a steep wall
    projects to a sliver or a line), which would truncate the depth band
    exactly on the walls ribs must climb — those get their bounding box
    filled.  Plain bbox fill everywhere would inflate the band on every
    oblique triangle (a diagonal flank paints its full depth span over
    columns it merely brushes), costing megavoxels of pointless queries.
    """
    Dlo = np.full(shape2d, 1e9, np.float32)
    Dhi = np.full(shape2d, -1e9, np.float32)
    covered = np.zeros(shape2d, bool)
    u0, v0 = origin

    def _bbox(a, b, c):
        i0 = max(int(math.floor((min(flat[a, 0], flat[b, 0], flat[c, 0])
                                 - u0) / cell)), 0)
        i1 = min(int(math.ceil((max(flat[a, 0], flat[b, 0], flat[c, 0])
                                - u0) / cell)) + 1, shape2d[0])
        j0 = max(int(math.floor((min(flat[a, 1], flat[b, 1], flat[c, 1])
                                 - v0) / cell)), 0)
        j1 = min(int(math.ceil((max(flat[a, 1], flat[b, 1], flat[c, 1])
                                - v0) / cell)) + 1, shape2d[1])
        return i0, i1, j0, j1

    for a, b, c in tris:
        i0, i1, j0, j1 = _bbox(a, b, c)
        if i1 <= i0 or j1 <= j0:
            continue
        p = (np.array([flat[a], flat[b], flat[c]]) - (u0, v0)) / cell
        det = ((p[1, 0] - p[0, 0]) * (p[2, 1] - p[0, 1])
               - (p[2, 0] - p[0, 0]) * (p[1, 1] - p[0, 1]))
        if abs(det) < 1e-12:
            continue
        gi, gj = np.meshgrid(np.arange(i0, i1), np.arange(j0, j1),
                             indexing="ij")
        w1 = ((gi - p[0, 0]) * (p[2, 1] - p[0, 1])
              - (p[2, 0] - p[0, 0]) * (gj - p[0, 1])) / det
        w2 = ((p[1, 0] - p[0, 0]) * (gj - p[0, 1])
              - (gi - p[0, 0]) * (p[1, 1] - p[0, 1])) / det
        w0 = 1.0 - w1 - w2
        inside = (w0 >= -0.02) & (w1 >= -0.02) & (w2 >= -0.02)
        if not inside.any():
            continue
        dmin = min(depth[a], depth[b], depth[c])
        dmax = max(depth[a], depth[b], depth[c])
        # interpolated depth: exact per-cell, no facet terracing in the
        # band bounds
        dval = (w0 * depth[a] + w1 * depth[b] + w2 * depth[c])
        sub_lo = Dlo[i0:i1, j0:j1]
        np.copyto(sub_lo, np.minimum(sub_lo, np.where(inside, dval, 1e9)))
        sub_hi = Dhi[i0:i1, j0:j1]
        np.copyto(sub_hi, np.maximum(sub_hi, np.where(inside, dval, -1e9)))
        sub_c = covered[i0:i1, j0:j1]
        np.copyto(sub_c, sub_c | inside)
    # edge-on slivers: bounding-box fill of the cells inside-fill missed
    for a, b, c in tris:
        i0, i1, j0, j1 = _bbox(a, b, c)
        if i1 <= i0 or j1 <= j0:
            continue
        win = ~covered[i0:i1, j0:j1]
        if not win.any():
            continue
        dmin = min(depth[a], depth[b], depth[c])
        dmax = max(depth[a], depth[b], depth[c])
        sub_lo = Dlo[i0:i1, j0:j1]
        np.copyto(sub_lo, np.minimum(sub_lo, np.where(win, dmin, 1e9)))
        sub_hi = Dhi[i0:i1, j0:j1]
        np.copyto(sub_hi, np.maximum(sub_hi, np.where(win, dmax, -1e9)))
        sub_c = covered[i0:i1, j0:j1]
        np.copyto(sub_c, sub_c | win)
    return Dlo, Dhi


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

    Surface-field mapping: the substrate is the exact distance field of
    the selected faces; the pattern rides the closest-point surface
    field.  Returns ([(verts, tris)], reports) — normally ONE watertight
    solid of ribs blended into a thin sub-surface slab.
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

    # The rib domain is the projection of the WHOLE selection: steep
    # walls and overhangs are rib territory now (the closest-point field
    # ribs them exactly like flat faces), so there is no front-facing
    # filter — that filter was the heightfield era's cull in disguise.
    domain = _kept_domain(flat, tris, np.arange(len(tris)), params)
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

    Dlo, Dhi = _rasterize_depth_range(flat, depth, tris, cell, origin,
                                      shape2d)
    unc = Dlo > 1e8
    if unc.any() and (~unc).any():
        # sealed separator strips borrow the nearest column's band so the
        # surface field stays evaluable across them
        _, (ri, ci) = distance_transform_edt(unc, return_indices=True)
        Dlo[unc] = Dlo[ri[unc], ci[unc]]
        Dhi[unc] = Dhi[ri[unc], ci[unc]]

    inset_mask = _rasterize_rings(inset, cell, origin, shape2d)
    if not inset_mask.any():
        raise RibbingError("margin leaves no rib area")

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
    # rib-termination distance measured from the FINAL domain (the inset
    # boundary): the -B clamp ends every rib on it
    B = (distance_transform_edt(inset_mask) * cell).astype(np.float32)
    Bo = B.copy()

    # --- substrate: the selection mesh itself, in frame coordinates -----
    V = np.ascontiguousarray(np.column_stack([flat, depth]), np.float64)
    F = np.ascontiguousarray(tris, np.int64)
    N = _vertex_normals(V, F)

    # closed BODY mesh for the negative-side cut (nTop boolean-with-the-
    # body): the slab of a wrapped selection gets fake walls in the void
    # between folded sheets where the closest-point map switches — the
    # body field cuts exactly there.  Threshold sits above the worst-case
    # substrate/body tessellation gap so real slab never clips.
    from .meshing import mesh_shape
    bv, bf, boff = [], [], 0
    for m in mesh_shape(shape, 0.3, 0.3):
        bv.append(np.asarray(m.vertices, np.float64))
        bf.append(np.asarray(m.triangles, np.int64) + boff)
        boff += len(m.vertices)
    BV = np.vstack(bv)
    BF = np.vstack(bf)
    body_tol = _slab_depth(params) + 1.2

    # --- OpenVDB-style narrow band --------------------------------------
    # padded per-column depth bounds.  Rib-corridor columns (near pattern
    # centerlines) get the tall band; the rest only enough for the
    # sub-surface slab — the dominant query saving on flat faces.
    lowpad = params.embed + _slab_depth(params) + max(0.6, 2.0 * res)
    highpad = params.height + max(params.fillet_root, params.fillet_top) \
        + max(0.6, 2.0 * res)
    dlo = (Dlo - lowpad).astype(np.float32)
    dhi = np.where(P <= reach, Dhi + highpad,
                   Dhi + max(0.6, 2.0 * res)).astype(np.float32)

    surface = SurfaceField(cell=cell, origin=origin, P=P, B=B,
                           mask=inset_mask, Bo=Bo, V=V, F=F, N=N,
                           dlo=dlo, dhi=dhi, BV=BV, BF=BF,
                           body_tol=body_tol, ctr=ctr,
                           M=np.ascontiguousarray(
                               np.column_stack([axes[:, 0], axes[:, 1],
                                                n_axis])))
    warn = []
    clusters = mesh_field(surface, params, resolution=res, reports=warn)
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
