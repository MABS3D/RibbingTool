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
import os
import time
from dataclasses import dataclass

import numpy as np

_SINK = 0.05                    # slab top sits this far under the surface

_CAP_RUNOUT = True              # capped run-out termination;
                                # False = legacy floor-stub to the B-cut
_CAP_STEEP = 3.0                # cap forward semi-axis = half/_CAP_STEEP:
                                # at 1x-2x the root smooth-min rakes the
                                # cap-height end wall into a ~55-degree
                                # upward-facing bevel (k exceeds the wall
                                # height); at 3x the skirt ring sinks
                                # below knee height (verified by sweep)

_CAP_GUSSET = True              # end gusset: the root smooth-min k ramps
                                # up to ~1.2x the plateau height over the
                                # last ~thickness of run-out and holds past
                                # the trim, melting the cap nose into the
                                # body (no proud shoulder).  False = plain
                                # nose + user fillet only (bit-identical
                                # pre-gusset engine)

_GATE_CAP = True                # capped termination at the facing gate:
                                # every rib/web ends in the same in-plane
                                # rounded nose as the taper cap, complete
                                # before the top-sheet facing falls under
                                # _GATE_HI — the raw 0.09 backstop can no
                                # longer slice standing material into the
                                # 81-85deg shard tatters.  False = binary
                                # cut only (bit-identical legacy engine)
_GATE_HI = 0.15                 # top-sheet facing of the cap contour
                                # (never below 0.09 + the observed
                                # +-0.045 facet wobble of raw nzc)

EXTRACTOR = "marching_cubes"    # default; env RIBBING_EXTRACTOR overrides.
                                # surface_nets (exact-snap dual, no
                                # Taubin) beats MC on every measurable
                                # crest metric but cannot push the
                                # stitch test's diagonal steep runs
                                # under 5um (those runs sample a window-
                                # edge chord envelope of the leaning
                                # crown flank, floor ~8um even at 16x
                                # density) — default stays MC until that
                                # contract is settled
SN_RELAX = 4                    # constrained relax iterations.  4 vs 6
                                # measured on wrap_cyl: fall-line crest
                                # runs 0.30-0.38um vs 0.16-0.28um (gate
                                # 0.5um), diag runs and crest radius
                                # unchanged, |f| p90 3.2um both — and
                                # the ghost halo shrinks with it
SN_EXACT = 2                    # last iterations snap to the exact field
SN_LAM = 0.5                    # Laplacian step weight
SN_GHOST = SN_RELAX + 1         # tile ghost cells: cells >= SN_RELAX+1
                                # from the pad rim relax on complete
                                # data, so tile welds are bitwise exact

_CREASE_TAU_LO = 1.0            # medial-crease detection (foot-uv jump
_CREASE_TAU_HI = 3.0            # floors, mm): opposing-normal tier /
                                # any-normal tier
_CREASE_NDOT = math.cos(math.radians(25.0))
_CREASE_BETA = 8.0              # distance-excess penalty slope: the far
                                # sheet's branch is exactly inert past
                                # ~(|f1-f2|+k_c)/beta of extra distance
                                # (env RIBTOOL_CREASE_BLEND=0 disables
                                # the whole blend at call time)


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
    OB: np.ndarray = None       # 2D raster: exact distance to the OPEN
                                # substrate boundary curves (front-facing
                                # edges only): rim termination + wedge cut
    Pgu: np.ndarray = None      # 2D rasters: gradient of P (unit-ish
    Pgv: np.ndarray = None      # inside reach) — surface-metric width
    NXr: np.ndarray = None      # 2D rasters: gaussian-smoothed surface
    NYr: np.ndarray = None      # normal (the stretch factor rings at
    NZr: np.ndarray = None      # facet pitch on raw pseudo-normals)
    Bv: np.ndarray = None       # 2D raster: distance INTO the void
                                # (0 inside the domain) — caps how far
                                # wall blades may lean over openings
    BV: np.ndarray = None       # closed BODY mesh (whole part, WORLD
    BF: np.ndarray = None       # coords): negative-side material must
    body_tol: float = 0.0       # live inside it (the nTop boolean-with-
                                # the-body; kills the fake walls the
                                # closest-point map grows in the void
                                # between folded sheets)
    ctr: np.ndarray = None      # frame->world transform for body queries
    M: np.ndarray = None        # (3,3) frame axis columns
    Bg: np.ndarray = None       # 2D raster: signed exact distance to the
                                # top-sheet-facing = _GATE_HI contour
                                # (positive on the ribbable side) — the
                                # gate-cap trim station


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
    pseudo-normal.  Also returns the cp facing (pseudo-normal component
    along the projection axis) for the fold-over gate.  Chunked to bound
    peak memory on multi-megavoxel bands.
    """
    import igl
    s = np.empty(len(X), np.float32)
    C = np.empty((len(X), 3))
    nrm = np.empty((len(X), 3), np.float32)
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
        # height coordinate: HYBRID of the normal projection and the
        # Euclidean distance.  |x-cp| arcs over every facet edge of the
        # tessellated substrate and stamps the facet rows into rib walls
        # and crowns as rings; the projection |side| is C1 across facets
        # (exact on smooth patches) — but goes ~0 in the tangent
        # extension past open boundaries, where material would escape
        # along the edge plane.  Blend by the divergence d-|side|: tiny
        # on facet arcs (projection wins, rings gone), large in tangent
        # zones (Euclidean wins, field stays contained).
        e = d - np.abs(side)
        w = np.clip(e / 0.35, 0.0, 1.0)
        mag = np.abs(side) + e * w
        s[sl] = np.where(ok, np.where(side >= 0.0, mag, -mag),
                         d).astype(np.float32)
        C[sl] = Cp
        nrm[sl] = n.astype(np.float32)
    return s, C, nrm


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


def _rib_field(s, P, B, inside, Bo, params, cell, nzc, ob=None, res=None,
               stretch=np.float32(1.0), bout=None, bg=None):
    """Rib prism + root blend + slab in TRUE surface coordinates.

    s: exact signed distance to the substrate (negative inside the
    body); P/mask sampled at the smooth normal foot, B/Bo additionally
    constrained by the actual closest point at open substrate rims;
    nzc: cp facing (pseudo-normal along the projection axis); ob: cp
    distance to the open substrate boundary; res: marching resolution
    (feature-resolvability floors track it, NOT the raster cell);
    stretch: local projected->surface metric factor along the pattern
    gradient (surface-constant rib width).
    """
    if res is None:
        res = 2.0 * cell
    height = np.float32(params.height)
    # cp facing clamped away from zero: the facing IS the local
    # cos(slope), shared by the taper run-out and the gate cap to keep
    # projected<->surface conversions finite at the silhouette
    nz_c = np.maximum(nzc, np.float32(0.2))
    q_end = None
    if params.taper_len > 0 and not params.border:
        # the run-out ramp follows the OPEN boundary only: domain rims
        # are interior transitions and must not fade the height.
        # taper_len is a SURFACE run-out length: Bo is a PROJECTED
        # distance and a slope stretches it by 1/cos on the surface —
        # uncorrected, a 75-degree roll turns a 5mm run-out into a 20mm
        # fan of pointed stubs.  The cp facing IS the local cos(slope).
        t_eff = np.float32(params.taper_len) * nz_c
        height = height * np.clip(Bo / t_eff, 0.0, 1.0)
        # a feather thinner than ~2 voxels cannot be resolved — it
        # aliases into torn lace along every tapered boundary.  Floor the
        # run-out height.
        h_end = np.float32(min(2.2 * res, 0.55 * params.height))
        if _CAP_RUNOUT:
            # capped termination: the ramp stops at TWO floors (same
            # 55%-of-height cap) and the rib ENDS in a rounded nose at
            # the station where the un-floored ramp would reach h_end,
            # kept clear of the binary B setback so the max() cut
            # cannot slice the tip into debris.  Profile: full ramp ->
            # short plateau -> blunt nose (the nTop swept-strut cap).
            # One floor is not enough: the nose's crown roll descends
            # ~r_top/2 below its crest and the root-blend skirt climbs
            # the end wall, and against a one-floor wall both land back
            # in the sub-resolvable stub band the cap exists to
            # eliminate (verified by sweep; two floors clear it).
            h_plat = np.float32(min(4.4 * res, 0.55 * params.height))
            height = np.maximum(height, h_plat)
            q_end = np.maximum(
                t_eff * np.float32(h_end / params.height),
                np.float32(params.thickness / (2.0 * _CAP_STEEP)) * nz_c
                + np.float32(1.2 * cell) + np.float32(max(res, 0.3)))
        else:
            height = np.maximum(height, h_end)

    half = np.float32(params.thickness / 2.0)
    if params.draft_deg > 0:
        half = half + math.tan(math.radians(params.draft_deg)) \
            * np.clip(height - s, 0.0, None)

    # rib WIDTH is surface-metric: P is a PROJECTED distance that maps to
    # P*stretch of actual surface — uncorrected, a rib crossing a steep
    # slope becomes a thickness/cos(slope) wide flat leaf on the surface
    # (with staircase tops — the "fern").  stretch is the local metric
    # factor along the pattern gradient, 1.0 on flats and for ribs
    # running down the fall line.
    lat = P * stretch
    wall = lat - half
    if q_end is not None:
        # trimmed-centerline SDF: hypot of the lateral distance and the
        # (steepened) surface-metric overshoot past the trim station —
        # the run-out ends in a rounded plan-view nose, C1 at m = 0 and
        # bit-identical wherever the rib survives (m == 0)
        m = np.clip(q_end - Bo, np.float32(0.0), None) \
            * (np.float32(_CAP_STEEP) / nz_c)
        wall = np.hypot(lat, m) - half
    if bg is not None:
        # in-plane gate cap: a second trim of the same centerline SDF,
        # its station the top-sheet facing contour at _GATE_HI plus a
        # q_g clearance (projected mm; the nose's forward semi-axis
        # half*nz_c/_CAP_STEEP always fits inside it, so the raw 0.09
        # backstop below can never slice a surviving tip).  The nested
        # hypot is the exact SDF of the doubly-trimmed centerline —
        # a rounded plan-view corner where an open rim meets a steep
        # band.  Wall-only: the sub-surface slab keeps its extent down
        # to the raw backstop (cluster connectivity, body coverage).
        q_g = np.float32(max(res, 0.3))
        mg = np.clip(q_g - bg, np.float32(0.0), None) \
            * (np.float32(_CAP_STEEP) / nz_c)
        wall = np.hypot(wall + half, mg) - half
    # sharp crown corners cannot be represented sub-voxel: marching cubes
    # makes the corner line wobble half a voxel along its length, which
    # reads as lengthwise wrinkles on every rib top.  Floor the rounding
    # at ~1.2 voxels — visually sharp at part scale, resolvable on the
    # grid.
    r_top = float(np.clip(max(params.fillet_top, 1.2 * res),
                          0.0, 0.35 * params.thickness))
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
    k_end = float(min(1.2 * float(h_plat), 0.6 * params.height, 2.0)) \
        if (_CAP_GUSSET and q_end is not None) else 0.0
    if k_end > k + 1e-6:
        # end gusset: the root blend radius swells to swallow the
        # plateau-high nose over the last ~thickness of surface run and
        # holds past the trim station, so the skirt ring melts the nose
        # into the body (the injection-molding end fillet).  The top
        # surface never fades (feather-proof by construction); the
        # fillet surface is the only legal path to zero.  Smootherstep
        # onset — a C0 kink in k(x) prints a faint crease.  Where the
        # user's fillet_root already exceeds k_end the guard above
        # keeps the scalar path bit-identical.
        L_p = np.float32(max(params.thickness, 3.0 * res)) * nz_c
        t = np.clip((q_end + L_p - Bo) / L_p, np.float32(0.0),
                    np.float32(1.0))
        t = t * t * (np.float32(3.0) - np.float32(2.0) * t)
        k_arr = np.maximum(np.float32(k), np.float32(k_end) * t)
        kk = np.maximum(k_arr, np.float32(1e-6))
        h = np.clip(0.5 + 0.5 * (shell - prism) / kk, 0.0, 1.0)
        f = prism * h + shell * (1 - h) - k_arr * h * (1 - h)
    elif k > 0:
        h = np.clip(0.5 + 0.5 * (shell - prism) / k, 0.0, 1.0)
        f = prism * h + shell * (1 - h) - k * h * (1 - h)
    else:
        f = np.minimum(prism, shell)

    f = np.maximum(f, -s - _slab_depth(params))          # slab floor cap
    # the slab must end on the domain boundary too: applied after the cap
    # (the prism clamp above cannot reach it), otherwise the sheet follows
    # the pixelated mask edge and frays at every recess rim
    f = np.maximum(f, 1.2 * np.float32(cell) - B)
    # facing gate: projected mapping cannot parameterize surface that
    # does not actually face the projection.  Back sheets (nz<0) would
    # get a mirrored lattice; NEAR-SILHOUETTE surface (|nz|~0) squashes
    # the whole pattern into a strip a few cells wide and shreds it into
    # torn flaps.  One smooth cut on the cp facing handles both: full
    # material by nz~0.15 (81 deg), gone below nz~0.09 (85 deg) — steep
    # walls keep their ribs, sealed-slit bridges keep their edge-anchored
    # closest points, and the cut wraps crests as a clean end cap.
    f = np.maximum(f, np.float32(25.0) * (np.float32(0.09) - nzc))
    # tangent-wedge cut: beyond an OPEN substrate boundary the closest
    # point snaps to the edge and the side test goes negative under the
    # edge's tangent plane — a floating slab wedge fans off every panel
    # edge into sealed strips and valleys (bare smears on real parts).
    # Kill slab-side material whose cp hugs the open boundary; rib
    # interiors (prism<0) keep their embedment so bridge ribs stay
    # rooted, and material above the surface is untouched.
    if ob is not None:
        f = np.maximum(f, np.where((s < 0.0) & (prism > 0.0),
                                   np.float32(2.0)
                                   * (np.float32(1.2) - ob),
                                   np.float32(-1e3)))
    # lean cap: wall blades extend horizontally over openings by up to
    # height*sin(slope) — a window must stay CLEAR.  Material may round
    # over the domain edge by ~a thickness, never hang further; measured
    # at the SAMPLE's own (u,v) (it is the blade body that leans, not
    # its root).
    if bout is not None:
        f = np.maximum(f, np.float32(2.0) * (bout - np.float32(1.6)))
    air = np.float32(max(4.0 * cell, 0.6))
    return np.where(inside, f.astype(np.float32), air)


def _safe_progress(progress):
    """Wrap a progress callback so it can never break a build: the first
    exception is logged once, then the callback is disabled.  None stays
    None, so the default path costs nothing."""
    if progress is None:
        return None
    box = [progress]

    def _cb(stage, done, total):
        if box[0] is None:
            return
        try:
            box[0](stage, done, total)
        except Exception:
            box[0] = None
            import logging
            logging.getLogger(__name__).warning(
                "progress callback raised; reporting disabled",
                exc_info=True)
    return _cb


def mesh_field(surface, params, resolution, tile=None, reports=None,
               extractor=None, progress=None, field=None):
    """Extract the field's zero set on the global grid; returns
    [(verts, tris)] in frame coordinates (one welded solid), or [] if
    the field is empty.  extractor: "marching_cubes" (default) |
    "surface_nets" (exact-snap dual); env RIBBING_EXTRACTOR overrides
    the module default at call time.  tile=None resolves per mode:
    256 for surface_nets (halves the relative SN_GHOST halo overhead;
    vol ~42MB), 192 for marching_cubes (path byte-identical).
    field: optional continuous scalar-field callable in frame coordinates.
    When supplied it owns geometry; legacy crease and orphan heuristics are
    bypassed, and the callable is preserved if extraction is retried.
    progress: optional (stage, done, total) callback — reporting only,
    never touches the build (a raising callback is disabled after one
    logged failure)."""
    progress = _safe_progress(progress)
    mode = extractor or os.environ.get("RIBBING_EXTRACTOR", EXTRACTOR)
    tile_arg = tile
    if tile is None:
        tile = 256 if mode == "surface_nets" else 192
    if mode != "surface_nets":
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
    maskf = g.mask.astype(np.float32)

    def _sample_at_foot(cu, cv, cp):
        """Raster block shared by both field branches: pattern,
        boundary, open-boundary and width-stretch data at a foot
        (u, v)."""
        P = _bilinear(g.P, cu, cv, g.cell, g.origin)
        B = _bilinear(g.B, cu, cv, g.cell, g.origin)
        Bo = _bilinear(g.Bo, cu, cv, g.cell, g.origin)
        if g.OB is not None:
            # The smooth pattern foot X-s*n can leave the selected mesh.
            # Near an opening another projected panel can still supply a
            # positive B there, growing a rib from the rim into empty air.
            # Terminate using the actual closest point, which stays on
            # that rim, while retaining the smooth foot for the lattice.
            obcp = _bilinear(g.OB, cp[:, 0], cp[:, 1], g.cell, g.origin)
            bcp = _bilinear(g.B, cp[:, 0], cp[:, 1], g.cell, g.origin)
            reach = params.height + params.margin + params.thickness
            # A sealed internal slit must still allow bridge ribs. Fade
            # this extra constraint beyond the possible rib reach from
            # the outer domain, instead of a binary edge classification.
            rim = obcp - params.margin + np.maximum(bcp - reach, 0.0)
            B = np.minimum(B, rim)
            Bo = np.minimum(Bo, rim)
        obv = (_bilinear(g.OB, cu, cv, g.cell, g.origin)
               if g.OB is not None else None)
        if g.Pgu is not None:
            # unit-gradient guard rejects the reach frontier; the cap
            # keeps the projected band >= ~4 raster cells so the wall
            # never sits sub-cell from the P crease (bamboo rings)
            gpu = _bilinear(g.Pgu, cu, cv, g.cell, g.origin)
            gpv = _bilinear(g.Pgv, cu, cv, g.cell, g.origin)
            nxs = _bilinear(g.NXr, cu, cv, g.cell, g.origin)
            nys = _bilinear(g.NYr, cu, cv, g.cell, g.origin)
            nzs = _bilinear(g.NZr, cu, cv, g.cell, g.origin)
            gl = np.hypot(gpu, gpv)
            okg = (gl > 0.3) & (gl < 3.0)
            along = np.where(
                okg,
                (nxs * gpu + nys * gpv)
                / (np.clip(gl, 0.3, None) * np.clip(nzs, 0.12, None)),
                0.0)
            cap = np.float32(max(1.0, params.thickness / (4.0 * g.cell)))
            stretch = np.minimum(np.sqrt(1.0 + along * along),
                                 cap).astype(np.float32)
            del gpu, gpv, nxs, nys, nzs, gl, okg, along
        else:
            stretch = np.float32(1.0)
        bg = (_bilinear(g.Bg, cu, cv, g.cell, g.origin)
              if g.Bg is not None else None)
        return P, B, Bo, obv, stretch, bg

    def _core(X, s, C, nrm):
        """Field from a precomputed substrate query: _field_at =
        _surface_eval + _core, and the tile loop calls _core directly so
        the crease blend can reuse the one query."""
        nzc = nrm[:, 2]
        foot = X - s[:, None] * nrm
        cu = foot[:, 0]
        cv = foot[:, 1]
        del foot
        P, B, Bo, obv, stretch, bgv = _sample_at_foot(cu, cv, C)
        bout = (_bilinear(g.Bv, X[:, 0], X[:, 1], g.cell, g.origin)
                if g.Bv is not None else None)
        ins = _bilinear(maskf, cu, cv, g.cell, g.origin) > 0.5
        # beyond the raster there is no surface: cap as air by the
        # sample's own position, or marching cubes leaves open sheets
        ins &= ((X[:, 0] >= u_lo) & (X[:, 0] <= u_hi)
                & (X[:, 1] >= v_lo) & (X[:, 1] <= v_hi))
        vals = _rib_field(s, P, B, ins, Bo, params, g.cell, nzc,
                          obv, res, stretch, bout, bgv)
        del P, B, Bo, ins, cu, cv, nzc, obv, stretch, bout, bgv
        # nTop-style boolean with the body: NEGATIVE-side material (the
        # slab) must live inside the BODY — the closest-point map grows
        # fake walls in the void between folded sheets otherwise.  Rib-
        # side voxels are exempt (a body cut at rib height would eat
        # every rib).  frame->world is @ M.T (+ctr): M holds the frame
        # axes as COLUMNS, and @ M is the world->frame rotation.
        # (Running before the crease blend is provably equivalent to
        # after it: blend-touched samples have s < ~0.5 only within
        # 0.5mm of the substrate, where the body SDF cannot exceed
        # body_tol, so the gate is inert on them either way.)
        if g.BV is not None:
            import igl
            cand = (vals < 0.0) & (s < 0.5)
            if cand.any():
                Xw = np.ascontiguousarray(X[cand] @ g.M.T + g.ctr)
                out = igl.signed_distance(Xw, g.BV, g.BF)
                vals[cand] = np.maximum(
                    vals[cand], out[0].astype(np.float32) - g.body_tol)
                del Xw, out
        return vals

    def _field_at(X):
        """Complete field at arbitrary frame points — the tile marcher
        and the vertex re-projection polish share this ONE definition.

        Pattern/boundary/mask are sampled at the PHONG FOOT x - s*n(cp):
        the interpolated pseudo-normal is smooth across substrate facets
        where the raw closest point jumps by ~height*dihedral (corduroy
        walls otherwise).  The width factor comes from the SMOOTHED
        normal rasters; the lean cap and the raster window use the
        sample's own position.
        """
        if field is not None:
            return field(X)
        s, C, nrm = _surface_eval(X, g.V, g.F, g.N)
        return _core(X, s, C, nrm)

    k_c = np.float32(min(0.35, max(1.2 * res, 0.2)))
    beta = np.float32(_CREASE_BETA)
    scap = np.float32(max(1.0, params.thickness / (4.0 * g.cell)))

    def _crease_blend(vals, band, s, C, nrm, us, vs, ds):
        """Two-branch medial-crease blend: a grid-only post-pass.

        Across a concave channel's medial sheet the closest-point foot
        flips wall sheets, so every foot-sampled raster (P/B/OB/
        stretch/facing) jumps and the COMPOSED field is discontinuous —
        ribs crossing the channel braid (s itself is C0; the jump lives
        in the samples).  Detect crease faces between adjacent in-band
        samples, harvest the second sheet from the neighbor across the
        jump (its cp + interpolated normal give a first-order tangent-
        plane branch — no second BVH query), evaluate the FULL rib
        field on that branch, and combine with the compact-support
        polynomial smooth-min after a distance-excess penalty
        beta*(|s2|-|s1|)+ that makes the far branch EXACTLY inert
        beyond ~2 voxels of the medial sheet (F == f1 wherever
        g2 >= f1 + k_c: no exp tails, no-crease tiles byte-identical).
        Returns (vals, cell_flags); cell_flags marks vol cells touching
        a blended sample — the SN exact-snap mask (the single-branch
        exact field is intentionally WRONG there, and snapping onto it
        would re-braid the seam).
        """
        shp = band.shape
        # staged detection: both crease tiers strictly require a foot
        # jump > tau1 across some grid face, and the feet need only an
        # 8B/sample FU/FV scatter — the S/NR/CP dense arrays (28B/sample
        # more) are deferred until a candidate face exists, so
        # crease-free slabs (the common case) never allocate them.
        bi = np.nonzero(band)
        FU = np.full(shp, np.nan, np.float32)
        FU[bi] = us.astype(np.float32)[bi[0]] - s * nrm[:, 0]
        FV = np.full(shp, np.nan, np.float32)
        FV[bi] = vs.astype(np.float32)[bi[1]] - s * nrm[:, 1]
        del bi
        tau1 = np.float32(max(_CREASE_TAU_LO, 4.0 * res))
        cand = False
        for ax in range(3):
            sl0 = tuple(slice(None, -1) if a == ax else slice(None)
                        for a in range(3))
            sl1 = tuple(slice(1, None) if a == ax else slice(None)
                        for a in range(3))
            ok = band[sl0] & band[sl1]
            if not ok.any():
                continue
            dj = np.where(ok, np.hypot(FU[sl1] - FU[sl0],
                                       FV[sl1] - FV[sl0]),
                          np.float32(0.0))
            cand = bool((dj > tau1).any())
            del dj
            if cand:
                break
        if not cand:
            return vals, None
        S = np.full(shp, np.nan, np.float32)
        S[band] = s
        NR = np.full(shp + (3,), np.nan, np.float32)
        NR[band] = nrm
        CP = np.full(shp + (3,), np.nan, np.float32)
        CP[band] = C
        out = _crease_partners(band, S, NR, CP, FU, FV, res, scap)
        del FU, FV
        if out is None:
            return vals, None
        flag, pdir = out
        fi, fj, fk = np.nonzero(flag)
        code = pdir[fi, fj, fk]
        okp = code >= 0
        if not okp.any():
            return vals, None
        fi, fj, fk, code = fi[okp], fj[okp], fk[okp], code[okp]
        stepv = np.where((code & 1) == 1, 1, -1)
        axc = code >> 1
        pi = fi + np.where(axc == 0, stepv, 0)
        pj = fj + np.where(axc == 1, stepv, 0)
        pk = fk + np.where(axc == 2, stepv, 0)
        Xf = np.column_stack([us[fi], vs[fj], ds[fk]])
        Cp = CP[pi, pj, pk].astype(np.float64)
        npn = NR[pi, pj, pk].astype(np.float64)
        del CP
        # first-order tangent-plane distance to the partner sheet
        # (error O(kappa*res^2)); its foot feeds the same raster block
        s2 = np.einsum("ij,ij->i", Xf - Cp, npn)
        cu2 = Xf[:, 0] - s2 * npn[:, 0]
        cv2 = Xf[:, 1] - s2 * npn[:, 1]
        P2, B2, Bo2, ob2, st2, bg2 = _sample_at_foot(cu2, cv2, Cp)
        ins2 = _bilinear(maskf, cu2, cv2, g.cell, g.origin) > 0.5
        ins2 &= ((Xf[:, 0] >= u_lo) & (Xf[:, 0] <= u_hi)
                 & (Xf[:, 1] >= v_lo) & (Xf[:, 1] <= v_hi))
        bout2 = (_bilinear(g.Bv, Xf[:, 0], Xf[:, 1], g.cell, g.origin)
                 if g.Bv is not None else None)
        f2 = _rib_field(s2.astype(np.float32), P2, B2, ins2, Bo2,
                        params, g.cell, npn[:, 2].astype(np.float32),
                        ob2, res, st2, bout2, bg2)
        F1 = np.zeros(shp, np.float32)
        F1[band] = vals
        f1 = F1[fi, fj, fk]
        exc = np.maximum(np.abs(s2).astype(np.float32)
                         - np.abs(S[fi, fj, fk]), np.float32(0.0))
        g2 = f2 + beta * exc
        h = np.clip(0.5 + 0.5 * (f1 - g2) / k_c, 0.0, 1.0)
        F1[fi, fj, fk] = (g2 * h + f1 * (1.0 - h)
                          - k_c * h * (1.0 - h)).astype(np.float32)
        vals = F1[band]
        touched = np.zeros(shp, bool)
        touched[fi, fj, fk] = True
        c0, c1, c2 = (n - 1 for n in shp)
        cfl = np.zeros((c0, c1, c2), bool)
        for du, dv, dd in _SN_CORNERS:
            cfl |= touched[du:du + c0, dv:dv + c1, dd:dd + c2]
        return vals, cfl

    blend = field is None and os.environ.get("RIBTOOL_CREASE_BLEND", "1") != "0"
    all_v, all_f, off = [], [], 0
    DSLAB = 128                       # max d-slices per marching box
    # marching_cubes: one-cell overlap — neighbors evaluate the shared
    # cells from identical samples and produce identical triangles; each
    # triangle is kept by the tile owning its centroid, so tile joints
    # cannot crack.  surface_nets: SN_GHOST-cell halo — every emitted
    # cell relaxes on complete data, ownership is exact per tail cell.
    # Crease blend on: a blended value at sample i depends on raw
    # samples [i-2, i+2] (flag = dilation(1) of crease faces at i+-1,
    # which read i+-2; partner at i+-1), so weld exactness needs +2
    # halo on both paths — MC shared-plane corners [ta, tb] then see
    # complete data [ta-2, tb+2] in both tiles; SN owned-cell vertices
    # depend on samples within SN_RELAX+1, blended: SN_RELAX+3.
    if mode == "surface_nets":
        G = SN_GHOST + (2 if blend else 0)
    else:
        G = 2 if blend else 1
    BIG = 1 << 30
    # progress: the (ta, tc) tile grid is known up front; every tile ticks
    # exactly once (skipped/empty tiles too, so done always reaches total)
    n_tiles = (math.ceil((iu1 - iu0) / tile)
               * math.ceil((iv1 - iv0) / tile))
    tiles_done = 0

    def _tile_done():
        nonlocal tiles_done
        tiles_done += 1
        if progress:
            progress("extracting tile", tiles_done, n_tiles)

    if progress:
        progress("extracting tile", 0, n_tiles)
    try:
        for ta in range(iu0, iu1, tile):
            tb = min(ta + tile, iu1)
            for tc in range(iv0, iv1, tile):
                td = min(tc + tile, iv1)
                us = np.arange(ta - G, tb + G + 1) * res
                vs = np.arange(tc - G, td + G + 1) * res
                # nearest raster column per sample
                gi = np.clip(np.round((us - u_lo)
                                      / g.cell).astype(np.int64),
                             0, nu - 1)
                gj = np.clip(np.round((vs - v_lo)
                                      / g.cell).astype(np.int64),
                             0, nv - 1)
                if not g.mask[gi][:, gj].any():
                    _tile_done()
                    continue
                blo = g.dlo[gi][:, gj]
                bhi = g.dhi[gi][:, gj]
                d0 = int(math.floor(float(blo.min()) / res))
                d1 = int(math.ceil(float(bhi.max()) / res))
                if d1 <= d0:
                    _tile_done()
                    continue
                # tall folded regions span enormous depth ranges; march
                # in d-slabs so no box explodes (same joint contracts)
                for e0 in range(d0, d1, DSLAB):
                    e1 = min(e0 + DSLAB, d1)
                    ds = np.arange(e0 - G, e1 + G + 1) * res
                    # OpenVDB-style narrow band: evaluate only within
                    # the per-column padded depth bounds (no meshgrids —
                    # the band mask is built by broadcasting and points
                    # are gathered by index)
                    band = ((ds[None, None, :] >= blo[:, :, None])
                            & (ds[None, None, :] <= bhi[:, :, None]))
                    if not band.any():
                        continue
                    iu, iv, iw = np.nonzero(band)
                    X = np.column_stack([us[iu], vs[iv], ds[iw]])
                    if field is not None:
                        vals = field(X)
                        eok = None
                        del X
                    else:
                        sQ, CQ, nrmQ = _surface_eval(X, g.V, g.F, g.N)
                        vals = _core(X, sQ, CQ, nrmQ)
                        eok = None
                        if blend:
                            vals, cfl = _crease_blend(vals, band, sQ, CQ,
                                                      nrmQ, us, vs, ds)
                            if cfl is not None:
                                eok = ~cfl
                            del cfl
                        del X, sQ, CQ, nrmQ
                    vol = np.full(band.shape, air, np.float32)
                    vol[band] = vals
                    del band, vals, iu, iv, iw
                    if vol.min() >= 0 or vol.max() <= 0:
                        continue
                    if mode == "surface_nets":
                        own_lo = np.array(
                            [G if ta != iu0 else -BIG,
                             G if tc != iv0 else -BIG,
                             G if e0 != d0 else -BIG], np.int64)
                        own_hi = np.array(
                            [G + (tb - ta) if tb < iu1 else BIG,
                             G + (td - tc) if td < iv1 else BIG,
                             G + (e1 - e0) if e1 < d1 else BIG], np.int64)
                        out = _surface_nets_tile(vol, us, vs, ds, own_lo,
                                                 own_hi, _field_at, res,
                                                 exact_ok=eok)
                        del vol
                        if out is None:
                            continue
                        verts, faces = out
                    else:
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
                _tile_done()
    except Exception as ex:
        # self-heal: one retry on the proven extractor, attributably
        if mode != "surface_nets":
            raise
        if reports is not None:
            reports.append(f"surface_nets extractor failed ({ex}); "
                           "retried with marching_cubes")
        return mesh_field(surface, params, resolution, tile_arg, reports,
                          extractor="marching_cubes", field=field,
                          **({"progress": progress} if progress else {}))

    if not all_v:
        return []
    if progress:
        progress("welding", 0, 0)
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
    kvol = []
    for cid in np.unique(lab[f[:, 0]]):
        fc = f[lab[f[:, 0]] == cid]
        vol = np.einsum("ij,ij->i", v[fc[:, 0]],
                        np.cross(v[fc[:, 1]], v[fc[:, 2]])).sum() / 6.0
        min_volume = 8.0 if field is None else res ** 3
        if abs(vol) < min_volume or len(fc) < 24:
            continue
        keep.append(fc[:, ::-1] if vol < 0 else fc)
        kvol.append(abs(vol))
    if not keep:
        return []
    if field is None and len(keep) > 1:
        # orphan-component guard: the leaning-crown band lid evaluates
        # field regions the per-column band used to clip, and the facing
        # gate can strand rib material on an isolated near-silhouette
        # sliver (an S-fold flank pinched between the cull and the open
        # boundary): a disconnected curl whose only roots are embed
        # grazes.  Every legitimately rooted component carries part of
        # its own slab FLOOR sheet (the -s-slab_depth cap surface,
        # carved under every masked column by construction; embed grazes
        # stop short of it), so a component with statistically no
        # floor-depth material is an orphan.  Multi-island selections
        # keep every island (each owns its floor); if NO component has a
        # floor there is no reference to trust, so keep everything.
        slabd = _slab_depth(params)
        tol = max(0.25, 0.6 * res)
        rooted = []
        for fc in keep:
            vi = np.unique(fc)
            vi = vi[::max(1, len(vi) // 20000)]
            sc, _, _ = _surface_eval(np.ascontiguousarray(v[vi]),
                                     g.V, g.F, g.N)
            rooted.append(float(np.mean(np.abs(sc + slabd) < tol)) > 0.02)
        if any(rooted) and not all(rooted):
            gone = sum(kv for kv, r in zip(kvol, rooted) if not r)
            keep = [fc for fc, r in zip(keep, rooted) if r]
            if reports is not None:
                reports.append(
                    f"dropped {sum(not r for r in rooted)} unrooted "
                    f"fragment(s), {gone:.0f}mm3 (no slab-floor material)")
    f = np.vstack(keep)
    # drop vertices orphaned by the crumb filter: they must not ride
    # along in the output (and must not be smoothed)
    used, f = np.unique(f, return_inverse=True)
    v = np.ascontiguousarray(v[used])
    f = np.ascontiguousarray(f.reshape(-1, 3))
    # NOTE on smoothing: SDF re-projection polish (smooth + Newton-project
    # back onto f=0 via _project_to_field) was benched and REJECTED here:
    # at 0.3mm voxels this lattice is crest-dominated, and projection is
    # ill-posed exactly at crests (flank-averaged normals disagree across
    # the crest line and zigzag it into sawteeth; selective-coherence
    # gating still nets out worse than plain Taubin).  Keep Taubin-6 on
    # the MC path.  The surface_nets path gets NO Taubin: its smoothing
    # budget lives inside the relax loop, bounded by the exact-field
    # projection and the cell clamp — post-hoc Taubin would drag crest
    # vertices off the surface (shrink) and re-create the stitch texture.
    if progress:
        progress("smoothing", 0, 0)
    if mode == "surface_nets":
        # naive SN's one non-manifold class _repair_pinches cannot split
        # (ambiguous-face duals: 4 quads ringing one mesh edge)
        v, f = _nonmanifold_edge_split(v, f)
    else:
        v = _taubin(v, f, rounds=6)
    # pinches first: boundary loops through pinched vertices cannot be
    # walked, so holes must be filled on the split mesh
    if progress:
        progress("repairing", 0, 0)
    v, f = _repair_pinches(v, f)
    if progress:
        progress("filling holes", 0, 0)
    v, f = _fill_microholes(v, f)
    if progress:
        progress("validating", 0, 0)
    # Validation is part of generation, even when the caller does not
    # collect reports. Never return an open mesh as a successful build.
    import manifold3d as m3d
    mesh = m3d.Mesh64(np.ascontiguousarray(v, np.float64),
                      np.ascontiguousarray(f, np.uint64))
    man = m3d.Manifold(mesh)
    if man.is_empty():
        mesh.merge()
        man = m3d.Manifold(mesh)
        if not man.is_empty():
            # merge() stores remapping metadata on mesh. Returning the
            # old v/f would discard the repair that just passed validation.
            repaired = man.to_mesh64()
            v = np.array(repaired.vert_properties[:, :3], dtype=np.float64,
                         order="C", copy=True)
            f = np.array(repaired.tri_verts, dtype=np.int64, order="C", copy=True)
    if man.is_empty():
        if mode == "surface_nets":
            if reports is not None:
                reports.append("surface_nets extractor failed manifold "
                               "validation; retried with marching_cubes")
            return mesh_field(surface, params, resolution, tile_arg, reports,
                              extractor="marching_cubes", progress=progress,
                              field=field)
        from .ribbing import RibbingError
        raise RibbingError("rib mesh is not a closed manifold solid "
                           f"({man.status()}); generation was not applied")
    return [(v, f.astype(np.int64))]


def _crease_partners(band, S, NR, CP, FU, FV, res, scap):
    """Grid-local medial-crease detection on one tile-slab.

    A grid face between adjacent in-band samples is a crease iff their
    pattern-sampling feet jump farther than legitimate same-sheet foot
    advection allows: tier 1 (opposing channel walls) jump >
    max(1mm, 4*res) with normals disagreeing by >25deg; tier 2 (any
    normals, large parallel-sheet flips) jump > max(3mm, 3x the
    expected steep-wall foot slide).  Either tier also requires the
    jump to be CARRIED by the closest point (|dcp| > 0.5*jump): at
    open-boundary tangent wedges the side test flips sign with cp
    pinned on the edge, throwing the foot by 2|s|*|n_uv| with |dcp| ~
    res — an edge artifact the OB cut already owns, not a two-sheet
    medial crease.  Returns (flag, pdir) — flagged samples (crease
    endpoints dilated by one within the band) and each sample's
    partner direction (the 6-neighbor with the largest foot jump,
    coded 2*axis + (1 if +step); -1 = none) — or None when no face is
    a crease (the caller's blend is then a provable no-op).
    """
    from scipy.ndimage import binary_dilation, generate_binary_structure
    shp = band.shape
    crease = np.zeros(shp, bool)
    tau1 = np.float32(max(_CREASE_TAU_LO, 4.0 * res))
    # partner argmax floors at tau1 (both crease tiers require dj > tau1):
    # ordinary same-sheet foot advection is ~res on every in-band face, so
    # an unfloored argmax hands every dilation-ring sample a same-sheet
    # partner with f2 ~ f1 — and smin(a, a) = a - k_c/4 stamps a proud
    # welt band beside every seam where the blend contract requires exact
    # inertness (F == f1).  Sub-threshold samples keep pdir = -1 and the
    # blend's okp gate skips them bit-exactly.
    bestj = np.full(shp, tau1, np.float32)
    pdir = np.full(shp, -1, np.int8)
    for ax in range(3):
        sl0 = tuple(slice(None, -1) if a == ax else slice(None)
                    for a in range(3))
        sl1 = tuple(slice(1, None) if a == ax else slice(None)
                    for a in range(3))
        ok = band[sl0] & band[sl1]
        if not ok.any():
            continue
        dj = np.where(ok, np.hypot(FU[sl1] - FU[sl0], FV[sl1] - FV[sl0]),
                      np.float32(0.0))
        # both tiers need dj > tau1 (tier 2's floor is higher), so the
        # expensive terms run only on the rare candidate faces
        cand = ok & (dj > tau1)
        if cand.any():
            ii = np.nonzero(cand)
            n0 = NR[sl0][ii]
            n1 = NR[sl1][ii]
            djv = dj[ii]
            dcv = np.linalg.norm(CP[sl1][ii] - CP[sl0][ii], axis=-1)
            ndv = (n0 * n1).sum(-1)
            nzv = 0.5 * np.abs(n0[:, 2] + n1[:, 2])
            mv = res * np.maximum(scap,
                                  np.sqrt(np.clip(1.0 - nzv * nzv,
                                                  0.0, 1.0))
                                  / np.clip(nzv, 0.09, None))
            crv = (dcv > 0.5 * djv) \
                & ((ndv < _CREASE_NDOT)
                   | (djv > np.maximum(np.float32(_CREASE_TAU_HI),
                                       3.0 * mv)))
            cr = np.zeros(dj.shape, bool)
            cr[ii] = crv
            crease[sl0] |= cr
            crease[sl1] |= cr
        upd = ok & (dj > bestj[sl0])
        pdir[sl0][upd] = 2 * ax + 1
        bestj[sl0][upd] = dj[upd]
        upd = ok & (dj > bestj[sl1])
        pdir[sl1][upd] = 2 * ax
        bestj[sl1][upd] = dj[upd]
    if not crease.any():
        return None
    flag = binary_dilation(crease, generate_binary_structure(3, 1)) & band
    return flag, pdir


# corner index = 4*du + 2*dv + dd; edges grouped d-, v-, u-axis
_SN_CORNERS = ((0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1),
               (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1))
_SN_EDGES = ((0, 1), (2, 3), (4, 5), (6, 7),
             (0, 2), (1, 3), (4, 6), (5, 7),
             (0, 4), (1, 5), (2, 6), (3, 7))
_SN_AXIS6 = ((-1, 0, 0), (1, 0, 0), (0, -1, 0),
             (0, 1, 0), (0, 0, -1), (0, 0, 1))


def _trilin_fg(CV, t):
    """Trilinear value + gradient (d/dt units) from 8 corner samples."""
    tu, tv, td = t[:, 0], t[:, 1], t[:, 2]
    c00 = CV[:, 0] * (1 - td) + CV[:, 1] * td
    c01 = CV[:, 2] * (1 - td) + CV[:, 3] * td
    c10 = CV[:, 4] * (1 - td) + CV[:, 5] * td
    c11 = CV[:, 6] * (1 - td) + CV[:, 7] * td
    c0 = c00 * (1 - tv) + c01 * tv
    c1 = c10 * (1 - tv) + c11 * tv
    f = c0 * (1 - tu) + c1 * tu
    gu = c1 - c0
    gv = (c01 - c00) * (1 - tu) + (c11 - c10) * tu
    gd = ((CV[:, 1] - CV[:, 0]) * (1 - tv)
          + (CV[:, 3] - CV[:, 2]) * tv) * (1 - tu) \
        + ((CV[:, 5] - CV[:, 4]) * (1 - tv)
           + (CV[:, 7] - CV[:, 6]) * tv) * tu
    return f, np.column_stack([gu, gv, gd])


def _surface_nets_tile(vol, us, vs, ds, own_lo, own_hi, field_at, res,
                       exact_ok=None):
    """SurfaceNets dual extractor for one tile: one free vertex per
    sign-changing cell, init at the mean of its edge crossings, then
    SN_RELAX iterations of (neighbor-average -> Newton-project toward
    f=0 along the field gradient -> clamp to cell); the last SN_EXACT
    projections evaluate the EXACT field so crowns land on the true
    zero set, not the trilinear surrogate.  Faces: one quad per
    sign-changing grid edge, wound by the tail sign, split on the
    shorter diagonal.  own_lo/own_hi are integer CELL bounds (tail-cell
    ownership partitions all edges exactly across tiles); every op is
    an elementwise gather with fixed term order, so shared cells come
    out bitwise identical in adjacent tiles and the weld is exact.

    exact_ok: optional per-cell bool mask (cell grid = vol.shape - 1).
    Cells masked False keep the trilinear surrogate even on the last
    iterations — for callers that modify grid values (medial crease
    blend), where the single-branch exact field is intentionally wrong.

    Returns (verts, tris) or None.
    """
    neg = vol < 0.0
    S0, S1, S2 = (n - 1 for n in vol.shape)
    occ = np.zeros((S0, S1, S2), np.uint8)
    for du, dv, dd in _SN_CORNERS:
        occ += neg[du:du + S0, dv:dv + S1, dd:dd + S2]
    cellmask = (occ > 0) & (occ < 8)
    del occ
    if not cellmask.any():
        return None
    K = int(cellmask.sum())
    cid = np.full(cellmask.shape, -1, np.int32)
    cid[cellmask] = np.arange(K, dtype=np.int32)
    ci, cj, ck = np.nonzero(cellmask)
    CV = np.stack([vol[ci + du, cj + dv, ck + dd].astype(np.float64)
                   for du, dv, dd in _SN_CORNERS], axis=1)      # (K,8)
    Cmin = np.column_stack([us[ci], vs[cj], ds[ck]])            # (K,3)
    CPOS = np.asarray(_SN_CORNERS, np.float64)
    # init: mean of edge crossings — already an O(res^2) surface
    # estimate, so the smoothing term never fights the surface term
    S = np.zeros((K, 3))
    W = np.zeros(K)
    for a, b in _SN_EDGES:
        fa, fb = CV[:, a], CV[:, b]
        x = (fa < 0.0) != (fb < 0.0)
        t = fa / np.where(x, fa - fb, 1.0)
        pt = CPOS[a] + t[:, None] * (CPOS[b] - CPOS[a])
        S[x] += pt[x]
        W[x] += 1.0
    P = Cmin + (S / W[:, None]) * res
    del S, W
    # 6-neighbor table (fixed order: the L-step sum must be
    # order-identical across tiles)
    NB = np.full((K, 6), -1, np.int32)
    for k6, (su, sv, sd) in enumerate(_SN_AXIS6):
        qi, qj, qk = ci + su, cj + sv, ck + sd
        ok = ((qi >= 0) & (qi < S0) & (qj >= 0) & (qj < S1)
              & (qk >= 0) & (qk < S2))
        NB[ok, k6] = cid[qi[ok], qj[ok], qk[ok]]
    eo = None if exact_ok is None else exact_ok[ci, cj, ck]
    eps = 1e-3 * res
    for it in range(SN_RELAX):
        nb = P[np.clip(NB, 0, None)]                         # (K,6,3)
        m = NB >= 0
        cnt = m.sum(axis=1)
        Lp = (nb * m[:, :, None]).sum(axis=1) \
            / np.clip(cnt, 1, None)[:, None]
        has = cnt > 0
        P[has] += SN_LAM * (Lp[has] - P[has])
        del nb, m, Lp
        t = np.clip((P - Cmin) / res, 0.0, 1.0)
        f, gg = _trilin_fg(CV, t)
        gg /= res
        if it >= SN_RELAX - SN_EXACT:
            # exact snap: gradient stays trilinear (recycled corners,
            # zero evals), the residual is the true field.  Restricted
            # to cells that can still influence owned output: owned
            # quads reference cells in [own_lo-1, own_hi), and a
            # position set in this iteration propagates one cell per
            # remaining L-step, so only cells within R = SN_RELAX-1-it
            # of that box matter — deeper ghost cells keep the
            # trilinear surrogate (bitwise no-op on owned vertices).
            # Cells within R of a tile plane sit inside BOTH adjacent
            # tiles' bands, so shared cells still relax on identical
            # choices and the weld stays bitwise exact.
            R = SN_RELAX - 1 - it
            sel = ((ci >= own_lo[0] - 1 - R) & (ci < own_hi[0] + R)
                   & (cj >= own_lo[1] - 1 - R) & (cj < own_hi[1] + R)
                   & (ck >= own_lo[2] - 1 - R) & (ck < own_hi[2] + R))
            if eo is not None:
                sel &= eo
            if sel.all():
                f = field_at(np.ascontiguousarray(P)).astype(np.float64)
            elif sel.any():
                f[sel] = field_at(
                    np.ascontiguousarray(P[sel])).astype(np.float64)
        gl = np.linalg.norm(gg, axis=1)
        ghat = gg / np.clip(gl, 1e-9, None)[:, None]
        # 0.6*res step clamp: the proven guard against max()-crease
        # kinks; |g| division handles the field's non-unit slopes
        P -= ghat * np.clip(f / np.clip(gl, 0.15, None),
                            -0.6 * res, 0.6 * res)[:, None]
        # cell box keeps one-vertex-per-cell topology (no fold-over)
        # and forbids exact coincidence on shared cell faces
        P = np.clip(P, Cmin + eps, Cmin + (res - eps))
    # quads: one per sign-changing grid edge, adjacent cells ringed
    # (s, s-eB, s-eB-eC, s-eC) — CCW seen from +A; ownership by tail
    # cell index (exact integer partition across tiles)
    tris = []
    for A in (0, 1, 2):
        Bx, Cx = (A + 1) % 3, (A + 2) % 3
        tl = tuple(slice(0, -1) if ax == A else slice(None)
                   for ax in range(3))
        hd = tuple(slice(1, None) if ax == A else slice(None)
                   for ax in range(3))
        cross = neg[tl] != neg[hd]
        si = np.stack(np.nonzero(cross)).astype(np.int64)
        if not si.shape[1]:
            continue
        ownm = ((own_lo[:, None] <= si)
                & (si < own_hi[:, None])).all(axis=0)
        si = si[:, ownm]
        if not si.shape[1]:
            continue
        eB = np.zeros(3, np.int64)
        eB[Bx] = 1
        eC = np.zeros(3, np.int64)
        eC[Cx] = 1
        s1 = si - eB[:, None]
        s2 = si - eB[:, None] - eC[:, None]
        s3 = si - eC[:, None]
        q0 = cid[si[0], si[1], si[2]]
        q1 = cid[s1[0], s1[1], s1[2]]
        q2 = cid[s2[0], s2[1], s2[2]]
        q3 = cid[s3[0], s3[1], s3[2]]
        # material at the tail -> outward normal along +A -> natural
        # (CCW-from-+A) ring; at the head -> reversed
        flip = neg[si[0], si[1], si[2]]
        ring = np.where(flip[None, :],
                        np.stack([q0, q1, q2, q3]),
                        np.stack([q0, q3, q2, q1]))
        # shorter-diagonal split: avoids fold-over slivers on
        # anisotropic quads; deterministic
        d02 = np.linalg.norm(P[ring[0]] - P[ring[2]], axis=1)
        d13 = np.linalg.norm(P[ring[1]] - P[ring[3]], axis=1)
        u02 = d02 <= d13
        t1 = np.where(u02[None, :],
                      np.stack([ring[0], ring[1], ring[2]]),
                      np.stack([ring[1], ring[2], ring[3]]))
        t2 = np.where(u02[None, :],
                      np.stack([ring[0], ring[2], ring[3]]),
                      np.stack([ring[1], ring[3], ring[0]]))
        tris.append(np.concatenate([t1.T, t2.T]))
    if not tris:
        return None
    F = np.concatenate(tris).astype(np.int64)
    used, inv = np.unique(F, return_inverse=True)
    return P[used], inv.reshape(-1, 3)


def _nonmanifold_edge_split(v, f):
    """Split mesh edges carried by FOUR triangles (SurfaceNets
    ambiguous-face duals: a checkerboard grid face rings all four of
    its edge-quads around one dual edge).  _repair_pinches cannot
    separate them — all four fans share both endpoints — so pair the
    triangles by normal coherence and give one pair duplicated
    endpoint vertices.

    Groups sharing a face are processed in separate passes: the rename
    matches ORIGINAL vertex ids against a face another group may have
    already renamed, which half-rewires the face (b renamed, a not) and
    emits incidence-1 edges no embedding can close.  A touched mask over
    all FOUR faces of each processed group defers conflicting groups to
    a recompute-and-repeat pass (a shared face renamed later would also
    strip an earlier group's kept edge to incidence 1).  The first group
    of a pass is never blocked, so every pass resolves at least one
    4-valent edge and the loop terminates; single-pass meshes come out
    bit-identical."""
    while True:
        e = np.sort(np.vstack([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]),
                    axis=1)
        eu, inv, cnt = np.unique(e, axis=0, return_inverse=True,
                                 return_counts=True)
        rows = np.nonzero(cnt[inv] == 4)[0]
        if not len(rows):
            return v, f
        rows = rows[np.argsort(inv[rows], kind="stable")]   # groups of 4
        f = f.copy()
        nf = len(f)
        fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
        fn /= np.clip(np.linalg.norm(fn, axis=1), 1e-14, None)[:, None]
        add = []
        nid = len(v)
        touched = np.zeros(nf, bool)
        deferred = False
        for g0 in range(0, len(rows), 4):
            grp = rows[g0:g0 + 4]
            fis = grp % nf
            if len(np.unique(fis)) != 4:
                continue
            if touched[fis].any():
                deferred = True
                continue
            a, b = eu[inv[grp[0]]]
            dots = fn[fis] @ fn[fis].T
            sc = (dots[0, 1] + dots[2, 3], dots[0, 2] + dots[1, 3],
                  dots[0, 3] + dots[1, 2])
            second = ((2, 3), (1, 3), (1, 2))[int(np.argmax(sc))]
            for fi in fis[list(second)]:
                tri = np.where(f[fi] == a, nid, f[fi])
                f[fi] = np.where(tri == b, nid + 1, tri)
            touched[fis] = True
            add += [v[a], v[b]]
            nid += 2
        if add:
            v = np.vstack([v, np.asarray(add)])
        if not deferred:
            return v, f


def _project_to_field(v, f, field_at, res, steps=2):
    """Newton-project vertices onto the f=0 iso-surface along their
    area-weighted normals (the field is ~unit-gradient near the surface;
    steps clamp to 0.6*res so clamp-crease kinks cannot launch
    vertices).

    SELECTIVE: only vertices whose incident face normals are coherent
    project — at crown crests the flank-averaged normal is ill-posed and
    projecting there zigzags the crest line into sawteeth.  Coherence
    |sum fn| / sum |fn| is ~1 on flats and walls (where the staircase
    lives) and drops at crests, which keep their smoothed positions.
    """
    for _ in range(steps):
        fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
        al = np.linalg.norm(fn, axis=1)
        vn = np.zeros_like(v)
        asum = np.zeros(len(v))
        for k in range(3):
            np.add.at(vn, f[:, k], fn)
            np.add.at(asum, f[:, k], al)
        ln = np.linalg.norm(vn, axis=1)
        coh = ln / np.clip(asum, 1e-14, None)
        ok = ln > 1e-14
        vn[ok] /= ln[ok, None]
        w = np.clip((coh - 0.88) / 0.07, 0.0, 1.0)
        fv = field_at(np.ascontiguousarray(v, np.float64))
        step = (np.clip(fv, -0.6 * res, 0.6 * res) * w).astype(v.dtype)
        v = v - vn * step[:, None]
    return v


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

def _column_triangle_depth(p, z, centers):
    """Depth extrema of a triangle clipped to each projected pixel square.

    Intersection vertices are clipped triangle edges or pixel corners
    inside the triangle. This also handles an exactly edge-on triangle.
    """
    low, high = centers - 0.5, centers + 0.5
    zlo = np.full(len(centers), np.inf)
    zhi = np.full(len(centers), -np.inf)
    for i in range(3):
        j = (i + 1) % 3
        edge = p[j] - p[i]
        t0, t1 = np.zeros(len(centers)), np.ones(len(centers))
        valid = np.ones(len(centers), bool)
        for axis in range(2):
            if abs(edge[axis]) < 1e-14:
                valid &= ((p[i, axis] >= low[:, axis] - 1e-12)
                          & (p[i, axis] <= high[:, axis] + 1e-12))
            else:
                a = (low[:, axis] - p[i, axis]) / edge[axis]
                b = (high[:, axis] - p[i, axis]) / edge[axis]
                t0 = np.maximum(t0, np.minimum(a, b))
                t1 = np.minimum(t1, np.maximum(a, b))
        valid &= t0 <= t1 + 1e-12
        a, b = z[i] + t0 * (z[j] - z[i]), z[i] + t1 * (z[j] - z[i])
        zlo[valid] = np.minimum(zlo[valid], np.minimum(a, b)[valid])
        zhi[valid] = np.maximum(zhi[valid], np.maximum(a, b)[valid])
    e1, e2 = p[1] - p[0], p[2] - p[0]
    det = e1[0] * e2[1] - e1[1] * e2[0]
    if abs(det) > 1e-14:
        for offset in ((-0.5, -0.5), (-0.5, 0.5), (0.5, -0.5), (0.5, 0.5)):
            q = centers + offset - p[0]
            w1 = (q[:, 0] * e2[1] - q[:, 1] * e2[0]) / det
            w2 = (e1[0] * q[:, 1] - e1[1] * q[:, 0]) / det
            inside = (w1 >= -1e-12) & (w2 >= -1e-12) & (w1 + w2 <= 1 + 1e-12)
            value = z[0] + w1 * (z[1] - z[0]) + w2 * (z[2] - z[0])
            zlo[inside] = np.minimum(zlo[inside], value[inside])
            zhi[inside] = np.maximum(zhi[inside], value[inside])
    return zlo, zhi


def _rasterize_depth_range(flat, depth, tris, cell, origin, shape2d):
    """Per-column surface depth, including overlapping edge-on sheets.

    Interpolate interior samples, then clip thin triangles and uncovered
    boundary pixels to their projected pixel squares. Using a triangle's
    entire depth range in each column would inflate the voxel band and
    make the facing gate query the wrong sheet on folded surfaces.
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
        # interpolated depth: exact per-cell, no facet terracing in the
        # band bounds
        dval = (w0 * depth[a] + w1 * depth[b] + w2 * depth[c])
        sub_lo = Dlo[i0:i1, j0:j1]
        np.copyto(sub_lo, np.minimum(sub_lo, np.where(inside, dval, 1e9)))
        sub_hi = Dhi[i0:i1, j0:j1]
        np.copyto(sub_hi, np.maximum(sub_hi, np.where(inside, dval, -1e9)))
        sub_c = covered[i0:i1, j0:j1]
        np.copyto(sub_c, sub_c | inside)
    # Each sheet contributes independently. A previously rasterized face
    # in the same column must not hide an upright lip or a folded sheet.
    # Keep the first-pass coverage fixed so fallback order cannot change
    # which of several overlapping sheets contributes its depth range.
    for a, b, c in tris:
        i0, i1, j0, j1 = _bbox(a, b, c)
        if i1 <= i0 or j1 <= j0:
            continue
        p = (np.array([flat[a], flat[b], flat[c]]) - (u0, v0)) / cell
        sides = np.roll(p, -1, axis=0) - p
        lengths2 = (sides * sides).sum(axis=1)
        area2 = abs(sides[0, 0] * (p[2, 1] - p[0, 1])
                    - sides[0, 1] * (p[2, 0] - p[0, 0]))
        thin = area2 <= np.sqrt(lengths2.max())
        win = ~covered[i0:i1, j0:j1] | thin
        if not win.any():
            continue
        # Restrict conservative fill to columns touching the projected
        # triangle. A diagonal sliver's bounding box can be much larger
        # than the sheet and must not inflate the band across empty space.
        gi, gj = np.meshgrid(np.arange(i0, i1), np.arange(j0, j1),
                             indexing="ij")
        near = np.zeros(win.shape, bool)
        for point, edge, length2 in zip(p, sides, lengths2):
            t = np.clip(((gi - point[0]) * edge[0]
                         + (gj - point[1]) * edge[1])
                        / max(length2, 1e-20), 0.0, 1.0)
            near |= ((gi - point[0] - t * edge[0]) ** 2
                     + (gj - point[1] - t * edge[1]) ** 2) <= 0.5 + 1e-12
        win &= near
        if not win.any():
            continue
        dmin, dmax = _column_triangle_depth(
            p, depth[[a, b, c]], np.column_stack([gi[win], gj[win]]))
        sub_lo = Dlo[i0:i1, j0:j1]
        sub_lo[win] = np.minimum(sub_lo[win], dmin)
        sub_hi = Dhi[i0:i1, j0:j1]
        sub_hi[win] = np.maximum(sub_hi[win], dmax)
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


def _smooth_rings(rings, win=2):
    """Circular moving-average of closed ring polylines.

    Boundary rings on steep slopes are the projected silhouette of
    tessellated facets — jagged at facet pitch.  Exact distance to a
    jagged line is still jagged, and the slope stretches every jag into
    a corduroy band across tapered crowns and end cuts; a ~1mm smoothing
    window straightens the rings while following the true boundary.
    """
    out = []
    for pts in rings:
        p = np.asarray(pts, float)
        if len(p) < 4 * win + 4 or not np.allclose(p[0], p[-1]):
            out.append(pts)
            continue
        p = p[:-1]
        acc = np.zeros_like(p)
        wsum = 0.0
        for k in range(-win, win + 1):
            w = win + 1 - abs(k)
            acc += w * np.roll(p, k, axis=0)
            wsum += w
        p = acc / wsum
        out.append(np.vstack([p, p[:1]]))
    return out


def _front_open_edges(V, F):
    """Open substrate edges whose incident triangle faces the projection."""
    edges = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    _, inverse, counts = np.unique(np.sort(edges, axis=1), axis=0,
                                    return_inverse=True, return_counts=True)
    # Edges are stacked by local edge number, each block containing ALL
    # triangles: their owners are [0..n-1, 0..n-1, 0..n-1].
    triangle = np.tile(np.arange(len(F)), 3)
    normal = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    length = np.linalg.norm(normal, axis=1)
    facing = normal[:, 2] / np.clip(length, 1e-14, None)
    return edges[(counts[inverse] == 1) & (facing[triangle] > -0.02)]


def build_rib_implicit(shape, face_ids, params, lin_defl=0.4, quality=1.0,
                       frame_cache=None, progress=None):
    """Implicit-engine entry: same contract as build_rib_meshes.

    Surface-field mapping: the substrate is the exact distance field of
    the selected faces; the pattern rides the closest-point surface
    field.  Returns ([(verts, tris)], reports) — normally ONE watertight
    solid of ribs blended into a thin sub-surface slab.  progress:
    optional (stage, done, total) reporting callback (see mesh_field).
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
    progress = _safe_progress(progress)
    if progress:
        progress("meshing selection", 0, 0)
    # the fast engine coarsens big selections (>25 faces) to keep OCCT
    # solid building affordable; the implicit substrate must NOT inherit
    # that — every field error (normal interpolation, chordal sag,
    # silhouette jaggies) scales with the deflection, and the only cost
    # of a finer substrate here is BVH query depth.  Angular deflection
    # is curvature-adaptive (it refines ONLY curved faces): 0.09 rad
    # keeps facet dihedrals ~5deg, below the threshold where the facet
    # rows stamp visible rings into rib walls on tight rolls (proven by
    # the 2x-refinement experiment; every cheaper decoupling attempt
    # left the rings in place).
    regions = region_meshes(shape, face_ids, min(lin_defl, 0.25),
                            ang_defl=0.09)
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

    if progress:
        progress("building rasters", 0, 0)
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
    # P gradient rasters for the surface-metric width correction (unit
    # magnitude inside reach; the 1e3 cliff at the reach frontier is
    # rejected by the magnitude guard at sampling time)
    Pgu, Pgv = np.gradient(P.astype(np.float32), cell)
    # rib-termination distance measured from the FINAL domain (the inset
    # boundary): the -B clamp ends every rib on it.  EXACT distance near
    # the rim, not pixel EDT: the EDT's half-cell scallops ride the taper
    # and termination coordinates, and on steep slopes they stretch into
    # visible corduroy bands across every tapered crown and end cut
    # (same defect class _exact_pattern_distance cures for the pattern).
    reach_b = max(params.taper_len, 0.0) + 2.0
    Bex = _exact_pattern_distance(
        _smooth_rings(_ring_chains(inset, max(cell, 0.6))),
        cell, origin, shape2d, reach_b)
    Bedt = (distance_transform_edt(inset_mask) * cell).astype(np.float32)
    B = np.where(inset_mask,
                 np.where(Bex <= reach_b, Bex, Bedt),
                 0.0).astype(np.float32)
    Bo = B.copy()
    # distance INTO the void (0 inside the domain): wall blades lean
    # horizontally over openings by up to height*sin(slope) — a window
    # must stay clear, so material is capped at ~a thickness of lean
    Bv = (distance_transform_edt(~inset_mask) * cell).astype(np.float32)

    # --- substrate: the selection mesh itself, in frame coordinates -----
    V = np.ascontiguousarray(np.column_stack([flat, depth]), np.float64)
    F = np.ascontiguousarray(tris, np.int64)
    N = _vertex_normals(V, F)
    # exact 2D distance raster to the OPEN substrate boundary curves,
    # for the tangent-wedge cut.  Only edges of FRONT-facing triangles
    # count: the facing gate already removes back-sheet material, and a
    # back rim projected inside front territory must not punch slab
    # holes there.  (A per-vertex graph distance fails on coarse planar
    # tessellations — a box face has no interior vertices at all.)
    ob_chains = [(V[a, :2], V[b, :2]) for a, b in _front_open_edges(V, F)]
    if ob_chains:
        # This raster also drives the positive rib's setback and taper.
        # Truncating it at 2.5mm creates a discontinuous height step when
        # margin + taper extends farther than that lookup band.
        OB = _exact_pattern_distance(ob_chains, cell, origin, shape2d,
                                     reach=max(2.5, params.margin
                                               + params.taper_len
                                               + params.height
                                               + params.thickness))
    else:
        OB = np.full(shape2d, 1e3, np.float32)

    if progress:
        progress("sampling substrate", 0, 0)
    # smoothed surface-normal raster for the stretch factor: per-point
    # pseudo-normals vary at facet pitch, and at stretch ~2+ a few
    # degrees of normal wiggle rings the rib walls like bamboo.  Sampled
    # at cell centers via the closest-point normal (from each column's
    # own mid-surface depth so the query lands on the local sheet), then
    # gaussian smoothed ~1.5mm and renormalized.
    from scipy.ndimage import gaussian_filter
    gu_ = (origin[0] + np.arange(shape2d[0]) * cell)
    gv_ = (origin[1] + np.arange(shape2d[1]) * cell)
    GU, GV = np.meshgrid(gu_, gv_, indexing="ij")
    Xg = np.column_stack([GU.ravel(), GV.ravel(),
                          ((Dlo + Dhi) * 0.5).ravel()])
    del GU, GV
    _, _, Ng = _surface_eval(Xg, V, F, N)
    del Xg
    sig = max(1.5 / cell, 2.0)
    NXr = gaussian_filter(Ng[:, 0].reshape(shape2d), sig)
    NYr = gaussian_filter(Ng[:, 1].reshape(shape2d), sig)
    NZr = gaussian_filter(Ng[:, 2].reshape(shape2d), sig)
    del Ng
    nl = np.clip(np.sqrt(NXr * NXr + NYr * NYr + NZr * NZr), 1e-6, None)
    NXr = (NXr / nl).astype(np.float32)
    NYr = (NYr / nl).astype(np.float32)
    NZr = (NZr / nl).astype(np.float32)

    # gate cap: signed exact distance to the {top-sheet facing =
    # _GATE_HI} contour — the trim station that ends every rib in a
    # rounded in-plane nose BEFORE the raw 0.09 backstop can slice it.
    # The facing must be sampled on the TOP sheet (query just above
    # each column's top surface depth: for near-vertical sheets the
    # point sits ~on the sheet, so the closest point stays on the sheet
    # the projection parameterizes): NZr's mid-band queries snap to
    # silhouette/fold-under sheets across overlap bands and read ~0
    # over legitimate 45-70deg ribs (measured -0.01 on the wrap
    # fixture, 0.28-vs-true-0.0 on the dome) — a contour of NZr would
    # amputate mid-slope material, and its cross-band gaussian smears
    # real dips away on tight rolls.  NZr serves only as the cheap
    # precheck: if even the smeared field never dips below 0.45, no
    # near-silhouette territory exists and the extra surface pass is
    # skipped.  Contour and sign both come from the same RAW field so
    # Bg is a continuous signed distance (a smoothed source would
    # detach the sign from the distance and print a C0 step at the raw
    # contour); the facet-pitch contour wobble (~0.05-0.11mm) sits
    # well inside the q_g clearance.  Exact segment distance to the
    # contour polylines, NEVER a pixel EDT (anti-scallop discipline).
    Bg = None
    if _GATE_CAP and float(NZr.min()) < 0.45:
        Xt = np.column_stack([np.repeat(gu_, shape2d[1]),
                              np.tile(gv_, shape2d[0]),
                              (Dhi + 2.0 * cell).ravel()])
        _, _, Nt = _surface_eval(Xt, V, F, N)
        NZg = Nt[:, 2].reshape(shape2d).astype(np.float32)
        del Xt, Nt
        if float(NZg.min()) < _GATE_HI:
            from skimage import measure
            chains = []
            for c in measure.find_contours(NZg, _GATE_HI):
                pts = c * cell + np.asarray(origin)  # (row,col)->(u,v)mm
                if len(pts) < 3:
                    continue
                if np.allclose(pts[0], pts[-1]):
                    # speck filter: closed rings enclosing < (3 cells)^2
                    a2 = 0.5 * abs(float(
                        (pts[:-1, 0] * pts[1:, 1]
                         - pts[:-1, 1] * pts[1:, 0]).sum()))
                    if a2 < (3.0 * cell) ** 2:
                        continue
                chains.append(pts)
            if chains:
                dg = _exact_pattern_distance(chains, cell, origin,
                                             shape2d, reach=2.5)
                Bg = np.where(NZg >= np.float32(_GATE_HI), dg,
                              -dg).astype(np.float32)
                del dg
        del NZg

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
    # lin_defl of chordal sag: the rasterized depth range follows the
    # TESSELLATED surface, which dips up to the deflection below the true
    # face — without this margin the band lid grazes rib crowns on curved
    # substrates (flat lid dents every facet)
    lowpad = params.embed + _slab_depth(params) + max(0.6, 2.0 * res) \
        + lin_defl
    highpad = params.height + max(params.fillet_root, params.fillet_top) \
        + max(0.6, 2.0 * res) + lin_defl
    # leaning-crown lid: a crown on a steep wall roots up to
    # ~height*sin(theta) away laterally (down-slope side), so the
    # column's own surface depth under-bounds it — the quantized lid
    # clips real crown material and ANY extractor terraces the crest
    # there (res/cos(theta) sawtooth on steep bands).  Max-dilate the
    # lid raster by the lean radius: inert on flats (Dhi locally
    # constant), so only columns within lean reach of real depth relief
    # pay.  Separable square footprint (superset of the disk, O(N)).
    from scipy.ndimage import maximum_filter
    r_top = min(max(params.fillet_top, 1.2 * res), 0.35 * params.thickness)
    krn = 2 * int(math.ceil((params.height + r_top) / cell)) + 1
    Dhid = maximum_filter(Dhi, size=krn)
    corr = P <= reach
    # a crown leaning FURTHER than reach lands on off-corridor columns
    # (possible once height*sin(theta) > reach): give those the tall
    # band too, but only near real relief so flat corridors do not widen
    tall = corr | (maximum_filter(corr, size=krn)
                   & (Dhid - Dhi > np.float32(0.3)))
    dlo = (Dlo - lowpad).astype(np.float32)
    dhi = np.where(tall, Dhid + highpad,
                   Dhi + max(0.6, 2.0 * res)).astype(np.float32)

    surface = SurfaceField(cell=cell, origin=origin, P=P, B=B,
                           mask=inset_mask, Bo=Bo, V=V, F=F, N=N, OB=OB,
                           Pgu=Pgu, Pgv=Pgv, NXr=NXr, NYr=NYr, NZr=NZr,
                           Bv=Bv, dlo=dlo, dhi=dhi, BV=BV, BF=BF,
                           body_tol=body_tol, ctr=ctr,
                           M=np.ascontiguousarray(
                               np.column_stack([axes[:, 0], axes[:, 1],
                                                n_axis])), Bg=Bg)
    warn = []
    # resolve the extractor ONCE and report what actually ran: the env is
    # re-read nowhere else, so a mid-build env change cannot misattribute,
    # and the SN self-heal fallback flips the reported name via the retry
    # marker mesh_field appends to this same list
    used = os.environ.get("RIBBING_EXTRACTOR", EXTRACTOR)
    # progress kwarg only when a callback exists: test spies monkeypatch
    # mesh_field with the historical signature, and reporting must never
    # be able to break a build path that does not use it
    clusters = mesh_field(surface, params, resolution=res, reports=warn,
                          extractor=used,
                          **({"progress": progress} if progress else {}))
    if any(w.startswith("surface_nets extractor failed") for w in warn):
        used = "marching_cubes"
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
        f"implicit engine: {res:.2f}mm voxels, "
        f"{used} extractor, "
        f"{time.time() - t0:.1f}s")
    return world, [rep]
