"""Run-out termination quality at tapered open boundaries (defect 3).

Reproduces the ragged rim run-outs seen on real dashboard panels: with
the user's production params (thickness 1.6, height 4, taper_len 5,
fillets 2/2, margin 2) the edge taper ramps the rib height down and then
meets the height floor  min(2.2*res, 0.55*height)  in _rib_field.  From
there the rib does NOT end — it rides the floor as a ~2-voxel-tall stub
feather until the B-setback / cull finally cuts it, and marching cubes +
Taubin turn that sub-resolvable ledge into torn scraps.

The nTop-style termination contract asserted here, method-agnostically:
  (a) NO stub band: the rib top surface in the last 30% of the taper
      band must not present a population of upward-facing samples
      hugging the floor height (that population IS the stub feather);
  (b) monotone, continuous run-out: the rib ends cleanly once its
      height would fall below the resolvable minimum — material must
      not creep beyond the floor-hit station by more than a rounded
      cap (~one thickness), the top profile must be connected and
      monotone down to the termination, and the ramp must complete;
  (c) no debris: no sub-few-voxel connected components in the band.

A clean capped termination PASSES all three: verified empirically by a
pseudo-cap build (B-setback moved to the floor-hit station so the rib
ends there at resolvable height with a rounded end wall) which measures
a stub fraction of 0.000 on metric (a), while today's floor-stub design
measures ~0.46.  Do NOT "fix" this by widening the stub window or
raising the floor — the contract is the capped ending itself.

Kernel-level (no OCCT): synthetic-plane harness replicated from
tests/test_implicit_kernel.py — a flat triangulated plane substrate at
z=10 with one rib along u, and B = Bo = u (an open boundary along u=0;
the real engine builds Bo = B.copy(), so the taper ramp and the
1.2*cell B-setback ride the same distance field, exactly as on parts).
Marched at the engine's interactive resolution 0.30mm so the floor is
the user-visible 0.66mm.  Runs in seconds.
"""
import numpy as np
import pytest

from server.geometry.implicit import (SurfaceField, _vertex_normals,
                                      mesh_field)
from server.geometry.patterns import RibParams

CELL = 0.25                 # raster pitch (harness constant)
RES = 0.30                  # marching resolution: engine interactive res
DEPTH = 10.0                # plane substrate height
RIB_V = 15.0                # rib centerline: v = 15, running along u

THICK = 1.6                 # the user's production params
HEIGHT = 4.0
TAPER = 5.0

# the height floor in _rib_field tracks the MARCHING res (round-4 rule)
FLOOR = min(2.2 * RES, 0.55 * HEIGHT)            # 0.66 mm
# station where the ideal taper ramp height*Bo/taper_len reaches FLOOR:
# a clean termination ends the rib here (plus at most a rounded cap)
U_FLOOR = TAPER * FLOOR / HEIGHT                 # 0.825 mm
BAND = 0.3 * TAPER          # the last 30% of the taper band: u < 1.5


def _params():
    return RibParams(pattern="rectangular", spacing=12.0, thickness=THICK,
                     height=HEIGHT, margin=2.0, taper_len=TAPER,
                     fillet_root=2.0, fillet_top=2.0)


def _surface(pattern_dist, size=(40.0, 30.0), depth=DEPTH):
    """Flat triangulated plane substrate + rasters (harness pattern)."""
    step = 1.0
    xs = np.arange(0.0, size[0] + 1e-9, step)
    ys = np.arange(0.0, size[1] + 1e-9, step)
    uu, vv = np.meshgrid(xs, ys, indexing="ij")
    zz = np.full_like(uu, depth)
    V = np.ascontiguousarray(
        np.column_stack([uu.ravel(), vv.ravel(), zz.ravel()]), np.float64)
    nu, nv = len(xs), len(ys)
    quads = []
    for i in range(nu - 1):
        for j in range(nv - 1):
            a = i * nv + j
            b = (i + 1) * nv + j
            c = (i + 1) * nv + j + 1
            d = i * nv + j + 1
            quads.append((a, b, c))
            quads.append((a, c, d))
    F = np.asarray(quads, np.int64)
    N = _vertex_normals(V, F)
    ru = int(size[0] / CELL) + 1
    rv = int(size[1] / CELL) + 1
    gu, gv = np.meshgrid(np.arange(ru) * CELL, np.arange(rv) * CELL,
                         indexing="ij")
    P = pattern_dist(gu, gv).astype(np.float32)
    B = np.full((ru, rv), 1e6, np.float32)
    mask = np.ones((ru, rv), bool)
    dlo = np.full((ru, rv), float(depth) - 3.0, np.float32)
    dhi = np.full((ru, rv), float(depth) + 8.0, np.float32)
    Pgu, Pgv = np.gradient(P, CELL)
    return SurfaceField(cell=CELL, origin=(0.0, 0.0), P=P, B=B, mask=mask,
                        Bo=B.copy(), V=V, F=F, N=N, dlo=dlo, dhi=dhi,
                        Pgu=Pgu.astype(np.float32),
                        Pgv=Pgv.astype(np.float32),
                        NXr=np.zeros((ru, rv), np.float32),
                        NYr=np.zeros((ru, rv), np.float32),
                        NZr=np.ones((ru, rv), np.float32))


def _vnormals(v, f):
    """Area-weighted vertex normals of the output mesh."""
    p0, p1, p2 = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    fn = np.cross(p1 - p0, p2 - p0)
    N = np.zeros_like(v)
    for k in range(3):
        np.add.at(N, f[:, k], fn)
    ln = np.linalg.norm(N, axis=1)
    ok = ln > 1e-14
    N[ok] /= ln[ok, None]
    return N


@pytest.fixture(scope="module")
def runout():
    """One rib running down an open-boundary taper ramp, meshed once.

    B = Bo = u: distance to an open boundary along the u=0 raster edge
    (the real engine sets Bo = B.copy(), so this is the faithful
    open-rim configuration: the taper ramp and the B-setback cut both
    ride it).
    """
    g = _surface(lambda uu, vv: np.abs(vv - RIB_V))     # rib along u
    nu = g.B.shape[0]
    ramp = (np.arange(nu, dtype=np.float32) * CELL)[:, None] \
        * np.ones_like(g.B)
    g.B = ramp
    g.Bo = ramp.copy()
    clusters = mesh_field(g, _params(), resolution=RES)
    assert clusters, "field produced no geometry"
    v = np.vstack([c[0] for c in clusters])
    n = np.vstack([_vnormals(c[0], c[1]) for c in clusters])
    return {"clusters": clusters, "v": v, "n": n, "h": v[:, 2] - DEPTH}


def _profile(v, h, u0=0.0, u1=6.3, binw=0.3):
    """Top profile h_top(u): max height in a thin centerline slice per
    u-bin (bin width >= RES so marching-cubes vertex spacing cannot
    alias empty bins).  The max ignores fillet skirts by construction."""
    sl = np.abs(v[:, 1] - RIB_V) < 0.4
    out = []
    for a in np.arange(u0, u1 - 1e-9, binw):
        m = sl & (v[:, 0] >= a) & (v[:, 0] < a + binw) & (h > 0.2)
        out.append((a, float(h[m].max()) if m.any() else None,
                    int(m.sum())))
    return out


def _fmt(prof):
    return "\n".join(
        f"    u [{a:4.2f},{a+0.3:4.2f})  " +
        (f"h_top={t:5.2f}  n={n:3d}" if t is not None else "-- none --")
        for a, t, n in prof)


def test_runout_ends_capped_not_as_floor_stub(runout):
    """(a) NO stub band hugging the height floor.

    Top-surface samples: upward-facing (nz > 0.5) vertices in a thin
    band around the rib centerline (|v - 15| < 0.6) — crest and crown
    shoulders only, never the fillet skirts (those are excluded by the
    h > 0.3*FLOOR cut and the upward filter).  Within the last 30% of
    the taper band, the fraction of these sitting in the floor window
    [0.3*FLOOR, 1.35*FLOOR] must be small.  The window top is 1.35x
    (not 1.2x) because the fillet_root=2 smooth-min blend measurably
    rides the 0.66mm floor stub up to ~0.87mm — still far below the
    ~0.95mm the legitimate ramp already reaches at the floor-hit
    station.

    Today this measures ~0.46 (the stub feather between the B-setback
    at u=0.30 and the floor-hit at u=0.825).  A clean capped
    termination measures 0.000 (verified with a pseudo-cap build:
    B-setback moved to the floor-hit station, rib ends in a rounded
    end wall at resolvable height)."""
    v, n, h = runout["v"], runout["n"], runout["h"]
    crest = (np.abs(v[:, 1] - RIB_V) < 0.6) & (n[:, 2] > 0.5)
    den = crest & (v[:, 0] < BAND) & (h > 0.3 * FLOOR)
    assert den.sum() >= 10, \
        f"dead fixture: only {den.sum()} top samples in the run-out band"
    num = den & (h < 1.35 * FLOOR)
    frac = num.sum() / den.sum()
    lowu = (f"[{v[num, 0].min():.2f}, {v[num, 0].max():.2f}]"
            if num.any() else "none")
    assert frac < 0.12, (
        f"floor-stub band in the run-out: {num.sum()}/{den.sum()} = "
        f"{frac:.3f} of the rib top samples in the last 30% of the taper "
        f"band (u < {BAND}) hug the height floor "
        f"(h in [{0.3 * FLOOR:.2f}, {1.35 * FLOOR:.2f}]mm, at u {lowu}; "
        f"floor {FLOOR:.2f}mm = min(2.2*res, 0.55*height), floor-hit "
        f"station u* = {U_FLOOR:.2f}mm).  The rib must END with a "
        f"rounded cap at resolvable height, not ride the floor as a "
        f"stub feather.  Top profile:\n{_fmt(_profile(v, h))}")


def test_runout_profile_monotone_and_terminates_in_band(runout):
    """(b) Monotone, continuous run-out that ends where it should.

    The rib may end anywhere at-or-before the floor-hit station plus a
    rounded cap (~one thickness of forward rounding), must actually
    reach the boundary region (no gross amputation of the run-out),
    and from its termination the top profile must climb monotonically
    and connectedly to full height — no torn scraps, no feather
    re-emerging past the end."""
    v, h = runout["v"], runout["h"]
    rib = (np.abs(v[:, 1] - RIB_V) < 3.0) & (h > 0.25)
    assert rib.any(), "no rib material at all"
    end_u = float(v[rib, 0].min())
    # termination creep: nothing beyond the floor-hit station + cap
    assert end_u >= U_FLOOR - THICK, (
        f"rib material creeps to u={end_u:.2f}, more than one thickness "
        f"beyond the floor-hit station u*={U_FLOOR:.2f}")
    # the run-out must approach the boundary (no amputation)
    assert end_u <= U_FLOOR + THICK, (
        f"run-out amputated: rib material starts only at u={end_u:.2f} "
        f"(floor-hit station u*={U_FLOOR:.2f})")
    prof = _profile(v, h, u0=0.3 * np.floor(end_u / 0.3), u1=6.0)
    tops = [t for _, t, _ in prof]
    # connected: once material exists, every bin up the ramp has some
    start = next(i for i, t in enumerate(tops) if t is not None)
    missing = [prof[i][0] for i in range(start, len(tops))
               if tops[i] is None]
    assert not missing, (
        f"top profile has gaps (torn run-out) at u bins {missing}:\n"
        f"{_fmt(prof)}")
    # monotone within voxel/smoothing wiggle
    bad = [(prof[i][0], tops[i], tops[i + 1])
           for i in range(start, len(tops) - 1)
           if tops[i + 1] < tops[i] - 0.2]
    assert not bad, (
        f"top profile not monotone along the run-out (feather/scraps): "
        f"{bad}\n{_fmt(prof)}")
    # and the ramp completes to full height past the taper band
    full = (np.abs(v[:, 1] - RIB_V) < 0.4) & (v[:, 0] > TAPER) \
        & (v[:, 0] < TAPER + 1.5)
    assert full.any() and h[full].max() > 0.85 * HEIGHT, \
        "taper ramp never completes to full height"


def test_runout_band_has_no_debris(runout):
    """(c) No sub-few-voxel connected components in the run-out band,
    and the output is watertight (the stub scraps of the current design
    must not be 'fixed' into detached crumbs)."""
    import manifold3d as m3d
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components
    vox = RES ** 3
    for verts, tris in runout["clusters"]:
        man = m3d.Manifold(m3d.Mesh(
            np.ascontiguousarray(verts, np.float32),
            np.ascontiguousarray(tris, np.uint32)))
        assert not man.is_empty(), "output cluster is not watertight"
        adj = sp.csr_matrix(
            (np.ones(len(tris) * 3, np.int8),
             (np.concatenate([tris[:, 0], tris[:, 1], tris[:, 2]]),
              np.concatenate([tris[:, 1], tris[:, 2], tris[:, 0]]))),
            shape=(len(verts), len(verts)))
        _, lab = connected_components(adj, directed=False)
        for cid in np.unique(lab[tris[:, 0]]):
            fc = tris[lab[tris[:, 0]] == cid]
            if verts[np.unique(fc), 0].min() >= 2.0:
                continue                      # component not in the band
            vol = abs(np.einsum(
                "ij,ij->i", verts[fc[:, 0]],
                np.cross(verts[fc[:, 1]], verts[fc[:, 2]])).sum() / 6.0)
            assert vol >= 10.0 * vox and len(fc) >= 24, (
                f"debris in the run-out band: component of {len(fc)} "
                f"faces, {vol:.3f}mm^3 (< 10 voxels)")
