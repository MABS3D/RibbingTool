"""Cap-to-body fade at tapered run-outs (Zone B: the end gusset).

With fillets 0/0 the capped run-out (_CAP_RUNOUT) ends the taper ramp as
a plateau at h_plat = min(4.4*res, 0.55*height) (1.32mm here) closed by
a 3x-steep hypot nose: a ~1.3mm-tall cliff standing proud of the body —
the "side bump" shoulder the user photographs at every tapered rim.  The
melt cannot come from ramping the top surface to zero (sub-voxel feather
= torn lace; the stub-band contract of test_runout_quality forbids it):
the only legal path to zero is the root fillet's body blend, solid-
backed.  The end-gusset contract asserted here:

  (m1) shoulder at the material front: the termination must descend
       into the body through a blend skirt — the front must reach the
       B-setback truncation, must not stand plateau-high within
       0.15mm of its lowest-u station, and the skirt must populate
       u in [0.28, 0.5] (bridging nose to body);
  (m2) junction dihedral across seam-zone edges — diagnostic print;
  (m3) feather honesty guard: no free rib-top sheet may fake the fade —
       every upward-facing vertex at gusset heights must be solid-
       backed (no downward-facing vertex within 0.45mm);
  (m4) no debris in the termination band, watertight output.

Calibrated on the pre-gusset engine (2026-08-27, post-round-8 fixes):
m1 measured u_end = 0.55 with front height 1.10 (material starts as a
~plateau-high wall; 0.1mm silhouette bins: nothing below u=0.5, then
0.81/1.10/1.14/1.29 — the binned-profile h_top of the spec draft read
1.29 ~ plateau 1.32 but cannot separate a cliff from the fillet arc,
hence the front-anchored form) — FAIL as designed; m2 measured 190
seam-zone edges, dihedral p50 6.2 p95 38.6 max 50.4 deg (Taubin
already spreads the raw 90deg corner over a few edges); m3 measured 0
feather pairs and m4 passed (contracts, not defects).  Post-gusset:
u_end = 0.39, front height S = 0.52 (0.1mm silhouette 0.22/0.52/1.07
from u=0.3 — the fillet arc rising from the body), m2 199 edges p95
31.6 max 39.0 deg.

Kernel-level (no OCCT): the exact harness of test_runout_quality.py —
flat plane substrate at z=10, one rib along u at v=15, B = Bo = u ramp,
CELL 0.25, RES 0.30 — with fillets 0/0 (thickness 1.6, height 4,
taper 5).  Derived: FLOOR 0.66, plateau 1.32, q_end 0.867, B-setback at
u = 0.3.  Do NOT copy the stub-window fraction metric into this
fillets-0 fixture: the gusset skirt legitimately sweeps upward-facing
material through [0.3, 1.35]*FLOOR near the nose — the root blend is
the explicitly legal path to zero; the window contract stays guarded by
the untouched fillets-2/2 module.
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

THICK = 1.6
HEIGHT = 4.0
TAPER = 5.0

FLOOR = min(2.2 * RES, 0.55 * HEIGHT)            # 0.66 mm
PLATEAU = min(4.4 * RES, 0.55 * HEIGHT)          # 1.32 mm (two-floor cap)
U_SETBACK = 1.2 * CELL                           # 0.30 mm: B-setback cut


def _params():
    return RibParams(pattern="rectangular", spacing=12.0, thickness=THICK,
                     height=HEIGHT, margin=2.0, taper_len=TAPER,
                     fillet_root=0.0, fillet_top=0.0)


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
    """One rib running down an open-boundary taper ramp, fillets 0/0."""
    g = _surface(lambda uu, vv: np.abs(vv - RIB_V))     # rib along u
    nu = g.B.shape[0]
    ramp = (np.arange(nu, dtype=np.float32) * CELL)[:, None] \
        * np.ones_like(g.B)
    g.B = ramp
    g.Bo = ramp.copy()
    clusters = mesh_field(g, _params(), resolution=RES)
    assert clusters, "field produced no geometry"
    v = np.vstack([c[0] for c in clusters])
    f = np.vstack([c[1] + off for c, off in
                   zip(clusters, np.cumsum([0] + [len(c[0]) for c in
                                                  clusters[:-1]]))])
    n = np.vstack([_vnormals(c[0], c[1]) for c in clusters])
    return {"clusters": clusters, "v": v, "f": f, "n": n,
            "h": v[:, 2] - DEPTH}


def _profile(v, h, u0=0.0, u1=6.3, binw=0.3):
    """Top profile h_top(u): max height in a thin centerline slice per
    u-bin (the runout-quality idiom)."""
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


def test_cap_shoulder_melts_into_body(runout):
    """(m1) The nose must not stand proud as a plateau-high cliff.

    Front-anchored shoulder metric (the 0.3mm binned h_top of the spec
    draft cannot separate a cliff from the steep-but-continuous fillet
    arc — the bin max always catches the arc's upper half; calibrated
    on the fine 0.1mm silhouettes of both builds):
      (i)  bridging: the material front u_end (lowest-u centerline vert
           with h > 0.15) must reach the B-setback truncation
           (u_end <= 0.45; pre-gusset 0.55 — the melt is missing);
      (ii) no cliff at the front: max h within [u_end, u_end + 0.15]
           must stay under 0.55*plateau = 0.73 (pre-gusset 1.10: the
           material STARTS as a ~plateau-high wall; post-gusset ~0.4:
           the fillet skirt rises from the body);
      (iii) skirt population: >= 5 centerline verts in u in [0.28, 0.5]
           with h > 0.05 (pre-gusset 0-2 strays, post ~6+)."""
    v, h = runout["v"], runout["h"]
    sl = np.abs(v[:, 1] - RIB_V) < 0.4
    front = sl & (h > 0.15)
    assert front.any(), "no rib material at all"
    u_end = float(v[front, 0].min())
    prof = _profile(v, h)
    assert u_end <= U_SETBACK + 0.15, (
        f"no gusset skirt: rib material starts only at u={u_end:.2f}, "
        f"never bridging down to the B-setback at u={U_SETBACK:.2f} — "
        f"the cap must melt into the body, not stop short as a "
        f"shoulder.  Top profile:\n{_fmt(prof)}")
    win = front & (v[:, 0] < u_end + 0.15)
    S = float(h[win].max())
    assert S < 0.55 * PLATEAU, (
        f"run-out front is a cliff: material within 0.15mm of its "
        f"lowest-u station u_end={u_end:.2f} already stands "
        f"h={S:.2f}mm ~ the {PLATEAU:.2f}mm plateau (limit "
        f"{0.55 * PLATEAU:.2f}).  The termination must descend through "
        f"the end-gusset blend skirt.  Top profile:\n{_fmt(prof)}")
    skirt = sl & (h > 0.05) & (v[:, 0] >= 0.28) & (v[:, 0] <= 0.5)
    assert int(skirt.sum()) >= 5, (
        f"no gusset skirt at the B-setback: only {int(skirt.sum())} "
        f"centerline verts in u [0.28, 0.5] (need >= 5) — the melt "
        f"must bridge the nose to the body.  Top profile:\n{_fmt(prof)}")


def test_junction_dihedral_diagnostic(runout):
    """(m2) p95 dihedral across seam-zone mesh edges — diagnostic print.

    Seam zone: edges with both endpoints at h < 0.6 and u in [0.2, 1.2]
    near the centerline.  Pre-gusset the nose wall meets the slab at
    ~90deg; the gusset's tangent body blend reads ~30-40deg (MC/Taubin
    noise makes the exact value build-dependent — print, no assert)."""
    v, f, h = runout["v"], runout["f"], runout["h"]
    fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    fn /= np.clip(np.linalg.norm(fn, axis=1), 1e-14, None)[:, None]
    e = np.sort(np.stack([np.concatenate([f[:, 0], f[:, 1], f[:, 2]]),
                          np.concatenate([f[:, 1], f[:, 2], f[:, 0]])],
                         axis=1), axis=1)
    fi = np.tile(np.arange(len(f)), 3)
    eu, inv, cnt = np.unique(e, axis=0, return_inverse=True,
                             return_counts=True)
    order = np.argsort(inv, kind="stable")
    two = cnt == 2
    pair = np.full((len(eu), 2), -1, np.int64)
    pos = np.zeros(len(eu), np.int64)
    for row in order:
        ei = inv[row]
        pair[ei, min(pos[ei], 1)] = fi[row]
        pos[ei] += 1
    sz = (np.abs(v[:, 1] - RIB_V) < 1.4) & (h < 0.6) \
        & (v[:, 0] >= 0.2) & (v[:, 0] <= 1.2)
    seam = two & sz[eu[:, 0]] & sz[eu[:, 1]]
    if not seam.any():
        print("m2: no seam-zone edges (no material below h=0.6 there)")
        return
    d = np.einsum("ij,ij->i", fn[pair[seam, 0]], fn[pair[seam, 1]])
    dih = np.degrees(np.arccos(np.clip(d, -1.0, 1.0)))
    print(f"m2: {int(seam.sum())} seam-zone edges, dihedral p50 "
          f"{np.percentile(dih, 50):.1f}deg p95 "
          f"{np.percentile(dih, 95):.1f}deg max {dih.max():.1f}deg")


def test_no_free_feather_sheets(runout):
    """(m3) Feather honesty guard, pre- AND post-gusset contract.

    The fade must be the solid-backed fillet skirt, never a free rib-top
    sheet ramping to zero: a free sheet has two nearby sides, so every
    upward-facing vertex at gusset heights (h in [0.2, 0.9]) must have
    NO downward-facing vertex within 0.45mm.  Pins the mechanism and
    guards future edits (measured 0 both pre- and post-gusset)."""
    from scipy.spatial import cKDTree
    v, n, h = runout["v"], runout["n"], runout["h"]
    up = (n[:, 2] > 0.5) & (h > 0.2) & (h < 0.9)
    dn = n[:, 2] < -0.5
    if not up.any() or not dn.any():
        return
    tree = cKDTree(v[dn])
    hits = tree.query_ball_point(v[up], 0.45)
    bad = int(sum(1 for hh in hits if hh))
    assert bad == 0, (
        f"{bad} upward-facing gusset-height vertices sit within 0.45mm "
        f"of a downward-facing vertex — a free feather sheet is faking "
        f"the fade (the only legal path to zero is the solid-backed "
        f"root-blend skirt)")


def test_cap_band_debris_watertight(runout):
    """(m4) No sub-few-voxel components in the termination band and the
    output is watertight (transplanted from test_runout_quality)."""
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
