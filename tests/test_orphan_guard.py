"""Orphan-component guard (S-fold blob regression, no OCCT).

The leaning-crown band lid evaluates field regions the per-column band
used to clip.  On the user's dashboard S-fold that exposed a
disconnected 38mm3 curled crown: rib material standing on an isolated
near-silhouette sliver of a separate flank sheet, its slab starved by
the open-boundary cut, its only roots embed grazes — a floating curl
over columns whose own surface is tens of mm away.  mesh_field must
drop such unrooted components (no slab-floor material) while keeping
every legitimately rooted island of a multi-island selection.

Fixture: flat plane substrate at z=0 plus a separate 2.2mm-wide
78-degree sliver strip hovering at z~18-20 (nz ~ 0.21, above the facing
gate), one rib family crossing both.  The strip sits entirely within
1.2mm of its open boundary, so the OB cut removes its slab exactly as
on the real part; its rib curls into a disconnected blob.
"""
import math

import numpy as np
import pytest

from server.geometry import implicit as imp
from server.geometry.implicit import (SurfaceField, _exact_pattern_distance,
                                      _vertex_normals, mesh_field)
from server.geometry.patterns import RibParams

CELL = 0.25
RES = 0.30


def _params(**kw):
    base = dict(pattern="rectangular", spacing=10, thickness=2.0, height=4.0,
                margin=0, taper_len=0, fillet_root=0.0, fillet_top=0.0)
    base.update(kw)
    return RibParams(**base)


def _grid_patch(us, vs, zfun):
    uu, vv = np.meshgrid(us, vs, indexing="ij")
    zz = zfun(uu, vv)
    V = np.column_stack([uu.ravel(), vv.ravel(), zz.ravel()])
    nu, nv = len(us), len(vs)
    quads = []
    for i in range(nu - 1):
        for j in range(nv - 1):
            a = i * nv + j
            b = (i + 1) * nv + j
            c = (i + 1) * nv + j + 1
            d = i * nv + j + 1
            quads.append((a, b, c))
            quads.append((a, c, d))
    return V, np.asarray(quads, np.int64)


def _open_boundary_raster(V, F, shape2d, origin):
    # engine-style OB: exact 2D distance to open (count-1) edges of
    # front-facing triangles
    ea = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    eas = np.sort(ea, axis=1)
    _, einv, ecnt = np.unique(eas, axis=0, return_inverse=True,
                              return_counts=True)
    chains = [(V[a, :2], V[b, :2]) for a, b in ea[ecnt[einv] == 1]]
    return _exact_pattern_distance(chains, CELL, origin, shape2d, reach=2.5)


def _orphan_surface():
    """Plane at z=0 + hovering near-silhouette sliver, one substrate."""
    V1, F1 = _grid_patch(np.arange(0.0, 40.0 + 1e-9, 1.0),
                         np.arange(0.0, 30.0 + 1e-9, 1.0),
                         lambda uu, vv: np.zeros_like(uu))
    # sliver strip: normal n = (-sin78, 0, cos78), long in v, 2.2mm wide
    th = math.radians(78.0)
    n = np.array([-math.sin(th), 0.0, math.cos(th)])
    # across-strip direction chosen so the (a,b,c) winding below yields
    # triangle normals along +n (front-facing, nz = cos78 ~ 0.21)
    t = np.array([math.cos(th), 0.0, math.sin(th)])
    ws = np.linspace(-1.1, 1.1, 5)
    vs = np.arange(5.0, 25.0 + 1e-9, 1.0)
    ctr = np.array([18.0, 0.0, 19.0])
    WW, VV = np.meshgrid(ws, vs, indexing="ij")
    P2 = (ctr[None, None, :] + WW[:, :, None] * t[None, None, :]
          + VV[:, :, None] * np.array([0.0, 1.0, 0.0])[None, None, :])
    V2 = P2.reshape(-1, 3)
    nu, nv = len(ws), len(vs)
    quads = []
    for i in range(nu - 1):
        for j in range(nv - 1):
            a = i * nv + j
            b = (i + 1) * nv + j
            c = (i + 1) * nv + j + 1
            d = i * nv + j + 1
            quads.append((a, b, c))
            quads.append((a, c, d))
    F2 = np.asarray(quads, np.int64)
    V = np.ascontiguousarray(np.vstack([V1, V2]), np.float64)
    F = np.ascontiguousarray(np.vstack([F1, F2 + len(V1)]), np.int64)
    N = _vertex_normals(V, F)
    ru = int(40.0 / CELL) + 1
    rv = int(30.0 / CELL) + 1
    gu, gv = np.meshgrid(np.arange(ru) * CELL, np.arange(rv) * CELL,
                         indexing="ij")
    P = np.abs(gv - 15.0).astype(np.float32)      # one rib along u
    B = np.full((ru, rv), 1e6, np.float32)
    OB = _open_boundary_raster(V, F, (ru, rv), (0.0, 0.0))
    return SurfaceField(cell=CELL, origin=(0.0, 0.0), P=P, B=B,
                        mask=np.ones((ru, rv), bool), Bo=B.copy(),
                        V=V, F=F, N=N, OB=OB,
                        dlo=np.full((ru, rv), -3.0, np.float32),
                        dhi=np.full((ru, rv), 26.0, np.float32))


def _components(v, f):
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components
    adj = sp.csr_matrix((np.ones(len(f) * 3, np.int8),
                         (np.concatenate([f[:, 0], f[:, 1], f[:, 2]]),
                          np.concatenate([f[:, 1], f[:, 2], f[:, 0]]))),
                        shape=(len(v), len(v)))
    _, lab = connected_components(adj, directed=False)
    return np.unique(lab[f[:, 0]]), lab


def test_orphan_curl_dropped_but_rooted_mesh_kept(monkeypatch):
    # anti-vacuity: capture the welded pre-filter mesh — the fixture
    # must genuinely grow the disconnected curl for the guard to kill
    pre = {}
    orig = imp._weld

    def spy(verts, faces, tol=1e-5):
        out = orig(verts, faces, tol)
        pre["v"], pre["f"] = out
        return out

    monkeypatch.setattr(imp, "_weld", spy)
    warn = []
    clusters = mesh_field(_orphan_surface(), _params(), resolution=RES,
                          reports=warn, extractor="marching_cubes")
    labs, lab = _components(pre["v"], pre["f"])
    curl = ((pre["v"][:, 2] > 14.0) & (pre["v"][:, 0] > 10.0)
            & (pre["v"][:, 0] < 22.0))
    assert len(labs) >= 2 and curl.sum() > 100, \
        "fixture no longer grows the orphan curl — test is vacuous"
    # the extracted result: ONE component, curl gone, plane ribs intact
    assert len(clusters) == 1
    v, f = clusters[0]
    labs2, _ = _components(v, f)
    assert len(labs2) == 1, f"{len(labs2)} components shipped"
    assert not ((v[:, 2] > 14.0) & (v[:, 0] > 10.0)
                & (v[:, 0] < 22.0)).any(), "orphan curl survived"
    assert (v[:, 2] > 2.0).any(), "plane ribs vanished with the curl"
    assert any(w.startswith("dropped") for w in warn), warn


def test_multi_island_components_all_kept():
    # two disjoint plane islands: legitimately TWO components, each
    # rooted by its own slab floor — the guard must keep both
    V1, F1 = _grid_patch(np.arange(0.0, 18.0 + 1e-9, 1.0),
                         np.arange(0.0, 30.0 + 1e-9, 1.0),
                         lambda uu, vv: np.zeros_like(uu))
    V2, F2 = _grid_patch(np.arange(22.0, 40.0 + 1e-9, 1.0),
                         np.arange(0.0, 30.0 + 1e-9, 1.0),
                         lambda uu, vv: np.zeros_like(uu))
    V = np.ascontiguousarray(np.vstack([V1, V2]), np.float64)
    F = np.ascontiguousarray(np.vstack([F1, F2 + len(V1)]), np.int64)
    N = _vertex_normals(V, F)
    ru = int(40.0 / CELL) + 1
    rv = int(30.0 / CELL) + 1
    gu, gv = np.meshgrid(np.arange(ru) * CELL, np.arange(rv) * CELL,
                         indexing="ij")
    mask = (gu <= 18.0) | (gu >= 22.0)
    P = np.abs(gv - 15.0).astype(np.float32)
    B = np.where(mask, 1e6, 0.0).astype(np.float32)
    g = SurfaceField(cell=CELL, origin=(0.0, 0.0), P=P, B=B, mask=mask,
                     Bo=np.full((ru, rv), 1e6, np.float32), V=V, F=F, N=N,
                     dlo=np.full((ru, rv), -3.0, np.float32),
                     dhi=np.full((ru, rv), 9.0, np.float32))
    warn = []
    clusters = mesh_field(g, _params(), resolution=RES, reports=warn,
                          extractor="marching_cubes")
    assert len(clusters) == 1
    v, f = clusters[0]
    labs, _ = _components(v, f)
    assert len(labs) == 2, \
        f"multi-island build shipped {len(labs)} components (want 2)"
    assert not any(w.startswith("dropped") for w in warn), warn
