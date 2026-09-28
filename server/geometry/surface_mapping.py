"""Surface-distance coordinates for pulling a lattice back onto a mesh.

Coordinates belong to vertices of the selected surface. We intersect each
triangle in parameter space and interpolate on that same 3D triangle; a
global UV overlap therefore never changes the sheet receiving a rib.
"""
import hashlib
from types import SimpleNamespace

import igl
import numpy as np
from scipy import sparse
from scipy.optimize import least_squares

from .flatten import FlattenError, _tri_areas, flatten


def _untangle(v, f, uv):
    """Repair small ARAP inversions before starting SLIM's flip barrier."""
    areas = _tri_areas(v, f)
    uv = uv.copy()
    for _ in range(4):
        bad = _tri_areas(uv, f) < .02*areas
        if not bad.any():
            break
        active = np.unique(f[bad])
        incident = np.isin(f, active).any(axis=1)
        ff = f[incident]
        den = np.maximum(areas[incident], 1e-6)
        local = np.full(len(v), -1)
        local[active] = np.arange(len(active))
        ids = local[ff]
        original = uv[active].copy()
        nr, nv = len(ff), len(active)*2
        mask = ids >= 0
        rows = np.broadcast_to(np.arange(nr)[:, None], ids.shape)[mask]
        cols = ids[mask]*2
        jac = sparse.coo_matrix((np.ones(2*len(rows)+nv),
            (np.r_[rows, rows, np.arange(nv)+nr],
             np.r_[cols, cols+1, np.arange(nv)])), shape=(nr+nv, nv)).tocsr()

        def residual(x):
            q = uv.copy()
            q[active] = x.reshape(-1, 2)
            return np.r_[np.minimum(_tri_areas(q, ff)/den-.03, 0),
                         .001*(x-original.ravel())]

        result = least_squares(residual, original.ravel(), jac_sparsity=jac,
                               ftol=1e-10, xtol=1e-10, gtol=1e-10,
                               max_nfev=200)
        uv[active] = result.x.reshape(-1, 2)
    if not np.isfinite(uv).all() or np.any(_tri_areas(uv, f) <= 1e-9*areas):
        raise FlattenError("surface mapping could not remove folded triangles; "
                           "split the selection into smaller surface patches")
    return uv


def stretch(v, f, uv):
    """Singular values of the surface-to-pattern Jacobian per triangle."""
    x, y = v[f[:, 1]]-v[f[:, 0]], v[f[:, 2]]-v[f[:, 0]]
    lx = np.linalg.norm(x, axis=1)
    ly = np.linalg.norm(np.cross(x, y), axis=1)/lx
    inv = np.zeros((len(f), 2, 2))
    inv[:, 0, 0], inv[:, 1, 1] = 1/lx, 1/ly
    inv[:, 0, 1] = -(x*y).sum(1)/(lx*lx*ly)
    delta = np.stack([uv[f[:, 1]]-uv[f[:, 0]],
                      uv[f[:, 2]]-uv[f[:, 0]]], axis=2)
    return np.linalg.svd(delta@inv, compute_uv=False)


def surface_coordinates(v, f, projection, cache=None):
    """Area-scaled, locally orientation-preserving map in millimetres.

    SLIM minimizes symmetric Dirichlet distortion with a positive Jacobian.
    A curved surface cannot in general be flattened isometrically. This
    balances stretch instead of collapsing steep walls into a frontal plane.
    Cache includes geometry AND frame; preview/export share the same map.
    """
    key = hashlib.sha256()
    for a in (v, f, projection):
        key.update(np.ascontiguousarray(a).tobytes())
    key = key.hexdigest()
    maps = cache.setdefault("surface_maps", {}) if cache is not None else {}
    if key in maps:
        return maps[key].copy()
    areas = _tri_areas(v, f)
    # Preserve exact scale and phase for a plane already in the frame.
    if (np.all(_tri_areas(projection, f) > 0)
            and np.max(np.abs(stretch(v, f, projection)-1)) < 1e-7):
        uv = projection.copy()
    else:
        uv = _untangle(v, f, flatten(SimpleNamespace(vertices=v, triangles=f)))
        data = igl.slim_precompute(
            np.asfortranarray(v, np.float64), np.asfortranarray(f, np.int32),
            np.asfortranarray(uv, np.float64), igl.SYMMETRIC_DIRICHLET,
            np.empty(0, np.int32), np.empty((0, 2), order="F"), 0.)
        uv = igl.slim_solve(data, 30)
        signed = _tri_areas(uv, f)
        if not np.isfinite(uv).all() or np.any(signed <= 1e-9*areas):
            raise FlattenError("surface mapping folded during refinement")
        uv *= np.sqrt(areas.sum()/signed.sum())
        # Area-weighted rigid alignment fixes the arbitrary flattening angle;
        # densely tessellated fillets must not determine lattice orientation.
        weights = np.bincount(f.ravel(), weights=np.repeat(areas/3, 3),
                              minlength=len(v))
        center = np.average(projection, axis=0, weights=weights)
        uv -= np.average(uv, axis=0, weights=weights)
        u, _, vt = np.linalg.svd((uv*weights[:, None]).T@(projection-center))
        rotation = u@np.diag([1., np.linalg.det(u@vt)])@vt
        uv = uv@rotation+center
    if len(maps) >= 16:
        maps.pop(next(iter(maps)))
    maps[key] = uv.copy()
    return uv
