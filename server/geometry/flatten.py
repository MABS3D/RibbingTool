import igl
import numpy as np


class FlattenError(Exception):
    pass


def boundary_loops(triangles):
    """All boundary loops as lists of vertex indices (mesh-winding order)."""
    return igl.boundary_loop_all(np.ascontiguousarray(triangles, dtype=np.int64))


def _tri_areas(v, f):
    a = v[f[:, 1]] - v[f[:, 0]]
    b = v[f[:, 2]] - v[f[:, 0]]
    if v.shape[1] == 3:
        return 0.5 * np.linalg.norm(np.cross(a, b), axis=1)
    return 0.5 * (a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])


def flatten(mesh):
    """LSCM-flatten a FaceMesh to 2D. Area-true scale, centered at origin.

    Raises FlattenError for closed surfaces or genus > 0.
    """
    v = np.ascontiguousarray(mesh.vertices, dtype=np.float64)
    f = np.ascontiguousarray(mesh.triangles, dtype=np.int64)
    loops = boundary_loops(f)
    if not loops:
        raise FlattenError(
            "face is a closed surface (not a topological disk) — "
            "split it in your CAD system first")
    edges = {(min(a, b), max(a, b)) for t in f for a, b in
             ((t[0], t[1]), (t[1], t[2]), (t[2], t[0]))}
    euler = len(v) - len(edges) + len(f)
    if euler != 2 - len(loops):  # chi = 2 - 2g - b must have g = 0
        raise FlattenError(
            "face topology unsupported (genus > 0) — "
            "split it in your CAD system first")

    outer = max(loops, key=len)
    p0 = outer[0]
    d = np.linalg.norm(v[outer] - v[p0], axis=1)
    p1 = outer[int(np.argmax(d))]
    b = np.array([p0, p1], dtype=np.int64)
    bc = np.array([[0.0, 0.0], [float(d.max()), 0.0]])
    uv = igl.lscm(v, f, b, bc)[0]
    if uv is None or len(uv) != len(v) or not np.isfinite(uv).all():
        raise FlattenError("LSCM flattening failed for this face")
    area2d = _tri_areas(uv, f).sum()
    if abs(area2d) < 1e-12:
        raise FlattenError("LSCM flattening collapsed (degenerate face)")
    if area2d < 0:  # keep triangle winding CCW in flat space
        uv = uv * np.array([1.0, -1.0])
        area2d = -area2d
    scale = np.sqrt(_tri_areas(v, f).sum() / area2d)
    uv = uv * scale
    return uv - uv.mean(axis=0)
