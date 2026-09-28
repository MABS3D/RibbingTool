"""Validate geometry at the precision actually stored in binary STL."""
import manifold3d as m3d
import numpy as np

from .booleans import BooleanError


def _closed_coordinates(v, f):
    """STL has no vertex IDs: closure must survive coordinate welding."""
    if not len(f) or not np.isfinite(v).all():
        return False
    p = v[f]
    if np.any(np.linalg.norm(np.cross(p[:, 1]-p[:, 0], p[:, 2]-p[:, 0]), axis=1) <= 1e-14):
        return False
    _, inverse = np.unique(v, axis=0, return_inverse=True)
    g = inverse[f]
    edges = np.vstack([g[:, [0, 1]], g[:, [1, 2]], g[:, [2, 0]]])
    keys, counts = np.unique(np.sort(edges, axis=1), axis=0, return_counts=True)
    if np.any(counts != 2):
        return False
    # Each undirected edge must be traversed once in either direction.
    directed = np.unique(edges, axis=0)
    return len(directed) == 2*len(keys)


def stl_export_mesh(vertices, triangles):
    """Resolve precision collapses without deleting individual boundary faces.

    The CAD and cached double-precision meshes remain untouched. If the
    float32 representation cannot retain closed components and volume, fail
    explicitly instead of writing an open or empty file.
    """
    v, f = np.asarray(vertices, np.float64), np.asarray(triangles, np.int64)
    with np.errstate(over='ignore', invalid='ignore'):
        q = v.astype(np.float32).astype(np.float64)
    if not np.isfinite(q).all() or not len(f):
        raise BooleanError('STL cannot represent empty or non-finite geometry')
    if _closed_coordinates(q, f):
        return q, f

    def solid(points, faces):
        return m3d.Manifold(m3d.Mesh64(np.ascontiguousarray(points),
                                      np.ascontiguousarray(faces, np.uint64)))

    original = solid(v, f)
    if original.is_empty() or original.status() != m3d.Error.NoError:
        raise BooleanError('STL source mesh is not a closed manifold solid')
    components = len(original.decompose())
    p = v[f]-v.mean(axis=0)
    volume = float(np.einsum('ij,ij->i', p[:, 0],
                            np.cross(p[:, 1], p[:, 2])).sum()/6)
    # One representable coordinate increment; never a modelling-scale
    # decimation tolerance. The constructor repairs collapsed edge topology.
    tolerance = float(np.max(np.abs(np.spacing(q.astype(np.float32)))))
    for _ in range(3):
        rebuilt = solid(q, f).set_tolerance(tolerance)
        if (rebuilt.is_empty() or rebuilt.status() != m3d.Error.NoError
                or len(rebuilt.decompose()) != components):
            break
        mesh = rebuilt.to_mesh64()
        q = np.asarray(mesh.vert_properties[:, :3]).astype(np.float32).astype(np.float64)
        f = np.asarray(mesh.tri_verts, np.int64)
        p = q[f]-q.mean(axis=0)
        new_volume = float(np.einsum('ij,ij->i', p[:, 0],
                                    np.cross(p[:, 1], p[:, 2])).sum()/6)
        if volume <= 0 or abs(new_volume-volume) > abs(volume)*1e-3:
            break
        if _closed_coordinates(q, f):
            return q, f
    raise BooleanError('STL precision would open or remove geometry; export aborted')
