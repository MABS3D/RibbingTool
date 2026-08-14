import numpy as np
import pytest

from server.geometry.step_io import load_step
from server.geometry.meshing import mesh_shape
from server.geometry.flatten import FlattenError, flatten


def _dist_ratio(mesh, flat):
    e = np.vstack([mesh.triangles[:, [0, 1]], mesh.triangles[:, [1, 2]],
                   mesh.triangles[:, [2, 0]]])
    d3 = np.linalg.norm(mesh.vertices[e[:, 0]] - mesh.vertices[e[:, 1]], axis=1)
    d2 = np.linalg.norm(flat[e[:, 0]] - flat[e[:, 1]], axis=1)
    keep = d3 > 1e-9
    return d2[keep] / d3[keep]


def test_planar_is_isometric(box_step):
    m = max(mesh_shape(load_step(box_step)), key=lambda x: x.area)
    r = _dist_ratio(m, flatten(m))
    assert np.allclose(r, 1.0, atol=1e-6)


def test_cylinder_patch_low_distortion(cyl_patch_step):
    ms = mesh_shape(load_step(cyl_patch_step))
    m = next(x for x in ms if x.surface_kind == "cylinder")
    r = _dist_ratio(m, flatten(m))
    assert r.mean() == pytest.approx(1.0, abs=0.05)
    assert r.max() < 1.3 and r.min() > 0.7


def test_sphere_patch_flattens(sphere_patch_step):
    ms = mesh_shape(load_step(sphere_patch_step))
    m = next(x for x in ms if x.surface_kind == "sphere")
    flat = flatten(m)
    assert np.isfinite(flat).all()
    # conformal map of a modest spherical patch: distortion bounded
    r = _dist_ratio(m, flat)
    assert r.max() < 2.0 and r.min() > 0.5


def test_full_cylinder_unrolls_at_seam(full_cyl_step):
    # OCCT cuts periodic faces at the parametric seam, so a full cylinder
    # lateral face arrives as a disk and unrolls almost isometrically.
    ms = mesh_shape(load_step(full_cyl_step))
    m = next(x for x in ms if x.surface_kind == "cylinder")
    r = _dist_ratio(m, flatten(m))
    assert r.mean() == pytest.approx(1.0, abs=0.05)


def test_boundaryless_mesh_rejected():
    # A closed tetrahedron has no boundary: must be rejected.
    from server.geometry.meshing import FaceMesh
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
    f = np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [0, 3, 2]], np.int32)
    m = FaceMesh(1, v, f, np.zeros((4, 2)), False, "other", 1.0)
    with pytest.raises(FlattenError, match="closed"):
        flatten(m)
