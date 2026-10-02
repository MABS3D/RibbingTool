"""The pattern's frame belongs to the surface, not its tessellation density."""
from types import SimpleNamespace

import numpy as np
import pytest

from server.geometry.ribbing import _projection_frame
from server.geometry.surface_mapping import surface_coordinates


def sheet(refinements=0):
    v = np.array([[-30., -20., 0.], [30., -20., 0.],
                  [30., 20., 0.], [-30., 20., 0.]])
    f = np.array([[0, 1, 2], [0, 2, 3]])
    for _ in range(refinements):
        vertices, triangles = list(v), []
        for tri in f:
            center = v[tri].mean(axis=0)
            if center[0] < 0 and center[1] > 0:
                new = len(vertices)
                vertices.append(center)
                triangles.extend([[tri[0], tri[1], new], [tri[1], tri[2], new],
                                  [tri[2], tri[0], new]])
            else:
                triangles.append(tri)
        v, f = np.asarray(vertices), np.asarray(triangles)
    return SimpleNamespace(vertices=v, triangles=f)


def test_uneven_refinement_preserves_surface_frame_and_lattice_phase():
    coarse, refined = sheet(), sheet(5)
    # Both cover exactly the same sheet. The dense upper-left corner must
    # not drag the pattern origin toward itself or rotate the ribs.
    center, axes = _projection_frame([coarse])
    refined_center, refined_axes = _projection_frame([refined])
    np.testing.assert_allclose(center, [0, 0, 0], atol=1e-12)
    np.testing.assert_allclose(refined_center, center, atol=1e-12)
    np.testing.assert_allclose(refined_axes, axes, atol=1e-12)
    base = surface_coordinates(coarse.vertices, coarse.triangles,
                               (coarse.vertices-center)@axes)
    dense = surface_coordinates(refined.vertices, refined.triangles,
                                (refined.vertices-refined_center)@refined_axes)
    np.testing.assert_allclose(dense[:4], base, atol=1e-12)


def test_frame_tracks_rigid_placement_without_world_origin_cancellation():
    region = sheet(4)
    center, axes = _projection_frame([region])
    angle = .43
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0.],
                         [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]])
    shift = np.array([1e7, -2e7, 3e7])
    moved = SimpleNamespace(vertices=region.vertices@rotation.T+shift,
                            triangles=region.triangles)
    moved_center, moved_axes = _projection_frame([moved])
    np.testing.assert_allclose(moved_center, center@rotation.T+shift, atol=1e-8, rtol=0)
    np.testing.assert_allclose(moved_axes, rotation@axes, atol=1e-9)


@pytest.mark.parametrize('reverse', [False, True])
def test_frame_keeps_outward_winding_and_deterministic_axis_sign(reverse):
    region = sheet(3)
    if reverse:
        region.triangles = region.triangles[:, ::-1]
    _, axes = _projection_frame([region])
    normal = np.cross(axes[:, 0], axes[:, 1])
    np.testing.assert_allclose(normal, [0, 0, -1 if reverse else 1], atol=1e-12)
    assert axes[np.argmax(np.abs(axes[:, 0])), 0] > 0
    np.testing.assert_allclose(axes.T@axes, np.eye(2), atol=1e-12)


def test_disjoint_regions_share_area_weighted_origin_independent_of_order():
    first, second = sheet(4), sheet()
    second.vertices = second.vertices*2+[200, 0, 0]
    center, axes = _projection_frame([first, second])
    reordered_center, reordered_axes = _projection_frame([second, first])
    # The second panel has four times the area of the first.
    np.testing.assert_allclose(center, [160, 0, 0], atol=1e-12)
    np.testing.assert_allclose(reordered_center, center, atol=1e-12)
    np.testing.assert_allclose(reordered_axes, axes, atol=1e-12)


def test_wide_cylinder_wrap_has_a_facing_frame_independent_of_mesh_density():
    angles = np.linspace(-np.deg2rad(130.), np.deg2rad(130.), 81)
    xy = 30*np.column_stack([np.cos(angles), np.sin(angles)])
    v = np.vstack([np.column_stack([xy, np.zeros(len(xy))]),
                   np.column_stack([xy, np.full(len(xy), 50.)])])
    i = np.arange(len(xy)-1)
    f = np.vstack([np.column_stack([i, i+1, i+1+len(xy)]),
                   np.column_stack([i, i+1+len(xy), i+len(xy)])])
    region = SimpleNamespace(vertices=v, triangles=f)
    center, axes = _projection_frame([region])
    # Spatial PCA alone picks the cylinder axis: every selected wall then
    # becomes an edge-on silhouette. The open wrap has a clear outward view.
    np.testing.assert_allclose(np.cross(axes[:, 0], axes[:, 1]), [1, 0, 0], atol=1e-12)
    np.testing.assert_allclose(axes.T@axes, np.eye(2), atol=1e-12)

    # Subdivide just one quadrant without changing its piecewise-flat surface.
    for _ in range(3):
        vertices, triangles = list(v), []
        for tri in f:
            point = v[tri].mean(axis=0)
            if point[1] > 0 and point[2] > 20:
                extra = len(vertices)
                vertices.append(point)
                triangles.extend([[tri[0], tri[1], extra], [tri[1], tri[2], extra],
                                  [tri[2], tri[0], extra]])
            else:
                triangles.append(tri)
        v, f = np.asarray(vertices), np.asarray(triangles)
    refined = SimpleNamespace(vertices=v, triangles=f)
    refined_center, refined_axes = _projection_frame([refined])
    np.testing.assert_allclose(refined_center, center, atol=1e-12)
    np.testing.assert_allclose(refined_axes, axes, atol=1e-12)


def test_closed_surface_does_not_use_cancelling_outward_normals_as_a_view():
    import manifold3d as m3d
    mesh = m3d.Manifold.cube([60., 40., 8.]).to_mesh64()
    region = SimpleNamespace(vertices=np.asarray(mesh.vert_properties[:, :3]),
                             triangles=np.asarray(mesh.tri_verts, np.int64))
    center, axes = _projection_frame([region])
    np.testing.assert_allclose(center, [30., 20., 4.], atol=1e-12)
    normal = np.cross(axes[:, 0], axes[:, 1])
    np.testing.assert_allclose(np.abs(normal), [0., 0., 1.], atol=1e-12)
    np.testing.assert_allclose(axes.T@axes, np.eye(2), atol=1e-12)
