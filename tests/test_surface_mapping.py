"""Surface spacing, finite UV paths and selection ownership regressions."""
from dataclasses import replace

import manifold3d as m3d
import numpy as np
import pytest

from server.geometry import graph_ribs as gr
from server.geometry.flatten import _tri_areas
from server.geometry.meshing import region_meshes
from server.geometry.ribbing import _projection_frame
from server.geometry.step_io import load_step
from server.geometry.surface_mapping import surface_coordinates, stretch, _untangle
from tests.test_graph_ribs import plate, params


def bent_patch():
    x, y = np.meshgrid(np.linspace(-20, 20, 17), np.linspace(-15, 15, 13))
    uv = np.column_stack([x.ravel(), y.ravel()])
    v = np.column_stack([np.minimum(uv[:, 0], 0), uv[:, 1], np.maximum(uv[:, 0], 0)])
    i = np.arange(len(v)).reshape(x.shape)[:-1, :-1].ravel()
    f = np.vstack([np.column_stack([i, i+1, i+18]),
                   np.column_stack([i, i+18, i+17])])
    return v, f


def test_lattice_line_coincident_with_triangle_edge_is_not_lost():
    v, f, n = plate()
    paths = gr.projected_paths(v, f, n, (-10., -10.), (10., 10.))
    assert len(paths) == 1
    p = paths[0][0]
    assert np.linalg.norm(np.diff(p, axis=0), axis=1).sum() == pytest.approx(20*2**.5)


def test_right_angle_fold_retains_surface_spacing_and_connected_paths():
    v, f = bent_patch()
    uv = surface_coordinates(v, f, v[:, :2])
    assert np.all(_tri_areas(uv, f) > 0)
    assert stretch(v, f, uv) == pytest.approx(np.ones((len(f), 2)), abs=.003)
    # Cross the fold with finite segments in the map. Segment length must
    # equal length along BOTH faces, including the edge-on vertical wall.
    center = uv.mean(0)
    n = gr.imp._vertex_normals(v, f)
    a, b = center+[-13., 0], center+[13., 0]
    paths = gr.projected_paths(v, f, n, a, b, coordinates=uv)
    assert len(paths) == 1
    points = paths[0][0]
    assert np.linalg.norm(np.diff(points, axis=0), axis=1).sum() == pytest.approx(26, abs=.04)
    assert points[:, 0].min() < -5 and points[:, 2].max() > 5


@pytest.mark.parametrize("fixture", ["cyl_patch_step", "wrap_cyl_step", "sphere_patch_step"])
def test_curved_cad_surfaces_have_positive_bounded_mapping(request, fixture):
    shape = load_step(request.getfixturevalue(fixture))
    from server.geometry.meshing import mesh_shape
    face = max((m for m in mesh_shape(shape) if not m.is_planar), key=lambda m: m.area)
    regions = region_meshes(shape, [face.face_id], .15, .075)
    r = regions[0]
    v, f = r.vertices, r.triangles
    center, axes = _projection_frame(regions)
    uv = surface_coordinates(v, f, (v-center)@axes)
    assert np.all(_tri_areas(uv, f) > 0)
    s = stretch(v, f, uv)
    assert np.quantile(s[:, 0]/s[:, 1], .99) < 1.8
    assert _tri_areas(uv, f).sum() == pytest.approx(_tri_areas(v, f).sum(), rel=1e-8)


def test_surface_cache_preserves_preview_export_map_and_tracks_frame():
    v, f = bent_patch()
    cache = {}
    uv = surface_coordinates(v, f, v[:, :2], cache)
    assert np.array_equal(uv, surface_coordinates(v.copy(), f.copy(), v[:, :2], cache))
    shifted = surface_coordinates(v, f, v[:, :2]+[7., -3.], cache)
    assert shifted == pytest.approx(uv+[7., -3.], abs=1e-6)
    assert len(cache['surface_maps']) == 2


def test_small_inverted_seed_is_repaired_before_barrier_solver():
    v, f = bent_patch()
    uv = np.column_stack([v[:, 0]+v[:, 2], v[:, 1]])
    uv[110] += [3., 3.]
    assert np.any(_tri_areas(uv, f) < 0)
    repaired = _untangle(v, f, uv)
    assert np.all(_tri_areas(repaired, f) > 0)


def test_root_blend_cannot_grow_onto_unselected_coplanar_face():
    # One half of a flat plate is selected. A large root fillet close to
    # the selection boundary used to spread onto the other half.
    v, f, n = plate()
    v[:, 0] = np.where(v[:, 0] < 0, -15, 0)
    other = v.copy(); other[:, 0] += 15
    p = replace(params(fillet_root=4.), margin=0)
    ribs, _ = gr.make_ribbons(v, f, n, [((-1.1, -12), (-1.1, 12))], p)
    body = m3d.Manifold.cube([30., 30., 8.]).translate([-15, -15, -8]).to_mesh64()
    kwargs = dict(substrate=(v, f), cull=False)
    untrimmed = gr.RibField(ribs, body.vert_properties[:, :3], body.tri_verts, p, **kwargs)
    trimmed = gr.RibField(ribs, body.vert_properties[:, :3], body.tri_verts, p,
                          excluded=(other, f), **kwargs)
    x = np.linspace(.1, 2, 30)
    q = np.column_stack([x, np.zeros(len(x)), np.full(len(x), .15)])
    assert np.any(untrimmed(q) < 0), 'fixture must expose the original leak'
    assert np.all(trimmed(q) > 0)
    assert trimmed(np.array([[-1.1, 0., .15], [-1.1, 0, -.1]])).max() < 0


def test_surface_engine_exports_one_body_and_preserves_excluded_side(box_step):
    from tests.test_ribbing import biggest_face_id
    from server.geometry.booleans import mesh_union
    shape = load_step(box_step)
    p = replace(params(fillet_root=1.2), mapping='surface')
    clusters, reports = gr.build_rib_graph(shape, [biggest_face_id(shape)], p)
    assert reports[0].lofted > 0
    joined = mesh_union(shape, clusters)
    assert len(joined) == 1
    v, f = joined[0]
    solid = m3d.Manifold(m3d.Mesh64(np.ascontiguousarray(v), np.ascontiguousarray(f, np.uint64)))
    assert solid.status() == m3d.Error.NoError
    assert len(solid.decompose()) == 1
    assert solid.volume() > 60*40*8


def test_disconnected_selection_does_not_bridge_an_excluded_groove(split_top_step):
    from server.geometry.meshing import mesh_shape
    shape = load_step(split_top_step)
    ids = [m.face_id for m in mesh_shape(shape)
           if np.allclose(m.vertices[:, 2], 10.)]
    assert len(ids) == 2
    p = replace(params(fillet_root=2.5), mapping='surface', margin=0)
    clusters, reports = gr.build_rib_graph(shape, ids, p)
    assert len(reports) == 2 and all(r.lofted for r in reports)
    gap = m3d.Manifold.cube([2., 36., 8.]).translate([39., 2., 6.1])
    for v, f in clusters:
        ribs = m3d.Manifold(m3d.Mesh64(np.ascontiguousarray(v),
                                      np.ascontiguousarray(f, np.uint64)))
        assert (ribs ^ gap).volume() < 1e-6


@pytest.mark.slow
@pytest.mark.parametrize("fixture,seed,count", [("part1_path", 65, 185),
                                                ("part2_path", 19, 158)])
def test_dashboard_mapping_does_not_collapse_lateral_cells(request, fixture, seed, count):
    from server.geometry.selection import grow_tangent
    shape = load_step(request.getfixturevalue(fixture))
    ids = grow_tangent(shape, [seed], angle_deg=20)
    assert len(ids) == count
    regions = region_meshes(shape, ids, .15, .075)
    assert len(regions) == 1
    r = regions[0]
    center, axes = _projection_frame(regions)
    matrix = np.column_stack([axes, np.cross(axes[:, 0], axes[:, 1])])
    v, f = (r.vertices-center)@matrix, r.triangles
    uv = surface_coordinates(v, f, v[:, :2])
    area = _tri_areas(v, f)
    assert np.all(_tri_areas(uv, f) > 0)
    s = stretch(v, f, uv)
    anisotropy = s[:, 0]/s[:, 1]
    order = np.argsort(anisotropy)
    median, p99 = np.interp([.5, .99], np.cumsum(area[order])/area.sum(),
                            anisotropy[order])
    assert median < 1.3 and p99 < 2.
    assert area[s[:, 1] < .1].sum()/area.sum() < .001
