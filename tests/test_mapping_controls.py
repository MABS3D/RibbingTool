"""Prescribed design fields: locality, safe mapping, shared curves and solids."""
from dataclasses import asdict, replace

import manifold3d as m3d
import numpy as np
import pytest

from server.geometry import graph_ribs as gr
from server.geometry.flatten import _tri_areas
from server.geometry.local_mapping import (MappingControlError, validate_controls,
                                          refine_control_mesh, control_fields)
from server.geometry.meshing import mesh_shape
from server.geometry.patterns import RibParams
from server.geometry.ribbing import RibbingError
from server.geometry.step_io import load_step
from tests.test_graph_ribs import plate


def control(**kwargs):
    return dict(dict(id='a', face_id=1, position=[0., 0., 0.], radius=10.,
                     angle_deg=0., spacing_scale=1., height_scale=1.), **kwargs)


def refined(controls):
    v, f, n = plate()
    return refine_control_mesh(v, f, v[:, :2].copy(), n, np.ones(len(f), int), controls)


@pytest.mark.parametrize('edit', [dict(radius=0), dict(angle_deg=91),
                                  dict(spacing_scale=.1), dict(height_scale=4),
                                  dict(position=[float('nan'), 0, 0]),
                                  dict(radius=float('inf')), dict(face_id=1.5)])
def test_invalid_controls_fail_before_mapping(edit):
    with pytest.raises(MappingControlError):
        validate_controls([control(**edit)], [1])


def test_controls_round_trip_without_mutable_defaults():
    p = RibParams(mapping='surface', mapping_controls=[control(height_scale=1.7)])
    assert RibParams.from_dict(asdict(p)) == p
    a, b = RibParams(), RibParams()
    a.mapping_controls.append(control())
    assert not b.mapping_controls
    with pytest.raises(MappingControlError, match='unselected'):
        validate_controls([control(face_id=2)], [1])


def test_control_refinement_preserves_base_map_and_shared_edges():
    v, f, uv, n, face = refined([control()])
    assert len(v) > 4  # a control inside a two-triangle CAD plane cannot vanish
    assert np.all(_tri_areas(uv, f) > 0)
    assert _tri_areas(v, f).sum() == pytest.approx(900.)
    assert uv == pytest.approx(v[:, :2])
    edges, counts = np.unique(np.sort(f[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1),
                              axis=0, return_counts=True)
    assert np.all(counts <= 2)
    boundary = v[edges[counts == 1]].mean(1)
    assert np.all(np.isclose(np.abs(boundary).max(1), 15.))


@pytest.mark.parametrize('spacing,angle', [(1., 35.), (.5, 0.), (2., 0.), (.7, 20.)])
def test_local_warp_rotates_scales_without_changing_outside_support(spacing, angle):
    controls = [control(spacing_scale=spacing, angle_deg=angle, height_scale=2.)]
    v, f, uv, _, face = refined(controls)
    mapped, heights, projected = control_fields(v, f, uv, face, controls)
    far = np.linalg.norm(v, axis=1) >= 10
    assert mapped[far] == pytest.approx(uv[far], abs=1e-12)
    assert heights[far] == pytest.approx(1.)
    assert np.min(_tri_areas(mapped, f)/_tri_areas(uv, f)) >= .08
    center = np.argmin(np.linalg.norm(v, axis=1))
    assert heights[center] == pytest.approx(2.)
    assert projected[0]['position'] == pytest.approx([0, 0, 0])
    assert np.max(np.linalg.norm(mapped-uv, axis=1)) > .1
    near = np.flatnonzero((v[:, 0] > 0) & np.isclose(v[:, 1], 0))
    probe = near[np.argmin(v[near, 0])]
    turn = np.degrees(np.arctan2(mapped[probe, 1], mapped[probe, 0]))
    assert turn == pytest.approx(-angle, abs=3.)
    physical_scale = np.linalg.norm(uv[probe])/np.linalg.norm(mapped[probe])
    assert physical_scale == pytest.approx(spacing, rel=.06)


def test_influence_cannot_jump_to_close_disconnected_sheet():
    controls = [control(radius=12, height_scale=2, angle_deg=20)]
    v, f, uv, _, face = refined(controls)
    count = len(v)
    v2 = np.vstack([v, v+[0, 0, .3]])
    f2 = np.vstack([f, f+count])
    uv2 = np.vstack([uv, uv])
    face2 = np.r_[face, np.full(len(f), 2)]
    mapped, height, _ = control_fields(v2, f2, uv2, face2, controls)
    assert np.max(height[:count]) > 1.9
    assert height[count:] == pytest.approx(1.)
    assert mapped[count:] == pytest.approx(uv)


def test_overlapping_edits_are_order_independent_and_fold_guard_is_explicit():
    controls = [control(position=[-3, 0, 0], angle_deg=20, height_scale=2),
                control(id='b', position=[3, 0, 0], spacing_scale=.8, height_scale=.5)]
    v, f, uv, _, face = refined(controls)
    a = control_fields(v, f, uv, face, controls)
    b = control_fields(v, f, uv, face, controls[::-1])
    assert a[0] == pytest.approx(b[0], abs=1e-12)
    assert a[1] == pytest.approx(b[1], abs=1e-12)
    # A strongly anisotropic base map can make a circular physical influence
    # incompatible with a UV rotation. It must be refused before solid work.
    squeezed = uv*np.array([20., .05])
    with pytest.raises(MappingControlError, match='fold|compress'):
        control_fields(v, f, squeezed, face, [control(angle_deg=90)])


def top_selection(shape):
    top = max(mesh_shape(shape), key=lambda m: m.vertices[:, 2].mean())
    center = top.vertices.mean(0).tolist()
    return top.face_id, center


def test_preview_uses_actual_trimmed_tapered_paths_and_has_no_solid_work(box_step, monkeypatch):
    shape = load_step(box_step)
    fid, center = top_selection(shape)
    p = RibParams(mapping='surface', taper_len=3, margin=2, border=True,
                  mapping_controls=[control(face_id=fid, position=center, radius=18,
                                            height_scale=1.8, angle_deg=18)])
    cache = {}
    graph = gr.prepare_rib_graph(shape, [fid], p, frame_cache=cache)
    monkeypatch.setattr(gr.imp, 'mesh_field', lambda *a, **kw: pytest.fail('preview must not extract solids'))
    preview = gr.preview_rib_graph(shape, [fid], p, frame_cache=cache)
    assert len(preview['paths']) == len(graph['paths']) > 0
    for path, visible in zip(graph['paths'], preview['paths']):
        points = np.asarray(visible['points']).reshape(-1, 3)
        top = np.asarray(visible['top']).reshape(-1, 3)
        assert points == pytest.approx(path['points']@graph['matrix'].T+graph['center'])
        assert np.linalg.norm(top-points, axis=1) == pytest.approx(path['height'])
    assert preview['controls'][0]['position'] == pytest.approx(center)
    assert preview['stats']['mapping_seconds'] >= 0
    second = gr.preview_rib_graph(shape, [fid], RibParams.from_dict(asdict(p)), frame_cache=cache)
    assert second['paths'] == preview['paths']


def test_non_surface_or_random_controls_are_rejected(box_step):
    shape = load_step(box_step)
    fid, center = top_selection(shape)
    p = RibParams(mapping_controls=[control(face_id=fid, position=center)])
    for bad in [replace(p, mapping='project'), replace(p, mapping='surface', pattern='stochastic')]:
        with pytest.raises(RibbingError):
            gr.preview_rib_graph(shape, [fid], bad)


def test_actual_controlled_solid_exceeds_base_height_and_stays_attached(box_step):
    from server.geometry.booleans import mesh_union
    shape = load_step(box_step)
    fid, center = top_selection(shape)
    p = RibParams(mapping='surface', spacing=16, height=3, thickness=2, taper_len=0,
                  fillet_root=.8, fillet_junction=.5,
                  mapping_controls=[control(face_id=fid, position=center, radius=20,
                                            height_scale=2, angle_deg=15, spacing_scale=.8)])
    clusters, reports = gr.build_rib_graph(shape, [fid], p, quality=.7)
    maximum = max(v[:, 2].max() for v, _ in clusters)
    assert maximum > 8+p.height*1.85  # the previous global cap would clip at 11 mm
    assert maximum <= 8+p.height*2+.15
    joined = mesh_union(shape, clusters)
    assert len(joined) == 1
    v, f = joined[0]
    solid = m3d.Manifold(m3d.Mesh64(np.ascontiguousarray(v), np.ascontiguousarray(f, np.uint64)))
    assert solid.status() == m3d.Error.NoError
    assert len(solid.decompose()) == 1
    assert solid.volume() > 60*40*8


def test_geodesic_support_cannot_jump_across_connected_hairpin():
    # The return wall is only 0.5 mm away in space, but >50 mm away along
    # the selected surface. A Euclidean brush would incorrectly edit both.
    arc = np.arange(0, 61, .5)
    yy, ss = np.meshgrid(np.arange(-2., 3.), arc)
    flat_s = ss.ravel()
    x = np.where(flat_s <= 30, flat_s, np.where(flat_s <= 30.5, 30, 60.5-flat_s))
    z = np.clip(flat_s-30, 0, .5)
    v = np.column_stack([x, yy.ravel(), z])
    uv = np.column_stack([flat_s, yy.ravel()])
    idx = np.arange(len(v)).reshape(ss.shape)[:-1, :-1].ravel()
    f = np.vstack([np.column_stack([idx, idx+5, idx+6]),
                   np.column_stack([idx, idx+6, idx+1])])
    edit = control(position=[3., 0, 0], radius=8., height_scale=2, angle_deg=15)
    mapped, height, _ = control_fields(v, f, uv, np.ones(len(f), int), [edit])
    returning = z > .49
    assert height[returning] == pytest.approx(1.)
    assert mapped[returning] == pytest.approx(uv[returning])
    assert height.max() == pytest.approx(2.)


def test_intrinsic_distance_is_smooth_across_long_thin_cad_triangles():
    from server.geometry.local_mapping import _surface_distances
    # Deliberately poor CAD triangulation, not poor surface mapping: an
    # edge-only shortest path produces large transverse distance gradients.
    x, y = np.meshgrid(np.linspace(-20, 20, 9), np.linspace(-.1, .1, 5))
    v = np.column_stack([x.ravel(), y.ravel(), np.zeros(x.size)])
    i = np.arange(len(v)).reshape(x.shape)[:-1, :-1].ravel()
    f = np.vstack([np.column_stack([i, i+1, i+10]),
                   np.column_stack([i, i+10, i+9])])
    source_face = 10
    point = v[f[source_face]].mean(0)
    distance = _surface_distances(v, f, source_face, point, 50.)
    assert distance == pytest.approx(np.linalg.norm(v-point, axis=1), abs=1e-7)


def test_adaptive_field_resolves_skinny_triangles_without_clamping_values():
    from server.geometry.local_mapping import apply_local_controls
    x, y = np.meshgrid(np.linspace(-20, 20, 5), [-.2, 0, .2])
    v = np.column_stack([x.ravel(), y.ravel(), np.zeros(x.size)])
    i = np.arange(len(v)).reshape(x.shape)[:-1, :-1].ravel()
    f = np.vstack([np.column_stack([i, i+1, i+6]),
                   np.column_stack([i, i+6, i+5])])
    uv, n, face = v[:, :2].copy(), np.tile([0., 0, 1], (len(v), 1)), np.ones(len(f), int)
    controls = [control(position=[-3., 0, 0], radius=15, angle_deg=25, spacing_scale=.85)]
    vv, ff, base, nn, tf = refine_control_mesh(v, f, uv, n, face, controls)
    with pytest.raises(MappingControlError):
        control_fields(vv, ff, base, tf, controls)
    v2, f2, mapped, _, _, projected = apply_local_controls(v, f, uv, n, face, controls)
    assert len(f2) > len(ff)
    assert np.all(_tri_areas(mapped, f2) > 0)
    assert projected[0]['angle_deg'] == 25
    assert projected[0]['spacing_scale'] == .85


def test_support_cache_reuses_geodesics_when_editing_fields(monkeypatch):
    from server.geometry import local_mapping as lm
    v, f, n = plate()
    face, uv = np.ones(len(f), int), v[:, :2].copy()
    a = [control(angle_deg=20, height_scale=1.4)]
    cache = {}
    first = lm.apply_local_controls(v, f, uv, n, face, a, cache)
    monkeypatch.setattr(lm, '_surface_distances', lambda *args: pytest.fail('unchanged support must reuse geodesics'))
    b = [control(angle_deg=10, height_scale=1.8)]
    second = lm.apply_local_controls(v, f, uv, n, face, b, cache)
    assert len(cache['local_control_supports']) == 1
    assert second[4].max() > first[4].max()
    assert not np.allclose(first[2], second[2])


def test_preview_rejects_a_margin_that_leaves_no_curves(box_step):
    shape = load_step(box_step)
    fid, _ = top_selection(shape)
    with pytest.raises(RibbingError, match='leave no ribs'):
        gr.preview_rib_graph(shape, [fid], RibParams(mapping='surface', margin=100))


def test_neutral_control_preserves_paths_and_does_not_dilute_another_field(box_step):
    from server.geometry.local_mapping import apply_local_controls
    v, f, n = plate()
    uv, face = v[:, :2].copy(), np.ones(len(f), int)
    neutral = [control()]
    same = apply_local_controls(v, f, uv, n, face, neutral)
    assert np.array_equal(same[0], v) and np.array_equal(same[1], f)
    assert np.array_equal(same[2], uv)
    active = [control(id='active', height_scale=2., angle_deg=20.)]
    a = apply_local_controls(v, f, uv, n, face, active)
    b = apply_local_controls(v, f, uv, n, face, active+neutral)
    assert np.array_equal(a[1], b[1])
    assert a[2] == pytest.approx(b[2])
    assert a[4] == pytest.approx(b[4])
    shape = load_step(box_step)
    fid, center = top_selection(shape)
    baseline = gr.preview_rib_graph(shape, [fid], RibParams(mapping='surface'))
    edited = gr.preview_rib_graph(shape, [fid], RibParams(mapping='surface',
                 mapping_controls=[control(face_id=fid, position=center)]))
    assert edited['paths'] == baseline['paths']


def test_small_radius_is_centered_on_projected_surface_before_refinement():
    from server.geometry.local_mapping import apply_local_controls
    v, f, n = plate()
    tiny = [control(position=[0, 0, .3], radius=.15, height_scale=2.)]
    vv, ff, _, _, height, projected = apply_local_controls(v, f, v[:, :2].copy(), n,
                                                          np.ones(len(f), int), tiny)
    assert projected[0]['position'] == pytest.approx([0, 0, 0])
    assert np.any(np.linalg.norm(vv, axis=1) < .03)
    assert height.max() > 1.9


def _cache_history_strip():
    x, y = np.meshgrid(np.linspace(-20, 20, 5), [-.2, 0, .2])
    v = np.column_stack([x.ravel(), y.ravel(), np.zeros(x.size)])
    i = np.arange(len(v)).reshape(x.shape)[:-1, :-1].ravel()
    f = np.vstack([np.column_stack([i, i+1, i+6]), np.column_stack([i, i+6, i+5])])
    return v, f, v[:, :2].copy(), np.tile([0., 0, 1], (len(v), 1)), np.ones(len(f), int)


def test_cache_history_cannot_change_mapping_geometry_or_acceptance():
    from server.geometry.local_mapping import apply_local_controls
    base = _cache_history_strip()
    edit = control(position=[-3., 0, 0], radius=15., angle_deg=25., spacing_scale=.85, height_scale=1.5)
    cold25 = apply_local_controls(*base, [edit], {})
    cold20 = apply_local_controls(*base, [dict(edit, angle_deg=20.)], {})
    cache = {}
    apply_local_controls(*base, [dict(edit, angle_deg=20.)], cache)
    after20 = apply_local_controls(*base, [edit], cache)
    back20 = apply_local_controls(*base, [dict(edit, angle_deg=20.)], cache)
    for cold, warm in [(cold25, after20), (cold20, back20)]:
        for a, b in zip(cold[:5], warm[:5]):
            assert np.array_equal(a, b)


def test_refined_fields_match_an_independent_exact_geodesic_solve():
    from server.geometry.local_mapping import apply_local_controls
    base = _cache_history_strip()
    edit = control(position=[-3., 0, 0], radius=15., angle_deg=25., spacing_scale=.85, height_scale=1.5)
    v, f, mapped, n, height, _ = apply_local_controls(*base, [edit], {})
    # For this planar support, the undeformed UV is the physical XY. A fresh
    # native exact solve on the returned mesh must reproduce every field.
    exact_uv, exact_height, _ = control_fields(v, f, v[:, :2], np.ones(len(f), int), [edit])
    assert np.allclose(mapped, exact_uv, atol=1e-10, rtol=0)
    assert np.allclose(height, exact_height, atol=1e-10, rtol=0)


def test_refinement_and_fields_are_invariant_to_control_order():
    from server.geometry.local_mapping import apply_local_controls
    v, f, n = plate()
    base = (v, f, v[:, :2].copy(), n, np.ones(len(f), int))
    edits = [control(id='a', position=[-3., 0, 0], radius=12., angle_deg=18., height_scale=1.8),
             control(id='b', position=[3., 0, 0], radius=12., angle_deg=-12., spacing_scale=.8)]
    a = apply_local_controls(*base, edits, {})
    b = apply_local_controls(*base, edits[::-1], {})
    for x, y in zip(a[:5], b[:5]):
        assert np.array_equal(x, y)


def test_intrinsic_support_cannot_cross_a_corner_touching_triangle_fan():
    from server.geometry.local_mapping import _surface_distances
    # The crop contains two nearby sheets sharing only a remote corner. A
    # vertex-connected crop passes both to the native solver, which can
    # report zero for unreachable fan vertices and activate the wrong wall.
    v = np.array([[0., -.5, 0], [3., 0, 0], [0., .5, 0],
                  [0., -.5, .3], [0., .5, .3]])
    f = np.array([[0, 1, 2], [3, 1, 4]])
    point = np.array([.3, 0, 0])
    distances = _surface_distances(v, f, 0, point, 1.5)
    assert distances[[0, 2]] == pytest.approx(np.linalg.norm(v[[0, 2]]-point, axis=1))
    assert np.isinf(distances[1])  # the shared corner is outside the support
    assert np.all(np.isinf(distances[3:]))


def test_invalid_native_geodesic_distances_are_rejected(monkeypatch):
    from server.geometry import local_mapping as lm
    v, f, _ = plate()
    monkeypatch.setattr(lm.igl, 'exact_geodesic', lambda *args: np.zeros(len(args[4])))
    with pytest.raises(MappingControlError, match='invalid surface path'):
        lm._surface_distances(v, f, 0, np.array([0., 0, 0]), 50.)


def test_remote_collar_bridge_cannot_hide_a_pinched_source_vertex(monkeypatch):
    from server.geometry import local_mapping as lm
    # The initial crop has two separate fans at the source. A bridge at
    # x=1.2 joins them only in the larger retry collar; this must not make
    # the locally ambiguous source valid on the second attempt.
    v = np.array([[0., 0, 0], [1.2, -.2, 0], [1.2, -.5, 0],
                  [1.2, .2, .3], [1.2, .5, .3]])
    f = np.array([[0, 1, 2], [0, 3, 4], [1, 3, 4], [1, 4, 2]])
    monkeypatch.setattr(lm.igl, 'exact_geodesic',
                        lambda *args: pytest.fail('pinched sources must be refused before solving'))
    with pytest.raises(MappingControlError, match='pinched surface boundary'):
        lm._surface_distances(v, f, 0, v[0], 1.)


@pytest.mark.slow
@pytest.mark.parametrize('fixture,seed,face_id', [('part1_path', 65, 295), ('part2_path', 19, 16)])
def test_dashboard_25_degree_control_is_independent_of_cached_20_degree_edit(request, fixture, seed, face_id):
    from server.geometry.local_mapping import apply_local_controls
    from server.geometry.meshing import region_meshes
    from server.geometry.ribbing import _projection_frame
    from server.geometry.selection import grow_tangent
    from server.geometry.surface_mapping import surface_coordinates
    shape = load_step(request.getfixturevalue(fixture))
    selected = grow_tangent(shape, [seed], angle_deg=20)
    regions = region_meshes(shape, selected, .15, .075)
    assert len(regions) == 1
    region = regions[0]
    center, axes = _projection_frame(regions)
    matrix = np.column_stack([axes, np.cross(axes[:, 0], axes[:, 1])])
    v, f = (region.vertices-center)@matrix, region.triangles
    uv = surface_coordinates(v, f, v[:, :2])
    n = gr.guide_normals(v, f, gr.imp._vertex_normals(v, f), 3.)
    eligible = np.flatnonzero(region.tri_face == face_id)
    areas = _tri_areas(v, f[eligible])
    point = v[f[eligible[areas.argmax()]]].mean(0)
    edit = control(face_id=face_id, position=point.tolist(), radius=70,
                   angle_deg=25, spacing_scale=.85, height_scale=1.6)
    base = (v, f, uv, n, region.tri_face)
    cold = apply_local_controls(*base, [edit], {})
    cache = {}
    apply_local_controls(*base, [dict(edit, angle_deg=20)], cache)
    warm = apply_local_controls(*base, [edit], cache)
    assert np.all(_tri_areas(cold[2], cold[1]) > 0)
    for a, b in zip(cold[:5], warm[:5]):
        assert np.array_equal(a, b)


def test_invalid_tight_crop_retries_a_validated_collar(monkeypatch):
    from server.geometry import local_mapping as lm
    original = lm._surface_distances_patch
    calls = []
    def cropped(v, f, source_face, point, radius, collar=False):
        calls.append(collar)
        if not collar:
            raise MappingControlError('intrinsic distance solver returned an invalid surface path')
        return original(v, f, source_face, point, radius, collar=True)
    monkeypatch.setattr(lm, '_surface_distances_patch', cropped)
    v, f, _ = plate()
    point = np.array([0., 0, 0])
    result = lm._surface_distances(v, f, 0, point, 50.)
    assert calls == [False, True]
    assert result == pytest.approx(np.linalg.norm(v-point, axis=1))
