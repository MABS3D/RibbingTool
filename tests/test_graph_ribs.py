"""Shape/parameter contracts for the explicit graph construction."""
from dataclasses import replace

import igl
import manifold3d as m3d
import numpy as np
import pytest

from server.geometry import graph_ribs as gr
from server.geometry.patterns import RibParams, generate_segments
from server.geometry.step_io import load_step
from tests.test_ribbing import biggest_face_id


def plate():
    v = np.array([[-15., -15, 0], [15, -15, 0], [15, 15, 0], [-15, 15, 0]])
    f = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    return v, f, np.tile([0., 0, 1], (4, 1))


def params(**kwargs):
    return RibParams(mapping="project", thickness=2, height=4, margin=1,
                     taper_len=0, fillet_junction=0, **kwargs)


def ribbon(p=None):
    return gr.make_ribbons(*plate(), [((-10., 0), (10., 0))], p or params())[0][0]


def test_projection_keeps_finite_endpoints():
    paths = gr.projected_paths(*plate(), (-4., 1), (3., 1))
    assert len(paths) == 1
    points, normals = paths[0]
    assert sorted(points[:, 0]) == pytest.approx([-4., 1., 3.])
    assert points[:, 1] == pytest.approx(1.)


def test_projection_continues_across_overhang():
    # Two selected faces meet at the silhouette and have opposite projected winding.
    v = np.array([[-5., -5, 0], [0, -5, 4], [-5, -5, 8],
                  [-5, 5, 0], [0, 5, 4], [-5, 5, 8]])
    f = np.array([[0, 1, 4], [0, 4, 3], [1, 2, 5], [1, 5, 4]])
    n = gr.imp._vertex_normals(v, f)
    paths = gr.projected_paths(v, f, n, (-10., .7), (3., .7))
    assert len(paths) == 1
    assert sorted(paths[0][0][[0, -1], 2]) == pytest.approx([0, 8])


def test_wall_thickness_height_and_top_radius_are_independent():
    r = ribbon()
    q = np.array([[0., 1, 2], [0, -1, 2], [0, 0, 4], [0, 1, 4], [0, 0, 4.1]])
    sharp = r.field(q, 1., 0., 0.)
    rounded = r.field(q, 1., .5, 0.)
    assert sharp == pytest.approx([0., 0., 0., 0., .1], abs=1e-8)
    assert rounded[:3] == pytest.approx([0., 0., 0.], abs=1e-8)
    assert rounded[3] == pytest.approx((2**.5-1)*.5)


def test_draft_changes_root_width_not_crown_width():
    r = ribbon()
    q = np.array([[0., 1.3, 1], [0, 1., 4]])
    assert r.field(q, 1., 0., .1) == pytest.approx([0., 0.], abs=1e-7)
    # Draft stops at the root; it cannot create a growing cone below the strip.
    q = np.array([[0., 1.5, -2.]])
    assert r.field(q, 1., 0., .2, height=4.)[0] > 0


def test_finite_network_junctions_do_not_taper_at_each_segment_end():
    p = replace(params(), taper_len=3.)
    segs = [((-8., 0), (0, 0)), ((0., 0), (4, 7)), ((0., 0), (4, -7))]
    ribs, _ = gr.make_ribbons(*plate(), segs, p)
    q = np.array([[0., 0, 3.]])
    assert max(float(r.field(q, 1., 0., 0.)[0]) for r in ribs) < -.9


@pytest.mark.parametrize("pattern", ["hexagonal", "stochastic"])
def test_finite_patterns_preserve_empty_cell_interiors(pattern):
    p = replace(params(), pattern=pattern, spacing=8, density=.018)
    segments = generate_segments(p, (-15, -15, 15, 15))
    ribs, _ = gr.make_ribbons(*plate(), segments, p)
    rng = np.random.default_rng(21)
    xy = rng.uniform(-8, 8, (400, 2))
    # Compare occupancy to the actual finite planar network, not infinite lines.
    a = np.array([s[0] for s in segments]); b = np.array([s[1] for s in segments])
    d = b-a
    t = np.clip(((xy[:, None]-a)*d).sum(2)/(d*d).sum(1), 0, 1)
    dist = np.linalg.norm(xy[:, None]-(a+t[:, :, None]*d), axis=2).min(1)
    q = np.column_stack([xy, np.full(len(xy), 2.)])
    field = np.stack([r.field(q, 1., 0., 0.) for r in ribs]).min(0)
    assert np.all(field[dist > 1.2] > 0)
    assert np.all(field[dist < .8] < 0)


def test_cached_body_distance_matches_closed_mesh_pseudonormals():
    mesh = m3d.Manifold.cube([12., 10, 2], center=True).to_mesh64()
    v, f = np.asarray(mesh.vert_properties[:, :3]), np.asarray(mesh.tri_verts, np.int64)
    q = np.random.default_rng(8).uniform([-8, -7, -4], [8, 7, 4], (4000, 3))
    expected = igl.signed_distance(q, v, f,
                 igl.SignedDistanceType.SIGNED_DISTANCE_TYPE_PSEUDONORMAL)[0]
    assert gr.MeshDistance(v, f, signed=True).signed(q) == pytest.approx(expected, abs=1e-8)


def test_junction_blend_is_symmetric_and_has_compact_support():
    f = np.array([[-.4, .2, 2., -.8], [.2, -.4, -.6, -.3], [3., 3., 4., -.1]])
    base = gr._round_union(f, 1.)
    assert gr._round_union(f[[2, 0, 1]], 1.) == pytest.approx(base)
    assert base[2] == pytest.approx(-.6)
    assert base[3] < f[:, 3].min()


def test_fold_field_has_no_closest_sheet_jump():
    v = np.array([[-8., -8, 0], [0, -8, -4], [8, -8, 0],
                  [-8, 8, 0], [0, 8, -4], [8, 8, 0]])
    f = np.array([[0, 1, 4], [0, 4, 3], [1, 2, 5], [1, 5, 4]])
    n = gr.guide_normals(v, f, gr.imp._vertex_normals(v, f), 3.)
    ribs, _ = gr.make_ribbons(v, f, n, [((-10., 0), (10., 0))], params())
    x = np.linspace(-.1, .1, 501)
    q = np.column_stack([x, np.full(len(x), .9), np.full(len(x), -.5)])
    field = ribs[0].field(q, 1., .4, 0.)
    assert np.max(np.abs(np.diff(field))) < .002
    assert np.all(field < 0)


def test_guide_is_rotation_equivariant_and_outward():
    v = np.array([[-5., -5, 0], [0, -5, 4], [-5, -5, 8],
                  [-5, 5, 0], [0, 5, 4], [-5, 5, 8]])
    f = np.array([[0, 1, 4], [0, 4, 3], [1, 2, 5], [1, 5, 4]])
    n = gr.imp._vertex_normals(v, f)
    guide = gr.guide_normals(v, f, n, 3.)
    from scipy.spatial.transform import Rotation
    rot = Rotation.from_euler("xyz", [23, -38, 17], degrees=True).as_matrix()
    got = gr.guide_normals(v@rot.T, f, n@rot.T, 3.)
    assert got == pytest.approx(guide@rot.T, abs=1e-10)
    assert np.all((guide*n).sum(1) > .1)


@pytest.mark.parametrize("border", [False, True])
def test_real_body_union_is_one_closed_solid(box_step, border):
    from server.geometry.booleans import mesh_union
    s = load_step(box_step)
    p = replace(params(fillet_root=1., fillet_top=.4), border=border,
                fillet_junction=.8)
    clusters, reports = gr.build_rib_graph(s, [biggest_face_id(s)], p)
    assert clusters and reports[0].lofted > 0
    for v, f in mesh_union(s, clusters):
        solid = m3d.Manifold(m3d.Mesh64(np.ascontiguousarray(v),
                                      np.ascontiguousarray(f, np.uint64)))
        assert solid.status() == m3d.Error.NoError
        assert len(solid.decompose()) == 1
        assert solid.volume() > 60*40*8 + 500


def test_rotating_body_preserves_rib_volume(rotated_box_step, box_step):
    def volume(path):
        s = load_step(path)
        clusters, _ = gr.build_rib_graph(s, [biggest_face_id(s)], params())
        return sum(np.einsum("ij,ij->i", v[f[:, 0]],
                            np.cross(v[f[:, 1]], v[f[:, 2]])).sum()/6
                   for v, f in clusters)
    assert volume(rotated_box_step) == pytest.approx(volume(box_step), rel=.015)


def test_extracted_wall_keeps_dimensions():
    v, f, n = plate()
    p = params(fillet_top=.4)
    ribs, _ = gr.make_ribbons(v, f, n, [((-10., 0), (10., 0))], p)
    body = m3d.Manifold.cube([30., 30, 8]).translate([-15, -15, -8]).to_mesh64()
    field = gr.RibField(ribs, np.asarray(body.vert_properties[:, :3]),
                        np.asarray(body.tri_verts, np.int64), p)
    grid = gr.extraction_grid(v, f, n, ribs, p, .16)
    [(v, f)] = gr.imp.mesh_field(grid, p, .16, field=field)
    # Intersect the Y-directed line through (x=0,z=2) with actual triangles.
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    dx, dz = b[:, 0]-a[:, 0], b[:, 2]-a[:, 2]
    ex, ez = c[:, 0]-a[:, 0], c[:, 2]-a[:, 2]
    determinant = dx*ez-dz*ex
    valid = np.abs(determinant) > 1e-12
    u = np.zeros(len(f)); w = np.zeros(len(f))
    u[valid] = (-a[valid, 0]*ez[valid]-(2-a[valid, 2])*ex[valid])/determinant[valid]
    w[valid] = (dx[valid]*(2-a[valid, 2])+dz[valid]*a[valid, 0])/determinant[valid]
    valid &= (u >= -1e-8) & (w >= -1e-8) & (u+w <= 1+1e-8)
    ys = (a[:, 1]+u*(b[:, 1]-a[:, 1])+w*(c[:, 1]-a[:, 1]))[valid]
    assert ys.max()-ys.min() == pytest.approx(2., abs=.08)
    assert v[:, 2].max() == pytest.approx(4., abs=.08)


def test_empty_space_culling_preserves_extracted_surface():
    v, f, n = plate()
    p = params(fillet_root=1.1, fillet_top=.4)
    p.fillet_junction = 1.2
    ribs, _ = gr.make_ribbons(v, f, n,
                              [((-10., 0), (10., 0)), ((0., -10), (0., 10))], p)
    body = m3d.Manifold.cube([30., 30, 8]).translate([-15, -15, -8]).to_mesh64()
    field = gr.RibField(ribs, np.asarray(body.vert_properties[:, :3]),
                        np.asarray(body.tri_verts, np.int64), p)
    grid = gr.extraction_grid(v, f, n, ribs, p, .25)
    [(a, af)] = gr.imp.mesh_field(grid, p, .25, field=field)
    field.cull = False
    [(b, bf)] = gr.imp.mesh_field(grid, p, .25, field=field)
    assert np.array_equal(af, bf)
    assert a == pytest.approx(b, abs=1e-12)


def test_large_root_radius_cannot_exceed_height():
    p = params(fillet_root=8.)
    ribs, _ = gr.make_ribbons(*plate(), [((-10., 0), (10., 0))], p)
    body = m3d.Manifold.cube([30., 30, 8]).translate([-15, -15, -8]).to_mesh64()
    field = gr.RibField(ribs, np.asarray(body.vert_properties[:, :3]),
                        np.asarray(body.tri_verts, np.int64), p)
    q = np.array([[0., 0, 4.1], [0, 0, 5], [0, 1, 4.1]])
    assert np.all(field(q) > 0)


def test_root_blend_cannot_print_through_thin_body():
    v, f, n = plate()
    p = params(fillet_root=2.)
    ribs, _ = gr.make_ribbons(v, f, n, [((-10., 0), (10., 0))], p)
    body = m3d.Manifold.cube([30., 30, .8]).translate([-15, -15, -.8]).to_mesh64()
    field = gr.RibField(ribs, np.asarray(body.vert_properties[:, :3]),
                        np.asarray(body.tri_verts, np.int64), p, substrate=(v, f))
    q = np.array([[0., 0, -.9], [0, .8, -1.1], [0, 0, .5], [0, 0, -.3]])
    values = field(q)
    assert np.all(values[:2] > 0), "ribs leaked out of the unselected back face"
    assert np.all(values[2:] < 0), "the rib lost its exterior wall or embedded root"


def test_excessive_border_margin_is_a_geometry_error(box_step):
    from server.geometry.ribbing import RibbingError
    s = load_step(box_step)
    p = replace(params(), border=True, margin=100)
    with pytest.raises(RibbingError, match="margin leaves no border"):
        gr.build_rib_graph(s, [biggest_face_id(s)], p)


def _arrays(solid):
    mesh = solid.to_mesh64()
    return (np.array(mesh.vert_properties[:, :3], copy=True),
            np.array(mesh.tri_verts, np.int64, copy=True))


def test_attachment_preserves_small_roots_and_removes_floating_and_touching_parts():
    body = m3d.Manifold.cube([12., 10., 2.], center=True)
    small = m3d.Manifold.cube([.1, .1, .1]).translate([0, 0, .975])
    floating = m3d.Manifold.cube([3., 3., 3.]).translate([0, 0, 3.])
    touching = m3d.Manifold.cube([1., 1., 1.]).translate([6., 5., 1.])
    warnings = []
    clusters = gr.attached_components([_arrays(p) for p in [small, floating, touching]],
                                      body, warnings)
    v, f = clusters[0]
    result = m3d.Manifold(m3d.Mesh64(v, np.asarray(f, np.uint64)))
    assert result.volume() == pytest.approx(.001)
    assert len((body+result).decompose()) == 1
    assert "removed 2" in warnings[0]


def test_attachment_uses_volume_overlap_even_without_interior_vertices():
    body = m3d.Manifold.cube([12., 10., 2.], center=True)
    # All eight bar vertices are outside the body; its middle intersects it.
    bar = m3d.Manifold.cube([16., .2, .2]).translate([-8, 0, .9])
    result = gr.attached_components([_arrays(bar)], body, [])
    assert len(result) == 1
    v, f = result[0]
    assert m3d.Manifold(m3d.Mesh64(v, np.asarray(f, np.uint64))).volume() == pytest.approx(.64)


def test_attachment_keeps_internal_cavities_when_removing_an_island():
    body = m3d.Manifold.cube([12., 10., 2.], center=True)
    hollow = (m3d.Manifold.cube([2., 2., 2.]).translate([0, 0, .5])
              - m3d.Manifold.cube([.5, .5, .5]).translate([.5, .5, 1.5]))
    floating = m3d.Manifold.cube([1., 1., 1.]).translate([0, 0, 4.])
    [(v, f)] = gr.attached_components([_arrays(hollow), _arrays(floating)], body, [])
    assert m3d.Manifold(m3d.Mesh64(v, np.asarray(f, np.uint64))).volume() == pytest.approx(8.-.125)


def test_attachment_rejects_a_completely_unrooted_result():
    from server.geometry.ribbing import RibbingError
    body = m3d.Manifold.cube([12., 10., 2.], center=True)
    floating = m3d.Manifold.cube([1., 1., 1.]).translate([0, 0, 4.])
    with pytest.raises(RibbingError, match="no solid attachment"):
        gr.attached_components([_arrays(floating)], body, [])


def test_attachment_unions_overlapping_components_before_export():
    body = m3d.Manifold.cube([12., 10., 2.], center=True)
    a = m3d.Manifold.cube([1., 1., 3.]).translate([0, 0, .5])
    b = a.translate([0, .5, 0])
    [(v, f)] = gr.attached_components([_arrays(a), _arrays(b)], body, [])
    result = m3d.Manifold(m3d.Mesh64(v, np.asarray(f, np.uint64)))
    assert result.volume() == pytest.approx(4.5)
    assert len(result.decompose()) == 1


def test_graph_export_rebuild_failure_is_reported(box_step, monkeypatch):
    from fastapi.testclient import TestClient
    from server import main
    client = TestClient(main.app)
    response = client.post("/api/load_path", json={"path": str(box_step)})
    biggest = max(response.json()["faces"], key=lambda face: face["area"])["id"]
    response = client.post("/api/ribs", json={"face_ids": [biggest],
                            "params": {"mapping": "project"}})
    assert response.status_code == 200
    entry = main.STATE["stack"][-1]
    assert entry["recipes"][0]["engine"] == "graph"
    def fail(*args, **kwargs):
        assert kwargs["quality"] == 3.
        raise ValueError("test rebuild failure")
    monkeypatch.setattr(main, "build_rib_graph", fail)
    response = client.get("/api/export/stl")
    assert response.status_code == 400
    assert "high-quality rib rebuild failed" in response.json()["detail"]
    assert "fine_shells" not in entry
