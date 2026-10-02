"""Geometric finishing contracts, measured on fields and exported material."""
from dataclasses import replace

import manifold3d as m3d
import numpy as np
import pytest
from scipy.optimize import brentq

from server.geometry import graph_ribs as gr
from server.geometry.patterns import RibParams


def _plate():
    vertices = np.array([[-15., -15, 0], [15, -15, 0],
                         [15, 15, 0], [-15, 15, 0]])
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    normals = np.tile([0., 0, 1], (4, 1))
    return vertices, faces, normals


def _params(**changes):
    return replace(RibParams(mapping="project", thickness=2., height=4.,
                            margin=1., taper_len=5., fillet_root=1.2,
                            fillet_top=.5, fillet_junction=0.), **changes)


def _fixture(params, segments=None, height_values=None):
    v, f, n = _plate()
    if segments is None:
        segments = [((-20., 0), (20., 0))]
    paths = gr.trace_rib_paths(v, f, n, segments, params,
                              height_values=height_values)
    ribbons = gr.ribbons_from_paths(paths, params)
    body = m3d.Manifold.cube([30., 30, 8]).translate([-15, -15, -8])
    mesh = body.to_mesh64()
    field = gr.RibField(ribbons, np.asarray(mesh.vert_properties[:, :3]),
                        np.asarray(mesh.tri_verts, np.int64), params,
                        substrate=(v, f), height_values=height_values)
    return field, paths, ribbons, body


def _crown(field, x, y, maximum=6.5):
    """Highest crossing of a vertical line through the actual scalar field."""
    z = np.linspace(-.2, maximum, 240)
    samples = np.column_stack([np.full(len(z), x), np.full(len(z), y), z])
    values = field(samples)
    inside = np.flatnonzero(values < -1e-9)
    if not len(inside):
        return 0.
    i = inside[-1]
    assert i < len(z)-1, "crown exceeds the measurement interval"
    return brentq(lambda height: float(field(np.array([[x, y, height]]))[0]),
                  z[i], z[i+1], xtol=1e-10)


@pytest.mark.parametrize("root_radius", [.5, 1.2, 2.])
@pytest.mark.parametrize("variable_height", [False, True])
def test_root_fillet_preserves_the_preview_runout_height(root_radius, variable_height):
    # The optional field models a gradual local height edit on the same plate.
    heights = np.array([.5, 1.5, 1.5, .5]) if variable_height else None
    field, [path], _, _ = _fixture(_params(fillet_root=root_radius),
                                   height_values=heights)
    order = np.argsort(path["points"][:, 0])
    xs = path["points"][order, 0]
    hs = path["height"][order]
    for x in [13., 12.9, 12.5, 12., 11., 8.]:
        expected = float(np.interp(x, xs, hs))
        actual = _crown(field, x, 0)
        assert actual == pytest.approx(expected, abs=.02), (
            f"root radius {root_radius} raises the run-out at x={x}: "
            f"preview {expected:.4f} mm, solid {actual:.4f} mm")


def test_top_fillet_is_tangent_to_the_junction_crown():
    params = _params(taper_len=0, fillet_root=0, fillet_junction=1.2)
    field, _, _, _ = _fixture(params,
        [((-10., 0), (10., 0)), ((0., -10), (0., 10))])
    # Follow the diagonal across an X intersection. A smooth top radius must
    # leave the constant-height crown tangentially, even after junction fillets.
    assert _crown(field, 0, 0) == pytest.approx(params.height, abs=1e-6)
    inside, outside = 0., 3.
    for _ in range(32):
        middle = (inside+outside)/2
        if _crown(field, middle, middle) >= params.height-1e-7:
            inside = middle
        else:
            outside = middle
    drop = params.height-_crown(field, outside+.01, outside+.01)
    assert drop < .0015, (
        f"junction crown leaves its plateau with a sharp edge: "
        f"{drop:.6f} mm drop over 0.01 mm despite a {params.fillet_top} mm top fillet")
    assert _crown(field, 7., .9) < params.height-.1, (
        "a smooth junction must retain the rounding on the straight rib")


def test_a_distant_full_height_rib_cannot_raise_another_ribs_runout():
    params = _params(fillet_junction=1.2)
    _, tapered, _, body = _fixture(params)
    v, f, n = _plate()
    tall = gr.trace_rib_paths(v, f, n, [((-10., 7.), (10., 7.))],
                             replace(params, taper_len=0.))
    for path in tall:
        path["group"] = 1
    # Separate field groups are intentional: finite networks and disconnected
    # selected regions can contain distinct, non-intersecting rib fields.
    ribbons = gr.ribbons_from_paths(tapered+tall, params)
    mesh = body.to_mesh64()
    field = gr.RibField(ribbons, np.asarray(mesh.vert_properties[:, :3]),
                        np.asarray(mesh.tri_verts, np.int64), params,
                        substrate=(v, f))
    assert _crown(field, 13., 0.) < .02
    assert _crown(field, 0., 7.) == pytest.approx(params.height, abs=.02)


def test_crossing_ribs_with_different_heights_preserve_the_taller_crown():
    params = _params(taper_len=0, fillet_root=0, fillet_junction=1.2)
    _, tall, _, body = _fixture(params, [((-10., 0), (10., 0))])
    v, f, n = _plate()
    short = gr.trace_rib_paths(v, f, n, [((0., -10.), (0., 10.))],
                              replace(params, height=2.))
    for path in short:
        path["group"] = 1
    ribbons = gr.ribbons_from_paths(tall+short, params)
    mesh = body.to_mesh64()
    field = gr.RibField(ribbons, np.asarray(mesh.vert_properties[:, :3]),
                        np.asarray(mesh.tri_verts, np.int64), params,
                        substrate=(v, f))
    # A mean of the two heights would cut a 1 mm dip into the taller rib.
    assert _crown(field, 0., 0.) == pytest.approx(4., abs=.1)
    assert _crown(field, 7., 0.) == pytest.approx(4., abs=.02)
    assert _crown(field, 0., 7.) == pytest.approx(2., abs=.02)


def test_crossing_crown_gradients_blend_without_a_crease():
    params = _params(taper_len=0, fillet_root=0, fillet_junction=1.2)
    _, paths, _, body = _fixture(params,
        [((-6., 0), (6., 0)), ((0., -6), (0., 6))])
    # Both prescribed crowns meet at 3 mm, with different tangent directions.
    # A hard nearest-crown switch makes a ridge along x=y at the junction.
    for path, axis in zip(paths, [0, 1]):
        path["height"] = 3.+.12*path["points"][:, axis]
        path["nominal_height"] = path["height"].copy()
    ribbons = gr.ribbons_from_paths(paths, params)
    v, f, _ = _plate()
    mesh = body.to_mesh64()
    field = gr.RibField(ribbons, np.asarray(mesh.vert_properties[:, :3]),
                        np.asarray(mesh.tri_verts, np.int64), params,
                        substrate=(v, f))
    left, center, right = (_crown(field, x, 0.) for x in [-.02, 0., .02])
    incoming, outgoing = (center-left)/.02, (right-center)/.02
    assert abs(incoming-outgoing) < .03, (
        f"crown has a tangent jump {incoming:.4f} -> {outgoing:.4f} "
        "where the two rib directions meet")
    assert center == pytest.approx(3., abs=.02)


def _width(field, x, height):
    ys = np.linspace(0., 5., 220)
    values = field(np.column_stack([np.full(len(ys), x), ys,
                                    np.full(len(ys), height)]))
    inside = np.flatnonzero(values < -1e-9)
    assert len(inside) and inside[-1] < len(ys)-1
    i = inside[-1]
    return 2*brentq(lambda y: float(field(np.array([[x, y, height]]))[0]),
                    ys[i], ys[i+1], xtol=1e-10)


@pytest.mark.parametrize("root_radius", [1.2, 2.])
def test_root_footprint_fades_with_the_runout_instead_of_leaving_a_flat_tongue(root_radius):
    params = _params(fillet_root=root_radius)
    field, [path], _, _ = _fixture(params)
    order = np.argsort(path["points"][:, 0])
    for x in [12.9, 12.5]:
        height = float(np.interp(x, path["points"][order, 0], path["height"][order]))
        width = _width(field, x, height/2)
        assert width <= params.thickness+2*height+.05, (
            f"{height:.4f} mm high run-out keeps a {width:.4f} mm wide root shelf; "
            "the footprint must fade with the remaining rib height")
    # Preserve the requested root fillet on the full-height part of the wall.
    assert _width(field, 0., .05) > params.thickness+.5


def _vertical_mesh_intersections(v, f, x, y):
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    ab, ac = b-a, c-a
    determinant = ab[:, 0]*ac[:, 1]-ab[:, 1]*ac[:, 0]
    usable = np.abs(determinant) > 1e-12
    u, w = np.zeros(len(f)), np.zeros(len(f))
    u[usable] = ((x-a[usable, 0])*ac[usable, 1]
                  -(y-a[usable, 1])*ac[usable, 0])/determinant[usable]
    w[usable] = (ab[usable, 0]*(y-a[usable, 1])
                  -ab[usable, 1]*(x-a[usable, 0]))/determinant[usable]
    usable &= (u >= -1e-8) & (w >= -1e-8) & (u+w <= 1+1e-8)
    return (a[:, 2]+u*ab[:, 2]+w*ac[:, 2])[usable]


def test_finished_runout_mesh_unions_to_one_closed_body_without_a_raised_tip():
    params = _params()
    field, _, ribbons, body = _fixture(params)
    grid = gr.extraction_grid(*_plate(), ribbons, params, .2)
    clusters = gr.imp.mesh_field(grid, params, resolution=.2, field=field)
    assert clusters
    material = []
    for v, f in clusters:
        solid = m3d.Manifold(m3d.Mesh64(np.ascontiguousarray(v, np.float64),
                                       np.ascontiguousarray(f, np.uint64)))
        assert solid.status() == m3d.Error.NoError
        material.append(solid)
    joined = m3d.Manifold.batch_boolean([body, *material], m3d.OpType.Add)
    assert joined.status() == m3d.Error.NoError
    assert len(joined.decompose()) == 1
    assert joined.volume() > body.volume()+100
    mesh = joined.to_mesh64()
    v, f = np.asarray(mesh.vert_properties[:, :3]), np.asarray(mesh.tri_verts)
    edges = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
    assert np.all(np.unique(edges, axis=0, return_counts=True)[1] == 2)
    assert v[:, 2].max() == pytest.approx(params.height, abs=.08)
    for x in [-13., 13.]:
        crossings = _vertical_mesh_intersections(v, f, x, 0.)
        assert len(crossings) >= 2
        assert crossings.max() <= .1, (
            f"taper should meet the body at x={x}; "
            f"extracted root leaves a {crossings.max():.4f} mm raised tip")
