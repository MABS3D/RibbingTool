"""Geometric invariants missing from the original small-fixture checks."""
import numpy as np
import pytest

from server.geometry import implicit as imp
from tests.test_implicit_kernel import _surface, _params


def _edge_counts(f):
    edges = np.sort(np.vstack([f[:, [0, 1]], f[:, [1, 2]],
                               f[:, [2, 0]]]), axis=1)
    return np.unique(edges, axis=0, return_counts=True)[1]


@pytest.mark.parametrize("shift", [0.0, 120.0])
def test_tiled_extraction_is_closed_before_hole_filling(monkeypatch, shift):
    # Production pitch is 0.30, unlike the exactly representable 0.25
    # used by the old kernel fixtures. A clean planar field needs no caps
    # or stitching repairs at internal tile planes.
    g = _surface(lambda u, v: np.abs(v - 0.4 * u - 2.0), size=(18, 12))
    g.V[:, 0] += shift
    g.origin = (shift, 0.0)
    monkeypatch.setattr(imp, "_taubin", lambda v, f, **kw: v)
    monkeypatch.setattr(imp, "_fill_microholes", lambda v, f: (v, f))
    clusters = imp.mesh_field(g, _params(), resolution=0.3, tile=24)
    assert clusters
    for v, f in clusters:
        counts = _edge_counts(f)
        assert np.all(counts == 2), (
            f"{(counts == 1).sum()} open edges, "
            f"{(counts > 2).sum()} nonmanifold edges before repairs")


@pytest.mark.parametrize("reverse", [False, True])
def test_depth_band_includes_vertical_sheet_above_covered_columns(reverse):
    # A dashboard lip stands above another face in the same projected
    # columns. The lower face must not hide the lip from the voxel band.
    flat = np.array([[0, 0], [6, 0], [0, 6],
                     [2, 2], [2, 4], [2, 3]], dtype=float)
    depth = np.array([0, 0, 0, 2, 2, 8], dtype=float)
    tris = np.array([[0, 1, 2], [3, 4, 5]])
    if reverse:
        tris = tris[::-1]
    lo, hi = imp._rasterize_depth_range(flat, depth, tris, 1.0,
                                       (0.0, 0.0), (7, 7))
    assert lo[2, 3] == 0.0
    assert hi[2, 3] >= 8.0, "overlapping upright sheet was clipped out"


@pytest.mark.parametrize("reverse", [False, True])
def test_open_edges_use_their_own_triangle_facing(reverse):
    # Two disjoint sheets with opposite windings. A back-facing rim must
    # never cut the slab under a front-facing rib; all three front edges
    # must be protected against tangent wedges regardless of face order.
    v = np.array([[0, 0, 0], [2, 0, 0], [0, 2, 0],
                  [0, 0, 5], [0, 2, 5], [2, 0, 5]], dtype=float)
    f = np.array([[0, 1, 2], [3, 4, 5]])
    edges = imp._front_open_edges(v, f[::-1] if reverse else f)
    assert {tuple(e) for e in np.sort(edges, axis=1)} == {
        (0, 1), (0, 2), (1, 2)}


@pytest.mark.parametrize("reports", [None, []])
def test_invalid_mesh_cannot_be_returned_as_success(monkeypatch, reports):
    from server.geometry.ribbing import RibbingError
    # Model a failed repair: a triangle is missing after extraction. The
    # production caller and a caller without a report list both must fail.
    monkeypatch.setattr(imp, "_fill_microholes", lambda v, f: (v, f[:-1]))
    g = _surface(lambda u, v: np.abs(v - 5), size=(12, 10))
    with pytest.raises(RibbingError, match="closed manifold"):
        imp.mesh_field(g, _params(), resolution=0.3, reports=reports)


def test_successful_merge_is_present_in_returned_geometry(monkeypatch):
    import manifold3d as m3d

    def split_corner(v, f):
        f = f.copy()
        corner = v[f[0, 0]].copy()
        corner[0] += 1e-12
        f[0, 0] = len(v)
        return np.vstack([v, corner]), f

    monkeypatch.setattr(imp, "_fill_microholes", split_corner)
    g = _surface(lambda u, v: np.abs(v - 5), size=(12, 10))
    v, f = imp.mesh_field(g, _params(), resolution=0.3)[0]
    # Validate returned arrays directly, without invoking merge again.
    man = m3d.Manifold(m3d.Mesh64(np.ascontiguousarray(v, np.float64),
                                 np.ascontiguousarray(f, np.uint64)))
    assert not man.is_empty()
    assert np.all(_edge_counts(f) == 2)


@pytest.mark.parametrize("reverse", [False, True])
def test_several_upright_sheets_share_a_depth_column(reverse):
    flat = np.array([[2, 2], [2, 4], [2, 3]] * 2, dtype=float)
    depth = np.array([2, 2, 8, -6, -6, -1], dtype=float)
    tris = np.array([[0, 1, 2], [3, 4, 5]])
    lo, hi = imp._rasterize_depth_range(flat, depth,
                                       tris[::-1] if reverse else tris,
                                       1.0, (0.0, 0.0), (7, 7))
    assert lo[2, 3] <= -6
    assert hi[2, 3] >= 8


def test_diagonal_sliver_does_not_fill_its_entire_bounding_box():
    flat = np.array([[0, 0], [6, 6], [3, 3]], dtype=float)
    lo, hi = imp._rasterize_depth_range(flat, np.array([1., 1., 9.]),
                                       np.array([[0, 1, 2]]),
                                       1.0, (0.0, 0.0), (7, 7))
    assert hi[3, 3] >= 9
    assert lo[0, 6] > 1e8


def test_sliver_depth_is_local_to_the_column():
    # A long narrow slope must not stamp its highest point into every
    # column: depth is also used to decide which folded sheet owns a rib.
    flat = np.array([[0, 0], [0, 10], [0.1, 10]], dtype=float)
    lo, hi = imp._rasterize_depth_range(flat, np.array([0., 100., 100.]),
                                       np.array([[0, 1, 2]]),
                                       1.0, (0.0, 0.0), (3, 12))
    assert lo[0, 1] == pytest.approx(5.0)
    assert hi[0, 1] == pytest.approx(15.0)


def test_export_rejects_open_ribs_instead_of_writing_broken_shells():
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from server.geometry.booleans import BooleanError, mesh_union
    from tests.test_faceted_step import V, T
    with pytest.raises(BooleanError, match="rib cluster 1"):
        mesh_union(BRepPrimAPI_MakeBox(20., 20., 5.).Shape(), [(V, T[:-1])])


def test_export_api_explains_invalid_geometry(box_step):
    from fastapi.testclient import TestClient
    from server.main import app, STATE
    from tests.test_faceted_step import V, T
    client = TestClient(app)
    assert client.post("/api/load_path", json={"path": str(box_step)}).status_code == 200
    STATE["stack"][-1]["overlay"] = [(V, T[:-1])]
    response = client.get("/api/export/stl")
    assert response.status_code == 400
    assert "rib cluster 1" in response.json()["detail"]


def test_shadowed_open_rim_cannot_grow_standing_ribs():
    # The projected silhouette of another panel covers this selected
    # panel's rim. Its global B therefore remains positive even when
    # the closest point is clamped to the open edge, as at the dashboard
    # window. The extrapolated pattern foot is not a valid rim coordinate.
    g = _surface(lambda u, v: np.abs(v - 15), slope=0.9, bdist=4.0)
    g.F = np.ascontiguousarray(g.F[(g.V[g.F, 0] >= 10).all(axis=1)])
    g.N = imp._vertex_normals(g.V, g.F)
    u = np.arange(g.P.shape[0])[:, None] * g.cell
    g.OB = np.broadcast_to(np.abs(u - 10), g.P.shape).astype(np.float32)
    p = _params(margin=2, taper_len=5, fillet_root=2)
    clusters = imp.mesh_field(g, p, resolution=0.3)
    v = np.ascontiguousarray(np.vstack([c[0] for c in clusters]))
    s, cp, _ = imp._surface_eval(v, g.V, g.F, g.N)
    assert (s[cp[:, 0] > 20] > 2.5).any(), "interior ribs were lost"
    assert not ((cp[:, 0] < 11.5) & (s > 0.8)).any(), (
        "standing ribs extend from the open CAD rim into the opening")


def test_export_keeps_thin_features_away_from_origin():
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCP.gp import gp_Pnt
    from server.geometry.booleans import _to_manifold, _man_to_arrays
    # A float32 round-trip collapses this perfectly valid thin solid.
    # Both the body conversion and the exported arrays need CAD precision.
    shape = BRepPrimAPI_MakeBox(gp_Pnt(1e6, 1e6, 1e6), 20., 20., 0.02).Shape()
    man = _to_manifold(shape, 0.2)
    v, f = _man_to_arrays(man, None)
    assert np.ptp(v[:, 2]) == pytest.approx(0.02, abs=1e-8)
    assert len(f) == 12


def test_export_simplification_cannot_detach_fragments():
    import manifold3d as m3d
    from server.geometry.booleans import _man_to_arrays
    solid = m3d.Manifold.cube((10., 10., 10.))
    detached = solid + m3d.Manifold.cube((0.01, 0.01, 0.01)).translate((20, 0, 0))

    class PinchingSimplifier:
        def decompose(self):
            return solid.decompose()

        def simplify(self, tolerance):
            return detached

        def to_mesh64(self):
            return solid.to_mesh64()

    v, f = _man_to_arrays(PinchingSimplifier(), 0.02)
    result = m3d.Manifold(m3d.Mesh64(v, np.ascontiguousarray(f, np.uint64)))
    assert len(result.decompose()) == 1
    assert result.volume() == pytest.approx(1000.)
