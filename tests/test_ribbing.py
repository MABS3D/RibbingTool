import pytest

from OCP.BRepCheck import BRepCheck_Analyzer

from server.geometry.meshing import mesh_shape
from server.geometry.patterns import RibParams
from server.geometry.ribbing import RibbingError, apply_ribs
from server.geometry.step_io import load_step, save_step, shape_volume


def biggest_face_id(shape, kind=None):
    meshes = mesh_shape(shape)
    if kind:
        meshes = [m for m in meshes if m.surface_kind == kind]
    return max(meshes, key=lambda m: m.area).face_id


@pytest.mark.parametrize("pattern", ["rectangular", "triangular", "hexagonal", "stochastic"])
def test_rib_box_top(box_step, pattern):
    s = load_step(box_step)
    fid = biggest_face_id(s)
    p = RibParams(pattern=pattern, spacing=10, thickness=1.6, height=4,
                  margin=2, seed=3, density=0.01)
    out, reports = apply_ribs(s, [fid], p)
    assert shape_volume(out) > shape_volume(s)
    assert reports[0].lofted > 0
    assert BRepCheck_Analyzer(out).IsValid()


def test_rib_with_draft_and_border(box_step):
    s = load_step(box_step)
    fid = biggest_face_id(s)
    p = RibParams(pattern="quadmesh", spacing=12, height=5, draft_deg=3, border=True)
    out, reports = apply_ribs(s, [fid], p)
    assert shape_volume(out) > shape_volume(s)
    assert reports[0].lofted > 0


def test_rib_cylinder_patch(cyl_patch_step):
    s = load_step(cyl_patch_step)
    fid = biggest_face_id(s, kind="cylinder")
    p = RibParams(pattern="isogrid", spacing=12, thickness=1.6, height=3)
    out, reports = apply_ribs(s, [fid], p)
    assert shape_volume(out) > shape_volume(s)
    assert reports[0].skipped < max(1, reports[0].segments * 0.2)


def test_rib_sphere_patch(sphere_patch_step):
    s = load_step(sphere_patch_step)
    fid = biggest_face_id(s, kind="sphere")
    p = RibParams(pattern="hexagonal", spacing=10, height=2.5, thickness=1.2)
    out, reports = apply_ribs(s, [fid], p)
    assert shape_volume(out) > shape_volume(s)
    assert reports[0].lofted > 0


def test_volume_increase_matches_expectation(box_step):
    # single parallel rib family on the 60x40 top: sanity-check added volume scale
    s = load_step(box_step)
    fid = biggest_face_id(s)
    p = RibParams(pattern="rectangular", spacing=10, spacing_y=0, thickness=2,
                  height=5, margin=2, embed=0.3)
    out, reports = apply_ribs(s, [fid], p)
    added = shape_volume(out) - shape_volume(s)
    # 5 ribs x ~56 long x 2 wide x 5 tall = ~2800 mm3 (caps add a little)
    assert 2000 < added < 4000, added


def test_no_faces_error(box_step):
    s = load_step(box_step)
    with pytest.raises(RibbingError):
        apply_ribs(s, [], RibParams())


def test_margin_too_big_error(box_step):
    s = load_step(box_step)
    fid = biggest_face_id(s)
    with pytest.raises(RibbingError, match="no ribs"):
        apply_ribs(s, [fid], RibParams(margin=100))


def test_roundtrip_after_rib(box_step, tmp_path):
    s = load_step(box_step)
    out, _ = apply_ribs(s, [biggest_face_id(s)], RibParams(pattern="rectangular"))
    p = tmp_path / "ribbed.step"
    save_step(out, p)
    s2 = load_step(p)
    assert abs(shape_volume(s2) - shape_volume(out)) / shape_volume(out) < 1e-3
