from server.geometry.step_io import load_step
from server.geometry.meshing import mesh_shape


def test_box_meshes(box_step):
    ms = mesh_shape(load_step(box_step))
    assert len(ms) == 6
    assert all(m.is_planar for m in ms)
    for m in ms:
        assert m.vertices.shape[1] == 3 and m.triangles.shape[1] == 3
        assert m.uvs.shape == (len(m.vertices), 2)
        assert m.triangles.max() < len(m.vertices)
    areas = sorted(round(m.area) for m in ms)
    assert areas == sorted([60 * 40, 60 * 40, 60 * 8, 60 * 8, 40 * 8, 40 * 8])


def test_cyl_patch_curved(cyl_patch_step):
    ms = mesh_shape(load_step(cyl_patch_step))
    kinds = {m.surface_kind for m in ms}
    assert "cylinder" in kinds
    curved = [m for m in ms if m.surface_kind == "cylinder"]
    assert all(not m.is_planar for m in curved)
