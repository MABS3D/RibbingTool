from server.geometry.meshing import mesh_shape
from server.geometry.selection import grow_tangent
from server.geometry.step_io import load_step


def test_box_faces_do_not_grow(box_step):
    # all box edges are sharp 90 degrees: growth stays at the seed
    s = load_step(box_step)
    big = max(mesh_shape(s), key=lambda m: m.area).face_id
    assert grow_tangent(s, [big]) == [big]


def test_cyl_patch_lateral_grows_nowhere_sharp(cyl_patch_step):
    # cylinder sector: lateral face meets caps/sides at 90 degrees
    s = load_step(cyl_patch_step)
    cyl = next(m.face_id for m in mesh_shape(s) if m.surface_kind == "cylinder")
    assert grow_tangent(s, [cyl]) == [cyl]


def test_part1_outer_skin_grows(part1_path):
    # face 63 is part of the tangent-continuous outer skin: growth should
    # pick up the smoothly-connected band (fillets, adjacent skin faces)
    s = load_step(part1_path)
    grown = grow_tangent(s, [63], angle_deg=25.0)
    assert 63 in grown
    assert len(grown) > 3
