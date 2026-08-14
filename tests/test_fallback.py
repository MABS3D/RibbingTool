from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox

from server.geometry.booleans import mesh_fallback_fuse
from server.geometry.step_io import shape_volume


def test_mesh_fallback_boxes():
    a = BRepPrimAPI_MakeBox(20, 20, 10).Shape()
    b = BRepPrimAPI_MakeBox(10, 10, 30).Shape()  # overlaps a
    out = mesh_fallback_fuse(a, [b])
    v = shape_volume(out)
    expected = 20 * 20 * 10 + 10 * 10 * 20
    assert abs(v - expected) / expected < 0.02
