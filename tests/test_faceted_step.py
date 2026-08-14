import numpy as np

from server.geometry.step_io import load_step, shape_volume, write_faceted_step

# unit cube as 12 triangles (CCW outward)
V = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
              [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], float) * 10.0
T = np.array([
    [0, 2, 1], [0, 3, 2],          # bottom (z=0, normal -z)
    [4, 5, 6], [4, 6, 7],          # top
    [0, 1, 5], [0, 5, 4],          # front
    [1, 2, 6], [1, 6, 5],          # right
    [2, 3, 7], [2, 7, 6],          # back
    [3, 0, 4], [3, 4, 7],          # left
])


def test_faceted_cube_roundtrip(tmp_path):
    p = tmp_path / "cube_faceted.step"
    write_faceted_step(p, V, T, name="cube")
    s = load_step(p)
    assert abs(shape_volume(s) - 1000.0) < 1.0
