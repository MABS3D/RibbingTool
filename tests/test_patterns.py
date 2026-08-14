import numpy as np
import pytest
from shapely.geometry import Polygon

from server.geometry.patterns import RibParams, capsule, clip_and_border, generate_segments

B = (0.0, 0.0, 100.0, 50.0)
SQUARE = Polygon([(0, 0), (100, 0), (100, 50), (0, 50)])


def test_rectangular_counts():
    segs = generate_segments(RibParams(pattern="rectangular", spacing=10), B)
    assert 14 <= len(segs) <= 22


def test_single_family():
    p = RibParams(pattern="rectangular", spacing=10, spacing_y=0)
    segs = generate_segments(p, B)
    assert 9 <= len(segs) <= 13
    d = [np.arctan2(s[1][1] - s[0][1], s[1][0] - s[0][0]) for s in segs]
    assert np.allclose(np.abs(np.cos(d)), np.abs(np.cos(d[0])), atol=1e-9)


def test_orientation_rotates():
    a = generate_segments(RibParams(pattern="rectangular", spacing=10), B)
    b = generate_segments(RibParams(pattern="rectangular", spacing=10, orientation_deg=30), B)
    va = np.array(a[0][1]) - np.array(a[0][0])
    vb = np.array(b[0][1]) - np.array(b[0][0])
    ang = np.degrees(np.arccos(abs(np.dot(va, vb)) / (np.linalg.norm(va) * np.linalg.norm(vb))))
    assert ang == pytest.approx(30, abs=1)


def test_all_patterns_nonempty():
    for pat in ("rectangular", "quadmesh", "triangular", "isogrid", "hexagonal", "stochastic"):
        segs = generate_segments(RibParams(pattern=pat, spacing=12), B)
        assert len(segs) > 3, pat


def test_stochastic_reproducible():
    a = generate_segments(RibParams(pattern="stochastic", seed=7), B)
    b = generate_segments(RibParams(pattern="stochastic", seed=7), B)
    assert len(a) == len(b) and np.allclose(np.array(a[0]), np.array(b[0]))


def test_clip_stays_inside():
    p = RibParams(pattern="triangular", spacing=10, margin=3)
    lines = clip_and_border(generate_segments(p, B), SQUARE, p)
    inset = SQUARE.buffer(-2.95)
    assert lines and all(inset.contains(l) for l in lines)


def test_border_ring():
    p_on = RibParams(pattern="rectangular", spacing=1000, margin=3, border=True)
    p_off = RibParams(pattern="rectangular", spacing=1000, margin=3, border=False)
    segs = generate_segments(p_on, B)
    with_b = sum(l.length for l in clip_and_border(segs, SQUARE, p_on))
    without = sum(l.length for l in clip_and_border(segs, SQUARE, p_off))
    # border contribution = inset rectangle perimeter (94 x 44)
    assert with_b - without == pytest.approx((94 + 44) * 2, rel=0.05)


def test_capsule_shape():
    a = capsule((0, 0), (30, 0), 1.0)
    b = capsule((0, 0), (30, 0), 0.6)  # drafted top: same structure
    assert a.shape == b.shape and a.shape[1] == 2
    assert len(a) >= 10
    pa = Polygon(a)
    assert pa.is_valid
    assert pa.area == pytest.approx(30 * 2 + np.pi, rel=0.05)  # rect + circle
    # CCW winding
    x, y = a[:, 0], a[:, 1]
    assert np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y) > 0


def test_capsule_cap_triangulation_covers_footprint():
    from server.geometry.patterns import capsule_cap_triangles
    pts, ns = capsule((0, 0), (30, 0), 1.0, return_meta=True)
    tris = capsule_cap_triangles(len(pts), ns, 5)
    assert len(tris) == len(pts) - 2          # proper simple-polygon count
    area = 0.0
    for a, b, c in tris:
        v1, v2 = pts[b] - pts[a], pts[c] - pts[a]
        signed = 0.5 * (v1[0] * v2[1] - v1[1] * v2[0])
        assert signed > 0                     # consistently CCW, no flips
        area += signed
    assert area == pytest.approx(Polygon(pts).area, rel=1e-6)


def test_from_dict_ignores_unknown():
    p = RibParams.from_dict({"pattern": "hexagonal", "spacing": 9, "bogus": 1})
    assert p.pattern == "hexagonal" and p.spacing == 9
