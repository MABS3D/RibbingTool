import numpy as np

from server.geometry.patterns import RibParams
from server.geometry.ribbing import build_rib_solids
from server.geometry.step_io import load_step, shape_volume
from tests.test_ribbing import biggest_face_id


def _components(solids):
    """Number of connected components of the fused rib lattice."""
    import manifold3d as m3d
    from server.geometry.booleans import _to_manifold
    mans = [_to_manifold(s, 0.3) for s in solids]
    union = m3d.Manifold.batch_boolean(mans, m3d.OpType.Add)
    assert not union.is_empty()
    return len(union.decompose())


def test_border_fuses_parallel_ribs_into_one_frame(box_step):
    # single-family parallel ribs are disconnected — the border rib must
    # weld them into one frame
    s = load_step(box_step)
    fid = biggest_face_id(s)
    base = dict(pattern="rectangular", spacing=10, spacing_y=0,
                thickness=2, height=4, margin=3, taper_len=0)
    solids_off, _ = build_rib_solids(s, [fid], RibParams(**base))
    solids_on, _ = build_rib_solids(s, [fid], RibParams(**base, border=True))
    assert _components(solids_off) > 1
    assert _components(solids_on) == 1


def test_taper_ramps_rib_ends(box_step):
    # ribs on the 60x40 top: tapered ends must reduce material and pull the
    # rib tops down toward the face (z=8) near the boundary
    s = load_step(box_step)
    fid = biggest_face_id(s)
    base = dict(pattern="rectangular", spacing=10, spacing_y=0,
                thickness=2, height=5, margin=2)
    abrupt, _ = build_rib_solids(s, [fid], RibParams(**base, taper_len=0))
    tapered, _ = build_rib_solids(s, [fid], RibParams(**base, taper_len=8))
    v_abrupt = sum(shape_volume(x) for x in abrupt)
    v_tapered = sum(shape_volume(x) for x in tapered)
    assert v_tapered < v_abrupt * 0.85

    # end regions: sample the tapered solids' bounding boxes — at least one
    # rib must top out well below full height somewhere (wedge run-out)
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib
    tops = []
    for x in tapered:
        box = Bnd_Box()
        BRepBndLib.Add_s(x, box)
        tops.append(box.Get()[5])
    assert max(tops) <= 8 + 5 + 0.3          # nothing exceeds full height
    # taper factor floor is 0.05: tips reach nearly the face
    # (verify via volume reduction above; bbox z-max stays at full height
    # because mid-rib keeps full height)


def test_border_with_taper_ignored(box_step):
    # border=True disables tapering (ribs meet the border at full height)
    s = load_step(box_step)
    fid = biggest_face_id(s)
    base = dict(pattern="rectangular", spacing=10, spacing_y=0,
                thickness=2, height=5, margin=3, border=True)
    a, _ = build_rib_solids(s, [fid], RibParams(**base, taper_len=0))
    b, _ = build_rib_solids(s, [fid], RibParams(**base, taper_len=8))
    va = sum(shape_volume(x) for x in a)
    vb = sum(shape_volume(x) for x in b)
    assert abs(va - vb) / va < 1e-6
