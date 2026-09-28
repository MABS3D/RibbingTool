import math
import pathlib

import pytest

from OCP.BRepPrimAPI import (
    BRepPrimAPI_MakeBox,
    BRepPrimAPI_MakeCylinder,
    BRepPrimAPI_MakeSphere,
)
from OCP.STEPControl import STEPControl_Writer, STEPControl_StepModelType
from OCP.IFSelect import IFSelect_RetDone
from OCP.BRepGProp import BRepGProp
from OCP.GProp import GProp_GProps

REPO = pathlib.Path(__file__).resolve().parents[1]


def write_step(shape, path):
    w = STEPControl_Writer()
    w.Transfer(shape, STEPControl_StepModelType.STEPControl_AsIs)
    assert w.Write(str(path)) == IFSelect_RetDone


def volume(shape):
    p = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, p)
    return p.Mass()


@pytest.fixture(scope="session")
def box_step(tmp_path_factory):
    p = tmp_path_factory.mktemp("fx") / "box.step"
    write_step(BRepPrimAPI_MakeBox(60.0, 40.0, 8.0).Shape(), p)
    return p


@pytest.fixture(scope="session")
def rotated_box_step(tmp_path_factory):
    # same box, rotated about a skew axis: the projection frame becomes a
    # NON-symmetric rotation (M.T != M), so any frame/world transform mixup
    # that an axis-aligned box hides (M ~ identity) must show here
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
    from OCP.gp import gp_Ax1, gp_Dir, gp_Pnt, gp_Trsf
    t = gp_Trsf()
    t.SetRotation(gp_Ax1(gp_Pnt(0, 0, 0), gp_Dir(1.0, 2.0, 3.0)),
                  math.radians(40.0))
    shape = BRepBuilderAPI_Transform(
        BRepPrimAPI_MakeBox(60.0, 40.0, 8.0).Shape(), t, True).Shape()
    p = tmp_path_factory.mktemp("fx") / "box_rot.step"
    write_step(shape, p)
    return p


@pytest.fixture(scope="session")
def cyl_patch_step(tmp_path_factory):
    # 120-degree cylinder sector: lateral face is a curved topological disk
    p = tmp_path_factory.mktemp("fx") / "cylpatch.step"
    write_step(BRepPrimAPI_MakeCylinder(30.0, 50.0, math.radians(120)).Shape(), p)
    return p


@pytest.fixture(scope="session")
def groove_step(tmp_path_factory):
    # box with a cylindrical groove: the groove face is CONCAVE, where
    # offset surfaces of a tessellated substrate crease at every facet
    # Voronoi wall (convex faces round them into C1 arcs instead)
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
    from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt
    box = BRepPrimAPI_MakeBox(80.0, 50.0, 20.0).Shape()
    cyl = BRepPrimAPI_MakeCylinder(
        gp_Ax2(gp_Pnt(-5.0, 25.0, 38.0), gp_Dir(1.0, 0.0, 0.0)),
        25.0, 90.0).Shape()
    shape = BRepAlgoAPI_Cut(box, cyl).Shape()
    p = tmp_path_factory.mktemp("fx") / "groove.step"
    write_step(shape, p)
    return p


@pytest.fixture(scope="session")
def split_top_step(tmp_path_factory):
    # box whose top face is split by a narrow groove: selecting the two
    # top faces (NOT the groove) gives a sealed separator strip with open
    # substrate boundaries on both flanks — the fixture for tangent-wedge
    # slab leakage into the strip void
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
    box = BRepPrimAPI_MakeBox(80.0, 40.0, 10.0).Shape()
    slot = BRepPrimAPI_MakeBox(2.5, 44.0, 6.0).Shape()
    from OCP.gp import gp_Trsf, gp_Vec
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
    t = gp_Trsf()
    t.SetTranslation(gp_Vec(38.75, -2.0, 6.0))
    slot = BRepBuilderAPI_Transform(slot, t, True).Shape()
    shape = BRepAlgoAPI_Cut(box, slot).Shape()
    p = tmp_path_factory.mktemp("fx") / "splittop.step"
    write_step(shape, p)
    return p


@pytest.fixture(scope="session")
def wrap_cyl_step(tmp_path_factory):
    # 260-degree cylinder sector: the lateral face wraps PAST the
    # silhouette, so its outer bands face away from any projection frame —
    # a fold-over fixture for projected-mapping gates
    p = tmp_path_factory.mktemp("fx") / "cylwrap.step"
    write_step(BRepPrimAPI_MakeCylinder(30.0, 50.0, math.radians(260)).Shape(), p)
    return p


@pytest.fixture(scope="session")
def dome_roll_step(tmp_path_factory):
    # mushroom dome: full sphere R=60 fused into a box top with its
    # center ABOVE the top plane, so the cap emerges through 97 deg of
    # polar angle: the silhouette ring floats 7mm above the flat
    # (fold-under exposed, not buried) and the flat top continues
    # beyond the contact circle — the 81-85deg facing band is INTERIOR
    # to the projected domain (B stays large), unlike wrap_cyl whose
    # silhouette IS the domain rim and dies to the margin inset.  The
    # sphere's x/y symmetry pins the PCA projection frame to +z
    # (cylinder-bulb variants nearly tie x/z vertex variance and flip
    # the frame sideways).  R=60 makes the [81, 85]deg band ~0.5mm of
    # projected width — resolvable at the engine's 0.30mm voxels, as on
    # the wide dashboard S-rolls that show the gate-shard defect.
    import math as _math
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Fuse
    from OCP.gp import gp_Pnt
    R = 60.0
    half = _math.ceil(R * _math.sin(_math.radians(97.0)) + 6.5)
    zc = 20.0 - R * _math.cos(_math.radians(97.0))
    box = BRepPrimAPI_MakeBox(2.0 * half, 2.0 * half, 20.0).Shape()
    sph = BRepPrimAPI_MakeSphere(gp_Pnt(half, half, zc), R).Shape()
    p = tmp_path_factory.mktemp("fx") / "domeroll.step"
    write_step(BRepAlgoAPI_Fuse(box, sph).Shape(), p)
    return p


@pytest.fixture(scope="session")
def sphere_patch_step(tmp_path_factory):
    # sphere sector: doubly-curved face, still a disk
    p = tmp_path_factory.mktemp("fx") / "spherepatch.step"
    shape = BRepPrimAPI_MakeSphere(
        30.0, math.radians(20), math.radians(70), math.radians(80)
    ).Shape()
    write_step(shape, p)
    return p


@pytest.fixture(scope="session")
def full_cyl_step(tmp_path_factory):
    # full cylinder: lateral face is closed in U -> must be rejected by flattening
    p = tmp_path_factory.mktemp("fx") / "fullcyl.step"
    write_step(BRepPrimAPI_MakeCylinder(30.0, 50.0).Shape(), p)
    return p


@pytest.fixture(scope="session")
def part1_path():
    return REPO / "testdata" / "part1.stp"


@pytest.fixture(scope="session")
def part2_path():
    return REPO / "testdata" / "part2.stp"


@pytest.fixture(scope="session")
def cruscotto_full_path():
    p = REPO / "testdata" / "cruscotto_full.stp"
    if not p.exists():
        pytest.skip("cruscotto_full.stp not present")
    return p
