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
def cyl_patch_step(tmp_path_factory):
    # 120-degree cylinder sector: lateral face is a curved topological disk
    p = tmp_path_factory.mktemp("fx") / "cylpatch.step"
    write_step(BRepPrimAPI_MakeCylinder(30.0, 50.0, math.radians(120)).Shape(), p)
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
