from pathlib import Path

from OCP.STEPControl import (
    STEPControl_Reader,
    STEPControl_StepModelType,
    STEPControl_Writer,
)
from OCP.IFSelect import IFSelect_RetDone
from OCP.TopAbs import TopAbs_FACE
from OCP.TopExp import TopExp
from OCP.TopTools import TopTools_IndexedMapOfShape
from OCP.TopoDS import TopoDS
from OCP.BRepGProp import BRepGProp
from OCP.GProp import GProp_GProps


class StepError(Exception):
    pass


def load_step(path):
    path = Path(path)
    if not path.exists():
        raise StepError(f"file not found: {path}")
    reader = STEPControl_Reader()
    if reader.ReadFile(str(path)) != IFSelect_RetDone:
        raise StepError(f"failed to parse STEP: {path.name}")
    reader.TransferRoots()
    shape = reader.OneShape()
    if shape.IsNull():
        raise StepError(f"no shape in STEP: {path.name}")
    return shape


def save_step(shape, path):
    writer = STEPControl_Writer()
    writer.Transfer(shape, STEPControl_StepModelType.STEPControl_AsIs)
    if writer.Write(str(path)) != IFSelect_RetDone:
        raise StepError(f"failed to write STEP: {path}")


def face_map(shape):
    m = TopTools_IndexedMapOfShape()
    TopExp.MapShapes_s(shape, TopAbs_FACE, m)
    return m


def get_face(shape, face_id):
    m = face_map(shape)
    if not 1 <= face_id <= m.Size():
        raise StepError(f"face id {face_id} out of range 1..{m.Size()}")
    return TopoDS.Face_s(m.FindKey(face_id))


def shape_volume(shape):
    p = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, p)
    return p.Mass()
