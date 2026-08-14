"""Smoke test: load the cruscotto STEP files, count solids/faces, time it."""
import sys
import time

from OCP.STEPControl import STEPControl_Reader
from OCP.IFSelect import IFSelect_RetDone
from OCP.TopExp import TopExp_Explorer
from OCP.TopAbs import TopAbs_FACE, TopAbs_SOLID
from OCP.BRepGProp import BRepGProp
from OCP.GProp import GProp_GProps


def inspect(path: str) -> None:
    t0 = time.perf_counter()
    reader = STEPControl_Reader()
    status = reader.ReadFile(path)
    if status != IFSelect_RetDone:
        print(f"{path}: READ FAILED ({status})")
        return
    reader.TransferRoots()
    shape = reader.OneShape()
    t1 = time.perf_counter()

    n_solids = 0
    ex = TopExp_Explorer(shape, TopAbs_SOLID)
    while ex.More():
        n_solids += 1
        ex.Next()

    n_faces = 0
    ex = TopExp_Explorer(shape, TopAbs_FACE)
    while ex.More():
        n_faces += 1
        ex.Next()

    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, props)
    vol = props.Mass()

    print(f"{path}: {n_solids} solid(s), {n_faces} faces, "
          f"volume={vol:.1f} mm^3, load={t1 - t0:.1f}s")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        inspect(p)
