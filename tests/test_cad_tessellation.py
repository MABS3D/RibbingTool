"""CAD recovery preserves contours, surface identity and closed body topology."""
import manifold3d as m3d
import numpy as np
import pytest
from OCP.BRep import BRep_Builder, BRep_Tool
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder, BRepPrimAPI_MakeSphere
from OCP.TopLoc import TopLoc_Location
from OCP.TopoDS import TopoDS, TopoDS_Compound
from OCP.gp import gp_Ax1, gp_Dir, gp_Pnt, gp_Trsf, gp_Vec

from server.geometry import cad_tessellation as ct
from server.geometry.booleans import _to_manifold, _weld
from server.geometry.meshing import face_mesh
from server.geometry.step_io import face_map, shape_volume


def closed_mesh(shape):
    fm = face_map(shape); vertices, faces, offset = [], [], 0
    for fid in range(1, fm.Size()+1):
        mesh = face_mesh(TopoDS.Face_s(fm.FindKey(fid)), fid)
        assert mesh is not None
        vertices.append(mesh.vertices); faces.append(mesh.triangles+offset)
        offset += len(mesh.vertices)
    v, f = _weld(np.vstack(vertices), np.vstack(faces))
    solid = m3d.Manifold(m3d.Mesh64(np.ascontiguousarray(v), np.ascontiguousarray(f, np.uint64)))
    assert solid.status() == m3d.Error.NoError
    return solid


@pytest.mark.parametrize('moved', [False, True])
@pytest.mark.parametrize('partial', [False, True])
def test_cylinder_recovery_reuses_adjacent_contours_and_periodic_seam(moved, partial):
    shape = BRepPrimAPI_MakeCylinder(6.5, 6.).Shape()
    if moved:
        tr = gp_Trsf(); tr.SetRotation(gp_Ax1(gp_Pnt(), gp_Dir(1, 1, 0)), .73)
        tr.SetTranslationPart(gp_Vec(24, -13, 8))
        shape.Move(TopLoc_Location(tr))
    fm, volume = face_map(shape), shape_volume(shape)
    BRepMesh_IncrementalMesh(shape, .1, False, .15, True)
    face = TopoDS.Face_s(fm.FindKey(1)); loc = TopLoc_Location()
    original = BRep_Tool.Triangulation_s(face, loc)
    if partial:
        original.ResizeTriangles(1, True)
    else:
        BRep_Builder().UpdateFace(face, None)
    report = ct.repair_tessellation(shape, .1, .15)
    assert report == {'recovered': [1], 'failed': {}}
    assert all(fm.FindKey(i).IsSame(face_map(shape).FindKey(i)) for i in range(1, fm.Size()+1))
    assert shape_volume(shape) == pytest.approx(volume, rel=1e-12)
    solid = closed_mesh(shape)
    assert len(solid.decompose()) == 1
    assert solid.volume() == pytest.approx(volume, rel=.002)
    mesh = face_mesh(face, 1)
    # Periodic UVs remain a full strip; no seam is folded onto one branch.
    assert np.ptp(mesh.uvs[:, 0]) == pytest.approx(2*np.pi)
    assert ct.repair_tessellation(shape, .1, .15) == {'recovered': [], 'failed': {}}


def test_dense_triangulation_with_an_interior_hole_cannot_pass_coverage():
    shape = BRepPrimAPI_MakeSphere(10.).Shape()
    BRepMesh_IncrementalMesh(shape, .1, False, .15, True)
    face = TopoDS.Face_s(face_map(shape).FindKey(1)); loc = TopLoc_Location()
    tri = BRep_Tool.Triangulation_s(face, loc)
    assert ct._complete_boundary(face, tri, loc)
    f = ct._triangles(tri)
    from collections import Counter
    edges = Counter(tuple(sorted((a, b))) for t in f for a, b in zip(t, np.roll(t, -1)))
    index = next(i for i, t in enumerate(f) if all(edges[tuple(sorted((a, b)))] == 2
                                                  for a, b in zip(t, np.roll(t, -1))))
    tri.SetTriangle(index+1, tri.Triangle(tri.NbTriangles()))
    tri.ResizeTriangles(tri.NbTriangles()-1, True)
    assert tri.NbTriangles() >= tri.NbNodes()-2
    assert not ct._complete_boundary(face, tri, loc)
    report = ct.repair_tessellation(shape, .1, .15)
    assert not report['recovered'] and 1 in report['failed']
    assert BRep_Tool.Triangulation_s(face, TopLoc_Location()) is None


def test_recovery_preserves_an_inner_wire_instead_of_filling_the_hole():
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
    from OCP.gp import gp_Ax2
    box = BRepPrimAPI_MakeBox(18., 12., 4.).Shape()
    cutter = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(9, 6, 0), gp_Dir(0, 0, 1)), 2., 4.).Shape()
    shape = BRepAlgoAPI_Cut(box, cutter).Shape()
    BRepMesh_IncrementalMesh(shape, .1, False, .15, True)
    fm = face_map(shape)
    fid = next(i for i in range(1, fm.Size()+1)
               if np.allclose(face_mesh(TopoDS.Face_s(fm.FindKey(i)), i).vertices[:, 2], 4.))
    BRep_Builder().UpdateFace(TopoDS.Face_s(fm.FindKey(fid)), None)
    assert ct.repair_tessellation(shape, .1, .15) == {'recovered': [fid], 'failed': {}}
    result = closed_mesh(shape)
    assert len(result.decompose()) == 1
    assert result.volume() == pytest.approx(shape_volume(shape), rel=.002)


@pytest.mark.parametrize('gap,components', [(0., 1), (2., 2)])
def test_assembly_conversion_unions_solids_without_welding_contact_faces(gap, components):
    a = BRepPrimAPI_MakeBox(18., 12., 4.).Shape()
    b = BRepPrimAPI_MakeBox(gp_Pnt(18.+gap, 0, 0), 18., 12., 4.).Shape()
    compound = TopoDS_Compound(); builder = BRep_Builder(); builder.MakeCompound(compound)
    builder.Add(compound, a); builder.Add(compound, b)
    result = _to_manifold(compound, .2)
    assert result.status() == m3d.Error.NoError
    assert len(result.decompose()) == components
    assert result.volume() == pytest.approx(2*18*12*4)
