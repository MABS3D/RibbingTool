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


_FACETED_HEADER = """ISO-10303-21;
HEADER;
FILE_DESCRIPTION((''),'2;1');
FILE_NAME('{name}','2026-01-01T00:00:00',(''),(''),'RibbingTool','','');
FILE_SCHEMA(('AUTOMOTIVE_DESIGN {{ 1 0 10303 214 1 1 1 1 }}'));
ENDSEC;
DATA;
#1=APPLICATION_CONTEXT('core data for automotive mechanical design processes');
#2=APPLICATION_PROTOCOL_DEFINITION('international standard','automotive_design',2000,#1);
#3=PRODUCT_CONTEXT('',#1,'mechanical');
#4=PRODUCT('{name}','{name}','',(#3));
#5=PRODUCT_DEFINITION_FORMATION('','',#4);
#6=PRODUCT_DEFINITION_CONTEXT('part definition',#1,'design');
#7=PRODUCT_DEFINITION('design','',#5,#6);
#8=PRODUCT_DEFINITION_SHAPE('','',#7);
#9=(GEOMETRIC_REPRESENTATION_CONTEXT(3)GLOBAL_UNCERTAINTY_ASSIGNED_CONTEXT((#13))GLOBAL_UNIT_ASSIGNED_CONTEXT((#10,#11,#12))REPRESENTATION_CONTEXT('',''));
#10=(LENGTH_UNIT()NAMED_UNIT(*)SI_UNIT(.MILLI.,.METRE.));
#11=(NAMED_UNIT(*)PLANE_ANGLE_UNIT()SI_UNIT($,.RADIAN.));
#12=(NAMED_UNIT(*)SI_UNIT($,.STERADIAN.)SOLID_ANGLE_UNIT());
#13=UNCERTAINTY_MEASURE_WITH_UNIT(LENGTH_MEASURE(0.001),#10,'distance_accuracy_value','');
#14=SHAPE_DEFINITION_REPRESENTATION(#8,#15);
#16=AXIS2_PLACEMENT_3D('',#18,$,$);
#18=CARTESIAN_POINT('',(0.,0.,0.));
"""


def write_faceted_step(path, vertices=None, triangles=None, name="ribbed",
                       shells=None):
    """Write triangle mesh(es) as FACETED_BREP STEP (pure text, no OCCT).

    Avoids the prohibitive cost of sewing large meshes back into B-rep just
    to serialize them. Pass one mesh via (vertices, triangles) or several
    closed shells via shells=[(v, t), ...] — each becomes its own
    FACETED_BREP solid in the representation.
    """
    import numpy as np
    if shells is None:
        shells = [(vertices, triangles)]

    lines = [_FACETED_HEADER.format(name=name)]
    nid = 20
    brep_ids = []
    for vertices_s, triangles_s in shells:
        # geometry is serialized at 5 decimals — filter degeneracy on the
        # ROUNDED coords, or rounding re-creates the slivers we filtered
        v = np.round(np.asarray(vertices_s, float), 5)
        t = np.asarray(triangles_s)
        e1 = v[t[:, 1]] - v[t[:, 0]]
        e2 = v[t[:, 2]] - v[t[:, 0]]
        e3 = v[t[:, 2]] - v[t[:, 1]]
        nrm = np.cross(e1, e2)
        nlen = np.linalg.norm(nrm, axis=1)
        ok = ((nlen > 1e-6)
              & (np.linalg.norm(e1, axis=1) > 1e-4)
              & (np.linalg.norm(e2, axis=1) > 1e-4)
              & (np.linalg.norm(e3, axis=1) > 1e-4))
        t, e1, nrm, nlen = t[ok], e1[ok], nrm[ok], nlen[ok]
        if not len(t):
            continue
        nrm = nrm / nlen[:, None]
        ref = e1 / np.linalg.norm(e1, axis=1, keepdims=True).clip(1e-12)

        pt_ids = np.empty(len(v), dtype=np.int64)
        for i, (x, y, z) in enumerate(v):
            pt_ids[i] = nid
            lines.append(
                f"#{nid}=CARTESIAN_POINT('',({x:.5f},{y:.5f},{z:.5f}));\n")
            nid += 1
        face_ids = []
        for k, (a, b, c) in enumerate(t):
            n, r = nrm[k], ref[k]
            loop, bound = nid, nid + 1
            dn, dr, ax, pl, face = nid + 2, nid + 3, nid + 4, nid + 5, nid + 6
            nid += 7
            lines.append(
                f"#{loop}=POLY_LOOP('',(#{pt_ids[a]},#{pt_ids[b]},#{pt_ids[c]}));\n"
                f"#{bound}=FACE_OUTER_BOUND('',#{loop},.T.);\n"
                f"#{dn}=DIRECTION('',({n[0]:.7f},{n[1]:.7f},{n[2]:.7f}));\n"
                f"#{dr}=DIRECTION('',({r[0]:.7f},{r[1]:.7f},{r[2]:.7f}));\n"
                f"#{ax}=AXIS2_PLACEMENT_3D('',#{pt_ids[a]},#{dn},#{dr});\n"
                f"#{pl}=PLANE('',#{ax});\n"
                f"#{face}=FACE_SURFACE('',(#{bound}),#{pl},.T.);\n")
            face_ids.append(face)
        # STEP physical files dislike very long lines — wrap the face list
        refs = [f"#{i}" for i in face_ids]
        chunks = [",".join(refs[i:i + 16]) for i in range(0, len(refs), 16)]
        shell_id, brep_id = nid, nid + 1
        nid += 2
        lines.append(f"#{shell_id}=CLOSED_SHELL('',("
                     + ",\n".join(chunks) + "));\n")
        lines.append(f"#{brep_id}=FACETED_BREP('',#{shell_id});\n")
        brep_ids.append(brep_id)

    items = ",".join(["#16"] + [f"#{i}" for i in brep_ids])
    lines.append(f"#15=FACETED_BREP_SHAPE_REPRESENTATION('',({items}),#9);\n")
    lines.append("ENDSEC;\nEND-ISO-10303-21;\n")
    Path(path).write_text("".join(lines), encoding="ascii")
