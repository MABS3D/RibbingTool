# RibbingTool Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Local web CAD tool: open STEP → click-select faces in 3D → apply parametric rib patterns (works on freeform curved faces) → export STEP/STL.

**Architecture:** FastAPI backend holds an OCCT shape stack (undo). Per selected face: OCCT triangulation (with UVs) → LSCM flatten to 2D → generate pattern segment network (shapely/scipy) → map capsule footprints back through barycentric→UV→exact surface eval → loft polygon-wire solids (ruled ThruSections) between `surface−embed` and `surface+height` → fuse ribs, fuse with body. Frontend is a static three.js page with per-face meshes for picking.

**Tech Stack:** Python 3.13 `.venv` (uv), cadquery-ocp 7.9.3 (OCCT), libigl 2.6 (LSCM), shapely 2.1, scipy, FastAPI/uvicorn, manifold3d (fallback), three.js (vendored), pytest.

## Global Constraints

- Units: mm and degrees everywhere (STEP files are mm).
- Python: run everything with `.venv\Scripts\python.exe` from repo root `C:\Users\Chello\Desktop\RibbingTool`.
- Face IDs = 1-based indices from `TopExp.MapShapes_s(shape, TopAbs_FACE, map)` — stable per shape, invalidated after every apply (frontend reloads full mesh set each time).
- OCP static methods carry `_s` suffix (e.g. `BRep_Tool.Surface_s`).
- All geometry failures must surface in API responses; never silent. Per-segment loft failures skip-and-count.
- Rib solids: polygon wires + ruled ThruSections (robust booleans) — smooth B-spline sides are explicitly out of scope for v1.
- Commit after every task (conventional commits).

---

### Task 1: Test scaffolding + STEP fixtures

**Files:**
- Create: `server/__init__.py`, `server/geometry/__init__.py` (empty)
- Create: `pytest.ini`
- Create: `tests/__init__.py` (empty), `tests/conftest.py`
- Test: `tests/test_fixtures.py`

**Interfaces:**
- Produces: pytest fixtures `box_step`, `cyl_patch_step`, `sphere_patch_step`, `full_cyl_step` (each → `pathlib.Path` to a generated STEP file); `part1_path`, `part2_path` (cruscotto files); helper `write_step(shape, path)` and `volume(shape)` in conftest importable by later tests.

- [ ] **Step 1: Write pytest.ini**

```ini
[pytest]
testpaths = tests
markers =
    slow: long-running acceptance tests (deselect with -m "not slow")
addopts = -m "not slow"
```

- [ ] **Step 2: Write conftest.py**

```python
import pathlib
import pytest

from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder, BRepPrimAPI_MakeSphere
from OCP.STEPControl import STEPControl_Writer, STEPControl_StepModelType
from OCP.IFSelect import IFSelect_RetDone
from OCP.BRepGProp import BRepGProp
from OCP.GProp import GProp_GProps
import math

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
    write_step(
        BRepPrimAPI_MakeSphere(30.0, math.radians(20), math.radians(70), math.radians(80)).Shape(), p)
    return p


@pytest.fixture(scope="session")
def full_cyl_step(tmp_path_factory):
    # full cylinder: lateral face is CLOSED in U -> must be rejected by flattening
    p = tmp_path_factory.mktemp("fx") / "fullcyl.step"
    write_step(BRepPrimAPI_MakeCylinder(30.0, 50.0).Shape(), p)
    return p


@pytest.fixture(scope="session")
def part1_path():
    return REPO / "testdata" / "part1.stp"


@pytest.fixture(scope="session")
def part2_path():
    return REPO / "testdata" / "part2.stp"
```

- [ ] **Step 3: Write failing test**

```python
# tests/test_fixtures.py
def test_fixtures_exist(box_step, cyl_patch_step, sphere_patch_step, full_cyl_step, part1_path):
    for p in (box_step, cyl_patch_step, sphere_patch_step, full_cyl_step, part1_path):
        assert p.exists() and p.stat().st_size > 0
```

- [ ] **Step 4: Run** `.venv\Scripts\python.exe -m pytest tests/test_fixtures.py -v` → PASS (fixtures generate; nothing to implement — this task's "failure mode" is import errors).
- [ ] **Step 5: Commit** `git add -A && git commit -m "test: scaffolding + STEP fixtures"`

---

### Task 2: STEP I/O (`step_io.py`)

**Files:**
- Create: `server/geometry/step_io.py`
- Test: `tests/test_step_io.py`

**Interfaces:**
- Produces:
  - `load_step(path: str|Path) -> TopoDS_Shape` (raises `StepError` on failure)
  - `save_step(shape, path) -> None`
  - `face_map(shape) -> TopTools_IndexedMapOfShape` (1-based; `.Size()`, `.FindKey(i)`)
  - `get_face(shape, face_id: int) -> TopoDS_Face`
  - `shape_volume(shape) -> float`
  - `class StepError(Exception)`

- [ ] **Step 1: Failing test**

```python
# tests/test_step_io.py
import pytest
from server.geometry.step_io import load_step, save_step, face_map, get_face, shape_volume, StepError


def test_load_box(box_step):
    s = load_step(box_step)
    assert face_map(s).Size() == 6
    assert abs(shape_volume(s) - 60 * 40 * 8) < 1e-6


def test_roundtrip(box_step, tmp_path):
    s = load_step(box_step)
    out = tmp_path / "rt.step"
    save_step(s, out)
    s2 = load_step(out)
    assert abs(shape_volume(s2) - shape_volume(s)) < 1e-3


def test_get_face(box_step):
    s = load_step(box_step)
    f = get_face(s, 1)
    assert not f.IsNull()
    with pytest.raises(StepError):
        get_face(s, 99)


def test_load_missing():
    with pytest.raises(StepError):
        load_step("nope.step")
```

- [ ] **Step 2: Run** → FAIL (module missing)
- [ ] **Step 3: Implement**

```python
# server/geometry/step_io.py
from pathlib import Path

from OCP.STEPControl import STEPControl_Reader, STEPControl_Writer, STEPControl_StepModelType
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
```

- [ ] **Step 4: Run** → PASS
- [ ] **Step 5: Commit** `feat: STEP I/O`

---

### Task 3: Per-face meshing with UVs (`meshing.py`)

**Files:**
- Create: `server/geometry/meshing.py`
- Test: `tests/test_meshing.py`

**Interfaces:**
- Produces:
  - `@dataclass FaceMesh: face_id:int; vertices:np.ndarray(n,3) f64; triangles:np.ndarray(m,3) i32 (CCW outward); uvs:np.ndarray(n,2); is_planar:bool; surface_kind:str; area:float`
  - `mesh_shape(shape, lin_defl=0.5, ang_defl=0.5) -> list[FaceMesh]` (one per face id, ascending; faces with no triangulation are skipped with a log)
  - `face_mesh(face, face_id) -> FaceMesh | None`

- [ ] **Step 1: Failing test**

```python
# tests/test_meshing.py
import numpy as np
from server.geometry.step_io import load_step
from server.geometry.meshing import mesh_shape


def test_box_meshes(box_step):
    ms = mesh_shape(load_step(box_step))
    assert len(ms) == 6
    assert all(m.is_planar for m in ms)
    for m in ms:
        assert m.vertices.shape[1] == 3 and m.triangles.shape[1] == 3
        assert m.uvs.shape == (len(m.vertices), 2)
        assert m.triangles.max() < len(m.vertices)
    areas = sorted(round(m.area) for m in ms)
    assert areas == sorted([60*40, 60*40, 60*8, 60*8, 40*8, 40*8])


def test_cyl_patch_curved(cyl_patch_step):
    ms = mesh_shape(load_step(cyl_patch_step))
    kinds = {m.surface_kind for m in ms}
    assert "cylinder" in kinds
    curved = [m for m in ms if m.surface_kind == "cylinder"]
    assert all(not m.is_planar for m in curved)
```

- [ ] **Step 2: Run** → FAIL
- [ ] **Step 3: Implement**

```python
# server/geometry/meshing.py
from dataclasses import dataclass
import numpy as np

from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.BRep import BRep_Tool
from OCP.TopLoc import TopLoc_Location
from OCP.TopAbs import TopAbs_REVERSED
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.GeomAbs import GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone, GeomAbs_Sphere, GeomAbs_Torus, GeomAbs_BSplineSurface, GeomAbs_BezierSurface
from OCP.BRepGProp import BRepGProp
from OCP.GProp import GProp_GProps

from .step_io import face_map
from OCP.TopoDS import TopoDS

_KIND = {GeomAbs_Plane: "plane", GeomAbs_Cylinder: "cylinder", GeomAbs_Cone: "cone",
         GeomAbs_Sphere: "sphere", GeomAbs_Torus: "torus",
         GeomAbs_BSplineSurface: "bspline", GeomAbs_BezierSurface: "bezier"}


@dataclass
class FaceMesh:
    face_id: int
    vertices: np.ndarray
    triangles: np.ndarray
    uvs: np.ndarray
    is_planar: bool
    surface_kind: str
    area: float


def face_mesh(face, face_id):
    loc = TopLoc_Location()
    tri = BRep_Tool.Triangulation_s(face, loc)
    if tri is None:
        return None
    trsf = loc.Transformation()
    n = tri.NbNodes()
    verts = np.empty((n, 3))
    uvs = np.zeros((n, 2))
    has_uv = tri.HasUVNodes()
    for i in range(1, n + 1):
        p = tri.Node(i).Transformed(trsf)
        verts[i - 1] = (p.X(), p.Y(), p.Z())
        if has_uv:
            q = tri.UVNode(i)
            uvs[i - 1] = (q.X(), q.Y())
    m = tri.NbTriangles()
    tris = np.empty((m, 3), dtype=np.int32)
    rev = face.Orientation() == TopAbs_REVERSED
    for i in range(1, m + 1):
        a, b, c = tri.Triangle(i).Get()
        tris[i - 1] = (a - 1, c - 1, b - 1) if rev else (a - 1, b - 1, c - 1)
    ad = BRepAdaptor_Surface(face)
    kind = _KIND.get(ad.GetType(), "other")
    gp = GProp_GProps()
    BRepGProp.SurfaceProperties_s(face, gp)
    return FaceMesh(face_id, verts, tris, uvs, kind == "plane", kind, gp.Mass())


def mesh_shape(shape, lin_defl=0.5, ang_defl=0.5):
    BRepMesh_IncrementalMesh(shape, lin_defl, False, ang_defl, True)
    fm = face_map(shape)
    out = []
    for fid in range(1, fm.Size() + 1):
        r = face_mesh(TopoDS.Face_s(fm.FindKey(fid)), fid)
        if r is not None:
            out.append(r)
    return out
```

- [ ] **Step 4: Run** → PASS
- [ ] **Step 5: Commit** `feat: per-face meshing with UVs`

---

### Task 4: LSCM flattening (`flatten.py`)

**Files:**
- Create: `server/geometry/flatten.py`
- Test: `tests/test_flatten.py`

**Interfaces:**
- Produces:
  - `flatten(mesh: FaceMesh) -> np.ndarray(n,2)` — flattened coords, rescaled so 2D area == 3D area, translated to centroid origin. Raises `FlattenError` (message mentions "closed" or "not a disk") for closed/genus>0 faces.
  - `class FlattenError(Exception)`
  - `boundary_loops(triangles) -> list[list[int]]` — vertex-index loops of all boundaries.

- [ ] **Step 1: Failing test**

```python
# tests/test_flatten.py
import numpy as np
import pytest
from server.geometry.step_io import load_step
from server.geometry.meshing import mesh_shape
from server.geometry.flatten import flatten, FlattenError


def _dist_ratio(mesh, flat):
    e = np.vstack([mesh.triangles[:, [0, 1]], mesh.triangles[:, [1, 2]], mesh.triangles[:, [2, 0]]])
    d3 = np.linalg.norm(mesh.vertices[e[:, 0]] - mesh.vertices[e[:, 1]], axis=1)
    d2 = np.linalg.norm(flat[e[:, 0]] - flat[e[:, 1]], axis=1)
    keep = d3 > 1e-9
    return d2[keep] / d3[keep]


def test_planar_is_isometric(box_step):
    m = max(mesh_shape(load_step(box_step)), key=lambda x: x.area)
    r = _dist_ratio(m, flatten(m))
    assert np.allclose(r, 1.0, atol=1e-6)


def test_cylinder_patch_low_distortion(cyl_patch_step):
    ms = mesh_shape(load_step(cyl_patch_step))
    m = next(x for x in ms if x.surface_kind == "cylinder")
    r = _dist_ratio(m, flatten(m))
    assert r.mean() == pytest.approx(1.0, abs=0.05)
    assert r.max() < 1.3 and r.min() > 0.7


def test_sphere_patch_flattens(sphere_patch_step):
    ms = mesh_shape(load_step(sphere_patch_step))
    m = next(x for x in ms if x.surface_kind == "sphere")
    flat = flatten(m)
    assert np.isfinite(flat).all()


def test_closed_face_rejected(full_cyl_step):
    ms = mesh_shape(load_step(full_cyl_step))
    m = next(x for x in ms if x.surface_kind == "cylinder")
    with pytest.raises(FlattenError):
        flatten(m)
```

- [ ] **Step 2: Run** → FAIL
- [ ] **Step 3: Implement**

```python
# server/geometry/flatten.py
import numpy as np
import igl


class FlattenError(Exception):
    pass


def boundary_loops(triangles):
    edges = {}
    for t in triangles:
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            k = (min(a, b), max(a, b))
            edges[k] = edges.get(k, 0) + 1
    bnd = [k for k, c in edges.items() if c == 1]
    nxt = {}
    for t in triangles:
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            if (min(a, b), max(a, b)) in dict.fromkeys(bnd) and (min(a, b), max(a, b)) in set(bnd):
                pass
    bset = set(bnd)
    for t in triangles:
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            if (min(a, b), max(a, b)) in bset:
                nxt[int(a)] = int(b)   # boundary edges keep face winding
    loops, seen = [], set()
    for start in list(nxt):
        if start in seen:
            continue
        loop, v = [], start
        while v not in seen:
            seen.add(v)
            loop.append(v)
            v = nxt.get(v)
            if v is None:
                break
        if loop:
            loops.append(loop)
    return loops


def _tri_areas(v, f):
    a = v[f[:, 1]] - v[f[:, 0]]
    b = v[f[:, 2]] - v[f[:, 0]]
    if v.shape[1] == 3:
        return 0.5 * np.linalg.norm(np.cross(a, b), axis=1)
    return 0.5 * np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])


def flatten(mesh):
    v = np.ascontiguousarray(mesh.vertices, dtype=np.float64)
    f = np.ascontiguousarray(mesh.triangles, dtype=np.int64)
    loops = boundary_loops(f)
    if not loops:
        raise FlattenError(
            "face is a closed surface (not a topological disk) — split it in your CAD first")
    nv, ne = len(v), len({(min(a, b), max(a, b)) for t in f for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0]))})
    euler = nv - ne + len(f)
    if euler != 2 - len(loops):   # genus-0 with b boundaries has euler 2-b... using chi = 2-2g-b
        raise FlattenError("face topology unsupported (genus > 0) — split it in your CAD first")
    outer = max(loops, key=len)
    # pin the two most distant vertices of the outer loop (approx: max dist from first)
    p0 = outer[0]
    d = np.linalg.norm(v[outer] - v[p0], axis=1)
    p1 = outer[int(np.argmax(d))]
    b = np.array([p0, p1], dtype=np.int64)
    bc = np.array([[0.0, 0.0], [float(d.max()), 0.0]])
    res = igl.lscm(v, f, b, bc)
    uv = res[1] if isinstance(res, tuple) else res
    if uv is None or not np.isfinite(uv).all() or abs(_tri_areas(uv, f).sum()) < 1e-12:
        raise FlattenError("LSCM flattening failed for this face")
    scale = np.sqrt(_tri_areas(v, f).sum() / abs(_tri_areas(uv, f).sum()))
    uv = uv * scale
    return uv - uv.mean(axis=0)
```

Note: clean up the dead loop block in `boundary_loops` when implementing (artifact of drafting — final code keeps only the `bset` pass).

- [ ] **Step 4: Run** → PASS (verify the closed-cylinder rejection specifically)
- [ ] **Step 5: Commit** `feat: LSCM face flattening with disk-topology guard`

---

### Task 5: Pattern generators (`patterns.py`)

**Files:**
- Create: `server/geometry/patterns.py`
- Test: `tests/test_patterns.py`

**Interfaces:**
- Produces:
  - `@dataclass RibParams: pattern:str; spacing:float=12; spacing_y:float|None=None; thickness:float=1.6; height:float=4; orientation_deg:float=0; draft_deg:float=0; margin:float=2; embed:float=0.3; border:bool=False; density:float=0.008; seed:int=1; base_angle_deg:float=0` (+ `from_dict(d)` classmethod ignoring unknown keys)
  - `generate_segments(params, bounds) -> list[tuple[(x0,y0),(x1,y1)]]` — infinite-pattern segments covering `bounds=(minx,miny,maxx,maxy)`, already rotated by `orientation_deg`.
  - `clip_and_border(segments, boundary: shapely.Polygon, params) -> list[LineString]` — clip to `boundary.buffer(-margin)`, append border ring edges if `params.border`.
  - `capsule(p0, p1, half_width, step=2.5, cap_pts=5) -> np.ndarray(k,2)` — CCW closed sample loop (not repeated last point), sides sampled every `step` mm; same k for any half_width given same segment.
  - Patterns: `rectangular` (spacing, spacing_y or same; spacing_y=0 → single family), `quadmesh` (= equal spacing), `triangular` (three families at base_angle+0/60/120), `isogrid` (= triangular), `hexagonal` (honeycomb walls, spacing = across-flats), `stochastic` (Voronoi edges of seeded points, ~density pts/mm²).

- [ ] **Step 1: Failing test**

```python
# tests/test_patterns.py
import numpy as np
from shapely.geometry import Polygon
from server.geometry.patterns import RibParams, generate_segments, clip_and_border, capsule

B = (0.0, 0.0, 100.0, 50.0)
SQUARE = Polygon([(0, 0), (100, 0), (100, 50), (0, 50)])


def test_rectangular_counts():
    segs = generate_segments(RibParams(pattern="rectangular", spacing=10), B)
    # ~11 vertical + ~6 horizontal lines covering a 100x50 box (edges inclusive)
    assert 14 <= len(segs) <= 22


def test_single_family():
    p = RibParams(pattern="rectangular", spacing=10, spacing_y=0)
    segs = generate_segments(p, B)
    assert 9 <= len(segs) <= 13
    d = [np.arctan2(s[1][1]-s[0][1], s[1][0]-s[0][0]) for s in segs]
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
    inset = SQUARE.buffer(-2.99)
    assert lines and all(inset.buffer(0.01).contains(l) for l in lines)


def test_border_ring():
    p = RibParams(pattern="rectangular", spacing=1000, margin=3, border=True)
    lines = clip_and_border(generate_segments(p, B), SQUARE, p)
    assert len(lines) >= 4


def test_capsule_same_count_for_draft():
    a = capsule((0, 0), (30, 0), 1.0)
    b = capsule((0, 0), (30, 0), 0.6)
    assert a.shape == b.shape and a.shape[1] == 2
    assert len(a) >= 10

import pytest  # noqa: E402  (used above)
```

- [ ] **Step 2: Run** → FAIL
- [ ] **Step 3: Implement**

```python
# server/geometry/patterns.py
from dataclasses import dataclass, fields
import math
import numpy as np
from shapely.geometry import LineString, MultiLineString, Polygon, box as shp_box
from shapely.affinity import rotate as shp_rotate
from scipy.spatial import Voronoi


@dataclass
class RibParams:
    pattern: str = "rectangular"
    spacing: float = 12.0
    spacing_y: float | None = None
    thickness: float = 1.6
    height: float = 4.0
    orientation_deg: float = 0.0
    draft_deg: float = 0.0
    margin: float = 2.0
    embed: float = 0.3
    border: bool = False
    density: float = 0.008
    seed: int = 1
    base_angle_deg: float = 0.0

    @classmethod
    def from_dict(cls, d):
        keys = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in keys})


def _expand(bounds, f=0.3):
    minx, miny, maxx, maxy = bounds
    dx, dy = (maxx - minx) * f, (maxy - miny) * f
    return minx - dx, miny - dy, maxx + dx, maxy + dy


def _family(bounds, spacing, angle_deg):
    """Parallel lines with given angle, covering bounds."""
    minx, miny, maxx, maxy = bounds
    cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
    r = 0.5 * math.hypot(maxx - minx, maxy - miny) + spacing
    a = math.radians(angle_deg)
    d = np.array([math.cos(a), math.sin(a)])      # line direction
    n = np.array([-d[1], d[0]])                   # normal
    segs = []
    k0, k1 = int(math.floor(-r / spacing)), int(math.ceil(r / spacing))
    for k in range(k0, k1 + 1):
        c = np.array([cx, cy]) + n * (k * spacing)
        segs.append((tuple(c - d * r), tuple(c + d * r)))
    return segs


def _hex_walls(bounds, size):
    """Honeycomb cell walls; size = across-flats distance."""
    minx, miny, maxx, maxy = _expand(bounds)
    s = size / math.sqrt(3.0)                     # hex edge length
    dx, dy = size, 1.5 * s
    segs = []
    j0, j1 = int(miny // dy) - 2, int(maxy // dy) + 2
    i0, i1 = int(minx // dx) - 2, int(maxx // dx) + 2
    for j in range(j0, j1 + 1):
        for i in range(i0, i1 + 1):
            cx = i * dx + (dx / 2 if j % 2 else 0.0)
            cy = j * dy
            pts = [(cx + (size / 2) * math.cos(a), cy + (size / 2) / math.cos(math.pi / 6) * math.sin(a))
                   for a in (math.radians(30 + 60 * k) for k in range(6))]
            # use exact hexagon: radius = s (edge length), flat-top orientation
            pts = [(cx + s * math.cos(math.radians(30 + 60 * k)),
                    cy + s * math.sin(math.radians(30 + 60 * k))) for k in range(6)]
            for k in range(6):
                a, b = pts[k], pts[(k + 1) % 6]
                if a <= b:                        # dedupe shared walls
                    segs.append((a, b))
    return segs


def _stochastic(bounds, density, seed):
    minx, miny, maxx, maxy = _expand(bounds, 0.15)
    area = (maxx - minx) * (maxy - miny)
    n = max(8, int(area * density))
    rng = np.random.default_rng(seed)
    pts = rng.uniform([minx, miny], [maxx, maxy], size=(n, 2))
    vor = Voronoi(pts)
    segs = []
    for a, b in vor.ridge_vertices:
        if a >= 0 and b >= 0:
            pa, pb = vor.vertices[a], vor.vertices[b]
            if np.isfinite(pa).all() and np.isfinite(pb).all():
                segs.append((tuple(pa), tuple(pb)))
    return segs


def generate_segments(params, bounds):
    p, sp = params.pattern, params.spacing
    base = params.base_angle_deg
    if p in ("rectangular", "quadmesh"):
        sy = sp if p == "quadmesh" else (params.spacing_y if params.spacing_y is not None else sp)
        segs = _family(bounds, sp, 90.0)
        if sy and sy > 0:
            segs += _family(bounds, sy, 0.0)
    elif p in ("triangular", "isogrid"):
        segs = (_family(bounds, sp, base) + _family(bounds, sp, base + 60)
                + _family(bounds, sp, base + 120))
    elif p == "hexagonal":
        segs = _hex_walls(bounds, sp)
    elif p == "stochastic":
        segs = _stochastic(bounds, params.density, params.seed)
    else:
        raise ValueError(f"unknown pattern: {p}")
    if params.orientation_deg:
        cx = (bounds[0] + bounds[2]) / 2
        cy = (bounds[1] + bounds[3]) / 2
        ml = shp_rotate(MultiLineString([LineString(s) for s in segs]),
                        params.orientation_deg, origin=(cx, cy))
        segs = [(tuple(g.coords[0]), tuple(g.coords[-1])) for g in ml.geoms]
    return segs


def clip_and_border(segments, boundary, params):
    inset = boundary.buffer(-params.margin)
    if inset.is_empty:
        return []
    out = []
    ml = MultiLineString([LineString(s) for s in segments])
    inter = inset.intersection(ml)
    geoms = getattr(inter, "geoms", [inter] if not inter.is_empty else [])
    for g in geoms:
        if isinstance(g, LineString) and g.length > max(0.5, params.thickness * 0.5):
            out.append(g)
    if params.border:
        polys = getattr(inset, "geoms", [inset])
        for poly in polys:
            for ring in [poly.exterior, *poly.interiors]:
                c = list(ring.coords)
                for i in range(len(c) - 1):
                    seg = LineString([c[i], c[i + 1]])
                    if seg.length > 0.05:
                        out.append(seg)
    return out


def capsule(p0, p1, half_width, step=2.5, cap_pts=5):
    p0 = np.asarray(p0, float)
    p1 = np.asarray(p1, float)
    d = p1 - p0
    L = np.linalg.norm(d)
    if L < 1e-9:
        d, L = np.array([1.0, 0.0]), 1e-9
    d = d / L
    n = np.array([-d[1], d[0]])
    ns = max(2, int(math.ceil(L / step)) + 1)
    ts = np.linspace(0.0, 1.0, ns)
    side1 = [tuple(p0 + d * (t * L) + n * half_width) for t in ts]
    side2 = [tuple(p1 - d * (t * L) - n * half_width) for t in ts]
    cap1 = [tuple(p1 + half_width * (math.cos(a) * n * -1 + math.sin(a) * d) * -1) for a in []]
    # semicircle caps, sampled cap_pts each (excluding endpoints already in sides)
    def cap(center, start_ang):
        return [tuple(center + half_width * np.array([math.cos(start_ang + k * math.pi / (cap_pts + 1)),
                                                      math.sin(start_ang + k * math.pi / (cap_pts + 1))]))
                for k in range(1, cap_pts + 1)]
    base = math.atan2(n[1], n[0])
    loop = side1 + cap(p1, base - math.pi) [::-1] + side2 + cap(p0, base)[::-1]
    # NOTE: implementer — verify winding CCW via shoelace and fix sign; drop the unused cap1 line.
    arr = np.array(loop)
    area2 = np.sum(arr[:, 0] * np.roll(arr[:, 1], -1) - np.roll(arr[:, 0], -1) * arr[:, 1])
    if area2 < 0:
        arr = arr[::-1]
    return arr
```

The capsule cap math above is sketched; implementer must make caps join sides continuously (side1 ends at p1+n·w, cap must run from angle(n) to angle(−n) around p1 passing through +d side). Verify with the test plus a quick shapely `Polygon(arr).is_valid` assert added to the test.

- [ ] **Step 4: Run** → PASS (add `Polygon(capsule(...)).is_valid` assertion)
- [ ] **Step 5: Commit** `feat: 2D rib pattern generators`

---

### Task 6: Rib construction + booleans (`ribbing.py`, `booleans.py`)

**Files:**
- Create: `server/geometry/ribbing.py`, `server/geometry/booleans.py`
- Test: `tests/test_ribbing.py`

**Interfaces:**
- Consumes: everything above.
- Produces:
  - `booleans.fuse_list(solids: list) -> TopoDS_Shape` (raises `BooleanError`)
  - `booleans.fuse(a, b) -> TopoDS_Shape`
  - `@dataclass RibReport: face_id:int; segments:int; lofted:int; skipped:int; warnings:list[str]`
  - `ribbing.apply_ribs(shape, face_ids: list[int], params: RibParams, lin_defl=0.4) -> tuple[TopoDS_Shape, list[RibReport]]` — full per-face pipeline; raises `RibbingError` with human message when nothing can be built.
  - `class RibbingError(Exception)`; `class BooleanError(Exception)`

Implementation notes (exact algorithm):
1. For each face: `face_mesh` (fresh triangulation at `lin_defl`), `flatten` → `flat (n,2)`.
2. Boundary polygon in flat space from `boundary_loops` (outer = largest |area| loop, holes = rest).
3. `generate_segments` over flat bounds → `clip_and_border` → LineStrings; split each LineString into straight sub-segments (consecutive coord pairs).
4. Per sub-segment: `capsule(p0,p1, t/2)` bottom loop, `capsule(p0,p1, max(t/2 − h·tan(draft), t*0.05))` top loop (same sample counts by construction).
5. Map 2D loop points to 3D: shapely `STRtree` of flat triangles → containing triangle (nearest fallback, clamp barycentric to [0,1] and renormalize) → barycentric blend of the triangle's 3 vertex UVs → `GeomLProp_SLProps(surf, u, v, 1, 1e-6)` point + normal; flip normal if face REVERSED; verify outward once per face via `BRepClass3d_SolidClassifier` on `p − n·(embed·2)` expecting `TopAbs_IN` (else flip all normals).
6. bottom pts = `P − n·embed`, top pts = `P + n·height`. Skip segment (count it) if any mapping failed or loops self-intersect (`Polygon(loop).is_valid` false) or normals vary > 85° within one segment (fold risk).
7. Wires: `BRepBuilderAPI_MakePolygon`, `Close()`; loft `BRepOffsetAPI_ThruSections(True, True, 1e-6)`; `CheckCompatibility(False)`. Wrap per-segment in try/except → skip+count.
8. `fuse_list(rib_solids)` → ribs; `fuse(body, ribs)` → out. Fuzzy 1e-4, parallel on. Validate with `BRepCheck_Analyzer(out).IsValid()`; if invalid → warning (not error). Volume must increase; if not, `RibbingError`.
9. Warning when `height > 0.8 × min curvature radius` sampled at segment midpoints (`props.MaxCurvature()` guarded).

```python
# server/geometry/booleans.py
from OCP.BRepAlgoAPI import BRepAlgoAPI_Fuse
from OCP.TopTools import TopTools_ListOfShape


class BooleanError(Exception):
    pass


def _fuse_args(args, tools):
    op = BRepAlgoAPI_Fuse()
    la, lt = TopTools_ListOfShape(), TopTools_ListOfShape()
    for s in args:
        la.Append(s)
    for s in tools:
        lt.Append(s)
    op.SetArguments(la)
    op.SetTools(lt)
    op.SetFuzzyValue(1e-4)
    op.SetRunParallel(True)
    op.Build()
    if not op.IsDone():
        raise BooleanError("boolean fuse failed")
    return op.Shape()


def fuse_list(solids):
    if not solids:
        raise BooleanError("nothing to fuse")
    if len(solids) == 1:
        return solids[0]
    return _fuse_args(solids[:1], solids[1:])


def fuse(a, b):
    return _fuse_args([a], [b])
```

Key ribbing.py skeleton (implementer fills the mapping/loft privates following the notes):

```python
# server/geometry/ribbing.py
from dataclasses import dataclass, field
import math
import numpy as np
from shapely.geometry import Polygon, LineString
from shapely.strtree import STRtree

from OCP.BRep import BRep_Tool
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakePolygon
from OCP.BRepOffsetAPI import BRepOffsetAPI_ThruSections
from OCP.BRepClass3d import BRepClass3d_SolidClassifier
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.GeomLProp import GeomLProp_SLProps
from OCP.gp import gp_Pnt
from OCP.TopAbs import TopAbs_REVERSED, TopAbs_IN

from .step_io import get_face, shape_volume
from .meshing import face_mesh
from .flatten import flatten, boundary_loops, FlattenError
from .patterns import RibParams, generate_segments, clip_and_border, capsule
from .booleans import fuse_list, fuse, BooleanError


class RibbingError(Exception):
    pass


@dataclass
class RibReport:
    face_id: int
    segments: int = 0
    lofted: int = 0
    skipped: int = 0
    warnings: list = field(default_factory=list)


MAX_SEGMENTS = 4000


def apply_ribs(shape, face_ids, params, lin_defl=0.4):
    all_ribs, reports = [], []
    for fid in face_ids:
        rep = RibReport(face_id=fid)
        reports.append(rep)
        face = get_face(shape, fid)
        mesh = face_mesh(face, fid)          # after fresh BRepMesh on the face
        ...
    if not all_ribs:
        raise RibbingError("no ribs could be built: " + "; ".join(
            w for r in reports for w in r.warnings) or "no ribs could be built")
    ribs = fuse_list(all_ribs)
    out = fuse(shape, ribs)
    v0, v1 = shape_volume(shape), shape_volume(out)
    if v1 <= v0 + 1e-6:
        raise RibbingError("ribbing produced no volume increase — fuse likely failed")
    if not BRepCheck_Analyzer(out).IsValid():
        for r in reports:
            r.warnings.append("result failed BRepCheck (may still export fine)")
    return out, reports
```

(`...` above is the per-face pipeline exactly as the numbered notes; the implementer writes it in helpers `_flat_boundary(mesh, flat)`, `_mapper(mesh, flat, face, body)` returning `map_pts(pts2d, offset)->np.ndarray|None`, `_loft(bottom, top)->solid`.)

- [ ] **Step 1: Failing tests**

```python
# tests/test_ribbing.py
import pytest
from OCP.BRepCheck import BRepCheck_Analyzer
from server.geometry.step_io import load_step, shape_volume, save_step, face_map
from server.geometry.meshing import mesh_shape
from server.geometry.patterns import RibParams
from server.geometry.ribbing import apply_ribs, RibbingError


def biggest_face_id(shape, kind=None):
    ms = mesh_shape(load_step_cache := shape) if False else None
    from server.geometry.meshing import mesh_shape as _ms
    meshes = _ms(shape)
    if kind:
        meshes = [m for m in meshes if m.surface_kind == kind]
    return max(meshes, key=lambda m: m.area).face_id


@pytest.mark.parametrize("pattern", ["rectangular", "triangular", "hexagonal", "stochastic"])
def test_rib_box_top(box_step, pattern):
    s = load_step(box_step)
    fid = biggest_face_id(s)
    p = RibParams(pattern=pattern, spacing=10, thickness=1.6, height=4, margin=2, seed=3, density=0.01)
    out, reports = apply_ribs(s, [fid], p)
    assert shape_volume(out) > shape_volume(s)
    assert reports[0].lofted > 0
    assert BRepCheck_Analyzer(out).IsValid()


def test_rib_with_draft_and_border(box_step):
    s = load_step(box_step)
    fid = biggest_face_id(s)
    p = RibParams(pattern="quadmesh", spacing=12, height=5, draft_deg=3, border=True)
    out, reports = apply_ribs(s, [fid], p)
    assert shape_volume(out) > shape_volume(s)


def test_rib_cylinder_patch(cyl_patch_step):
    s = load_step(cyl_patch_step)
    fid = biggest_face_id(s, kind="cylinder")
    p = RibParams(pattern="isogrid", spacing=12, thickness=1.6, height=3)
    out, reports = apply_ribs(s, [fid], p)
    assert shape_volume(out) > shape_volume(s)
    assert reports[0].skipped < reports[0].segments * 0.2


def test_rib_sphere_patch(sphere_patch_step):
    s = load_step(sphere_patch_step)
    fid = biggest_face_id(s, kind="sphere")
    p = RibParams(pattern="hexagonal", spacing=10, height=2.5, thickness=1.2)
    out, reports = apply_ribs(s, [fid], p)
    assert shape_volume(out) > shape_volume(s)


def test_closed_face_error(full_cyl_step):
    s = load_step(full_cyl_step)
    fid = biggest_face_id(s, kind="cylinder")
    with pytest.raises(RibbingError, match="disk|closed|split"):
        apply_ribs(s, [fid], RibParams())


def test_roundtrip_after_rib(box_step, tmp_path):
    s = load_step(box_step)
    out, _ = apply_ribs(s, [biggest_face_id(s)], RibParams(pattern="rectangular"))
    p = tmp_path / "ribbed.step"
    save_step(out, p)
    s2 = load_step(p)
    assert abs(shape_volume(s2) - shape_volume(out)) / shape_volume(out) < 1e-3
```

(Fix the sloppy `biggest_face_id` helper when writing the real file: it takes `shape`, meshes it, returns largest-area face id, optional kind filter.)

- [ ] **Step 2: Run** → FAIL
- [ ] **Step 3: Implement** per notes. FlattenError inside `apply_ribs` for a face → convert to `RibbingError` carrying the message (so the closed-face test passes).
- [ ] **Step 4: Run** → PASS. Also rerun the whole suite.
- [ ] **Step 5: Commit** `feat: rib loft pipeline + fuse`

---

### Task 7: Manifold fallback (`booleans.py` extension)

**Files:**
- Modify: `server/geometry/booleans.py`
- Test: `tests/test_fallback.py`

**Interfaces:**
- Produces: `mesh_fallback_fuse(body_shape, rib_solids, lin_defl=0.3) -> TopoDS_Shape` — meshes everything, welds vertices, unions via manifold3d, rebuilds a sewn faceted OCCT solid. Raises `BooleanError` if non-manifold. `ribbing.apply_ribs` gains `allow_fallback=True` param: on `BooleanError` from OCCT path, retries via fallback and appends warning `"OCCT fuse failed — output is faceted (mesh boolean fallback)"`.

- [ ] **Step 1: Failing test**

```python
# tests/test_fallback.py
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
from server.geometry.booleans import mesh_fallback_fuse
from server.geometry.step_io import shape_volume


def test_mesh_fallback_boxes():
    a = BRepPrimAPI_MakeBox(20, 20, 10).Shape()
    b = BRepPrimAPI_MakeBox(10, 10, 30).Shape()   # overlaps a
    out = mesh_fallback_fuse(a, [b])
    v = shape_volume(out)
    assert abs(v - (20*20*10 + 10*10*20)) / v < 0.02
```

- [ ] **Step 2: Run** → FAIL
- [ ] **Step 3: Implement**

```python
# append to server/geometry/booleans.py
import numpy as np


def _shape_to_mesh(shape, lin_defl):
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.BRep import BRep_Tool
    from OCP.TopLoc import TopLoc_Location
    from OCP.TopExp import TopExp
    from OCP.TopTools import TopTools_IndexedMapOfShape
    from OCP.TopAbs import TopAbs_FACE, TopAbs_REVERSED
    from OCP.TopoDS import TopoDS
    BRepMesh_IncrementalMesh(shape, lin_defl, False, 0.4, True)
    fm = TopTools_IndexedMapOfShape()
    TopExp.MapShapes_s(shape, TopAbs_FACE, fm)
    V, F = [], []
    for i in range(1, fm.Size() + 1):
        face = TopoDS.Face_s(fm.FindKey(i))
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation_s(face, loc)
        if tri is None:
            continue
        t0 = len(V)
        tr = loc.Transformation()
        for k in range(1, tri.NbNodes() + 1):
            p = tri.Node(k).Transformed(tr)
            V.append((p.X(), p.Y(), p.Z()))
        rev = face.Orientation() == TopAbs_REVERSED
        for k in range(1, tri.NbTriangles() + 1):
            a, b, c = tri.Triangle(k).Get()
            F.append((t0+a-1, t0+c-1, t0+b-1) if rev else (t0+a-1, t0+b-1, t0+c-1))
    return np.array(V, np.float32), np.array(F, np.uint32)


def _weld(v, f, tol=1e-3):
    key = np.round(v / tol).astype(np.int64)
    _, idx, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    return v[idx], inv[f].astype(np.uint32)


def _to_manifold(shape, lin_defl):
    import manifold3d as m3d
    v, f = _weld(*_shape_to_mesh(shape, lin_defl))
    mesh = m3d.Mesh(vert_properties=v, tri_verts=f)
    man = m3d.Manifold(mesh)
    if man.is_empty():
        raise BooleanError("mesh is not manifold — fallback impossible")
    return man


def mesh_fallback_fuse(body_shape, rib_solids, lin_defl=0.3):
    import manifold3d as m3d
    man = _to_manifold(body_shape, lin_defl)
    for s in rib_solids:
        man = man + _to_manifold(s, lin_defl)
    mesh = man.to_mesh()
    return _mesh_to_shape(np.asarray(mesh.vert_properties, float),
                          np.asarray(mesh.tri_verts, np.int64))


def _mesh_to_shape(v, f):
    from OCP.BRepBuilderAPI import (BRepBuilderAPI_MakePolygon, BRepBuilderAPI_MakeFace,
                                    BRepBuilderAPI_Sewing, BRepBuilderAPI_MakeSolid)
    from OCP.gp import gp_Pnt
    from OCP.TopoDS import TopoDS
    from OCP.TopAbs import TopAbs_SHELL
    from OCP.TopExp import TopExp_Explorer
    sew = BRepBuilderAPI_Sewing(1e-3)
    for (a, b, c) in f:
        poly = BRepBuilderAPI_MakePolygon(gp_Pnt(*v[a]), gp_Pnt(*v[b]), gp_Pnt(*v[c]), True)
        try:
            sew.Add(BRepBuilderAPI_MakeFace(poly.Wire()).Face())
        except Exception:
            continue
    sew.Perform()
    shell_shape = sew.SewedShape()
    ex = TopExp_Explorer(shell_shape, TopAbs_SHELL)
    if not ex.More():
        raise BooleanError("sewing produced no shell")
    mk = BRepBuilderAPI_MakeSolid()
    while ex.More():
        mk.Add(TopoDS.Shell_s(ex.Current()))
        ex.Next()
    return mk.Solid()
```

Check the actual manifold3d 3.5 Python API names at implementation time (`Mesh` vs `MeshGL`, `to_mesh()` naming) with a quick `dir()` probe; adjust accordingly.

- [ ] **Step 4: Run** → PASS
- [ ] **Step 5: Commit** `feat: manifold3d mesh-boolean fallback`

---

### Task 8: FastAPI app (`main.py`)

**Files:**
- Create: `server/main.py`
- Test: `tests/test_api.py`

**Interfaces:**
- Produces (all JSON unless noted):
  - `POST /api/load` multipart file → `{filename, faces:[{id, kind, planar, area, positions:[...f32 xyz], indices:[...], }], volume, bbox:[minx..maxz], nfaces}`
  - `POST /api/ribs` `{face_ids:[int], params:{...RibParams fields}}` → same mesh payload + `{reports:[{face_id, segments, lofted, skipped, warnings}], volume}`; 400 with `{detail}` on RibbingError/FlattenError
  - `POST /api/undo` → mesh payload (400 if stack empty)
  - `GET /api/export/step` → STEP file download of current shape
  - `GET /api/export/stl` → binary STL download
  - `GET /` → `web/index.html`; `/web/*` static mount
  - Module-level `STATE = {"stack": [], "filename": None, "meshes": None}`

- [ ] **Step 1: Failing test**

```python
# tests/test_api.py
from fastapi.testclient import TestClient
from server.main import app, STATE


def _load(client, step_path):
    with open(step_path, "rb") as f:
        r = client.post("/api/load", files={"file": ("box.step", f, "application/step")})
    assert r.status_code == 200, r.text
    return r.json()


def test_load_and_mesh(box_step):
    c = TestClient(app)
    data = _load(c, box_step)
    assert data["nfaces"] == 6
    assert len(data["faces"]) == 6
    f0 = data["faces"][0]
    assert len(f0["positions"]) % 9 == 0 or (f0["indices"] and len(f0["positions"]) % 3 == 0)


def test_ribs_undo_export(box_step):
    c = TestClient(app)
    d0 = _load(c, box_step)
    big = max(d0["faces"], key=lambda f: f["area"])["id"]
    r = c.post("/api/ribs", json={"face_ids": [big], "params": {"pattern": "quadmesh", "spacing": 12}})
    assert r.status_code == 200, r.text
    d1 = r.json()
    assert d1["volume"] > d0["volume"]
    assert d1["reports"][0]["lofted"] > 0
    r = c.get("/api/export/step")
    assert r.status_code == 200 and len(r.content) > 1000
    r = c.post("/api/undo")
    assert r.status_code == 200
    assert abs(r.json()["volume"] - d0["volume"]) < 1e-6


def test_bad_face(box_step):
    c = TestClient(app)
    _load(c, box_step)
    r = c.post("/api/ribs", json={"face_ids": [999], "params": {}})
    assert r.status_code == 400
```

- [ ] **Step 2: Run** → FAIL
- [ ] **Step 3: Implement**

```python
# server/main.py
import io
import struct
import tempfile
from pathlib import Path

import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .geometry.step_io import load_step, save_step, shape_volume, StepError
from .geometry.meshing import mesh_shape
from .geometry.patterns import RibParams
from .geometry.ribbing import apply_ribs, RibbingError
from .geometry.flatten import FlattenError
from .geometry.booleans import BooleanError

app = FastAPI(title="RibbingTool")
WEB = Path(__file__).resolve().parents[1] / "web"
STATE = {"stack": [], "filename": None, "meshes": None}


class RibsRequest(BaseModel):
    face_ids: list[int]
    params: dict = {}


def _mesh_payload():
    shape = STATE["stack"][-1]
    meshes = mesh_shape(shape, 0.5, 0.5)
    STATE["meshes"] = meshes
    allv = np.vstack([m.vertices for m in meshes]) if meshes else np.zeros((1, 3))
    faces = []
    for m in meshes:
        faces.append({
            "id": m.face_id,
            "kind": m.surface_kind,
            "planar": m.is_planar,
            "area": float(m.area),
            "positions": np.round(m.vertices, 4).astype(float).ravel().tolist(),
            "indices": m.triangles.ravel().tolist(),
        })
    return {
        "filename": STATE["filename"],
        "nfaces": len(faces),
        "faces": faces,
        "volume": float(shape_volume(shape)),
        "bbox": [*allv.min(0).tolist(), *allv.max(0).tolist()],
    }


@app.post("/api/load")
async def api_load(file: UploadFile = File(...)):
    data = await file.read()
    with tempfile.NamedTemporaryFile(suffix=".step", delete=False) as tf:
        tf.write(data)
        tmp = tf.name
    try:
        shape = load_step(tmp)
    except StepError as e:
        raise HTTPException(400, str(e))
    finally:
        Path(tmp).unlink(missing_ok=True)
    STATE["stack"] = [shape]
    STATE["filename"] = file.filename
    return _mesh_payload()


@app.post("/api/ribs")
def api_ribs(req: RibsRequest):
    if not STATE["stack"]:
        raise HTTPException(400, "no model loaded")
    params = RibParams.from_dict(req.params)
    try:
        out, reports = apply_ribs(STATE["stack"][-1], req.face_ids, params)
    except (RibbingError, FlattenError, BooleanError, StepError) as e:
        raise HTTPException(400, str(e))
    STATE["stack"].append(out)
    if len(STATE["stack"]) > 10:
        STATE["stack"] = STATE["stack"][:1] + STATE["stack"][-9:]
    payload = _mesh_payload()
    payload["reports"] = [vars(r) for r in reports]
    return payload


@app.post("/api/undo")
def api_undo():
    if len(STATE["stack"]) < 2:
        raise HTTPException(400, "nothing to undo")
    STATE["stack"].pop()
    return _mesh_payload()


@app.get("/api/export/step")
def api_export_step():
    if not STATE["stack"]:
        raise HTTPException(400, "no model loaded")
    out = Path(tempfile.gettempdir()) / "ribbingtool_export.step"
    save_step(STATE["stack"][-1], out)
    name = (STATE["filename"] or "model").rsplit(".", 1)[0] + "_ribbed.step"
    return FileResponse(out, filename=name, media_type="application/step")


@app.get("/api/export/stl")
def api_export_stl():
    if not STATE["stack"]:
        raise HTTPException(400, "no model loaded")
    meshes = STATE["meshes"] or mesh_shape(STATE["stack"][-1], 0.2, 0.3)
    buf = io.BytesIO()
    buf.write(b"\0" * 80)
    ntri = sum(len(m.triangles) for m in meshes)
    buf.write(struct.pack("<I", ntri))
    for m in meshes:
        v, t = m.vertices, m.triangles
        for (a, b, c) in t:
            n = np.cross(v[b] - v[a], v[c] - v[a])
            ln = np.linalg.norm(n)
            n = n / ln if ln > 0 else n
            buf.write(struct.pack("<3f", *n))
            for p in (v[a], v[b], v[c]):
                buf.write(struct.pack("<3f", *p))
            buf.write(b"\0\0")
    name = (STATE["filename"] or "model").rsplit(".", 1)[0] + "_ribbed.stl"
    return Response(buf.getvalue(), media_type="model/stl",
                    headers={"Content-Disposition": f'attachment; filename="{name}"'})


@app.get("/", response_class=HTMLResponse)
def index():
    return (WEB / "index.html").read_text(encoding="utf-8")


app.mount("/web", StaticFiles(directory=str(WEB)), name="web")
```

(Static mount requires `web/` to exist — Task 9 creates it; for this task create `web/index.html` containing `<!doctype html><title>RibbingTool</title>placeholder`.)

- [ ] **Step 4: Run** → PASS (also install `httpx` for TestClient: `uv pip install --python .venv httpx` and add to a `requirements.txt` capturing all deps)
- [ ] **Step 5: Commit** `feat: FastAPI backend`

---

### Task 9: Frontend (three.js viewer + panel)

**Files:**
- Create: `web/index.html`, `web/app.js`, `web/viewer.js`
- Vendor: `web/vendor/three.module.js`, `web/vendor/OrbitControls.js`
- Test: manual browser drive (Claude drives the in-app browser); plus `tests/test_static.py` asserting `GET /` returns HTML containing `app.js`.

**Steps:**

- [ ] **Step 1: Vendor three.js**

```bash
cd /c/Users/Chello/Desktop/RibbingTool
npm init -y && npm i three@0.170.0
mkdir -p web/vendor
cp node_modules/three/build/three.module.js web/vendor/
cp node_modules/three/examples/jsm/controls/OrbitControls.js web/vendor/
```

Then edit `web/vendor/OrbitControls.js`: replace `from 'three'` with `from './three.module.js'`. Add `node_modules/` and `package*.json` to `.gitignore` (vendored copies are committed instead).

- [ ] **Step 2: index.html** — dark UI, left 3D canvas, right panel: file input, pattern `<select>` (rectangular/quadmesh/triangular/isogrid/hexagonal/stochastic), numeric inputs (spacing, spacing_y, thickness, height, orientation, draft, margin, density, seed), border checkbox, selected-faces readout, Apply / Undo / Export STEP / Export STL buttons, status banner div, busy overlay. All ids referenced by app.js (`file-input, pattern, spacing, spacing-y, thickness, height, orientation, draft, margin, density, seed, border, sel-info, btn-apply, btn-undo, btn-step, btn-stl, banner, busy`).

- [ ] **Step 3: viewer.js** — exports `initViewer(canvas)`, `loadModel(data)` (one `THREE.Mesh` per face, `MeshStandardMaterial({color:0x8a8f98, metalness:.1, roughness:.75})`), `onPick(cb)` raycast click → toggles face selection (selected → orange `0xff8c2f`, hovered → lighten emissive), `getSelection() -> int[]`, `clearSelection()`, camera fit to bbox, OrbitControls, hemisphere+directional lights, resize handling.

- [ ] **Step 4: app.js** — wires panel: upload posts to `/api/load`, renders; Apply gathers params → `/api/ribs` with `getSelection()`; banner shows `detail` on 400 and per-face reports (`lofted/skipped/warnings`) on success; busy overlay during fetches; Undo/Export wired; selection info line updates on pick.

- [ ] **Step 5: Static test + run server**

```python
# tests/test_static.py
from fastapi.testclient import TestClient
from server.main import app

def test_index_served():
    r = TestClient(app).get("/")
    assert r.status_code == 200 and "app.js" in r.text
```

Create `.claude/launch.json` entry `ribbingtool` → `.venv\Scripts\python.exe -m uvicorn server.main:app --port 8317`, then drive the browser: load `testdata/part1.stp`, click the external skin face, apply hexagonal ribs, verify visually + export.

- [ ] **Step 6: Commit** `feat: three.js frontend`

---

### Task 10: Cruscotto acceptance tests (slow)

**Files:**
- Create: `tests/test_acceptance.py`

- [ ] **Step 1: Test**

```python
# tests/test_acceptance.py
import pytest
from server.geometry.step_io import load_step, save_step, shape_volume
from server.geometry.meshing import mesh_shape
from server.geometry.patterns import RibParams
from server.geometry.ribbing import apply_ribs


@pytest.mark.slow
@pytest.mark.parametrize("path_fx", ["part1_path", "part2_path"])
def test_cruscotto_external_skin(path_fx, request, tmp_path):
    s = load_step(request.getfixturevalue(path_fx))
    meshes = mesh_shape(s)
    target = max(meshes, key=lambda m: m.area)      # external skin = largest face (verify in UI)
    p = RibParams(pattern="hexagonal", spacing=14, thickness=1.6, height=3.0, margin=3)
    out, reports = apply_ribs(s, [target.face_id], p)
    assert shape_volume(out) > shape_volume(s)
    r = reports[0]
    assert r.lofted > 10 and r.skipped <= r.segments * 0.25
    save_step(out, tmp_path / "ribbed.step")
    s2 = load_step(tmp_path / "ribbed.step")
    assert abs(shape_volume(s2) - shape_volume(out)) / shape_volume(out) < 1e-3
```

- [ ] **Step 2: Run** `.venv\Scripts\python.exe -m pytest tests/test_acceptance.py -m slow -v` → PASS (expect minutes; if the largest face is not the white-highlighted skin, adjust selection to match what the UI shows and note the actual face id in the test).
- [ ] **Step 3: Commit** `test: cruscotto acceptance`

---

### Task 11: Polish + README

**Files:**
- Create: `README.md` (what it is, `uv venv` + `uv pip install -r requirements.txt` setup, run command, usage walkthrough, parameter reference table, known limits from spec)
- Create: `requirements.txt` (all deps pinned)
- Modify: banner copy for fallback/faceted warning if not already surfaced.

- [ ] **Step 1:** Full suite run `.venv\Scripts\python.exe -m pytest -v` → all PASS; then slow suite once.
- [ ] **Step 2:** README written with real commands.
- [ ] **Step 3: Commit** `docs: README + pins`
