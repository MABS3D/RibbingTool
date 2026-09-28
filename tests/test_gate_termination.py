"""Capped termination at the facing gate (Zone A: gate shards).

The facing gate f = max(f, 25*(0.09 - nzc)) is BINARY at raw nzc =
0.09: ribs stand at full height right up to a 3D cut surface that
wanders with the facet-pitch wobble of the interpolated pseudo-normal
(+-0.045 in nz near 85deg at ang_defl 0.09), slicing ribs mid-body and
littering the 81-85deg band of every wide dashboard roll with
partial-height torn fragments and gate-orphaned crown scraps.  The
termination contract asserted here, method-agnostically:

  (a) band emptiness: standing rib material (signed s > 0.4 above the
      substrate) must not exist where the top-sheet facing at its foot
      is deep inside the gate band (nzg_foot < 0.13) — every rib and
      web must complete in a clean capped end before the smoothed
      cut region, as the round-4 comment always promised ("full
      material by nz~0.15, gone below 0.09");
  (b) no partial-height shards: in the terminal fringe (nzg_foot <
      0.15) every connected piece of standing material must reach
      crest height (max s >= 0.7*height) — full ribs and webs crossing
      the band pass, torn tatters do not.  Anti-vacuity guard: the
      >=55deg region must still hold real rib material (webs/steep
      ribs survive — the trim must not become a slope cull);
  (c) no debris in the gate band and watertight output (transplanted
      from test_runout_quality).

Fixture: dome_roll_step — a sphere-R=60 mushroom cap emerging through
97deg from a flat box top, projected along +z.  The spec draft named
wrap_cyl_step, but measured on the current engine that fixture is dead
for this defect: its silhouette IS the projected domain rim, so the
default margin-2 inset plus the B-setback end all material at ~68deg
(measured raw cp-facing min 0.373) — the gate never touches anything.
The dome keeps the steep band INTERIOR (flat continues beyond the
contact circle, B large), which is the real defect geometry (interior
S-rolls and transition walls).

Classification is by the TOP-SHEET facing raster NZg — the
pseudo-normal sampled just above each column's top surface depth
(Dhi + 2 cells), the sheet the projection actually parameterizes.  The
engine's smoothed NZr raster is sampled at MID-band depth: in fold-over
overlap bands that query snaps to silhouette/under sheets and reads ~0
across whole bands of legitimate 45-70deg ribs (measured -0.01 on
wrap_cyl, and 0.275-vs-true-0.0 blindness on this dome), so it can
classify nothing here — the same measurement that forced the engine's
gate-cap raster onto the top-sheet field.

Both standing-material metrics carry a z >= 24 window (the defect
band 82.5-87deg lives at z 30-36 on this dome; the excluded z < 24
shell is the dome/flat contact zone, where the crease blend
legitimately bridges flat ribs under the overhang rim and the
classifier cannot attribute: those verts' closest points snap to the
rim, so their cp-feet read nzg ~0.03-0.13 although the material is
flat-rib — measured post-cap as the entire residual population, all
with engine trim coordinate bg < 0 at their feet, i.e. kept by the
partner branch, z p100 = 23.53).

Calibrated on the pre-cap engine (2026-08-27, gusset round landed):
(a) measured 15103 standing verts with nzg_foot < 0.13 at z >= 24
(threshold 1325) — FAIL; (b) measured 10 partial-height shard
components (max-s 0.59..2.06 < 2.1) among 51 fringe components at
nzg < 0.15, z >= 24 — FAIL (at nzg < 0.13 every pre-cap component
still reaches crest height: the [0.13, 0.15) shell is what
disconnects the diagonal gate-cut remnants); anti-vacuity measured
113762 (PASS, both before and after: 110488 post); (c) watertight, 1
cluster, 0 band debris (PASS, both).  Post-cap: (a) 0, (b) fringe
subgraph empty.
"""
import numpy as np
import pytest

from server.geometry import implicit as imp
from server.geometry.implicit import (_bilinear, _surface_eval,
                                      build_rib_implicit)
from server.geometry.patterns import RibParams
from server.geometry.step_io import load_step

HEIGHT = 3.0
Z_CREASE = 24.0             # below: dome/flat contact-crease zone (excluded)


def _params():
    return RibParams(pattern="isogrid", spacing=12, thickness=1.6,
                     height=HEIGHT, taper_len=0, mapping="project")


@pytest.fixture(scope="module")
def gatecap(dome_roll_step):
    """One dome build + captured rasters + per-vertex classification."""
    s = load_step(dome_roll_step)
    from server.geometry.meshing import mesh_shape
    fids = []
    for m in mesh_shape(s, 0.4, 0.3):
        v = np.asarray(m.vertices)
        if not len(v):
            continue
        if m.is_planar and np.allclose(v[:, 2], 20.0, atol=1e-6):
            fids.append(m.face_id)          # flat top around the dome
        elif not m.is_planar and v[:, 2].max() > 25.0:
            fids.append(m.face_id)          # dome cap
    assert len(fids) >= 2, f"dome fixture faces not found: {fids}"

    cap = {}
    omf = imp.mesh_field
    ord_ = imp._rasterize_depth_range

    def spy_mf(surface, params, resolution, tile=None, reports=None,
               extractor=None):
        cap["g"] = surface
        cap["res"] = resolution
        return omf(surface, params, resolution, tile, reports, extractor)

    def spy_rd(*a, **k):
        out = ord_(*a, **k)
        cap["Dhi"] = out[1]         # infilled in place by the engine
        return out

    imp.mesh_field = spy_mf
    imp._rasterize_depth_range = spy_rd
    try:
        clusters, _ = build_rib_implicit(s, fids, _params())
    finally:
        imp.mesh_field = omf
        imp._rasterize_depth_range = ord_
    g = cap["g"]

    # top-sheet facing raster: pseudo-normal just above each column's
    # top surface depth — for near-vertical sheets the query point sits
    # ~on the sheet (distance 2*cell*nz), so the closest point stays on
    # the sheet the projection parameterizes, fold-under sheets never
    # answer, and infilled beyond-silhouette columns read the rim
    sh = g.dlo.shape
    gu = g.origin[0] + np.arange(sh[0]) * g.cell
    gv = g.origin[1] + np.arange(sh[1]) * g.cell
    GU, GV = np.meshgrid(gu, gv, indexing="ij")
    Xt = np.column_stack([GU.ravel(), GV.ravel(),
                          (cap["Dhi"] + 2.0 * g.cell).ravel()])
    del GU, GV
    _, _, Nt = _surface_eval(Xt, g.V, g.F, g.N)
    NZg = Nt[:, 2].reshape(sh).astype(np.float32)
    del Xt, Nt

    v = np.vstack([c[0] for c in clusters])
    f = np.vstack([c[1] + off for c, off in
                   zip(clusters, np.cumsum([0] + [len(c[0]) for c in
                                                  clusters[:-1]]))])
    X = np.ascontiguousarray((v - g.ctr) @ g.M)          # world -> frame
    s_v, _, nrm = _surface_eval(X, g.V, g.F, g.N)
    foot = X - s_v[:, None] * nrm
    nzg = _bilinear(NZg, foot[:, 0], foot[:, 1], g.cell, g.origin)
    return {"clusters": clusters, "v": v, "f": f, "s": s_v,
            "nzg": nzg, "nzc": nrm[:, 2], "res": cap["res"]}


def test_gate_band_empty_of_standing_material(gatecap):
    """(a) No standing rib material deep inside the gate band.

    Pre-cap: 15103 violations (full-height blades and torn fragments
    stand right down to the raw 0.09 cut).  Post-cap: every wall ends
    in a rounded in-plane nose with clearance before the 0.15 contour,
    so feet stay at nzg >= ~0.17; 0.13 leaves margin for bilinear and
    query noise.  The sub-surface slab is exempt by the signed s > 0.4
    filter — the trim is wall-only and the slab legitimately keeps its
    extent down to the raw backstop."""
    v, s, nzg = gatecap["v"], gatecap["s"], gatecap["nzg"]
    bad = (s > 0.4) & (nzg < 0.13) & (v[:, 2] >= Z_CREASE)
    lim = max(20, int(0.001 * len(v)))
    assert int(bad.sum()) < lim, (
        f"{int(bad.sum())} standing rib verts (s > 0.4) on gate-band "
        f"surface (top-sheet facing at foot < 0.13; limit {lim}) — the "
        f"facing gate is slicing ribs mid-body instead of capping them "
        f"before the band")


def test_no_partial_height_shards_in_fringe(gatecap):
    """(b) Every standing piece in the terminal fringe reaches crest
    height; the >=55deg band still holds material (anti-vacuity)."""
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components
    v, f, s = gatecap["v"], gatecap["f"], gatecap["s"]
    nzg, nzc = gatecap["nzg"], gatecap["nzc"]
    # anti-vacuity FIRST: webs/steep ribs must survive on the 55-78deg
    # band — the executable form of "the trim is not a slope cull"
    keep = (s > 0.4) & (nzc > 0.2) & (nzc < 0.5)
    assert int(keep.sum()) > 100, (
        f"only {int(keep.sum())} standing verts on the 55-78deg band — "
        f"steep ribs/webs vanished, the termination became a slope cull")
    sel = (s > 0.4) & (nzg < 0.15) & (v[:, 2] >= Z_CREASE)
    fs = f[sel[f].all(axis=1)]
    if not len(fs):
        return                       # fringe empty: trivially shard-free
    adj = sp.csr_matrix(
        (np.ones(len(fs) * 3, np.int8),
         (np.concatenate([fs[:, 0], fs[:, 1], fs[:, 2]]),
          np.concatenate([fs[:, 1], fs[:, 2], fs[:, 0]]))),
        shape=(len(v), len(v)))
    _, lab = connected_components(adj, directed=False)
    labs = np.unique(lab[fs[:, 0]])
    mx = np.array([float(s[np.unique(fs[lab[fs[:, 0]] == c])].max())
                   for c in labs])
    shards = mx[mx < 0.7 * HEIGHT]
    assert len(shards) == 0, (
        f"{len(shards)} partial-height shard component(s) in the gate "
        f"fringe (max heights {np.round(np.sort(shards), 2)} < "
        f"{0.7 * HEIGHT:.1f}mm, of {len(labs)} fringe components) — "
        f"the binary gate cut is tearing ribs into tatters instead of "
        f"ending them in full-height capped noses")


def test_gate_band_debris_and_watertight(gatecap):
    """(c) No sub-few-voxel components touching the gate band, output
    watertight (transplanted from test_runout_quality)."""
    import manifold3d as m3d
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components
    vox = gatecap["res"] ** 3
    nzg = gatecap["nzg"]
    off = 0
    for verts, tris in gatecap["clusters"]:
        man = m3d.Manifold(m3d.Mesh(
            np.ascontiguousarray(verts, np.float32),
            np.ascontiguousarray(tris, np.uint32)))
        assert not man.is_empty(), "output cluster is not watertight"
        adj = sp.csr_matrix(
            (np.ones(len(tris) * 3, np.int8),
             (np.concatenate([tris[:, 0], tris[:, 1], tris[:, 2]]),
              np.concatenate([tris[:, 1], tris[:, 2], tris[:, 0]]))),
            shape=(len(verts), len(verts)))
        _, lab = connected_components(adj, directed=False)
        for cid in np.unique(lab[tris[:, 0]]):
            fc = tris[lab[tris[:, 0]] == cid]
            if nzg[off + np.unique(fc)].min() >= 0.2:
                continue             # component never touches the band
            vol = abs(np.einsum(
                "ij,ij->i", verts[fc[:, 0]],
                np.cross(verts[fc[:, 1]], verts[fc[:, 2]])).sum() / 6.0)
            assert vol >= 10.0 * vox and len(fc) >= 24, (
                f"debris in the gate band: component of {len(fc)} "
                f"faces, {vol:.3f}mm^3 (< 10 voxels)")
        off += len(verts)
