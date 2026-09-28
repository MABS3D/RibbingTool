"""SurfaceNets repair-pass and extractor-attribution unit tests.

_nonmanifold_edge_split splits 4-valent mesh edges by renaming one
normal-coherent face pair onto duplicated endpoint vertices.  When one
triangle carries TWO 4-valent edges, a single-snapshot rename matches
ORIGINAL vertex ids against a face the first group already renamed: the
second group half-rewires it (renames b but not a), emitting cross-pair
edges with incidence 1 that no embedding — including a closed mesh —
can close.  The deferred-group pass loop must instead defer any group
sharing a face with an already-processed group and re-run on the
mutated mesh.
"""
import numpy as np

from server.geometry.implicit import _nonmanifold_edge_split


def _edge_counts(faces):
    e = np.sort(np.vstack([faces[:, [0, 1]], faces[:, [1, 2]],
                           faces[:, [2, 0]]]), axis=1)
    return np.unique(e, axis=0, return_counts=True)


def _two_group_mesh():
    """7-face mesh: T=(2,0,1) carries BOTH 4-valent edges (0,1) and
    (1,2); apex coordinates put T in the renamed (`second`) pair of
    BOTH groups by the function's own normal-coherence argmax."""
    v = np.array([
        [0.0, 0.0, 0.0],   # 0
        [1.0, 0.0, 0.0],   # 1
        [1.0, 1.0, 0.0],   # 2
        [0.5, 0.0, -1.0],  # 3  F1 apex -> normal +y
        [0.5, 0.0, 1.0],   # 4  F2 apex -> normal +y
        [0.5, 1.0, 0.0],   # 5  F3 apex -> normal +z (parallel to T)
        [1.0, 0.5, 1.0],   # 6  G1 apex -> normal +x
        [1.0, 0.5, -1.0],  # 7  G2 apex -> normal +x
        [0.0, 0.5, 0.0],   # 8  G3 apex -> normal +z (parallel to T)
    ])
    f = np.array([
        [2, 0, 1],  # T (both 4-valent edges)
        [0, 1, 3],  # F1
        [1, 0, 4],  # F2
        [0, 1, 5],  # F3
        [1, 2, 6],  # G1
        [2, 1, 7],  # G2
        [1, 2, 8],  # G3
    ], dtype=np.int64)
    return v, f


def test_split_two_groups_sharing_a_triangle():
    # the corruption fixture: pre-fix, group 2 half-rewired T (renamed
    # b=2 -> 12 while a=1 was already 10) leaving cross-pair edges
    # (9,12)/(10,12) and the pairing edge (11,12) at incidence 1
    v, f = _two_group_mesh()
    eu0, cnt0 = _edge_counts(f)
    assert (cnt0 == 4).sum() == 2          # two 4-valent edge groups
    v2, f2 = _nonmanifold_edge_split(v, f)
    n0 = len(v)
    assert len(f2) == len(f)
    pairs = [(n, n + 1) for n in range(n0, len(v2), 2)]
    assert pairs, "split processed no group"
    eu, cnt = _edge_counts(f2)
    emap = {tuple(ed): int(c) for ed, c in zip(eu.tolist(), cnt.tolist())}
    # (1) no half-renamed face: a face referencing either member of a
    # duplicated pair must reference BOTH (it carried the full edge)
    for tri in f2:
        ts = set(int(x) for x in tri)
        for a2, b2 in pairs:
            assert (a2 in ts) == (b2 in ts), f"half-renamed face {tri}"
    # (2) every pairing edge is carried by exactly its split face pair
    for a2, b2 in pairs:
        assert emap.get((a2, b2), 0) == 2, \
            f"pairing edge ({a2},{b2}) incidence {emap.get((a2, b2), 0)}"
    # (3) no edge joins new vertices of DIFFERENT pairs (the buggy
    # (9,12)/(10,12) class — structurally unclosable in any embedding)
    pset = set(pairs)
    for ed in emap:
        if ed[0] >= n0 and ed[1] >= n0:
            assert ed in pset, f"cross-pair edge {ed}"
    # (4) new vertices are exact copies of their sources
    for a2, b2 in pairs:
        assert (v == v2[a2]).all(axis=1).any()
        assert (v == v2[b2]).all(axis=1).any()


def test_split_single_group_one_pass():
    # no cross-group face sharing: one pass, the known rename (second
    # pair = the two +z-parallel faces T and F3 by the coherence argmax)
    v, f = _two_group_mesh()
    v, f = v[:6], f[:4]                    # drop the G group entirely
    v2, f2 = _nonmanifold_edge_split(v, f)
    assert len(v2) == 8 and len(f2) == 4
    eu, cnt = _edge_counts(f2)
    emap = {tuple(ed): int(c) for ed, c in zip(eu.tolist(), cnt.tolist())}
    assert emap.get((6, 7), 0) == 2        # duplicated edge on T + F3
    assert (0, 1) in emap and emap[(0, 1)] == 2   # F1 + F2 keep original
    # untouched faces stay bit-identical
    np.testing.assert_array_equal(f2[1], f[1])
    np.testing.assert_array_equal(f2[2], f[2])


def test_split_noop_on_manifold_mesh():
    # closed tetrahedron: no 4-valent edges — identity, same objects
    v = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    f = np.array([[0, 2, 1], [0, 3, 2], [0, 1, 3], [1, 2, 3]], np.int64)
    v2, f2 = _nonmanifold_edge_split(v, f)
    assert v2 is v and f2 is f


def test_extractor_report_names_what_ran(box_step, monkeypatch):
    # SN self-heal fallback: mesh_field retries with marching_cubes when
    # surface_nets raises, so the engine summary must attribute the
    # extraction to what actually produced the mesh — not re-read the
    # env it was requested with
    from server.geometry import implicit as imp
    from server.geometry.patterns import RibParams
    from server.geometry.step_io import load_step
    from tests.test_ribbing import biggest_face_id

    monkeypatch.setenv("RIBBING_EXTRACTOR", "surface_nets")

    def boom(*a, **k):
        raise MemoryError("stubbed SN failure")

    monkeypatch.setattr(imp, "_surface_nets_tile", boom)
    s = load_step(box_step)
    p = RibParams(pattern="quadmesh", spacing=12, thickness=2.0, height=4,
                  margin=2, taper_len=0, mapping="project")
    clusters, reports = imp.build_rib_implicit(s, [biggest_face_id(s)], p)
    assert clusters
    w = reports[0].warnings
    assert any(x.startswith("surface_nets extractor failed") for x in w)
    summary = [x for x in w if x.startswith("implicit engine:")]
    assert summary and "marching_cubes extractor" in summary[-1], summary
