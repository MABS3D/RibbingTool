"""Visual verification: apply ribs, render body+ribs with a z-buffer.

Usage:
  python scripts/verify_render.py <step_path> <face_ids_csv> [--view name]
Renders output/vr_<name>.png from several angles.
"""
import sys
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "output"


def zbuffer_render(verts, tris, color_ids, eye, target, up=(0, 0, 1),
                   width=1500, height=1000, ortho_scale=None):
    """Software z-buffer flat-shaded render. color_ids: per-triangle 0/1."""
    eye = np.asarray(eye, float)
    target = np.asarray(target, float)
    up = np.asarray(up, float)
    fwd = target - eye
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, up)
    right /= np.linalg.norm(right)
    upv = np.cross(right, fwd)

    rel = verts - eye
    x = rel @ right
    y = rel @ upv
    z = rel @ fwd                     # depth along view
    if ortho_scale is None:
        ortho_scale = max(x.max() - x.min(), y.max() - y.min()) * 0.55
    sx = width / (2 * ortho_scale)
    px = (x / ortho_scale + 0.5) * width
    py = (0.5 - y / ortho_scale * (width / height)) * height

    img = np.ones((height, width, 3), np.float32)
    zbuf = np.full((height, width), -1e18, np.float32)

    tv = np.stack([px[tris], py[tris], z[tris]], axis=2)   # (m,3,3)
    tv[..., 1] = height - tv[..., 1]                       # flip y

    light = np.array([0.35, -0.45, 0.83])
    light /= np.linalg.norm(light)
    p0, p1, p2 = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    n = np.cross(p1 - p0, p2 - p0)
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    ln[ln < 1e-12] = 1
    n = n / ln
    # flip normals toward camera for shading
    cen = (p0 + p1 + p2) / 3
    to_cam = eye - cen
    flip = (n * to_cam).sum(1) < 0
    n[flip] *= -1
    shade = 0.30 + 0.70 * np.clip(n @ light, 0, 1)
    cols = np.where(color_ids[:, None] == 0,
                    np.array([0.62, 0.66, 0.72]),
                    np.array([0.80, 0.68, 0.52]))
    tri_col = cols * shade[:, None]

    order = np.argsort(-tv[:, :, 2].mean(axis=1))          # far first
    for ti in order:
        tri = tv[ti]
        xmin = max(int(np.floor(tri[:, 0].min())), 0)
        xmax = min(int(np.ceil(tri[:, 0].max())) + 1, width)
        ymin = max(int(np.floor(tri[:, 1].min())), 0)
        ymax = min(int(np.ceil(tri[:, 1].max())) + 1, height)
        if xmin >= xmax or ymin >= ymax:
            continue
        xs = np.arange(xmin, xmax) + 0.5
        ys = np.arange(ymin, ymax) + 0.5
        gx, gy = np.meshgrid(xs, ys)
        ax, ay = tri[0, 0], tri[0, 1]
        bx, by = tri[1, 0], tri[1, 1]
        cx, cy = tri[2, 0], tri[2, 1]
        det = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
        if abs(det) < 1e-9:
            continue
        w0 = ((by - cy) * (gx - cx) + (cx - bx) * (gy - cy)) / det
        w1 = ((cy - ay) * (gx - cx) + (ax - cx) * (gy - cy)) / det
        w2 = 1 - w0 - w1
        inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        if not inside.any():
            continue
        zt = w0 * tri[0, 2] + w1 * tri[1, 2] + w2 * tri[2, 2]
        sub = zbuf[ymin:ymax, xmin:xmax]
        upd = inside & (zt > sub)
        sub[upd] = zt[upd]
        img[ymin:ymax, xmin:xmax][upd] = tri_col[ti]
    return (img * 255).astype(np.uint8)


def main():
    step = sys.argv[1]
    faces = [int(x) for x in sys.argv[2].split(",")]
    from server.geometry.step_io import load_step
    from server.geometry.meshing import mesh_shape
    from server.geometry.patterns import RibParams
    from server.geometry.selection import grow_tangent
    from server.geometry.implicit import build_rib_implicit

    s = load_step(step)
    if faces == [0]:
        curved = [m for m in mesh_shape(s, 0.5, 0.5) if not m.is_planar]
        faces = [max(curved, key=lambda m: m.area).face_id]
    grown = grow_tangent(s, faces, angle_deg=20.0) if len(faces) == 1 \
        and "--nogrow" not in sys.argv else faces
    p = RibParams(pattern="isogrid", spacing=12, thickness=1.6, height=4,
                  margin=2, taper_len=5, fillet_root=2, fillet_top=2,
                  mapping="project")
    t0 = time.time()
    clusters, reports = build_rib_implicit(s, grown, p)
    print(f"apply {time.time()-t0:.0f}s")
    for r in reports:
        print("  warnings:", r.warnings)
    rv = np.vstack([c[0] for c in clusters])
    rt = np.vstack([c[1] for c in clusters])

    meshes = mesh_shape(s, 0.5, 0.5)
    vs, ts, off = [], [], 0
    for m in meshes:
        vs.append(np.asarray(m.vertices))
        ts.append(np.asarray(m.triangles, np.int64) + off)
        off += len(m.vertices)
    bv = np.vstack(vs)
    bt = np.vstack(ts)

    allv = np.vstack([bv, rv])
    allt = np.vstack([bt, rt + len(bv)])
    color_ids = np.concatenate([np.zeros(len(bt)), np.ones(len(rt))])
    # proper decimation for render speed (keeps watertight look);
    # simplify body and ribs separately to keep per-part colors
    import manifold3d as m3d

    def decim(V, F, eps):
        mesh = m3d.Manifold(m3d.Mesh(np.ascontiguousarray(V, np.float32),
                                     np.ascontiguousarray(F, np.uint32)))
        mesh = mesh.simplify(eps)
        out = mesh.to_mesh64()
        return (np.asarray(out.vert_properties[:, :3]),
                np.asarray(out.tri_verts))

    bv2, bt2 = decim(bv, bt, 0.3)
    rv2, rt2 = decim(rv, rt, 0.35)
    allv = np.vstack([bv2, rv2])
    allt = np.vstack([bt2, rt2 + len(bv2)])
    color_ids = np.concatenate([np.zeros(len(bt2)), np.ones(len(rt2))])
    print(f"render tris: {len(allt)}")

    ctr = allv.mean(0)
    span = np.linalg.norm(allv.max(0) - allv.min(0))
    views = {
        "front": ctr + np.array([0.3, -1.0, 0.55]) * span,
        "top": ctr + np.array([0.05, -0.35, 1.0]) * span,
        "side": ctr + np.array([1.0, -0.3, 0.25]) * span,
    }
    OUT.mkdir(exist_ok=True)
    for name, eye in views.items():
        img = zbuffer_render(allv, allt, color_ids, eye, ctr)
        out = OUT / f"vr_{name}.png"
        plt.imsave(out, img)
        print("saved", out)


if __name__ == "__main__":
    main()
