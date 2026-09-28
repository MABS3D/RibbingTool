"""Bounded, repeatable RibbingTool geometry/export benchmark.

Run with the project's virtualenv Python. Every invocation imports exactly
one --code-root in a child process. The parent limits wall time on all
platforms and private memory on Windows, and preserves a partial JSON report
if the worker fails. Private-memory monitoring is unavailable on other OSes.
Nothing is sent to a running app and source trees are never modified.

Examples (PowerShell):
  .venv/Scripts/python.exe benchmark_generation.py --code-root BASELINE `
      --case plate --output baseline-plate.json --profile
  .venv/Scripts/python.exe benchmark_generation.py --code-root REPO `
      --case dashboard --faces 123 --output current-dashboard.json

Use --inspect-dashboard to list bounded curved-face candidates without
running generation. The dashboard defaults to testdata/part1.stp; --model can
select another STEP. Selected faces remain on the actual CAD body, including
its excluded faces, throughout generation and union.
"""
from __future__ import annotations

import argparse
import cProfile
import ctypes
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import pstats
import struct
import subprocess
import sys
import time
import traceback


def process_memory(pid=None):
    if sys.platform != "win32":
        return {}
    from ctypes import wintypes

    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
                    ("PrivateUsage", ctypes.c_size_t)]
    kernel, psapi = ctypes.windll.kernel32, ctypes.windll.psapi
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
    handle = kernel.GetCurrentProcess() if pid is None else kernel.OpenProcess(0x1010, False, pid)
    if not handle:
        return {}
    try:
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        if not psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
            return {}
        return {"working_set_mb": round(counters.WorkingSetSize / 2**20, 2),
                "peak_working_set_mb": round(counters.PeakWorkingSetSize / 2**20, 2),
                "private_mb": round(counters.PrivateUsage / 2**20, 2),
                "peak_private_mb": round(counters.PeakPagefileUsage / 2**20, 2)}
    finally:
        if pid is not None:
            kernel.CloseHandle(handle)


def write_report(output, report):
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(output)


def digest_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def worker(args):
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Windows virtualenv python.exe is a redirector: Popen.pid can identify
    # the small launcher instead of this actual Python worker.
    args.output.with_suffix(".pid").write_text(str(os.getpid()), encoding="ascii")
    import faulthandler
    faulthandler.enable()
    sys.dont_write_bytecode = True
    code_root = args.code_root.resolve()
    sys.path.insert(0, str(code_root))
    import numpy as np
    import manifold3d as m3d
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from server.geometry import graph_ribs as graph
    from server.geometry.booleans import mesh_union, _to_manifold
    from server.geometry.meshing import mesh_shape
    from server.geometry.patterns import RibParams
    from server.geometry.step_io import load_step, shape_volume
    import server.main as api

    assert Path(graph.__file__).resolve().is_relative_to(code_root), graph.__file__
    report = {"status": "running", "worker_pid": os.getpid(), "case": args.case, "code_root": str(code_root),
              "source_sha256": {str(path.relative_to(code_root)): digest_file(path)
                                for path in sorted((code_root / "server").rglob("*.py"))},
              "python": sys.version, "platform": platform.platform(), "cpu_count": os.cpu_count(),
              "profile_enabled": args.profile, "timeout_seconds": args.timeout,
              "max_private_gb": args.max_private_gb, "memory_limit_supported": sys.platform == "win32",
              "phases": {}, "measurements": {}}
    start = time.perf_counter()

    def save():
        report["worker_elapsed_seconds"] = round(time.perf_counter() - start, 6)
        report["memory"] = process_memory()
        write_report(args.output, report)

    def phase(name, function):
        report["active_phase"] = name
        save()
        print(json.dumps({"phase": name, "event": "start", "elapsed": report["worker_elapsed_seconds"]}), flush=True)
        profile = cProfile.Profile() if args.profile else None
        wall, cpu = time.perf_counter(), time.process_time()
        try:
            result = profile.runcall(function) if profile else function()
        finally:
            values = {"wall_seconds": round(time.perf_counter() - wall, 6),
                      "cpu_seconds": round(time.process_time() - cpu, 6), **process_memory()}
            report["phases"][name] = values
            if profile:
                prefix = args.output.with_suffix("")
                profile.dump_stats(str(prefix) + "." + name + ".pstats")
                stream = io.StringIO()
                pstats.Stats(profile, stream=stream).strip_dirs().sort_stats("cumulative").print_stats(40)
                Path(str(prefix) + "." + name + ".profile.txt").write_text(stream.getvalue(), encoding="utf-8")
            print(json.dumps({"phase": name, "event": "done", **values}), flush=True)
            save()
        return result

    def cache_summary(cache):
        arrays, seen = [], set()
        def visit(value):
            if id(value) in seen:
                return
            seen.add(id(value))
            if isinstance(value, np.ndarray):
                arrays.append(value)
            elif isinstance(value, dict):
                for child in value.values():
                    visit(child)
            elif isinstance(value, (list, tuple)):
                for child in value:
                    visit(child)
        visit(cache)
        return {"keys": sorted(cache), "array_count": len(arrays),
                "array_megabytes": round(sum(array.nbytes for array in arrays) / 2**20, 4)}

    def concat(clusters):
        vertices, triangles, offset = [], [], 0
        for v, f in clusters:
            vertices.append(np.asarray(v, np.float64))
            triangles.append(np.asarray(f, np.int64) + offset)
            offset += len(v)
        return np.vstack(vertices), np.vstack(triangles)

    def mesh_metrics(clusters):
        v, f = concat(clusters)
        finite = bool(np.isfinite(v).all())
        valid_indices = bool(f.min() >= 0 and f.max() < len(v))
        area2 = np.linalg.norm(np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]]), axis=1)
        _, incidence = np.unique(np.sort(np.vstack([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1),
                                 axis=0, return_counts=True)
        man = m3d.Manifold(m3d.Mesh64(np.ascontiguousarray(v), np.ascontiguousarray(f, np.uint64)))
        parts = man.decompose()
        result = {"vertices": len(v), "triangles": len(f), "input_shells": len(clusters),
                  "finite": finite, "valid_indices": valid_indices,
                  "collapsed_triangles": int((area2 <= 1e-14).sum()),
                  "boundary_edges": int((incidence == 1).sum()),
                  "nonmanifold_edges": int((incidence > 2).sum()),
                  "manifold_status": str(man.status()), "volume_mm3": man.volume(),
                  "positive_components": sum(part.volume() > 0 for part in parts),
                  "negative_components": sum(part.volume() < 0 for part in parts),
                  "bbox": [*v.min(0).tolist(), *v.max(0).tolist()],
                  "array_sha256": hashlib.sha256(v.tobytes() + f.tobytes()).hexdigest()}
        result["closed_valid_mesh"] = bool(finite and valid_indices and not result["collapsed_triangles"]
            and not result["boundary_edges"] and not result["nonmanifold_edges"]
            and man.status() == m3d.Error.NoError and not man.is_empty())
        return result

    def done():
        report["status"] = "complete"
        report.pop("active_phase", None)
        save()
        print(json.dumps({"result": str(args.output), "seconds": report["worker_elapsed_seconds"]}), flush=True)

    try:
        shape = phase("load_model", lambda: BRepPrimAPI_MakeBox(60., 40., 8.).Shape()
                      if args.case == "plate" else load_step(args.model))
        meshes = phase("initial_body_mesh", lambda: mesh_shape(shape, .5, .5))
        report["body"] = {"faces": len(meshes), "volume_mm3": shape_volume(shape),
                          "display_triangles": sum(len(mesh.triangles) for mesh in meshes)}
        if args.case == "dashboard":
            report["model"] = {"path": str(args.model), "sha256": digest_file(args.model)}
        if args.inspect_dashboard:
            report["face_catalog"] = [{"face_id": m.face_id, "kind": m.surface_kind,
                                        "area_mm2": m.area, "triangles": len(m.triangles),
                                        "bbox": [*m.vertices.min(0).tolist(), *m.vertices.max(0).tolist()],
                                        "extent_mm": np.ptp(m.vertices, axis=0).tolist()}
                                       for m in meshes]
            candidates = [item for item in report["face_catalog"] if item["kind"] != "plane"
                          and 400 <= item["area_mm2"] <= 3500 and max(item["extent_mm"]) < 110]
            print(json.dumps({"bounded_curved_candidates": sorted(candidates, key=lambda item: item["area_mm2"])}), flush=True)
            done()
            return
        if args.faces:
            face_ids = sorted(set(map(int, args.faces.split(","))))
        elif args.case == "plate":
            face_ids = [max(meshes, key=lambda mesh: mesh.vertices[:, 2].mean()).face_id]
        else:
            raise ValueError("dashboard generation requires explicit --faces; use --inspect-dashboard first")
        valid = {m.face_id for m in meshes}
        if not set(face_ids) <= valid:
            raise ValueError("selected face does not exist")
        params = RibParams(pattern="isogrid", spacing=12., thickness=1.6, height=4., margin=2.,
                           taper_len=5., fillet_root=1.2, fillet_top=.5, fillet_junction=1.2,
                           guide_smoothing=3., mapping="surface")
        if args.params_json:
            params = RibParams.from_dict({**vars(params), **json.loads(args.params_json.read_text())})
        report.update(face_ids=face_ids, params=vars(params), fine_quality=args.fine_quality,
                      selection_area_mm2=sum(m.area for m in meshes if m.face_id in face_ids))
        cache = {}
        for label in ("cold", "warm"):
            preview = phase("preview_" + label, lambda: graph.preview_rib_graph(shape, face_ids, params, frame_cache=cache))
            report["measurements"]["preview_" + label] = {
                **preview["stats"], "warnings": preview["warnings"],
                "paths_sha256": hashlib.sha256(json.dumps(preview["paths"], separators=(",", ":")).encode()).hexdigest(),
                "cache": cache_summary(cache)}
            save()
        report["measurements"]["preview_cold_warm_identical"] = (
            report["measurements"]["preview_cold"]["paths_sha256"] == report["measurements"]["preview_warm"]["paths_sha256"])
        if args.stop_after == "preview":
            done()
            return
        last_progress = [None, 0.]
        def progress(stage, done_count, total):
            now = time.perf_counter()
            if stage != last_progress[0] or now - last_progress[1] >= 5 or done_count == total:
                print(json.dumps({"progress": stage, "done": done_count, "total": total,
                                  "elapsed": round(now - start, 3)}), flush=True)
                last_progress[:] = [stage, now]
        clusters, reports = phase("apply_quality1", lambda: graph.build_rib_graph(
            shape, face_ids, params, quality=1., frame_cache=cache, progress=progress))
        report["measurements"]["apply"] = phase("validate_apply", lambda: mesh_metrics(clusters))
        report["measurements"]["apply"]["reports"] = [vars(item) for item in reports]
        report["measurements"]["cache_after_apply"] = cache_summary(cache)
        if args.save_meshes:
            v, f = concat(clusters)
            np.savez_compressed(str(args.output.with_suffix("")) + ".apply.npz", v=v, f=f)
        if args.stop_after == "apply":
            done()
            return
        fine, reports = phase("fine_build", lambda: graph.build_rib_graph(
            shape, face_ids, params, quality=args.fine_quality, frame_cache=cache, progress=progress))
        report["measurements"]["fine"] = phase("validate_fine", lambda: mesh_metrics(fine))
        report["measurements"]["fine"]["reports"] = [vars(item) for item in reports]
        if args.save_meshes:
            v, f = concat(fine)
            np.savez_compressed(str(args.output.with_suffix("")) + ".fine.npz", v=v, f=f)
        if args.stop_after == "fine":
            done()
            return
        shells = phase("union_body_fine", lambda: mesh_union(shape, fine, lin_defl=.2))
        union_info = phase("validate_union", lambda: mesh_metrics(shells))
        body_man = phase("reference_export_body", lambda: _to_manifold(shape, .2))
        body_volume = body_man.volume()
        union_info.update(body_mesh_volume_mm3=body_volume,
                          added_volume_mm3=union_info["volume_mm3"] - body_volume,
                          all_ribs_attached=(union_info["input_shells"] == 1
                                             and union_info["positive_components"] == 1
                                             and union_info["volume_mm3"] > body_volume))
        report["measurements"]["union"] = union_info
        v, f = concat(shells)
        if args.save_meshes:
            np.savez_compressed(str(args.output.with_suffix("")) + ".union.npz", v=v, f=f)
        if args.stop_after == "union":
            done()
            return
        v, f = phase("prepare_stl_precision", lambda: api.stl_export_mesh(v, f))
        stl = phase("serialize_stl", lambda: api._stl_bytes(v, f))
        def stl_metrics():
            count = struct.unpack_from("<I", stl, 80)[0]
            dtype = np.dtype([("normal", "<f4", 3), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")])
            records = np.frombuffer(stl, dtype=dtype, offset=84)
            points = records["vertices"].astype(np.float64)
            vectors = np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0])
            return {"bytes": len(stl), "triangles": count,
                    "record_count_correct": count == len(f) and len(stl) == 84 + count * 50,
                    "finite": bool(np.isfinite(points).all()),
                    "collapsed_triangles_after_float32": int((np.linalg.norm(vectors, axis=1) <= 1e-14).sum()),
                    "signed_volume_mm3": float(np.einsum("ij,ij->i", points[:, 0], vectors).sum() / 6),
                    "sha256": hashlib.sha256(stl).hexdigest()}
        report["measurements"]["stl"] = phase("validate_stl", stl_metrics)
        if args.save_stl:
            args.output.with_suffix(".stl").write_bytes(stl)
        report["quality_pass"] = all(report["measurements"][key]["closed_valid_mesh"]
                                     for key in ("apply", "fine", "union")) and union_info["all_ribs_attached"] \
            and report["measurements"]["preview_cold_warm_identical"] \
            and report["measurements"]["stl"]["record_count_correct"] \
            and report["measurements"]["stl"]["finite"] \
            and not report["measurements"]["stl"]["collapsed_triangles_after_float32"]
        done()
    except BaseException as exc:
        report.update(status="error", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        save()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--case", choices=["plate", "dashboard"], default="plate")
    parser.add_argument("--model", type=Path,
                        help="dashboard STEP path; defaults to --code-root/testdata/part1.stp")
    parser.add_argument("--faces", help="comma-separated original CAD face IDs")
    parser.add_argument("--inspect-dashboard", action="store_true")
    parser.add_argument("--params-json", type=Path, help="optional parameter overrides; same file for both code roots")
    parser.add_argument("--fine-quality", type=float, default=3.)
    parser.add_argument("--stop-after", choices=["preview", "apply", "fine", "union", "stl"], default="stl")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--save-stl", action="store_true")
    parser.add_argument("--save-meshes", action="store_true")
    parser.add_argument("--timeout", type=float, default=180.)
    parser.add_argument("--max-private-gb", type=float, default=8.,
                        help="Windows-only private-memory limit in GiB; other platforms enforce timeout only")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.inspect_dashboard:
        args.case = "dashboard"
    if args.model is None:
        args.model = args.code_root / "testdata" / "part1.stp"
    if args.case == "dashboard" and not args.model.is_file():
        parser.error("dashboard STEP not found; provide an explicit --model path")
    if args.timeout <= 0 or args.max_private_gb <= 0:
        parser.error("time and memory budgets must be positive")
    args.output = args.output.resolve()
    if args.worker:
        worker(args)
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    log_path = args.output.with_suffix(".log")
    pid_path = args.output.with_suffix(".pid")
    args.output.unlink(missing_ok=True)
    pid_path.unlink(missing_ok=True)
    command = [sys.executable, "-u", str(Path(__file__).resolve()), *sys.argv[1:], "--worker"]
    beginning = time.perf_counter()
    peak_private = 0.
    limit_reason = None
    actual_pid = None
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        while process.poll() is None:
            elapsed = time.perf_counter() - beginning
            if actual_pid is None and pid_path.exists():
                try:
                    actual_pid = int(pid_path.read_text())
                except ValueError:
                    pass
            memory = process_memory(actual_pid or process.pid)
            peak_private = max(peak_private, memory.get("private_mb", 0))
            if elapsed > args.timeout:
                limit_reason = f"wall-time budget exceeded ({args.timeout}s)"
            elif peak_private > args.max_private_gb * 1024:
                limit_reason = f"private-memory budget exceeded ({args.max_private_gb}GiB)"
            if limit_reason:
                if sys.platform == "win32":
                    # Kill only this benchmark's own process tree, including
                    # the actual worker behind the virtualenv redirector.
                    subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10)
                else:
                    process.kill()
                process.wait(timeout=10)
                break
            time.sleep(.25)
    report = json.loads(args.output.read_text()) if args.output.exists() else {
        "case": args.case, "code_root": str(args.code_root.resolve())}
    if limit_reason or process.returncode:
        report.update(status="limited" if limit_reason else "worker_failed",
                      failure=limit_reason or f"worker exit code {process.returncode}")
    report["supervisor"] = {"elapsed_seconds": round(time.perf_counter() - beginning, 3),
                            "sampled_peak_private_mb": round(peak_private, 2),
                            "launcher_pid": process.pid, "worker_pid": actual_pid,
                            "memory_limit_supported": sys.platform == "win32",
                            "exit_code": process.returncode, "log": str(log_path)}
    write_report(args.output, report)
    summary = {"status": report.get("status"), "case": args.case, "output": str(args.output),
               "quality_pass": report.get("quality_pass"),
               "phases": {key: value["wall_seconds"] for key, value in report.get("phases", {}).items()},
               "supervisor": report["supervisor"]}
    print(json.dumps(summary, indent=2))
    if process.returncode or limit_reason or report.get("quality_pass") is False:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
