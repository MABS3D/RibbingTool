"""Steep-crown crest quality: voxel-pitch stitch texture must be absent.

Rib crowns are the s=height offset of the substrate, a smooth surface
whose crest lines are smooth space curves: on the R=30 wrap cylinder
the crest of every rib is an arc of radius ~33mm, so its own curvature
contributes ~4nm of sagitta over a 1mm window -- nothing.  The implicit
FIELD's own crest is smooth (probed at r = 33.00 +- 0.02 from 0deg to
64deg on this fixture), so any crest roughness in the output mesh is
pure surface-extraction texture -- terrace steps whose spatial period
is the voxel pitch stretched by the slope (res/cos(theta)), the
"stitched seam" look on every steep rib crown of a real dashboard
panel.

The measurement is spectral and method-agnostic (it sees only the
output mesh): crown crest polylines are sampled by cross-sections along
each rib centerline, parameterized by crest arc length, high-passed
(subtracting a 0.6mm-sigma gaussian baseline removes rib shape, height
trends and any smooth truncation profile), and the residual RMS
amplitude in the wavelength band [res, 2.5*res] is asserted.  Plane-fit
patch RMS is NOT used: crown-rounding sagitta dominates such fits and
hides the stitches.

RE-TARGET RATIONALE (2026-08-25, sanctioned): the original sampler took
the outermost mesh crossing inside a FIXED +-1.4mm lateral window
around each projected centerline.  That is correct for the 4 fall-line
runs (their crest lean is ALONG the rib) but wrong for the 8
diagonal-family runs: a crown on a theta-steep band leans laterally by
height*sin(theta)*sin(60deg) -- ~2.25mm at 60deg, OUTSIDE the window
-- so those "crest" samples pinned at the window edge (|q| =
1.31-1.34) and measured a chord envelope of the rib FLANK, where
dr/dq ~ 1.3 amplifies lateral sampling jitter into fake radial
residual.  Probe evidence that this was the metric, not the mesh
(probe_diag_field / probe_metric / probe_subdiv, session scratchpad):
  * the true crest is present and exact in the same mesh: max r within
    |q| < 4 reads 32.999-33.001 at q ~ 2.25 on the worst diagonal runs
    (the windowed profile read 32.2-32.4);
  * surface-nets vertices sit ON the smooth field (|f| p50 = 0.0um,
    p90 = 3.2um) -- >5um of real geometric texture is impossible;
  * the old diagonal reading converges toward zero under mesh
    subdivision + exact-field snap (20.2um at 4x faces, 8.55um at 16x)
    -- the signature of sampling noise, not geometry; no extraction at
    res = 0.30 can pass the old window-edge metric (floor ~8um).
The sampler now TRACKS the crest: same stations, same outermost-
crossing rule, over a +-4.0mm window covering the full rib
cross-section including the maximum lean (2.35mm at 65deg).  The
spectral band, the 5um threshold, the fall-line runs and the
shallow-crown control are unchanged.  Two assertions are ADDED, so the
re-target is strictly STRONGER than the original:
  * absolute crest accuracy: the median crest radius over the steep
    runs must sit within +-50um of the field-truth crest (asserted
    against the analytic 33.00; the field's own tolerance is +-0.02,
    leaving >=30um for extraction).  The original band-lid erosion
    defect read 31.6-32.3 at 55-65deg -- this catches it directly, no
    spectral argument needed.
  * synthetic-injection honesty guard: a deterministic 30um sawtooth
    at 0.45mm arc wavelength (in-band; the real terrace period is
    res/cos(theta) = 0.44-0.71mm on these slopes) is added to the
    measured crest profiles and the same metric must read it > 5um.
    Per-station offsets commute with the max-r crest sampler
    (max(r_i) + c == max(r_i + c)), so profile injection is exactly a
    coherent radial displacement of the crest-band surface -- the
    amplitude class the user photographed.

Threshold 5um justification:
  * crowns below 44deg of slope on the SAME build measure 0.2-0.4um
    median through the identical pipeline, and the crest-tracked
    fall-line steep runs measure 0.16-0.28um -- the floor is
    demonstrably reachable by whatever produces the output mesh, on
    this very lattice at this very resolution;
  * 5um is 1.7% of the 0.30mm voxel pitch, invisible at any zoom and
    far below print resolution;
  * the band-lid erosion this test was written against measured 7-20um
    per run with the dominant residual wavelength in the voxel-texture
    range -- not smooth curvature.

Verified numbers (this file, wrap_cyl at 0.30mm, 12 steep runs):
  surface_nets:   crest-tracked band RMS 0.16-0.56um (median 0.39),
                  per-run median crest radii 32.9952-32.9965
                  (median 32.9964); control 0.22um over 34 runs.
  marching_cubes: crest-tracked band RMS 0.17-2.91um (median 2.20;
                  fall-line 0.17-0.19, diagonal 2.06-2.91), radii
                  33.0028-33.0186 (median 33.0057); control 0.37um.
  honesty guard:  the 30um/0.45mm sawtooth reads 8.0-13.4um per run
                  (median 11.4 SN / 12.4 MC) through the same metric.
"""
import math
import re

import numpy as np

from server.geometry.implicit import build_rib_implicit
from server.geometry.patterns import RibParams, generate_segments
from server.geometry.step_io import load_step
from tests.test_ribbing import biggest_face_id

R_CYL = 30.0        # wrap_cyl_step cylinder radius, axis = world z

DT = 0.04           # crest cross-section station pitch (projected mm)
DS = 0.06           # uniform arc-length resampling pitch (mm)
SIG_HP = 0.6        # gaussian baseline sigma (mm): keeps sub-mm texture
TRIM = 0.7          # crest run end trim (mm): termination ramps out
CLEAR = 2.5         # junction clearance to other rib centerlines (mm)
MIN_ARC = 4.4       # minimum crest run arc length worth a spectrum (mm)
QWIN = 4.0          # lateral crest-tracking half-window (mm): covers the
                    # crown's lateral lean height*sin(theta)*sin(60deg)
                    # (2.35mm at 65deg); the pre-re-target 1.4mm window
                    # pinned diagonal-family samples on the rib flank
R_CREST = 33.0      # analytic crest radius R_CYL + height (field truth
                    # probed at 33.00 +- 0.02 from 0 to 64deg)
ACC_TOL = 0.05      # absolute crest-accuracy budget (mm)
INJ_AMP = 0.030     # honesty-guard sawtooth amplitude (mm)
INJ_LAM = 0.45      # honesty-guard sawtooth arc wavelength (mm), inside
                    # the asserted [res, 2.5*res] band


def _crest_runs(v, e, uv, r, seglist, nz_axis, nz_lo, nz_hi):
    """Crown crest profiles per rib: (max-radius, crest-xyz) per station.

    For each 2D rib centerline, cross-section planes every DT sample the
    mesh edges near the crown (both endpoints within QWIN laterally --
    the FULL rib cross-section, so the tracked crest is the true
    outermost point wherever the crown leans -- and radially above
    R_CYL + 0.8); the outermost crossing point is the crest sample.
    Stations near lattice junctions are dropped (crowns blend there),
    and stations are gated by the crest point's own slope (angle of its
    radial direction to the projection axis).
    """
    runs = []
    for si, ((p0x, p0y), (p1x, p1y)) in enumerate(seglist):
        p0 = np.array([p0x, p0y])
        ev = np.array([p1x, p1y]) - p0
        ev /= np.hypot(*ev)
        m = np.array([-ev[1], ev[0]])
        t = (uv - p0) @ ev
        q = (uv - p0) @ m
        cand = (np.abs(q) < QWIN) & (r > R_CYL + 0.8)
        ce = e[cand[e[:, 0]] & cand[e[:, 1]]]
        if len(ce) < 40:
            continue
        ta, tb = t[ce[:, 0]], t[ce[:, 1]]
        t0, t1 = np.minimum(ta, tb), np.maximum(ta, tb)
        k0 = np.ceil(t0 / DT).astype(np.int64)
        k1 = np.floor(t1 / DT).astype(np.int64)
        n_st = k1 - k0 + 1
        keep = n_st > 0
        ce, ta, tb, k0, n_st = (ce[keep], ta[keep], tb[keep], k0[keep],
                                n_st[keep])
        if not len(ce):
            continue
        # every (edge, station) crossing pair, fully vectorized
        ei = np.repeat(np.arange(len(ce)), n_st)
        kk = k0[ei] + (np.arange(len(ei))
                       - np.repeat(np.cumsum(n_st) - n_st, n_st))
        al = (kk * DT - ta[ei]) / (tb[ei] - ta[ei])
        P3 = v[ce[ei, 0]] + al[:, None] * (v[ce[ei, 1]] - v[ce[ei, 0]])
        rp = np.hypot(P3[:, 0], P3[:, 1])
        kmin, kmax = kk.min(), kk.max()
        nst = kmax - kmin + 1
        rmax = np.full(nst, -1e9)
        np.maximum.at(rmax, kk - kmin, rp)
        cnt = np.zeros(nst, np.int64)
        np.add.at(cnt, kk - kmin, 1)
        order = np.lexsort((rp, kk))
        last = np.r_[np.nonzero(np.diff(kk[order]))[0], len(order) - 1]
        top = order[last]
        crest = np.full((nst, 3), np.nan)
        crest[kk[top] - kmin] = P3[top]
        # junction clearance: station centerline point vs other segments
        st_t = np.arange(kmin, kmax + 1) * DT
        st2 = p0[None, :] + st_t[:, None] * ev[None, :]
        dmin = np.full(nst, 1e9)
        for sj, ((a0x, a0y), (a1x, a1y)) in enumerate(seglist):
            if sj == si:
                continue
            a0 = np.array([a0x, a0y])
            av = np.array([a1x, a1y]) - a0
            tt = np.clip(((st2 - a0) @ av) / (av @ av), 0.0, 1.0)
            dd = np.hypot(*(st2 - a0 - tt[:, None] * av).T)
            dmin = np.minimum(dmin, dd)
        good = np.isfinite(crest[:, 0])
        nz = np.full(nst, np.nan)
        nz[good] = (crest[good, :2] / np.hypot(
            crest[good, 0], crest[good, 1])[:, None]) @ nz_axis[:2]
        ok = (cnt >= 3) & (dmin > CLEAR) & good \
            & (nz >= nz_lo) & (nz <= nz_hi)
        idx = np.nonzero(ok)[0]
        if not len(idx):
            continue
        brk = np.nonzero(np.diff(idx) > 1)[0]
        for a, b in zip(np.r_[idx[0], idx[brk + 1]],
                        np.r_[idx[brk], idx[-1]] + 1):
            if b - a >= 40:
                runs.append((rmax[a:b].copy(), crest[a:b].copy()))
    return runs


def _band_rms(profile_r, xyz, band):
    """RMS residual amplitude within the wavelength band, or None.

    Crest points are re-parameterized by arc length (computed on a
    mildly smoothed copy so the texture itself cannot inflate it),
    resampled uniformly, high-passed against a 0.6mm gaussian baseline,
    Hann-windowed, and integrated over the band via Parseval.  Also
    returns the dominant residual wavelength for diagnosis.
    """
    from scipy.ndimage import gaussian_filter1d
    p = xyz.copy()
    for k in range(3):
        p[:, k] = gaussian_filter1d(p[:, k], 2.0, mode="nearest")
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    if s[-1] < 2 * TRIM + MIN_ARC - 1.4:
        return None
    su = np.arange(TRIM, s[-1] - TRIM, DS)
    if len(su) < MIN_ARC / DS:
        return None
    ru = np.interp(su, s, profile_r)
    hp = ru - gaussian_filter1d(ru, SIG_HP / DS, mode="nearest")
    n = len(hp)
    win = np.hanning(n)
    ps = np.abs(np.fft.rfft(hp * win)) ** 2
    freq = np.fft.rfftfreq(n, DS)
    lam = np.where(freq > 0, 1.0 / np.maximum(freq, 1e-9), np.inf)
    wn = (win ** 2).mean()
    inb = (lam >= band[0]) & (lam <= band[1])
    rms = math.sqrt(2.0 * ps[inb].sum() / n ** 2 / wn)
    scan = (lam > 0.12) & (lam < 1.5)
    lam_dom = float(lam[scan][np.argmax(ps[scan])])
    return rms, lam_dom


def test_steep_crowns_free_of_voxel_stitch(wrap_cyl_step):
    s = load_step(wrap_cyl_step)
    fid = biggest_face_id(s, kind="cylinder")
    p = RibParams(pattern="isogrid", spacing=12, thickness=1.6, height=3,
                  taper_len=0, mapping="project")
    clusters, reports = build_rib_implicit(s, [fid], p)
    v = np.vstack([c[0] for c in clusters])
    f = np.vstack([c[1] + off for c, off in
                   zip(clusters, np.cumsum([0] + [len(c[0]) for c in
                                                  clusters[:-1]]))])

    # marching resolution as the engine reports it (0.30mm interactive)
    res = 0.30
    for rep in reports:
        for w in rep.warnings:
            mt = re.search(r"([0-9.]+)mm voxels", w)
            if mt:
                res = float(mt.group(1))
    band = (res, 2.5 * res)

    # rib centerlines + projection frame, reconstructed from the same
    # geometry inputs the engine uses (pattern definition, not engine
    # internals: the lattice window is origin-symmetric on whole
    # periods, so segment lines are phase-exact)
    from server.geometry.meshing import region_meshes
    from server.geometry.ribbing import (_kept_domain, _lattice_window,
                                         _merge_regions, _projection_frame)
    regions = region_meshes(s, [fid], 0.25, ang_defl=0.09)
    if len(regions) > 1:
        regions = [_merge_regions(regions)]
    region = regions[0]
    ctr, axes = _projection_frame(regions)
    n_axis = np.cross(axes[:, 0], axes[:, 1])
    flat = (region.vertices - ctr) @ axes
    tris = np.asarray(region.triangles, np.int64)
    domain = _kept_domain(flat, tris, np.arange(len(tris)), p)
    seglist = np.asarray(
        generate_segments(p, _lattice_window(p, domain.bounds)), float)

    uv = (v - ctr) @ axes
    r = np.hypot(v[:, 0], v[:, 1])
    e = np.sort(np.vstack([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]),
                axis=1)
    e = np.unique(e, axis=0)

    def gather(nz_lo, nz_hi):
        rms, doms, rmed, runs = [], [], [], []
        for prof, xyz in _crest_runs(v, e, uv, r, seglist, n_axis,
                                     nz_lo, nz_hi):
            out = _band_rms(prof, xyz, band)
            if out is not None:
                rms.append(out[0])
                doms.append(out[1])
                rmed.append(float(np.median(prof)))
                runs.append((prof, xyz))
        return np.asarray(rms), np.asarray(doms), np.asarray(rmed), runs

    # control: crowns below 44deg of slope on this same mesh.  This is
    # the smoothness the pipeline demonstrably delivers -- it proves
    # both that the metric's own noise floor is far below the threshold
    # and that the threshold is reachable on this exact lattice.
    ctl_rms, _, _, _ = gather(math.cos(math.radians(44.0)), 1.01)
    assert len(ctl_rms) >= 10, \
        f"only {len(ctl_rms)} shallow-crown crest runs found"
    ctl_med = float(np.median(ctl_rms))
    print(f"control: {len(ctl_rms)} shallow runs, median "
          f"{ctl_med * 1e3:.2f}um")
    assert ctl_med < 5e-3, \
        f"control violated: shallow crowns measure {ctl_med * 1e3:.1f}um " \
        f"in the {band[0]:.2f}-{band[1]:.2f}mm band"

    # the defect: crowns on the 47-65deg bands, crest-TRACKED (the
    # outermost point of the full cross-section, wherever it leans)
    stp_rms, stp_dom, stp_rmed, stp_runs = gather(
        math.cos(math.radians(65.0)), math.cos(math.radians(47.0)))
    assert len(stp_rms) >= 8, \
        f"only {len(stp_rms)} steep-crown crest runs found"
    stp_med = float(np.median(stp_rms))
    print(f"steep crest-tracked band RMS (um, sorted): "
          f"{np.round(np.sort(stp_rms) * 1e3, 2)} (median "
          f"{stp_med * 1e3:.2f})")
    assert stp_med < 5e-3, (
        f"steep rib crowns carry voxel-pitch stitch texture: median "
        f"crest residual {stp_med * 1e3:.1f}um in the "
        f"{band[0]:.2f}-{band[1]:.2f}mm wavelength band over "
        f"{len(stp_rms)} crest-tracked runs at 47-65deg slope "
        f"(threshold 5.0um; crowns below 44deg on this same mesh "
        f"measure {ctl_med * 1e3:.1f}um).  Dominant residual wavelength "
        f"{float(np.median(stp_dom)):.2f}mm ~ "
        f"{float(np.median(stp_dom)) / res:.1f}x the {res:.2f}mm voxel "
        f"pitch -- lattice terracing stretched by the slope "
        f"(res/cos(theta)), not surface curvature (the crest arc's own "
        f"sagitta is ~4nm/mm)")

    # absolute crest accuracy: the crest must not just be smooth, it
    # must sit AT the field-truth radius.  This is the assertion that
    # catches the original band-lid erosion defect directly (steep
    # crests read 31.6-32.3 there) with no spectral argument.
    crest_med = float(np.median(stp_rmed))
    print(f"steep per-run median crest radii: {np.round(stp_rmed, 4)} "
          f"(median {crest_med:.4f}, target {R_CREST:.2f} "
          f"+- {ACC_TOL})")
    assert abs(crest_med - R_CREST) < ACC_TOL, (
        f"steep crown crests sit at r = {crest_med:.3f}mm, "
        f"{abs(crest_med - R_CREST) * 1e3:.0f}um from the field-truth "
        f"crest {R_CREST:.2f} (budget {ACC_TOL * 1e3:.0f}um; the "
        f"field's own crest is 33.00 +- 0.02 from 0 to 64deg) -- the "
        f"extractor erodes or inflates steep crowns")

    # honesty guard: the crest-tracked metric must still SEE voxel-
    # pitch texture of the photographed amplitude class.  Inject a
    # deterministic 30um sawtooth at 0.45mm arc wavelength (in-band;
    # the real terrace period is res/cos(theta)) into the measured
    # profiles -- per-station offsets commute with the max-r crest
    # sampler, so this is exactly a coherent radial displacement of
    # the crest-band surface -- and the same metric must read it.
    inj = []
    for prof, xyz in stp_runs:
        s_arc = np.r_[0.0, np.cumsum(
            np.linalg.norm(np.diff(xyz, axis=0), axis=1))]
        saw = INJ_AMP * (2.0 * ((s_arc / INJ_LAM) % 1.0) - 1.0)
        out = _band_rms(prof + saw, xyz, band)
        assert out is not None
        inj.append(out[0])
    inj = np.asarray(inj)
    inj_med = float(np.median(inj))
    print(f"honesty guard: injected {INJ_AMP * 1e3:.0f}um sawtooth "
          f"reads (um, sorted) {np.round(np.sort(inj) * 1e3, 2)} "
          f"(median {inj_med * 1e3:.2f})")
    assert inj_med > 5e-3, (
        f"HONESTY GUARD FAILED: a {INJ_AMP * 1e3:.0f}um sawtooth at "
        f"{INJ_LAM}mm arc wavelength injected into the crest profiles "
        f"reads only {inj_med * 1e3:.2f}um (per-run "
        f"{np.round(np.sort(inj) * 1e3, 2)}) -- the re-targeted "
        f"sampler/spectrum can no longer detect the photographed "
        f"defect class and the primary assertion is vacuous")
