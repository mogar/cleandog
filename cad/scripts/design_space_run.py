"""Design-space study, compute step: solve every design, then cross-check it in MuJoCo.

Writes outputs/<date>_design-space/results.json and traces.npz. The report is built from them by
design_space_report.py, so figures can be iterated without recomputing (~10 min on 8 cores).

Usage: python cad/scripts/design_space_run.py [--only B N ...]
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from cleandog_cad import design_space as ds

OUT = ds.REPO / "outputs" / f"{dt.date.today().isoformat()}_design-space"


def run_one(key: str) -> dict:
    t0 = time.time()
    from cleandog_cad.mj_check import replay

    ref = ds.load_reference()
    des = next(d for d in ds.DESIGNS if d.key == key)
    if des.notch:
        des = ds.with_(des, nose_half=15.0)  # solve with a slim nose, then widen it as far as it goes
    sw = ds.solve(des, ref)
    nose = None
    if des.notch:
        nose = ds.widest_nose(des, sw.y_leg, ref)
        des = ds.with_(des, nose_half=nose)
        sw = ds.sweep_clearance(des, sw.y_leg, ref, per_frame=True)
    su = ds.summarise(des, sw)
    ik_frames = ds.sweep_clearance(des, sw.y_leg, ref, per_frame=True, ik_only_frames=True).per_frame
    rp = replay(des, sw.y_leg, ref)
    pf = ik_frames[rp.frames]
    near = (pf < ds.CULL) & (rp.trunk < 40)
    stack = [p for p in rp.stairs if p in ("hip motor", "knee motor", "belt", "thigh hub")]
    legs = [p for p in rp.stairs if p not in stack]
    res = dict(
        key=key, label=des.label, design={k: (v.name if hasattr(v, "name") else v)
                                          for k, v in dataclasses.asdict(des).items()
                                          if not isinstance(v, dict)},
        y_leg=sw.y_leg, ya=des.ya, nose_half=nose, notch_x=des.notch_x if des.notch else None,
        min_clear=sw.min_clear, worst_part=sw.worst_part, worst_fx=sw.worst_fx,
        worst_q_deg=np.degrees(sw.worst_q).tolist(), per_part=sw.per_part, max_abad_ik=sw.max_abad,
        infeasible=sw.infeasible, static_roll=ds.static_roll_clearance(des, sw.y_leg),
        overall_width=su.overall_width, track=su.track, abad_offset=su.abad_offset, hip_width=su.hip_width,
        length=su.length, leg_motor_mass=su.leg_mass, knee_peak=su.knee_peak, knee_rated=su.knee_rated,
        hip_peak=su.hip_peak, hip_rated=su.hip_rated, trunk_area_cm2=su.trunk_area,
        mj_trunk_min=float(rp.trunk.min()),
        mj_vs_np_max_diff=float(np.abs(pf[near] - rp.trunk[near]).max()) if near.any() else None,
        mj_frames_compared=int(near.sum()),
        stairs_worst={p: dict(dist=d, frame=f, leg=leg) for p, (d, f, leg) in rp.worst_stairs.items()},
        stack_stairs_min=min(rp.worst_stairs[p][0] for p in stack),
        leg_stairs_min=min(rp.worst_stairs[p][0] for p in legs),
        seconds=time.time() - t0,
    )
    traces = dict(clear=sw.per_frame.astype(np.float32), mj_frames=rp.frames, mj_trunk=rp.trunk.astype(np.float32),
                  stack_stairs=np.min([rp.stairs[p] for p in stack], 0).astype(np.float32))
    print(f"{key:4s} width {su.overall_width:6.1f}  leg {sw.y_leg:6.1f}  clr {sw.min_clear:5.2f} ({sw.worst_part})  "
          f"stack-stairs {res['stack_stairs_min']:6.1f}  infeasible {sw.infeasible:.3f}  {res['seconds']:.0f} s",
          flush=True)
    return res, traces


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="design keys (default: all)")
    args = ap.parse_args(argv)
    keys = args.only or [d.key for d in ds.DESIGNS]
    OUT.mkdir(parents=True, exist_ok=True)
    rj, tz = OUT / "results.json", OUT / "traces.npz"
    results = json.loads(rj.read_text()) if rj.exists() else {}
    traces = dict(np.load(tz)) if tz.exists() else {}
    with ProcessPoolExecutor(max_workers=min(8, len(keys))) as ex:
        for res, tr in ex.map(run_one, keys):
            results[res["key"]] = res
            traces.update({f"{res['key']}/{k}": v for k, v in tr.items()})
    ref = ds.load_reference()
    traces["dt"], traces["segment"] = np.array(ref.dt), ref.segment
    traces["segment_names"] = np.array(ref.segment_names)
    rj.write_text(json.dumps(results, indent=1))
    np.savez_compressed(tz, **traces)
    print(rj)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
