"""M0 step 3a: plan both whole flights and build the round-trip reference.

Usage:
  python scripts/rl_reference.py [--morph SEG_CM KNEE SPACING_CM] [--reuse-plans]

Writes outputs/<date>_rl/plans.json, reference.npz and reference.json (segment
timings, landing crawl log, sway reductions, 3D static margins). Then run
scripts/rl_train.py.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from visualize import OUTPUTS, load_transitions, recommend  # noqa: E402

from cleandog_sim import StairParams  # noqa: E402
from cleandog_sim.transitions import morphology  # noqa: E402


def plan_one(args):
    from cleandog_sim.reference import plan_flight

    key, flight = args
    t0 = time.time()
    _, path, status = plan_flight(morphology(*key), StairParams(), flight)
    return flight, status, path, time.time() - t0


def clearance_audit(p, s, ref, out: Path) -> dict:
    """3D clearances of every frame (legs vs trunk, hip stacks and limbs vs the stairs): writes
    reference_clearance.npz and returns the summary for reference.json."""
    from cleandog_sim.reference import Clearance, audit

    clr = audit(p, s, ref)
    np.savez_compressed(out / "reference_clearance.npz", **clr)
    chk = Clearance(p, s)
    clearance = {}
    for k, v in clr.items():
        f = int(np.argmin(v))
        _, worst = chk(ref.qpos[f], detail=True)
        clearance[k] = dict(min=float(v[f]), frame=f, segment=ref.segment_names[ref.segment[f]],
                            pair=list(worst.get(k, ())), frames_below_0=int((v < 0).sum()),
                            frames_below_3=int((v < 3).sum()))
    return clearance


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--morph", nargs=3, metavar=("SEG_CM", "KNEE", "SPACING_CM"))
    ap.add_argument("--audit-only", action="store_true",
                    help="re-measure the clearances of the latest reference without re-planning")
    ap.add_argument("--reuse-plans", action="store_true",
                    help="rebuild the reference from today's plans.json instead of re-planning (~40 min)")
    args = ap.parse_args(argv)
    if args.audit_only:
        from cleandog_sim.reference import Reference

        d = sorted(OUTPUTS.glob("*_rl/reference.json"))[-1].parent
        info = json.loads((d / "reference.json").read_text())
        ref = Reference.load(d / "reference.npz")
        info["clearance"] = clearance_audit(morphology(*info["morphology"]), StairParams(), ref, d)
        (d / "reference.json").write_text(json.dumps(info))
        for k, c in info["clearance"].items():
            print(f"  clearance {k}: min {c['min']:.1f} mm at frame {c['frame']} ({c['segment']}, {c['pair']}), "
                  f"{c['frames_below_0']} frames below 0, {c['frames_below_3']} below 3")
        return 0
    if args.morph:
        key = (float(args.morph[0]) / 100, args.morph[1], float(args.morph[2]) / 100)
    else:
        key, _ = recommend(load_transitions()[1]["rows"])
    out = OUTPUTS / f"{dt.date.today().isoformat()}_rl"
    out.mkdir(parents=True, exist_ok=True)

    if args.reuse_plans:
        plans = json.loads((out / "plans.json").read_text())
        key, paths, plan_info = tuple(plans["morphology"]), plans["paths"], plans["info"]
        print(f"morphology {key}: reusing {out / 'plans.json'}", flush=True)
    else:
        print(f"morphology {key}: planning both whole flights", flush=True)
        with mp.Pool(2) as pool:
            res = pool.map(plan_one, [(key, fl) for fl in ("lower", "upper")])
        paths, plan_info = {}, {}
        for flight, status, path, secs in res:
            print(f"  {flight}: {status}, {len(path or []) - 1} moves, {secs:.0f} s", flush=True)
            if path is None:
                return 1
            paths[flight] = [[*st[:5], list(st[5])] for st in path]
            plan_info[flight] = dict(status=status, moves=len(path) - 1, seconds=round(secs, 1))
        (out / "plans.json").write_text(json.dumps(dict(morphology=key, paths=paths, info=plan_info)))

    from cleandog_sim.dynamics import pin_model
    from cleandog_sim.reference import round_trip, static_margins

    p, s = morphology(*key), StairParams()
    tup = {fl: [(*st[:5], tuple(st[5])) for st in pth] for fl, pth in paths.items()}
    ref, extra = round_trip(p, s, tup)
    ref.save(out / "reference.npz")
    margins = static_margins(pin_model(p), ref)
    clearance = clearance_audit(p, s, ref, out)
    n_up = extra["n_up"]
    segs = []
    for i, name in enumerate(ref.segment_names):
        sel = np.flatnonzero(ref.segment == i)
        segs.append(dict(name=name, start=float(sel[0] * ref.dt), duration=float(len(sel) * ref.dt),
                         min_margin=float(margins[sel].min())))
    segs.sort(key=lambda sg: sg["start"])
    crawl = [dict(base=c["base"].tolist(), leg=c["leg"], margin=c["margin"], com=c["com"].tolist(),
                  feet={l: f.tolist() for l, f in c["feet"].items()}) for c in extra["crawl_log"]]
    sway = {fl: dict(zip(*(a.tolist() for a in np.unique(sc, return_counts=True))))
            for fl, sc in extra["sway_scale"].items()}
    (out / "reference.json").write_text(json.dumps(dict(
        morphology=key, dt=ref.dt, duration=ref.duration, n_up=n_up, segments=segs, crawl=crawl,
        sway_frames={fl: {str(k): v for k, v in d.items()} for fl, d in sway.items()},
        margins=margins[::5].round(4).tolist(), margins_every=5, clearance=clearance,
        clear_viol={fl: float(v.max()) for fl, v in extra["clear_viol"].items()},
        margin_short={fl: float(v.max()) for fl, v in extra["margin_short"].items()},
        generated=dt.datetime.now().isoformat(timespec="seconds"))))
    print(f"reference: {ref.duration:.0f} s round trip, min 3D static margin {margins.min() * 100:.2f} cm -> {out}")
    for k, c in clearance.items():
        print(f"  clearance {k}: min {c['min']:.1f} mm at frame {c['frame']} ({c['segment']}, {c['pair']}), "
              f"{c['frames_below_0']} frames below 0")
    return 0


if __name__ == "__main__":
    sys.exit(main())
