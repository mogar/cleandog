"""M0 step 3a: plan both whole flights and build the round-trip reference.

Usage:
  python scripts/rl_reference.py [--morph SEG_CM KNEE SPACING_CM]

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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--morph", nargs=3, metavar=("SEG_CM", "KNEE", "SPACING_CM"))
    args = ap.parse_args(argv)
    if args.morph:
        key = (float(args.morph[0]) / 100, args.morph[1], float(args.morph[2]) / 100)
    else:
        key, _ = recommend(load_transitions()[1]["rows"])
    out = OUTPUTS / f"{dt.date.today().isoformat()}_rl"
    out.mkdir(parents=True, exist_ok=True)

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
        margins=margins[::5].round(4).tolist(), margins_every=5,
        generated=dt.datetime.now().isoformat(timespec="seconds"))))
    print(f"reference: {ref.duration:.0f} s round trip, min 3D static margin {margins.min() * 100:.2f} cm -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
