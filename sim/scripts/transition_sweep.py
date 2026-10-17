"""Sweep the quasi-static transition planner over morphologies (M0 step 1).

Usage:
  python scripts/transition_sweep.py [--quick] [--jobs N]

Writes outputs/<date>_transitions/results.json: one row per (morphology,
scenario), with the planned path for successful plans. Then run
`python scripts/visualize.py transitions` to build the report.
"""

from __future__ import annotations

import argparse
import datetime as dt
import itertools
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

from cleandog_sim import StairParams
from cleandog_sim.transitions import KNEES, Planner, all_scenarios, expand_plan, morphology

REPO = Path(__file__).resolve().parents[2]


def run_one(args) -> dict:
    seg, knee, hs, scenario_name, budget, time_limit = args
    p = morphology(seg, knee, hs)
    sc = next(s for s in all_scenarios(StairParams(), p.body_length) if s.name == scenario_name)
    t0 = time.time()
    pl = Planner(p, sc, budget=budget)
    path, status = pl.plan(time_limit=time_limit)
    row = dict(seg=seg, knee=knee, hip_spacing=hs, scenario=scenario_name, status=status,
               expanded=pl.expanded, seconds=round(time.time() - t0, 1))
    if path:
        frames = expand_plan(pl, path)
        row.update(steps=len(path) - 1, min_margin=min(f.margin for f in frames),
                   peak_hip=max(t[0] for f in frames for t in f.torques.values()),
                   peak_knee=max(t[1] for f in frames for t in f.torques.values()),
                   path=[[*st[:5], list(st[5])] for st in path])
    elif pl.best_path:
        row["best_path"] = [[*st[:5], list(st[5])] for st in pl.best_path]
    return row


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="small grid for a smoke test")
    ap.add_argument("--jobs", type=int, default=max(1, mp.cpu_count() - 2))
    ap.add_argument("--budget", type=int, default=600)
    ap.add_argument("--time-limit", type=float, default=90.0)
    args = ap.parse_args(argv)

    if args.quick:
        segs, knees, spacings = (0.13, 0.16), ("back",), (0.292,)
    else:
        segs, knees, spacings = (0.10, 0.12, 0.13, 0.14, 0.16, 0.18), tuple(KNEES), (0.25, 0.292, 0.33)
    names = [s.name for s in all_scenarios(StairParams(), 0.42)]
    jobs = [(seg, k, hs, n, args.budget, args.time_limit)
            for seg, k, hs, n in itertools.product(segs, knees, spacings, names)]
    out = REPO / "outputs" / f"{dt.date.today().isoformat()}_transitions"
    out.mkdir(parents=True, exist_ok=True)
    print(f"{len(jobs)} plans on {args.jobs} workers -> {out}/results.json", flush=True)
    rows = []
    t0 = time.time()
    with mp.Pool(args.jobs) as pool:
        for i, row in enumerate(pool.imap_unordered(run_one, jobs), 1):
            rows.append(row)
            if i % 20 == 0 or i == len(jobs):
                ok = sum(r["status"] == "ok" for r in rows)
                print(f"  {i}/{len(jobs)} done, {ok} ok, {time.time() - t0:.0f}s", flush=True)
    (out / "results.json").write_text(json.dumps(dict(
        generated=dt.datetime.now().isoformat(timespec="seconds"), budget=args.budget,
        time_limit=args.time_limit, rows=rows), default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
