"""Does capping trunk pitch force a bigger robot? (M0 step 3e)

Usage:
  python scripts/pitch_cap_sweep.py [--jobs N] [--time-limit S]

The step-1 planner is rerun on the four uphill-facing transitions (the ones the
round trip uses) with |trunk pitch| capped at several levels, for knees-back
robots of several leg lengths and hip spacings. A morphology passes a cap if all
four transitions plan. Failures are labelled as in step 1: "no plan within
discretisation" means the search ran out of options; "search budget exhausted"
is unresolved.

Writes outputs/<date>_pitch-cap/results.json. Then run
`python scripts/visualize.py pitch_cap`.
"""

from __future__ import annotations

import argparse
import datetime as dt
import itertools
import json
import math
import multiprocessing as mp
import sys
import time
from pathlib import Path

from cleandog_sim import StairParams
from cleandog_sim.transitions import Planner, all_scenarios, expand_plan, morphology

REPO = Path(__file__).resolve().parents[2]
SCENARIOS = ("lower-bottom-up", "lower-top-up", "upper-bottom-up", "upper-top-up")
CAPS_DEG = (None, 50.0, 45.0, math.degrees(StairParams().slope), 40.0, 35.0)


def run_one(args) -> dict:
    seg, hs, cap, name, budget, time_limit = args
    p = morphology(seg, "back", hs)
    sc = next(s for s in all_scenarios(StairParams(), p.body_length) if s.name == name)
    t0 = time.time()
    pl = Planner(p, sc, budget=budget, max_pitch=None if cap is None else math.radians(cap))
    path, status = pl.plan(time_limit=time_limit)
    row = dict(seg=seg, hip_spacing=hs, cap=cap, scenario=name, status=status, expanded=pl.expanded,
               seconds=round(time.time() - t0, 1))
    if path:
        frames = expand_plan(pl, path)
        row.update(steps=len(path) - 1, max_pitch=math.degrees(max(abs(st[2]) for st in path)),
                   min_margin=min(f.margin for f in frames),
                   peak_knee=max(t[1] for f in frames for t in f.torques.values()),
                   path=[[*st[:5], list(st[5])] for st in path])
    elif pl.best_path:
        row["best_path"] = [[*st[:5], list(st[5])] for st in pl.best_path]
    return row


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", type=int, default=max(1, mp.cpu_count() - 2))
    ap.add_argument("--budget", type=int, default=1500)
    ap.add_argument("--time-limit", type=float, default=240.0)
    args = ap.parse_args(argv)
    segs, spacings = (0.16, 0.17, 0.18, 0.20), (0.25, 0.292)
    jobs = [(seg, hs, cap, n, args.budget, args.time_limit)
            for seg, hs, cap, n in itertools.product(segs, spacings, CAPS_DEG, SCENARIOS)]
    out = REPO / "outputs" / f"{dt.date.today().isoformat()}_pitch-cap"
    out.mkdir(parents=True, exist_ok=True)
    print(f"{len(jobs)} plans on {args.jobs} workers -> {out}/results.json", flush=True)
    rows, t0 = [], time.time()
    with mp.Pool(args.jobs) as pool:
        for i, row in enumerate(pool.imap_unordered(run_one, jobs), 1):
            rows.append(row)
            if i % 16 == 0 or i == len(jobs):
                ok = sum(r["status"] == "ok" for r in rows)
                print(f"  {i}/{len(jobs)} done, {ok} ok, {time.time() - t0:.0f}s", flush=True)
            (out / "results.json").write_text(json.dumps(dict(
                generated=dt.datetime.now().isoformat(timespec="seconds"), budget=args.budget,
                time_limit=args.time_limit, caps=CAPS_DEG, rows=rows), default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
