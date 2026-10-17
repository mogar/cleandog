"""Re-run selected (morphology, scenario) plans with a larger search budget and merge them into
the latest transitions results, so 'budget exhausted' unknowns near the decision boundary
become answers.

Usage: python scripts/transition_recheck.py --segs 13 14 --knee back --facing up --time-limit 900
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from transition_sweep import run_one  # noqa: E402
from visualize import load_transitions  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--segs", nargs="+", type=float, required=True, help="segment lengths, cm")
    ap.add_argument("--knee", default="back")
    ap.add_argument("--facing", default="up")
    ap.add_argument("--budget", type=int, default=6000)
    ap.add_argument("--time-limit", type=float, default=900)
    ap.add_argument("--jobs", type=int, default=22)
    args = ap.parse_args(argv)
    src, data = load_transitions()
    segs = {round(s / 100, 4) for s in args.segs}
    todo = [r for r in data["rows"] if round(r["seg"], 4) in segs and r["knee"] == args.knee
            and r["scenario"].endswith(args.facing) and r["status"] != "ok"]
    print(f"rechecking {len(todo)} plans", flush=True)
    jobs = [(r["seg"], r["knee"], r["hip_spacing"], r["scenario"], args.budget, args.time_limit) for r in todo]
    with mp.Pool(min(args.jobs, len(jobs))) as pool:
        new = pool.map(run_one, jobs)
    index = {(r["seg"], r["knee"], r["hip_spacing"], r["scenario"]): i for i, r in enumerate(data["rows"])}
    for r in new:
        r["rechecked"] = dict(budget=args.budget, time_limit=args.time_limit)
        data["rows"][index[(r["seg"], r["knee"], r["hip_spacing"], r["scenario"])]] = r
        print(f"  {r['seg'] * 100:.0f} cm {r['knee']} {r['hip_spacing'] * 100:.1f} {r['scenario']}: {r['status']} "
              f"({r['expanded']} expansions)", flush=True)
    (src / "results.json").write_text(json.dumps(data, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
