"""M0 first pass: print the ladder-posture stroke/torque sweep as text tables.

Usage: python scripts/ladder_sweep.py [--csv out.csv]
For figures, use: python scripts/visualize.py ladder
"""

from __future__ import annotations

import argparse
import csv
import sys

from cleandog_sim import StairParams
from cleandog_sim.ladder import belly_rest_friction, run_sweep

CSV_FIELDS = ("seg", "gap", "knee", "corner_stroke", "mid_stroke", "best_stroke", "hip_torque", "knee_torque")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", help="write every row to this CSV")
    args = ap.parse_args(argv)

    s = StairParams()
    print(f"Stairs: slope {s.slope * 57.2958:.1f} deg, nosing pitch {s.nosing_pitch * 100:.1f} cm, "
          f"pocket depth {s.pocket_depth * 100:.1f} cm, total rise {s.total_rise:.2f} m\n")

    mus = (0.15, 0.3, 0.5, 0.9)
    print("Foot friction needed to rest with part of the weight on the skid plate:")
    print("  belly share  " + "  ".join(f"skid mu={mu:.2f}" for mu in mus))
    for share in (0.25, 0.5, 0.75, 1.0):
        print(f"  {share:>10.2f}   " + "  ".join(f"{belly_rest_friction(s, share, mu):>11.2f}" for mu in mus))

    rows = run_sweep(s)
    print("\nStance stroke with the foot planted (cm). Torques: foot carries 1/2 body weight.")
    print(f"{'seg cm':>6} {'gap cm':>6} {'knee':>5} | {'corner':>6} {'mid':>6} {'best':>6} "
          f"| {'hip Nm':>6} {'knee Nm':>7}")
    for r in rows:
        print(f"{r['seg'] * 100:>6.0f} {r['gap'] * 100:>6.0f} {r['knee']:>5} | "
              f"{r['corner_stroke'] * 100:>6.1f} {r['mid_stroke'] * 100:>6.1f} {r['best_stroke'] * 100:>6.1f} | "
              f"{r['hip_torque']:>6.1f} {r['knee_torque']:>7.1f}")
    print(f"\nOne nosing pitch is {s.nosing_pitch * 100:.1f} cm; a stroke at least that long "
          "lets each foot advance one pocket per stride.")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
