"""M0 step 2: Crocoddyl trajectory optimisation along the recommended morphology's plans.

Usage:
  python scripts/dynamics_run.py [--morph SEG_CM KNEE SPACING_CM] [--jobs N]

Reads the latest outputs/*_transitions/results.json. Writes
outputs/<date>_dynamics/<scenario>.npz plus summary.json. Then run
`python scripts/visualize.py dynamics` for the report.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from visualize import OUTPUTS, load_transitions, recommend  # noqa: E402

from cleandog_sim import StairParams, build_scene_xml  # noqa: E402
from cleandog_sim.transitions import Planner, all_scenarios, morphology  # noqa: E402


def penetration(p, s, q_traj, flight: str, facing: str) -> dict[str, float]:
    """Worst robot-world penetration per robot geom over a trajectory, replayed in MuJoCo."""
    import mujoco

    from cleandog_sim.render import planner_to_world_qpos

    m = mujoco.MjModel.from_xml_string(build_scene_xml(p, s))
    d = mujoco.MjData(m)
    worst: dict[str, float] = {}
    for q in q_traj:
        d.qpos[:] = planner_to_world_qpos(q, s, flight, facing)
        mujoco.mj_forward(m, d)
        for i in range(d.ncon):
            c = d.contact[i]
            names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) for g in (c.geom1, c.geom2)]
            robot = next((n for n in names if n and n != "floor" and n != "partition"), None)
            if robot and -c.dist > worst.get(robot, 0.0):
                worst[robot] = -c.dist
    return worst


def run_one(args):
    key, scenario, raw_path, out = args
    from cleandog_sim import dynamics as D

    s = StairParams()
    p = morphology(*key)
    sc = next(x for x in all_scenarios(s, p.body_length) if x.name == scenario)
    pl = Planner(p, sc)
    path = [(st[0], st[1], st[2], st[3], st[4], tuple(st[5])) for st in raw_path]
    t0 = time.time()
    r = D.solve(p, pl, path, max_iter=300)
    pen = penetration(p, s, r.q, sc.flight, sc.facing)
    np.savez_compressed(out / f"{scenario}.npz", t=r.t, tau=r.tau, qd=r.qd, q=r.q, friction=r.friction_used,
                        **{f"force_{leg}": f for leg, f in r.forces.items()})
    return dict(scenario=scenario, converged=r.converged, iterations=r.iterations, cost=r.cost,
                seconds=round(time.time() - t0, 1), duration=float(r.t[-1]), foot_err=r.foot_err,
                peak_tau=np.abs(r.tau).max(0).tolist(), rms_tau=np.sqrt((r.tau**2).mean(0)).tolist(),
                peak_qd=np.abs(r.qd).max(0).tolist(), friction_max=float(r.friction_used.max()),
                penetration=pen, joints=r.joint_names)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--morph", nargs=3, metavar=("SEG_CM", "KNEE", "SPACING_CM"))
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args(argv)
    _, data = load_transitions()
    rows = data["rows"]
    if args.morph:
        key = (float(args.morph[0]) / 100, args.morph[1], float(args.morph[2]) / 100)
    else:
        key, _ = recommend(rows)
    todo = [(key, r["scenario"], r["path"]) for r in rows
            if (r["seg"], r["knee"], r["hip_spacing"]) == key and r["status"] == "ok"]
    out = OUTPUTS / f"{dt.date.today().isoformat()}_dynamics"
    out.mkdir(parents=True, exist_ok=True)
    print(f"morphology {key}: {len(todo)} plans -> {out}", flush=True)
    with mp.Pool(min(args.jobs, len(todo))) as pool:
        results = []
        for res in pool.imap_unordered(run_one, [(*t, out) for t in todo]):
            print(f"  {res['scenario']}: converged={res['converged']} iters={res['iterations']} "
                  f"{res['seconds']}s friction={res['friction_max']:.2f} "
                  f"worst penetration={max(res['penetration'].values(), default=0) * 1000:.1f} mm", flush=True)
            results.append(res)
    (out / "summary.json").write_text(json.dumps(dict(morphology=key, results=results,
                                                      generated=dt.datetime.now().isoformat(timespec="seconds"))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
