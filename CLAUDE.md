# CleanDog — house-tidying quadruped

`PLAN.md` is the project plan, with milestones M0–M7. M0 (sim sizing and stair
feasibility) lives in `sim/`. See `sim/README.md` for commands.

## Priority: visualizations and human understanding

This project puts **visualizations and human understanding** first. Every
analysis or major decision should come with something a person can look at
and check: figures, renders, and the maths written out.

- Every analysis gets a figure, not just a number. Prefer diagrams drawn in the
  real geometry (side views of the stairs, robot poses) over abstract plots.
- Show the 2D analysis next to the 3D MuJoCo model whenever both exist, and
  report how far apart they are, so each checks the other.
- Write out the derivation behind each major decision in the report: equations,
  assumptions, and what is *not* modelled.
- Look at every generated figure before reporting results. Many M0 bugs were
  found only by looking. Among them: a missing link thickness, wrong camera
  angles, a planner deadlock at the first riser, an IK branch flip, and swing
  legs cutting through nosings.
- Cross-check models against each other. The 2D planner, the MuJoCo replay and
  the Pinocchio/Crocoddyl model must agree; reports state the disagreement
  (for example, the penetration in the replays).

## Experimental outputs: `outputs/` (git-ignored)

Graphics, plots, renders and the write-ups behind decisions go in
`outputs/<YYYY-MM-DD>_<topic>/`. Each folder has a `report.md` that embeds its
images; open it with VS Code's Markdown preview. The folders are regenerated
by scripts, so they are not committed. When a result changes the plan,
record the conclusion in `PLAN.md`, because outputs are not versioned.

```bash
conda activate cleandog
cd sim && python scripts/visualize.py stairs|ladder|bins   # fast, self-contained reports
python scripts/transition_sweep.py && python scripts/visualize.py transitions   # M0 step 1 (~35 min)
python scripts/dynamics_run.py && python scripts/visualize.py dynamics          # M0 step 2 (~1 min)
python scripts/rl_reference.py && python scripts/rl_train.py                    # M0 step 3 (GPU, ~1 h)
python scripts/rl_eval.py && python scripts/visualize.py rl                     # round trips (~25 min)
python scripts/rl_diagnose.py && python scripts/visualize.py rl_failures        # shove map + forensics (~10 min)
```

Compute-heavy steps write their data (JSON/NPZ) into `outputs/`, and the
`visualize.py` command rebuilds the report from it, so figures can be
iterated without recomputing.

## Conventions

- **MJCF is the source of truth** for the robot. `cleandog_sim.robot`
  generates it, and Pinocchio and Crocoddyl load it directly.
- SI units everywhere. Frames: x forward/uphill, y left, z up. Knee sign
  convention: −1 means the knee points backward.
- Values tagged `ESTIMATE` in `params.py` are placeholders until M1 sysID or a
  datasheet replaces them.
- The stairwell model is the real one: 8 risers, a 78" × 35" landing with a
  180° turn, then 7 risers, with balusters on the open side of the lower
  flight. The return direction (`turn_left`) is still an assumption.
- Only one GPU job at a time: training preallocates most of the 3090, and a
  second job (an eval, a render) caused an out-of-memory crash.
- Environment: conda env `cleandog` (`environment.yml`). There is no display
  in agent sessions: render offscreen with `MUJOCO_GL=egl`. The interactive
  viewer (`visualize.py view`) is for Morgan's desktop.
