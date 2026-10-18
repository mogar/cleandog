# cleandog_sim — M0 sizing and stair feasibility

This project prioritizes **visualizations and human understanding** (see
`../CLAUDE.md`). Every analysis here writes figures and a Markdown report to
`../outputs/<date>_<topic>/`, which is git-ignored.

```bash
conda env create -f ../environment.yml   # once; then: conda activate cleandog
pytest                                    # geometry, model and kinematics checks

python scripts/visualize.py all           # every report below
python scripts/visualize.py stairs        # stairwell geometry + 3D views
python scripts/visualize.py ladder        # ladder-posture sweep, gallery, 2D-vs-3D poses
python scripts/visualize.py bins          # bin rim height study

# Interactive MuJoCo viewer (needs a display, i.e. your desktop terminal):
python scripts/visualize.py view                  # stairwell, robot standing at the bottom
python scripts/visualize.py view --ladder 13 3    # 13 cm segments in ladder posture, 3 cm belly gap
python scripts/visualize.py view --flat

# M0 steps 1-2 (compute first, then report):
python scripts/transition_sweep.py        # quasi-static stair-transition plans, all morphologies (~40 min, 22 cores)
python scripts/visualize.py transitions   # report + recommended morphology
python scripts/dynamics_run.py            # Crocoddyl on the recommended morphology's plans (~5 min)
python scripts/visualize.py dynamics      # torque/speed/friction report, actuator spec, 3D replays

# M0 step 3 (RL round trips; GPU):
python scripts/rl_reference.py            # plan both whole flights, build the round-trip reference (~3 min)
python scripts/rl_train.py                # PPO tracking policy on MJX/Warp, 4096 envs (~1 h for 400M steps)
python scripts/rl_eval.py                 # whole round trips: randomized, nominal, friction sweep, CPU MuJoCo cross-check
python scripts/visualize.py rl            # report, written into the same outputs/<date>_rl folder

python scripts/ladder_sweep.py            # the sweep as text tables (--csv out.csv)
python scripts/build_models.py            # write MJCF scenes to sim/models/
```

| Module | What it does |
| --- | --- |
| `params.py` | `RobotParams` (the sweep knobs), `StairParams` (your stairwell), actuator specs |
| `robot.py` | Parametric 12-DoF + 2-DoF neck MJCF; mass budget; stand pose |
| `stairs.py` | Bullnose-tread stairwell: 2D profile for analysis, plus the MJCF switchback with partition, landing walls and balusters |
| `scene.py` | Full scene assembly and `load_model()` |
| `ladder.py` | Ladder-posture analysis: stance stroke, static torque, belly friction, whole-body poses |
| `bin_reach.py` | Grid search over drop poses at a bin rim |
| `terrain.py` | 2D terrain for planners: shapes, walking surfaces, envelope, mirroring, cached occupancy grid |
| `transitions.py` | Quasi-static free-gait crawl planner through the stair transitions, with verified swing paths |
| `dynamics.py` | Crocoddyl whole-body trajectory optimisation along a plan (friction cones, torque bounds) |
| `reference.py` | Round-trip kinematic reference: whole-flight plans, the 3D landing crawl, reversal for rear-first descent |
| `rl.py` | MJX (Warp) tracking environment: residual PD policy, torque-speed envelope, rewards, domain randomization |
| `viz.py`, `render.py` | Matplotlib figures and offscreen MuJoCo renders |

MJCF is the single source of truth. Pinocchio and Crocoddyl load it directly
(`pinocchio.buildModelFromMJCF`), so a URDF export can wait until ROS needs one.

Actuator values marked `ESTIMATE` in `params.py` are placeholders until M1
sysID. The torque-speed envelope (`JointActuation.torque_limit`) can't be
expressed in MJCF, so the RL environment (`rl.py`) applies it to every PD torque.
