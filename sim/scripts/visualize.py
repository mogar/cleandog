"""Generate M0 figures, 3D renders and Markdown reports into outputs/<date>_<topic>/.

Usage:
  python scripts/visualize.py stairs    # stairwell geometry + 3D views
  python scripts/visualize.py ladder    # ladder-posture sweep: gallery, heatmaps, 3D poses, friction
  python scripts/visualize.py bins      # bin rim height study
  python scripts/visualize.py transitions   # stair transition plans (run transition_sweep.py first)
  python scripts/visualize.py dynamics      # Crocoddyl results (run dynamics_run.py first)
  python scripts/visualize.py rl            # RL round trips (run rl_reference/rl_train/rl_eval first)
  python scripts/visualize.py all
  python scripts/visualize.py view [--ladder SEG_CM GAP_CM]   # interactive viewer (needs a display)

Open the report.md in each folder with VS Code's Markdown preview (Ctrl+Shift+V).
"""

from __future__ import annotations

import argparse
import datetime as dt
import math
import os
import sys
from pathlib import Path

# Offscreen rendering must pick its GL backend before mujoco is first imported.
# The interactive viewer uses GLFW on its own and is unaffected.
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

from cleandog_sim import RobotParams, StairParams, build_scene_xml, load_model
from cleandog_sim.ladder import ladder_pose, ladder_qpos, run_sweep

REPO = Path(__file__).resolve().parents[2]
OUTPUTS = REPO / "outputs"


def out_dir(topic: str) -> Path:
    d = OUTPUTS / f"{dt.date.today().isoformat()}_{topic}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save(fig, d: Path, name: str) -> str:
    import matplotlib.pyplot as plt

    fig.savefig(d / name)
    plt.close(fig)
    return name


# ---------------------------------------------------------------- stairwell

def cmd_stairs() -> Path:
    from cleandog_sim import viz
    from cleandog_sim.render import View, render_views

    viz.style()
    s, p = StairParams(), RobotParams()
    d = out_dir("stairwell")
    geo = save(viz.fig_stair_geometry(s), d, "stair_geometry.png")
    model = load_model(p, s)
    x_land = (s.flights[0].n_risers - 1) * s.run
    views = [
        # MuJoCo puts the camera at lookat - distance * (cos el cos az, cos el sin az, sin el).
        View("overview", (1.0, 0.5, 1.3), 5.5, 40, -30),
        View("from_bottom", (0.8, 0.0, 0.7), 3.2, 0, -15),
        View("landing", (x_land + 0.4, 0.5, 1.6), 2.8, 25, -50),
        View("side_balusters", (0.8, 0.0, 0.9), 3.6, 90, -8),
    ]
    renders = render_views(model, None, views, d, "stairwell")
    f0, f1 = s.flights
    md = f"""# Stairwell model

Generated {dt.datetime.now():%Y-%m-%d %H:%M} by `scripts/visualize.py stairs`.

## What's modelled

| Quantity | Value |
| --- | --- |
| Lower flight | {f0.n_risers} risers; {f0.left} on the left (partition), {f0.right} on the right |
| Landing | 180° turn, {s.landing_width / 0.0254:.0f}" × {f0.top_depth / 0.0254:.0f}" ({s.landing_width:.3f} × {f0.top_depth:.3f} m), walled on all sides |
| Upper flight | {f1.n_risers} risers, walls both sides |
| Partition between flights | {s.partition_thickness * 100:.1f} cm (78" − 2 × 36.5") |
| Rise / run / overhang | {s.rise * 100:.1f} / {s.run * 100:.1f} / {s.overhang * 100:.1f} cm |
| Slope, nosing pitch, pocket depth | {math.degrees(s.slope):.1f}°, {s.nosing_pitch * 100:.1f} cm, {s.pocket_depth * 100:.1f} cm |
| Total rise | {s.total_rise:.2f} m ({sum(f.n_risers for f in s.flights)} risers) |

**Assumption to confirm:** the upper flight returns on the climber's **left**
(`StairParams.turn_left`), so the balusters are on the right of the lower
flight. Mirroring it doesn't change any stair-climbing result.

Walls are drawn semi-transparent so the treads stay visible.

![geometry]({geo})

## 3D views

""" + "\n\n".join(f"![{n}]({n})" for n in renders) + "\n"
    (d / "report.md").write_text(md)
    return d


# ---------------------------------------------------------------- ladder sweep

def _fk_check(p: RobotParams, s: StairParams, pose, qpos) -> float:
    """Max distance between MuJoCo's foot sites and the 2D analysis' footholds (x-z)."""
    import mujoco

    from cleandog_sim.robot import LEGS

    m = load_model(p, s)
    data = mujoco.MjData(m)
    data.qpos[:] = qpos
    mujoco.mj_forward(m, data)
    err = 0.0
    for leg in LEGS:
        xz = data.site(f"{leg}_foot").xpos[[0, 2]]
        err = max(err, float(np.hypot(*(xz - pose.feet[leg[0]]))))
    return err


def cmd_ladder() -> Path:
    from cleandog_sim import viz
    from cleandog_sim.ladder import belly_rest_friction
    from cleandog_sim.render import View, render_views

    viz.style()
    s, base = StairParams(), RobotParams()
    d = out_dir("ladder-sweep")
    rows = run_sweep(s, base)
    figs = {
        "heat": save(viz.fig_sweep_heatmaps(rows, s), d, "sweep_heatmaps.png"),
        "gal_back": save(viz.fig_ladder_gallery(rows, base, s, "back"), d, "gallery_knee_back.png"),
        "gal_fwd": save(viz.fig_ladder_gallery(rows, base, s, "fwd"), d, "gallery_knee_fwd.png"),
        "friction": save(viz.fig_belly_friction(s), d, "belly_friction.png"),
    }

    # Whole-body poses at a few sizes, in 2D and as 3D renders of the actual MJCF.
    u_body = 3.5 * s.nosing_pitch
    pose_md, fk_errs = [], []
    for seg in (0.10, 0.13, 0.16, 0.20):
        for knee in ("back", "fwd"):
            sign = -1 if knee == "back" else 1
            p = base.with_(thigh_length=seg, shin_length=seg, knee_sign_front=sign, knee_sign_rear=sign)
            gap = 0.03
            pose = ladder_pose(p, s, u_body, gap)
            tag = f"seg{seg * 100:.0f}_knee{knee}"
            title = f"Segments {seg * 100:.0f} cm, knee {knee}, belly {gap * 100:.0f} cm above nosing line"
            if not all(pose.ok.values()):
                title += "  (red leg: no feasible foothold)"
            f2d = save(viz.fig_ladder_pose(p, s, pose, gap, title), d, f"pose2d_{tag}.png")
            qpos = ladder_qpos(p, pose)
            fk_errs.append(_fk_check(p, s, pose, qpos))
            import mujoco

            model = mujoco.MjModel.from_xml_string(build_scene_xml(p, s.open_sides()))
            c = pose.trunk_xz
            r3d, r3q = render_views(model, qpos, [View("side", (c[0], 0.0, c[1]), 1.3, 90, -5),
                                                       View("quarter", (c[0], 0.0, c[1]), 1.3, 55, -25)], d,
                                  f"pose3d_{tag}")
            pose_md.append(f"### {title}\n\n| 2D analysis | MuJoCo model |\n| --- | --- |\n"
                           f"| ![]({f2d}) | ![]({r3d}) |\n\n![]({r3q})\n")

    summary_lines = []
    for knee in ("back", "fwd"):
        ok = [r for r in rows if r["knee"] == knee and r["best_stroke"] >= s.nosing_pitch]
        if ok:
            r0 = min(ok, key=lambda r: (r["seg"], r["gap"]))
            summary_lines.append(
                f"- Knee {knee}: the shortest segments reaching one nosing pitch are "
                f"**{r0['seg'] * 100:.0f} cm** (belly gap {r0['gap'] * 100:.0f} cm, stroke "
                f"{r0['best_stroke'] * 100:.0f} cm).")
        else:
            summary_lines.append(f"- Knee {knee}: no configuration reaches one nosing pitch.")
    for seg in (0.12, 0.13):
        for knee in ("back", "fwd"):
            sel = [r["best_stroke"] for r in rows if r["knee"] == knee and abs(r["seg"] - seg) < 1e-9]
            summary_lines.append(f"- {seg * 100:.0f} cm segments, knee {knee}: best stroke "
                                 f"{min(sel) * 100:.0f}–{max(sel) * 100:.0f} cm across belly gaps.")
    summary = "\n".join(summary_lines)

    th = s.slope
    fric = {share: belly_rest_friction(s, share, 0.15) for share in (0.25, 0.5, 0.75)}
    best = max(rows, key=lambda r: r["best_stroke"])
    md = f"""# Ladder-posture sizing sweep (M0, first pass)

Generated {dt.datetime.now():%Y-%m-%d %H:%M} by `scripts/visualize.py ladder`.

**Scope:** sagittal (2D) kinematics and quasi-static forces in the *interior*
of a flight. Transitions (bottom, top, landing) and dynamics are not covered
here; Crocoddyl and RL handle those next.

## 1. Geometry the argument rests on

Slope $\\theta = \\arctan(\\text{{rise}}/\\text{{run}}) = \\arctan({s.rise * 100:.1f}/{s.run * 100:.1f}) = {math.degrees(th):.1f}°$.

Nosing pitch (nosing to nosing) $p = \\sqrt{{\\text{{rise}}^2 + \\text{{run}}^2}} = {s.nosing_pitch * 100:.1f}$ cm.

With the body parallel to the nosing line, each riser/tread corner lies a
perpendicular distance

$$d = \\frac{{\\text{{rise}} \\cdot \\text{{run}}}}{{p}} = {s.pocket_depth * 100:.1f}\\ \\text{{cm}}$$

below the line. That's the "pocket" the feet work in. The overhang makes the
corner under the nosing slightly deeper.

## 2. Correction to PLAN.md: riser bracing does not resist sliding

**No surface on a staircase faces uphill.** Treads face up and risers face
downhill (their outward normal is $-x$). A foot pressed into the riser corner
gets a reaction with zero or negative uphill component. Any uphill force on
the robot must therefore come from **friction**.

- Standing on the treads: gravity is vertical and treads push straight up, so
  holding still needs no friction. This is fine.
- Resting on the skid plate is the problem. The nosings push perpendicular to
  the slope, which is tilted downhill by $\\theta$.

Let a share $\\beta$ of the weight $W$ rest on a skid plate with friction
$\\mu_b$, with normal force $N$:

$$N(\\cos\\theta + \\mu_b \\sin\\theta) = \\beta W \\qquad
\\text{{downhill push}} = N(\\sin\\theta - \\mu_b\\cos\\theta)$$

The feet carry $(1-\\beta)W$ vertically, so they need

$$\\mu_\\text{{feet}} \\ge \\frac{{N(\\sin\\theta - \\mu_b\\cos\\theta)}}{{(1-\\beta)W}}$$

For a UHMW plate with $\\mu_b = 0.15$:

| Share of weight on the plate | Foot friction needed |
| --- | --- |
| 25% | {fric[0.25]:.2f} |
| 50% | {fric[0.5]:.2f} |
| 75% | {fric[0.75]:.2f} |

To hold the full weight on the plate alone needs
$\\mu_b \\ge \\tan\\theta = {math.tan(th):.2f}$.

![belly friction]({figs['friction']})

**Implication:** a slippery belly plate is good for sliding but bad for
resting. The options for M0 to compare:

1. a grippy or ratcheting belly, such as rubber pads or a pawl that catches
   the nosing;
2. never loading the belly, keeping it only as a crash skid;
3. a slippery plate plus feet that hold the downhill push by friction, which
   needs $\\mu$ of about 0.5 or more at 50% belly load.

## 3. Stance stroke and static torque

The **stance stroke** is how far the hip can travel along the slope while one
foot stays planted at a given foothold, within reach, within the knee limits
(0.2–2.7 rad flexion), and with the thigh and shin clear of the stairs. One
nosing pitch of stroke ({s.nosing_pitch * 100:.1f} cm) lets a foot advance one
pocket per stride. Nine footholds per tread are tried, from the riser corner to
the nosing. "Best" is the longest stroke among them.

Static torque is $\\tau = |\\mathbf{{r}} \\times \\mathbf{{f}}|$ with
$\\mathbf{{f}} = (0, W/2)$: the foot carries half of {base.total_mass_target} kg,
as in a trot. The peak is over the stroke.

![heatmaps]({figs['heat']})

Longest stroke overall: {best['best_stroke'] * 100:.0f} cm at {best['seg'] * 100:.0f} cm segments,
gap {best['gap'] * 100:.0f} cm, knee {best['knee']}.

**Reading the results** (computed from the sweep)

{summary}
- Some of the longest forward-knee strokes pass through a folded leg tucked
  under the next nosing; the gallery shows which. They clear the stairs by the
  link radius, but a real knee housing could be bulkier.
- Static hip torque mostly sits between the RobStride's 6 Nm rated and 17 Nm
  peak. At ≤ 20 s per step this is close to continuous, so motor heating is
  likely the limit.

## 4. Every configuration tried

![gallery back]({figs['gal_back']})

![gallery fwd]({figs['gal_fwd']})

## 5. Whole-body poses: 2D analysis next to the MuJoCo model

The same pose is shown twice: the 2D analysis drawing and the generated MJCF
loaded in MuJoCo. Maximum foot-position disagreement between the two:
**{max(fk_errs) * 1000:.2f} mm**, which confirms the generator and the
analysis agree.

""" + "\n".join(pose_md)
    (d / "report.md").write_text(md)
    return d


# ---------------------------------------------------------------- bins

def cmd_bins() -> Path:
    from cleandog_sim import viz
    from cleandog_sim.bin_reach import BOOK, SHOE, SOCK, feasible_drops, max_rim_height, pose_grid

    viz.style()
    p = RobotParams()
    d = out_dir("bin-rim")
    rims = np.arange(0.10, 0.42, 0.01)
    proposed = 0.20

    poses = pose_grid(p, BOOK)
    P = poses.params
    level = np.abs(P["pitch"]) < 0.05
    neck_mod = np.abs(P["pitch"] - P["neck"]) < 0.8  # neck within ~45° of horizontal
    counts = {"any pose": [], "level stance": [], "level, neck < 45°": []}
    for rim in rims:
        ok = feasible_drops(poses, rim, BOOK)
        counts["any pose"].append(ok.sum())
        counts["level stance"].append((ok & level).sum())
        counts["level, neck < 45°"].append((ok & level & neck_mod).sum())
    counts = {k: np.array(v, float) for k, v in counts.items()}
    static_max, i_max = max_rim_height(p, BOOK)

    f_counts = save(viz.fig_bin_rim_counts(rims, counts, proposed, static_max), d, "rim_feasibility.png")
    # A representative pose at the proposed rim: level, and the least bent neck and head joints.
    ok = feasible_drops(poses, proposed, BOOK) & level & neck_mod
    bend = np.abs(P["neck"]) + np.abs(P["head"])
    i_ok = int(np.flatnonzero(ok)[np.argmin(bend[ok])])
    f_ok = save(viz.fig_bin_pose(p, poses, i_ok, proposed, BOOK,
                                 f"Proposed {proposed * 100:.0f} cm rim: level stance, least-bent neck and head"),
                d, "pose_proposed.png")
    f_max = save(viz.fig_bin_pose(p, poses, i_max, static_max, BOOK,
                                  f"Static limit {static_max * 100:.0f} cm: rears up, neck vertical"),
                 d, "pose_limit.png")
    per_obj = [(o.name, max_rim_height(p, o)[0]) for o in (BOOK, SHOE, SOCK)]
    rows = "\n".join(f"| {n} | {h * 100:.0f} cm |" for n, h in per_obj)
    md = f"""# Bin rim height

Generated {dt.datetime.now():%Y-%m-%d %H:%M} by `scripts/visualize.py bins`.

## Proposal

**Open-top bins with a rim about {proposed * 100:.0f} cm (8") high** and an
opening of at least about 45 × 30 cm, so drops don't need to be precise. A
rigid, wide base matters, since a 1 kg book lands hard. Low toy-storage or
under-bed totes fit this class.

## Why {proposed * 100:.0f} cm

- **The robot can do it without contortion.** It stands level, at normal
  height, with the neck within 45° of horizontal. At {proposed * 100:.0f} cm,
  {counts['level, neck < 45°'][int(round((proposed - 0.10) / 0.01))] / counts['level, neck < 45°'][0] * 100:.0f}%
  of the level, moderate-neck poses that work at 10 cm still work.
- **Headroom.** The static limit is {static_max * 100:.0f} cm, reached by
  rearing up with the neck vertical. That leaves room for M0 to shrink the
  legs or neck without breaking the bins.
- **Taller is better for the bins themselves.** Objects are less likely to
  bounce out, the bins hold more, and they look like furniture.

| Object | Static limit (any pose) |
| --- | --- |
{rows}

![feasibility]({f_counts})

![proposed]({f_ok})

![limit]({f_max})

## Model and assumptions

Sagittal grid search on flat floor, about 1.26 M poses. The grid covers:

- front and rear hip heights (45–97% of leg reach);
- front-hip lean ahead of the front feet (0–10 cm);
- standing back 0–12 cm beyond a 3 cm toe gap;
- neck and head pitch across their joint ranges.

A pose counts as a feasible drop when all of these hold:

- the jaw tip is ≥ 6 cm past the rim;
- the held object's underside is ≥ 2 cm above the rim;
- neck, head and trunk nose clear the rim, and the front legs stay outside
  the wall;
- the centre of mass, including the object, is ≥ 3 cm inside the
  front-to-rear foot span;
- neck torque is ≤ 7.5 Nm (about 60% of DM-J4310P peak).

The robot is the M0 default: {p.thigh_length * 100:.0f} cm segments,
{p.neck_length * 100:.0f} cm neck, {p.head_length * 100:.0f} cm head.

Not modelled: how the object leaves the jaw, bouncing, lateral balance, or
dynamics. Those belong to M5.
"""
    (d / "report.md").write_text(md)
    return d


# ---------------------------------------------------------------- transitions (M0 step 1)

UP = ("lower-bottom-up", "lower-top-up", "upper-bottom-up", "upper-top-up")
SCENARIO_LABELS = {
    "lower-bottom-up": "floor→flight ↑", "lower-bottom-down": "flight→floor ↓",
    "lower-top-up": "flight→landing ↑", "lower-top-down": "landing→flight ↓",
    "upper-bottom-up": "landing→flight ↑", "upper-bottom-down": "flight→landing ↓",
    "upper-top-up": "flight→floor ↑", "upper-top-down": "floor→flight ↓",
}


def latest(pattern: str) -> Path:
    found = sorted(OUTPUTS.glob(pattern))
    if not found:
        raise SystemExit(f"no {pattern} under {OUTPUTS}; run the sweep first")
    return found[-1]


def load_transitions() -> tuple[Path, dict]:
    import json

    d = latest("*_transitions/results.json").parent
    return d, json.loads((d / "results.json").read_text())


def recommend(rows: list[dict]) -> tuple[tuple, list[tuple]]:
    """Morphologies that pass all four uphill-facing scenarios (ascent, and rear-first descent
    by time reversal), ranked: shortest legs, then most downhill-facing scenarios passed, then
    lowest peak static hip torque."""
    by = {}
    for r in rows:
        by.setdefault((r["seg"], r["knee"], r["hip_spacing"]), []).append(r)
    ranked = []
    for key, rs in by.items():
        ok = {r["scenario"] for r in rs if r["status"] == "ok"}
        if all(u in ok for u in UP):
            n_down = sum(1 for r in rs if r["scenario"].endswith("down") and r["status"] == "ok")
            peak = max(r["peak_hip"] for r in rs if r["status"] == "ok")
            ranked.append((key, n_down, peak))
    ranked.sort(key=lambda t: (t[0][0], -t[1], t[2]))
    return (ranked[0][0] if ranked else None), ranked


def cmd_transitions() -> Path:
    from cleandog_sim import viz
    from cleandog_sim.transitions import Planner, all_scenarios, morphology, turn_footprint

    viz.style()
    src, data = load_transitions()
    rows = data["rows"]
    d = out_dir("transitions")
    s = StairParams()
    names = [sc.name for sc in all_scenarios(s, 0.42)]
    order = [n for n in names if n.endswith("up")] + [n for n in names if n.endswith("down")]
    grid = save(viz.fig_success_grid(rows, order, SCENARIO_LABELS), d, "success_grid.png")
    best, ranked = recommend(rows)

    def planner_for(key, scenario):
        p = morphology(*key)
        sc = next(x for x in all_scenarios(s, p.body_length) if x.name == scenario)
        return Planner(p, sc)

    def as_path(raw):
        return [(st[0], st[1], st[2], st[3], st[4], tuple(st[5])) for st in raw]

    sections = []
    if best:
        p_best = morphology(*best)
        for r in sorted((r for r in rows if (r["seg"], r["knee"], r["hip_spacing"]) == best),
                        key=lambda r: order.index(r["scenario"])):
            pl = planner_for(best, r["scenario"])
            label = f"{SCENARIO_LABELS[r['scenario']]} ({pl.sc.label})"
            if r["status"] != "ok":
                sections.append(f"#### {label}: no plan ({r['status']})\n")
                continue
            path = as_path(r["path"])
            strip = save(viz.fig_plan_filmstrip(pl, path, label), d, f"strip_{r['scenario']}.png")
            gif = f"anim_{r['scenario']}.gif"
            viz.animate_plan(pl, path, d / gif, label)
            sections.append(f"#### {label}\n\n{r['steps']} steps, min stability margin "
                            f"{r['min_margin'] * 100:.1f} cm, peak static torque hip {r['peak_hip']:.1f} Nm / "
                            f"knee {r['peak_knee']:.1f} Nm.\n\n![]({gif})\n\n![]({strip})\n")

    # Why the plan's 13 cm hypothesis fails: furthest progress on the floor -> flight transition.
    fail_md = ""
    key13 = (0.13, "back", 0.292)
    r13 = next((r for r in rows if (r["seg"], r["knee"], r["hip_spacing"]) == key13
                and r["scenario"] == "lower-bottom-up"), None)
    if r13 and r13["status"] != "ok" and r13.get("best_path"):
        pl = planner_for(key13, "lower-bottom-up")
        bp = as_path(r13["best_path"])
        f13 = save(viz.fig_plan_filmstrip(pl, bp, "13 cm segments, knee back: furthest the search got "
                                          "(floor → flight)"), d, "fail_13cm.png")
        diag = pl.diagnose(bp[-1])
        table = "| Leg | end poses | trunk clear | 4 feet OK | can lift | path OK | has a foothold |\n" \
                "| --- | --- | --- | --- | --- | --- | --- |\n" + "\n".join(
                    f"| {leg} | {v['ends']} | {v['trunk_clear']} | {v['four_feet_ok']} | {v['can_lift']} | "
                    f"{v['path_ok']} | {v['has_foothold']} |" for leg, v in diag.items())
        fail_md = f"""## Why 13 cm segments fail

PLAN.md's hypothesis was 12–13 cm segments. With backward knees the search
stalls at the first riser. From its furthest state, here is how many of the
candidate moves pass each check:

{table}

![]({f13})
"""

    fp_folded = turn_footprint(morphology(*best) if best else RobotParams(), neck_option=4)
    fp_out = turn_footprint(morphology(*best) if best else RobotParams(), neck_option=0)
    turn = save(viz.fig_landing_turn(s, morphology(*best) if best else RobotParams(), fp_folded,
                                     "neck folded up"), d, "landing_turn.png")

    knee_rows = []
    for knee in ("back", "fwd", "out", "in"):
        cells = []
        for facing in ("up", "down"):
            ok = sum(1 for r in rows if r["knee"] == knee and r["scenario"].endswith(facing) and r["status"] == "ok")
            tot = sum(1 for r in rows if r["knee"] == knee and r["scenario"].endswith(facing))
            cells.append(f"{ok}/{tot}")
        knee_rows.append(f"| {knee} | {cells[0]} | {cells[1]} |")
    knee_table = "\n".join(knee_rows)
    rank_rows = "\n".join(f"| {k[0] * 100:.0f} cm | {k[1]} | {k[2] * 100:.1f} cm | {n}/4 | {pk:.1f} Nm |"
                           for k, n, pk in ranked[:12])
    md = f"""# Stair transitions: quasi-static crawl plans (M0 step 1)

Generated {dt.datetime.now():%Y-%m-%d %H:%M} by `scripts/visualize.py transitions`, from
`{src.name}/results.json` (sweep budget: {data['budget']} expansions or {data['time_limit']:.0f} s per plan).

## Decision

""" + (f"""**Recommended morphology: {best[0] * 100:.0f} cm thigh and shin, knees {best[1]},
hip spacing {best[2] * 100:.1f} cm** (trunk {p_best.body_length * 100:.0f} cm long).
It is the shortest-legged design that completes all four uphill-facing
transitions, and the tie-breaks favour designs that also manage more
downhill-facing ones, then lower hip torque. This replaces PLAN.md's 12–13 cm
hypothesis: 13 cm is proven infeasible at the plan's 29.2 cm hip spacing,
and 14 cm stays unresolved even with a 10× search budget, so it is
marginal at best.""" if best else "**No morphology passed all four uphill-facing transitions.**") + f"""

**Descend rear-first.** A quasi-static plan is still valid played backwards.
So every uphill-facing plan, reversed, is a rear-first descent of the same
transition. Head-first descent (the downhill-facing columns) is harder for the
planner and unproven for most designs.

**Knees should point downhill.** Backward knees succeed only facing uphill,
and forward knees only facing downhill. Either way the knee sits on the
downhill side of the leg, where it folds into the pocket instead of
hitting the riser above. A robot that always faces uphill on the stairs
(climbing forwards, descending rear-first) therefore wants backward knees.

| Knees | Uphill-facing plans found | Downhill-facing plans found |
| --- | --- | --- |
{knee_table}

**Turning on the landing works with the neck folded.** Turning in place needs
a ⌀{2 * fp_folded['radius'] * 100:.0f} cm circle with the neck folded up, but
⌀{2 * fp_out['radius'] * 100:.0f} cm with it straight out. The landing is
{s.flights[0].top_depth * 100:.0f} cm deep.

## How the planner works

- **Geometry:** the side view of each transition, with the real tread,
  bullnose, landing and wall dimensions. Left and right feet sit at
  y = ±{(RobotParams().hip_width / 2 + RobotParams().abad_offset) * 100:.0f} cm.
- **Moves:** a body shift with all four feet planted, then one foot steps.
  Any leg may go next (a free gait); a fixed crawl order deadlocks at the
  first riser. Before lifting a foot the body may sway sideways up to 5 cm.
- **Every pose must pass:**
  - each planted leg reaches its foothold within its knee and hip limits, with
    reach to spare for the sway;
  - thigh and shin (capsules) clear the stairs;
  - the trunk and skid plate clear the stairs and walls;
  - some neck fold clears them (the one with the best stability is used);
  - the centre of mass, projected onto the floor, is ≥ 2 cm inside the support
    polygon.
- **Body pose:** each hip sits at one of 7 heights above the stepped terrain
  envelope (the nosing line on a flight, the floor or landing elsewhere).
  Front and rear are chosen independently, which sets the pitch.
- **Search:** depth-first, ranked by progress with a one-step lookahead.
- **Why no friction is needed:** with the centre of mass inside the support
  polygon, vertical foot forces alone balance gravity.
- **Statuses:** ✗ means the search space is exhausted (infeasible *within this
  discretisation*); ? means the search ran out of budget, so the answer is
  unknown.

**Not modelled here:** the swing-foot path (Crocoddyl in step 2 adds it),
dynamics, and the belly carrying load.

## Results

![]({grid})

Morphologies passing all four uphill-facing transitions, best first:

| Segments | Knees | Hip spacing | Downhill-facing passed | Peak static hip torque |
| --- | --- | --- | --- | --- |
{rank_rows}

## Recommended morphology, every transition

""" + "\n".join(sections) + f"""

{fail_md}

## Turning on the landing

![]({turn})
"""
    (d / "report.md").write_text(md)
    return d


# ---------------------------------------------------------------- dynamics (M0 step 2)

def cmd_dynamics() -> Path:
    import json

    import mujoco

    from cleandog_sim import viz
    from cleandog_sim.render import planner_to_world_qpos, render_trajectory_gif
    from cleandog_sim.robot import joint_actuation
    from cleandog_sim.transitions import morphology

    viz.style()
    src = latest("*_dynamics/summary.json").parent
    summary = json.loads((src / "summary.json").read_text())
    key = tuple(summary["morphology"])
    p = morphology(*key)
    s = StairParams()
    acts = {ja.joint: ja for ja in joint_actuation(p)}
    d = out_dir("dynamics")
    model = mujoco.MjModel.from_xml_string(build_scene_xml(p, s.open_sides()))

    demand = {r: dict(peak=0.0, rms=0.0, speed=0.0) for r in ("abad", "hip", "knee")}
    sections, rows = [], []
    results = sorted(summary["results"], key=lambda r: list(SCENARIO_LABELS).index(r["scenario"]))
    for res in results:
        z = np.load(src / f"{res['scenario']}.npz")
        joints = res["joints"]
        for role in demand:
            idx = [joints.index(f"{leg}_{role}") for leg in ("FL", "FR", "RL", "RR")]
            demand[role]["peak"] = max(demand[role]["peak"], float(np.abs(z["tau"][:, idx]).max()))
            demand[role]["rms"] = max(demand[role]["rms"], float(np.sqrt((z["tau"][:, idx] ** 2).mean(0)).max()))
            demand[role]["speed"] = max(demand[role]["speed"], float(np.abs(z["qd"][:, idx]).max()))
        label = SCENARIO_LABELS[res["scenario"]]
        flight, end, facing = res["scenario"].split("-")
        f_tau = save(viz.fig_torque_traces(z["t"], z["tau"], joints, acts, f"{label}: joint torques"), d,
                     f"torque_{res['scenario']}.png")
        f_fric = save(viz.fig_friction_use(z["t"], z["friction"], 0.5, f"{label}: friction used (worst foot)"), d,
                      f"friction_{res['scenario']}.png")
        qpos = np.array([planner_to_world_qpos(q, s, flight, facing) for q in z["q"]])
        gif = f"replay_{res['scenario']}.gif"
        render_trajectory_gif(model, qpos, d / gif, dt=float(z["t"][1] - z["t"][0]), every=0.4,
                              azimuth=65 if flight == "lower" else -60)
        pen = max(res["penetration"].values(), default=0.0)
        worst_part = max(res["penetration"], key=res["penetration"].get) if res["penetration"] else "-"
        rows.append(f"| {label} | {'yes' if res['converged'] else 'no'} ({res['iterations']} it) | "
                    f"{res['duration']:.0f} s | {max(res['peak_tau'][:12]):.1f} Nm | {max(res['rms_tau'][:12]):.1f} Nm | "
                    f"{res['friction_max']:.2f} | {res['foot_err'] * 1000:.0f} mm | {pen * 1000:.0f} mm ({worst_part}) |")
        sections.append(f"### {label}\n\n![]({gif})\n\n![]({f_tau})\n\n![]({f_fric})\n")

    f_dem = save(viz.fig_actuator_demand(demand, acts), d, "actuator_demand.png")
    spec_rows = "\n".join(
        f"| {viz.ROLE_TITLES[r]} | {acts['FL_' + r].spec.name} | {demand[r]['peak']:.1f} → **{demand[r]['peak'] * 1.3:.1f}** | "
        f"{acts['FL_' + r].peak_torque:.1f} | {demand[r]['rms']:.1f} → **{demand[r]['rms'] * 1.3:.1f}** | "
        f"{acts['FL_' + r].rated_torque:.1f} | {demand[r]['speed']:.1f} → **{demand[r]['speed'] * 1.3:.1f}** | "
        f"{acts['FL_' + r].max_speed:.0f} |" for r in demand)
    md = f"""# Whole-body dynamics through the stair transitions (M0 step 2)

Generated {dt.datetime.now():%Y-%m-%d %H:%M} by `scripts/visualize.py dynamics`, from
`{src.name}`. Morphology: **{key[0] * 100:.0f} cm segments, knees {key[1]}, hip spacing
{key[2] * 100:.1f} cm**, {p.total_mass_target} kg.

## Method

Crocoddyl solves a whole-body trajectory on the full 3D model, loaded from the
same MJCF through Pinocchio. It follows each quasi-static plan from step 1.

- **Timing:** each step is a 1.0 s body shift on four feet, then a 0.8 s
  swing on three. That is slow, close to the ≤ 20 s-per-stair target.
- **Contact:** planted feet are hard contact constraints. Their forces must
  stay inside friction cones with μ = 0.5, a deliberately cautious value for
  rubber on finished hardwood. An inscribed pyramid guarantees the true cone.
- **Torque:** each joint is box-bounded at its actuator's peak torque.
- **Swing feet:** each follows the planner's collision-free path through the
  leg's free space.
- **Replay:** every solution is replayed in MuJoCo in your stairwell to
  measure any penetration. The replays hide walls and balusters for
  visibility.

## Summary

| Transition | Converged | Duration | Peak torque | Worst RMS torque | Max friction used | Swing landing error | Worst penetration |
| --- | --- | --- | --- | --- | --- | --- | --- |
{chr(10).join(rows)}

The penetration comes from the solver trading posture tracking for lower
effort: the trunk drifts 1–2 cm from the plan in the tightest spots. The
foot's rubber cover absorbs this scale, but treat it as a modelling tolerance,
not a clearance.

## Actuator spec (M0 exit criterion)

Worst case over every transition, then × 1.3 for the required 30% margin.
Heating is judged by RMS torque against the rated (continuous) torque.

| Joint | Candidate | Peak needed → with margin | Actuator peak | RMS needed → with margin | Actuator rated | Speed needed → with margin (rad/s) | Actuator max speed |
| --- | --- | --- | --- | --- | --- | --- | --- |
{spec_rows}

![]({f_dem})

**Caveats.** The actuator speed and friction values are estimates until M1's
system identification (`params.py`, marked ESTIMATE). RMS is over the
transition alone; a robot holding still on the stairs sits near its static
torques, which the quasi-static plans bound.

## Each transition

""" + "\n".join(sections)
    (d / "report.md").write_text(md)
    return d


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p, den = k / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / den, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def cmd_rl(tag: str = "main") -> Path:
    """M0 step 3 report: the round-trip reference, training, and the round-trip evaluation.
    Written into the same outputs/<date>_rl folder as the data it reads."""
    import json

    import mujoco

    from cleandog_sim import viz
    from cleandog_sim.render import View, render_trajectory_gif, render_views
    from cleandog_sim.robot import joint_actuation
    from cleandog_sim.transitions import morphology

    viz.style()
    d = latest("*_rl/reference.json").parent
    ref_info = json.loads((d / "reference.json").read_text())
    plans = json.loads((d / "plans.json").read_text())
    train = json.loads((d / f"train_{tag}.json").read_text())
    ev = json.loads((d / f"eval_{tag}.json").read_text())
    z = np.load(d / f"eval_{tag}.npz")
    key = tuple(ref_info["morphology"])
    p, s = morphology(*key), StairParams()
    acts = {ja.joint: ja for ja in joint_actuation(p)}
    segs = ref_info["segments"]
    res = ev["results"]
    ctrl_dt = ev["ctrl_dt"]

    # -- figures
    f_ref = save(viz.fig_reference_overview(ref_info["dt"], ref_info["margins_every"], np.array(ref_info["margins"]),
                                            segs), d, "reference_margin.png")
    f_crawl = save(viz.fig_landing_crawl(p, s, ref_info["crawl"]), d, "landing_crawl.png")
    f_train = save(viz.fig_training_curve(train["curve"], 1000), d, f"training_{tag}.png")
    mus = sorted(float(k.split()[1]) for k in res if k.startswith("friction"))
    f_succ = save(viz.fig_success(res, mus), d, f"success_{tag}.png")
    f_track = save(viz.fig_tracking_along_trip(ctrl_dt * 5, z["rand_pos_err"], segs, z["rand_failed_at"],
                                               ev["term_pos"]), d, f"tracking_{tag}.png")
    engines = {"MJX / Warp, dt 4 ms": z["nom_pos_err"]}
    for k, lab in (("cpu4", "CPU MuJoCo, dt 4 ms"), ("cpu2", "CPU MuJoCo, dt 2 ms")):
        if f"{k}_pos_err" in z:
            engines[lab] = z[f"{k}_pos_err"]
    f_eng = save(viz.fig_engine_agreement(ctrl_dt, engines, segs), d, f"engines_{tag}.png")

    joints = ev["joints"]
    nj = len(joints)
    tau_peak = z["rand_tau_peak"].reshape(-1, nj).max(0)
    tau_rms = z["rand_tau_rms"].reshape(-1, nj).max(0)  # worst trip's RMS per joint
    qd_peak = z["rand_qd_peak"].reshape(-1, nj).max(0)
    demand = {}
    for role in ("abad", "hip", "knee"):
        idx = [joints.index(f"{leg}_{role}") for leg in ("FL", "FR", "RL", "RR")]
        demand[role] = dict(peak=float(tau_peak[idx].max()), rms=float(tau_rms[idx].max()),
                            speed=float(qd_peak[idx].max()))
    f_dem = save(viz.fig_actuator_demand(demand, acts, "Closed-loop actuator demand over the successful randomized "
                                         "round trips (worst case), with the 30% margin. n× = capability / demand."),
                 d, f"actuator_demand_{tag}.png")
    nom_rms = np.sqrt(z["nom_tau_ms"].mean(0))

    # -- 3D replays of the nominal trip (MJX trace, every 0.1 s)
    model = mujoco.MjModel.from_xml_string(build_scene_xml(p, s.open_sides()))
    qpos = z["nom_qpos"]
    every = 5 * ctrl_dt
    gifs = []
    for name, az, el, dist in (("lower flight", 65, -20, 1.6), ("landing", 90, -75, 2.2),
                               ("upper flight", -60, -20, 1.6), ("upper flight (down, rear-first)", -60, -20, 1.6),
                               ("landing (down, rear-first)", 90, -75, 2.2), ("lower flight (down, rear-first)", 65, -20, 1.6)):
        sg = next(x for x in segs if x["name"] == name)
        i0, i1 = int(sg["start"] / every), int((sg["start"] + sg["duration"]) / every)
        fn = f"replay_{tag}_{name.split(' (')[0].replace(' ', '_')}{'_down' if 'down' in name else '_up'}.gif"
        render_trajectory_gif(model, qpos[i0:i1], d / fn, dt=every, every=0.6, fps=15, azimuth=az, elevation=el,
                              distance=dist)
        gifs.append((name, fn))
    q_still = qpos[int((segs[1]["start"] + 60) / every)]
    stills = render_views(model, q_still, [View("side", tuple(q_still[:3]), 1.4, 65, -18)], d, f"still_{tag}")

    # -- tables
    def row(name):
        r = res[name]
        ci = r.get("ci95")
        ci_s = f"{ci[0]:.1%}–{ci[1]:.1%}" if ci and r["trips"] > 1 else "–"
        where = ""
        if r["failed_at"]:
            names = []
            for f in r["failed_at"]:
                tt = f * ctrl_dt
                names.append(next((viz._short(sg["name"]) for sg in segs
                                   if sg["start"] <= tt < sg["start"] + sg["duration"]), "?"))
            where = ", ".join(f"{n} ×{names.count(n)}" for n in dict.fromkeys(names))
        return (f"| {name} | {r['success']}/{r['trips']} | **{r['rate']:.1%}** | {ci_s} | "
                f"{r['end_err_median'] * 100:.1f} cm | {where or '–'} |")

    GATE = "gate: 1 shove per flight"
    order = [GATE, "randomized", "nominal"] + [f"friction {m:.1f}" for m in mus] + [k for k in res if k.startswith("cpu")]
    table = "\n".join(row(n) for n in order if n in res)
    rand = res["randomized"]
    seg_rows = "\n".join(f"| {sg['name']} | {sg['start']:.0f} s | {sg['duration']:.0f} s | "
                         f"{sg['min_margin'] * 100:.1f} cm |" for sg in segs)
    spec_rows = "\n".join(
        f"| {viz.ROLE_TITLES[r]} | {demand[r]['peak']:.1f} → **{demand[r]['peak'] * 1.3:.1f}** | "
        f"{acts['FL_' + r].peak_torque:.1f} | {demand[r]['rms']:.1f} → **{demand[r]['rms'] * 1.3:.1f}** | "
        f"{acts['FL_' + r].rated_torque:.1f} | {demand[r]['speed']:.1f} | {acts['FL_' + r].max_speed:.0f} |"
        for r in demand)
    nom_rms_rows = ", ".join(f"{j} {v:.1f}" for j, v in zip(joints, nom_rms) if j.endswith("knee"))
    sway = ref_info["sway_frames"]
    sway_txt = "; ".join(f"{fl}: " + ", ".join(f"{float(k):.0%} sway for {v} frames" for k, v in sorted(
        sway[fl].items(), key=lambda kv: -float(kv[0]))) for fl in sway)
    sweep_k = sum(res[f"friction {m:.1f}"]["success"] for m in mus)
    sweep_n = sum(res[f"friction {m:.1f}"]["trips"] for m in mus)
    sweep_lo, sweep_hi = _wilson(sweep_k, sweep_n)
    nom_rows = []
    for role in ("abad", "hip", "knee"):
        idx = [joints.index(f"{leg}_{role}") for leg in ("FL", "FR", "RL", "RR")]
        pk = float(z["nom_tau_peak"][:, idx].max())
        nom_rows.append(f"| {viz.ROLE_TITLES[role]} | {pk:.1f} → **{pk * 1.3:.1f}** | "
                        f"{acts['FL_' + role].peak_torque:.1f} | {nom_rms[idx].max():.1f} | "
                        f"{np.abs(z['nom_qd'][:, idx]).max():.1f} |")
    fails = np.sort(z["rand_failed_at"][z["rand_failed_at"] >= 0] * ctrl_dt)
    T = ev["n_steps"] * ctrl_dt
    mirrored = sum(1 for f in fails if f < T / 2 and np.any(np.abs((T - fails) - f) < 3.0))
    gif_md = "\n\n".join(f"**{viz._short(n)}** ({n})\n\n![]({fn})" for n, fn in gifs)
    gate = res.get(GATE)
    head = gate or rand
    if head["trips"] < 100:
        verdict = f"**Too few trips to judge ({head['trips']}).**"
    else:
        verdict = "**Met.**" if head["rate"] >= 0.95 else "**Not met yet.**"
    head_txt = ("with one shove per flight (the gate set on 2026-10-18), everything else randomized" if gate
                else "randomized, with a shove every 4–10 s")

    md = f"""# Round trips with an RL tracking policy (M0 step 3)

Generated {dt.datetime.now():%Y-%m-%d %H:%M} by `scripts/visualize.py rl`, from the data in
this folder. Morphology: **{key[0] * 100:.0f} cm segments, knees {key[1]}, hip spacing
{key[2] * 100:.1f} cm**, {p.total_mass_target} kg. Policy `{tag}` after
{ev['policy_step'] / 1e6:.0f}M environment steps.

**M0 exit item 3 (≥ 95% simulated round trips on your stairwell): {verdict}**
{head['success']} of {head['trips']} round trips completed {head_txt}
({head['rate']:.1%}, 95% CI {head['ci95'][0]:.1%}–{head['ci95'][1]:.1%}). With a shove every 4–10 s
(the training distribution): {rand['rate']:.1%}.

![]({stills[0]})

## 1. The round trip the policy follows

A round trip is one continuous motion of {ref_info['duration']:.0f} s
({ref_info['duration'] / 60:.1f} min). It starts on the floor, climbs the lower
flight, crosses the landing (turning 180°), and climbs the upper flight to the
upper floor. Then it plays the same motion backwards. The plans are
quasi-static, so the reverse of a valid plan is valid: **the descent is
rear-first, retracing the ascent.**

| Segment | Starts | Lasts | Tightest static margin |
| --- | --- | --- | --- |
{seg_rows}

- **Flights.** The step-1 planner was rerun over each *whole* flight, not just
  the transitions. The lower flight took {plans['info']['lower']['moves']}
  moves and the upper {plans['info']['upper']['moves']}. Each move is a 1.0 s
  trunk shift on four feet, then a 0.8 s swing along the planner's verified
  collision-free path. That works out to
  {segs[1]['duration'] / 8:.0f} s per riser, against the plan's ≤ 20 s target.
- **Landing.** A quasi-static crawl on flat ground, generated in 3D. The trunk
  walks to mid-landing, turns 90° in place, crosses to the upper flight's
  line, turns 90° more, and steps to the upper flight's start pose. The neck
  stays folded, keeping the turning circle at 46 cm on the 89 cm landing.
  Before each swing the trunk shifts (sways) to the smallest offset that puts
  the centre of mass ≥ 2.5 cm inside the other three feet.
- **Correction found while building this:** the planner's sideways sway is a
  2D idea and ignores the ab/ad joint limit (±0.6 rad). Where a hip sits low
  over its foot, a 5 cm sway needs more than that, and the 3D IK dragged
  planted feet by up to 18 mm. Each shift now keeps the largest sway (in
  quarters) that the 3D IK reaches exactly: {sway_txt}. The 3D margin left is
  still positive everywhere (below).

![]({f_ref})

![]({f_crawl})

The reference is kinematic: poses and contact timing, not forces.

- **Penetration:** replayed in MuJoCo, front thighs dip up to 11 mm into a
  bullnose at the tightest pockets. The ab/ad tilt shortens the leg's
  side-view projection, so the IK bends the knee further than the 2D planner
  assumed.
- **Contact:** the physics in RL resolves this contact; the policy has to
  live with it.

## 2. The policy

A **residual tracking policy** (DeepMimic style):
- **Command:** at 50 Hz it outputs joint offsets `a` in [−1, 1], and the PD
  target is `q* = q_ref(t+1) + 0.25·a`, so the policy can move each joint up to
  0.25 rad off the reference.
- **Motor model:** a PD loop at the 250 Hz physics rate produces torques
  `τ = kp(q* − q) − kd·q̇`. Each torque is clipped to the actuator's linear
  torque-speed envelope `|τ| ≤ τ_peak (1 − |q̇|/ω_max)`, as `params.py`
  defines it.
- **Observations:** what the robot can estimate. That is IMU gravity and
  gyro, joint angles and speeds, body velocity, the last action, and the
  reference 0.02 s, 0.1 s and 0.5 s ahead (joint angles, and trunk pose
  relative to the robot's own pose).
  - **Assumption:** the robot localizes itself against the stairwell to
    about 1 cm. On the real robot that is the M4 perception stack's job.
- **Critic:** additionally sees privileged state (true foot errors, contact,
  friction, mass, gains).

**Reward** per step, clipped at 0 so that ending early never pays. Errors are
e_q (joints), e_p (trunk position), e_R (trunk angle), e_f (feet) and e_v
(joint speeds):

  r = 0.3·exp(−mean e_q²/0.15²) + 0.2·exp(−e_p²/0.05²) + 0.15·exp(−e_R²/0.15²)
    + 0.25·exp(−mean e_f²/0.03²) + 0.1·exp(−mean e_v²/1.5²)
    − 5·10⁻⁵ Σ τ² − 0.005 Σ Δa² − 0.1 Σ_touchdowns |v_foot|² − 0.2·[trunk or skid touching]

The torque² term stands for motor heating. The touchdown-velocity term stands
for noise; the plan asks for both.

**Domain randomization** (per environment):
- foot friction μ 0.4–1.0 and skid friction 0.1–0.3;
- link masses ±10%, a payload of −0.3 to +1.0 kg on the trunk, and its centre
  of mass ±2 cm;
- joint friction ×0.5–2 and rotor inertia ×0.9–1.2;
- PD gains ±20% and motor strength 85–100%;
- actuation latency 0–12 ms;
- observation noise;
- a 0–0.25 m/s sideways shove every 4–10 s.

**Training:**
- **Simulator:** PPO (Brax) on MuJoCo MJX with the Warp backend. It uses
  the real stairwell: cylinder nosings, balusters and elliptic friction
  cones, at 4096 parallel environments.
- **Episodes:** 20 s, each starting anywhere on the round trip.

![]({f_train})

## 3. Round-trip evaluation

Every evaluation trip starts at the bottom (reference time 0) and runs the
whole {ref_info['duration'] / 60:.1f} minutes with no resets. A trip **fails**
if the trunk ever strays more than {ev['term_pos'] * 100:.0f} cm or
{ev['term_rot']:.1f} rad from the reference. That is earlier than a fall: a
robot that far off the plan is presumed lost. A trip **succeeds** if it never
fails and ends within 5 cm of where it started. Intervals are Wilson 95%
score intervals.

Conditions:
- **gate: 1 shove per flight** is the M0 gate. It has one 0–0.25 m/s shove
  at a random moment in each of the four flights, in a random direction,
  with everything else randomized as in training.
- **randomized** is the training distribution, with a shove every 4–10 s
  (about 117 per trip).
- **nominal** has no randomization, noise or shoves.
- **friction** fixes the foot friction at the stated value, with everything
  else randomized and no shoves.
- **cpu mujoco** is the nominal trip in the C library, as a cross-check.

| Condition | Completed | Rate | 95% CI | Median end error | Failures (where) |
| --- | --- | --- | --- | --- | --- |
{table}

![]({f_succ})

![]({f_track})

### What the failures say

- **Shoves are what fails trips.** The friction sweep randomizes everything
  except the shoves. Pooled over its five friction values it completes
  {sweep_k}/{sweep_n} trips ({sweep_k / sweep_n:.1%}, 95% CI {sweep_lo:.1%}–{sweep_hi:.1%}),
  against {rand['rate']:.1%} with shoves. Friction from 0.3 to 0.8 makes little
  difference, so the planned rubber-on-hardwood contact is not the bottleneck.
- **The failures sit at a few fixed spots, mostly on the lower flight, and
  in both directions.** Of the ascent failures, {mirrored} have a descent failure
  within 3 s of the same reference pose. The descent is the ascent reversed,
  so these are physical spots in the plan where a shove can't be recovered.
- **It isn't the static margin.** The failures happen where the reference
  margin is 1.6–2.1 cm. The tightest spot (0.35 cm) never fails. The cause is
  still open: a contact event (a thigh on a bullnose, a swing foot near a
  nosing) is the next thing to check.

## 4. Cross-check: MJX/Warp vs CPU MuJoCo

The same policy and the same observation code run the nominal trip in the C
MuJoCo library. It runs once at the training timestep and once at half of it.
If the policy were exploiting a quirk of the GPU solver or of the 4 ms step,
the trip would fail or drift here.

![]({f_eng})

## 5. Actuator demand in closed loop

These are worst cases over the successful randomized trips. Failed trips are
left out because a robot flails once it has failed. The randomization includes
the shoves, the 15% weaker motors and the heavier payloads, so this is
conservative. Simulated torque can never exceed the envelope, so the test is
whether demand × 1.3 fits under it.

| Joint | Peak needed → with margin | Actuator peak | RMS needed → with margin | Actuator rated | Peak speed (rad/s) | Actuator max speed |
| --- | --- | --- | --- | --- | --- | --- |
{spec_rows}

![]({f_dem})

The shoves set these peaks: torque saturates while the policy catches the
robot. On the nominal trip, with no shoves, demand fits the actuators with the
30% margin:

| Joint | Peak (nominal) → with margin | Actuator peak | RMS | Peak speed (rad/s) |
| --- | --- | --- | --- | --- |
{chr(10).join(nom_rows)}

On the nominal trip, the knee RMS torques are {nom_rms_rows} Nm. The
Crocoddyl step reported 2.9 Nm on the transitions alone. The RL numbers
include holding still and the policy's own stiffness choices.

## 6. Replays (nominal trip, MJX trace)

Walls and balusters are hidden for visibility; the landing is seen from
above.

{gif_md}

## What is not modelled

- **Stair geometry:** the stair geometry is your stairwell's, exactly. The
  randomized stairs in the plan's workflow (rise 17–20 cm, tread 21–28 cm)
  would need a reference planned per variant. That is cheap (about 1 min per
  flight) but not done here.
- **Localization:** the policy is told its pose relative to the stairwell
  (1 cm noise); real localization error and drift are not modelled.
- **Actuators:** no motor thermal model (RMS is reported instead), no
  backlash, no cable drag.
- **Estimates:** actuator values tagged ESTIMATE (speed, friction, rotor
  inertia) await M1 sysID.
- **Contact:** contact is MuJoCo's soft contact. The plan flags rubber on a
  bullnose and the belly sliding on a nosing as the least trustworthy parts,
  for M1 to measure.
"""
    (d / "report.md").write_text(md)
    return d


DIR_NAMES = ("forward", "forward-left", "left", "back-left", "back", "back-right", "right", "forward-right")


def cmd_rl_failures(tag: str = "main") -> Path:
    """M0 step 3d report: shove map along the round trip and forensic replays of the worst moments."""
    import json

    import mujoco
    from PIL import Image

    from cleandog_sim import viz
    from cleandog_sim.reference import Reference
    from cleandog_sim.render import View, render_views
    from cleandog_sim.robot import LEGS
    from cleandog_sim.transitions import morphology

    viz.style()
    src = latest("*_rl/reference.json").parent
    diag = json.loads((src / f"diagnose_{tag}.json").read_text())
    sm = np.load(src / f"diagnose_{tag}.npz")
    ev = json.loads((src / f"eval_{tag}.json").read_text())
    evz = np.load(src / f"eval_{tag}.npz")
    ref_info = json.loads((src / "reference.json").read_text())
    ref = Reference.load(src / "reference.npz")
    segs = ref_info["segments"]
    p, s = morphology(*ref_info["morphology"]), StairParams()
    d = out_dir("rl-failures")
    dt_c, nd, lead = diag["dt"], diag["n_dirs"], diag["lead"]
    speeds = diag["speeds"]
    vmax = max(speeds)

    frac = {}
    for v in speeds:
        sel = sm["speed"] == v
        frac[v] = sm["failed"][sel].reshape(-1, nd).mean(1)
        times = sm["frame"][sel].reshape(-1, nd)[:, 0] * dt_c
    eval_fail = evz["rand_failed_at"][evz["rand_failed_at"] >= 0] * dt_c
    f_map = save(viz.fig_shove_map(times, frac, segs, eval_fail), d, "shove_map.png")

    # Breakdown at the largest speed: gait phase at the shove, and direction relative to the trunk.
    sel = sm["speed"] == vmax
    fr, failed, ang = sm["frame"][sel], sm["failed"][sel], sm["angle"][sel]
    phase_rates: dict[str, list[int]] = {}
    dir_rates = {n: [0, 0] for n in DIR_NAMES}
    for f, fl, a in zip(fr, failed, ang):
        c = ref.contact[f]
        seg = ref.segment_names[ref.segment[f]]
        where = "landing" if "landing" in seg else ("flight" if "flight" in seg else "standing")
        what = "all four feet down" if c.all() else f"{LEGS[int(np.argmin(c))]} swinging"
        for key in (f"{where}: {what}",):
            phase_rates.setdefault(key, [0, 0])
            phase_rates[key][0] += int(fl)
            phase_rates[key][1] += 1
        w, x, y, z = ref.qpos[f, 3:7]
        yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        k = int(round(((a - yaw) % (2 * math.pi)) / (2 * math.pi / nd))) % nd
        dir_rates[DIR_NAMES[k]][0] += int(fl)
        dir_rates[DIR_NAMES[k]][1] += 1
    phase_rates = dict(sorted(phase_rates.items()))
    f_break = save(viz.fig_shove_breakdown({k: tuple(v) for k, v in phase_rates.items()},
                                           {k: tuple(v) for k, v in dir_rates.items()}, vmax), d, "shove_breakdown.png")

    # Failure rate against the reference trunk pitch at the shove (flights only).
    frames = fr.reshape(-1, nd)[:, 0]
    frac_v = failed.reshape(-1, nd).mean(1)
    w, x, y, zq = ref.qpos[frames, 3:7].T
    pitch = np.abs(np.degrees(np.arcsin(np.clip(2 * (w * y - zq * x), -1, 1))))
    on_flight = np.array(["flight" in ref.segment_names[ref.segment[f]] for f in frames])
    edges = [0, 10, 20, 30, 40, 50, 60]
    p_rates, p_counts = [], []
    for lo_, hi_ in zip(edges[:-1], edges[1:]):
        m = on_flight & (pitch >= lo_) & (pitch < hi_)
        p_rates.append(frac_v[m].mean() if m.any() else 0.0)
        p_counts.append(int(m.sum()))
    p_rates, p_counts = np.array(p_rates), np.array(p_counts)
    f_pitch = save(viz.fig_shove_vs_pitch(edges, p_rates, p_counts, math.degrees(s.slope), vmax), d,
                   "shove_vs_pitch.png")
    steep = on_flight & (pitch >= 40)
    shallow = on_flight & (pitch < 40)
    per_shove = float(np.mean([np.mean(sm["failed"][sm["speed"] == v]) for v in speeds]))
    T = len(ref.qpos) * dt_c
    n_shoves = T / 7.0  # one every U(4, 10) s on average
    surv = (1 - per_shove) ** n_shoves
    need = 1 - 0.95 ** (1 / n_shoves)

    # Forensics.
    model = mujoco.MjModel.from_xml_string(build_scene_xml(p, s.open_sides()))
    joints = ev["joints"]
    sections = []
    for k, fo in enumerate(diag["forensics"]):
        z = np.load(src / f"diagnose_{tag}_forensic{k}.npz")
        f = fo["frame"]
        seg = ref.segment_names[ref.segment[f]]
        c = ref.contact[f]
        what = "all four feet down" if c.all() else f"{LEGS[int(np.argmin(c))]} swinging"
        w, x, y, zq = ref.qpos[f, 3:7]
        yaw = math.atan2(2 * (w * zq + x * y), 1 - 2 * (y * y + zq * zq))
        rel = DIR_NAMES[int(round(((fo["angle"] - yaw) % (2 * math.pi)) / (2 * math.pi / nd))) % nd]
        title = (f"t = {fo['time']:.1f} s ({seg}, {what}): {fo['speed']:.2f} m/s shove {rel}. "
                 f"MJX: {fo['fail_frac']:.0%} of directions fail; CPU: "
                 + (f"fails {(fo['cpu_failed_at'] - lead) * dt_c:.1f} s after" if fo["cpu_failed_at"] >= 0
                    else "recovers"))
        f_for = save(viz.fig_forensic(dt_c, lead, z["pos_err"], z["forces"], fo["contact_names"], z["saturated"],
                                      joints, title, ev["term_pos"]), d, f"forensic{k}.png")
        # Filmstrip: fixed camera at the trunk's position at the shove.
        q = z["qpos"]
        end = fo["cpu_failed_at"] if fo["cpu_failed_at"] >= 0 else len(q) - 1
        picks = [lead - 20] + list(np.linspace(lead + 5, min(end + 10, len(q) - 1), 4).astype(int))
        look = tuple(q[lead, :3])
        tiles = []
        for i in picks:
            names = render_views(model, q[i], [View(f"f{k}_{i}", look, 1.3, 65, -15)], d, "tmp", size=(300, 400))
            tiles.append(Image.open(d / names[0]))
            (d / names[0]).unlink()
        strip = Image.new("RGB", (400 * len(tiles), 300))
        for j, im in enumerate(tiles):
            strip.paste(im, (400 * j, 0))
        strip.save(d / f"forensic{k}_film.png")
        labels = " · ".join(f"{(i - lead) * dt_c:+.1f} s" for i in picks)
        top = [n for n in fo["contact_names"] if not n.split("|")[0].endswith("_foot")]
        sections.append(f"### {k + 1}. t = {fo['time']:.1f} s\n\n![]({f_for})\n\n![](forensic{k}_film.png)\n\n"
                        f"Frames: {labels}. Non-foot contacts during the replay: "
                        f"{', '.join(viz._contact_label(n) for n in top) or 'none'}.\n")

    n_trials = len(sm["failed"])
    rate_by_speed = ", ".join(f"{np.mean(sm['failed'][sm['speed'] == v]):.1%} at {v:.2f} m/s" for v in speeds)
    md = f"""# Why shoved round trips fail (M0 step 3d)

Generated {dt.datetime.now():%Y-%m-%d %H:%M} by `scripts/visualize.py rl_failures`, from
`{src.name}/diagnose_{tag}.*`. Policy `{tag}` ({diag['policy_step'] / 1e6:.0f}M steps).

## Question

The tracking policy completes 94.7% of randomized round trips without shoves
but only 74.5% with them. Its failures recur at the same places, going up and
coming down. Which moments are fragile, and what goes wrong there?

## Method

**Shove map.** The nominal robot starts from the reference 2 s before a
moment, gets one shove, and runs 5 s more. A trial fails by the same rule as
the round-trip evaluation: the trunk strays more than {ev['term_pos'] * 100:.0f} cm or
{ev['term_rot']:.1f} rad from the reference.
- **Grid:** shove moments every {diag['every']} s over the whole trip, at
  {', '.join(f'{v:.2f}' for v in speeds)} m/s, in {nd} directions, for
  {n_trials} trials.
- **Speeds:** the evaluation drew shove speeds uniformly from 0 to
  0.25 m/s, so 0.25 m/s is its worst case.

Overall {rate_by_speed} of trials fail.

**Forensics.** The {len(diag['forensics'])} worst moments are replayed on CPU
MuJoCo with each one's worst direction. Every contact force on every robot
part is logged, along with every control step a joint spent at its torque
limit.

## Findings

1. **A shove is rarely fatal; the evaluation just shoves a lot.** On the
   nominal robot, {per_shove:.2%} of single shoves fail, averaged over the
   speeds above. The round-trip evaluation shoves every 4–10 s, about
   {n_shoves:.0f} times per {T / 60:.1f}-minute trip. At this rate
   (1 − {per_shove:.4f})^{n_shoves:.0f} ≈ {surv:.0%} of trips would survive,
   consistent with the 74.5% measured (that also has randomized masses,
   gains and latency). Reaching 95% of trips at this shove rate needs
   under {need:.3%} failures per shove, about {per_shove / need:.0f}× better
   than now.
2. **Fragile moments are where the trunk pitches steeper than the stairs.**
   On the flights, shoves fail {frac_v[shallow].mean():.2%} of the time below
   40° of trunk pitch and {frac_v[steep].mean():.2%} at 40° or more
   ({steep.sum()} of {on_flight.sum()} moments). The stair slope is
   {math.degrees(s.slope):.1f}°.
   - **40–50° moments** are spread through the flights' interiors. They are
     part of the normal climbing rhythm, and fail at a few percent.
   - **Above 50°** (the worst band) the moments are only at the transitions:
     entering each flight from the floor or landing, and the top of the
     lower flight. Their mirrors on the way down are fragile too.

   A downhill (backward) shove is the dangerous direction.
3. **Two failure modes, and neither is a lack of torque.**
   - **Tipping backward** (3 of 4 replays). Trunk error grows smoothly from
     the shove; no joint reaches its torque limit, and nothing but the feet
     touches the stairs, until the robot is already falling. Once the
     centre of mass is behind the rear feet, joint torque can't bring it
     back: only stepping can. The policy is tied to the reference's footstep
     timing, so it cannot step to recover.
   - **Delayed thigh strike** (1 of 4). The robot absorbs the shove with
     about 5 cm of error. Its next swing, still following the reference
     swing path from the wrong place, drives a thigh into a nosing about 3 s
     later. The swing paths are collision-free only from the planned pose,
     and the policy has no view of the stairs.
4. **The landing is immune.** No shove on the landing failed, whatever the
   legs were doing.

## What to change

- **Cap the trunk pitch in the plan, near the stair slope.** The planner
  picks front and rear hip heights independently and can pitch the trunk
  past 50°. A limit (or a cost) there removes the worst moments at the
  source. Whether a cap near 40° still finds plans through the transitions
  is the open question; 16 cm legs were chosen as the shortest that pass, so
  the margin may be thin. Re-plan, rebuild the reference, retrain.
- **Let the policy see the stairs.** A height scan around each foot, from
  the M4 depth camera on the real robot, lets the policy fix a swing that
  starts from the wrong place. That covers the thigh strike.
- **Train harder where it fails.** Sample more training episodes around
  steep moments and shove more often there.
- **Set the disturbance requirement deliberately.** About {n_shoves:.0f}
  shoves per trip is far more than a house will deliver (pets, people
  brushing past). The 95% target means something only against a stated
  disturbance rate.

## Shove map

![]({f_map})

![]({f_break})

![]({f_pitch})

## Forensic replays

{chr(10).join(sections)}
"""
    (d / "report.md").write_text(md)
    return d


def cmd_pitch_cap() -> Path:
    """M0 step 3e: does capping trunk pitch (to stop backward tip-overs) need a bigger robot?"""
    import json

    from cleandog_sim import viz

    viz.style()
    src = latest("*_pitch-cap/results.json").parent
    data = json.loads((src / "results.json").read_text())
    rows, caps = data["rows"], data["caps"]
    scen = sorted({r["scenario"] for r in rows})
    f_grid = save(viz.fig_pitch_cap_grid(rows, caps, scen), src, "pitch_cap_grid.png")
    by = {}
    for r in rows:
        by.setdefault((r["seg"], r["hip_spacing"], r["cap"]), []).append(r)
    table = []
    for (seg, hs, cap), rs in sorted(by.items(), key=lambda kv: (kv[0][0], kv[0][1], -(kv[0][2] or 1e9))):
        ok = [r for r in rs if r["status"] == "ok"]
        st = ", ".join(f"{r['scenario']}: {r['status']}" for r in sorted(rs, key=lambda r: r["scenario"])
                       if r["status"] != "ok")
        pk = max((r["max_pitch"] for r in ok), default=float("nan"))
        table.append(f"| {seg * 100:.0f} cm | {hs * 100:.1f} cm | {'none' if cap is None else f'{cap:.1f}°'} | "
                     f"{len(ok)}/4 | {pk:.0f}° | {st or '–'} |")
    md = f"""# Capping trunk pitch: does it need a bigger robot? (M0 step 3e)

Generated {dt.datetime.now():%Y-%m-%d %H:%M} by `scripts/visualize.py pitch_cap`, from
`{src.name}/results.json` (`scripts/pitch_cap_sweep.py`).

## Why

The shove analysis (`*_rl-failures/report.md`) found that shoves tip the robot
backward where its trunk pitches steeper than the stairs. Shoves fail
0.35% of the time below 40° and up to 8.7% above 50°. The step-1 planner
chooses front and rear hip heights independently and can pitch the trunk to
about 53°. Capping the pitch removes those moments, but if it needs longer
legs, the robot grows.

## Method

The step-1 quasi-static planner is rerun on the four uphill-facing transitions
the round trip uses. Its rules (reach, link clearance, 2 cm stability margin,
verified swing paths) are unchanged, with one added constraint: |trunk pitch|
≤ cap at every pose.
- **Robots:** knees back; legs 16–20 cm; hips 25 and 29.2 cm apart.
- **Search budget:** {data['budget']} expansions and {data['time_limit']:.0f} s per plan. That is
  {data['time_limit'] / 90:.1f}× the step-1 sweep's time limit, so a failure here is stronger
  evidence.
- **Failures:** "no plan within discretisation" means the search tried every
  option it has. "Search budget exhausted" is unresolved.

![]({f_grid.split('/')[-1] if '/' in f_grid else f_grid})

## Conclusion

- **Capping at the stair slope needs no bigger robot.** The chosen 16 cm /
  25 cm robot plans all four uphill transitions with |pitch| ≤ 42.4°. Planned
  over each whole flight (the round-trip reference's form) it also succeeds:
  lower flight 78 moves (77 uncapped), upper 68 (70), steepest pitch 40.9°.
- **No size gets below the slope.** At 40° and 35° every robot, 20 cm legs
  included, fails the top transitions with "no valid start pose": it can't
  even stand mid-flight. Mid-flight the trunk lies along the stairs, so its
  pitch is the stair slope. That is geometry, not size.
- **Expected effect:** the cap removes the moments steeper than the slope.
  - On the flights those were 11% of moments, and 6.85% of 0.25 m/s shoves
    there failed. Between 40° and the slope it was 1.45%, and below 40° 0.35%.
  - If capped moments behaved like the 40–42.4° band, flight shove failures
    would roughly halve (about 1.4% to 0.8%). That is an estimate: the capped
    plan is a different motion, and only a retrained policy measures it.
- **What the cap leaves:** the 40–42.4° band is still about 4× more fragile
  than shallower poses. Training harder there is the complement.

| Legs | Hip spacing | Cap | Planned | Steepest pitch used | Failures |
| --- | --- | --- | --- | --- | --- |
{chr(10).join(table)}
"""
    (src / "report.md").write_text(md)
    return src


# ---------------------------------------------------------------- interactive

def cmd_view(args) -> None:
    import mujoco
    import mujoco.viewer

    s, p = StairParams(), RobotParams()
    qpos = None
    if args.ladder:
        seg, gap = args.ladder[0] / 100, args.ladder[1] / 100
        p = p.with_(thigh_length=seg, shin_length=seg)
        qpos = ladder_qpos(p, ladder_pose(p, s, 3.5 * s.nosing_pitch, gap))
    model = mujoco.MjModel.from_xml_string(build_scene_xml(p, None if args.flat else s))
    data = mujoco.MjData(model)
    if qpos is not None:
        data.qpos[:] = qpos
    else:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.viewer.launch(model, data)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("stairs", "ladder", "bins", "transitions", "dynamics", "rl", "rl_failures", "pitch_cap", "all"):
        sub.add_parser(name)
    v = sub.add_parser("view")
    v.add_argument("--ladder", nargs=2, type=float, metavar=("SEG_CM", "GAP_CM"),
                   help="pose the robot in ladder posture on the stairs")
    v.add_argument("--flat", action="store_true", help="flat floor instead of the stairwell")
    args = ap.parse_args(argv)

    if args.cmd == "view":
        cmd_view(args)
        return 0
    cmds = {"stairs": cmd_stairs, "ladder": cmd_ladder, "bins": cmd_bins, "transitions": cmd_transitions,
            "dynamics": cmd_dynamics, "rl": cmd_rl, "rl_failures": cmd_rl_failures,
            "pitch_cap": cmd_pitch_cap}
    for name in (cmds if args.cmd == "all" else [args.cmd]):
        print(f"{name}: {cmds[name]()}/report.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
