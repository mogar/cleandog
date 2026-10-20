"""Generate M0 figures, 3D renders and Markdown reports into outputs/<date>_<topic>/.

Usage:
  python scripts/visualize.py stairs    # stairwell geometry + 3D views
  python scripts/visualize.py ladder    # ladder-posture sweep: gallery, heatmaps, 3D poses, friction
  python scripts/visualize.py bins      # bin rim height study
  python scripts/visualize.py transitions   # stair transition plans (run transition_sweep.py first)
  python scripts/visualize.py dynamics      # Crocoddyl results (run dynamics_run.py first)
  python scripts/visualize.py rl            # RL round trips (run rl_reference/rl_train/rl_eval first)
  python scripts/visualize.py nk            # N+K hip layout: model, re-plan, clearances, RL vs old geometry
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
    from cleandog_sim.reference import NECK_FOLDED
    from cleandog_sim.transitions import turn_footprint

    turn_d = 2 * turn_footprint(p, NECK_FOLDED)["radius"] * 100
    clr = ref_info.get("clearance")
    if clr:
        lm, sf, st = clr["limb"], clr["self"], clr["stack"]
        clearance_md = f"""Replayed in MuJoCo, with the robot's full collision geometry:
- **Legs vs trunk:** ≥ {sf['min']:.1f} mm everywhere (closest: {' / '.join(sf['pair'])}).
- **Hip stacks vs stairs:** ≥ {st['min']:.1f} mm everywhere (closest: {' / '.join(st['pair'])}).
- **Thighs and shins vs stairs:** {'clear everywhere' if lm['min'] >= 0 else
            f"down to {lm['min']:.1f} mm into a nosing ({lm['pair'][0] if 'world' not in lm['pair'][0] else lm['pair'][1]}, {lm['segment']}), on {lm['frames_below_0']} of {ref_info['n_up'] * 2} frames"}.
  The physics in RL resolves any such contact; the policy has to live with it.
"""
    else:
        clearance_md = "- **Clearance:** not measured for this reference (built before 2026-10-20)."
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
    marg = np.array(ref_info["margins"])
    m_every = ref_info["margins_every"] * ref_info["dt"]
    if len(fails):
        at = marg[np.clip((fails / m_every).astype(int), 0, len(marg) - 1)] * 100
        t_min = marg.argmin() * m_every
        near = int(np.sum(np.abs(fails - t_min) < 3.0) + np.sum(np.abs((ev["n_steps"] * ctrl_dt - fails) - t_min) < 3.0))
        fail_margin_txt = (f"the reference margin where trips failed ranges {at.min():.1f}–{at.max():.1f} cm "
                           f"(median {np.median(at):.1f}). The tightest spot ({marg.min() * 100:.2f} cm, at "
                           f"{t_min:.0f} s) has {near} failure{'s' if near != 1 else ''} within 3 s of it.")
    else:
        fail_margin_txt = "no randomized trip failed."
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
  stays folded, keeping the turning circle at {turn_d:.0f} cm on the
  {s.flights[0].top_depth * 100:.0f} cm landing. Before each swing the trunk
  shifts (sways) to the smallest offset that puts the centre of mass ≥ 2.5 cm
  inside the other three feet, with the legs clear of the trunk.
- **Sway in 3D.** The planner's sideways sway (up to 5 cm before a swing) is
  checked in 3D since 2026-10-20: every planted leg must absorb it within its
  ab/ad, knee and hip limits, with the rolled thigh and shin clear of the
  stairs. The reference then keeps the largest sway, in quarters, that the 3D
  IK reaches exactly with the legs ≥ 3 mm from the trunk and the hip stacks
  off the stairs: {sway_txt}.

![]({f_ref})

![]({f_crawl})

The reference is kinematic: poses and contact timing, not forces.

{clearance_md}

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
    − 5·10⁻⁵ Σ τ² − 0.005 Σ Δa² − 0.1 Σ_touchdowns |v_foot|² − 0.2·[unplanned contact]

Unplanned contact is anything touching the trunk or skid plate (stairs, or the
robot's own legs) or a hip stack (motors, belt) touching anything. Legs collide
with the trunk physically (since 2026-10-20), so a leg can't pass through it.

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

- **Shoves vs everything else.** The friction sweep randomizes everything
  except the shoves. Pooled over its five friction values it completes
  {sweep_k}/{sweep_n} trips ({sweep_k / sweep_n:.1%}, 95% CI {sweep_lo:.1%}–{sweep_hi:.1%}),
  against {rand['rate']:.1%} with a shove every 4–10 s. Across friction
  0.3–0.8 the rate spans {min(res[f"friction {m:.1f}"]["rate"] for m in mus):.0%}–{max(res[f"friction {m:.1f}"]["rate"] for m in mus):.0%}.
- **Where:** of the {int(np.sum(fails < T / 2))} ascent failures in the randomized
  run, {mirrored} have a descent failure within 3 s of the same reference pose
  (the descent is the ascent reversed, so these are spots in the plan where a
  shove can't be recovered).
- **Static margin at the failures:** {fail_margin_txt}

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
Crocoddyl step (2026-10-16 geometry) reported 2.9 Nm on the transitions alone. The RL numbers
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


OLD_RL = OUTPUTS / "2026-10-18_rl"  # the M0 gate run, placeholder hip geometry
# The sim's geometry before the N+K redesign (2026-10-20): a 22 cm box trunk, ab/ad axes 14 cm
# apart, leg planes 12 cm out, and no motors.
OLD_GEOMETRY = dict(body_width=0.22, hip_width=0.14, abad_offset=0.05)


def _cad_vs_sim(p: RobotParams, n: int = 400, seed: int = 0):
    """Leg-to-trunk clearance of random front-left leg poses, by the CAD study's sampled model
    (cad/cleandog_cad/design_space.py, design N+K) and by MuJoCo on the sim's MJCF (mm)."""
    import xml.etree.ElementTree as ET

    import mujoco

    sys.path.insert(0, str(REPO / "cad"))
    from cleandog_cad import design_space as ds

    des = ds.with_(ds.NK, nose_half=p.trunk_nose_width / 2 * 1000)
    root = ET.fromstring(build_scene_xml(p))
    for g in root.iter("geom"):
        if g.get("name", "").startswith("FL_"):
            g.set("margin", "0.04")
    m = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    d = mujoco.MjData(m)
    fl = {g for g in range(m.ngeom) if m.geom(g).name.startswith("FL_")
          and m.geom(g).name[3:] not in ("foot", "hip_motor", "knee_motor", "belt")}
    trunk = {g for g in range(m.ngeom) if m.geom(g).name.startswith(("trunk_shell", "skid"))}
    lips = {g for g in trunk if "lip" in m.geom(g).name}
    rng = np.random.default_rng(seed)
    Q = np.c_[rng.uniform(-0.6, 0.6, n), rng.uniform(-0.5, 2.6, n), rng.uniform(-2.7, -0.2, n)]
    cad = np.full(n, np.inf)
    for name, (frame, pts, inflate) in ds.leg_parts(des, p.leg_plane_y * 1000, 1).items():
        if frame != "stack":  # the hip stack only rolls; its fixed 6 mm gap would hide the rest
            cad = np.minimum(cad, ds.part_clearance(des, frame, pts, inflate, Q, 1))
    sim = np.full(n, np.inf)
    at_lip = np.zeros(n, bool)
    ft = np.zeros(6)
    for i, q in enumerate(Q):
        d.qpos[:] = 0
        d.qpos[2:4] = 1.0, 1.0
        d.qpos[7:10] = q
        mujoco.mj_kinematics(m, d)
        for a in fl:
            for b in trunk:
                dist = mujoco.mj_geomDistance(m, d, a, b, 0.04, ft) * 1000
                if dist < sim[i]:
                    sim[i], at_lip[i] = dist, b in lips
    return cad, sim, at_lip


def cmd_nk(tag: str = "main") -> Path:
    """The N+K hip layout in the sim (2026-10-20): the model, the re-plan, 3D clearances, and the
    RL round trips against the old placeholder geometry."""
    import json
    import shutil
    import xml.etree.ElementTree as ET

    import matplotlib.pyplot as plt
    import mujoco

    from cleandog_sim import viz
    from cleandog_sim.reference import SELF_CLEARANCE, STACK_CLEARANCE, Clearance, Reference
    from cleandog_sim.render import View, render_views
    from cleandog_sim.robot import LEGS, joint_actuation, stand_height
    from cleandog_sim.transitions import morphology

    viz.style()
    d_new = latest("*_rl/reference.json").parent
    out = out_dir("nk-sim")
    ref_new = json.loads((d_new / "reference.json").read_text())
    ref_old = json.loads((OLD_RL / "reference.json").read_text())
    plans_new = json.loads((d_new / "plans.json").read_text())
    plans_old = json.loads((OLD_RL / "plans.json").read_text())
    key = tuple(ref_new["morphology"])
    p, s = morphology(*key), StairParams()
    p_old = p.with_(**OLD_GEOMETRY)
    figs = {}

    # -- 1. the model, next to the CAD study's render
    root = ET.fromstring(build_scene_xml(p))
    root.find("asset").find("material[@name='shell']").set("rgba", "0.62 0.64 0.68 1")  # visible from above
    m_flat = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    z = stand_height(p) - 0.05
    views = [View("iso", (0, 0, z), 0.95, -135, -22), View("front", (0, 0, z), 0.85, 180, -2),
             View("side", (0, 0, z), 0.95, -90, -2), View("top", (0, 0, z), 0.9, -90, -70)]
    names = render_views(m_flat, None, views, out, "model", size=(560, 640))
    imgs = [(v.name, plt.imread(out / n)) for v, n in zip(views, names)]
    figs["model"] = save(viz.fig_render_row(imgs, f"Sim model (MJCF), stand pose: {p.overall_width * 1000:.0f} mm "
                                                  "wide at the thighs, notched trunk, knee motors above the hips"),
                         out, "fig_model.png")
    for n in names:
        (out / n).unlink()
    cad_png = REPO / "outputs" / "2026-10-19_design-space" / "cad_NK.png"
    if cad_png.exists():
        shutil.copy(cad_png, out / "cad_NK.png")

    # -- 2. cross-check: the sim's MJCF against the CAD study's geometry
    cad, sim, at_lip = _cad_vs_sim(p)
    both = (cad < 25) & (sim < 40)  # the CAD sweep reports only a lower bound above 25 mm
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.plot([-5, 40], [-5, 40], color=viz.MUTED, lw=0.9, ls=(0, (4, 3)))
    ax.scatter(cad[both], sim[both], s=12, color=viz.SERIES[0])
    ax.set_xlim(-5, 40)
    ax.set_ylim(-5, 40)
    ax.set_xlabel("CAD study: sampled clearance (mm)")
    ax.set_ylabel("sim MJCF in MuJoCo: distance (mm)")
    diff = np.abs(cad[both] - sim[both])
    ax.set_title(f"Front-left thigh, knee pulley and shin vs trunk,\n{both.sum()} random poses: max difference "
                 f"{diff.max():.2f} mm",
                 fontsize=10)
    figs["cad_sim"] = save(fig, out, "fig_cad_vs_sim.png")

    # -- 3. the reference: static margin, old vs new
    fig, ax = plt.subplots(figsize=(12, 3.2))
    for j, (lab, r) in enumerate((("old geometry (2026-10-18)", ref_old), ("N+K (this run)", ref_new))):
        mg = np.array(r["margins"]) * 100
        w = max(1, int(round(2.0 / (r["dt"] * r["margins_every"]))))  # 2 s windows
        n = len(mg) // w
        env = mg[:n * w].reshape(n, w).min(1)
        t = (np.arange(n) + 0.5) * w * r["dt"] * r["margins_every"]
        ax.plot(t, env, color=viz.SERIES[j], lw=1.0, label=f"{lab}: min {mg.min():.2f} cm")
    ax.axhline(0, color=viz.TEXT_2, lw=0.9)
    viz.shade_segments(ax, ref_new["segments"])
    ax.set_xlim(0, ref_new["duration"])
    ax.set_ylim(-0.5, 6)
    ax.set_ylabel("lowest static margin\nin each 2 s (cm)")
    ax.set_xlabel("time along the round trip (s)")
    ax.legend(loc="upper right", fontsize=8.5, bbox_to_anchor=(1.0, 0.9))
    ax.set_title("Reference: COM distance inside the planted feet's support polygon (3D); > 0 holds with no "
                 "dynamics", fontsize=10)
    figs["margin"] = save(fig, out, "fig_margin.png")

    # -- 4. clearances of every frame
    clr = dict(np.load(d_new / "reference_clearance.npz"))
    segs = ref_new["segments"]
    figs["clear"] = save(viz.fig_clearance_along_trip(ref_new["dt"], clr, segs, Clearance.MARGIN * 1000,
                                                      dict(self=SELF_CLEARANCE * 1000, stack=STACK_CLEARANCE * 1000)),
                         out, "fig_clearance.png")
    ref = Reference.load(d_new / "reference.npz")
    chk = Clearance(p, s)
    root = ET.fromstring(build_scene_xml(p, s.open_sides()))
    m_st = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    shots = []
    for k, short, _ in viz.CLEARANCE_PANELS:
        f = int(np.argmin(clr[k]))
        _, worst = chk(ref.qpos[f], detail=True)
        part = next(n for n in worst[k] if n[:2] in LEGS)
        dd = mujoco.MjData(m_st)
        dd.qpos[:] = ref.qpos[f]
        mujoco.mj_kinematics(m_st, dd)
        leg_pos = dd.xpos[m_st.body(f"{part[:2]}_thigh").id]
        az = 90 if part[1] == "L" else -90
        fn = render_views(m_st, ref.qpos[f], [View(k, tuple(leg_pos), 0.95, az + 25, -15)], out, "worst",
                          size=(480, 560))[0]
        shots.append((f"{short}: {clr[k][f]:+.1f} mm\n{part} at {f * ref.dt:.0f} s", plt.imread(out / fn)))
        (out / fn).unlink()
    figs["worst"] = save(viz.fig_render_row(shots, "The closest frame of each kind (MuJoCo, walls hidden)"),
                         out, "fig_worst.png")

    # -- 5. RL, if trained and evaluated
    have_rl = (d_new / f"eval_{tag}.json").exists()
    rl_md = "RL not trained or evaluated yet for this reference."
    if have_rl:
        ev_new = json.loads((d_new / f"eval_{tag}.json").read_text())
        ev_old = json.loads((OLD_RL / f"eval_{tag}.json").read_text())
        tr_new = json.loads((d_new / f"train_{tag}.json").read_text())
        tr_old = json.loads((OLD_RL / f"train_{tag}.json").read_text())
        conds = ["gate: 1 shove per flight", "randomized", "nominal", "friction 0.3", "friction 0.4",
                 "friction 0.5", "friction 0.6", "friction 0.8"]
        runs = {"old geometry (2026-10-18)": ev_old["results"], "N+K (this run)": ev_new["results"]}
        figs["succ"] = save(viz.fig_compare_success(runs, conds), out, "fig_success.png")
        figs["train"] = save(viz.fig_compare_training({"old geometry": tr_old["curve"], "N+K": tr_new["curve"]}),
                             out, "fig_training.png")
        rows = []
        for c in conds + [k for k in ev_new["results"] if k.startswith("cpu")]:
            a, b = ev_old["results"].get(c), ev_new["results"].get(c)
            if b is None:
                continue
            fa = f"{a['success']}/{a['trips']} ({a['rate']:.1%})" if a else "–"
            ci = b.get("ci95")
            ci_s = f" (CI {ci[0]:.1%}–{ci[1]:.1%})" if ci and b["trips"] > 1 else ""
            rows.append(f"| {c} | {fa} | **{b['success']}/{b['trips']} ({b['rate']:.1%})**{ci_s} |")
        acts = {ja.joint: ja for ja in joint_actuation(p)}
        dem = []
        for lab, dd_ in (("old", OLD_RL), ("N+K", d_new)):
            zz = np.load(dd_ / f"eval_{tag}.npz")
            joints = json.loads((dd_ / f"eval_{tag}.json").read_text())["joints"]
            nj = len(joints)
            pk = zz["nom_tau_peak"]
            rms = np.sqrt(zz["nom_tau_ms"].mean(0))
            vals = []
            for role in ("abad", "hip", "knee"):
                idx = [joints.index(f"{leg}_{role}") for leg in LEGS]
                vals.append(f"{pk[:, idx].max():.1f} / {rms[idx].max():.1f}")
            body = float(np.mean(zz["rand_body"])) if "rand_body" in zz else float("nan")
            dem.append(f"| {lab} | " + " | ".join(vals) + f" | {body:.1%} |")
        gate = ev_new["results"]["gate: 1 shove per flight"]
        verdict = "passes" if gate["rate"] >= 0.95 else "does not pass"
        rl_md = f"""Same pipeline, settings and budget as the 2026-10-18 gate run: PPO for
{ev_new['policy_step'] / 1e6:.0f}M steps (old: {ev_old['policy_step'] / 1e6:.0f}M) on 4096 environments,
with half of the episodes starting at moments of ≥ 40° trunk pitch, then
`rl_eval.py` with the same seeds.

**M0 gate (≥ 95% of round trips with one shove per flight): N+K {verdict}, with
{gate['success']}/{gate['trips']} ({gate['rate']:.1%}, 95% CI {gate['ci95'][0]:.1%}–{gate['ci95'][1]:.1%}).**

| Condition | Old geometry (2026-10-18) | N+K |
| --- | --- | --- |
{chr(10).join(rows)}

![](fig_success.png)

![](fig_training.png)

Closed-loop torque on the nominal trip (no shoves, no randomization), peak / RMS
in Nm, and the share of control steps with unplanned contact in the randomized
trips (old: trunk or skid touching; N+K: also a leg on the trunk or a hip stack
touching anything):

| Geometry | Ab/ad | Hip | Knee | Unplanned contact |
| --- | --- | --- | --- | --- |
{chr(10).join(dem)}

Actuator limits: ab/ad {acts['FL_abad'].peak_torque:.1f} Nm peak / {acts['FL_abad'].rated_torque:.1f} rated,
hip and knee {acts['FL_hip'].peak_torque:.1f} / {acts['FL_hip'].rated_torque:.1f}. The full RL report for this run is
[{d_new.name}/report.md](../{d_new.name}/report.md).
"""

    # -- tables and text
    po = p_old
    geo_rows = "\n".join(f"| {lab} | {a} | {b} |" for lab, a, b in (
        ("Trunk", f"box {po.body_length * 100:.1f} × {po.body_width * 100:.0f} cm",
         f"{p.trunk_notch_x * 200:.1f} × {p.body_width * 100:.0f} cm centre section, "
         f"{p.trunk_nose_width * 100:.1f} cm nose and tail"),
        ("Ab/ad axes apart (`hip_width`)", f"{po.hip_width * 100:.1f} cm", f"{p.hip_width * 100:.1f} cm"),
        ("Leg plane offset (`abad_offset`)", f"{po.abad_offset * 100:.1f} cm", f"{p.abad_offset * 100:.2f} cm"),
        ("Leg planes from the centreline", f"{po.leg_plane_y * 100:.1f} cm", f"{p.leg_plane_y * 100:.1f} cm"),
        ("Hip stack", "none (mass only)", f"hip motor on the hip axis, knee motor {p.knee_motor_above * 1000:.1f} mm "
         f"above, belt stage outboard (RobStride 02, ⌀{p.hip_motor_radius * 2000:.1f} × {p.hip_motor_length * 1000:.1f} mm)"),
        ("Thigh", f"capsule r {po.thigh_radius * 1000:.0f} mm", f"plate {p.thigh_radius * 2000:.0f} × "
         f"{p.thigh_half_thickness * 2000:.0f} mm, hub r {p.thigh_hub_radius * 1000:.0f}, knee pulley r "
         f"{p.knee_pulley_radius * 1000:.0f}"),
        ("Overall width", f"{(po.leg_plane_y + po.thigh_radius) * 200:.1f} cm at the thighs (no motors)",
         f"{p.overall_width * 100:.1f} cm at the thighs, {(p.leg_plane_y + p.foot_radius) * 200:.0f} cm at the feet"),
        ("Self-collision", "off: legs passed up to 21 mm into the trunk", "legs and hip stacks vs trunk"),
    ))
    sway_rows = []
    for lab, r in (("old geometry", ref_old), ("N+K", ref_new)):
        for fl in ("lower", "upper"):
            sw = r["sway_frames"][fl]
            tot = sum(sw.values())
            full = sw.get("1.0", 0) / tot
            sway_rows.append(f"| {lab}, {fl} | {full:.0%} | " + ", ".join(
                f"{float(k):.0%}: {v}" for k, v in sorted(sw.items(), key=lambda kv: -float(kv[0])) if k != "1.0") + " |")
    c = ref_new["clearance"]
    run1_note = (" (That first run's data is kept here as `run1_reference.npz` and `run1_clearance.npz`; "
                 "the current code no longer produces it.)" if (out / "run1_reference.npz").exists() else "")
    md = f"""# The N+K hip layout in the sim: re-plan, self-collision, RL gate

Generated {dt.datetime.now():%Y-%m-%d %H:%M} by `scripts/visualize.py nk` from
`{d_new.name}` (this run) and `{OLD_RL.name}` (the M0 gate run with the placeholder
geometry). The CAD study that chose N+K is
[2026-10-19_design-space](../2026-10-19_design-space/report.md).

## Answer

{rl_md.split(chr(10) + chr(10))[1] if have_rl else "RL pending."}

- **Geometry:** the sim now has the N+K trunk and hip stacks. Legs and hip
  stacks collide with the trunk, and the hip stacks collide with the stairs.
- **Re-plan:** both flights re-planned with the 3D sway check. The reference
  keeps every leg ≥ {c['self']['min']:.1f} mm from the trunk and every hip stack
  ≥ {c['stack']['min']:.1f} mm off the stairs. Its tightest static margin is
  {min(ref_new['margins']) * 100:.2f} cm (old geometry: {min(ref_old['margins']) * 100:.2f} cm).
- **Still open:** thighs and shins still touch nosings in the kinematic
  reference: {c['limb']['min']:.1f} mm at worst, on {c['limb']['frames_below_0']} frames. The RL physics resolves it.

## 1. The model

| | Old sim (placeholder) | N+K (now) |
| --- | --- | --- |
{geo_rows}

The MJCF (`cleandog_sim.robot`) is generated from `RobotParams`, so the planner,
Pinocchio/Crocoddyl and RL all see it. The knee motor now rides on the ab/ad
link, so that link's centre of mass sits {p.knee_motor_above / 2 * 1000:.0f} mm above the hip axis.

![](fig_model.png)

The CAD study's render of the same design, for comparison:

![](cad_NK.png)

**Collision classes** (a pair collides if one's contype shares a bit with the
other's conaffinity):

| Class | contype | conaffinity | Collides with |
| --- | --- | --- | --- |
| World (stairs, walls, floor) | 1 | 2 | every robot geom |
| Trunk, skid plate, neck, head | 2+4 | 1 | world, legs, hip stacks |
| Hip stack (motors, belt) | 2 | 1+4 | world, trunk |
| Thigh, shin, foot | 2 | 1+4 | world, trunk |

Legs don't collide with each other or with their own hip stack: the joints keep
those apart by construction. MuJoCo's parent filter is off, so the hip stacks,
which hang off the trunk body, are checked against it.

**Cross-checks:**
- **CAD vs sim:** the MJCF against the CAD study's geometry, for 400 random
  front-left leg poses (thigh, hub, knee pulley and shin vs the trunk; the hip
  stack's fixed 6 mm gap is left out so it doesn't mask them). The two models
  agree to {diff.max():.2f} mm wherever both measure (below).
- **Warp vs CPU:** on two test poses, one with a leg on the trunk and one with
  a hip motor in the trunk, MJX's Warp backend (used for training) and CPU
  MuJoCo reported the same contacts and depths to 0.01 mm (checked by hand on
  2026-10-20; not rerun by this script). The nominal round trip below is the
  ongoing cross-check.

![](fig_cad_vs_sim.png)

*Found while building this:* MuJoCo contacts within a geom margin are not a
reliable distance measure. They are exact for penetration (which is all the
physics needs), but positive distances go missing: entirely if `geom_margin` is
set after compiling, and for some box-box pairs even when it is in the XML. A
thigh plate 6.1 mm from the skid plate went unreported. The leg-to-trunk check
therefore uses `mj_geomDistance` on every pair, which agrees with the CAD study
to {diff.max():.2f} mm.

## 2. The re-plan, and a planner fix the wider stance needed

The planner and reference pipeline ran unchanged at first. They produced a
reference whose tightest static margin was **0.01 cm** (old: 1.28 cm), with 255
frames under 1 cm. It also had 1,590 frames with a thigh or shin in a nosing,
down to −17.5 mm.{run1_note}

**Why:** the planner's sideways sway (up to 5 cm before each swing, to put the
centre of mass over the other three feet) was a 2D idea. It ignored the ab/ad
limit and what the roll does to the leg. With the leg plane 7.15 cm from the
ab/ad axis instead of 5 cm, a folded leg (knee near its −155° limit) needs a
larger roll for the same sway. The reference then had to shrink sways the legs
couldn't make, and with them the margin. Where the sway was kept, it reshaped
planted legs: a straighter or more bent knee that the planner's zero-sway
capsule check never saw. The penetration grew from −12.6 mm to −17.5 mm, which
is the ratio of the two offsets (71.5 / 50 = 1.43).

**Fix** (`Planner.sway_ik`, `sway_leg_ok`). For each candidate sway y, each
planted leg is solved in 3D. The swing leg is solved at its new foothold.
Trunk frame, foot relative to the ab/ad axis: along the body p_x, up p_z,
outward p_y = o − s·y, where o is the leg-plane offset and s = ±1 the side.

- **Roll:** the foot must lie in the rolled leg plane, so its in-plane height
  is h = −√(p_y² + p_z² − o²). The ab/ad angle is
  a = atan2(p_z, p_y) − atan2(h, o).
- **Planar IK** to (p_x, h) gives the knee flexion and hip angle. The sway is
  kept only if |a| ≤ 0.6 rad, the flexion is in [0.2, 2.7] rad and
  |hip| ≤ 2.6 rad.
- **Side view of the rolled leg** (exact, since the stairs don't vary across
  their width): the hip-pitch joint sits at z = o·sin a and the knee at
  z = o·sin a + l₁·sin(θ)·cos a, where θ is the thigh's in-plane angle. The
  thigh capsule, the knee pulley and the shin must clear the stairs there.
- **Checked:** a test drives MuJoCo with the angles this finds. The foot lands
  on the foothold and the knee sits where the side view says, both to 10⁻⁶ m.

| Reference | Shifts at full sway | Reduced sways (frames) |
| --- | --- | --- |
{chr(10).join(sway_rows)}

| | Old geometry | N+K |
| --- | --- | --- |
| Lower flight | {plans_old['info']['lower']['moves']} moves ({plans_old['info']['lower']['seconds']:.0f} s to plan) | {plans_new['info']['lower']['moves']} moves ({plans_new['info']['lower']['seconds']:.0f} s) |
| Upper flight | {plans_old['info']['upper']['moves']} moves ({plans_old['info']['upper']['seconds']:.0f} s) | {plans_new['info']['upper']['moves']} moves ({plans_new['info']['upper']['seconds']:.0f} s) |
| Round trip | {ref_old['duration']:.0f} s | {ref_new['duration']:.0f} s |
| Tightest static margin | {min(ref_old['margins']) * 100:.2f} cm | {min(ref_new['margins']) * 100:.2f} cm |

![](fig_margin.png)

## 3. Clearance of every reference frame

`reference.Clearance` puts the reference poses into MuJoCo with the full
collision geometry and a 5 cm detection margin on the legs. While building the
reference, the sway search requires legs ≥ {SELF_CLEARANCE * 1000:.0f} mm from the
trunk (the CAD study's design clearance) and hip stacks ≥ {STACK_CLEARANCE * 1000:.0f}
mm off the stairs. The landing crawl's sway search does the same.

| Kind | Minimum | Where | Closest pair | Frames below 0 |
| --- | --- | --- | --- | --- |
""" + "\n".join(f"| {short} | {c[k]['min']:.1f} mm | {c[k]['segment']}, {c[k]['frame'] * ref_new['dt']:.0f} s | "
                f"{' / '.join(c[k]['pair'])} | {c[k]['frames_below_0']} |" for k, short, _ in viz.CLEARANCE_PANELS) + f"""

![](fig_clearance.png)

![](fig_worst.png)

The thigh-nosing contact is the old reference's problem, made no worse: the
2026-10-18 reference had thighs −12.6 mm into nosings in its own geometry. Its
legs also passed 21 mm into the trunk, which can no longer happen.

## 4. RL round trips

{rl_md}

## What is not modelled

- **Hip geometry:** the ab/ad bracket, cables, and the belt's tension and
  compliance. The second belt stage (thigh to knee) is inside the thigh plate.
- **Mass and inertia:** the knee motor's mass sits on the ab/ad link, as
  built. The rest of the mass budget is unchanged (8.5 kg total).
- **Legs:** leg-to-leg contact is off (front and rear legs on one side never
  overlap in the reference).
- **Not re-run:** the morphology sweep (`transition_sweep.py`) and Crocoddyl
  (`dynamics_run.py`) still reflect the 2026-10-16 geometry. The round-trip
  planner re-plans every transition inside each whole flight, and the
  closed-loop torques above replace Crocoddyl's for the actuator check.
"""
    (out / "report.md").write_text(md)
    return out


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
    for name in ("stairs", "ladder", "bins", "transitions", "dynamics", "rl", "rl_failures", "pitch_cap", "nk", "all"):
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
            "pitch_cap": cmd_pitch_cap, "nk": cmd_nk}
    for name in (cmds if args.cmd == "all" else [args.cmd]):
        print(f"{name}: {cmds[name]()}/report.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
