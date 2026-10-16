"""Generate M0 figures, 3D renders and Markdown reports into outputs/<date>_<topic>/.

Usage:
  python scripts/visualize.py stairs    # stairwell geometry + 3D views
  python scripts/visualize.py ladder    # ladder-posture sweep: gallery, heatmaps, 3D poses, friction
  python scripts/visualize.py bins      # bin rim height study
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
from cleandog_sim.ladder import ladder_pose, ladder_qpos, run_sweep, slope_frame

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
    fr = slope_frame(s)
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
    for name in ("stairs", "ladder", "bins", "all"):
        sub.add_parser(name)
    v = sub.add_parser("view")
    v.add_argument("--ladder", nargs=2, type=float, metavar=("SEG_CM", "GAP_CM"),
                   help="pose the robot in ladder posture on the stairs")
    v.add_argument("--flat", action="store_true", help="flat floor instead of the stairwell")
    args = ap.parse_args(argv)

    if args.cmd == "view":
        cmd_view(args)
        return 0
    cmds = {"stairs": cmd_stairs, "ladder": cmd_ladder, "bins": cmd_bins}
    for name in (cmds if args.cmd == "all" else [args.cmd]):
        print(f"{name}: {cmds[name]()}/report.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
