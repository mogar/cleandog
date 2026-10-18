"""Matplotlib figures for M0 reports. Every function returns a Figure; callers save it.

Colours follow the reference data-viz palette: categorical slots in fixed order,
a one-hue blue ramp for magnitude, and neutral inks for text. The stairs and
robot body use neutral fills, so colour stays reserved for data.
"""

from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Circle as CirclePatch  # noqa: E402
from matplotlib.patches import Polygon, Rectangle  # noqa: E402

from .bin_reach import DropPoses, HeldObject  # noqa: E402
from .ladder import LadderPose, belly_rest_friction, hip_height, leg_ik, slope_frame, stance_stroke  # noqa: E402
from .params import FlightParams, RobotParams, StairParams  # noqa: E402
from .stairs import Circle, Rect, flight_profile  # noqa: E402

# Reference palette (light mode).
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")  # blue, orange, aqua: first three slots
SERIES4 = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")  # + yellow, for four lines (adjacent-pair safe)
TEXT, TEXT_2, MUTED = "#0b0b0b", "#52514e", "#8d8b85"
SURFACE, GRID = "#fcfcfb", "#e4e2dd"
STAIR_FILL, STAIR_EDGE = "#ddd6cb", "#9b8f7d"
BODY_FILL, LIMB = "#c9ccd1", "#33363b"
BLUES = LinearSegmentedColormap.from_list(
    "blues", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
CM = 100.0


def style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": MUTED, "axes.labelcolor": TEXT_2, "xtick.color": TEXT_2, "ytick.color": TEXT_2,
        "text.color": TEXT, "axes.titlecolor": TEXT, "axes.titleweight": "bold", "axes.titlesize": 11,
        "font.size": 9.5, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
        "figure.dpi": 110, "savefig.dpi": 160, "savefig.bbox": "tight",
    })


# ---------------------------------------------------------------- drawing primitives

def draw_stairs(ax, s: StairParams, n_risers: int = 8, top_depth: float = 0.6, floor_extent: float = 0.6):
    """Draw a flight's side profile in centimetres."""
    for sh in flight_profile(s, FlightParams(n_risers, top_depth), floor_extent=floor_extent):
        if isinstance(sh, Rect):
            ax.add_patch(Rectangle((sh.x0 * CM, sh.z0 * CM), (sh.x1 - sh.x0) * CM, (sh.z1 - sh.z0) * CM,
                                   facecolor=STAIR_FILL, edgecolor="none", zorder=1))
        elif isinstance(sh, Circle):
            ax.add_patch(CirclePatch((sh.x * CM, sh.z * CM), sh.r * CM, facecolor=STAIR_FILL,
                                     edgecolor="none", zorder=1))
    # Outline the walking surface, with each bullnose as a half-circle arc.
    r = s.tread_thickness / 2
    xs, zs = [-floor_extent, 0.0], [0.0, 0.0]
    for k in range(1, n_risers + 1):
        xr, z = (k - 1) * s.run, k * s.rise
        cx, cz = xr - s.overhang + r, z - r
        arc = np.linspace(-np.pi / 2, -3 * np.pi / 2, 16)  # bottom, round the front, to top
        xs += [xr, xr, *(cx + r * np.cos(arc)), xr + (s.run if k < n_risers else top_depth)]
        zs += [zs[-1], z - s.tread_thickness, *(cz + r * np.sin(arc)), z]
    ax.plot(np.array(xs) * CM, np.array(zs) * CM, color=STAIR_EDGE, lw=1.0, zorder=2)


def draw_nosing_line(ax, s: StairParams, u0: float, u1: float, **kw):
    fr = slope_frame(s)
    pts = fr.to_world(np.array([u0, u1]), np.zeros(2)) * CM
    ax.plot(pts[:, 0], pts[:, 1], **({"color": MUTED, "lw": 1, "ls": (0, (4, 3)), "zorder": 3} | kw))


def draw_trunk(ax, p: RobotParams, centre: np.ndarray, pitch: float, alpha: float = 1.0):
    ex = np.array([math.cos(pitch), math.sin(pitch)])
    ez = np.array([-math.sin(pitch), math.cos(pitch)])
    hl, hh = p.body_length / 2, p.body_height / 2
    corners = [centre + sx * hl * ex + sz * hh * ez for sx, sz in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    ax.add_patch(Polygon(np.array(corners) * CM, closed=True, facecolor=BODY_FILL, edgecolor=LIMB,
                         lw=1.0, alpha=alpha, zorder=5))
    skid = [centre + sx * hl * ex + sz * ez for sx, sz in
            ((-1, -hh), (1, -hh), (1, -hh - p.skid_thickness), (-1, -hh - p.skid_thickness))]
    ax.add_patch(Polygon(np.array(skid) * CM, closed=True, facecolor=LIMB, edgecolor="none",
                         alpha=alpha, zorder=5))


def draw_leg(ax, hip, knee, foot, foot_r: float, color=LIMB, alpha: float = 1.0, lw: float = 3.0):
    pts = np.array([hip, knee, foot]) * CM
    ax.plot(pts[:, 0], pts[:, 1], color=color, lw=lw, alpha=alpha, solid_capstyle="round", zorder=6)
    ax.plot(*pts[:2].T, "o", color=color, ms=4, alpha=alpha, zorder=7)
    ax.add_patch(CirclePatch(tuple(pts[2]), foot_r * CM, facecolor=color, edgecolor="none",
                             alpha=alpha, zorder=7))


def _equal(ax, xlim, ylim):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.set_xlabel("x along stairs (cm)")
    ax.set_ylabel("height (cm)")


# ---------------------------------------------------------------- figures

def fig_stair_geometry(s: StairParams):
    """Side view of the flight with the numbers the ladder-posture argument rests on."""
    fig, ax = plt.subplots(figsize=(9, 5.6))
    draw_stairs(ax, s, n_risers=4, top_depth=0.3, floor_extent=0.25)
    fr = slope_frame(s)
    draw_nosing_line(ax, s, -0.2, 1.05)
    ax.annotate(f"nosing line, {math.degrees(s.slope):.1f}°", fr.to_world(0.95, 0.0) * CM + [0, 4],
                color=TEXT_2, ha="right")

    # Pocket depth: perpendicular from the riser-3 corner up to the nosing line.
    corner = np.array([2 * s.run, 2 * s.rise])
    v = np.dot(corner - fr.origin, fr.ev)
    foot_of_perp = corner - v * fr.ev
    ax.annotate("", foot_of_perp * CM, corner * CM,
                arrowprops=dict(arrowstyle="<->", color=SERIES[0], lw=1.4))
    ax.text(*(corner * CM + [1.5, 2.5]),
            f"pocket depth {-v * CM:.1f} cm (⊥ to nosing line)\n{s.pocket_depth * CM:.1f} cm from rise × run / pitch;\n"
            "the overhang and bullnose add the rest",
            color=TEXT, fontsize=8.5)

    # Rise / run / pitch dimensions on step 1-2.
    x1, z1 = 0.0, s.rise
    ax.annotate("", (x1 * CM - 6, 0), (x1 * CM - 6, z1 * CM), arrowprops=dict(arrowstyle="<->", color=TEXT_2))
    ax.text(x1 * CM - 8, z1 * CM / 2, f"rise {s.rise * CM:.1f}", rotation=90, ha="right", va="center",
            color=TEXT_2)
    ax.annotate("", (0, 3 * s.rise * CM + 3), (s.run * CM, 3 * s.rise * CM + 3),
                arrowprops=dict(arrowstyle="<->", color=TEXT_2))
    ax.text(s.run * CM / 2, 3 * s.rise * CM + 4, f"run {s.run * CM:.1f}", ha="center", color=TEXT_2)
    ax.text((s.run - s.overhang) * CM, 2 * s.rise * CM - 6,
            f"overhang {s.overhang * CM:.1f}\nbullnose ⌀{s.tread_thickness * CM:.1f}",
            ha="center", va="top", color=TEXT_2, fontsize=8.5)
    tips = fr.to_world(np.array([s.nosing_pitch, 2 * s.nosing_pitch]), np.zeros(2)) * CM
    ax.annotate("", tips[0] + [-3, 3], tips[1] + [-3, 3], arrowprops=dict(arrowstyle="<->", color=SERIES[1]))
    ax.text(*(tips.mean(0) + [-6, 5]), f"nosing pitch {s.nosing_pitch * CM:.1f}", color=TEXT,
            rotation=math.degrees(s.slope), ha="center")
    ax.set_title(f"Your stairs in side view: a {math.degrees(s.slope):.0f}° ladder with "
                 f"~{-v * CM:.0f} cm pockets")
    _equal(ax, (-30, 95), (-5, 90))
    return fig


def fig_ladder_pose(p: RobotParams, s: StairParams, pose: LadderPose, belly_gap: float, title: str):
    """Full-body ladder pose with each planted foot's stance stroke drawn as a band on the hip path."""
    fig, ax = plt.subplots(figsize=(9, 6))
    draw_stairs(ax, s, n_risers=7, top_depth=0.3, floor_extent=0.2)
    fr = slope_frame(s)
    u_mid = float(np.dot(pose.trunk_xz - fr.origin, fr.eu))
    draw_nosing_line(ax, s, u_mid - 0.6, u_mid + 0.6)
    h = hip_height(p, belly_gap)
    for i, end in enumerate(("F", "R")):
        sign = p.knee_sign_front if end == "F" else p.knee_sign_rear
        res = stance_stroke(p, s, belly_gap, pose.feet[end], knee_back=(sign == -1))
        if res.u_range:
            band = fr.to_world(np.array(res.u_range), np.full(2, h)) * CM
            ax.plot(band[:, 0], band[:, 1], color=SERIES[i], lw=7, alpha=0.45, solid_capstyle="round", zorder=8)
            ax.text(*(band[1] + [1, 2.5]), f"{'front' if end == 'F' else 'rear'} stroke {res.stroke * CM:.0f} cm",
                    color=TEXT, fontsize=8.5, zorder=9)
    draw_trunk(ax, p, pose.trunk_xz, pose.pitch)
    for i, end in enumerate(("F", "R")):
        draw_leg(ax, pose.hips[end], pose.knees[end], pose.feet[end], p.foot_radius,
                 color=SERIES[i] if pose.ok[end] else "#e34948")
    ax.set_title(title)
    c = pose.trunk_xz * CM
    _equal(ax, (c[0] - 50, c[0] + 45), (c[1] - 42, c[1] + 25))
    ax.text(0.01, 0.01, "Bands: hip travel along the slope over which that foot can stay planted.",
            transform=ax.transAxes, color=TEXT_2, fontsize=8.5)
    return fig


def fig_ladder_gallery(rows: list[dict], p_base: RobotParams, s: StairParams, knee: str):
    """Small multiples: one panel per (segment length, belly gap). Each panel shows the leg at both
    ends of its best stance stroke (light = start, dark = end)."""
    segs = sorted({r["seg"] for r in rows})
    gaps = sorted({r["gap"] for r in rows})
    fig, axes = plt.subplots(len(segs), len(gaps), figsize=(2.6 * len(gaps), 2.3 * len(segs)),
                             sharex=False, sharey=False)
    fr = slope_frame(s)
    for r in rows:
        if r["knee"] != knee:
            continue
        ax = axes[segs.index(r["seg"]), gaps.index(r["gap"])]
        p = p_base.with_(thigh_length=r["seg"], shin_length=r["seg"])
        draw_stairs(ax, s, n_risers=8, top_depth=0.3, floor_extent=0.0)
        foot = r["best_foot"]
        h = hip_height(p, r["gap"])
        if r["best_u_range"]:
            us = np.array(r["best_u_range"])
            hips = fr.to_world(us, np.full(2, h))
            ik = leg_ik(p, s, hips, foot[None, :], knee_back=(knee == "back"))
            for j, alpha in ((0, 0.35), (1, 1.0)):
                draw_leg(ax, hips[j], ik.knee[j], foot, p.foot_radius, color=SERIES[0], alpha=alpha, lw=2)
            band = hips * CM
            ax.plot(band[:, 0], band[:, 1], color=SERIES[1], lw=4, alpha=0.5, solid_capstyle="round", zorder=4)
        draw_nosing_line(ax, s, 0.3, 1.6, lw=0.8)
        ok = r["best_stroke"] >= s.nosing_pitch
        ax.set_title(f"{r['seg'] * CM:.0f} cm · gap {r['gap'] * CM:.0f} · {r['best_stroke'] * CM:.0f} cm"
                     + (" ✓" if ok else ""), fontsize=8.5, fontweight="normal")
        c = foot * CM
        ax.set_xlim(c[0] - 40, c[0] + 22)
        ax.set_ylim(c[1] - 6, c[1] + 50)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=7)
        ax.grid(False)
    for i, seg in enumerate(segs):
        axes[i, 0].set_ylabel(f"segments {seg * CM:.0f} cm", fontsize=9)
    fig.suptitle(f"Ladder posture, knee {knee}: leg at the start (light) and end (dark) of its longest "
                 f"stance stroke.\nOrange band = hip travel. Title: segment · belly gap · stroke "
                 f"(✓ = at least one {s.nosing_pitch * CM:.1f} cm nosing pitch).", fontsize=10, y=1.0)
    fig.tight_layout()
    return fig


def fig_sweep_heatmaps(rows: list[dict], s: StairParams):
    """Best stroke and peak hip torque over segment length x belly gap, per knee direction."""
    segs = sorted({r["seg"] for r in rows})
    gaps = sorted({r["gap"] for r in rows})
    fig, axes = plt.subplots(2, 2, figsize=(10, 8.5))
    metrics = (("best_stroke", "Longest stance stroke (cm)", CM, s.nosing_pitch * CM),
               ("hip_torque", "Peak static hip torque (Nm), foot holds ½ weight", 1.0, None))
    for row_i, knee in enumerate(("back", "fwd")):
        for col_i, (key, label, scale, ref) in enumerate(metrics):
            ax = axes[row_i, col_i]
            grid = np.full((len(segs), len(gaps)), np.nan)
            for r in rows:
                if r["knee"] == knee:
                    grid[segs.index(r["seg"]), gaps.index(r["gap"])] = r[key] * scale
            vmax = np.nanmax([r[key] * scale for r in rows])
            im = ax.imshow(grid, cmap=BLUES, vmin=0, vmax=vmax, aspect="auto", origin="lower")
            for (i, j), val in np.ndenumerate(grid):
                if np.isnan(val):
                    continue
                dark = val > 0.55 * vmax
                mark = " ✓" if (ref is not None and val >= ref) else ""
                ax.text(j, i, f"{val:.0f}{mark}", ha="center", va="center", fontsize=8.5,
                        color="#ffffff" if dark else TEXT)
            ax.set_xticks(range(len(gaps)), [f"{g * CM:.0f}" for g in gaps])
            ax.set_yticks(range(len(segs)), [f"{v * CM:.0f}" for v in segs])
            ax.set_xlabel("belly gap above nosing line (cm)")
            ax.set_ylabel("thigh = shin length (cm)")
            ax.set_title(f"{label}\nknee {knee}", fontsize=10)
            ax.grid(False)
            fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle(f"✓ = stroke ≥ one nosing pitch ({s.nosing_pitch * CM:.1f} cm). "
                 "RobStride 02: 6 Nm rated, 17 Nm peak.", fontsize=10)
    fig.tight_layout()
    return fig


def fig_belly_friction(s: StairParams):
    """Foot friction needed to hold still vs share of weight on the skid plate."""
    fig, ax = plt.subplots(figsize=(8, 5))
    shares = np.linspace(0, 0.95, 200)
    for i, mu in enumerate((0.15, 0.4, 0.7)):
        need = [belly_rest_friction(s, x, mu) for x in shares]
        ax.plot(shares * 100, need, color=SERIES[i], lw=2)
        k = 150
        ax.text(shares[k] * 100 + 1, need[k], f"skid plate μ = {mu}", color=TEXT, fontsize=9, va="bottom")
    ax.axhspan(0.4, 0.6, color=GRID, zorder=0)
    ax.text(2, 0.62, "rubber on finished hardwood, roughly 0.4–0.6 (estimate; M1 measures it)",
            color=TEXT_2, fontsize=8.5)
    ax.set_ylim(0, 2)
    ax.set_xlim(0, 95)
    ax.set_xlabel("share of body weight resting on the skid plate (%)")
    ax.set_ylabel("friction coefficient the feet need")
    ax.set_title(f"Resting on the belly at {math.degrees(s.slope):.0f}° pushes the robot downhill.\n"
                 f"A skid plate needs μ ≥ tan θ = {math.tan(s.slope):.2f} to hold the full weight by itself.")
    return fig


def fig_bin_rim_counts(rims: np.ndarray, counts: dict[str, np.ndarray], recommended: float, static_max: float):
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for i, (label, c) in enumerate(counts.items()):
        ax.plot(rims * CM, c / c[0] * 100, color=SERIES[i], lw=2, label=label)
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.88))
    ax.axvline(recommended * CM, color=TEXT_2, lw=1, ls=(0, (4, 3)))
    ax.text(recommended * CM + 0.5, 95, f"proposed rim {recommended * CM:.0f} cm", color=TEXT, fontsize=9)
    ax.axvline(static_max * CM, color=MUTED, lw=1, ls=(0, (1, 2)))
    ax.text(static_max * CM - 0.5, 95, f"static limit {static_max * CM:.0f} cm", color=TEXT_2, fontsize=8.5,
            ha="right")
    ax.set_xlabel("bin rim height (cm)")
    ax.set_ylabel("drop poses still feasible (% of those at 10 cm)")
    ax.set_ylim(0, 105)
    ax.set_xlim(rims[0] * CM, max(rims[-1], static_max) * CM + 2)
    ax.set_title("Dropping a 1 kg book over the rim: how much posture freedom is left")
    return fig


def fig_bin_pose(p: RobotParams, poses: DropPoses, i: int, rim: float, obj: HeldObject, title: str):
    """Side view of one drop pose against the bin."""
    fig, ax = plt.subplots(figsize=(8, 5.2))
    P = {k: float(v[i]) for k, v in poses.params.items()}
    pitch = P["pitch"]
    ex = np.array([math.cos(pitch), math.sin(pitch)])
    ez = np.array([-math.sin(pitch), math.cos(pitch)])
    hip_f = poses.hip_front[i]
    centre = hip_f - ex * p.hip_spacing / 2 - ez * p.hip_z
    hip_r = centre - ex * p.hip_spacing / 2 + ez * p.hip_z
    # Floor and bin.
    ax.axhline(0, color=STAIR_EDGE, lw=1.2)
    ax.add_patch(Rectangle((0, 0), 45, rim * CM, facecolor="none", edgecolor=STAIR_EDGE, lw=1.5, hatch="////"))
    ax.text(22, rim * CM / 2, f"bin\nrim {rim * CM:.0f} cm", ha="center", va="center", color=TEXT_2)
    draw_trunk(ax, p, centre, pitch)
    foot_f = np.array([hip_f[0] - P["lean"], p.foot_radius])
    foot_r = np.array([hip_r[0], p.foot_radius])
    for hip, foot, col in ((hip_f, foot_f, SERIES[0]), (hip_r, foot_r, SERIES[1])):
        # Planar IK on flat ground; choose knee-back solution.
        d = foot - hip
        dist = np.hypot(*d)
        l1, l2 = p.thigh_length, p.shin_length
        a = math.acos(np.clip((l1**2 + dist**2 - l2**2) / (2 * l1 * dist), -1, 1))
        ang = math.atan2(d[1], d[0]) - a
        knee = hip + l1 * np.array([math.cos(ang), math.sin(ang)])
        draw_leg(ax, hip, knee, foot, p.foot_radius, color=col)
    nb, nt, jaw = poses.neck_base[i] * CM, poses.neck_tip[i] * CM, poses.jaw[i] * CM
    ax.plot([nb[0], nt[0]], [nb[1], nt[1]], color=LIMB, lw=5, solid_capstyle="round", zorder=6)
    ax.plot([nt[0], jaw[0]], [nt[1], jaw[1]], color=LIMB, lw=8, solid_capstyle="butt", zorder=6)
    ax.add_patch(Rectangle((jaw[0] - 3, jaw[1] - obj.hang * CM), 6, obj.hang * CM, facecolor=SERIES[2],
                           edgecolor="none", zorder=7))
    ax.text(jaw[0] + 4, jaw[1] - obj.hang * CM / 2, obj.name, color=TEXT, fontsize=8.5, va="center")
    ax.set_title(title)
    ax.text(0.01, 0.97, f"pitch {math.degrees(pitch):.0f}°, neck torque {poses.neck_torque[i]:.1f} Nm",
            transform=ax.transAxes, color=TEXT_2, fontsize=8.5, va="top")
    _equal(ax, (-65, 50), (-2, 65))
    ax.set_xlabel("x from bin wall (cm)")
    return fig


# ---------------------------------------------------------------- transition plans

def draw_terrain(ax, terrain):
    from .stairs import Circle as C, Rect as R

    for sh in terrain.shapes:
        if isinstance(sh, R):
            ax.add_patch(Rectangle((sh.x0 * CM, sh.z0 * CM), (sh.x1 - sh.x0) * CM, (sh.z1 - sh.z0) * CM,
                                   facecolor=STAIR_FILL, edgecolor=STAIR_EDGE, lw=0.6, zorder=1))
        elif isinstance(sh, C):
            ax.add_patch(CirclePatch((sh.x * CM, sh.z * CM), sh.r * CM, facecolor=STAIR_FILL,
                                     edgecolor=STAIR_EDGE, lw=0.6, zorder=1))


def draw_planner_state(ax, pl, pose: tuple[float, float, float], feet: dict, lifted: str | None = None,
                       swing_from=None, show_com: bool = True, swing_leg: str | None = None):
    """Side view of one planner state (trunk pose = (x, z, pitch)): left legs solid, right legs
    dashed with hollow feet; the lifted leg in red."""
    from .robot import LEGS

    P = pl.pose(*pose)
    ok, info = pl.evaluate(P, feet, lifted=lifted)
    draw_trunk(ax, pl.p, P.centre[0], float(P.pitch[0]))
    base, tip, jaw = (v[0] * CM for v in pl.neck_geometry(P, max(int(info["neck"][0]), 0)))
    ax.plot([base[0], tip[0], jaw[0]], [base[1], tip[1], jaw[1]], color=LIMB, lw=4, solid_capstyle="round",
            zorder=6)
    for leg in LEGS:
        hip = P.hip(pl.p, leg[0] == "F")[0]
        knee = info["knees"][leg][0]
        col = SERIES[0] if leg[0] == "F" else SERIES[1]
        if leg == lifted:
            col = "#e34948"
        pts = np.array([hip, knee, feet[leg]]) * CM
        ax.plot(pts[:, 0], pts[:, 1], color=col, lw=2.2, ls="-" if leg[1] == "L" else (0, (2, 1.5)),
                solid_capstyle="round", zorder=7)
        ax.add_patch(CirclePatch(tuple(pts[2]), pl.p.foot_radius * CM, facecolor=col if leg[1] == "L" else SURFACE,
                                 edgecolor=col, lw=1.2, zorder=8))
    if swing_from is not None and (swing_leg or lifted):
        leg = swing_leg or lifted
        path = pl.swing_path(tuple(pose), leg, np.asarray(swing_from), feet[leg])
        if path is None:
            path = np.array([swing_from, feet[leg]])
        ax.plot(path[:, 0] * CM, path[:, 1] * CM, color="#e34948", lw=1.2, zorder=9)
        ax.annotate("", path[-1] * CM, path[-2] * CM,
                    arrowprops=dict(arrowstyle="->", color="#e34948", lw=1.2), zorder=9)
    if show_com:
        cx = float(info["com_x"][0])
        ax.axvline(cx * CM, color=MUTED, lw=0.8, ls=(0, (1, 2)), zorder=3)
    return ok[0], info


def _state_parts(pl, st):
    x, z, beta, fi, ri, idx = st
    return (x, z, beta), pl.feet_of(tuple(idx))


def fig_plan_filmstrip(pl, path, title: str, n_panels: int = 8):
    """Evenly spaced keyframes from a plan, each a side view."""
    picks = np.unique(np.linspace(0, len(path) - 1, n_panels).round().astype(int))
    cols = 4
    rows = math.ceil(len(picks) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.4 * cols, 3.6 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, i in zip(axes, picks):
        pose, feet = _state_parts(pl, path[i])
        prev_feet = _state_parts(pl, path[i - 1])[1] if i > 0 else feet
        moved = [l for l in feet if not np.allclose(feet[l], prev_feet[l])]
        draw_terrain(ax, pl.t)
        draw_planner_state(ax, pl, pose, feet, lifted=None,
                           swing_from=prev_feet[moved[0]] if moved else None,
                           swing_leg=moved[0] if moved else None)
        if moved:
            ax.add_patch(CirclePatch(tuple(feet[moved[0]] * CM), pl.p.foot_radius * CM * 1.6, fill=False,
                                     edgecolor="#e34948", lw=1.2, zorder=9))
        ax.set_aspect("equal")
        ax.set_xlim(pose[0] * CM - 45, pose[0] * CM + 45)
        ax.set_ylim(pose[1] * CM - 40, pose[1] * CM + 32)
        ax.set_title(f"step {i}/{len(path) - 1}" + (f" · {moved[0]} stepped" if moved else ""), fontsize=9,
                     fontweight="normal")
        ax.tick_params(labelsize=7)
    for ax in axes[len(picks):]:
        ax.axis("off")
    fig.suptitle(title + "\nBlue = front legs, orange = rear; solid = left, dashed/hollow = right. "
                 "Red line = the swing path of the foot that just moved. Dotted line = centre of mass.", fontsize=10)
    fig.tight_layout()
    return fig


def animate_plan(pl, path, out_file, title: str, sub: int = 6, fps: int = 12):
    """GIF of a plan: each step = body shift (feet planted), then the swing foot arcing over."""
    from matplotlib.animation import FuncAnimation, PillowWriter

    frames = []
    for a, b in zip(path[:-1], path[1:]):
        pa, fa = _state_parts(pl, a)
        pb, fb = _state_parts(pl, b)
        for t in np.linspace(0, 1, sub, endpoint=False):
            frames.append((tuple(np.add(pa, t * (np.subtract(pb, pa)))), fa, None))
        moved = [l for l in fb if not np.allclose(fa[l], fb[l])]
        if moved:
            leg = moved[0]
            f0, f1 = fa[leg], fb[leg]
            lift = 0.05 + max(0.0, f1[1] - f0[1])
            for t in np.linspace(0, 1, sub):
                f = dict(fb)
                f[leg] = f0 + t * (f1 - f0) + np.array([0.0, lift * math.sin(math.pi * t)])
                frames.append((pb, f, leg))
    frames += [frames[-1]] * fps  # hold the last frame for a second

    fig, ax = plt.subplots(figsize=(7.5, 5))
    xs = [st[0] for st in path]
    zs = [st[1] for st in path]

    def draw(k):
        ax.clear()
        pose, feet, lifted = frames[k]
        draw_terrain(ax, pl.t)
        draw_planner_state(ax, pl, pose, feet, lifted=lifted, show_com=False)
        ax.set_aspect("equal")
        ax.set_xlim(min(xs) * CM - 45, max(xs) * CM + 45)
        ax.set_ylim(min(zs) * CM - 40, max(zs) * CM + 35)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("x (cm)")
        return []

    anim = FuncAnimation(fig, draw, frames=len(frames), blit=False)
    anim.save(out_file, writer=PillowWriter(fps=fps), dpi=70)
    plt.close(fig)


def fig_success_grid(rows: list[dict], scenarios: list[str], labels: dict[str, str]):
    """Small multiples, one per knee layout × hip spacing: segment length (rows) × scenario (cols)."""
    knees = ["back", "fwd", "out", "in"]
    spacings = sorted({r["hip_spacing"] for r in rows})
    segs = sorted({r["seg"] for r in rows})
    fig, axes = plt.subplots(len(spacings), len(knees), figsize=(4.2 * len(knees), 2.6 + 0.42 * len(segs) * len(spacings)),
                             squeeze=False)
    lookup = {(r["seg"], r["knee"], r["hip_spacing"], r["scenario"]): r for r in rows}
    for i, hs in enumerate(spacings):
        for j, knee in enumerate(knees):
            ax = axes[i, j]
            grid = np.zeros((len(segs), len(scenarios)))
            for a, seg in enumerate(segs):
                for b, sc in enumerate(scenarios):
                    r = lookup.get((seg, knee, hs, sc))
                    if r is None:
                        grid[a, b] = np.nan
                        continue
                    grid[a, b] = 1.0 if r["status"] == "ok" else 0.0
                    mark = "✓" if r["status"] == "ok" else ("?" if "budget" in r["status"] else "✗")
                    ax.text(b, a, mark, ha="center", va="center", fontsize=9,
                            color="#ffffff" if r["status"] == "ok" else TEXT_2)
            ax.imshow(grid, cmap=LinearSegmentedColormap.from_list("ok", ["#f1efea", SERIES[0]]), vmin=0, vmax=1,
                      aspect="auto")
            ax.set_xticks(range(len(scenarios)), [labels[s] for s in scenarios], rotation=60, ha="right", fontsize=7.5)
            ax.set_yticks(range(len(segs)), [f"{s * CM:.0f} cm" for s in segs], fontsize=8)
            ax.set_title(f"knee {knee}, hip spacing {hs * CM:.1f} cm", fontsize=9.5)
            ax.grid(False)
            up_cols = [b for b, sc in enumerate(scenarios) if sc.endswith("up")]
            n_up = int(np.nansum(grid[:, up_cols].all(axis=1)))
            ax.text(1.0, 1.02, f"{n_up} pass all ↑", transform=ax.transAxes, ha="right", fontsize=8, color=TEXT_2)
    fig.suptitle("Quasi-static transition plans. ✓ plan found · ✗ no plan exists in the search space · "
                 "? search ran out of budget (unknown)", fontsize=10)
    fig.tight_layout()
    return fig


def fig_landing_turn(s, p, fp: dict, neck_label: str):
    """Top view of the landing with the robot arriving, turning, and leaving."""
    fig, ax = plt.subplots(figsize=(7.5, 8))
    w, L, D = s.width, s.landing_width, s.flights[0].top_depth
    x0 = 0.0  # landing edge = lower flight's top riser face
    ax.add_patch(Rectangle((x0 * CM, -w / 2 * CM), D * CM, L * CM, facecolor=STAIR_FILL, edgecolor=STAIR_EDGE))
    ax.add_patch(Rectangle(((x0 - 0.9) * CM, -w / 2 * CM), 0.9 * CM, w * CM, facecolor="#ece7de",
                           edgecolor=STAIR_EDGE, hatch="--"))
    ax.add_patch(Rectangle(((x0 - 0.9) * CM, (L - 1.5 * w) * CM), 0.9 * CM, w * CM,
                           facecolor="#ece7de", edgecolor=STAIR_EDGE, hatch="--"))
    ax.text((x0 - 0.45) * CM, 0, "lower flight\n(arrive)", ha="center", va="center", fontsize=9, color=TEXT_2)
    ax.text((x0 - 0.45) * CM, (L - w) * CM, "upper flight\n(leave)", ha="center", va="center", fontsize=9,
            color=TEXT_2)
    yc_up = L - w
    for yc, heading, col, lab in ((0.0, 0.0, SERIES[0], "arrive"), (yc_up, math.pi, SERIES[1], "leave")):
        # Arrive: rear just past the landing edge, facing the far wall. Leave: facing back toward
        # the upper flight, front just behind the landing edge.
        R = np.array([[math.cos(heading), -math.sin(heading)], [math.sin(heading), math.cos(heading)]])
        c = np.array([x0 + 0.02 + (fp["front"] if heading else -fp["rear"]), yc])
        pts = (R @ fp["points"].T).T + c
        hull_order = np.argsort(np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0]))
        ax.add_patch(Polygon(pts[hull_order] * CM, closed=True, facecolor=col, alpha=0.18, edgecolor=col))
        for f in (R @ fp["feet"].T).T + c:
            ax.add_patch(CirclePatch(tuple(f * CM), 2, color=col))
        ax.text(c[0] * CM, c[1] * CM, lab, ha="center", va="center", fontsize=9, color=TEXT)
    turn_c = np.array([x0 + D / 2, (L - w) / 2])
    ax.add_patch(CirclePatch(tuple(turn_c * CM), fp["radius"] * CM, fill=False, edgecolor=TEXT_2, lw=1.2,
                             ls=(0, (4, 3))))
    ax.text(turn_c[0] * CM, (turn_c[1] - fp["radius"]) * CM - 4,
            f"turn-in-place circle ⌀{2 * fp['radius'] * CM:.0f} cm\n({neck_label})", ha="center", va="top",
            fontsize=8.5, color=TEXT)
    ax.set_aspect("equal")
    ax.set_xlim((x0 - 1.0) * CM, (x0 + D + 0.1) * CM)
    ax.set_ylim((-w / 2 - 0.1) * CM, (L - w / 2 + 0.1) * CM)
    ax.set_xlabel("x (cm), along the lower flight")
    ax.set_ylabel("y (cm)")
    ax.set_title(f"Landing {D * CM:.0f} × {L * CM:.0f} cm, top view")
    return fig


# ---------------------------------------------------------------- dynamics (M0 step 2)

LEG_ORDER = ("FL", "FR", "RL", "RR")
ROLE_TITLES = {"abad": "Hip ab/ad", "hip": "Hip pitch", "knee": "Knee"}


def fig_torque_traces(t, tau, joints, acts: dict, title: str):
    """Joint torques over a transition, one panel per joint role, one line per leg, against the
    actuator's rated (dashed) and peak (dotted) torque."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for ax, role in zip(axes, ("abad", "hip", "knee")):
        for i, leg in enumerate(LEG_ORDER):
            j = joints.index(f"{leg}_{role}")
            ax.plot(t, tau[:, j], color=SERIES4[i], lw=1.3, label=leg)
        a = acts[f"FL_{role}"]
        for v, ls, lab in ((a.rated_torque, (0, (4, 3)), "rated"), (a.peak_torque, (0, (1, 2)), "peak")):
            for sgn in (1, -1):
                ax.axhline(sgn * v, color=TEXT_2, lw=0.9, ls=ls)
            ax.text(t[-1], v, f" {lab} {v:.1f}", va="center", fontsize=8, color=TEXT_2)
        ax.set_ylabel(f"{ROLE_TITLES[role]} (Nm)")
        lim = max(a.peak_torque * 1.08, np.abs(tau).max() * 1.08)
        ax.set_ylim(-lim, lim)
    axes[0].legend(ncol=4, loc="upper left", fontsize=8)
    axes[-1].set_xlabel("time (s)")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    return fig


def fig_actuator_demand(demand: dict, acts: dict, title: str = "Actuator demand across all planned transitions "
                        "(worst case), with the 30% margin M0 requires. n× = capability / demand."):
    """Per joint role: worst peak torque, worst RMS torque and worst speed across all transitions,
    each with a 30% margin, against the actuator's capability."""
    roles = ("abad", "hip", "knee")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    specs = [("peak", "peak torque (Nm)", lambda a: a.peak_torque),
             ("rms", "RMS torque (Nm) — heating", lambda a: a.rated_torque),
             ("speed", "joint speed (rad/s)", lambda a: a.max_speed)]
    for ax, (key, label, cap_of) in zip(axes, specs):
        y = np.arange(len(roles))
        need = np.array([demand[r][key] for r in roles])
        cap = np.array([cap_of(acts[f"FL_{r}"]) for r in roles])
        ax.barh(y + 0.18, need * 1.3, height=0.32, color=SERIES[0], label="demand × 1.3")
        ax.barh(y - 0.18, cap, height=0.32, color=GRID, edgecolor=MUTED, label="actuator")
        for yi, n, c in zip(y, need, cap):
            ok = "✓" if n * 1.3 <= c else "✗"
            ax.text(max(n * 1.3, c) * 1.03, yi, f"{ok} {c / n:.1f}×" if n > 0 else "", va="center", fontsize=8.5)
        ax.set_yticks(y, [ROLE_TITLES[r] for r in roles])
        ax.set_xlabel(label)
        ax.set_xlim(0, max((need * 1.3).max(), cap.max()) * 1.35)
        ax.grid(axis="y", visible=False)
    axes[0].legend(loc="lower right", fontsize=8)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    return fig


def fig_friction_use(t, fric, mu: float, title: str):
    fig, ax = plt.subplots(figsize=(10, 2.8))
    ax.plot(t, fric, color=SERIES[0], lw=1.2)
    ax.axhline(mu, color=TEXT_2, lw=0.9, ls=(0, (4, 3)))
    ax.text(t[-1], mu, f" μ limit {mu}", va="center", fontsize=8, color=TEXT_2)
    ax.set_ylim(0, max(mu * 1.25, fric.max() * 1.1))
    ax.set_xlabel("time (s)")
    ax.set_ylabel("max |f_t| / f_n")
    ax.set_title(title, fontsize=10)
    return fig


# ---------------------------------------------------------------- M0 step 3: RL round trips

def _short(name: str) -> str:
    base = name.replace(" (down, rear-first)", "")
    arrow = "↓" if "down" in name else ("↑" if "flight" in name else "")
    return {"lower flight": "lower", "upper flight": "upper"}.get(base, base) + (f" {arrow}" if arrow else "")


def shade_segments(ax, segments: list[dict], label: bool = True, ymax_frac: float = 0.98):
    """Alternate light bands for the round trip's segments (times in seconds), labelled at the top."""
    for k, sg in enumerate(segments):
        x0, x1 = sg["start"], sg["start"] + sg["duration"]
        if k % 2 == 0:
            ax.axvspan(x0, x1, color=GRID, alpha=0.45, lw=0, zorder=0)
        if label and sg["duration"] > 20:
            ax.text((x0 + x1) / 2, ymax_frac, _short(sg["name"]), transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=8, color=TEXT_2)


def fig_reference_overview(dt: float, margins_every: int, margins: np.ndarray, segments: list[dict]):
    t = np.arange(len(margins)) * dt * margins_every
    fig, ax = plt.subplots(figsize=(12, 3.0))
    ax.plot(t, margins * CM, color=SERIES[0], lw=0.8)
    ax.axhline(0, color=TEXT_2, lw=0.9)
    ax.set_ylim(min(-0.5, margins.min() * CM * 1.1), max(margins.max() * CM * 1.15, 5))
    shade_segments(ax, segments)
    ax.set_xlim(0, t[-1])
    ax.set_xlabel("time along the round trip (s)")
    ax.set_ylabel("static margin (cm)")
    ax.set_title("Reference: COM distance inside the planted feet's support polygon (3D, top view); > 0 means the "
                 "pose holds with no dynamics", fontsize=10)
    return fig


def fig_landing_crawl(p: RobotParams, s: StairParams, crawl: list[dict], legs=("FL", "FR", "RL", "RR")):
    """Top view of the landing: the trunk at every swing, the footholds and the COM path."""
    lower = s.flights[0]
    x_top = (lower.n_risers - 1) * s.run
    x_end = x_top + lower.top_depth
    ys = 1 if s.turn_left else -1
    y_lo, y_hi = sorted((-ys * s.width / 2, ys * (s.landing_width - s.width / 2)))
    fig, ax = plt.subplots(figsize=(7.5, 9.5))
    ax.add_patch(Rectangle((x_top * CM, y_lo * CM), (x_end - x_top) * CM, (y_hi - y_lo) * CM,
                           facecolor="none", edgecolor=TEXT_2, lw=1.4))
    for y0, y1, lab in ((-s.width / 2, s.width / 2, "lower flight (down to the floor)"),
                        (ys * (s.landing_width - s.width) - s.width / 2, ys * (s.landing_width - s.width) + s.width / 2,
                         "upper flight (up)")):
        ax.add_patch(Rectangle(((x_top - 0.45) * CM, y0 * CM), 0.45 * CM, (y1 - y0) * CM, facecolor=STAIR_FILL,
                               edgecolor="none"))
        for k in range(3):
            ax.plot([(x_top - k * s.run) * CM] * 2, [y0 * CM, y1 * CM], color=STAIR_EDGE, lw=0.8)
        ax.text((x_top - 0.22) * CM, (y0 + y1) / 2 * CM, lab, rotation=90, ha="center", va="center", fontsize=8,
                color=TEXT_2)
    hl, hw = p.body_length / 2, p.body_width / 2
    corners = np.array([[hl, hw], [hl, -hw], [-hl, -hw], [-hl, hw]])
    for i, c in enumerate(crawl):
        x, y, yaw = c["base"][0], c["base"][1], c["base"][3]
        R = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
        pts = (corners @ R.T + [x, y]) * CM
        ax.add_patch(Polygon(pts, closed=True, facecolor=BODY_FILL, edgecolor=LIMB, lw=0.4,
                             alpha=0.12 if 0 < i < len(crawl) - 1 else 0.5))
    com = np.array([c["com"] for c in crawl]) * CM
    ax.plot(com[:, 0], com[:, 1], color=TEXT, lw=1.0, label="centre of mass (at each lift)")
    for k, leg in enumerate(legs):
        f = np.array([c["feet"][leg][:2] for c in crawl if c["leg"] == leg]) * CM
        ax.scatter(f[:, 0], f[:, 1], s=14, color=SERIES4[k], label=f"{leg} footholds", zorder=3)
    worst = min(range(len(crawl)), key=lambda i: crawl[i]["margin"])
    c = crawl[worst]
    tri = np.array([c["feet"][l][:2] for l in legs if l != c["leg"]]) * CM
    ax.add_patch(Polygon(tri, closed=True, facecolor="none", edgecolor=TEXT, lw=1.0, ls=(0, (3, 2))))
    ax.annotate(f"tightest support triangle\n(COM {c['margin'] * CM:.1f} cm inside)", tri.mean(0),
                xytext=(18, -30), textcoords="offset points", fontsize=8, arrowprops=dict(arrowstyle="-", lw=0.6))
    ax.set_aspect("equal")
    ax.set_xlim((x_top - 0.5) * CM, (x_end + 0.08) * CM)
    ax.set_ylim((y_lo - 0.08) * CM, (y_hi + 0.08) * CM)
    ax.set_xlabel("x (cm, uphill on the lower flight)")
    ax.set_ylabel("y (cm)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8)
    ax.set_title(f"Landing crawl, top view: {len(crawl)} swings, trunk drawn at each", fontsize=10)
    return fig


def fig_training_curve(curve: list[dict], episode_length: int):
    steps = np.array([r["step"] for r in curve]) / 1e6
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.2))
    rew = np.array([r.get("eval/episode_reward", np.nan) for r in curve])
    length = np.array([r.get("eval/avg_episode_length", np.nan) for r in curve])
    axes[0].plot(steps, rew, color=SERIES[0], lw=1.5, marker="o", ms=3)
    axes[0].set_ylabel("episode reward")
    axes[0].set_title("Evaluation reward (128 randomized episodes)", fontsize=10)
    axes[1].plot(steps, length * CTRL_DT_REPORT, color=SERIES[0], lw=1.5, marker="o", ms=3)
    axes[1].axhline(episode_length * CTRL_DT_REPORT, color=TEXT_2, lw=0.9, ls=(0, (4, 3)))
    axes[1].text(steps[0], episode_length * CTRL_DT_REPORT, " episode cap", va="bottom", fontsize=8, color=TEXT_2)
    axes[1].set_ylabel("survival (s)")
    axes[1].set_title("Time before the trunk strays from the reference", fontsize=10)
    for ax in axes:
        ax.set_xlabel("environment steps (millions)")
    fig.tight_layout()
    return fig


CTRL_DT_REPORT = 0.02


def fig_success(results: dict, sweep_mus: list[float]):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), gridspec_kw=dict(width_ratios=[1, 1.3]))
    names = [n for n in ("nominal", "randomized") if n in results]
    for i, n in enumerate(names):
        r = results[n]
        lo, hi = r.get("ci95", [r["rate"], r["rate"]])
        axes[0].barh(i, r["rate"] * 100, height=0.5, color=SERIES[0])
        axes[0].plot([lo * 100, hi * 100], [i, i], color=TEXT, lw=1.2)
        axes[0].text(2, i, f"{r['success']}/{r['trips']}", va="center", fontsize=8.5, color="white",
                     fontweight="bold")
    axes[0].axvline(95, color=TEXT_2, lw=0.9, ls=(0, (4, 3)))
    axes[0].text(96, -0.45, "95% target", fontsize=8, color=TEXT_2)
    axes[0].set_yticks(range(len(names)), names)
    axes[0].set_xlim(0, 115)
    axes[0].set_xlabel("round trips completed (%), bar = 95% CI")
    axes[0].grid(axis="y", visible=False)
    axes[0].set_title("Whole round trips", fontsize=10)
    rates = np.array([results[f"friction {m:.1f}"]["rate"] for m in sweep_mus]) * 100
    ci = np.array([results[f"friction {m:.1f}"]["ci95"] for m in sweep_mus]) * 100
    axes[1].fill_between(sweep_mus, ci[:, 0], ci[:, 1], color=SERIES[0], alpha=0.18, lw=0)
    axes[1].plot(sweep_mus, rates, color=SERIES[0], lw=1.8, marker="o", ms=5)
    axes[1].axhline(95, color=TEXT_2, lw=0.9, ls=(0, (4, 3)))
    axes[1].axvspan(0.4, 1.0, color=GRID, alpha=0.4, lw=0)
    axes[1].text(0.82, 4, "trained on\nμ 0.4–1.0", fontsize=8, color=TEXT_2, ha="center")
    axes[1].set_xlim(min(sweep_mus) - 0.03, max(sweep_mus) + 0.03)
    axes[1].set_ylim(0, 105)
    axes[1].set_xlabel("foot friction μ (rubber on hardwood), all else randomized, no pushes")
    axes[1].set_ylabel("round trips completed (%)")
    axes[1].set_title("Sensitivity to foot friction", fontsize=10)
    fig.tight_layout()
    return fig


def fig_tracking_along_trip(dt: float, pos_err: np.ndarray, segments: list[dict], failed_at: np.ndarray,
                            term: float):
    """pos_err: (T, B) trunk distance from the reference, sampled every dt seconds."""
    t = np.arange(pos_err.shape[0]) * dt
    pos_err = pos_err.astype(float).copy()
    for b, f in enumerate(failed_at):  # a failed trip leaves the statistics once it fails
        if f >= 0:
            pos_err[int(f * CTRL_DT_REPORT / dt) + 1:, b] = np.nan
    fig, ax = plt.subplots(figsize=(12, 3.6))
    with np.errstate(all="ignore"):
        q = np.nanpercentile(pos_err, [50, 95, 100], axis=1) * CM
    ax.fill_between(t, 0, q[1], color=SERIES[0], alpha=0.25, lw=0, label="95th percentile")
    ax.plot(t, q[0], color=SERIES[0], lw=1.0, label="median")
    ax.plot(t, q[2], color=SERIES[0], lw=0.5, alpha=0.6, label="worst trip")
    ax.axhline(term * CM, color=TEXT_2, lw=0.9, ls=(0, (4, 3)))
    ax.text(t[-1], term * CM, f" failure: {term * CM:.0f} cm", va="center", fontsize=8, color=TEXT_2)
    f = failed_at[failed_at >= 0] * CTRL_DT_REPORT
    if len(f):
        ax.plot(f, np.full(len(f), term * CM * 1.08), "|", color=TEXT, ms=10, label="failures")
    shade_segments(ax, segments)
    ax.set_xlim(0, t[-1])
    ax.set_ylim(0, term * CM * 1.25)
    ax.set_xlabel("time along the round trip (s)")
    ax.set_ylabel("trunk from reference (cm)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), fontsize=8, ncol=4)
    ax.set_title(f"Tracking error over {pos_err.shape[1]} randomized round trips (a failed trip drops out "
                 "when it fails)", fontsize=10)
    return fig


def fig_engine_agreement(dt: float, traces: dict[str, np.ndarray], segments: list[dict]):
    fig, ax = plt.subplots(figsize=(12, 3.0))
    for k, (name, y) in enumerate(traces.items()):
        t = np.arange(len(y)) * dt
        ax.plot(t, y * CM, color=SERIES[k], lw=0.9, label=name)
    shade_segments(ax, segments)
    ax.set_xlim(0, max(len(y) for y in traces.values()) * dt)
    ax.set_ylim(0, None)
    ax.set_xlabel("time along the round trip (s)")
    ax.set_ylabel("trunk from reference (cm)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), fontsize=8, ncol=3)
    ax.set_title("Same policy, nominal robot: MJX/Warp (training engine) vs CPU MuJoCo at two timesteps", fontsize=10)
    return fig
