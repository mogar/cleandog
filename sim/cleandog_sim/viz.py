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
