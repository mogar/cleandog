"""Design-space study, report step: figures + report.md from design_space_run.py's data.

Usage: python cad/scripts/design_space_report.py [--rerender]
(--rerender redoes the slow 3D CAD renders, ~2 min each; otherwise existing PNGs are kept.)
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, Polygon, Rectangle  # noqa: E402

from cleandog_cad import design_space as ds  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from design_space_run import OUT  # noqa: E402

# Reference palette (dataviz skill), the same as cleandog_cad.render.
BLUE, ORANGE, AQUA, YELLOW, GREY, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8d8b85", "#0b0b0b", "#52514e"
TRUNK_C = "#c9ccd1"
COLS = {"trunk": TRUNK_C, "ab/ad motor": AQUA, "hip-pitch motor": BLUE, "thigh hub": YELLOW, "knee motor": ORANGE,
        "belt": "#3a3a38", "thigh": "#52514e", "shin": GREY}
TOP = ("NK", "N", "K")
SHORT = {"B": "B: baseline", "N": "N: corner notches", "K": "K: knee motor above hip", "Kf": "Kf: knee motor ahead of hip",
         "M": "M: DM-J4310P hip + knee", "NK": "N+K", "NM": "N+M", "NKM": "N+K+M"}  # the three ideas carried forward, best first (set after the run; see report)
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})


def design(res: dict) -> ds.Design:
    d = next(x for x in ds.DESIGNS if x.key == res["key"])
    return ds.with_(d, nose_half=res["nose_half"]) if d.notch else d


# ------------------------------------------------------------------ figure 1: where the legs go

def fig_leg_sweep(out):
    ref = ds.load_reference()
    q = ref.qpos[::2, 7:19].reshape(-1, 4, 3)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9), sharey=True)
    s = np.linspace(0, 1, 80)[None, :]
    for ax, legs, col, name in ((axes[0], (0, 1), BLUE, "front legs"), (axes[1], (2, 3), ORANGE, "rear legs")):
        pts = []
        for i in legs:
            hx = ds.HIP_X if i < 2 else -ds.HIP_X
            h, k = q[:, i, 1:2], q[:, i, 2:3]
            kx, kz = hx - ds.L1 * np.sin(h), -ds.L1 * np.cos(h)
            pts.append(np.c_[(hx + (kx - hx) * s).ravel(), (kz * s).ravel()])
            pts.append(np.c_[(kx - ds.L2 * np.sin(h + k) * s).ravel(), (kz - ds.L2 * np.cos(h + k) * s).ravel()])
        p = np.vstack(pts)
        H, xe, ze = np.histogram2d(p[:, 0], p[:, 1], bins=(150, 110), range=((-420, 330), (-360, 190)))
        H = np.where(H > 0, np.log10(H + 1), np.nan)
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("c", ["#ffffff", col])
        ax.pcolormesh(xe, ze, H.T, cmap=cmap, vmin=0, shading="auto")
        ax.add_patch(Rectangle((-ds.BODY_LEN / 2, ds.BODY_BOT), ds.BODY_LEN, ds.BODY_TOP - ds.BODY_BOT, fill=False,
                               ec=INK, lw=1.4))
        for hx in (ds.HIP_X, -ds.HIP_X):
            ax.add_patch(Circle((hx, 0), ds.ROBSTRIDE_02.outer_d / 2, fill=False, ec=INK, lw=0.8, ls=(0, (3, 2))))
        ax.set_aspect("equal")
        ax.set_xlim(-420, 330)
        ax.set_ylim(-360, 190)
        ax.set_xlabel("x, body frame (mm)  →  forward")
        ax.set_title(f"{name}: thigh and shin over the whole round trip", fontsize=9.5)
    axes[0].set_ylabel("z, body frame (mm)")
    axes[0].annotate("front thighs swing back\nbeside the mid-trunk\n(x from −82 to +136 mm)", (20, 25), (-330, 120),
                     fontsize=8, color=INK, arrowprops=dict(arrowstyle="->", color=MUTED))
    axes[1].annotate("rear legs never come\nforward of the rear hip", (-150, 30), (-60, 120), fontsize=8, color=INK,
                     arrowprops=dict(arrowstyle="->", color=MUTED))
    fig.text(0.5, -0.01, "Black box: trunk (shell + skid plate). Dashed circles: hip-motor housings. Shading: log of "
             "how often a leg passes there. Hip pitch reaches 129° (front) and 142° (rear).", ha="center",
             fontsize=8, color=MUTED)
    fig.tight_layout()
    fig.savefig(out / "fig_leg_sweep.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ figures 2-3: the designs drawn

def _rot(pts, ang, ya):
    c, s = math.cos(ang), math.sin(ang)
    p = np.asarray(pts, float) - [ya, 0]
    return p @ np.array([[c, s], [-s, c]]) + [ya, 0]


def _front_shapes(des, y_leg):
    """Left-side shapes in the y-z plane at the front hip (rolling parts), as (polygon, colour, z-order)."""
    spans = des.stack_y(y_leg)
    out = []

    def rect(y0, y1, z0, z1, c, zo=3):
        out.append(([(y0, z0), (y1, z0), (y1, z1), (y0, z1)], c, zo))

    R = des.hip_r
    rect(*spans["hip motor"], -R, R, BLUE)
    kx, kz = des.knee_axis_xz(1)
    Rk = des.knee_motor.outer_d / 2
    rect(*spans["knee motor"], kz - Rk, kz + Rk, ORANGE, 2 if des.knee_mount == "forward" else 3)
    if "belt" in spans:
        (cx, cy, cz), (bx, by, bz) = ds.belt_box(des, y_leg, 1)
        rect(cy - by, cy + by, cz - bz, cz + bz, COLS["belt"], 2)
    t = des.thigh_half_t
    hub_r = 28.0 if "belt" in spans else R
    rect(y_leg - t, y_leg + t, -hub_r, hub_r, YELLOW, 4)
    # Leg in the stand pose, seen from the front: thigh and shin hang in the leg plane.
    zk = -ds.L1 * math.cos(0.7226)
    rect(y_leg - t, y_leg + t, zk, 0, COLS["thigh"], 4)
    rect(y_leg - des.shin_r, y_leg + des.shin_r, -240, zk, GREY, 4)
    return out


def draw_front(ax, des, res):
    y_leg, ya, ang = res["y_leg"], des.ya, math.radians(34.4)
    hw = des.body_width / 2
    if des.notch:  # the full-width centre section is behind the hip; the nose is at the hip
        ax.add_patch(Rectangle((-hw, ds.BODY_BOT), 2 * hw, ds.BODY_TOP - ds.BODY_BOT, fc="none", ec=MUTED, lw=0.8,
                               ls=(0, (3, 2)), zorder=1))
        n = des.nose_half
        ax.add_patch(Rectangle((-n, ds.BODY_BOT), 2 * n, ds.BODY_TOP - ds.BODY_BOT, fc=TRUNK_C, ec=MUTED, lw=0.8,
                               zorder=1))
    else:
        ax.add_patch(Rectangle((-hw, ds.BODY_BOT), 2 * hw, ds.BODY_TOP - ds.BODY_BOT, fc=TRUNK_C, ec=MUTED, lw=0.8,
                               zorder=1))
    for sgn in (1, -1):
        ax.add_patch(Circle((sgn * ya, 0), des.abad_motor.outer_d / 2, fc=AQUA, ec="white", lw=0.6, zorder=2))
        for a, alpha in ((ang, 0.16), (-ang, 0.16), (0.0, 1.0)):
            for poly, c, zo in _front_shapes(des, y_leg):
                p = _rot(poly, a, ya)
                p[:, 0] *= sgn
                ax.add_patch(Polygon(p, fc=c, ec="white" if alpha == 1 else "none", lw=0.5, alpha=alpha,
                                     zorder=zo if alpha == 1 else 1.5))
    W = res["overall_width"]
    ax.annotate("", (-W / 2, 150), (W / 2, 150), arrowprops=dict(arrowstyle="<->", color=INK, lw=0.8))
    ax.text(0, 158, f"{W:.0f} mm", ha="center", fontsize=9, color=INK, weight="bold")
    ax.set_xlim(-280, 280)
    ax.set_ylim(-250, 185)
    ax.set_aspect("equal")
    ax.set_xticks([-200, -100, 0, 100, 200])
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_title(SHORT[res["key"]], fontsize=9, loc="left")


def draw_top(ax, des, res):
    y_leg, ya = res["y_leg"], des.ya
    for x0, x1, y0, y1 in des.boxes():
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc=TRUNK_C, ec="none", zorder=1))
    spans = des.stack_y(y_leg)
    face = des.notch_x if des.notch else ds.HIP_X - des.hip_r - 6.0
    for fx in (1, -1):
        kx, _ = des.knee_axis_xz(fx)
        for sy in (1, -1):
            def r(xc, half_x, span, c, zo=3):
                y0, y1 = sorted((sy * span[0], sy * span[1]))
                ax.add_patch(Rectangle((xc - half_x, y0), 2 * half_x, y1 - y0, fc=c, ec="white", lw=0.5, zorder=zo))
            L = des.abad_motor.length
            r(fx * (face - L / 2), L / 2, (ya - 28.5, ya + 28.5), AQUA)
            r(fx * ds.HIP_X, des.hip_r, spans["hip motor"], BLUE)
            r(fx * ds.HIP_X + kx, des.knee_motor.outer_d / 2, spans["knee motor"], ORANGE,
              4 if des.knee_mount == "above" else 3)
            t = des.thigh_half_t
            xk = fx * ds.HIP_X - ds.L1 * math.sin(0.7226)
            ax.add_patch(Rectangle((xk, sy * y_leg - t), fx * ds.HIP_X - xk, 2 * t, fc=COLS["thigh"], ec="white",
                                   lw=0.5, zorder=5))
    ax.set_xlim(-300, 300)
    ax.set_ylim(-270, 270)
    ax.set_aspect("equal")
    ax.set_xticks([-200, 0, 200])
    ax.set_yticks([-200, 0, 200])
    ax.text(-290, 250, f"{res['length']:.0f} mm long", fontsize=7.5, color=MUTED)


def fig_designs(results, out):
    keys = [d.key for d in ds.DESIGNS]
    for name, fn, h in (("fig_designs_front.png", draw_front, 3.0), ("fig_designs_top.png", draw_top, 3.1)):
        fig, axes = plt.subplots(2, 4, figsize=(13, 2 * h))
        for ax, k in zip(axes.ravel(), keys):
            fn(ax, design(results[k]), results[k])
            if fn is draw_top:
                ax.set_title(SHORT[k] + f" — {results[k]['overall_width']:.0f} mm wide", fontsize=9, loc="left")
        fig.tight_layout()
        fig.savefig(out / name, dpi=140)
        plt.close(fig)


# ------------------------------------------------------------------ figure 4: widths

def fig_widths(results, out):
    keys = sorted(results, key=lambda k: results[k]["overall_width"])
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    y = np.arange(len(keys))
    w = [results[k]["overall_width"] for k in keys]
    cols = [ORANGE if k in TOP else BLUE for k in keys]
    ax.barh(y, w, color=cols, height=0.62)
    for i, k in enumerate(keys):
        r = results[k]
        extra = f"   {r['length']:.0f} mm long" if r["length"] > ds.BODY_LEN + 1 else ""
        ax.text(w[i] + 4, i, f"{w[i]:.0f}{extra}", va="center", fontsize=8, color=INK)
    ax.set_yticks(y, [SHORT[k] for k in keys])
    ax.set_ylim(-1.2, len(keys) - 0.5)
    sim = 2 * (ds.SIM_LEG + 18)
    ax.axvline(sim, color=MUTED, lw=1, ls=(0, (4, 3)))
    ax.text(sim + 3, -0.75, f"sim placeholder {sim:.0f} mm", fontsize=7.5, color=MUTED, va="center")
    ax.set_xlim(0, 590)
    ax.set_xlabel("overall width (mm), trunk centre section 200 mm wide in every design")
    ax.set_title("Overall width of each design (orange: N+K, recommended, and the two ideas it combines)", fontsize=9.5,
                 loc="left")
    fig.tight_layout()
    fig.savefig(out / "fig_widths.png", dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------ figures 5-6: clearance and stairs

def _segments(ax, tr, ymax):
    seg, names, dt_ = tr["segment"], [str(s) for s in tr["segment_names"]], float(tr["dt"])
    t = np.arange(len(seg)) * dt_ / 60
    edges = np.flatnonzero(np.diff(seg)) + 1
    starts = np.r_[0, edges]
    ends = np.r_[edges, len(seg)]
    for i, (a, b) in enumerate(zip(starts, ends)):
        if i % 2:
            ax.axvspan(t[a], t[b - 1], color="#f1f0ed", zorder=0)
    return t, starts, ends, names, seg


def fig_clearance(results, tr, out):
    keys = ["B", *TOP]
    fig, axes = plt.subplots(len(keys), 1, figsize=(10, 1.55 * len(keys) + 0.6), sharex=True)
    for ax, k in zip(axes, keys):
        t, starts, ends, names, seg = _segments(ax, tr, 25)
        c = np.minimum(tr[f"{k}/clear"], ds.CULL)
        ax.plot(t, c, color=BLUE, lw=0.7)
        ax.axhline(3, color=ORANGE, lw=1)
        ax.set_ylim(0, 27)
        ax.set_yticks([0, 3, 10, 20])
        ax.set_ylabel("mm", rotation=0, labelpad=12)
        ax.set_title(f"{SHORT[k]}: min {results[k]['min_clear']:.1f} mm ({results[k]['worst_part']})", fontsize=8.5,
                     loc="left", pad=14 if k == keys[0] else 4)
        if results[k]["notch_x"]:
            ax.text(t[-1], 7.2, "6 mm floor: hip motor to the centre section's end wall (fixed bracket gap)",
                    fontsize=7, color=MUTED, ha="right")
    for a, b in zip(starts, ends):
        axes[0].text((t[a] + t[b - 1]) / 2, 28.5, names[seg[a]].replace(" (down, rear-first)", " ↓"), fontsize=6.5,
                     ha="center", color=MUTED, rotation=0)
    axes[-1].set_xlabel("time in the round trip (min)")
    fig.suptitle("Smallest gap between any leg part and the trunk, every frame (capped at 25 mm; orange: 3 mm "
                 "requirement)", fontsize=9.5, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out / "fig_clearance_trace.png", dpi=150)
    plt.close(fig)


def fig_stairs(results, out):
    keys = [d.key for d in ds.DESIGNS]
    fig, ax = plt.subplots(figsize=(8, 3.3))
    x = np.arange(len(keys))
    st = [results[k]["stack_stairs_min"] for k in keys]
    lg = [results[k]["leg_stairs_min"] for k in keys]
    ax.bar(x - 0.19, st, 0.36, color=ORANGE, label="motor stack (hip + knee motors, hub, belt)")
    ax.bar(x + 0.19, lg, 0.36, color=BLUE, label="thigh, knee pulley, shin")
    ax.axhline(0, color=INK, lw=0.8)
    ax.axhline(-12.6, color=MUTED, lw=1, ls=(0, (4, 3)))
    ax.text(len(keys) - 0.5, -11.5, "sim's own legs: −12.6 mm", fontsize=7.5, color=MUTED, ha="right")
    for i, (a, b) in enumerate(zip(st, lg)):
        ax.text(i - 0.19, min(a, 0) - 1.5, f"{a:.0f}", ha="center", va="top", fontsize=7, color=INK)
    ax.set_xticks(x, keys)
    ax.set_ylabel("worst gap to stairs (mm)\nnegative = penetration")
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    ax.set_ylim(-40, 26)
    ax.set_title("Deepest penetration into the stairs over the round trip (MuJoCo replay, feasible footholds only)",
                 fontsize=9.5, loc="left")
    fig.tight_layout()
    fig.savefig(out / "fig_stairs.png", dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------ figure 7: MuJoCo views of the worst stair frame

def fig_mujoco(results, out):
    from cleandog_cad.mj_check import render_frame

    ref = ds.load_reference()
    frame = results["B"]["stairs_worst"]["knee motor"]["frame"]
    files = []
    for k in ("B", TOP[0]):
        des, y = design(results[k]), results[k]["y_leg"]
        qpos = ds.mj_qpos(ref, ds.motion_for(des.ya, y, ref))
        f = out / f"mj_{k}_frame{frame}.png"
        render_frame(des, y, qpos[frame], f, dict(distance=0.75, azimuth=-150, elevation=-12))
        files.append(f)
    return frame, files


# ------------------------------------------------------------------ figure 8: 3D CAD renders

def fig_cad(results, out, rerender):
    from cleandog_cad.design_cad import robot
    from cleandog_cad.render import render

    files = []
    for k in ("B", *TOP):
        f = out / f"cad_{k}.png"
        files.append(f)
        if f.exists() and not rerender:
            continue
        r = results[k]
        render([(s, COLS[label], 0.35 if label == "trunk" else 1.0) for s, label in robot(design(r), r["y_leg"])],
               f, views=("iso", "front", "side", "top"), title=f"{r['label']} — {r['overall_width']:.0f} mm wide (stand pose)",
               tol=0.8, size=3.6)
    return files


# ------------------------------------------------------------------ figure 9: electronics packing

PACK = {  # (x size, y size, height) mm. Typical parts; ESTIMATE until chosen.
    "battery": (110, 72, 44),  # 10S1P 21700, cells lying 5 x 2, with BMS
    "pi": (88, 58, 30),  # Pi 5 + 4-channel CAN HAT
    "power": (50, 62, 25),  # power board, contactor, DC-DC
    "radio": (40, 26, 12),  # e-stop radio
    "imu": (25, 25, 5),
    "neck": (49, 57, 57),  # neck-pitch DM-J4310P, axis along y
}


def fig_packing(results, out):
    k = TOP[0]
    des, r = design(results[k]), results[k]
    fig, ax = plt.subplots(figsize=(8.4, 4.0))
    for x0, x1, y0, y1 in des.boxes():
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc="#ecebe8", ec=MUTED, lw=0.8))
    face, L = des.notch_x, des.abad_motor.length
    for fx in (1, -1):
        for sy in (1, -1):
            ax.add_patch(Rectangle((fx * (face - L / 2) - L / 2, sy * des.ya - 28.5), L, 57, fc=AQUA, ec="white"))
    ax.text(face - L / 2, des.ya, "ab/ad", ha="center", va="center", fontsize=7, color="white")

    def box(name, xc, yc, label, fc, tc="white"):
        sx, sy_, h = PACK[name]
        ax.add_patch(Rectangle((xc - sx / 2, yc - sy_ / 2), sx, sy_, fc=fc, ec="white", lw=0.8))
        ax.text(xc, yc, f"{label}\n{sx}×{sy_}×{h}" if sx > 30 else label, ha="center", va="center", fontsize=6.5,
                color=tc)

    box("battery", 0, 0, "battery 10S1P 21700", YELLOW, INK)
    box("power", 0, 100 - 1.5 - PACK["power"][1] / 2, "power board", GREY)
    box("imu", 0, -100 + 4 + 12.5, "IMU", GREY)
    box("pi", -des.notch_x - 8 - PACK["pi"][0] / 2 - 4, 0, "Pi 5 + CAN HAT", BLUE)
    box("radio", des.notch_x + 12 + PACK["radio"][0] / 2, 0, "e-stop\nradio", GREY)
    box("neck", ds.BODY_LEN / 2 - 6 - PACK["neck"][0] / 2, 0, "neck motor", AQUA)
    ax.set_xlim(-200, 200)
    ax.set_ylim(-108, 108)
    ax.set_aspect("equal")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title(f"{SHORT[k]}: one way to pack the trunk (top view; interior 74 mm tall; part sizes are ESTIMATEs)",
                 fontsize=9, loc="left")
    fig.tight_layout()
    fig.savefig(out / "fig_packing.png", dpi=150)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rerender", action="store_true")
    args = ap.parse_args(argv)
    out = OUT
    results = json.loads((out / "results.json").read_text())
    tr = dict(np.load(out / "traces.npz"))
    fig_leg_sweep(out)
    fig_designs(results, out)
    fig_widths(results, out)
    fig_clearance(results, tr, out)
    fig_stairs(results, out)
    fig_packing(results, out)
    frame, mj_files = fig_mujoco(results, out)
    cad_files = fig_cad(results, out, args.rerender)
    from design_space_text import write_report

    write_report(out, results, frame, mj_files, cad_files)
    print(out / "report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
