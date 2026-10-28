"""Sagittal-plane checks of the "ladder posture": first-pass sizing before Crocoddyl/RL.

The body sits parallel to the nosing line, with the belly at height `belly_gap`
above it. Each foot is planted in a pocket. This module answers two
questions, both in the 2D x-z plane of one flight's interior:

1. Stance stroke. Over how much body travel along the slope can a foot stay
   planted at a given spot, within reach, within the knee limits, and without
   the thigh or shin passing through the stairs?
2. Quasi-static torques and friction. What do the hip and knee need while the
   foot pushes, and what friction coefficient does the body need to rest on
   its belly?

Because the flight interior is periodic, each leg can be analysed alone. Hip
spacing only matters at the transitions, which are left to Crocoddyl and RL.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .params import RobotParams, StairParams
from .terrain import Terrain, interior_terrain

G = 9.81


@dataclass(frozen=True)
class SlopeFrame:
    """Nosing line of a flight: origin on the line, unit vectors along it (eu) and normal (ev)."""

    origin: np.ndarray
    eu: np.ndarray
    ev: np.ndarray

    def to_world(self, u, v) -> np.ndarray:
        return self.origin + np.multiply.outer(u, self.eu) + np.multiply.outer(v, self.ev)


def slope_frame(s: StairParams) -> SlopeFrame:
    th = s.slope
    eu = np.array([math.cos(th), math.sin(th)])
    ev = np.array([-math.sin(th), math.cos(th)])
    r = s.tread_thickness / 2
    first_nose_centre = np.array([-s.overhang + r, s.rise - r])
    # The line tangent to the tops of all bullnoses (what a flat skid plate rests on).
    return SlopeFrame(first_nose_centre + r * ev, eu, ev)


def pocket_targets(s: StairParams, k: int, foot_radius: float, n: int = 9) -> np.ndarray:
    """Foot-centre positions on tread k (k>=1), from the riser corner back toward the nosing.

    Index 0 is the corner of tread k and riser k+1, where the plan proposes bracing the foot.
    """
    x_riser_next = k * s.run
    z = k * s.rise + foot_radius
    x_corner = x_riser_next - foot_radius
    x_nose = (k - 1) * s.run - s.overhang + foot_radius
    return np.stack([np.linspace(x_corner, x_nose, n), np.full(n, z)], axis=1)


@dataclass
class LegIK:
    """Vectorised planar IK result; arrays share the broadcast shape of hips and feet."""

    ok: np.ndarray  # reachable, within knee limits, and clear of the stairs
    knee: np.ndarray  # (..., 2)
    flex: np.ndarray  # knee flexion magnitude, rad (0 = straight)


def leg_ik(p: RobotParams, terrain: StairParams | Terrain, hips: np.ndarray, feet: np.ndarray, knee_back: bool,
           flex_limits: tuple[float, float] = (0.2, 2.7), n_samples: int = 24, clearance: float = 0.0) -> LegIK:
    """Planar two-link IK in the world x-z plane, for a body facing uphill along the slope.
    `clearance` is kept between the links' surfaces and the stairs (0: touching is allowed).

    knee_back puts the knee on the -x side of the hip-foot line (behind it, for a robot
    facing +x). `terrain` may be StairParams, meaning the interior of an endless flight.
    """
    if isinstance(terrain, StairParams):
        terrain = interior_terrain(terrain)
    l1, l2 = p.thigh_length, p.shin_length
    hips, feet = np.broadcast_arrays(np.asarray(hips, float), np.asarray(feet, float))
    d_vec = feet - hips
    d = np.hypot(d_vec[..., 0], d_vec[..., 1])
    reach_ok = (d < l1 + l2 - 1e-6) & (d > abs(l1 - l2) + 1e-6)
    d_safe = np.where(reach_ok, d, (l1 + l2) / 2)
    flex = np.pi - np.arccos(np.clip((l1**2 + l2**2 - d_safe**2) / (2 * l1 * l2), -1, 1))
    a = np.arccos(np.clip((l1**2 + d_safe**2 - l2**2) / (2 * l1 * d_safe), -1, 1))
    base = np.arctan2(d_vec[..., 1], d_vec[..., 0])
    # Rotating the hip-foot direction counter-clockwise (+a) swings the knee toward +x
    # (uphill, forward) for a leg pointing down, so a backward knee uses -a.
    ang = base + (-a if knee_back else a)
    knee = hips + l1 * np.stack([np.cos(ang), np.sin(ang)], axis=-1)

    ok = reach_ok & (flex >= flex_limits[0]) & (flex <= flex_limits[1])
    t = np.linspace(0.0, 1.0, n_samples)
    # Each link is a capsule: test its centreline against the stairs grown by its radius.
    for a_pt, b_pt, skip, radius in ((hips, knee, 0.0, p.thigh_radius),
                                     (knee, feet, p.foot_radius * 1.5, p.shin_radius)):
        seg = b_pt - a_pt
        length = np.maximum(np.hypot(seg[..., 0], seg[..., 1]), 1e-9)
        tt = t * np.maximum(1.0 - skip / length, 0.0)[..., None]  # stop short of the foot
        pts = a_pt[..., None, :] + tt[..., None] * seg[..., None, :]
        ok &= ~terrain.solid(pts[..., 0], pts[..., 1], clearance=radius + clearance).any(axis=-1)
    return LegIK(ok=ok, knee=knee, flex=flex)


def _cross2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


@dataclass
class StanceResult:
    stroke: float  # longest contiguous hip travel along the slope with the foot planted, m
    u_range: tuple[float, float] | None
    peak_hip_torque: float  # over that stroke, Nm, foot carrying `load_share` of weight
    peak_knee_torque: float


def hip_height(p: RobotParams, belly_gap: float) -> float:
    """Hip axis height above the nosing line when the belly is `belly_gap` above it."""
    return belly_gap + p.hip_above_belly


def stance_stroke(p: RobotParams, s: StairParams, belly_gap: float, foot_target: np.ndarray,
                  knee_back: bool, load_share: float = 0.5, du: float = 0.002) -> StanceResult:
    """Sweep the hip along the slope, holding the foot at `foot_target` (world x-z)."""
    fr = slope_frame(s)
    reach = p.leg_reach
    u_target = float(np.dot(foot_target - fr.origin, fr.eu))
    us = np.arange(u_target - reach, u_target + reach, du)
    hips = fr.to_world(us, hip_height(p, belly_gap))
    ik = leg_ik(p, s, hips, foot_target[None, :], knee_back)
    f_vert = np.array([0.0, load_share * p.total_mass_target * G])  # ground reaction on foot
    tau_h = np.abs(_cross2(foot_target - hips, f_vert))
    tau_k = np.abs(_cross2(foot_target - ik.knee, f_vert))

    # Longest contiguous run of feasible hip positions.
    best, start, best_rng = 0, None, None
    for i, v in enumerate(np.append(ik.ok, False)):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start > best:
                best, best_rng = i - start, (start, i)
            start = None
    if best_rng is None:
        return StanceResult(0.0, None, math.nan, math.nan)
    a, b = best_rng
    return StanceResult(
        stroke=(b - a) * du,
        u_range=(float(us[a]), float(us[b - 1])),
        peak_hip_torque=float(tau_h[a:b].max()),
        peak_knee_torque=float(tau_k[a:b].max()),
    )


@dataclass
class LadderPose:
    """A whole-body ladder posture: trunk on the slope, each foot in a pocket (sagittal)."""

    trunk_xz: np.ndarray  # trunk origin, world x-z
    pitch: float  # nose-up pitch, rad (= stair slope)
    hips: dict[str, np.ndarray]  # "F"/"R" -> hip x-z
    knees: dict[str, np.ndarray]
    feet: dict[str, np.ndarray]
    ok: dict[str, bool]


def ladder_pose(p: RobotParams, s: StairParams, u_body: float, belly_gap: float) -> LadderPose:
    """Place the trunk at slope coordinate `u_body` and pick a comfortable foothold per leg.

    Candidate footholds are 9 spots on each of the nearby treads. The chosen one is
    feasible and puts the hip-foot distance closest to 75% of full leg reach.
    """
    fr = slope_frame(s)
    h = hip_height(p, belly_gap)
    v_trunk = belly_gap + p.body_height / 2 + p.skid_thickness
    cands = np.concatenate([pocket_targets(s, k, p.foot_radius) for k in range(1, 30)])
    out = LadderPose(fr.to_world(u_body, v_trunk), s.slope, {}, {}, {}, {})
    for end, du, sign in (("F", p.hip_spacing / 2, p.knee_sign_front),
                          ("R", -p.hip_spacing / 2, p.knee_sign_rear)):
        hip = fr.to_world(u_body + du, h)
        ik = leg_ik(p, s, hip[None, :], cands, knee_back=(sign == -1))
        dist = np.hypot(*(cands - hip).T)
        score = np.where(ik.ok, np.abs(dist - 0.75 * p.leg_reach), np.inf)
        i = int(np.argmin(score))
        out.hips[end], out.knees[end], out.feet[end] = hip, ik.knee[i], cands[i]
        out.ok[end] = bool(np.isfinite(score[i]))
    return out


def ladder_qpos(p: RobotParams, pose: LadderPose, y0: float = 0.0) -> np.ndarray:
    """MuJoCo qpos (free joint + legs + neck) for a LadderPose, trunk centred at y=y0."""
    from .robot import LEGS

    c, sn = math.cos(pose.pitch / 2), math.sin(pose.pitch / 2)
    qpos = [pose.trunk_xz[0], y0, pose.trunk_xz[1], c, 0.0, -sn, 0.0]  # pitch nose-up about -y
    rot = np.array([[math.cos(pose.pitch), math.sin(pose.pitch)],
                    [-math.sin(pose.pitch), math.cos(pose.pitch)]])  # world x-z -> trunk x-z

    def joint_angle(direction: np.ndarray) -> float:
        # At angle q about +y, a link pointing down has trunk-frame x-z direction (-sin q, -cos q).
        dx, dz = rot @ direction
        return math.atan2(-dx, -dz)

    for leg in LEGS:
        end = leg[0]
        q_thigh = joint_angle(pose.knees[end] - pose.hips[end])
        q_shin = joint_angle(pose.feet[end] - pose.knees[end])
        qpos += [0.0, q_thigh, (q_shin - q_thigh + math.pi) % (2 * math.pi) - math.pi]
    if p.include_neck:
        qpos += [0.0, 0.0]
    return np.array(qpos)


SWEEP_SEGMENTS = (0.10, 0.12, 0.13, 0.14, 0.15, 0.16, 0.18, 0.20)
SWEEP_GAPS = (0.0, 0.03, 0.06, 0.10)


def run_sweep(s: StairParams, base: RobotParams = RobotParams(), segments=SWEEP_SEGMENTS,
              gaps=SWEEP_GAPS, pocket: int = 4) -> list[dict]:
    """Stance stroke for equal thigh/shin lengths x belly gaps x knee direction.

    For each config it records the riser-corner foothold, the mid-tread foothold, and the
    best of 9 footholds along the tread (with that foothold and its hip range, for drawing).
    """
    rows = []
    for seg in segments:
        p = base.with_(thigh_length=seg, shin_length=seg)
        targets = pocket_targets(s, k=pocket, foot_radius=p.foot_radius)
        for gap in gaps:
            for knee_back in (True, False):
                res = [stance_stroke(p, s, gap, t, knee_back) for t in targets]
                i_best = max(range(len(res)), key=lambda i: res[i].stroke)
                best = res[i_best]
                rows.append(dict(
                    seg=seg, gap=gap, knee="back" if knee_back else "fwd",
                    corner_stroke=res[0].stroke, mid_stroke=res[len(res) // 2].stroke,
                    best_stroke=best.stroke, hip_torque=best.peak_hip_torque,
                    knee_torque=best.peak_knee_torque,
                    best_foot=targets[i_best], best_u_range=best.u_range,
                ))
    return rows


def belly_rest_friction(s: StairParams, belly_share: float, skid_mu: float) -> float:
    """Foot friction coefficient needed to hold still with `belly_share` of the weight on the skid plate.

    No surface of a staircase faces uphill: treads face up, risers face downhill. Feet in the
    riser corner therefore cannot supply an uphill force by normal contact, only by
    friction. The skid plate's normal force is tilted downhill by the slope angle, so
    resting on it creates a downhill push that the feet must hold through friction.
    Returns inf if the feet would need to pull.
    """
    th = s.slope
    # Belly normal N (perpendicular to slope) plus skid friction up the slope (mu_b N).
    # The belly carries belly_share of the weight vertically: N (cos th + mu_b sin th) = share.
    n = belly_share / (math.cos(th) + skid_mu * math.sin(th))
    horiz_downhill = n * (math.sin(th) - skid_mu * math.cos(th))
    if horiz_downhill <= 0:
        return 0.0  # skid friction alone holds it
    foot_vertical = 1.0 - belly_share
    return math.inf if foot_vertical <= 1e-9 else horiz_downhill / foot_vertical
