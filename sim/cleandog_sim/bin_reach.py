"""Can the robot drop an object over a bin rim? Sagittal grid search on a flat floor.

The robot faces the bin with its front feet at least `toe_gap` from the bin wall. It may:
- stand farther back,
- rear up (front hips higher than rear),
- lean its front hips ahead of the front feet,
- set the neck and head pitch.

A pose counts as a drop when all of these hold:
- the jaw tip is past the rim by `past_rim`, with the held object's underside
  `clearance` above the rim;
- the neck and head clear the rim;
- the centre of mass, including the object, stays over the support span;
- neck-pitch torque stays under its limit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .params import RobotParams
from .robot import mass_breakdown

G = 9.81


@dataclass(frozen=True)
class HeldObject:
    name: str
    mass: float
    hang: float  # how far the object's underside sits below the jaw tip, m


BOOK = HeldObject("hardback book (held flat in the scoop)", 1.0, 0.04)
SHOE = HeldObject("child's shoe", 0.3, 0.08)
SOCK = HeldObject("sock", 0.05, 0.03)


@dataclass(frozen=True)
class DropSpec:
    toe_gap: float = 0.03  # front foot surface to bin wall
    past_rim: float = 0.06  # jaw tip beyond the rim's inner edge
    clearance: float = 0.02  # object underside above the rim
    com_margin: float = 0.03  # COM inside the support span by this much
    neck_torque_limit: float = 7.5  # Nm, about 60% of DM-J4310P peak, for a brief hold


@dataclass
class DropPoses:
    """Every pose in the grid, flattened, with the derived quantities used for plotting."""

    jaw: np.ndarray  # (N, 2) jaw tip, x measured from the bin's outer wall
    neck_base: np.ndarray
    neck_tip: np.ndarray
    neck_torque: np.ndarray
    stable: np.ndarray
    hip_front: np.ndarray  # (N, 2)
    trunk_front: np.ndarray  # (N, 2 corners, 2): bottom-front and top-front of the trunk
    params: dict[str, np.ndarray]


def pose_grid(p: RobotParams, obj: HeldObject, spec: DropSpec = DropSpec(), n: int = 10) -> DropPoses:
    m = mass_breakdown(p)
    r = p.foot_radius
    ext = np.linspace(0.45, 0.97, n) * p.leg_reach  # leg extension (hip to foot centre)
    lean = np.linspace(0.0, 0.10, 6)  # front hip ahead of front foot
    standoff = np.array([0.0, 0.04, 0.08, 0.12])  # front foot farther back than toe_gap
    a_rel = np.linspace(-1.6, 1.2, 21)  # neck pitch joint (+ tips down)
    b_rel = np.linspace(-1.6, 1.6, 25)  # head pitch joint
    grids = np.meshgrid(ext + r, ext + r, lean, standoff, a_rel, b_rel, indexing="ij")
    hf_, hr_, le, so, a, b = (g.ravel() for g in grids)

    sin_beta = np.clip((hf_ - hr_) / p.hip_spacing, -1, 1)
    beta = np.arcsin(sin_beta)  # nose-up pitch
    ex = np.stack([np.cos(beta), np.sin(beta)], -1)  # trunk forward
    ez = np.stack([-np.sin(beta), np.cos(beta)], -1)  # trunk up

    foot_front_x = -(spec.toe_gap + r + so)
    hip_f = np.stack([foot_front_x + le, hf_], -1)
    centre = hip_f - ex * p.hip_spacing / 2 - ez * p.hip_z
    hip_r = centre - ex * p.hip_spacing / 2 + ez * p.hip_z
    foot_rear_x = hip_r[:, 0]  # rear feet straight under the rear hips

    neck_base = centre + ex * (p.body_length / 2 - 0.03) + ez * (p.body_height / 2)
    a_abs = beta - a
    neck_tip = neck_base + p.neck_length * np.stack([np.cos(a_abs), np.sin(a_abs)], -1)
    b_abs = a_abs - b
    jaw = neck_tip + p.head_length * np.stack([np.cos(b_abs), np.sin(b_abs)], -1)

    # Static stability: COM between rear and front feet.
    leg_mass = m["hip_link"] + m["thigh"] + m["shin_with_foot"]
    parts = [
        (m["trunk"], centre),
        (2 * leg_mass, (hip_f + np.stack([foot_front_x, np.full_like(hf_, r)], -1)) / 2),
        (2 * leg_mass, (hip_r + np.stack([foot_rear_x, np.full_like(hf_, r)], -1)) / 2),
        (m["neck"], (neck_base + neck_tip) / 2),
        (m["head"], (neck_tip + jaw) / 2),
        (obj.mass, jaw),
    ]
    total = sum(w for w, _ in parts)
    com_x = sum(w * c[:, 0] for w, c in parts) / total
    stable = (com_x > foot_rear_x + spec.com_margin) & (com_x < foot_front_x - spec.com_margin)

    neck_torque = G * np.abs(
        m["neck"] * ((neck_base + neck_tip) / 2 - neck_base)[:, 0]
        + m["head"] * ((neck_tip + jaw) / 2 - neck_base)[:, 0]
        + obj.mass * (jaw - neck_base)[:, 0]
    )
    # The trunk's front corners: anything past the bin wall must be above the rim.
    half_l, half_h = p.body_length / 2, p.body_height / 2 + p.skid_thickness
    trunk_front = np.stack([centre + ex * half_l + ez * dz for dz in (-half_h, half_h)], 1)
    return DropPoses(jaw, neck_base, neck_tip, neck_torque, stable, hip_f, trunk_front,
                     dict(hip_front=hf_, hip_rear=hr_, lean=le, standoff=so, neck=a, head=b, pitch=beta))


def _segment_min_height_over(x0, z0, x1, z1, xq: float) -> np.ndarray:
    """Height of each segment where it crosses x=xq, or +inf where it doesn't cross."""
    lo, hi = np.minimum(x0, x1), np.maximum(x0, x1)
    crosses = (lo <= xq) & (xq <= hi)
    t = np.where(np.abs(x1 - x0) > 1e-9, (xq - x0) / np.where(np.abs(x1 - x0) > 1e-9, x1 - x0, 1), 0)
    return np.where(crosses, z0 + t * (z1 - z0), np.inf)


def feasible_drops(poses: DropPoses, rim: float, obj: HeldObject, spec: DropSpec = DropSpec(),
                   rim_thickness: float = 0.005) -> np.ndarray:
    jaw = poses.jaw
    x_inner = rim_thickness
    ok = (jaw[:, 0] >= x_inner + spec.past_rim) & (jaw[:, 1] - obj.hang >= rim + spec.clearance)
    for a, b in ((poses.neck_base, poses.neck_tip), (poses.neck_tip, jaw)):
        for xq in (0.0, x_inner):
            z = _segment_min_height_over(a[:, 0], a[:, 1], b[:, 0], b[:, 1], xq)
            ok &= z >= rim + spec.clearance
    # Front legs must stay on this side of the wall; trunk corners past it must clear the rim.
    ok &= poses.hip_front[:, 0] <= -0.01
    for c in range(poses.trunk_front.shape[1]):
        corner = poses.trunk_front[:, c]
        ok &= (corner[:, 0] <= 0.0) | (corner[:, 1] >= rim + spec.clearance)
    return ok & poses.stable & (poses.neck_torque <= spec.neck_torque_limit)


def max_rim_height(p: RobotParams, obj: HeldObject, spec: DropSpec = DropSpec()) -> tuple[float, int]:
    """Highest rim (1 cm resolution) with at least one feasible drop pose, and that pose's index."""
    poses = pose_grid(p, obj, spec)
    best, best_i = 0.0, -1
    for rim in np.arange(0.0, 0.6, 0.01):
        ok = feasible_drops(poses, rim, obj, spec)
        if not ok.any():
            break
        # Prefer the least extreme pose: lowest neck torque among feasible ones.
        best, best_i = float(rim), int(np.flatnonzero(ok)[np.argmin(poses.neck_torque[ok])])
    return best, best_i
