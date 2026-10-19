"""Hip layout study: where must the leg plane sit, given the real motors?

One front-left leg, body frame (x forward, y left, z up), millimetres, hip at body mid-height.

From the body outward along y:
- ab/ad motor (DM-J4310P): axis along x at (y = ya, z = 0), inside the body, just behind the
  hip-pitch motor. Its output carries the hip bracket.
- hip-pitch motor (RobStride 02): axis along +y, output facing outboard, inboard face at y0.
- thigh hub: plate of thickness `thigh_hub_t` bolted to the hip-pitch output. Its mid-plane is
  the leg plane.
- knee motor (RobStride 02): output facing inboard and bolted to the far side of the thigh hub,
  so it turns with the thigh and the knee angle equals its output angle.

The hip stack rolls with the ab/ad joint. The trunk is a box. For a body width W, the ab/ad axis
sits as far out as the ab/ad motor allows (ya = W/2 - r_abad - wall), and y0 is the smallest that
keeps `clearance` between the rolled hip stack and the trunk over the whole ab/ad range. That
fixes the ab/ad offset (leg plane - ya) and the overall width.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from build123d import Axis, Box, Cylinder, Location, Pos, Rot

from .vendor import DM_J4310P, ROBSTRIDE_02


@dataclass(frozen=True)
class HipLayout:
    body_width: float = 200.0
    body_height: float = 80.0  # trunk shell (sim RobotParams.body_height)
    skid: float = 12.0  # skid plate under the trunk
    abad_range_deg: float = 40.0  # ± (round trip uses ±34.4°)
    thigh_hub_t: float = 24.0  # thigh hub plate, holds the knee drive's input
    abad_wall: float = 3.0  # ab/ad motor housing to trunk wall
    clearance: float = 3.0  # moving hip stack to trunk, at every ab/ad angle


def hip_stack(y0: float, lay: HipLayout):
    """Envelope of the rolling hip stack (hip motor + thigh hub + knee motor) as solids, with the
    hip-pitch motor's inboard face at y = y0. Cylinders stand in for the vendor models here:
    the clearance search needs fast distance queries."""
    rs = ROBSTRIDE_02
    hip = Cylinder(rs.outer_d / 2, rs.length).moved(Location((0, y0 + rs.length / 2, 0)) * Rot(90, 0, 0))
    hub = Cylinder(rs.outer_d / 2, lay.thigh_hub_t).moved(
        Location((0, y0 + rs.length + lay.thigh_hub_t / 2, 0)) * Rot(90, 0, 0))
    knee = Cylinder(rs.outer_d / 2, rs.length).moved(
        Location((0, y0 + rs.length + lay.thigh_hub_t + rs.length / 2, 0)) * Rot(90, 0, 0))
    return [hip, hub, knee]


def trunk(lay: HipLayout):
    h = lay.body_height + lay.skid
    return Box(400, lay.body_width, h).moved(Pos(0, 0, -lay.skid / 2))


def min_clearance(y0: float, ya: float, lay: HipLayout, step_deg: float = 5.0) -> float:
    body = trunk(lay)
    worst = math.inf
    n = int(round(2 * lay.abad_range_deg / step_deg))
    for k in range(n + 1):
        th = -lay.abad_range_deg + k * step_deg
        roll = Location((0, ya, 0)) * Rot(th, 0, 0) * Location((0, -ya, 0))  # roll about the ab/ad axis
        for part in hip_stack(y0, lay):
            worst = min(worst, part.moved(roll).distance_to(body))
    return worst


@dataclass
class LayoutResult:
    lay: HipLayout
    ya: float  # ab/ad axis, mm from centreline
    y0: float  # hip-pitch motor inboard face
    leg_plane: float  # thigh mid-plane, mm from centreline
    abad_offset: float  # leg plane - ab/ad axis (sim RobotParams.abad_offset)
    outer: float  # knee motor outer face, mm from centreline
    clearance: float

    @property
    def overall_width(self) -> float:
        return 2 * self.outer


def solve(lay: HipLayout, tol: float = 0.25) -> LayoutResult:
    """Smallest y0 (bisection) with the required clearance over the ab/ad range."""
    ya = lay.body_width / 2 - DM_J4310P.outer_d / 2 - lay.abad_wall
    lo, hi = ya, ya + 200.0
    assert min_clearance(hi, ya, lay) >= lay.clearance
    while hi - lo > tol:
        mid = (lo + hi) / 2
        if min_clearance(mid, ya, lay) >= lay.clearance:
            hi = mid
        else:
            lo = mid
    rs = ROBSTRIDE_02
    leg = hi + rs.length + lay.thigh_hub_t / 2
    return LayoutResult(lay, ya, hi, leg, leg - ya, hi + 2 * rs.length + lay.thigh_hub_t,
                        min_clearance(hi, ya, lay))


def analytic_y0(lay: HipLayout) -> float:
    """Hand check of the search: the hip motor's inboard face is a disc of radius R at distance d
    outboard of the ab/ad axis. Rolled by θ, its lowest inboard point moves to
    ya + d·cosθ − R·sinθ, and it must stay clearance outside the wall at W/2. (This ignores the
    trunk's finite height, so it is conservative when the disc swings below the skid.)"""
    ya = lay.body_width / 2 - DM_J4310P.outer_d / 2 - lay.abad_wall
    th = math.radians(lay.abad_range_deg)
    R = ROBSTRIDE_02.outer_d / 2
    d = (lay.body_width / 2 + lay.clearance - ya + R * math.sin(th)) / math.cos(th)
    return ya + d


def assembly(res: LayoutResult, abad_deg: float = 0.0):
    """Real vendor models placed for one leg (front-left), for renders. Returns [(shape, label)]."""
    lay, rs, dm = res.lay, ROBSTRIDE_02, DM_J4310P
    roll = Location((0, res.ya, 0)) * Rot(abad_deg, 0, 0) * Location((0, -res.ya, 0))
    # Hip motor: standard frame output +Z -> world +y, inboard face at y0.
    hip = rs.shape().moved(roll * Location((0, res.y0 + rs.length, 0)) * Rot(-90, 0, 0))
    hub = Cylinder(rs.outer_d / 2, lay.thigh_hub_t).moved(
        roll * Location((0, res.y0 + rs.length + lay.thigh_hub_t / 2, 0)) * Rot(90, 0, 0))
    # Knee motor: output faces inboard (-y), bolted to the hub's outboard face.
    knee = rs.shape().moved(roll * Location((0, res.y0 + rs.length + lay.thigh_hub_t, 0)) * Rot(90, 0, 0))
    # Ab/ad motor: output +x toward the hip, axis at (ya, 0), face just behind the hip motor.
    x_face = -rs.outer_d / 2 - 6.0
    abad = dm.shape().moved(Location((x_face, res.ya, 0)) * Rot(0, 90, 0))
    # The trunk as the study models it: alongside the hip (front hip at x = 0, body length 378 mm
    # centred 125 mm behind it, as in the sim).
    body = Box(378, lay.body_width, lay.body_height + lay.skid).moved(Pos(-125, 0, -lay.skid / 2))
    return [(body, "trunk"), (abad, "ab/ad motor"), (hip, "hip-pitch motor"), (hub, "thigh hub"),
            (knee, "knee motor")]


__all__ = ["Axis", "HipLayout", "LayoutResult", "solve", "analytic_y0", "assembly", "min_clearance"]
