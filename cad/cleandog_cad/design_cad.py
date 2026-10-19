"""Whole-robot build123d assembly of a design-space design (real vendor motor models), for renders
and for OCP CAD Viewer. Geometry and joint conventions match design_space (which the clearance
numbers come from); this module only draws it."""

from __future__ import annotations

import math

from build123d import Box, Cylinder, Location, Plane, Pos, Rot, Sphere

from .design_space import BODY_BOT, BODY_TOP, HIP_X, L1, L2, Design, belt_box

STAND = (0.0, math.radians(41.4), math.radians(-82.8))  # sim stand_pose at 75 % reach (ab/ad, hip, knee)


def trunk(des: Design):
    zc, h = (BODY_TOP + BODY_BOT) / 2, BODY_TOP - BODY_BOT
    parts = [Box(x1 - x0, y1 - y0, h).moved(Pos((x0 + x1) / 2, (y0 + y1) / 2, zc)) for x0, x1, y0, y1 in des.boxes()]
    out = parts[0]
    for p in parts[1:]:
        out = out.fuse(p)
    return out


def left_leg(des: Design, y_leg: float, fx: int, q=STAND, vendor: bool = True) -> list[tuple]:
    """[(shape, label)] for the left leg at hip x = fx * HIP_X, joint angles q (radians)."""
    a, h, k = (math.degrees(v) for v in q)
    hip_m, knee_m, abad_m = des.hip_motor, des.knee_motor, des.abad_motor
    spans = des.stack_y(y_leg)
    hx, ya, t = fx * HIP_X, des.ya, des.thigh_half_t

    def motor(m, axis_x, axis_z, span, outboard: bool):
        """Motor with its axis along y; output faces +y (outboard) or -y."""
        if vendor:
            if outboard:
                return m.shape().moved(Pos(axis_x, span[1] - ya, axis_z) * Rot(-90, 0, 0))
            return m.shape().moved(Pos(axis_x, span[0] - ya, axis_z) * Rot(90, 0, 0))
        return Cylinder(m.outer_d / 2, span[1] - span[0]).moved(
            Pos(axis_x, (span[0] + span[1]) / 2 - ya, axis_z) * Rot(90, 0, 0))

    roll = Pos(hx, ya, 0) * Rot(a, 0, 0)
    out = []
    face = des.notch_x if des.notch else HIP_X - des.hip_r - 6.0  # ab/ad output face, behind the hip motor
    if vendor:
        abad = abad_m.shape().moved(Pos(fx * face, ya, 0) * Rot(0, 90 * fx, 0))
    else:
        abad = Cylinder(abad_m.outer_d / 2, abad_m.length).moved(Pos(fx * (face - abad_m.length / 2), ya, 0)
                                                                 * Rot(0, 90, 0))
    out.append((abad, "ab/ad motor"))
    out.append((motor(hip_m, 0, 0, spans["hip motor"], True).moved(roll), "hip-pitch motor"))
    kx, kz = des.knee_axis_xz(fx)
    coax = des.knee_mount == "coaxial"
    out.append((motor(knee_m, kx, kz, spans["knee motor"], not coax).moved(roll), "knee motor"))
    if "belt" in spans:
        (cx, cy, cz), (bx, by, bz) = belt_box(des, y_leg, fx)
        out.append((Box(2 * bx, 2 * by, 2 * bz).moved(roll * Pos(cx, cy - ya, cz)), "belt"))
    thigh_loc = roll * Pos(0, y_leg - ya, 0) * Rot(0, h, 0)
    hub_r = 28.0 if "belt" in spans else des.hip_r
    out.append((Cylinder(hub_r, 2 * t).moved(thigh_loc * Rot(90, 0, 0)), "thigh hub"))
    thigh = Box(2 * des.thigh_half_h, 2 * t, L1).moved(Pos(0, 0, -L1 / 2)).fuse(
        Cylinder(des.knee_pulley_r, 2 * t).moved(Pos(0, 0, -L1) * Rot(90, 0, 0)))
    out.append((thigh.moved(thigh_loc), "thigh"))
    shin_loc = thigh_loc * Pos(0, 0, -L1) * Rot(0, k, 0)
    shin = Cylinder(des.shin_r, L2).moved(Pos(0, 0, -L2 / 2)).fuse(Sphere(20.0).moved(Pos(0, 0, -L2)))
    out.append((shin.moved(shin_loc), "shin"))
    return out


def robot(des: Design, y_leg: float, q_legs=None, vendor: bool = True) -> list[tuple]:
    """Whole robot. q_legs: {leg: (ab/ad, hip, knee)} in the left-side convention (design_space),
    default the stand pose. Right legs are mirrored left legs."""
    out = [(trunk(des), "trunk")]
    for leg in ("FL", "FR", "RL", "RR"):
        fx = 1 if leg[0] == "F" else -1
        q = (q_legs or {}).get(leg, STAND)
        for shape, label in left_leg(des, y_leg, fx, q, vendor):
            out.append((shape.mirror(Plane.XZ) if leg[1] == "R" else shape, label))
    return out


__all__ = ["robot", "trunk", "left_leg", "STAND", "Location"]
