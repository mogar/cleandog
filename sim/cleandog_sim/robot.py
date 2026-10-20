"""Parametric 12-DoF quadruped (+ optional 2-DoF neck) emitted as MJCF.

Frames: trunk x forward, y left, z up. With every joint at zero, the legs hang
straight down and the neck points straight forward. The legs, in order, are
FL, FR, RL, RR, and each has the joints abad (about x), hip (about y) and
knee (about y).

Geometry follows design N+K of the CAD design-space study: a notched trunk
(full-width centre section, narrow nose and tail), and on each hip a stack that
rolls with ab/ad: the hip-pitch motor, the knee motor above it and the first
belt stage. The thigh is a plate (box), the shin a capsule.

Collision bits (a pair collides if contype of one & conaffinity of the other):

| class | contype | conaffinity | collides with |
| --- | --- | --- | --- |
| world (scene.py) | 1 | 2 | every robot geom |
| trunk, skid, neck, head | 2+4 | 1 | world, legs, hip stacks |
| hip stack (motors, belt) | 2 | 1+4 | world, trunk |
| thigh, shin, foot | 2 | 1+4 | world, trunk |

Legs don't collide with each other or with their own hip stack (the joints keep
those apart by construction). Parent filtering is off (scene.py), so the hip
stacks, which hang off the trunk body, are checked against it. Robot contact
geoms set priority=1 so that their friction overrides the world's.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np

from .params import ActuatorSpec, RobotParams

LEGS = ("FL", "FR", "RL", "RR")
TRUNK_COLLISION = dict(contype="6", conaffinity="1")
LEG_COLLISION = dict(contype="2", conaffinity="5")


def _fmt(*v: float) -> str:
    return " ".join(f"{x:.6g}" for x in v)


def _box_inertia(m: float, sx: float, sy: float, sz: float) -> tuple[float, float, float]:
    return (
        m * (sy**2 + sz**2) / 12,
        m * (sx**2 + sz**2) / 12,
        m * (sx**2 + sy**2) / 12,
    )


def _rod_inertia(m: float, length: float, axis: str, r: float = 0.015) -> tuple[float, float, float]:
    """Diagonal inertia of a thin rod along `axis`, with a small radial floor for stability."""
    radial = m * length**2 / 12 + m * r**2 / 4
    axial = m * r**2 / 2
    return {"x": (axial, radial, radial), "z": (radial, radial, axial)}[axis]


@dataclass(frozen=True)
class JointActuation:
    """How one joint is driven: actuator spec scaled by any transmission after it."""

    joint: str
    spec: ActuatorSpec
    ratio: float = 1.0  # extra reduction after the actuator (e.g. knee belt)

    @property
    def peak_torque(self) -> float:
        return self.spec.peak_torque * self.ratio

    @property
    def rated_torque(self) -> float:
        return self.spec.rated_torque * self.ratio

    @property
    def max_speed(self) -> float:
        return self.spec.no_load_speed / self.ratio

    def torque_limit(self, speed: float) -> float:
        return max(0.0, self.peak_torque * (1.0 - abs(speed) / self.max_speed))


def joint_actuation(p: RobotParams) -> list[JointActuation]:
    out = []
    for leg in LEGS:
        out.append(JointActuation(f"{leg}_abad", p.actuator("abad")))
        out.append(JointActuation(f"{leg}_hip", p.actuator("hip")))
        out.append(JointActuation(f"{leg}_knee", p.actuator("knee"), p.knee_belt_ratio))
    if p.include_neck:
        out.append(JointActuation("neck_pitch", p.actuator("neck")))
        out.append(JointActuation("head_pitch", p.actuator("neck")))
    return out


def mass_breakdown(p: RobotParams) -> dict[str, float]:
    """Mass of each body. The trunk absorbs whatever the target leaves over."""
    hip_link = p.actuator("hip").mass + p.actuator("knee").mass + p.hip_bracket_mass
    neck = p.neck_link_mass + p.actuator("neck").mass if p.include_neck else 0.0
    head = p.head_mass if p.include_neck else 0.0
    legs = 4 * (hip_link + p.thigh_mass + p.shin_mass + p.foot_mass)
    trunk = p.total_mass_target - legs - neck - head
    if trunk <= 0:
        raise ValueError(f"mass target {p.total_mass_target} kg is below leg+neck mass")
    # The trunk carries the ab/ad motors, the neck-pitch motor, battery and compute.
    fixed_in_trunk = 4 * p.actuator("abad").mass + (p.actuator("neck").mass if p.include_neck else 0)
    return {
        "trunk": trunk,
        "trunk_payload_available": trunk - fixed_in_trunk,
        "hip_link": hip_link,
        "thigh": p.thigh_mass,
        "shin_with_foot": p.shin_mass + p.foot_mass,
        "neck": neck,
        "head": head,
        "total": p.total_mass_target,
    }


def stand_pose(p: RobotParams, height_frac: float = 0.75) -> dict[str, float]:
    """Joint angles with every foot directly under its hip, at height_frac of full reach."""
    l1, l2 = p.thigh_length, p.shin_length
    d = height_frac * (l1 + l2)
    knee = math.pi - math.acos((l1**2 + l2**2 - d**2) / (2 * l1 * l2))
    hip_off = math.asin(l2 * math.sin(knee) / d)  # thigh angle from vertical
    q = {}
    for leg in LEGS:
        sign = p.knee_sign_front if leg[0] == "F" else p.knee_sign_rear
        q[f"{leg}_abad"] = 0.0
        # Knee backward (sign -1): thigh tilts back (+hip), shin swings forward (-knee).
        q[f"{leg}_hip"] = -sign * hip_off
        q[f"{leg}_knee"] = sign * knee
    if p.include_neck:
        q["neck_pitch"] = 0.0
        q["head_pitch"] = 0.0
    return q


def stand_height(p: RobotParams, height_frac: float = 0.75) -> float:
    """Trunk-origin height above the ground in stand_pose."""
    return height_frac * p.leg_reach + p.foot_radius - p.hip_z


def add_robot(worldbody: ET.Element, actuator: ET.Element, sensor: ET.Element,
              p: RobotParams, pos=(0.0, 0.0, 0.4), euler=(0.0, 0.0, 0.0)) -> None:
    """Add the robot under `worldbody`. `euler` is in degrees (intrinsic xyz)."""
    masses = mass_breakdown(p)
    trunk = ET.SubElement(worldbody, "body", name="trunk", pos=_fmt(*pos),
                          euler=_fmt(*(math.radians(e) for e in euler)))
    ET.SubElement(trunk, "freejoint", name="root_joint")  # name Crocoddyl expects
    ET.SubElement(
        trunk, "inertial", pos="0 0 0", mass=_fmt(masses["trunk"]),
        diaginertia=_fmt(*_box_inertia(masses["trunk"], p.body_length, p.body_width, p.body_height)),
    )
    _add_trunk_geoms(trunk, p)
    ET.SubElement(trunk, "site", name="imu", pos="0 0 0", size="0.01")

    for leg in LEGS:
        _add_leg(trunk, p, leg, masses)
    if p.include_neck:
        _add_neck(trunk, p, masses)

    for ja in joint_actuation(p):
        ET.SubElement(actuator, "motor", name=ja.joint, joint=ja.joint,
                      ctrlrange=_fmt(-ja.peak_torque, ja.peak_torque))
    ET.SubElement(sensor, "framequat", name="trunk_quat", objtype="site", objname="imu")
    ET.SubElement(sensor, "gyro", name="trunk_gyro", site="imu")
    ET.SubElement(sensor, "accelerometer", name="trunk_acc", site="imu")


def trunk_boxes(p: RobotParams) -> list[tuple[str, float, float, float]]:
    """Trunk planform as (name, x centre, half length, half width) boxes: the centre section
    and, if notched, the nose and tail."""
    hl = p.body_length / 2
    if not p.notched:
        return [("", 0.0, hl, p.body_width / 2)]
    xn, nw = p.trunk_notch_x, p.trunk_nose_width / 2
    return [("", 0.0, xn, p.body_width / 2), ("_nose", (xn + hl) / 2, (hl - xn) / 2, nw),
            ("_tail", -(xn + hl) / 2, (hl - xn) / 2, nw)]


def trunk_outline(p: RobotParams) -> np.ndarray:
    """Top-view outline of the trunk (closed polygon, counter-clockwise), x forward, y left."""
    hl, hw = p.body_length / 2, p.body_width / 2
    if not p.notched:
        return np.array([[hl, hw], [-hl, hw], [-hl, -hw], [hl, -hw]])
    xn, nw = p.trunk_notch_x, p.trunk_nose_width / 2
    q = [[hl, nw], [xn, nw], [xn, hw], [-xn, hw], [-xn, nw], [-hl, nw]]
    return np.array(q + [[x, -y] for x, y in reversed(q)])


def hip_stack_outlines(p: RobotParams) -> list[np.ndarray]:
    """Top-view rectangles (4 corners each) of the four hip stacks and thighs at zero ab/ad:
    from the hip motor's inner face out to the thigh plate's outer face."""
    y0, y1 = p.stack_y()["motors"][0], p.overall_width / 2
    r = p.hip_motor_radius
    out = []
    for fx in (1, -1):
        for sd in (1, -1):
            x = fx * p.hip_spacing / 2
            out.append(np.array([[x + r, sd * y0], [x + r, sd * y1], [x - r, sd * y1], [x - r, sd * y0]]))
    return out


def _add_trunk_geoms(trunk: ET.Element, p: RobotParams) -> None:
    """Shell and skid plate. The skid plate follows the planform: a flat plate under the belly,
    with cylindrical lips at both ends (as wide as the nose) so it rides onto nosings without
    catching."""
    for suffix, xc, hx, hy in trunk_boxes(p):
        ET.SubElement(trunk, "geom", name=f"trunk_shell{suffix}", type="box", material="shell",
                      pos=_fmt(xc, 0, 0), size=_fmt(hx, hy, p.body_height / 2), **TRUNK_COLLISION)
    skid_z = -p.body_height / 2 - p.skid_thickness / 2
    lip_r = p.skid_lip_radius
    flat_half = p.body_length / 2 - lip_r
    skid_common = dict(material="skid", priority="1", friction=_fmt(p.skid_friction, 0.005, 0.0001),
                       **TRUNK_COLLISION)
    for suffix, xc, hx, hy in trunk_boxes(p):
        x0, x1 = max(xc - hx, -flat_half), min(xc + hx, flat_half)  # the lips cap both ends
        ET.SubElement(trunk, "geom", name=f"skid{suffix}", type="box", pos=_fmt((x0 + x1) / 2, 0, skid_z),
                      size=_fmt((x1 - x0) / 2, hy, p.skid_thickness / 2), **skid_common)
    lip_half_w = p.trunk_nose_width / 2 if p.notched else p.body_width / 2
    for name, sx in (("skid_lip_front", 1), ("skid_lip_rear", -1)):
        ET.SubElement(
            trunk, "geom", name=name, type="cylinder", zaxis="0 1 0",
            pos=_fmt(sx * flat_half, 0, -p.body_height / 2 - p.skid_thickness + lip_r),
            size=_fmt(lip_r, lip_half_w), **skid_common,
        )


def hip_link_inertia(p: RobotParams) -> tuple[tuple[float, float, float], float, tuple[float, ...]]:
    """(COM, mass, full inertia xx yy zz xy xz yz) of the parts that roll with ab/ad, left hip,
    in the hip frame (origin on the ab/ad axis): the two motors as solid cylinders along y, and
    the ab/ad output bracket as a point mass midway between the motors."""
    m = mass_breakdown(p)["hip_link"]
    y0, y1 = p.stack_y()["motors"]
    yc = (y0 + y1) / 2 - p.hip_width / 2
    r, ln = p.hip_motor_radius, p.hip_motor_length
    parts = [(p.actuator("hip").mass, (0.0, yc, 0.0), True), (p.actuator("knee").mass, (0.0, yc, p.knee_motor_above), True),
             (m - p.actuator("hip").mass - p.actuator("knee").mass, (0.0, yc, p.knee_motor_above / 2), False)]
    com = sum(mi * np.array(c) for mi, c, _ in parts) / m
    inertia = np.zeros((3, 3))
    for mi, c, cyl in parts:
        if cyl:  # solid cylinder, axis y
            radial = mi * (3 * r**2 + ln**2) / 12
            inertia += np.diag([radial, mi * r**2 / 2, radial])
        d = np.array(c) - com
        inertia += mi * (d @ d * np.eye(3) - np.outer(d, d))
    full = (inertia[0, 0], inertia[1, 1], inertia[2, 2], inertia[0, 1], inertia[0, 2], inertia[1, 2])
    return tuple(com), m, full


def _joint_attrs(ja: JointActuation) -> dict[str, str]:
    s, r = ja.spec, ja.ratio
    return dict(armature=_fmt(s.armature * r**2), damping=_fmt(s.damping * r**2),
                frictionloss=_fmt(s.frictionloss * r))


def _add_leg(trunk: ET.Element, p: RobotParams, leg: str, masses: dict[str, float]) -> None:
    fx = 1 if leg[0] == "F" else -1
    side = 1 if leg[1] == "L" else -1
    knee_sign = p.knee_sign_front if fx == 1 else p.knee_sign_rear
    act = {ja.joint: ja for ja in joint_actuation(p)}
    l1, l2 = p.thigh_length, p.shin_length

    hip = ET.SubElement(trunk, "body", name=f"{leg}_hip",
                        pos=_fmt(fx * p.hip_spacing / 2, side * p.hip_width / 2, p.hip_z))
    com, m_hip, full = hip_link_inertia(p)
    ET.SubElement(hip, "inertial", pos=_fmt(com[0], side * com[1], com[2]), mass=_fmt(m_hip),
                  fullinertia=_fmt(full[0], full[1], full[2], side * full[3], full[4], side * full[5]))
    ET.SubElement(hip, "joint", name=f"{leg}_abad", axis="1 0 0", range="-0.6 0.6",
                  **_joint_attrs(act[f"{leg}_abad"]))
    # Hip stack: hip-pitch motor on the hip axis, knee motor above it, first belt stage outboard.
    spans = p.stack_y()
    y0, y1 = spans["motors"]
    yc = side * ((y0 + y1) / 2 - p.hip_width / 2)
    for name, z in (("hip_motor", 0.0), ("knee_motor", p.knee_motor_above)):
        ET.SubElement(hip, "geom", name=f"{leg}_{name}", type="cylinder", zaxis="0 1 0", pos=_fmt(0, yc, z),
                      size=_fmt(p.hip_motor_radius, p.hip_motor_length / 2), material="motor", **LEG_COLLISION)
    b0, b1 = spans["belt"]
    ET.SubElement(hip, "geom", name=f"{leg}_belt", type="box",
                  pos=_fmt(0, side * ((b0 + b1) / 2 - p.hip_width / 2), p.knee_motor_above / 2),
                  size=_fmt(p.belt_radius, (b1 - b0) / 2, p.knee_motor_above / 2 + p.belt_radius),
                  material="belt", **LEG_COLLISION)

    thigh = ET.SubElement(hip, "body", name=f"{leg}_thigh", pos=_fmt(0, side * p.abad_offset, 0))
    ET.SubElement(thigh, "inertial", pos=_fmt(0, 0, -l1 / 2), mass=_fmt(p.thigh_mass),
                  diaginertia=_fmt(*_rod_inertia(p.thigh_mass, l1, "z")))
    ET.SubElement(thigh, "joint", name=f"{leg}_hip", axis="0 1 0", range=_fmt(-2.6, 2.6),
                  **_joint_attrs(act[f"{leg}_hip"]))
    t = p.thigh_half_thickness
    ET.SubElement(thigh, "geom", name=f"{leg}_thigh_hub", type="cylinder", zaxis="0 1 0", pos="0 0 0",
                  size=_fmt(p.thigh_hub_radius, t), material="limb", **LEG_COLLISION)
    ET.SubElement(thigh, "geom", name=f"{leg}_thigh", type="box", pos=_fmt(0, 0, -l1 / 2),
                  size=_fmt(p.thigh_radius, t, l1 / 2), material="limb", **LEG_COLLISION)
    ET.SubElement(thigh, "geom", name=f"{leg}_knee_pulley", type="cylinder", zaxis="0 1 0", pos=_fmt(0, 0, -l1),
                  size=_fmt(p.knee_pulley_radius, t), material="limb", **LEG_COLLISION)

    shin = ET.SubElement(thigh, "body", name=f"{leg}_shin", pos=_fmt(0, 0, -l1))
    m_shin = p.shin_mass + p.foot_mass
    com_z = -(p.shin_mass * l2 / 2 + p.foot_mass * l2) / m_shin
    ET.SubElement(shin, "inertial", pos=_fmt(0, 0, com_z), mass=_fmt(m_shin),
                  diaginertia=_fmt(*_rod_inertia(m_shin, l2, "z")))
    knee_range = sorted((knee_sign * 2.7, knee_sign * 0.2))
    ET.SubElement(shin, "joint", name=f"{leg}_knee", axis="0 1 0", range=_fmt(*knee_range),
                  **_joint_attrs(act[f"{leg}_knee"]))
    ET.SubElement(shin, "geom", name=f"{leg}_shin", type="capsule", fromto=_fmt(0, 0, 0, 0, 0, -l2),
                  size=_fmt(p.shin_radius), material="limb", **LEG_COLLISION)
    ET.SubElement(shin, "geom", name=f"{leg}_foot", type="sphere", pos=_fmt(0, 0, -l2),
                  size=_fmt(p.foot_radius), material="foot", priority="1",
                  friction=_fmt(p.foot_friction, 0.01, 0.001), condim="6", **LEG_COLLISION)
    ET.SubElement(shin, "site", name=f"{leg}_foot", pos=_fmt(0, 0, -l2), size="0.005")


def _add_neck(trunk: ET.Element, p: RobotParams, masses: dict[str, float]) -> None:
    act = {ja.joint: ja for ja in joint_actuation(p)}
    neck = ET.SubElement(trunk, "body", name="neck", pos=_fmt(p.body_length / 2 - 0.03, 0, p.body_height / 2))
    ET.SubElement(neck, "inertial", pos=_fmt(p.neck_length / 2, 0, 0), mass=_fmt(masses["neck"]),
                  diaginertia=_fmt(*_rod_inertia(masses["neck"], p.neck_length, "x", r=0.03)))
    # Positive pitch about +y tips the neck down.
    ET.SubElement(neck, "joint", name="neck_pitch", axis="0 1 0", range="-1.6 1.2",
                  **_joint_attrs(act["neck_pitch"]))
    ET.SubElement(neck, "geom", name="neck", type="capsule", fromto=_fmt(0, 0, 0, p.neck_length, 0, 0),
                  size="0.02", material="limb", **TRUNK_COLLISION)

    head = ET.SubElement(neck, "body", name="head", pos=_fmt(p.neck_length, 0, 0))
    ET.SubElement(head, "inertial", pos=_fmt(p.head_length / 2, 0, 0), mass=_fmt(masses["head"]),
                  diaginertia=_fmt(*_box_inertia(masses["head"], p.head_length, 0.08, 0.06)))
    ET.SubElement(head, "joint", name="head_pitch", axis="0 1 0", range="-1.6 1.6",
                  **_joint_attrs(act["head_pitch"]))
    ET.SubElement(head, "geom", name="head", type="box", pos=_fmt(p.head_length / 2, 0, 0),
                  size=_fmt(p.head_length / 2, 0.04, 0.03), material="shell", **TRUNK_COLLISION)
    ET.SubElement(head, "site", name="mouth", pos=_fmt(p.head_length, 0, -0.02), size="0.008")
