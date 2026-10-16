"""Parametric 12-DoF quadruped (+ optional 2-DoF neck) emitted as MJCF.

Frames: trunk x forward, y left, z up. With every joint at zero, the legs hang
straight down and the neck points straight forward. The legs, in order, are
FL, FR, RL, RR, and each has the joints abad (about x), hip (about y) and
knee (about y).

Collision bits: robot geoms use contype=2/conaffinity=1 and the world uses
contype=1/conaffinity=2. The robot therefore hits the world but not itself.
Robot contact geoms set priority=1 so that their friction overrides the
world's.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass

from .params import ActuatorSpec, RobotParams

LEGS = ("FL", "FR", "RL", "RR")
ROBOT_COLLISION = dict(contype="2", conaffinity="1")


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
    ET.SubElement(trunk, "freejoint", name="root")
    ET.SubElement(
        trunk, "inertial", pos="0 0 0", mass=_fmt(masses["trunk"]),
        diaginertia=_fmt(*_box_inertia(masses["trunk"], p.body_length, p.body_width, p.body_height)),
    )
    ET.SubElement(
        trunk, "geom", name="trunk_shell", type="box", material="shell",
        size=_fmt(p.body_length / 2, p.body_width / 2, p.body_height / 2), **ROBOT_COLLISION,
    )
    # Skid plate: a flat plate under the belly, with cylindrical lips at both ends so it
    # rides onto nosings without catching.
    skid_z = -p.body_height / 2 - p.skid_thickness / 2
    lip_r = p.skid_lip_radius
    flat_half = p.body_length / 2 - lip_r
    skid_common = dict(material="skid", priority="1", friction=_fmt(p.skid_friction, 0.005, 0.0001),
                       **ROBOT_COLLISION)
    ET.SubElement(trunk, "geom", name="skid", type="box", pos=_fmt(0, 0, skid_z),
                  size=_fmt(flat_half, p.body_width / 2, p.skid_thickness / 2), **skid_common)
    for name, sx in (("skid_lip_front", 1), ("skid_lip_rear", -1)):
        ET.SubElement(
            trunk, "geom", name=name, type="cylinder", zaxis="0 1 0",
            pos=_fmt(sx * flat_half, 0, -p.body_height / 2 - p.skid_thickness + lip_r),
            size=_fmt(lip_r, p.body_width / 2), **skid_common,
        )
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
    ET.SubElement(hip, "inertial", pos=_fmt(0, side * p.abad_offset / 2, 0), mass=_fmt(masses["hip_link"]),
                  diaginertia=_fmt(*_box_inertia(masses["hip_link"], 0.08, 0.08, 0.08)))
    ET.SubElement(hip, "joint", name=f"{leg}_abad", axis="1 0 0", range="-0.6 0.6",
                  **_joint_attrs(act[f"{leg}_abad"]))

    thigh = ET.SubElement(hip, "body", name=f"{leg}_thigh", pos=_fmt(0, side * p.abad_offset, 0))
    ET.SubElement(thigh, "inertial", pos=_fmt(0, 0, -l1 / 2), mass=_fmt(p.thigh_mass),
                  diaginertia=_fmt(*_rod_inertia(p.thigh_mass, l1, "z")))
    ET.SubElement(thigh, "joint", name=f"{leg}_hip", axis="0 1 0", range=_fmt(-2.6, 2.6),
                  **_joint_attrs(act[f"{leg}_hip"]))
    ET.SubElement(thigh, "geom", name=f"{leg}_thigh", type="capsule", fromto=_fmt(0, 0, 0, 0, 0, -l1),
                  size=_fmt(p.thigh_radius), material="limb", **ROBOT_COLLISION)

    shin = ET.SubElement(thigh, "body", name=f"{leg}_shin", pos=_fmt(0, 0, -l1))
    m_shin = p.shin_mass + p.foot_mass
    com_z = -(p.shin_mass * l2 / 2 + p.foot_mass * l2) / m_shin
    ET.SubElement(shin, "inertial", pos=_fmt(0, 0, com_z), mass=_fmt(m_shin),
                  diaginertia=_fmt(*_rod_inertia(m_shin, l2, "z")))
    knee_range = sorted((knee_sign * 2.7, knee_sign * 0.2))
    ET.SubElement(shin, "joint", name=f"{leg}_knee", axis="0 1 0", range=_fmt(*knee_range),
                  **_joint_attrs(act[f"{leg}_knee"]))
    ET.SubElement(shin, "geom", name=f"{leg}_shin", type="capsule", fromto=_fmt(0, 0, 0, 0, 0, -l2),
                  size=_fmt(p.shin_radius), material="limb", **ROBOT_COLLISION)
    ET.SubElement(shin, "geom", name=f"{leg}_foot", type="sphere", pos=_fmt(0, 0, -l2),
                  size=_fmt(p.foot_radius), material="foot", priority="1",
                  friction=_fmt(p.foot_friction, 0.01, 0.001), condim="6", **ROBOT_COLLISION)
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
                  size="0.02", material="limb", **ROBOT_COLLISION)

    head = ET.SubElement(neck, "body", name="head", pos=_fmt(p.neck_length, 0, 0))
    ET.SubElement(head, "inertial", pos=_fmt(p.head_length / 2, 0, 0), mass=_fmt(masses["head"]),
                  diaginertia=_fmt(*_box_inertia(masses["head"], p.head_length, 0.08, 0.06)))
    ET.SubElement(head, "joint", name="head_pitch", axis="0 1 0", range="-1.6 1.6",
                  **_joint_attrs(act["head_pitch"]))
    ET.SubElement(head, "geom", name="head", type="box", pos=_fmt(p.head_length / 2, 0, 0),
                  size=_fmt(p.head_length / 2, 0.04, 0.03), material="shell", **ROBOT_COLLISION)
    ET.SubElement(head, "site", name="mouth", pos=_fmt(p.head_length, 0, -0.02), size="0.008")
