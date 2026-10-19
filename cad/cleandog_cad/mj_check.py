"""MuJoCo cross-check of the design-space sweep.

Builds one design as MuJoCo primitives (trunk boxes, motor cylinders, thigh box, shin capsule),
puts it in the sim's stairwell and replays the round-trip reference kinematically. Contacts are
detected with a margin, so each frame yields signed distances for:
- leg parts vs the trunk: the same quantity as design_space.sweep_clearance (the two must agree);
- leg parts vs the stairs and floor: the motors are not in the sim model, so this is new. It shows
  whether the rolled hip stack dips into a nosing.

The base pose comes from the reference; the joints are the design's own IK onto the reference
footholds (design_space.motion_for).
"""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import mujoco
import numpy as np

from .design_space import BODY_BOT, BODY_TOP, HIP_X, L1, L2, REPO, Design, Reference, belt_box, mj_qpos, motion_for

sys.path.insert(0, str(REPO / "sim"))
from cleandog_sim.params import RobotParams, StairParams  # noqa: E402
from cleandog_sim.robot import LEGS  # noqa: E402
from cleandog_sim.scene import build_scene_xml  # noqa: E402

MARGIN = 0.04  # m: report distances below 40 mm
LEG = dict(contype="0", conaffinity="5", margin=f"{MARGIN}")  # meets world (1) and trunk (4)
TRUNK = dict(contype="4", conaffinity="0")
STACK_PARTS = ("hip motor", "knee motor", "belt")


def _f(*v):
    return " ".join(f"{x:.6g}" for x in v)


def scene_xml(des: Design, y_leg: float) -> str:
    """The sim's stairwell scene with the robot replaced by this design's collision geometry."""
    root = ET.fromstring(build_scene_xml(RobotParams(), StairParams()))
    world = root.find("worldbody")
    world.remove(world.find("body[@name='trunk']"))
    for tag in ("actuator", "sensor", "keyframe"):
        root.remove(root.find(tag))
    ET.SubElement(root.find("option"), "flag", filterparent="disable")
    mm = 1e-3
    trunk = ET.SubElement(world, "body", name="trunk", pos="0 0 0.4")
    ET.SubElement(trunk, "freejoint", name="root_joint")
    ET.SubElement(trunk, "inertial", pos="0 0 0", mass="5", diaginertia="0.1 0.1 0.1")
    zc, zh = (BODY_TOP + BODY_BOT) / 2 * mm, (BODY_TOP - BODY_BOT) / 2 * mm
    for i, (x0, x1, y0, y1) in enumerate(des.boxes()):
        ET.SubElement(trunk, "geom", name=f"trunk{i}", type="box", pos=_f((x0 + x1) / 2 * mm, (y0 + y1) / 2 * mm, zc),
                      size=_f((x1 - x0) / 2 * mm, (y1 - y0) / 2 * mm, zh), rgba="0.85 0.85 0.8 1", **TRUNK)
    spans = des.stack_y(y_leg)
    for leg in LEGS:
        fx = 1 if leg[0] == "F" else -1
        sd = 1 if leg[1] == "L" else -1
        hip = ET.SubElement(trunk, "body", name=f"{leg}_hip", pos=_f(fx * HIP_X * mm, sd * des.ya * mm, 0))
        ET.SubElement(hip, "inertial", pos="0 0 0", mass="0.1", diaginertia="1e-4 1e-4 1e-4")
        ET.SubElement(hip, "joint", name=f"{leg}_abad", axis="1 0 0")

        def cyl(parent, name, x, z, r, y0, y1, rgba):
            yc = sd * ((y0 + y1) / 2 - des.ya)
            ET.SubElement(parent, "geom", name=f"{leg}_{name}", type="cylinder", zaxis="0 1 0",
                          pos=_f(x * mm, yc * mm, z * mm), size=_f(r * mm, (y1 - y0) / 2 * mm), rgba=rgba, **LEG)

        cyl(hip, "hip motor", 0, 0, des.hip_r, *spans["hip motor"], "0.16 0.47 0.84 1")
        kx, kz = des.knee_axis_xz(fx)
        cyl(hip, "knee motor", kx, kz, des.knee_motor.outer_d / 2, *spans["knee motor"], "0.92 0.41 0.2 1")
        if "belt" in spans:
            (cx, cy, cz), (hx, hy, hz) = belt_box(des, y_leg, fx)
            ET.SubElement(hip, "geom", name=f"{leg}_belt", type="box", pos=_f(cx * mm, sd * (cy - des.ya) * mm, cz * mm),
                          size=_f(hx * mm, hy * mm, hz * mm), rgba="0.3 0.3 0.3 1", **LEG)
        thigh = ET.SubElement(hip, "body", name=f"{leg}_thigh", pos=_f(0, sd * (y_leg - des.ya) * mm, 0))
        ET.SubElement(thigh, "inertial", pos="0 0 0", mass="0.1", diaginertia="1e-4 1e-4 1e-4")
        ET.SubElement(thigh, "joint", name=f"{leg}_hip", axis="0 1 0")
        t = des.thigh_half_t
        hub_r = 28.0 if "belt" in spans else des.hip_r
        ET.SubElement(thigh, "geom", name=f"{leg}_thigh hub", type="cylinder", zaxis="0 1 0", pos="0 0 0",
                      size=_f(hub_r * mm, t * mm), rgba="0.93 0.63 0 1", **LEG)
        ET.SubElement(thigh, "geom", name=f"{leg}_thigh", type="box", pos=_f(0, 0, -L1 / 2 * mm),
                      size=_f(des.thigh_half_h * mm, t * mm, L1 / 2 * mm), rgba="0.25 0.27 0.3 1", **LEG)
        ET.SubElement(thigh, "geom", name=f"{leg}_knee pulley", type="cylinder", zaxis="0 1 0",
                      pos=_f(0, 0, -L1 * mm), size=_f(des.knee_pulley_r * mm, t * mm), rgba="0.25 0.27 0.3 1",
                      **LEG)
        shin = ET.SubElement(thigh, "body", name=f"{leg}_shin", pos=_f(0, 0, -L1 * mm))
        ET.SubElement(shin, "inertial", pos="0 0 0", mass="0.1", diaginertia="1e-4 1e-4 1e-4")
        ET.SubElement(shin, "joint", name=f"{leg}_knee", axis="0 1 0")
        ET.SubElement(shin, "geom", name=f"{leg}_shin", type="capsule", fromto=_f(0, 0, 0, 0, 0, -L2 * mm),
                      size=_f(des.shin_r * mm), rgba="0.25 0.27 0.3 1", **LEG)
    neck = ET.SubElement(trunk, "body", name="neck", pos="0.16 0 0.04")  # keeps nq = 21; no geometry
    ET.SubElement(neck, "inertial", pos="0 0 0", mass="0.1", diaginertia="1e-4 1e-4 1e-4")
    ET.SubElement(neck, "joint", name="neck_pitch", axis="0 1 0")
    head = ET.SubElement(neck, "body", name="head", pos="0.1 0 0")
    ET.SubElement(head, "inertial", pos="0 0 0", mass="0.1", diaginertia="1e-4 1e-4 1e-4")
    ET.SubElement(head, "joint", name="head_pitch", axis="0 1 0")
    return ET.tostring(root, encoding="unicode")


@dataclass
class Replay:
    frames: np.ndarray  # reference frame indices checked
    trunk: np.ndarray  # per frame: min leg-part-to-trunk distance, mm (MARGIN*1000 when none in range)
    stairs: dict[str, np.ndarray]  # part -> per frame min distance to stairs/floor, mm (feasible legs only)
    worst_stairs: dict[str, tuple[float, int, str]]  # part -> (distance, frame, leg)


def replay(des: Design, y_leg: float, ref: Reference, stride: int = 4) -> Replay:
    motion = motion_for(des.ya, y_leg, ref)
    qpos = mj_qpos(ref, motion)
    model = mujoco.MjModel.from_xml_string(scene_xml(des, y_leg))
    data = mujoco.MjData(model)
    trunk_ids = {i for i in range(model.ngeom) if model.geom(i).name.startswith("trunk")}
    leg_part = {}
    for i in range(model.ngeom):
        n = model.geom(i).name
        if n[:2] in LEGS and n[2] == "_":
            leg_part[i] = (n[:2], n[3:])
    parts = sorted({p for _, p in leg_part.values()})
    frames = np.arange(0, ref.n_frames, stride)
    far = MARGIN * 1000
    trunk = np.full(len(frames), far)
    stairs = {p: np.full(len(frames), far) for p in parts}
    worst = {p: (far, -1, "") for p in parts}
    for k, f in enumerate(frames):
        data.qpos[:] = qpos[f]
        mujoco.mj_fwdPosition(model, data)
        for c in data.contact[:data.ncon]:
            g1, g2 = int(c.geom1), int(c.geom2)
            leg_g, other = (g1, g2) if g1 in leg_part else (g2, g1)
            if leg_g not in leg_part:
                continue
            d = float(c.dist) * 1000
            if other in trunk_ids:
                trunk[k] = min(trunk[k], d)
            else:
                leg, p = leg_part[leg_g]
                if motion.infeasible[f, LEGS.index(leg)]:
                    continue  # this foothold must be re-planned anyway; its clipped pose is not meaningful
                stairs[p][k] = min(stairs[p][k], d)
                if d < worst[p][0]:
                    worst[p] = (d, int(f), leg)
    return Replay(frames, trunk, stairs, worst)


def render_frame(des: Design, y_leg: float, qpos: np.ndarray, out_file, cam: dict, size=(640, 480)) -> None:
    """Offscreen render (MUJOCO_GL=egl) of the design at one reference pose. cam: lookat, distance,
    azimuth, elevation."""
    import matplotlib.pyplot as plt

    model = mujoco.MjModel.from_xml_string(scene_xml(des, y_leg))
    model.vis.global_.offwidth, model.vis.global_.offheight = size
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    with mujoco.Renderer(model, size[1], size[0]) as r:
        c = mujoco.MjvCamera()
        c.lookat[:] = cam.get("lookat", data.qpos[:3])
        c.distance, c.azimuth, c.elevation = cam["distance"], cam["azimuth"], cam["elevation"]
        r.update_scene(data, camera=c)
        plt.imsave(out_file, r.render())
