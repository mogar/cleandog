"""Assemble a full MJCF scene: options, assets, floor, optional stairwell, robot."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import mujoco

from .params import RobotParams, StairParams
from .robot import add_robot, stand_height, stand_pose
from .stairs import add_stairwell

WORLD_COLLISION = dict(contype="1", conaffinity="2")


def build_scene_xml(robot: RobotParams, stairs: StairParams | None = None,
                    robot_pos: tuple[float, float, float] | None = None,
                    robot_euler: tuple[float, float, float] = (0.0, 0.0, 0.0),
                    timestep: float = 0.002) -> str:
    root = ET.Element("mujoco", model="cleandog")
    ET.SubElement(root, "compiler", angle="radian", autolimits="true")
    ET.SubElement(root, "option", timestep=f"{timestep:g}", integrator="implicitfast", cone="elliptic")
    ET.SubElement(root, "visual").append(ET.Element("global", offwidth="1280", offheight="720"))

    default = ET.SubElement(root, "default")
    ET.SubElement(default, "geom", solref="0.005 1", solimp="0.9 0.95 0.001")

    asset = ET.SubElement(root, "asset")
    ET.SubElement(asset, "texture", name="grid", type="2d", builtin="checker", width="512", height="512",
                  rgb1="0.82 0.80 0.76", rgb2="0.74 0.72 0.68")
    ET.SubElement(asset, "texture", name="sky", type="skybox", builtin="gradient", width="512",
                  height="512", rgb1="0.9 0.92 0.95", rgb2="0.6 0.65 0.7")
    ET.SubElement(asset, "material", name="floor", texture="grid", texrepeat="8 8", reflectance="0.05")
    for name, rgba in (("wood", "0.62 0.43 0.26 1"), ("wall", "0.9 0.9 0.88 0.25"),
                       ("shell", "0.95 0.95 0.93 1"), ("limb", "0.25 0.27 0.3 1"),
                       ("foot", "0.1 0.1 0.1 1"), ("rail", "0.35 0.24 0.15 1"), ("skid", "0.85 0.85 0.95 1")):
        ET.SubElement(asset, "material", name=name, rgba=rgba)

    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "light", pos="0 0 4", dir="0 0 -1", directional="true")
    ET.SubElement(world, "light", pos="-2 -3 3", dir="0.4 0.6 -0.5", directional="true", diffuse="0.4 0.4 0.4",
                  castshadow="false")
    ET.SubElement(world, "geom", name="floor", type="plane", size="0 0 0.05", material="floor",
                  **WORLD_COLLISION)
    if stairs is not None:
        stair_parent = ET.SubElement(world, "body", name="stairwell")
        add_stairwell(stair_parent, stairs)
        for g in stair_parent.iter("geom"):
            g.attrib.update(WORLD_COLLISION, friction=f"{stairs.friction:g} 0.005 0.0001")

    actuator = ET.SubElement(root, "actuator")
    sensor = ET.SubElement(root, "sensor")
    if robot_pos is None:
        robot_pos = (-0.8 if stairs else 0.0, 0.0, stand_height(robot))
    add_robot(world, actuator, sensor, robot, pos=robot_pos, euler=robot_euler)

    # Keyframe: standing pose at the spawn position.
    q = stand_pose(robot)
    qpos = list(robot_pos) + [1.0, 0.0, 0.0, 0.0] + list(q.values())
    if any(robot_euler):
        qpos[3:7] = _euler_deg_to_quat(robot_euler)
    keyframe = ET.SubElement(root, "keyframe")
    ET.SubElement(keyframe, "key", name="stand", qpos=" ".join(f"{v:.6g}" for v in qpos))

    ET.indent(root)
    return ET.tostring(root, encoding="unicode")


def _euler_deg_to_quat(euler_deg) -> list[float]:
    import numpy as np

    quat = np.zeros(4)
    mujoco.mju_euler2Quat(quat, np.deg2rad(euler_deg), "xyz")
    return quat.tolist()


def load_model(robot: RobotParams, stairs: StairParams | None = None, **kw) -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string(build_scene_xml(robot, stairs, **kw))
