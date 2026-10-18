"""Offscreen MuJoCo renders (EGL), for reports. Needs no display.

MUJOCO_GL must be set before mujoco is first imported; this module sets it, but only
helps if it is imported before anything else that imports mujoco.
"""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

from dataclasses import dataclass  # noqa: E402

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402


@dataclass(frozen=True)
class View:
    name: str
    lookat: tuple[float, float, float]
    distance: float
    azimuth: float
    elevation: float


def render_views(model: mujoco.MjModel, qpos: np.ndarray | None, views: list[View], out_dir,
                 prefix: str, size=(720, 1280)) -> list[str]:
    """Render each view to <out_dir>/<prefix>_<view>.png and return the file names."""
    data = mujoco.MjData(model)
    if qpos is not None:
        data.qpos[:] = qpos
    elif model.nkey:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    names = []
    with mujoco.Renderer(model, *size) as renderer:
        cam = mujoco.MjvCamera()
        for v in views:
            cam.lookat[:], cam.distance, cam.azimuth, cam.elevation = v.lookat, v.distance, v.azimuth, v.elevation
            renderer.update_scene(data, cam)
            name = f"{prefix}_{v.name}.png"
            Image.fromarray(renderer.render()).save(os.path.join(out_dir, name))
            names.append(name)
    return names


def planner_to_world_qpos(q_pin: np.ndarray, s, flight: str, facing: str) -> np.ndarray:
    """Map a Pinocchio configuration in the planner frame to a MuJoCo qpos in the stairwell.

    Downhill plans live in an x-mirrored frame. A mirror image of this left-right
    symmetric robot is the same robot with left and right legs swapped, ab/ad angles
    negated, and pitch joints unchanged. The upper flight sits beyond the landing,
    rotated 180 degrees.
    """
    import pinocchio as pin

    pos = q_pin[:3].copy()
    R = pin.Quaternion(q_pin[3:7]).matrix()  # coeffs are (x, y, z, w)
    joints = q_pin[7:].copy()
    if facing == "down":
        M, D = np.diag([-1.0, 1.0, 1.0]), np.diag([1.0, -1.0, 1.0])
        pos[0] = -pos[0]
        R = M @ R @ D
        legs = joints[:12].reshape(4, 3).copy()  # FL, FR, RL, RR
        legs = legs[[1, 0, 3, 2]]
        legs[:, 0] *= -1
        joints[:12] = legs.ravel()
    if flight == "upper":
        lower = s.flights[0]
        ys = 1 if s.turn_left else -1
        T = np.array([(lower.n_risers - 1) * s.run, ys * (s.landing_width - s.width), lower.n_risers * s.rise])
        Rz = np.diag([-1.0, -1.0, 1.0])
        pos = T + Rz @ pos
        R = Rz @ R
    quat = pin.Quaternion(R).coeffs()  # x, y, z, w
    return np.concatenate([pos, [quat[3], quat[0], quat[1], quat[2]], joints])


def render_trajectory_gif(model: mujoco.MjModel, qpos_traj: np.ndarray, out_file, dt: float, every: float = 0.25,
                          size=(360, 640), azimuth: float = 120, elevation: float = -20, distance: float = 1.6,
                          fps: int = 12, smooth: float = 12.0) -> int:
    """Kinematic replay of a trajectory as a GIF, camera following the trunk. Returns frame count.

    The camera target is the trunk position averaged over a `smooth`-second window, so
    the view glides with the robot's progress instead of copying every sway and step.
    """
    data = mujoco.MjData(model)
    step = max(1, int(round(every / dt)))
    pos = np.asarray(qpos_traj)[:, :3]
    half = max(0, int(round(smooth / dt / 2)))
    padded = np.concatenate([np.repeat(pos[:1], half, 0), pos, np.repeat(pos[-1:], half, 0)])
    kernel = np.hanning(2 * half + 3)[1:-1] if half else np.ones(1)
    kernel /= kernel.sum()
    lookat = np.stack([np.convolve(padded[:, c], kernel, mode="valid") for c in range(3)], 1)
    images = []
    with mujoco.Renderer(model, *size) as renderer:
        cam = mujoco.MjvCamera()
        cam.distance, cam.azimuth, cam.elevation = distance, azimuth, elevation
        for q, target in zip(qpos_traj[::step], lookat[::step]):
            data.qpos[:] = q
            mujoco.mj_forward(model, data)
            cam.lookat[:] = target
            renderer.update_scene(data, cam)
            images.append(Image.fromarray(renderer.render()))
    images[0].save(out_file, save_all=True, append_images=images[1:], duration=int(1000 / fps), loop=0)
    return len(images)
