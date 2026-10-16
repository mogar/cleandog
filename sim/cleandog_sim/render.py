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
