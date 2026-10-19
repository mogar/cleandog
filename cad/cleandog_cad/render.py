"""Headless preview renders of build123d shapes (matplotlib), for reports and for checking parts
without a display. Interactive viewing is OCP CAD Viewer in VS Code."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

PART_COLOURS = ("#8d8b85", "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#52514e")
# (elevation, azimuth). Azimuth 0 looks along -x (the front, showing y-z); -90 looks along +y (the side).
VIEWS = {"iso": (25, -60), "front": (0, 0), "side": (0, -90), "top": (90, -90)}


def mesh(shape, tol: float = 0.2):
    """Triangles of a shape, face by face. Faces that fail to triangulate (seen in some vendor
    STEP files) are skipped: these meshes are for looking at, not for geometry checks."""
    vs, ts, n = [], [], 0
    for face in shape.faces():
        try:
            verts, tris = face.tessellate(tol)
        except AttributeError:
            continue
        if not tris:
            continue
        vs.append(np.array([[v.X, v.Y, v.Z] for v in verts]))
        ts.append(np.array(tris) + n)
        n += len(verts)
    return np.concatenate(vs), np.concatenate(ts)


def render(parts: list[tuple], out_file, views=("iso", "front", "side", "top"), title: str = "",
           size: float = 4.2, tol: float = 0.3, axes_labels: bool = True):
    """parts: [(shape, colour or None, alpha)]. One panel per view; equal axis scaling."""
    meshes = [(mesh(s, tol), c or PART_COLOURS[i % len(PART_COLOURS)], a) for i, (s, c, a) in enumerate(parts)]
    allv = np.concatenate([m[0][0] for m in meshes])
    lo, hi = allv.min(0), allv.max(0)
    centre, half = (lo + hi) / 2, (hi - lo).max() / 2 * 1.05
    fig = plt.figure(figsize=(size * len(views), size + 0.4))
    light = np.array([0.4, -0.5, 0.75])
    light /= np.linalg.norm(light)
    for k, name in enumerate(views):
        ax = fig.add_subplot(1, len(views), k + 1, projection="3d")
        for (v, t), colour, alpha in meshes:
            tri = v[t]
            n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
            n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
            shade = 0.45 + 0.55 * np.abs(n @ light)
            base = np.array(matplotlib.colors.to_rgb(colour))
            fc = np.clip(base[None] * shade[:, None], 0, 1)
            ax.add_collection3d(Poly3DCollection(tri, facecolors=np.c_[fc, np.full(len(fc), alpha)],
                                                 edgecolors="none"))
        ax.set_xlim(centre[0] - half, centre[0] + half)
        ax.set_ylim(centre[1] - half, centre[1] + half)
        ax.set_zlim(centre[2] - half, centre[2] + half)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(*VIEWS[name])
        ax.set_title(name, fontsize=10)
        if axes_labels:
            ax.set_xlabel("x (mm)", fontsize=8)
            ax.set_ylabel("y (mm)", fontsize=8)
            ax.set_zlabel("z (mm)", fontsize=8)
        ax.tick_params(labelsize=7)
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_file, dpi=110)
    plt.close(fig)
