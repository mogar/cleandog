"""Stairwell geometry: a 2D sagittal profile (for analysis) and MJCF geoms (for sim).

Flight-local frame: x runs uphill, z up, y across the stair centred on 0.
The face of the first riser is at x = 0 and the floor at z = 0. Riser k
(1-based) has its face at x = (k-1)*run, and the tread above it has its top
at z = k*rise. Each tread is a board of thickness `tread_thickness`. It
overhangs its riser by `overhang` and ends in a full-round bullnose.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np

from .params import FlightParams, StairParams


@dataclass(frozen=True)
class Rect:
    x0: float
    x1: float
    z0: float
    z1: float

    def contains(self, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        return (x >= self.x0) & (x <= self.x1) & (z >= self.z0) & (z <= self.z1)


@dataclass(frozen=True)
class Circle:
    x: float
    z: float
    r: float

    def contains(self, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        return (x - self.x) ** 2 + (z - self.z) ** 2 <= self.r**2


def flight_profile(s: StairParams, flight: FlightParams, floor_extent: float = 1.0):
    """Solid shapes of one flight in its local x-z plane, including the floor in front."""
    t, o, nose_r = s.tread_thickness, s.overhang, s.tread_thickness / 2
    shapes: list[Rect | Circle] = [Rect(-floor_extent, 0.0, -0.05, 0.0)]  # floor slab
    for k in range(1, flight.n_risers + 1):
        xr = (k - 1) * s.run
        z = k * s.rise
        depth = s.run if k < flight.n_risers else flight.top_depth
        shapes.append(Rect(xr, xr + depth, -0.05, z - t))  # riser + stringer fill
        shapes.append(Rect(xr - o + nose_r, xr + depth, z - t, z))  # tread board
        shapes.append(Circle(xr - o + nose_r, z - nose_r, nose_r))  # bullnose
    return shapes


def inflate(shapes, r: float):
    """Grow every shape by r (rectangles stay rectangles, so corners are conservative)."""
    return [Rect(s.x0 - r, s.x1 + r, s.z0 - r, s.z1 + r) if isinstance(s, Rect) else Circle(s.x, s.z, s.r + r)
            for s in shapes]


def solid_mask(shapes, x: np.ndarray, z: np.ndarray) -> np.ndarray:
    inside = np.zeros(np.broadcast(x, z).shape, dtype=bool)
    for sh in shapes:
        inside |= sh.contains(x, z)
    return inside


def _fmt(*v: float) -> str:
    return " ".join(f"{x:.6g}" for x in v)


def _box(parent: ET.Element, lo, hi, material: str, name: str | None = None) -> None:
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    attrs = dict(type="box", pos=_fmt(*(lo + hi) / 2), size=_fmt(*(hi - lo) / 2), material=material)
    if name:
        attrs["name"] = name
    ET.SubElement(parent, "geom", **attrs)


def _rod(parent: ET.Element, a, b, radius: float, material: str, kind: str = "cylinder") -> None:
    ET.SubElement(parent, "geom", type=kind, fromto=_fmt(*a, *b), size=_fmt(radius), material=material)


def nosing_line_z(s: StairParams, x: float) -> float:
    """Height of the line through the nosing tips, at flight-local x."""
    return s.rise + (x + s.overhang) * s.rise / s.run


def _add_flight(parent: ET.Element, s: StairParams, flight: FlightParams, y_span=None,
                wall_x0: float = 0.0, wall_z0: float = 0.0) -> None:
    """Treads, risers and side treatments of one flight, in its local frame.

    `y_span` widens the top tread (the landing). `wall_z0` extends walls down below the
    flight's base, so the upper flight's walls reach the floor.
    """
    w, n = s.width, flight.n_risers
    nose_r = s.tread_thickness / 2
    shapes = flight_profile(s, flight, floor_extent=0.0)[1:]
    for idx, sh in enumerate(shapes):
        is_top = idx >= 3 * (n - 1)
        y0, y1 = y_span if (is_top and y_span) else (-w / 2, w / 2)
        if isinstance(sh, Rect):
            _box(parent, (sh.x0, y0, sh.z0), (sh.x1, y1, sh.z1), "wood")
        else:
            _rod(parent, (sh.x, y0, sh.z), (sh.x, y1, sh.z), nose_r, "wood")

    x_last = (n - 1) * s.run  # face of the top riser
    for side, sign in ((flight.left, 1), (flight.right, -1)):
        if side == "wall":
            y_in = sign * w / 2
            y_out = sign * (w / 2 + s.wall_thickness)
            _box(parent, (wall_x0, min(y_in, y_out), wall_z0),
                 (x_last, max(y_in, y_out), n * s.rise + s.wall_height), "wall")
        elif side == "balusters":
            y = sign * (w / 2 - s.baluster_inset)
            for k in range(1, n):
                x_nose = (k - 1) * s.run - s.overhang
                for j in range(s.balusters_per_tread):
                    x = x_nose + (j + 0.5) * s.run / s.balusters_per_tread
                    _rod(parent, (x, y, k * s.rise), (x, y, nosing_line_z(s, x) + s.handrail_height),
                         s.baluster_radius, "rail")
            x_newel = -0.1
            _rod(parent, (x_newel, y, 0.0), (x_newel, y, nosing_line_z(s, x_newel) + s.handrail_height),
                 0.04, "rail")
            _rod(parent, (x_newel, y, nosing_line_z(s, x_newel) + s.handrail_height),
                 (x_last, y, nosing_line_z(s, x_last) + s.handrail_height), 0.025, "rail", "capsule")
        elif side != "open":
            raise ValueError(f"unknown side treatment {side!r}")


def add_stairwell(worldbody: ET.Element, s: StairParams) -> None:
    """Append the stairwell as static bodies to an MJCF <worldbody>.

    The lower flight's frame is the world frame. With two flights, the upper one
    returns 180 degrees beside it, beyond a partition wall.
    """
    if len(s.flights) == 1:
        _add_flight(ET.SubElement(worldbody, "body", name="flight0"), s, s.flights[0], wall_x0=-0.3)
        return
    if len(s.flights) != 2:
        raise ValueError("stairwell supports one flight or two with a 180-degree landing")
    if s.partition_thickness < 0:
        raise ValueError("landing_width is narrower than two flights")

    lower, upper = s.flights
    w, L, ys = s.width, s.landing_width, (1 if s.turn_left else -1)
    x_top = (lower.n_risers - 1) * s.run
    z_top = lower.n_risers * s.rise
    z_ceiling = s.total_rise + s.wall_height
    landing_y = sorted((-ys * w / 2, ys * (L - w / 2)))
    x_end = x_top + lower.top_depth

    _add_flight(ET.SubElement(worldbody, "body", name="flight0"), s, lower,
                y_span=tuple(landing_y), wall_x0=-0.3)
    upper_body = ET.SubElement(worldbody, "body", name="flight1",
                               pos=_fmt(x_top, ys * (L - w), z_top), euler=_fmt(0, 0, math.pi))
    _add_flight(upper_body, s, upper, wall_z0=-z_top)

    walls = ET.SubElement(worldbody, "body", name="stairwell_walls")
    # Partition between the flights.
    gap_y = sorted((ys * w / 2, ys * (w / 2 + s.partition_thickness)))
    if s.partition_thickness > 0:
        _box(walls, (-0.3, gap_y[0], 0.0), (x_top, gap_y[1], z_ceiling), "wall", "partition")
    # Landing: far end and both sides.
    t = s.wall_thickness
    _box(walls, (x_end, landing_y[0] - t, 0.0), (x_end + t, landing_y[1] + t, z_ceiling), "wall")
    for y_edge, out in ((landing_y[0], -t), (landing_y[1], t)):
        _box(walls, (x_top, min(y_edge, y_edge + out), 0.0), (x_end, max(y_edge, y_edge + out), z_ceiling),
             "wall")
