"""2D sagittal terrain: solid shapes, walkable surfaces and a body-height envelope.

A Terrain is what the planners see. Its x axis is the direction the robot
faces. For walking down the stairs, the uphill geometry is mirrored, so the
robot still moves toward +x.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .params import FlightParams, StairParams
from .stairs import Circle, Rect, flight_profile, inflate, solid_mask

GRID_RES = 0.003  # occupancy-grid cell size for collision queries, m


@dataclass(frozen=True)
class Surface:
    """A horizontal walking surface from x0 to x1 at height z.

    `sid` numbers surfaces in the uphill frame: 0 is the floor or landing below
    the flight, 1..n-1 are treads, and n is the top.
    """

    x0: float
    x1: float
    z: float
    sid: int


@dataclass(eq=False)
class Terrain:
    shapes: list
    surfaces: list[Surface]
    nosing_line: tuple[float, float, float, float] | None  # (x_a, z_a, x_b, z_b) first/last nosing tips
    slope_limit: float = 1.0  # max |dz/dx| of the body envelope
    _inflated: dict = field(default_factory=dict)
    _grids: dict = field(default_factory=dict)

    def inflated(self, r: float):
        key = round(r, 6)
        if key not in self._inflated:
            self._inflated[key] = inflate(self.shapes, r)
        return self._inflated[key]

    def solid(self, x, z, clearance: float = 0.0) -> np.ndarray:
        """Is (x, z) inside the terrain grown by `clearance`? Uses a cached occupancy grid
        (GRID_RES cells, sampled at cell centres) over the terrain's bounding box; outside
        the box is open air."""
        x, z = np.broadcast_arrays(np.asarray(x, float), np.asarray(z, float))
        key = round(clearance, 6)
        if key not in self._grids:
            self._grids[key] = self._build_grid(clearance)
        x0, z0, grid = self._grids[key]
        ix = np.floor((x - x0) / GRID_RES).astype(int)
        iz = np.floor((z - z0) / GRID_RES).astype(int)
        inside = (ix >= 0) & (ix < grid.shape[0]) & (iz >= 0) & (iz < grid.shape[1])
        out = np.zeros(x.shape, bool)
        out[inside] = grid[ix[inside], iz[inside]]
        return out

    def _build_grid(self, clearance: float):
        shapes = self.inflated(clearance)
        xs = [v for s in shapes for v in ((s.x0, s.x1) if isinstance(s, Rect) else (s.x - s.r, s.x + s.r))]
        zs = [v for s in shapes for v in ((s.z0, s.z1) if isinstance(s, Rect) else (s.z - s.r, s.z + s.r))]
        x0, x1, z0, z1 = min(xs) - 0.01, max(xs) + 0.01, min(zs) - 0.01, max(zs) + 0.01
        gx = x0 + (np.arange(int((x1 - x0) / GRID_RES) + 1) + 0.5) * GRID_RES
        gz = z0 + (np.arange(int((z1 - z0) / GRID_RES) + 1) + 0.5) * GRID_RES
        grid = np.zeros((len(gx), len(gz)), bool)
        for sh in shapes:  # rasterise each shape into its own sub-window
            if isinstance(sh, Rect):
                ia, ib = np.searchsorted(gx, sh.x0, "left"), np.searchsorted(gx, sh.x1, "right")
                ja, jb = np.searchsorted(gz, sh.z0, "left"), np.searchsorted(gz, sh.z1, "right")
                grid[ia:ib, ja:jb] = True
            else:
                ia, ib = np.searchsorted(gx, sh.x - sh.r, "left"), np.searchsorted(gx, sh.x + sh.r, "right")
                ja, jb = np.searchsorted(gz, sh.z - sh.r, "left"), np.searchsorted(gz, sh.z + sh.r, "right")
                X, Z = np.meshgrid(gx[ia:ib], gz[ja:jb], indexing="ij")
                grid[ia:ib, ja:jb] |= (X - sh.x) ** 2 + (Z - sh.z) ** 2 <= sh.r**2
        return x0, z0, grid

    def raw_envelope(self, x: np.ndarray) -> np.ndarray:
        """Highest walking surface or nosing line at each x (step-shaped)."""
        x = np.asarray(x, float)
        z = np.full(x.shape, -np.inf)
        for s in self.surfaces:
            z = np.where((x >= s.x0) & (x <= s.x1), np.maximum(z, s.z), z)
        if self.nosing_line:
            xa, za, xb, zb = self.nosing_line
            lo, hi = min(xa, xb), max(xa, xb)
            line = za + (x - xa) * (zb - za) / (xb - xa)
            z = np.where((x >= lo) & (x <= hi), np.maximum(z, line), z)
        if np.isinf(z).any():  # outside every surface: hold the nearest value
            xs = np.array([v for s in self.surfaces for v in (s.x0, s.x1)])
            zs = np.array([s.z for s in self.surfaces for _ in (0, 1)])
            nearest = zs[np.argmin(np.abs(x[..., None] - xs), axis=-1)]
            z = np.where(np.isinf(z), nearest, z)
        return z

    def envelope(self, x) -> np.ndarray:
        """Envelope smoothed so its slope never exceeds slope_limit (a cone dilation).

        A step up becomes a ramp that starts before the step, so the body starts
        pitching up before the front legs reach the first riser.
        """
        x = np.atleast_1d(np.asarray(x, float))
        grid = np.arange(x.min() - 1.0, x.max() + 1.0, 0.005)
        zg = self.raw_envelope(grid)
        return np.max(zg[None, :] - self.slope_limit * np.abs(x[:, None] - grid[None, :]), axis=1)

    def mirrored(self) -> "Terrain":
        shapes = [Rect(-s.x1, -s.x0, s.z0, s.z1) if isinstance(s, Rect) else Circle(-s.x, s.z, s.r)
                  for s in self.shapes]
        surfaces = [Surface(-s.x1, -s.x0, s.z, s.sid) for s in self.surfaces]
        nl = self.nosing_line
        return Terrain(shapes, surfaces, (-nl[0], nl[1], -nl[2], nl[3]) if nl else None, self.slope_limit)


def flight_terrain(s: StairParams, n_risers: int, low_len: float, low_wall: bool,
                   top_len: float, top_wall: bool) -> Terrain:
    """One flight in the uphill frame: a flat of `low_len` before riser 1, the flight, and a
    flat of `top_len` beyond the top nosing. Walls bound either flat if requested."""
    shapes = flight_profile(s, FlightParams(n_risers, top_len), floor_extent=low_len)
    x_top_riser = (n_risers - 1) * s.run
    if low_wall:
        shapes.append(Rect(-low_len - 0.1, -low_len, -0.05, 3.5))
    if top_wall:
        shapes.append(Rect(x_top_riser + top_len, x_top_riser + top_len + 0.1, -0.05, 3.5))
    surfaces = [Surface(-low_len, 0.0, 0.0, 0)]
    for k in range(1, n_risers + 1):
        xr = (k - 1) * s.run
        x1 = xr + (s.run if k < n_risers else top_len)
        surfaces.append(Surface(xr - s.overhang, x1, k * s.rise, k))
    tips = (-s.overhang, s.rise, x_top_riser - s.overhang, n_risers * s.rise)
    return Terrain(shapes, surfaces, tips)


def interior_terrain(s: StairParams) -> Terrain:
    """A long single flight, so every pocket analysed is an interior one."""
    return _INTERIOR.setdefault(s, flight_terrain(s, 40, 1.0, False, 1.0, False))


_INTERIOR: dict[StairParams, Terrain] = {}


def foothold_candidates(t: Terrain, foot_radius: float, step: float = 0.02,
                        nosing_margin: float = 0.01) -> tuple[np.ndarray, np.ndarray]:
    """Foot-centre positions on every surface, at least a foot radius plus a margin clear of edges.

    Returns (points (N, 2), surface ids (N,)).
    """
    pts, sids = [], []
    for s in t.surfaces:
        lo = s.x0 + foot_radius + nosing_margin
        hi = s.x1 - foot_radius
        if hi < lo:
            continue
        xs = np.arange(lo, hi + 1e-9, step)
        # Keep only spots whose foot sphere clears every solid (e.g. walls).
        # Exact test here (not the grid): the foot sphere touches the surface by construction.
        xs = xs[~solid_mask(t.inflated(foot_radius - 1e-4), xs, np.full_like(xs, s.z + foot_radius))]
        pts.append(np.stack([xs, np.full_like(xs, s.z + foot_radius)], 1))
        sids.append(np.full(len(xs), s.sid))
    return np.concatenate(pts), np.concatenate(sids)
