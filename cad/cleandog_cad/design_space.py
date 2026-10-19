"""Design-space study: how narrow can the robot be, given the real motors and the real stair motion?

Body frame, millimetres (x forward, y left, z up), hips at body mid-height, as in the sim. One
design = a trunk planform + a hip arrangement + motor choices. Each design is tested by sweeping
its left-side leg geometry through every joint configuration of the M0 round-trip reference
(40,644 frames, all four legs; right legs are mirrored onto the left) and measuring the gap to
the trunk. `solve` then finds the narrowest leg plane that keeps `clearance` everywhere.

Two checkers, so each checks the other:
- `sweep_clearance`: point samples of every moving part against a signed-distance model of the
  trunk (numpy; fast enough to bisect).
- `mujoco_replay` (in mj_check.py): the same geometry as MuJoCo primitives, replayed through the
  reference in the stairwell; it also measures the motors against the stairs.

Motion: a design does not replay the reference's joint angles (with a wider ab/ad offset the same
angles would move the feet up or down by centimetres). It keeps the reference footholds, each moved
sideways to sit under its own leg plane, and solves its own IK (`motion_for`). The trunk follows
the reference path. A full re-plan with the new `hip_width` / `abad_offset` is the follow-up.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .vendor import DM_J4310P, ROBSTRIDE_02, Motor

REPO = Path(__file__).resolve().parents[2]
REFERENCE = REPO / "outputs" / "2026-10-18_rl" / "reference.npz"

HIP_X = 125.0  # sim hip_spacing / 2
BODY_LEN = 378.0  # sim body_length
BODY_TOP, BODY_BOT = 40.0, -52.0  # shell top; skid plate bottom (80 mm shell + 12 mm skid)
L1 = L2 = 160.0


@dataclass(frozen=True)
class Design:
    key: str
    label: str
    hip_motor: Motor = ROBSTRIDE_02
    knee_motor: Motor = ROBSTRIDE_02
    abad_motor: Motor = DM_J4310P
    knee_mount: str = "coaxial"  # "coaxial" (outboard of the thigh) | "forward" | "above"
    knee_offset: float = 0.0  # knee motor axis from the hip axis (forward/above mounts), mm
    knee_belt_ratio: float = 1.0  # extra reduction in the knee belt
    body_width: float = 200.0  # trunk (centre section) width
    notch: bool = False  # corner notches at the hips
    nose_half: float = 30.0  # half-width of the trunk ahead of / behind the notches
    notch_gap: float = 6.0  # hip-motor housing to the centre section's end wall (ab/ad bracket flange)
    clearance: float = 3.0  # moving part to trunk, everywhere in the round trip
    abad_wall: float = 3.0  # ab/ad motor housing to trunk wall
    thigh_half_t: float = 12.0  # thigh plate, half-thickness along y (holds the knee belt)
    thigh_half_h: float = 18.0  # thigh, half-depth in the leg plane
    shin_r: float = 13.0
    knee_pulley_r: float = 20.0
    belt_band: float = 10.0  # first-stage belt + pulleys, off-axis knee only
    notes: str = ""

    @property
    def ya(self) -> float:
        """Ab/ad axis from the centreline: the ab/ad motor as far out as the trunk wall allows."""
        return self.body_width / 2 - self.abad_motor.outer_d / 2 - self.abad_wall

    @property
    def hip_r(self) -> float:
        return self.hip_motor.outer_d / 2

    @property
    def notch_x(self) -> float:
        """Front/rear end of the full-width centre section: just behind the hip-motor housing."""
        r = self.hip_r
        if self.knee_mount == "coaxial":
            r = max(r, self.knee_motor.outer_d / 2)
        return HIP_X - r - self.notch_gap

    def stack_y(self, y_leg: float) -> dict[str, tuple[float, float]]:
        """y spans of the parts that roll with ab/ad (but not pitch), for a given leg plane."""
        t = self.thigh_half_t
        if self.knee_mount == "coaxial":
            y0 = y_leg - t - self.hip_motor.length
            return {"hip motor": (y0, y_leg - t), "knee motor": (y_leg + t, y_leg + t + self.knee_motor.length)}
        y1 = y_leg - t - self.belt_band
        return {"hip motor": (y1 - self.hip_motor.length, y1), "knee motor": (y1 - self.knee_motor.length, y1),
                "belt": (y1, y_leg - t)}

    def knee_axis_xz(self, fx: int) -> tuple[float, float]:
        """Knee motor axis relative to the hip axis, in the sagittal plane. fx = +1 front, -1 rear."""
        if self.knee_mount == "forward":
            return fx * self.knee_offset, 0.0
        if self.knee_mount == "above":
            return 0.0, self.knee_offset
        return 0.0, 0.0

    def outer_y(self, y_leg: float) -> float:
        """Outermost point of the robot (half the overall width)."""
        spans = self.stack_y(y_leg)
        return max(max(b for _, b in spans.values()), y_leg + self.thigh_half_t)

    def boxes(self) -> list[tuple[float, float, float, float]]:
        """Trunk planform as a union of rectangles (x0, x1, y0, y1), extruded over BODY_BOT..BODY_TOP."""
        hx, hw = BODY_LEN / 2, self.body_width / 2
        if not self.notch:
            return [(-hx, hx, -hw, hw)]
        xn, n = self.notch_x, self.nose_half
        return [(-xn, xn, -hw, hw), (xn, hx, -n, n), (-hx, -xn, -n, n)]


# ------------------------------------------------------------------ the reference motion

SIM_YA, SIM_LEG = 70.0, 120.0  # the sim's ab/ad axis (hip_width / 2) and leg plane (+ abad_offset)


@dataclass
class Reference:
    """The M0 round trip, with every foot in the body frame (mm), right legs mirrored onto the left."""
    qpos: np.ndarray = field(repr=False)  # (N, 21) MuJoCo qpos, sim geometry
    feet: np.ndarray = field(repr=False)  # (N, 4, 3) LEGS order, y > 0 for every leg
    dt: float = 0.02
    segment: np.ndarray = field(default=None, repr=False)
    segment_names: list[str] = field(default_factory=list)

    @property
    def n_frames(self) -> int:
        return len(self.qpos)


def leg_fk(q: np.ndarray, ya: float, y_leg: float, fx: int) -> np.ndarray:
    """Foot centre (left-side convention), body frame, for joint triples q (..., 3)."""
    a, h, k = q[..., 0], q[..., 1], q[..., 2]
    x = -L1 * np.sin(h) - L2 * np.sin(h + k)
    z = -L1 * np.cos(h) - L2 * np.cos(h + k)
    o = y_leg - ya
    return np.stack([x + fx * HIP_X, ya + o * np.cos(a) - z * np.sin(a), o * np.sin(a) + z * np.cos(a)], -1)


def leg_ik(feet: np.ndarray, ya: float, y_leg: float, fx: int) -> np.ndarray:
    """Inverse of leg_fk with the knee pointing backward (knee angle < 0)."""
    px, py, pz = feet[..., 0] - fx * HIP_X, feet[..., 1] - ya, feet[..., 2]
    o = y_leg - ya
    h = -np.sqrt(np.maximum(py**2 + pz**2 - o**2, 0.0))  # foot below the hip, in the leg plane
    a = np.arctan2(pz, py) - np.arctan2(h, o)
    a = (a + np.pi) % (2 * np.pi) - np.pi
    c = np.clip((px**2 + h**2 - L1**2 - L2**2) / (2 * L1 * L2), -1.0, 1.0)
    k = -np.arccos(c)
    hip = np.arctan2(-px, -h) - np.arctan2(L2 * np.sin(k), L1 + L2 * np.cos(k))
    return np.stack([a, hip, k], -1)


def load_reference(path: Path = REFERENCE) -> Reference:
    d = np.load(path)
    q = d["qpos"][:, 7:19].reshape(-1, 4, 3).copy()  # LEGS = FL, FR, RL, RR
    q[:, [1, 3], 0] *= -1  # right legs: mirror y, so ab/ad changes sign
    feet = np.stack([leg_fk(q[:, i], SIM_YA, SIM_LEG, 1 if i < 2 else -1) for i in range(4)], 1)
    return Reference(d["qpos"], feet, float(d["dt"]), d["segment"], [str(s) for s in d["segment_names"]])


ABAD_LIMIT = math.radians(34.4)  # sim joint range is +-0.6 rad


@dataclass
class Motion:
    """One design's joint motion, two ways:
    - replay: the reference's own joint angles (what the trained policy commands);
    - foothold IK: the reference footholds, each moved sideways to sit under the design's leg plane
      (as a re-plan with the wider stance would), solved with this design's IK and clipped to the
      ab/ad limit. Frames whose foothold needs more ab/ad than the limit, or is out of reach, are
      `infeasible`: a re-plan must move those footholds.
    The clearance sweep uses both sets of configurations (`configs`)."""
    q_ik: np.ndarray = field(repr=False)  # (N, 4, 3) left-side convention, clipped
    q_rep: np.ndarray = field(repr=False)
    infeasible: np.ndarray = field(repr=False)  # (N, 4) bool
    configs: dict[int, np.ndarray] = field(repr=False)  # fx -> (M, 3) unique triples (radians)
    inverse: dict[int, np.ndarray] = field(repr=False)  # fx -> (N, 4) index into configs: 2 legs x 2 ways
    max_abad_ik: float = 0.0  # degrees, unclipped: what the footholds would need
    n_frames: int = 0


def motion_for(ya: float, y_leg: float, ref: Reference, step_deg: float = 0.25) -> Motion:
    feet = ref.feet.copy()
    feet[..., 1] += y_leg - SIM_LEG
    q_ik = np.stack([leg_ik(feet[:, i], ya, y_leg, 1 if i < 2 else -1) for i in range(4)], 1)
    reach = np.stack([np.linalg.norm(leg_fk(q_ik[:, i], ya, y_leg, 1 if i < 2 else -1) - feet[:, i], axis=-1)
                      for i in range(4)], 1)
    max_abad = float(np.degrees(np.abs(q_ik[..., 0]).max()))
    infeasible = (np.abs(q_ik[..., 0]) > ABAD_LIMIT + 1e-9) | (reach > 1.0)
    q_ik[..., 0] = np.clip(q_ik[..., 0], -ABAD_LIMIT, ABAD_LIMIT)
    q_rep = ref.qpos[:, 7:19].reshape(-1, 4, 3).copy()
    q_rep[:, [1, 3], 0] *= -1
    configs, inverse = {}, {}
    for fx, legs in ((1, [0, 1]), (-1, [2, 3])):
        both = np.concatenate([q_ik[:, legs], q_rep[:, legs]], 1)  # (N, 4, 3)
        key = np.round(np.degrees(both.reshape(-1, 3)) / step_deg).astype(np.int32)
        uniq, inv = np.unique(key, axis=0, return_inverse=True)
        configs[fx] = np.radians(uniq * step_deg)
        inverse[fx] = inv.reshape(-1, 4)
    return Motion(q_ik, q_rep, infeasible, configs, inverse, max_abad, len(q_ik))


def mj_qpos(ref: Reference, motion: Motion) -> np.ndarray:
    """MuJoCo qpos for the design's motion (base and neck from the reference)."""
    q = motion.q_ik.copy()
    q[:, [1, 3], 0] *= -1
    out = ref.qpos.copy()
    out[:, 7:19] = q.reshape(len(q), 12)
    return out


# ------------------------------------------------------------------ geometry as point samples

def _cyl_y(xc, zc, r, y0, y1, h=3.0):
    """Surface points of a cylinder with its axis along y."""
    nth = max(12, int(2 * math.pi * r / h))
    th = np.linspace(0, 2 * math.pi, nth, endpoint=False)
    ys = np.linspace(y0, y1, max(2, int((y1 - y0) / h) + 1))
    side = np.array([(xc + r * math.cos(a), y, zc + r * math.sin(a)) for y in ys for a in th])
    caps = [np.array([(xc + rr * math.cos(a), y, zc + rr * math.sin(a)) for a in th])
            for rr in np.arange(r, 0, -h) for y in (y0, y1)]
    return np.vstack([side, *caps, [(xc, y0, zc), (xc, y1, zc)]])


def _box(c, half, h=3.0):
    """Surface points of an axis-aligned box (centre c, half sizes)."""
    g = [np.linspace(-s, s, max(2, int(2 * s / h) + 1)) for s in half]
    pts = []
    for ax in range(3):
        o = [i for i in range(3) if i != ax]
        A, B = np.meshgrid(g[o[0]], g[o[1]], indexing="ij")
        for sgn in (-1, 1):
            p = np.zeros((A.size, 3))
            p[:, ax] = sgn * half[ax]
            p[:, o[0]], p[:, o[1]] = A.ravel(), B.ravel()
            pts.append(p)
    return np.vstack(pts) + np.asarray(c)


BELT_R = 16.0  # first-stage pulley radius (belt envelope half-height)


def belt_box(des: Design, y_leg: float, fx: int) -> tuple[tuple, tuple]:
    """(centre, half sizes) of the first-stage belt envelope, hip frame at zero ab/ad."""
    y0, y1 = des.stack_y(y_leg)["belt"]
    kx, kz = des.knee_axis_xz(fx)
    return (kx / 2, (y0 + y1) / 2, kz / 2), (abs(kx) / 2 + BELT_R, (y1 - y0) / 2, abs(kz) / 2 + BELT_R)


def leg_parts(des: Design, y_leg: float, fx: int) -> dict[str, tuple[str, np.ndarray, float]]:
    """name -> (frame, points, inflate). Frames: 'stack' rolls about the ab/ad axis; 'thigh' also
    pitches about the hip axis; 'shin' also bends at the knee. Points are in the body frame at
    zero joint angles, hip axis at x = 0 (shifted to fx * HIP_X later). `inflate` is subtracted
    from the distance (capsules are sampled on their axis)."""
    parts = {}
    spans = des.stack_y(y_leg)
    parts["hip motor"] = ("stack", _cyl_y(0, 0, des.hip_r, *spans["hip motor"]), 0.0)
    kx, kz = des.knee_axis_xz(fx)
    parts["knee motor"] = ("stack", _cyl_y(kx, kz, des.knee_motor.outer_d / 2, *spans["knee motor"]), 0.0)
    if "belt" in spans:  # first belt stage: box envelope from the knee motor pulley to the hip-axis idler
        parts["belt"] = ("stack", _box(*belt_box(des, y_leg, fx)), 0.0)
        hub_r = 28.0
    else:
        hub_r = des.hip_r
    t = des.thigh_half_t
    parts["thigh hub"] = ("thigh", _cyl_y(0, 0, hub_r, y_leg - t, y_leg + t), 0.0)
    parts["thigh"] = ("thigh", _box((0, y_leg, -L1 / 2), (des.thigh_half_h, t, L1 / 2)), 0.0)
    parts["knee pulley"] = ("thigh", _cyl_y(0, -L1, des.knee_pulley_r, y_leg - t, y_leg + t), 0.0)
    s = np.linspace(0, L2, 60)
    parts["shin"] = ("shin", np.c_[np.zeros_like(s), np.full_like(s, y_leg), -s], des.shin_r)
    return parts


def _rx(a):
    c, s = np.cos(a), np.sin(a)
    R = np.zeros((len(a), 3, 3))
    R[:, 0, 0] = 1
    R[:, 1, 1], R[:, 1, 2], R[:, 2, 1], R[:, 2, 2] = c, -s, s, c
    return R


def _ry(a):
    c, s = np.cos(a), np.sin(a)
    R = np.zeros((len(a), 3, 3))
    R[:, 1, 1] = 1
    R[:, 0, 0], R[:, 0, 2], R[:, 2, 0], R[:, 2, 2] = c, s, -s, c
    return R


def transform(des: Design, frame: str, pts: np.ndarray, q: np.ndarray, fx: int) -> np.ndarray:
    """Body-frame positions (M, P, 3) of `pts` (in `frame`) for joint triples q (M, 3).
    Matches the MJCF: ab/ad about x through (hx, ya, 0); hip about y through the hip axis;
    knee about y at L1 down the thigh."""
    p = np.broadcast_to(pts, (len(q),) + pts.shape).copy()
    if frame == "shin":
        p = p @ _ry(q[:, 2]).transpose(0, 2, 1)
        p[..., 2] -= L1
    if frame in ("thigh", "shin"):
        p = p @ _ry(q[:, 1]).transpose(0, 2, 1)
    p[..., 1] -= des.ya
    p = p @ _rx(q[:, 0]).transpose(0, 2, 1)
    p[..., 1] += des.ya
    p[..., 0] += fx * HIP_X
    return p


def trunk_sdf(des: Design, p: np.ndarray) -> np.ndarray:
    """Signed distance to the trunk (union of extruded rectangles); negative inside."""
    zc, zh = (BODY_TOP + BODY_BOT) / 2, (BODY_TOP - BODY_BOT) / 2
    best = np.full(p.shape[:-1], np.inf)
    for x0, x1, y0, y1 in des.boxes():
        c = np.array([(x0 + x1) / 2, (y0 + y1) / 2, zc])
        h = np.array([(x1 - x0) / 2, (y1 - y0) / 2, zh])
        dq = np.abs(p - c) - h
        d = np.linalg.norm(np.maximum(dq, 0), axis=-1) + np.minimum(dq.max(-1), 0)
        best = np.minimum(best, d)
    return best


@dataclass
class Sweep:
    y_leg: float
    min_clear: float
    worst_part: str
    worst_fx: int
    worst_q: np.ndarray
    per_part: dict[str, float]
    max_abad: float  # degrees, unclipped foothold IK
    infeasible: float  # fraction of leg-frames whose foothold this design cannot reach within the ab/ad limit
    per_frame: np.ndarray | None = None  # min clearance per reference frame (both sides, all legs)


CULL = 25.0  # mm. Clearances above this are reported as a lower bound, not computed exactly.


def _clusters(pts: np.ndarray, size: int = 150) -> list[np.ndarray]:
    """Split a point cloud along its widest axis until each piece has <= size points."""
    if len(pts) <= size:
        return [pts]
    order = np.argsort(pts[:, int(np.argmax(pts.var(0)))])
    h = len(pts) // 2
    return _clusters(pts[order[:h]], size) + _clusters(pts[order[h:]], size)


def part_clearance(des: Design, frame: str, pts: np.ndarray, inflate: float, Q: np.ndarray, fx: int,
                   chunk: int = 2000, stop_below: float | None = None) -> np.ndarray:
    """Clearance of one part to the trunk for every config in Q. Each cluster of points is first
    bounded by its sphere; only configs whose bound is under CULL get the exact point query.
    With stop_below, returns as soon as any config is found below it (closest clusters first)."""
    out = np.full(len(Q), np.inf)
    bounds = []
    for c in _clusters(pts):
        cc = c.mean(0)
        rb = float(np.linalg.norm(c - cc, axis=1).max())
        bounds.append((trunk_sdf(des, transform(des, frame, cc[None], Q, fx))[:, 0] - rb - inflate, c))
    bounds.sort(key=lambda b: b[0].min())
    for lb, c in bounds:
        d = lb.copy()
        idx = np.flatnonzero(lb < CULL)
        if stop_below is not None:
            idx = idx[np.argsort(lb[idx])]
        for i in range(0, len(idx), chunk):
            sel = idx[i:i + chunk]
            d[sel] = trunk_sdf(des, transform(des, frame, c, Q[sel], fx)).min(1) - inflate
            if stop_below is not None and d[sel].min() < stop_below:
                return np.minimum(out, d)
        out = np.minimum(out, d)
    return out


def sweep_clearance(des: Design, y_leg: float, ref: Reference, per_frame: bool = False, ik_only_frames: bool = False,
                    stop_below: float | None = None) -> Sweep:
    """Min clearance over both motions. per_frame: also the min per reference frame (over the foothold-IK
    motion only if ik_only_frames, for comparing with the MuJoCo replay, which plays that one)."""
    motion = motion_for(des.ya, y_leg, ref)
    worst = (math.inf, "", 0, None)
    per_part: dict[str, float] = {}
    frame_min = np.full(motion.n_frames, np.inf) if per_frame else None
    for fx in (1, -1):
        Q = motion.configs[fx]
        cfg_min = np.full(len(Q), np.inf)
        for name, (frame, pts, inflate) in leg_parts(des, y_leg, fx).items():
            dmin = part_clearance(des, frame, pts, inflate, Q, fx, stop_below=stop_below)
            k = int(dmin.argmin())
            if stop_below is not None and dmin[k] < stop_below:  # bisection probe: failing is all it needs
                return Sweep(y_leg, float(dmin[k]), name, fx, Q[k], {}, motion.max_abad_ik,
                             float(motion.infeasible.mean()))
            per_part[name] = min(per_part.get(name, math.inf), float(dmin[k]))
            if dmin[k] < worst[0]:
                worst = (float(dmin[k]), name, fx, Q[k])
            cfg_min = np.minimum(cfg_min, dmin)
        if per_frame:
            cols = motion.inverse[fx][:, :2] if ik_only_frames else motion.inverse[fx]
            frame_min = np.minimum(frame_min, cfg_min[cols].min(1))
    return Sweep(y_leg, worst[0], worst[1], worst[2], worst[3], per_part, motion.max_abad_ik, float(motion.infeasible.mean()), frame_min)


def solve(des: Design, ref: Reference, tol: float = 0.25) -> Sweep:
    """Narrowest leg plane (bisection) with `clearance` over the whole round trip."""
    lo = des.body_width / 2 * (0.0 if des.notch else 1.0)
    hi = des.body_width / 2 + 250.0
    assert sweep_clearance(des, hi, ref).min_clear >= des.clearance
    while hi - lo > tol:
        mid = (lo + hi) / 2
        if sweep_clearance(des, mid, ref, stop_below=des.clearance).min_clear >= des.clearance:
            hi = mid
        else:
            lo = mid
    return sweep_clearance(des, hi, ref, per_frame=True)


def static_roll_clearance(des: Design, y_leg: float, abad_deg: float = 34.4, step: float = 1.0) -> float:
    """Ab/ad-only check (hip and knee at the stand pose): the stack rolled over ±abad_deg."""
    a = np.radians(np.arange(-abad_deg, abad_deg + 1e-9, step))
    q = np.c_[a, np.full_like(a, 0.6), np.full_like(a, -1.2)]
    worst = math.inf
    for fx in (1, -1):
        for frame, pts, inflate in leg_parts(des, y_leg, fx).values():
            worst = min(worst, float(trunk_sdf(des, transform(des, frame, pts, q, fx)).min()) - inflate)
    return worst


# ------------------------------------------------------------------ summary numbers

@dataclass
class Summary:
    des: Design
    sweep: Sweep
    overall_width: float
    track: float  # foot-to-foot width at zero ab/ad (2 * leg plane)
    abad_offset: float  # leg plane - ab/ad axis (sim RobotParams.abad_offset)
    hip_width: float  # 2 * ab/ad axis (sim RobotParams.hip_width)
    length: float  # overall length incl. anything ahead of the trunk
    leg_mass: float  # motors per leg, kg
    knee_peak: float
    knee_rated: float
    hip_peak: float
    hip_rated: float
    trunk_area: float  # planform area of the trunk, cm^2


def motor_torque(m: Motor) -> tuple[float, float]:
    """(peak, rated) at the output, from the vendor notes ("17 Nm peak / 6 Nm rated")."""
    import re
    peak, rated = re.search(r"([\d.]+) Nm peak / ([\d.]+) Nm rated", m.notes).groups()
    return float(peak), float(rated)


def summarise(des: Design, sw: Sweep) -> Summary:
    y = sw.y_leg
    hp, hr = motor_torque(des.hip_motor)
    kp, kr = motor_torque(des.knee_motor)
    eff = 0.95 if des.knee_belt_ratio > 1 else 1.0
    kx, _ = des.knee_axis_xz(1)
    front = max(BODY_LEN / 2, HIP_X + kx + des.knee_motor.outer_d / 2, HIP_X + des.hip_r)
    area = sum((x1 - x0) * (y1 - y0) for x0, x1, y0, y1 in des.boxes()) / 100
    return Summary(des, sw, 2 * des.outer_y(y), 2 * y, y - des.ya, 2 * des.ya, 2 * front,
                   des.hip_motor.mass + des.knee_motor.mass + des.abad_motor.mass,
                   kp * des.knee_belt_ratio * eff, kr * des.knee_belt_ratio * eff, hp, hr, area)


def with_(des: Design, **kw) -> Design:
    return dataclasses.replace(des, **kw)


# ------------------------------------------------------------------ the designs tested

def _knee_gap(hip: Motor, knee: Motor, gap: float = 10.0) -> float:
    return hip.outer_d / 2 + knee.outer_d / 2 + gap


BASE = Design("B", "Baseline: box trunk, coaxial hip and knee motors (RS02)")
NOTCH = with_(BASE, key="N", label="N: corner notches at the hips", notch=True)
KNEE_UP = with_(BASE, key="K", label="K: knee motor off-axis, above the hip (2-stage belt)", knee_mount="above",
                knee_offset=_knee_gap(ROBSTRIDE_02, ROBSTRIDE_02))
KNEE_FWD = with_(KNEE_UP, key="Kf", label="Kf: knee motor off-axis, ahead of / behind the hip", knee_mount="forward")
DM = with_(BASE, key="M", label="M: DM-J4310P for hip and knee (2:1 knee belt)", hip_motor=DM_J4310P,
           knee_motor=DM_J4310P, knee_belt_ratio=2.0)
NK = with_(NOTCH, key="NK", label="N+K", knee_mount="above", knee_offset=KNEE_UP.knee_offset)
NM = with_(NOTCH, key="NM", label="N+M", hip_motor=DM_J4310P, knee_motor=DM_J4310P, knee_belt_ratio=2.0)
NKM = with_(NM, key="NKM", label="N+K+M", knee_mount="above", knee_offset=_knee_gap(DM_J4310P, DM_J4310P))
DESIGNS = [BASE, NOTCH, KNEE_UP, KNEE_FWD, DM, NK, NM, NKM]


def widest_nose(des: Design, y_leg: float, ref: Reference, tol: float = 0.5) -> float:
    """Widest trunk nose/tail (half-width) that keeps clearance with the leg plane at y_leg."""
    lo, hi = 0.0, des.body_width / 2
    if sweep_clearance(with_(des, nose_half=hi), y_leg, ref).min_clear >= des.clearance:
        return hi
    while hi - lo > tol:
        mid = (lo + hi) / 2
        ok = sweep_clearance(with_(des, nose_half=mid), y_leg, ref, stop_below=des.clearance).min_clear >= des.clearance
        lo, hi = (mid, hi) if ok else (lo, mid)
    return lo
