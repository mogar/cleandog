"""Quasi-static crawl planner for stair transitions (sagittal geometry, top-view stability).

The robot always moves toward +x in the planner frame. For walking down, the
uphill terrain is mirrored (see terrain.Terrain.mirrored).

A plan alternates two kinds of move:

- **Shift:** all four feet stay planted while the body moves. The trunk
  follows a smoothed terrain envelope at a chosen hip height `h`; its pitch
  comes from where the front and rear hips sit on that envelope.
- **Swing:** one leg moves to a new foothold while the other three hold the
  robot up. This is a free gait: any leg may go next. A fixed crawl order
  deadlocks at the first riser. Before a swing the body may
  sway sideways by up to `max_sway`, as real quadrupeds do in a crawl. Each
  sway is checked in 3D (`sway_leg_ok`): every planted leg, and the swing leg
  at its new foothold, must absorb it within the ab/ad, knee and hip limits.
  (Until 2026-10-20 it wasn't, and the reference had to shrink sways the legs
  couldn't make; the wider N+K stance made that common.)

Every pose must satisfy all of these:

- each planted leg reaches its foothold within the knee and hip limits, with
  its thigh and shin clear of the stairs (links treated as capsules);
- the trunk and skid plate clear the stairs and walls;
- some neck/head fold clears them too;
- the centre of mass, projected onto the floor, is inside the support polygon
  of the planted feet by `margin`.

Under those conditions vertical foot forces alone hold the robot, so no
friction is needed. Time reversal also holds: a valid plan played backwards
is valid. So "facing uphill" covers both ascent and rear-first descent, and
"facing downhill" covers both head-first descent and backing up.

Swings are checked too: with the body fixed, the foot must be able to travel
from the old foothold to the new one through foot positions where the whole
leg is valid (a 1 cm grid of the leg's free space), and that path becomes the
swing reference for Crocoddyl.

Not modelled here: dynamics, left/right differences in terrain, and the
belly carrying load.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import lsq_linear

from .ladder import leg_ik
from .params import RobotParams, StairParams
from .robot import LEGS, hip_link_inertia, mass_breakdown
from .terrain import Terrain, flight_terrain, foothold_candidates

G = 9.81
ABAD_LIMIT = 0.6  # rad, the MJCF ab/ad range
FLEX_LIMITS = (0.2, 2.7)  # rad, knee flexion range (ladder.leg_ik's default)
# (neck, head) pitch relative to the trunk, positive tips down. The planner takes the collision-free
# fold with the best margin per pose. The last four reach forward, level with a nose-up trunk, to move
# the COM uphill (the head and neck are ~10% of the mass); indices 0-6 are fixed (NECK_FOLDED etc.).
NECK_OPTIONS = ((0.0, 0.0), (-0.5, 0.5), (-1.0, 1.0), (-1.45, 1.45), (-1.45, 0.3), (0.4, 0.8), (0.0, 1.4),
                (0.3, 0.3), (0.6, 0.0), (0.6, 0.6), (0.9, 0.0))
HIP_LIMIT = 2.6


KNEES = {"back": (-1, -1), "fwd": (1, 1), "out": (1, -1), "in": (-1, 1)}  # (front, rear) knee signs
# "out": front knees forward and rear knees back, so the knees point away from each other.


def morphology(seg: float, knee: str, hip_spacing: float, base: RobotParams = RobotParams()) -> RobotParams:
    """A sweep variant: equal thigh/shin `seg`, a knee layout, and hip spacing. The trunk length
    tracks hip spacing, keeping the trunk's overhang beyond the hips constant."""
    kf, kr = KNEES[knee]
    return base.with_(thigh_length=seg, shin_length=seg, knee_sign_front=kf, knee_sign_rear=kr,
                      hip_spacing=hip_spacing, body_length=hip_spacing + (base.body_length - base.hip_spacing))


# ---------------------------------------------------------------- scenarios

@dataclass
class Scenario:
    flight: str  # "lower" | "upper"
    end: str  # "bottom" | "top"
    facing: str  # "up" | "down"
    terrain: Terrain  # planner frame (robot moves +x)
    n_risers: int
    s0: float  # initial trunk-centre x (planner frame)
    start_sids: tuple[int, ...]
    goal_sids: tuple[int, ...]

    @property
    def name(self) -> str:
        return f"{self.flight}-{self.end}-{self.facing}"

    @property
    def label(self) -> str:
        where = {("lower", "bottom"): "floor ↔ lower flight", ("lower", "top"): "lower flight ↔ landing",
                 ("upper", "bottom"): "landing ↔ upper flight", ("upper", "top"): "upper flight ↔ upper floor"}
        return f"{where[(self.flight, self.end)]}, facing {self.facing}hill"

    def goal(self, sids) -> bool:
        return all(s in self.goal_sids for s in sids)


def make_scenario(s: StairParams, flight: str, end: str, facing: str, body_length: float) -> Scenario:
    lower, upper = s.flights
    if flight == "lower":
        n, low_len, low_wall, top_len, top_wall = lower.n_risers, 1.5, False, lower.top_depth, True
    else:
        n, low_len, low_wall, top_len, top_wall = upper.n_risers, lower.top_depth, True, upper.top_depth, False
    t = flight_terrain(s, n, low_len, low_wall, top_len, top_wall)
    half = body_length / 2
    x_tip_top = (n - 1) * s.run - s.overhang
    treads = tuple(range(1, n))
    if end == "bottom" and facing == "up":
        s0 = max(-half - 0.15, -low_len + 0.03 + half) if low_wall else -half - 0.15
        start, goal = (0,), tuple(k for k in treads if k >= 3)
    elif end == "bottom" and facing == "down":
        s0 = -(3.5 * s.run)
        start, goal = treads, (0,)
    elif end == "top" and facing == "up":
        s0 = (n - 4.5) * s.run
        start, goal = treads, (n,)
    else:  # top, down
        s0 = -x_tip_top - half - 0.15
        if top_wall:
            s0 = min(s0, -(x_tip_top + top_len) + 0.03 + half)
        start, goal = (n,), tuple(k for k in treads if k <= n - 3)
    if facing == "down":
        t = t.mirrored()
    return Scenario(flight, end, facing, t, n, s0, start, goal)


def all_scenarios(s: StairParams, body_length: float) -> list[Scenario]:
    return [make_scenario(s, fl, end, fa, body_length)
            for fl in ("lower", "upper") for end in ("bottom", "top") for fa in ("up", "down")]


# ---------------------------------------------------------------- robot geometry

@dataclass
class Poses:
    """Vectorised trunk poses (N,)."""

    centre: np.ndarray  # (N, 2)
    pitch: np.ndarray  # (N,)

    @property
    def ex(self):
        return np.stack([np.cos(self.pitch), np.sin(self.pitch)], -1)

    @property
    def ez(self):
        return np.stack([-np.sin(self.pitch), np.cos(self.pitch)], -1)

    def hip(self, p: RobotParams, front: bool) -> np.ndarray:
        sign = 1 if front else -1
        return self.centre + sign * self.ex * p.hip_spacing / 2 + self.ez * p.hip_z

    def __getitem__(self, i):
        return Poses(self.centre[i], self.pitch[i])


class Planner:
    def __init__(self, p: RobotParams, sc: Scenario, margin: float = 0.02,
                 n_heights: int = 7, budget: int = 1500, max_sway: float = 0.05, max_pitch: float | None = None,
                 prefer: float | None = None, limb_clear: float = 0.0):
        self.p, self.sc, self.t, self.margin, self.budget = p, sc, sc.terrain, margin, budget
        # Extra gap (m) between the leg links and the stairs, on top of the link radii. At 0 a
        # stance shin or thigh may graze a nosing; a policy tracking a few mm off then hits it
        # (every run-7 gate failure had a limb on the stairs just before it diverged).
        self.limb_clear = limb_clear
        self.footholds = None  # optional {leg index: allowed candidate indices}, set by repair()
        # Optional soft margin target (m). `margin` is a hard floor and the search is otherwise
        # indifferent to margin, so a plan can sit just above the floor for long stretches (the
        # N+K robot spent 42 s per ascent below 2 cm against a 1.5 cm floor, and a randomized trunk
        # COM tipped it there). With `prefer`, moves are ranked by their worst margin (move_margin)
        # in 5 mm steps up to `prefer`, ahead of progress.
        self.prefer = prefer
        # Optional cap on |trunk pitch| (rad). Shoves tip the robot backward where the trunk is
        # steeper than the stairs (M0 step 3d), so a cap trades reach for robustness.
        self.max_pitch = max_pitch
        # Hip heights above the stepped terrain envelope (nosing line on the flight, floor or
        # landing elsewhere): from belly 1 cm above it up to 95% of leg reach. Front and rear
        # are chosen independently.
        self.hs = list(np.round(np.linspace(p.hip_above_belly + 0.01, 0.95 * p.leg_reach + p.foot_radius,
                                            n_heights), 4))
        self.cand, self.cand_sid = foothold_candidates(self.t, p.foot_radius)
        xs = np.concatenate([[s.x0, s.x1] for s in self.t.surfaces])
        self._env_x = np.arange(xs.min() - 0.5, xs.max() + 0.5, 0.004)
        self._env_z = self.t.envelope(self._env_x)
        self._raw_z = self.t.raw_envelope(self._env_x)
        self.masses = mass_breakdown(p)
        self.wy = p.hip_width / 2 + p.abad_offset
        self.knee_back = {leg: (p.knee_sign_front if leg[0] == "F" else p.knee_sign_rear) == -1 for leg in LEGS}
        self.expanded = 0
        self.sways = np.linspace(-max_sway, max_sway, 11)
        self._swing_cache: dict = {}
        # Reach left for 3D effects the sagittal model ignores: the sway adds a sideways
        # component, and the ab/ad tilt that absorbs it lifts the hip-pitch joint (which sits
        # abad_offset outboard) by about 1 cm.
        self.reach_2d = math.sqrt((p.leg_reach - 0.015) ** 2 - max_sway**2)

    # -------------------------------------------------- poses

    def env(self, x):
        return np.interp(x, self._env_x, self._env_z)

    def body(self, s, h) -> Poses:
        s, h = np.atleast_1d(np.asarray(s, float)), np.broadcast_to(np.asarray(h, float), np.shape(np.atleast_1d(s)))
        hs = self.p.hip_spacing
        beta = np.zeros_like(s)
        for _ in range(25):
            c = np.cos(beta) * hs / 2
            zr, zf = self.env(s - c) + h, self.env(s + c) + h
            beta = np.arcsin(np.clip((zf - zr) / hs, -0.95, 0.95))
        c = np.cos(beta) * hs / 2
        mid = np.stack([s, (self.env(s - c) + self.env(s + c)) / 2 + h], -1)
        ez = np.stack([-np.sin(beta), np.cos(beta)], -1)
        return Poses(mid - ez * self.p.hip_z, beta)

    def pose(self, x, z, beta) -> Poses:
        x, z, beta = np.broadcast_arrays(*(np.atleast_1d(np.asarray(v, float)) for v in (x, z, beta)))
        return Poses(np.stack([x, z], -1), beta)

    def raw_env(self, x):
        return np.interp(x, self._env_x, self._raw_z)

    def target(self, x: float, h_front: float, h_rear: float) -> tuple[float, float, float]:
        """Trunk pose with centre at x and each hip at its own height above the stepped envelope."""
        hs, beta = self.p.hip_spacing, 0.0
        for _ in range(12):
            c = math.cos(beta) * hs / 2
            zf, zr = float(self.raw_env(x + c)) + h_front, float(self.raw_env(x - c)) + h_rear
            beta = math.asin(float(np.clip((zf - zr) / hs, -0.8, 0.8)))
        z_mid = (zf + zr) / 2
        return (round(x, 4), round(z_mid - math.cos(beta) * self.p.hip_z, 4), round(beta, 4))

    # -------------------------------------------------- checks (vectorised over poses)

    def leg_check(self, P: Poses, leg: str, feet: np.ndarray):
        """feet (N, 2) or (2,). Returns (ok (N,), knee (N, 2))."""
        hip = P.hip(self.p, leg[0] == "F")
        feet = np.broadcast_to(feet, hip.shape)
        ik = leg_ik(self.p, self.t, hip, feet, self.knee_back[leg], clearance=self.limb_clear)
        # Leave reach for the sideways sway: the 3D hip-foot distance grows to sqrt(d^2 + sway^2).
        in_reach = np.hypot(*(feet - hip).T) <= self.reach_2d
        d = ik.knee - hip
        c, s_ = np.cos(P.pitch), np.sin(P.pitch)
        q_hip = np.arctan2(-(c * d[:, 0] + s_ * d[:, 1]), -(-s_ * d[:, 0] + c * d[:, 1]))
        return ik.ok & in_reach & (np.abs(q_hip) <= HIP_LIMIT), ik.knee

    def trunk_clear(self, P: Poses) -> np.ndarray:
        p = self.p
        hl, top, bot = p.body_length / 2, p.body_height / 2, -(p.body_height / 2 + p.skid_thickness)
        pts = [(x, bot) for x in np.linspace(-hl, hl, 15)] + [(x, top) for x in np.linspace(-hl, hl, 7)]
        pts += [(sx * hl, z) for sx in (-1, 1) for z in np.linspace(bot, top, 4)]
        loc = np.array(pts)
        world = P.centre[:, None, :] + loc[None, :, :1] * P.ex[:, None, :] + loc[None, :, 1:] * P.ez[:, None, :]
        ok = ~self.t.solid(world[..., 0], world[..., 1], clearance=0.005).any(axis=1)
        # Hip stacks, seen from the side at zero ab/ad: the hip motor round the hip axis and the
        # knee motor above it. (Rolling with ab/ad lowers their outboard ends; the 3D check on the
        # reference, reference.stack_and_self_clearance, covers that.)
        for front in (True, False):
            hip = P.hip(p, front)
            for c in (hip, hip + P.ez * p.knee_motor_above):
                ok &= ~self.t.solid(c[:, 0], c[:, 1], clearance=p.hip_motor_radius + 0.005)
        return ok

    def neck_geometry(self, P: Poses, opt: int):
        p = self.p
        neck_rel, head_rel = NECK_OPTIONS[opt]
        base = P.centre + P.ex * (p.body_length / 2 - 0.03) + P.ez * (p.body_height / 2)
        a = P.pitch - neck_rel
        tip = base + p.neck_length * np.stack([np.cos(a), np.sin(a)], -1)
        b = a - head_rel
        jaw = tip + p.head_length * np.stack([np.cos(b), np.sin(b)], -1)
        return base, tip, jaw

    def neck_feasible(self, P: Poses) -> np.ndarray:
        """(N, n_options) mask of collision-free neck/head folds per pose."""
        if not self.p.include_neck:
            return np.ones((len(P.pitch), 1), bool)
        t = np.linspace(0, 1, 10)
        cols = []
        for opt in range(len(NECK_OPTIONS)):
            base, tip, jaw = self.neck_geometry(P, opt)
            ok = np.ones(len(P.pitch), bool)
            for a, b, r in ((base, tip, 0.022), (tip, jaw, 0.035)):
                pts = a[:, None] + t[None, :, None] * (b - a)[:, None]
                ok &= ~self.t.solid(pts[..., 0], pts[..., 1], clearance=r).any(axis=1)
            cols.append(ok)
        return np.stack(cols, 1)

    def neck_option(self, P: Poses) -> np.ndarray:
        """Index of the first collision-free neck/head fold per pose, or -1."""
        feas = self.neck_feasible(P)
        return np.where(feas.any(1), feas.argmax(1), -1)

    def com_parts(self, P: Poses, knees: dict, feet: dict):
        """Mass-weighted x sum and total mass of everything except the neck and head."""
        m, p = self.masses, self.p
        num = m["trunk"] * P.centre[:, 0]
        tot = m["trunk"]
        # Same COMs as the MJCF: the hip stack's sits above the hip axis (the knee motor is
        # knee_motor_above up), which on a pitched trunk moves it along the stairs.
        hip_com_up = hip_link_inertia(p)[0][2]
        shin_frac = (p.shin_mass / 2 + p.foot_mass) / (p.shin_mass + p.foot_mass)
        for leg in LEGS:
            hip = P.hip(p, leg[0] == "F")
            k, f = knees[leg], np.broadcast_to(feet[leg], hip.shape)
            num = num + m["hip_link"] * (hip[:, 0] + hip_com_up * P.ez[:, 0]) \
                + m["thigh"] * (hip[:, 0] + k[:, 0]) / 2 \
                + m["shin_with_foot"] * ((1 - shin_frac) * k[:, 0] + shin_frac * f[:, 0])
            tot += m["hip_link"] + m["thigh"] + m["shin_with_foot"]
        return num, tot

    def com_x(self, P: Poses, knees: dict, feet: dict, neck_opt) -> np.ndarray:
        num, tot = self.com_parts(P, knees, feet)
        if not self.p.include_neck:
            return num / tot
        m = self.masses
        opt = np.broadcast_to(np.maximum(neck_opt, 0), P.pitch.shape)
        geo = [self.neck_geometry(P, o) for o in range(len(NECK_OPTIONS))]
        base, tip, jaw = (np.choose(opt[:, None], [g[i] for g in geo]) for i in range(3))
        num = num + m["neck"] * (base[:, 0] + tip[:, 0]) / 2 + m["head"] * (tip[:, 0] + jaw[:, 0]) / 2
        return num / (tot + m["neck"] + m["head"])

    def support_margin(self, com_x: np.ndarray, stance_xy: list[np.ndarray], com_y=0.0) -> np.ndarray:
        """Signed distance from (com_x, com_y) to the edges of the support polygon (positive inside)."""
        pts = np.stack(stance_xy, 1)  # (N, k, 2)
        c = pts.mean(1, keepdims=True)
        order = np.argsort(np.arctan2(pts[..., 1] - c[..., 1], pts[..., 0] - c[..., 0]), axis=1)
        pts = np.take_along_axis(pts, order[..., None], 1)
        nxt = np.roll(pts, -1, axis=1)
        e = nxt - pts
        q = np.stack([com_x, np.broadcast_to(com_y, com_x.shape)], -1)[:, None, :] - pts
        cross = e[..., 0] * q[..., 1] - e[..., 1] * q[..., 0]
        return (cross / np.maximum(np.hypot(e[..., 0], e[..., 1]), 1e-9)).min(axis=1)

    def sway_ik(self, P: Poses, leg: str, foot, sways: np.ndarray) -> dict[str, np.ndarray]:
        """Exact 3D leg IK with the trunk at pose P (pitch only, no roll) swayed sideways by each
        of `sways` (k,), the foot held at `foot` (2D, planner frame). The sway moves the foot by -y
        across the leg plane; the ab/ad roll takes that up and the planar IK reaches the foot in
        the rolled leg plane. Returns (N, k) arrays: joint angles (abad, hip, knee, in the left
        leg's convention), flex, h2 (< 0: unreachable) and the trunk-frame (x, z) offsets from the
        ab/ad axis of the hip-pitch joint, knee and foot as seen from the side."""
        p = self.p
        o, l1, l2 = p.abad_offset, p.thigh_length, p.shin_length
        c, s_ = np.cos(P.pitch), np.sin(P.pitch)
        hip = P.hip(p, leg[0] == "F")
        d = np.broadcast_to(foot, hip.shape) - hip
        px = (c * d[:, 0] + s_ * d[:, 1])[:, None]  # trunk frame: along the body
        pz = (-s_ * d[:, 0] + c * d[:, 1])[:, None]  # trunk frame: up (the foot is below, < 0)
        side = 1 if leg[1] == "L" else -1
        py = o - side * np.asarray(sways, float)[None, :]  # foot from the ab/ad axis, outward
        h2 = py**2 + pz**2 - o**2
        h = -np.sqrt(np.maximum(h2, 0.0))  # foot height in the rolled leg plane
        a = np.arctan2(pz, py) - np.arctan2(h, o)
        a = (a + np.pi) % (2 * np.pi) - np.pi
        dist2 = px**2 + h**2
        cos_k = (dist2 - l1**2 - l2**2) / (2 * l1 * l2)
        flex = np.arccos(np.clip(cos_k, -1.0, 1.0))
        k = -flex if self.knee_back[leg] else flex
        q_hip = np.arctan2(-px, -h) - np.arctan2(l2 * np.sin(k), l1 + l2 * np.cos(k))
        q_hip = (q_hip + np.pi) % (2 * np.pi) - np.pi
        # Side view (the stairs don't vary across their width, so this is all the collision check
        # needs): the hip-pitch joint rides o*sin(a) above the ab/ad axis, and the knee sits at
        # its in-plane height times cos(a) above that.
        dd = np.sqrt(np.maximum(dist2, 1e-12))
        alpha = np.arccos(np.clip((l1**2 + dist2 - l2**2) / (2 * l1 * dd), -1.0, 1.0))
        ang = np.arctan2(h, px) + (-alpha if self.knee_back[leg] else alpha)
        lift = o * np.sin(a)
        return dict(abad=a, hip=q_hip, knee=k, flex=flex, h2=h2, dist=np.sqrt(dist2), cos_k=cos_k,
                    hip_xz=np.stack([np.zeros_like(lift), lift], -1),
                    knee_xz=np.stack([l1 * np.cos(ang), lift + l1 * np.sin(ang) * np.cos(a)], -1),
                    foot_xz=np.stack([np.broadcast_to(px, a.shape), np.broadcast_to(pz, a.shape)], -1),
                    hip_world=hip, ex=np.stack([c, s_], -1), ez=np.stack([-s_, c], -1))

    def sway_leg_ok(self, P: Poses, leg: str, foot, sways: np.ndarray, collide: bool = True) -> np.ndarray:
        """(N, k) mask: can `leg` keep its foot planted with the trunk swayed by each of `sways`
        (sway_ik), within the ab/ad, knee and hip limits? With `collide`, the rolled thigh, knee
        pulley and shin must also clear the stairs: the sway reshapes the leg (a straighter or
        more bent knee), which the zero-sway check in leg_check can't see."""
        p = self.p
        ik = self.sway_ik(P, leg, foot, sways)
        a, flex = ik["abad"], ik["flex"]
        ok = ((ik["h2"] >= 0) & (np.abs(a) <= ABAD_LIMIT) & (np.abs(ik["cos_k"]) <= 1.0)
              & (ik["dist"] <= self.reach_2d) & (flex >= FLEX_LIMITS[0]) & (flex <= FLEX_LIMITS[1])
              & (np.abs(ik["hip"]) <= HIP_LIMIT))
        if not collide:
            return ok
        hip_t, knee_t, foot_t = ik["hip_xz"], ik["knee_xz"], ik["foot_xz"]
        hip, ex, ez = ik["hip_world"], ik["ex"][:, None, None, :], ik["ez"][:, None, None, :]

        def world(v):  # (N, k, n, 2) trunk-frame offsets from the ab/ad axis -> planner frame
            return hip[:, None, None, :] + v[..., :1] * ex + v[..., 1:] * ez

        t = np.linspace(0.0, 1.0, 12)[None, None, :, None]
        thigh = world(hip_t[:, :, None] + t * (knee_t - hip_t)[:, :, None])
        shin_len = np.maximum(np.linalg.norm(foot_t - knee_t, axis=-1, keepdims=True), 1e-9)
        tt = t * np.maximum(1.0 - p.foot_radius * 1.5 / shin_len, 0.0)[:, :, None]
        shin = world(knee_t[:, :, None] + tt * (foot_t - knee_t)[:, :, None])
        c = self.limb_clear
        clear = ~self.t.solid(thigh[..., 0], thigh[..., 1], clearance=p.thigh_radius + c).any(-1)
        clear &= ~self.t.solid(shin[..., 0], shin[..., 1], clearance=p.shin_radius + c).any(-1)
        kw = world(knee_t[:, :, None])[:, :, 0]
        clear &= ~self.t.solid(kw[..., 0], kw[..., 1], clearance=p.knee_pulley_radius + c)
        return ok & clear

    def sway_feasible(self, P: Poses, feet: dict) -> np.ndarray:
        """(N, n_sways) mask: every planted leg can absorb each of `self.sways` (sway_leg_ok)."""
        ok = np.ones((len(P.pitch), len(self.sways)), bool)
        for leg in LEGS:
            ok &= self.sway_leg_ok(P, leg, feet[leg], self.sways)
        return ok

    def foot_xy(self, leg: str, foot_x) -> np.ndarray:
        y = self.wy if leg[1] == "L" else -self.wy
        x = np.atleast_1d(foot_x)
        return np.stack([x, np.full_like(x, y)], -1)

    def evaluate(self, P: Poses, feet: dict, lifted: str | None = None, neck: int | None = None):
        """Check poses with planted feet; `lifted` leg is ignored for reach and support. With
        `neck`, only that NECK_OPTIONS fold is considered (otherwise the best one is chosen).

        Returns (ok (N,), info dict)."""
        ok = self.trunk_clear(P)
        if self.max_pitch is not None:
            ok &= np.abs(P.pitch) <= self.max_pitch
        feas = self.neck_feasible(P)
        ok &= feas.any(axis=1)
        knees = {}
        for leg in LEGS:
            leg_ok, knees[leg] = self.leg_check(P, leg, feet[leg])
            if leg != lifted:
                ok &= leg_ok
        all_xy = {leg: self.foot_xy(leg, np.broadcast_to(feet[leg], P.centre.shape)[:, 0]) for leg in LEGS}
        four = [all_xy[leg] for leg in LEGS]
        three = [all_xy[leg] for leg in LEGS if leg != lifted]
        n = len(P.pitch)
        opts = range(feas.shape[1]) if neck is None else (neck,)
        sway_ok = self.sway_feasible(P, feet) if lifted is not None else None
        marg, sway = np.full(n, -np.inf), np.zeros(n)
        neck, cx = np.full(n, -1), np.zeros(n)
        # The neck and head carry ~10% of the mass: fold them to whichever collision-free
        # pose gives the best stability margin (head back when going downhill).
        for opt in opts:
            c = self.com_x(P, knees, feet, opt)
            if lifted is None:
                m_o, y_o = self.support_margin(c, four), np.zeros(n)
            else:
                # Sway the body sideways (all four feet down, so any sway between is also
                # stable) to the y that best supports lifting this leg.
                m_o, y_o = np.full(n, -np.inf), np.zeros(n)
                for k, y in enumerate(self.sways):
                    m = np.minimum(self.support_margin(c, three, y), self.support_margin(c, four, y))
                    m = np.where(sway_ok[:, k], m, -np.inf)
                    m_o, y_o = np.maximum(m, m_o), np.where(m > m_o, y, y_o)
            m_o = np.where(feas[:, opt], m_o, -np.inf)
            better = m_o > marg
            marg, sway = np.where(better, m_o, marg), np.where(better, y_o, sway)
            neck, cx = np.where(better, opt, neck), np.where(better, c, cx)
        return ok & (marg >= self.margin), dict(knees=knees, neck=neck, com_x=cx, margin=marg, sway=sway)

    # -------------------------------------------------- search

    def diagnose(self, state) -> dict[str, dict[str, int]]:
        """Why can't each leg move from `state`? Counts of candidate end poses passing each check."""
        x, z, beta, fi, ri, feet_idx = state
        feet = self.feet_of(feet_idx)
        nh = len(self.hs)
        ends = [self.target(x + ds, self.hs[f], self.hs[r]) for ds in (0.10, 0.07, 0.04, 0.02, 0.0, -0.02, -0.04)
                for f in range(nh) for r in range(nh)]
        P_end = self.pose(*np.array(ends).T)
        ok4, info4 = self.evaluate(P_end, feet)
        trunk = self.trunk_clear(P_end)
        neck = self.neck_option(P_end) >= 0
        out = {}
        for leg_i, leg in enumerate(LEGS):
            ok_lift, _ = self.evaluate(P_end, feet, lifted=leg)
            both = np.flatnonzero(ok4 & ok_lift)
            n_path = sum(self.path_ok((x, z, beta), ends[i], feet) for i in both[:30])
            n_swing = sum(bool(self.swing_candidates(ends[i], feet_idx, leg_i)) for i in both[:30])
            out[leg] = dict(ends=len(ends), trunk_clear=int(trunk.sum()), neck_clear=int(neck.sum()),
                            four_feet_ok=int(ok4.sum()), can_lift=int((ok4 & ok_lift).sum()),
                            path_ok=int(n_path), has_foothold=int(n_swing))
        return out

    def feet_of(self, idx: tuple[int, ...]) -> dict:
        return {leg: self.cand[i] for leg, i in zip(LEGS, idx)}

    def same_side_ok(self, feet: dict) -> bool:
        return all(feet[f"F{side}"][0] - feet[f"R{side}"][0] >= 0.06 for side in "LR")

    def initial_state(self):
        """State tuple: (x, z, pitch, front height index, rear height index, feet indices)."""
        p = self.p
        allowed = np.isin(self.cand_sid, self.sc.start_sids)
        for ds in (0.0, -0.03, 0.03, -0.06, 0.06, -0.1, 0.1):
            P = self.body(self.sc.s0 + ds, 0.7 * p.leg_reach)  # envelope pose, only to pick feet
            idx = []
            for leg in LEGS:
                hip = P.hip(p, leg[0] == "F")[0]
                ok, _ = self.leg_check(Poses(np.repeat(P.centre, len(self.cand), 0),
                                             np.repeat(P.pitch, len(self.cand))), leg, self.cand)
                ok &= allowed
                if not ok.any():
                    break
                d = np.hypot(*(self.cand - hip).T)
                score = np.abs(d - 0.75 * p.leg_reach) + 0.5 * np.abs(self.cand[:, 0] - hip[0])
                idx.append(int(np.argmin(np.where(ok, score, np.inf))))
            if len(idx) < 4:
                continue
            feet = self.feet_of(tuple(idx))
            if not self.same_side_ok(feet):
                continue
            mid = 0.7 * p.leg_reach
            for hi in sorted(range(len(self.hs)), key=lambda j: abs(self.hs[j] - mid)):
                pose = self.target(self.sc.s0 + ds, self.hs[hi], self.hs[hi])
                good, _ = self.evaluate(self.pose(*pose), feet)
                if good[0]:
                    return (*pose, hi, hi, tuple(idx))
        return None

    def path_ok(self, pose0, pose1, feet) -> bool:
        """All four feet planted while the trunk moves linearly from pose0 to pose1."""
        dist = max(abs(pose1[0] - pose0[0]), abs(pose1[1] - pose0[1]), 0.15 * abs(pose1[2] - pose0[2]))
        n = int(max(2, math.ceil(dist / 0.01) + 1))
        t = np.linspace(0, 1, n)[1:-1]
        if not len(t):
            return True
        P = self.pose(*(a + t * (b - a) for a, b in zip(pose0, pose1)))
        return bool(self.evaluate(P, feet)[0].all())

    # -------------------------------------------------- swing space

    SWING_RES = 0.01

    def swing_space(self, pose, leg: str):
        """Foot positions (1 cm grid) where the whole leg is valid with the body at `pose`:
        reachable within joint limits, thigh and shin clear, and the foot sphere clear of the
        terrain. Returns (gx, gz, free mask (nx, nz), component labels). Cached."""
        key = (pose, leg, self.limb_clear)
        if key in self._swing_cache:
            return self._swing_cache[key]
        from scipy import ndimage

        P = self.pose(*pose)
        hip = P.hip(self.p, leg[0] == "F")[0]
        r = self.p.leg_reach
        gx = np.arange(hip[0] - r, hip[0] + r + 1e-9, self.SWING_RES)
        gz = np.arange(hip[1] - r, hip[1] + 0.12, self.SWING_RES)
        X, Z = np.meshgrid(gx, gz, indexing="ij")
        pts = np.stack([X.ravel(), Z.ravel()], 1)
        n = len(pts)
        ok, _ = self.leg_check(Poses(np.repeat(P.centre, n, 0), np.repeat(P.pitch, n)), leg, pts)
        # The swinging foot keeps 5 mm clear of every surface; it leaves and reaches its
        # footholds (which touch the surface) through the nearest free cell, ~1 cm away.
        ok &= ~self.t.solid(pts[:, 0], pts[:, 1], clearance=self.p.foot_radius + 0.005)
        free = ok.reshape(X.shape)
        labels, _ = ndimage.label(free)
        out = (gx, gz, free, labels)
        if len(self._swing_cache) > 4000:
            self._swing_cache.clear()
        self._swing_cache[key] = out
        return out

    def _cell(self, gx, gz, pt):
        i = int(np.clip(round((pt[0] - gx[0]) / self.SWING_RES), 0, len(gx) - 1))
        j = int(np.clip(round((pt[1] - gz[0]) / self.SWING_RES), 0, len(gz) - 1))
        return i, j

    def _component(self, gx, gz, free, labels, pt) -> int:
        """Label of the free component containing pt (or the nearest free cell within 2 cm)."""
        i, j = self._cell(gx, gz, pt)
        best, best_d = 0, 9
        for di in range(-2, 3):
            for dj in range(-2, 3):
                a, b = i + di, j + dj
                if 0 <= a < len(gx) and 0 <= b < len(gz) and free[a, b] and di * di + dj * dj < best_d:
                    best, best_d = labels[a, b], di * di + dj * dj
        return best

    def swing_path(self, pose, leg: str, f0, f1) -> np.ndarray | None:
        """Shortest 8-connected foot path through the leg's free space from f0 to f1 (2D points),
        or None if they are not connected."""
        import heapq

        gx, gz, free, labels = self.swing_space(pose, leg)
        c0, c1 = self._component(gx, gz, free, labels, f0), self._component(gx, gz, free, labels, f1)
        if not c0 or c0 != c1:
            return None
        start = min(((self._cell(gx, gz, f0)[0] + di, self._cell(gx, gz, f0)[1] + dj)
                     for di in range(-2, 3) for dj in range(-2, 3)),
                    key=lambda c: (not (0 <= c[0] < len(gx) and 0 <= c[1] < len(gz) and labels[c] == c0),
                                   abs(c[0] - self._cell(gx, gz, f0)[0]) + abs(c[1] - self._cell(gx, gz, f0)[1])))
        goal_cells = {(a, b) for a in range(len(gx)) for b in range(len(gz))
                      if labels[a, b] == c0 and abs(gx[a] - f1[0]) <= 0.021 and abs(gz[b] - f1[1]) <= 0.021}
        dist, prev = {start: 0.0}, {}
        heap = [(0.0, start)]
        end = None
        while heap:
            d, c = heapq.heappop(heap)
            if c in goal_cells:
                end = c
                break
            if d > dist.get(c, np.inf):
                continue
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    nb = (c[0] + di, c[1] + dj)
                    if (di or dj) and 0 <= nb[0] < len(gx) and 0 <= nb[1] < len(gz) and free[nb]:
                        nd = d + math.hypot(di, dj)
                        if nd < dist.get(nb, np.inf):
                            dist[nb], prev[nb] = nd, c
                            heapq.heappush(heap, (nd, nb))
        if end is None:
            return None
        cells = [end]
        while cells[-1] != start:
            cells.append(prev[cells[-1]])
        path = np.array([[gx[a], gz[b]] for a, b in reversed(cells)])
        return np.vstack([np.asarray(f0)[None], path, np.asarray(f1)[None]])

    def swing_candidates(self, pose, feet_idx, leg_i, sway: float | None = None) -> list[int]:
        """Up to 3 new footholds for leg_i from `pose`; with `sway`, only footholds the leg reaches
        in 3D with the trunk swayed that far (it holds the sway through the swing). With
        `self.footholds` (leg index -> candidate indices), only those."""
        leg = LEGS[leg_i]
        cur = self.cand[feet_idx[leg_i]]
        P = self.pose(*pose)
        idx = np.flatnonzero(self.cand[:, 0] > cur[0] + 0.02)
        if self.footholds is not None:
            idx = idx[np.isin(idx, list(self.footholds[leg_i]))]
        if not len(idx):
            return []
        PP = Poses(np.repeat(P.centre, len(idx), 0), np.repeat(P.pitch, len(idx)))
        ok, _ = self.leg_check(PP, leg, self.cand[idx])
        idx = idx[ok]
        if sway is not None and len(idx):
            idx = idx[self.sway_leg_ok(PP[:len(idx)], leg, self.cand[idx], np.array([sway]))[:, 0]]
        partner = ("R" if leg[0] == "F" else "F") + leg[1]
        px = self.cand[feet_idx[LEGS.index(partner)]][0]
        if leg[0] == "F":
            idx = idx[self.cand[idx, 0] - px >= 0.06]
        else:
            idx = idx[px - self.cand[idx, 0] >= 0.06]
        # The foot must be able to get there: same free component of the leg's swing space.
        if len(idx):
            gx, gz, free, labels = self.swing_space(pose, leg)
            c0 = self._component(gx, gz, free, labels, cur)
            idx = idx[[c0 != 0 and self._component(gx, gz, free, labels, self.cand[i]) == c0 for i in idx]]
        hip = P.hip(self.p, leg[0] == "F")[0]
        comfy = np.hypot(*(self.cand[idx] - hip).T) <= 0.9 * self.p.leg_reach
        order = np.lexsort((-self.cand[idx, 0], ~comfy))  # comfortable first, then farthest
        return [int(i) for i in idx[order][:3]]

    def _tier(self, m):
        """Margin graded in 5 mm steps, capped at `prefer` (larger is better)."""
        return np.floor(np.minimum(m, self.prefer) / 0.005 + 1e-9)

    def move_margins(self, feet_idx, child, neck: int | None = None) -> list[float]:
        """The four static margins of move_margin, with the best fold or the given `neck`."""
        P = self.pose(*child[:3])
        old, new = self.feet_of(feet_idx), self.feet_of(child[5])
        leg = next((l for l, a, b in zip(LEGS, feet_idx, child[5]) if a != b), None)
        ms = [self.evaluate(P, old, neck=neck)[1]["margin"][0], self.evaluate(P, new, neck=neck)[1]["margin"][0]]
        if leg:
            ms += [self.evaluate(P, old, lifted=leg, neck=neck)[1]["margin"][0],
                   self.evaluate(P, new, lifted=leg, neck=neck)[1]["margin"][0]]
        return ms

    def move_margin(self, feet_idx, child) -> float:
        """Worst static margin over one move: the shift to the child's pose (all four feet on the
        old footholds), the swing leg lifting there, the same leg about to touch down on its new
        foothold (its mass has moved forward, toward the edge of the other three feet: up to
        0.57 cm on the N+K robot), and the four feet down after the swing. Lift and touchdown use
        the best sway for each, which the reference holds within ~1 mm."""
        return float(min(self.move_margins(feet_idx, child)))

    def move_ok(self, feet_idx, child) -> bool:
        """The four states of move_margin all pass evaluate: margin floor, reach, limb clearance,
        trunk and neck clear, and a feasible sway for the lift and the touchdown."""
        P = self.pose(*child[:3])
        old, new = self.feet_of(feet_idx), self.feet_of(child[5])
        leg = next((l for l, a, b in zip(LEGS, feet_idx, child[5]) if a != b), None)
        checks = [(old, None), (new, None)] + ([(old, leg), (new, leg)] if leg else [])
        return all(bool(self.evaluate(P, f, lifted=l)[0][0]) for f, l in checks)

    def smooth_necks(self, path: list, slack: float = 0.0025) -> list[int]:
        """One NECK_OPTIONS fold per path state (index 0: the start pose), held from that move's
        shift through its swing, chosen for the least total neck motion over the plan (Viterbi,
        cost = largest joint change, rad). Per move, a fold is allowed if its worst margin
        (move_margins) clears the floor and comes within `slack` of the best fold's, capped at
        `prefer`. Picking the best fold per pose instead switched the neck by 0.5-2.35 rad on most
        moves (run 6), and the policy failed on one of those swings."""
        n_opt = len(NECK_OPTIONS)
        _, start = self.evaluate(self.pose(*path[0][:3]), self.feet_of(path[0][5]))
        allowed = [{int(start["neck"][0]): float(start["margin"][0])}]
        cap = np.inf if self.prefer is None else self.prefer
        for a, b in zip(path[:-1], path[1:]):
            m = np.array([min(self.move_margins(a[5], b, neck=o)) for o in range(n_opt)])
            best = m.max()
            ok = (m >= self.margin) & (m >= min(best, cap) - slack)
            allowed.append({o: float(m[o]) for o in (np.flatnonzero(ok) if ok.any() else [int(m.argmax())])})
        vec = np.array(NECK_OPTIONS)
        step = lambda i, j: float(np.abs(vec[i] - vec[j]).max())
        # cost to reach each allowed fold; ties go to the fold with more margin
        cost = [{o: -1e-3 * m for o, m in allowed[0].items()}]
        back = [{}]
        for k in range(1, len(allowed)):
            ck, bk = {}, {}
            for o, m in allowed[k].items():
                prev = min(cost[k - 1], key=lambda p: cost[k - 1][p] + step(p, o))
                ck[o], bk[o] = cost[k - 1][prev] + step(prev, o) - 1e-3 * m, prev
            cost.append(ck)
            back.append(bk)
        o = min(cost[-1], key=cost[-1].get)
        out = [int(o)]
        for k in range(len(allowed) - 1, 0, -1):
            o = back[k][o]
            out.append(int(o))
        return out[::-1]

    def repair(self, path: list, floors=(0.02, 0.018, 0.0165), limb_clear: float = 0.0,
               windows=((3, 4), (6, 8)), time_limit: float = 60.0, progress=None) -> tuple[list, list[dict]]:
        """Re-plan a finished plan locally around its weak moves, without re-planning the flight.

        A whole flight at a 2 cm floor doesn't plan within an hour (N+K robot, 2026-10-20), but
        only a dozen moves per flight sit near the 1.5 cm floor. A move is weak if its move_margin
        is below floors[0], or (with `limb_clear`) if it fails move_ok with that much extra gap
        between the leg links and the stairs. For each weak move: search again from `back` states
        earlier, on the same footholds, with every move held to a raised floor and the extra gap
        (strict), until the feet match an original state at most `ahead` states past the weak move
        and the trunk can shift from there to the next original pose; then splice. Floors go from the highest down to just
        above the move's own margin (a move weak only on clearance keeps floors[0]); windows from
        the smallest. Without a plan that keeps the gap, the gap is dropped. Returns (new path, one
        log entry per weak move); `progress(entry)` is called after each."""
        import copy

        path, log = list(path), []
        prog = lambda st: st[0] + self.cand[list(st[5]), 0].mean()
        same = lambda a, b: a[3:] == b[3:] and np.allclose(a[:3], b[:3], atol=1e-6)
        chk = copy.copy(self)
        chk.margin, chk.limb_clear = -np.inf, limb_clear
        k = 1
        while k < len(path):
            m0 = self.move_margin(path[k - 1][5], path[k])
            clear0 = not limb_clear or chk.move_ok(path[k - 1][5], path[k])
            if m0 >= floors[0] and clear0:
                k += 1
                continue
            leg = next(l for l, a, b in zip(LEGS, path[k - 1][5], path[k][5]) if a != b)
            entry = dict(state=k, leg=leg, x=float(path[k][0]), margin=m0, clear=clear0, floor=None,
                         limb_clear=None, replaced=0, added=0)
            t0 = time.time()
            tries = [f for f in floors if f > m0 + 5e-4] or [floors[0]]
            attempts = [(f, c, w) for c in dict.fromkeys((limb_clear, 0.0)) for f in tries for w in windows]
            if m0 >= floors[0]:  # weak on clearance only: nothing to gain without the gap
                attempts = [a for a in attempts if a[1] == limb_clear]
            for floor, gap, (back, ahead) in attempts:
                i0, last = max(0, k - 1 - back), len(path) - 1
                targets = range(k, min(k + ahead, last) + 1)
                hit = {}

                def goal(st):
                    for j in targets:
                        if tuple(st[5]) != tuple(path[j][5]):
                            continue
                        # rejoin: the next original shift (4 feet down, base floor) must still work;
                        # the last state is where the flight hands over, so it must match exactly
                        if same(st, path[j]) or (j < last and self.path_ok(st[:3], path[j + 1][:3],
                                                                            self.feet_of(st[5]))):
                            hit["j"] = j
                            return True
                    return False

                local = copy.copy(self)
                local.margin, local.limb_clear, local.expanded, local.budget = floor, gap, 0, 10**6
                # Keep the original footholds: the search re-chooses trunk poses, shift sizes, sway,
                # neck and leg order. Free footholds rarely land back on the original set, which
                # the rejoin needs (tried: it wanders off at any floor above the original's).
                local.footholds = {i: {path[t][5][i] for t in range(i0, targets[-1] + 1)} for i in range(4)}
                lim = prog(path[targets[-1]]) + 0.02
                sub, _ = local.plan(time_limit=time_limit, goal=goal, start=path[i0],
                                    prune=lambda st: prog(st) > lim, strict=True)
                if sub is None:
                    continue
                j = hit["j"]
                path = path[:i0] + sub + path[j + 1:]
                entry.update(floor=floor, limb_clear=gap, replaced=j - i0, added=len(sub) - 1, back=back,
                             new_margin=min(self.move_margin(a[5], b) for a, b in zip(sub[:-1], sub[1:])))
                k = i0 + len(sub)
                break
            entry["seconds"] = round(time.time() - t0, 1)
            log.append(entry)
            if progress:
                progress(entry)
            if entry["floor"] is None:
                k += 1
        return path, log

    def plan(self, time_limit: float = 60.0, goal=None, start=None, prune=None, strict: bool = False):
        """Depth-first search. Returns (list of states, status string).

        `goal(state) -> bool` overrides the scenario's goal, `start` the initial state, and
        `prune(state) -> bool` drops states the search shouldn't expand. With `strict`, every move's
        four margins (move_margin, touchdown included) must clear the floor, not only lift-off and
        stance. On failure, `self.best_path` holds the path that got furthest, for diagnosis.
        """
        start = self.initial_state() if start is None else start
        self.best_path = None
        if start is None:
            return None, "no valid start pose"
        goal = goal or (lambda st: self.sc.goal(self.cand_sid[list(st[5])]))
        best_progress = -np.inf
        visited = set()
        t0 = time.time()
        deltas = (0.10, 0.07, 0.04, 0.02, 0.0, -0.02, -0.04)
        nh = len(self.hs)
        stack = [(start, [start])]
        while stack:
            if self.expanded >= self.budget or time.time() - t0 > time_limit:
                return None, "search budget exhausted"
            state, path = stack.pop()
            x, z, beta, fi, ri, feet_idx = state
            if goal(state):
                return path, "ok"
            progress = x + self.cand[list(feet_idx), 0].mean()
            if progress > best_progress:
                best_progress, self.best_path = progress, path
            key = (x, z, beta, feet_idx)
            if key in visited or (prune is not None and prune(state)):
                continue
            visited.add(key)
            self.expanded += 1
            feet = self.feet_of(feet_idx)

            # Candidate end poses: shift size x (front, rear) hip heights, smallest height change first.
            combos = sorted(((f, r) for f in range(nh) for r in range(nh)),
                            key=lambda c: abs(c[0] - fi) + abs(c[1] - ri))
            ends = [(ds, f, r, self.target(x + ds, self.hs[f], self.hs[r])) for ds in deltas for f, r in combos]
            P_end = self.pose(*np.array([e[3] for e in ends]).T)
            ok4, info4 = self.evaluate(P_end, feet)

            children = []
            # Free gait: any leg may swing next. Rear-most legs first, so ties favour a crawl.
            for leg_i in sorted(range(4), key=lambda i: self.cand[feet_idx[i]][0]):
                ok_lift, info_lift = self.evaluate(P_end, feet, lifted=LEGS[leg_i])
                found = 0
                order = np.flatnonzero(ok4 & ok_lift)  # largest shift first
                if self.prefer is not None:  # then, ahead of that, the most margin (stable sort)
                    pre = np.minimum(info4["margin"][order], info_lift["margin"][order])
                    order = order[np.argsort(-self._tier(pre), kind="stable")]
                for e_i in order:
                    ds, f, r, pose1 = ends[e_i]
                    if not self.path_ok((x, z, beta), pose1, feet):
                        continue
                    cands = self.swing_candidates(pose1, feet_idx, leg_i, sway=float(info_lift["sway"][e_i]))[:2]
                    for c in cands:
                        new_idx = list(feet_idx)
                        new_idx[leg_i] = c
                        children.append((*pose1, f, r, tuple(new_idx)))
                    found += bool(cands)
                    if found >= 2:
                        break

            # One-step lookahead: prefer moves after which some leg can lift without moving the
            # body, then the most progress (body plus feet).
            def rank(c):
                P = self.pose(*c[:3])
                cf = self.feet_of(c[5])
                can_lift = any(self.evaluate(P, cf, lifted=l)[0][0] for l in LEGS)
                tier = 0.0 if self.prefer is None else -self._tier(self.move_margin(feet_idx, c))
                return (not can_lift, tier, -(c[0] + self.cand[list(c[5]), 0].mean()))

            if strict:
                children = [c for c in children if self.move_ok(feet_idx, c)]
            children = sorted(children, key=rank)[:8]
            for c in reversed(children):  # best first off the stack
                stack.append((c, path + [c]))
        return None, "no plan within discretisation"


# ---------------------------------------------------------------- plan post-processing

@dataclass
class Frame:
    """One keyframe of a plan, with everything needed to draw it or seed Crocoddyl."""

    kind: str  # "start" | "shift" | "swing"
    s: float
    h: float
    feet: dict
    swing_leg: str | None
    swing_from: np.ndarray | None = None
    centre: np.ndarray = field(default_factory=lambda: np.zeros(2))
    pitch: float = 0.0
    knees: dict = field(default_factory=dict)
    neck: int = 0
    com_x: float = 0.0
    margin: float = 0.0
    torques: dict = field(default_factory=dict)  # leg -> (hip, knee) Nm, quasi-static


def foot_forces(com_x: float, stance: dict[str, np.ndarray], wy: float, weight: float) -> dict[str, float]:
    """Vertical forces at the stance feet balancing the weight (non-negative, least norm)."""
    legs = list(stance)
    A = np.array([[1.0] * len(legs), [stance[l][0] for l in legs], [wy if l[1] == "L" else -wy for l in legs]])
    b = np.array([weight, weight * com_x, 0.0])
    reg = 1e-3
    A_aug = np.vstack([A, reg * np.eye(len(legs))])
    b_aug = np.concatenate([b, np.zeros(len(legs))])
    sol = lsq_linear(A_aug, b_aug, bounds=(0, np.inf))
    return dict(zip(legs, sol.x))


def expand_plan(pl: Planner, path: list, necks: list[int] | None = None) -> list[Frame]:
    """Turn search states into keyframes: each step is a body shift (worst case: the next
    leg about to lift) followed by a swing. With `necks` (Planner.smooth_necks), each state's
    shift and swing use that fold instead of the best one per keyframe."""
    frames = []
    weight = pl.p.total_mass_target * G
    prev = None
    for i, st in enumerate(path):
        x, z, beta, fi, ri, feet_idx = st
        feet = pl.feet_of(feet_idx)
        if prev is None:
            steps = [("start", feet, None, None)]
        else:
            moved = [l for l, a, b in zip(LEGS, prev[5], feet_idx) if a != b]
            old = pl.feet_of(prev[5])
            leg = moved[0] if moved else None
            steps = [("shift", old, leg, None)]
            if leg:
                steps.append(("swing", feet, None, old[leg]))
        for kind, use_feet, lifted, swing_from in steps:
            P = pl.pose(x, z, beta)
            ok, info = pl.evaluate(P, use_feet, lifted=lifted, neck=None if necks is None else necks[i])
            stance = {l: use_feet[l] for l in LEGS if l != lifted}
            forces = foot_forces(float(info["com_x"][0]), stance, pl.wy, weight)
            torques = {}
            for l, f in forces.items():
                hip = P.hip(pl.p, l[0] == "F")[0]
                knee = info["knees"][l][0]
                torques[l] = (abs((use_feet[l][0] - hip[0]) * f), abs((use_feet[l][0] - knee[0]) * f))
            swing_leg = lifted if kind == "shift" else (steps[0][2] if kind == "swing" else None)
            frames.append(Frame(kind, x, pl.hs[fi], dict(use_feet), swing_leg, swing_from, P.centre[0], beta,
                                {l: v[0] for l, v in info["knees"].items()}, int(info["neck"][0]),
                                float(info["com_x"][0]), float(info["margin"][0]), torques))
            frames[-1].sway = float(info["sway"][0])
        prev = st
    return frames


@dataclass
class PlanResult:
    scenario: str
    status: str
    steps: int
    expanded: int
    seconds: float
    min_margin: float
    peak_hip: float
    peak_knee: float
    frames: list[Frame] | None = None


def plan_scenario(p: RobotParams, sc: Scenario, keep_frames: bool = False, **kw) -> PlanResult:
    t0 = time.time()
    pl = Planner(p, sc, **kw)
    path, status = pl.plan()
    if path is None:
        return PlanResult(sc.name, status, 0, pl.expanded, time.time() - t0, math.nan, math.nan, math.nan)
    frames = expand_plan(pl, path)
    torques = [t for f in frames for t in f.torques.values()]
    return PlanResult(sc.name, status, len(path) - 1, pl.expanded, time.time() - t0,
                      min(f.margin for f in frames), max(t[0] for t in torques), max(t[1] for t in torques),
                      frames if keep_frames else None)


# ---------------------------------------------------------------- landing turn (top view)

def turn_footprint(p: RobotParams, neck_option: int = 3) -> dict:
    """Top-view extents of the robot standing level with the neck folded (NECK_OPTIONS index).

    Returns the corner points of the body, neck and head, and feet (x forward, y left, relative
    to the trunk centre) and the radius swept by turning in place about the trunk centre.
    """
    from .robot import hip_stack_outlines, trunk_outline

    hl = p.body_length / 2
    pts = [tuple(c) for c in trunk_outline(p)] + [tuple(c) for o in hip_stack_outlines(p) for c in o]
    wy = p.hip_width / 2 + p.abad_offset
    feet = [(sx * p.hip_spacing / 2, sy * (wy + p.foot_radius)) for sx in (-1, 1) for sy in (-1, 1)]
    neck_x = []
    if p.include_neck:
        neck_rel, head_rel = NECK_OPTIONS[neck_option]
        base = hl - 0.03
        tip = base + p.neck_length * math.cos(-neck_rel)
        jaw = tip + p.head_length * math.cos(-neck_rel - head_rel)
        neck_x = [tip, jaw]
        pts += [(x, sy * 0.04) for x in (tip, jaw) for sy in (-1, 1)]
    pts += feet
    radius = max(math.hypot(x, y) for x, y in pts)
    return dict(points=np.array(pts), feet=np.array(feet), radius=radius,
                front=max(x for x, _ in pts), rear=min(x for x, _ in pts), neck_x=neck_x)
