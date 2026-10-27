"""Whole-round-trip kinematic reference for the RL tracking policy (M0 step 3).

A round trip is: floor -> lower flight -> landing (turn 180 degrees and step
across to the upper flight) -> upper flight -> upper floor, then the same
motion played backwards. Playing it backwards is a rear-first descent: the
plans are quasi-static, so the reverse of a valid plan is valid (see
transitions.py).

The pieces:

- **Flights:** the step-1 planner, run from floor to top of each whole flight
  (not just the transitions), turned into timed whole-body poses with the
  phases of the Crocoddyl step (dynamics.plan_phases), with each shift
  slowed to `SHIFT_RATES`.
- **Landing:** a quasi-static crawl on flat ground in 3D, generated here. The
  trunk follows straight segments and turns in place with the neck folded.
  Before each swing the trunk shifts (sways) so the centre of mass is inside
  the other three feet's support triangle by `MARGIN`. Each foot steps to
  its nominal spot under the hip, placed about two steps ahead.

Everything is in the MuJoCo world frame at the policy rate, with per-frame
contact flags. A reference is kinematic: it says where the robot should be,
not which forces get it there. The RL policy finds those.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass

import numpy as np
import pinocchio as pin

from .params import RobotParams, StairParams
from .robot import LEGS, stand_height
from .transitions import NECK_OPTIONS, Planner, expand_plan, make_scenario

MARGIN = 0.025  # landing crawl: COM inside the support triangle by this much, m
NECK_FOLDED = 3  # NECK_OPTIONS index used on the landing (the 46 cm turning circle)


@dataclass
class Reference:
    dt: float
    qpos: np.ndarray  # (N, nq) MuJoCo world qpos
    contact: np.ndarray  # (N, 4) bool, LEGS order: foot planted
    segment: np.ndarray  # (N,) int index into segment_names
    segment_names: list[str]

    @property
    def duration(self) -> float:
        return len(self.qpos) * self.dt

    def reversed(self, rename=lambda n: n) -> "Reference":
        seg = self.segment[::-1]
        order = list(dict.fromkeys(seg.tolist()))  # segment ids in their new time order
        remap = np.empty(len(self.segment_names), int)
        remap[order] = np.arange(len(order))
        return Reference(self.dt, self.qpos[::-1].copy(), self.contact[::-1].copy(), remap[seg],
                         [rename(self.segment_names[i]) for i in order])

    def save(self, path) -> None:
        np.savez_compressed(path, dt=self.dt, qpos=self.qpos, contact=self.contact, segment=self.segment,
                            segment_names=np.array(self.segment_names))

    @classmethod
    def load(cls, path) -> "Reference":
        d = np.load(path)
        return cls(float(d["dt"]), d["qpos"], d["contact"], d["segment"], [str(n) for n in d["segment_names"]])


def concat(parts: list[Reference]) -> Reference:
    names: list[str] = []
    segs = []
    for r in parts:
        remap = []
        for n in r.segment_names:
            if n not in names:
                names.append(n)
            remap.append(names.index(n))
        segs.append(np.array(remap)[r.segment])
    return Reference(parts[0].dt, np.concatenate([r.qpos for r in parts]), np.concatenate([r.contact for r in parts]),
                     np.concatenate(segs), names)


def hold(q: np.ndarray, contact, seconds: float, dt: float, name: str) -> Reference:
    n = int(round(seconds / dt))
    return Reference(dt, np.repeat(q[None], n, 0), np.repeat(np.asarray(contact, bool)[None], n, 0),
                     np.zeros(n, int), [name])


# ---------------------------------------------------------------- frame conversions

def pin_to_mj(q: np.ndarray) -> np.ndarray:
    """Pinocchio (quat x, y, z, w) to MuJoCo (quat w, x, y, z) for a free-floating robot."""
    return np.concatenate([q[:3], [q[6], q[3], q[4], q[5]], q[7:]])


def mj_to_pin(q: np.ndarray) -> np.ndarray:
    return np.concatenate([q[:3], [q[4], q[5], q[6], q[3]], q[7:]])


def smooth(t):
    return 0.5 - 0.5 * np.cos(np.pi * np.clip(t, 0.0, 1.0))


def foot_positions(model: pin.Model, data: pin.Data, q: np.ndarray) -> dict[str, np.ndarray]:
    pin.framesForwardKinematics(model, data, q)
    return {leg: data.oMf[model.getFrameId(f"{leg}_foot")].translation.copy() for leg in LEGS}


# ---------------------------------------------------------------- 3D clearance

SELF_CLEARANCE = 0.003  # leg parts to the trunk, everywhere in the reference (the CAD study's 3 mm)
STACK_CLEARANCE = 0.0  # hip stacks to the stairs


class Clearance:
    """MuJoCo distance queries on whole-body poses, in the full stairwell, for what the sagittal
    planner can't see: ab/ad roll brings the legs toward the trunk and lowers the outboard end of
    each hip stack. Per pose (mm, negative = penetration, capped at MARGIN):

    - `self`: any leg or hip-stack part to the trunk, skid plate, neck or head (exact, every pair);
    - `stack`: hip stacks (motors, belt) to the stairs, walls and floor;
    - `limb`: thigh, hub, knee pulley and shin to the stairs (feet excluded: they stand there).
    """

    MARGIN = 0.05

    def __init__(self, p: RobotParams, s: StairParams):
        import xml.etree.ElementTree as ET

        import mujoco

        from .scene import build_scene_xml

        # The margin must be in the XML: the compiler sizes the broadphase bounds with it, and a
        # margin set on the compiled model is missed by the box and cylinder colliders.
        root = ET.fromstring(build_scene_xml(p, s))
        for g in root.find("worldbody").find("body[@name='trunk']").iter("geom"):
            name = g.get("name", "")
            if name[:2] in LEGS and name[2] == "_":
                g.set("margin", f"{self.MARGIN:g}")
        self.mj = mujoco
        self.m = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        self.d = mujoco.MjData(self.m)
        m = self.m
        trunk = m.body("trunk").id
        robot = {g for g in range(m.ngeom) if m.body_rootid[m.geom_bodyid[g]] == trunk}
        self.kind = np.zeros(m.ngeom, int)  # 0 world, 1 trunk class, 2 stack, 3 limb, 4 foot
        for g in robot:
            name = m.geom(g).name
            if name[:2] in LEGS and name[2] == "_":
                part = name[3:]
                self.kind[g] = 2 if part in ("hip_motor", "knee_motor", "belt") else 4 if part == "foot" else 3
            else:
                self.kind[g] = 1
        self.names = [m.geom(g).name or f"world{g}" for g in range(m.ngeom)]
        # Leg-to-trunk distances come from mj_geomDistance on every pair: contacts within a
        # margin are exact for penetration but can miss positive distances (box-box pairs).
        legs = [g for g in robot if self.kind[g] in (2, 3, 4)]
        body = [g for g in robot if self.kind[g] == 1]
        self.self_pairs = [(a, b) for a in legs for b in body]
        self._ft = np.zeros(6)

    def __call__(self, qpos_mj: np.ndarray, detail: bool = False):
        mj, m, d = self.mj, self.m, self.d
        d.qpos[:] = qpos_mj
        mj.mj_kinematics(m, d)
        mj.mj_collision(m, d)
        far = self.MARGIN * 1000
        out = dict(self=far, stack=far, limb=far)
        worst = {}
        for a, b in self.self_pairs:
            dist = mj.mj_geomDistance(m, d, a, b, self.MARGIN, self._ft) * 1000
            if dist < out["self"]:
                out["self"] = dist
                worst["self"] = (self.names[a], self.names[b])
        for c in d.contact[:d.ncon]:
            k1, k2 = self.kind[c.geom1], self.kind[c.geom2]
            ks = {k1, k2}
            dist = float(c.dist) * 1000
            if 1 in ks and ks & {2, 3, 4}:
                continue  # measured exactly above
            elif 0 in ks and 2 in ks:
                key = "stack"
            elif 0 in ks and 3 in ks:
                key = "limb"
            else:
                continue
            if dist < out[key]:
                out[key] = dist
                worst[key] = (self.names[c.geom1], self.names[c.geom2])
        return (out, worst) if detail else out

    def ok(self, qpos_mj: np.ndarray) -> tuple[bool, float]:
        """(passes, worst violation in mm: <= 0 when it passes)."""
        c = self(qpos_mj)
        viol = max(SELF_CLEARANCE * 1000 - c["self"], STACK_CLEARANCE * 1000 - c["stack"])
        return viol <= 0, viol


def audit(p: RobotParams, s: StairParams, ref: "Reference") -> dict[str, np.ndarray]:
    """Clearance of every reference frame (mm): see Clearance."""
    chk = Clearance(p, s)
    rows = [chk(q) for q in ref.qpos]
    return {k: np.array([r[k] for r in rows]) for k in ("self", "stack", "limb")}


# ---------------------------------------------------------------- flights

def full_flight_scenario(p: RobotParams, s: StairParams, flight: str):
    """The bottom-transition scenario with its goal moved to the top: the whole flight."""
    sc = make_scenario(s, flight, "bottom", "up", p.body_length)
    return dataclasses.replace(sc, goal_sids=(sc.n_risers,))


def flight_planner(p: RobotParams, s: StairParams, flight: str) -> Planner:
    """Whole-flight planner with |trunk pitch| capped at the stair slope. Shoves tip the robot
    backward where the trunk is steeper than the stairs, and the cap costs nothing for the
    16 cm robot (M0 step 3e, outputs/*_pitch-cap). It can't go lower: mid-flight the trunk
    lies along the stairs.

    Static margin floor FLIGHT_MARGIN: 1.5 cm is the most both flights plan with for the N+K robot
    (2 cm does not plan within an hour; 2026-10-20). It covers most of the trunk-COM randomization
    (whole-body COM shift up to ~1 cm), which tipped the robot backward at 1.1 cm."""
    return Planner(p, full_flight_scenario(p, s, flight), margin=FLIGHT_MARGIN, budget=20000, max_pitch=s.slope,
                   prefer=FLIGHT_PREFER)


FLIGHT_MARGIN = 0.015
# Soft target above the floor (Planner.prefer): the run-4 probe found the N+K robot tipping forward
# where the plan sat at 1.6-2.0 cm for seconds, with the randomized trunk COM taking ~0.9 cm of it.
FLIGHT_PREFER = 0.025


def plan_flight(p: RobotParams, s: StairParams, flight: str, time_limit: float = 3600.0):
    pl = flight_planner(p, s, flight)
    path, status = pl.plan(time_limit=time_limit)
    return pl, path, status


# Shift timing: each shift takes at least SHIFT_MIN_S, and long enough that the trunk pitches at
# most 10 deg/s, the neck turns at most 1.2 rad/s and the trunk travels (x, z) at most 0.1 m/s on
# average. A fixed 1 s shift put a 15 deg pitch, a 9 cm rise and a 1-2 rad neck swing into one
# second, and the run-3 failures sit on those shifts. The rates slow the biggest ~20% of shifts by
# 1.5-3x and speed up the small ones, +21 s over both flights (inside 20 s per step).
SHIFT_RATES = dict(pitch=math.radians(10.0), neck=1.2, travel=0.1)
SHIFT_MIN_S = 0.8


def shift_duration(base_start, base_end, neck0, neck1) -> float:
    """Flight shift, bases (x, sway, z, pitch): the sway is left out of the travel."""
    d = np.asarray(base_end) - np.asarray(base_start)
    return rate_duration(d[3], np.abs(neck1 - neck0).max(), math.hypot(d[0], d[2]))


def rate_duration(d_pitch, d_neck, travel, min_s: float = SHIFT_MIN_S) -> float:
    return max(min_s, abs(d_pitch) / SHIFT_RATES["pitch"], float(d_neck) / SHIFT_RATES["neck"],
               travel / SHIFT_RATES["travel"])


def flight_reference(p: RobotParams, s: StairParams, model: pin.Model, pl: Planner, path: list, flight: str,
                     dt: float = 0.02, T_swing: float = 0.8) -> Reference:
    """Timed whole-body poses along a planner path, in the world frame.

    The trunk moves with a smooth (cosine) profile during shifts, timed by `shift_duration`; swing feet follow the
    planner's collision-free swing paths. The neck takes the fold `Planner.smooth_necks` picks
    for each move (least total neck motion within 2.5 mm of the best margin), blended during
    shifts and held during swings.

    The planner's sideways sway is a 2D idea: it doesn't know the ab/ad joint limits
    (+-0.6 rad). Where a hip sits low over its foot, a 5 cm sway would need more than
    that, and the IK would drag the planted foot. So each shift (with the swing after it)
    is built with the largest sway, in quarters of the planned one, that the 3D IK
    reaches exactly. `sway_scale` on the result records what was kept; the 3D COM margin
    that remains is measured by `static_margins`.

    The same search also keeps the 3D clearances (`Clearance.ok`): sway moves the trunk over the
    feet, so the ab/ad roll it needs can bring a leg against the trunk or a hip stack down onto a
    nosing. A group that no scale makes clear keeps the scale with the smallest violation;
    `clear_viol` records it (mm, <= 0 is clear).

    After clearance, the search keeps the planner's margin floor (FLIGHT_MARGIN) in 3D, over every
    frame of the group, not just the keyframes the planner checked: less sway can leave the COM too
    close to the edge of the stance triangle, and where the planned sway itself falls short in 3D
    (diagonal edges), up to 1.5x the planned sway is tried. `margin_short` records what no scale
    recovers (m).
    """
    from .dynamics import plan_phases, reference_q
    from .render import planner_to_world_qpos

    chk = Clearance(p, s)
    data = model.createData()
    folds = pl.smooth_necks(path)
    phases = plan_phases(pl, path, 1.0, T_swing, necks=folds)
    frames = expand_plan(pl, path, necks=folds)[1:]  # one keyframe per phase, same order
    assert len(frames) == len(phases)
    necks = [np.array(NECK_OPTIONS[f.neck]) for f in frames]
    neck_prev = np.array(NECK_OPTIONS[folds[0]])
    held = neck_prev  # a swing holds the fold of the shift before it (its own keyframe is the next pose's)
    for j, ph in enumerate(phases):
        if ph.kind == "shift":
            phases[j] = dataclasses.replace(ph, duration=shift_duration(ph.base_start, ph.base_end, held, necks[j]))
            held = necks[j]

    def margin(out):
        lo = np.inf
        for q, c in out:
            qp = mj_to_pin(q)
            feet = foot_positions(model, data, qp)
            com = pin.centerOfMass(model, data, qp)[:2]
            lo = min(lo, support_margin(com, [feet[l][:2] for l, on in zip(LEGS, c) if on]))
        return lo

    def phase_frames(ph, base_start, base_end, neck0, neck1, first):
        n = max(2, int(round(ph.duration / dt)))
        out, err = [], 0.0
        for k in range(n + first):  # the very first phase also emits its t = 0 frame
            t = (k + (not first)) / n
            feet = dict(ph.feet)
            if ph.kind == "shift":
                base = base_start + smooth(t) * (base_end - base_start)
                nk = neck0 + smooth(t) * (neck1 - neck0)
            else:
                base, nk = base_start, neck0
                leg = ph.swing_leg
                f0, f1 = ph.swing_from, ph.feet[leg]
                u = np.linspace(0, 1, len(ph.swing_ref))
                xz = [np.interp(t, u, ph.swing_ref[:, c]) for c in (0, 1)]
                feet[leg] = np.array([xz[0], f0[1] + smooth(t) * (f1[1] - f0[1]), xz[1]])
            q = reference_q(p, model, base, feet)
            q[-2:] = nk
            err = max(err, max(float(np.linalg.norm(f - feet[l]))
                               for l, f in foot_positions(model, data, q).items()))
            planted = [l in ph.stance or (ph.kind == "swing" and t >= 1.0 and l == ph.swing_leg) for l in LEGS]
            out.append((planner_to_world_qpos(q, s, flight, "up"), planted))
        return out, err

    qs, contact, scales, viols, shorts = [], [], [], [], []
    base_prev = phases[0].base_start.copy()
    i = 0
    while i < len(phases):
        group = [i] + ([i + 1] if i + 1 < len(phases) and phases[i + 1].kind == "swing" else [])
        best = None
        for scale in (1.0, 0.75, 0.5, 0.25, 0.0, 1.1, 1.2, 1.35, 1.5):  # more than planned only if nothing else holds
            end = phases[i].base_end.copy()
            end[1] *= scale
            out, err, n0 = [], 0.0, neck_prev
            for j in group:
                ph = phases[j]
                n1 = necks[j] if ph.kind == "shift" else n0
                start = base_prev if ph.kind == "shift" else end
                f, e = phase_frames(ph, start, end, n0, n1, first=not qs and not out)
                out += f
                err = max(err, e)
                n0 = n1
            viol = max(chk.ok(o[0])[1] for o in out)
            short = max(FLIGHT_MARGIN - margin(out), 0.0)
            score = (err >= 1.5e-3, max(viol, 0.0), short, err)
            if best is None or score < best[0]:
                best = (score, out, scale, end, n0, viol)
            if score[:3] == (False, 0.0, 0.0):
                break
        (_, _, short, _), out, scale, base_prev, neck_prev, viol = best
        qs += [o[0] for o in out]
        contact += [o[1] for o in out]
        scales += [scale] * len(out)
        viols += [viol] * len(out)
        shorts += [short] * len(out)
        i += len(group)
    q = np.array(qs)
    ref = Reference(dt, q, np.array(contact), np.zeros(len(q), int), [f"{flight} flight"])
    ref.sway_scale = np.array(scales)
    ref.clear_viol = np.array(viols)
    ref.margin_short = np.array(shorts)
    return ref


# ---------------------------------------------------------------- landing crawl

def _yaw_rot(yaw: float) -> np.ndarray:
    c, s_ = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s_, 0.0], [s_, c, 0.0], [0.0, 0.0, 1.0]])


def _base_q(model: pin.Model, x, y, z, yaw, pitch, joints) -> np.ndarray:
    q = pin.neutral(model)
    q[:3] = (x, y, z)
    q[3:7] = pin.Quaternion(pin.utils.rpyToMatrix(0.0, -pitch, yaw)).coeffs()
    q[7:] = joints
    return q


def support_margin(com_xy: np.ndarray, feet_xy: list[np.ndarray]) -> float:
    """Signed distance from the COM to the edge of the support polygon (convex hull of the
    stance feet, top view); positive inside."""
    pts = np.array(feet_xy)
    c = pts.mean(0)
    ang = np.arctan2(*(pts - c).T[::-1])
    pts = pts[np.argsort(ang)]
    m = np.inf
    for a, b in zip(pts, np.roll(pts, -1, 0)):
        e = b - a
        nrm = np.array([-e[1], e[0]]) / (np.linalg.norm(e) + 1e-12)  # inward for CCW order
        m = min(m, float(nrm @ (com_xy - a)))
    return m


class LandingCrawl:
    """Quasi-static crawl on the flat landing, from the lower flight's last pose to the upper
    flight's first pose."""

    def __init__(self, p: RobotParams, s: StairParams, model: pin.Model, dt: float = 0.02,
                 T_shift: float = 1.0, T_swing: float = 0.8, step_len: float = 0.035, step_yaw: float = 0.12,
                 lift: float = 0.05):
        self.p, self.s, self.model, self.dt = p, s, model, dt
        self.data = model.createData()
        self.T_shift, self.T_swing, self.step_len, self.step_yaw, self.lift = T_shift, T_swing, step_len, step_yaw, lift
        self.z_ground = s.flights[0].n_risers * s.rise
        self.h = stand_height(p)
        self.wy = p.hip_width / 2 + p.abad_offset
        self.nominal = {leg: np.array([(1 if leg[0] == "F" else -1) * p.hip_spacing / 2,
                                       self.wy if leg[1] == "L" else -self.wy, 0.0]) for leg in LEGS}
        self.chk = Clearance(p, s)
        self.qs: list[np.ndarray] = []
        self.contact: list[list[bool]] = []
        self.log: list[dict] = []  # per swing: base, feet, margin (for the report)

    # -- geometry
    def nominal_foot(self, pose) -> dict[str, np.ndarray]:
        x, y, yaw = pose
        R = _yaw_rot(yaw)
        return {leg: np.array([x, y, self.z_ground + self.p.foot_radius]) + R @ self.nominal[leg] for leg in LEGS}

    def com(self, q) -> np.ndarray:
        return pin.centerOfMass(self.model, self.data, q).copy()

    def ik(self, q_guess, feet) -> tuple[np.ndarray, float]:
        from .dynamics import _refine_legs

        q = _refine_legs(self.model, q_guess, feet, iters=80)
        err = max(float(np.linalg.norm(f - feet[l])) for l, f in foot_positions(self.model, self.data, q).items())
        return q, err

    def reachable(self, q, feet) -> bool:
        """Every foot within 92% of full reach of its hip-pitch joint (3D)."""
        pin.framesForwardKinematics(self.model, self.data, q)
        for leg in LEGS:
            hip = self.data.oMf[self.model.getFrameId(f"{leg}_thigh")].translation
            if np.linalg.norm(feet[leg] - hip) > 0.92 * self.p.leg_reach:
                return False
        return True

    # -- emitting frames
    def _emit(self, q, planted):
        self.qs.append(q.copy())
        self.contact.append(list(planted))

    def shift(self, q0, base1, feet, neck1=None):
        """Move the trunk from q0's pose to base1 = (x, y, z, yaw, pitch) with all feet planted, in
        T_shift or longer at SHIFT_RATES (yaw counts as pitch). Most crawl shifts are small; the
        first (levelling from the flight) and the last (into the upper flight's pose) are not."""
        x0 = np.array([*q0[:3], *self._yaw_pitch(q0)])
        x1 = np.array(base1, float)
        x1[3] = x0[3] + (x1[3] - x0[3] + math.pi) % (2 * math.pi) - math.pi  # shortest way round
        neck0 = q0[-2:].copy()
        neck1 = neck0 if neck1 is None else np.asarray(neck1)
        T = rate_duration(max(abs(x1[3] - x0[3]), abs(x1[4] - x0[4])), np.abs(neck1 - neck0).max(),
                          float(np.linalg.norm(x1[:3] - x0[:3])), min_s=self.T_shift)
        n = max(2, int(round(T / self.dt)))
        q = q0
        for k in range(1, n + 1):
            b = x0 + smooth(k / n) * (x1 - x0)
            q = _base_q(self.model, *b, q[7:])
            q[-2:] = neck0 + smooth(k / n) * (neck1 - neck0)
            q, err = self.ik(q, feet)
            if err > 2e-3:
                raise RuntimeError(f"landing shift unreachable (foot error {err * 1000:.1f} mm)")
            self._emit(q, [True] * 4)
        return q

    def swing(self, q0, feet, leg, target):
        f0 = feet[leg].copy()
        n = max(2, int(round(self.T_swing / self.dt)))
        q = q0
        for k in range(1, n + 1):
            t = k / n
            f = dict(feet)
            f[leg] = f0 + smooth(t) * (target - f0) + np.array([0, 0, self.lift * math.sin(math.pi * t)])
            q, err = self.ik(q, f)
            if err > 2e-3:
                raise RuntimeError(f"landing swing unreachable ({leg}, {err * 1000:.1f} mm)")
            self._emit(q, [l != leg or k == n for l in LEGS])
        feet = dict(feet)
        feet[leg] = target.copy()
        return q, feet

    @staticmethod
    def _yaw_pitch(q):
        R = pin.Quaternion(q[3:7]).matrix()
        rpy = pin.rpy.matrixToRpy(R)
        return float(rpy[2]), float(-rpy[1])

    def sway_for(self, q, base_xy_yaw, feet, leg) -> tuple[np.ndarray, np.ndarray, float]:
        """Smallest sideways/fore-aft trunk offset from the target pose that puts the COM inside
        the other three feet by MARGIN, with every foot still reachable and the legs clear of the
        trunk (Clearance). Returns (q, base, margin)."""
        x, y, yaw = base_xy_yaw
        stance = [feet[l][:2] for l in LEGS if l != leg]
        best = None
        offsets = [(dx, dy) for dx in np.linspace(-0.08, 0.08, 17) for dy in np.linspace(-0.08, 0.08, 17)]
        for dx, dy in sorted(offsets, key=lambda o: math.hypot(*o)):
            if best is not None and math.hypot(dx, dy) > best[0] + 1e-9:
                break
            b = (x + dx, y + dy, self.z_ground + self.h, yaw, 0.0)
            qb = _base_q(self.model, *b, q[7:])
            if not self.reachable(qb, feet):
                continue
            qb, err = self.ik(qb, feet)
            if err > 2e-3 or not self.chk.ok(pin_to_mj(qb))[0]:
                continue
            m = support_margin(self.com(qb)[:2], stance)
            if m >= MARGIN and (best is None or m > best[3]):
                best = (math.hypot(dx, dy), b, qb, m)
        if best is None:
            raise RuntimeError(f"no sway gives {MARGIN} m margin before lifting {leg}")
        return best[2], np.array(best[1]), best[3]

    # -- the maneuver
    def waypoints(self, start, goal) -> list[tuple[float, float, float]]:
        """Trunk (x, y, yaw) path: to mid-landing, turn 90 degrees, cross to the upper flight's
        line, turn 90 more, then to the upper flight's start pose."""
        s = self.s
        x_top = (s.flights[0].n_risers - 1) * s.run
        x_mid = x_top + s.flights[0].top_depth / 2
        corners = [start, (x_mid, start[1], start[2]), (x_mid, start[1], start[2] + math.pi / 2),
                   (x_mid, goal[1], start[2] + math.pi / 2), (x_mid, goal[1], goal[2]), goal]
        pts = [corners[0]]
        for a, b in zip(corners[:-1], corners[1:]):
            dyaw = (b[2] - a[2] + math.pi) % (2 * math.pi) - math.pi
            n = max(1, math.ceil(max(math.hypot(b[0] - a[0], b[1] - a[1]) / self.step_len, abs(dyaw) / self.step_yaw)))
            for k in range(1, n + 1):
                pts.append((a[0] + k / n * (b[0] - a[0]), a[1] + k / n * (b[1] - a[1]), a[2] + k / n * dyaw))
        return pts

    def run(self, q_start: np.ndarray, q_goal: np.ndarray) -> Reference:
        """q_start/q_goal are Pinocchio world configurations (lower flight end, upper flight start)."""
        feet = foot_positions(self.model, self.data, q_start)
        goal_feet = foot_positions(self.model, self.data, q_goal)
        yaw0, _ = self._yaw_pitch(q_start)
        yawg, pitch_g = self._yaw_pitch(q_goal)
        path = self.waypoints((q_start[0], q_start[1], yaw0), (q_goal[0], q_goal[1], yawg))
        # The flight can end with the feet staggered fore-aft: square the stance up in place first
        # (the first swings target the start pose's nominal footprint).
        path = [path[0]] * 5 + path
        folded = np.array(NECK_OPTIONS[NECK_FOLDED])
        # Level the trunk at standing height and fold the neck, feet where the flight left them.
        q = self.shift(q_start, (q_start[0], q_start[1], self.z_ground + self.h, yaw0, 0.0), feet, folded)
        order = ("RL", "FL", "RR", "FR")
        k, i = 0, 0
        K = len(path) - 1
        while True:
            done_path = k >= K
            leg = order[i % 4]
            i += 1
            if done_path:
                targets = goal_feet
            else:
                targets = self.nominal_foot(path[min(k + 2, K)])
            k = min(k + 1, K)
            if np.linalg.norm(feet[leg] - targets[leg]) < 1e-3:
                if done_path and all(np.linalg.norm(feet[l] - goal_feet[l]) < 1e-3 for l in LEGS):
                    break
                continue
            # A stance left by the flight (feet far apart fore-aft) can leave too little support
            # for the scheduled leg: then lift the first other leg that still has to move.
            for cand in [leg] + [l for l in order if l != leg]:
                if cand != leg and np.linalg.norm(feet[cand] - targets[cand]) < 1e-3:
                    continue
                try:
                    qb, base, m = self.sway_for(q, path[k], feet, cand)
                except RuntimeError:
                    continue
                leg = cand
                break
            else:
                raise RuntimeError(f"no leg can lift with {MARGIN} m margin (swing {len(self.log)})")
            q = self.shift(q, base, feet)
            q, feet = self.swing(q, feet, leg, targets[leg])
            self.log.append(dict(base=base.copy(), feet={l: f.copy() for l, f in feet.items()}, leg=leg, margin=m,
                                 com=self.com(q)[:2].copy()))
            if i > 400:
                raise RuntimeError("landing crawl did not converge")
        neck_g = q_goal[-2:]
        q = self.shift(q, (*q_goal[:3], yawg, pitch_g), feet, neck_g)
        qm = np.array([pin_to_mj(x) for x in self.qs])
        return Reference(self.dt, qm, np.array(self.contact), np.zeros(len(qm), int), ["landing"])


# ---------------------------------------------------------------- checks

def static_margins(model: pin.Model, ref: Reference) -> np.ndarray:
    """Per frame: signed top-view distance from the COM to the edge of the planted feet's
    support polygon (m). Positive means the pose holds without any dynamics."""
    data = model.createData()
    out = np.empty(len(ref.qpos))
    for i, (q, c) in enumerate(zip(ref.qpos, ref.contact)):
        qp = mj_to_pin(q)
        feet = foot_positions(model, data, qp)
        com = pin.centerOfMass(model, data, qp)[:2]
        out[i] = support_margin(com, [feet[l][:2] for l, on in zip(LEGS, c) if on])
    return out


# ---------------------------------------------------------------- round trip

def round_trip(p: RobotParams, s: StairParams, paths: dict[str, list], dt: float = 0.02,
               hold_s: float = 1.5) -> tuple[Reference, dict]:
    """Full round trip from the two whole-flight plans. Returns (reference, extras for the report)."""
    from .dynamics import pin_model

    model = pin_model(p)
    flights = {}
    for fl in ("lower", "upper"):
        pl = flight_planner(p, s, fl)
        flights[fl] = flight_reference(p, s, model, pl, paths[fl], fl, dt)
    crawl = LandingCrawl(p, s, model, dt)
    landing = crawl.run(mj_to_pin(flights["lower"].qpos[-1]), mj_to_pin(flights["upper"].qpos[0]))
    all_down = [True] * 4
    up = concat([hold(flights["lower"].qpos[0], all_down, hold_s, dt, "start"),
                 flights["lower"], landing, flights["upper"],
                 hold(flights["upper"].qpos[-1], all_down, hold_s, dt, "top")])
    down = up.reversed(lambda n: {"start": "finish", "top": "top"}.get(n, f"{n} (down, rear-first)"))
    extras = dict(crawl_log=crawl.log, n_up=len(up.qpos),
                  sway_scale={fl: r.sway_scale for fl, r in flights.items()},
                  clear_viol={fl: r.clear_viol for fl, r in flights.items()},
                  margin_short={fl: r.margin_short for fl, r in flights.items()})
    return concat([up, down]), extras
