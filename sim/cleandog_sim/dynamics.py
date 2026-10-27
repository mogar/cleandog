"""Crocoddyl whole-body trajectory optimisation along a quasi-static transition plan (M0 step 2).

The quasi-static planner (transitions.py) fixes the contact sequence. Here it
becomes a dynamic problem on the full 3D model, loaded from the same MJCF
through Pinocchio.

Each plan step becomes two phases:

- **Shift** (`T_shift`): four feet in contact; the trunk tracks the planned
  pose, including the sideways sway.
- **Swing** (`T_swing`): three feet in contact; the swing foot follows the
  planner's collision-free path through the leg's free space and lands with
  low velocity.

Contact forces must stay inside friction cones (μ = `mu`, normals vertical:
every foothold is on a tread, floor or landing), and joint torques are
box-bounded at each actuator's peak. The outputs are the torque, speed and
friction demands of the motion, which feed the actuator spec.

Not modelled: collisions are not constraints here (the replay in MuJoCo
measures any penetration), motor torque-speed curves (speeds are checked
afterwards), and contact compliance.
"""

from __future__ import annotations

import math
import os
import tempfile
from dataclasses import dataclass

import crocoddyl
import numpy as np
import pinocchio as pin

from .params import RobotParams
from .robot import LEGS, joint_actuation
from .scene import build_scene_xml
from .transitions import NECK_OPTIONS, Planner


def pin_model(p: RobotParams) -> pin.Model:
    xml = build_scene_xml(p, None)
    with tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False) as f:
        f.write(xml)
        path = f.name
    try:
        return pin.buildModelFromMJCF(path)
    finally:
        os.unlink(path)


# ---------------------------------------------------------------- plan -> phases

@dataclass
class Phase:
    kind: str  # "shift" | "swing"
    duration: float
    stance: tuple[str, ...]
    feet: dict  # leg -> 3D foothold (stance feet, and the swing target for the swing leg)
    base_start: np.ndarray  # (x, y, z, pitch)
    base_end: np.ndarray
    swing_leg: str | None = None
    swing_from: np.ndarray | None = None
    swing_ref: np.ndarray | None = None  # (n, 2) planner x-z foot path for the swing leg


def swing_reference(pl: Planner, pose, leg: str, f0_2d, f1_2d, n: int = 40) -> np.ndarray:
    """Foot path (n x 2, planner x-z) for a swing: the shortest path through the leg's free
    space, lightly smoothed where smoothing stays in free space, timed so the foot starts and
    stops gently (cosine profile along arc length)."""
    raw = pl.swing_path(pose, leg, f0_2d, f1_2d)
    if raw is None:
        raise ValueError(f"no swing path for {leg}")
    path = raw.copy()
    if len(path) > 4:
        sm = path.copy()
        sm[1:-1] = (path[:-2] + 2 * path[1:-1] + path[2:]) / 4
        gx, gz, free, labels = pl.swing_space(pose, leg)
        cells = [pl._cell(gx, gz, q) for q in sm[1:-1]]
        if all(free[c] for c in cells):
            path = sm
    seg = np.hypot(*np.diff(path, axis=0).T)
    s_cum = np.concatenate([[0.0], np.cumsum(seg)])
    t = np.linspace(0, 1, n)
    s_t = (0.5 - 0.5 * np.cos(np.pi * t)) * s_cum[-1]
    return np.stack([np.interp(s_t, s_cum, path[:, 0]), np.interp(s_t, s_cum, path[:, 1])], 1)


def plan_phases(pl: Planner, path: list, T_shift: float = 1.0, T_swing: float = 0.8,
                necks: list[int] | None = None) -> list[Phase]:
    wy = pl.wy
    to3 = {leg: (lambda f, leg=leg: np.array([f[0], wy if leg[1] == "L" else -wy, f[1]])) for leg in LEGS}
    phases = []
    prev_base = None
    for k, (a, b) in enumerate(zip(path[:-1], path[1:]), 1):
        fa, fb = pl.feet_of(tuple(a[5])), pl.feet_of(tuple(b[5]))
        moved = [l for l in LEGS if not np.allclose(fa[l], fb[l])]
        leg = moved[0] if moved else None
        # Sway: the y offset that best supports lifting `leg` at the shifted pose.
        sway = 0.0
        if leg:
            _, info = pl.evaluate(pl.pose(*b[:3]), fa, lifted=leg, neck=None if necks is None else necks[k])
            sway = float(info["sway"][0])
        start = prev_base if prev_base is not None else np.array([a[0], 0.0, a[1], a[2]])
        end = np.array([b[0], sway, b[1], b[2]])
        feet_a = {l: to3[l](fa[l]) for l in LEGS}
        phases.append(Phase("shift", T_shift, tuple(LEGS), feet_a, start, end))
        if leg:
            feet_b = {l: to3[l](fb[l]) for l in LEGS}
            ref = swing_reference(pl, tuple(b[:3]), leg, fa[leg], fb[leg])
            phases.append(Phase("swing", T_swing, tuple(l for l in LEGS if l != leg), feet_b, end, end,
                                leg, feet_a[leg], ref))
        prev_base = end
    return phases


def reference_q(p: RobotParams, model: pin.Model, base: np.ndarray, feet: dict, neck_opt: int = 3) -> np.ndarray:
    """Configuration for a trunk pose (x, y, z, pitch) and 3D footholds, by planar IK per leg.

    The knee direction comes from the morphology; ab/ad takes up the body's sideways sway.
    """
    from .transitions import Poses

    x, y, z, beta = base
    q = pin.neutral(model)
    q[:3] = (x, y, z)
    q[3:7] = pin.Quaternion(pin.utils.rpyToMatrix(0.0, -beta, 0.0)).coeffs()  # nose-up pitch
    P = Poses(np.array([[x, z]]), np.array([beta]))
    c, s_ = math.cos(beta), math.sin(beta)
    i = 7
    for leg in LEGS:
        front = leg[0] == "F"
        hip = P.hip(p, front)[0]
        f = feet[leg][[0, 2]]
        knee_back = (p.knee_sign_front if front else p.knee_sign_rear) == -1
        l1, l2 = p.thigh_length, p.shin_length
        d_vec = f - hip
        d = min(float(np.hypot(*d_vec)), l1 + l2 - 1e-4)
        a = math.acos(np.clip((l1**2 + d**2 - l2**2) / (2 * l1 * d), -1, 1))
        ang = math.atan2(d_vec[1], d_vec[0]) + (-a if knee_back else a)
        knee = hip + l1 * np.array([math.cos(ang), math.sin(ang)])

        def joint_angle(v):
            dx, dz = c * v[0] + s_ * v[1], -s_ * v[0] + c * v[1]
            return math.atan2(-dx, -dz)

        q_hip = joint_angle(knee - hip)
        q_knee = (joint_angle(f - knee) - q_hip + math.pi) % (2 * math.pi) - math.pi
        side_y = (p.hip_width / 2 + p.abad_offset) * (1 if leg[1] == "L" else -1)
        q_abad = math.asin(np.clip((feet[leg][1] - (y + side_y)) / max(d, 1e-3), -0.5, 0.5))
        q[i:i + 3] = (q_abad, q_hip, q_knee)
        i += 3
    if p.include_neck:
        q[i:i + 2] = NECK_OPTIONS[neck_opt]
    return _refine_legs(model, q, feet)


def _refine_legs(model: pin.Model, q: np.ndarray, feet: dict, iters: int = 60) -> np.ndarray:
    """Exact 3D IK: damped least squares on each leg's three joints so its foot hits feet[leg]."""
    data = model.createData()
    q = q.copy()
    for it in range(iters):
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        worst = 0.0
        for li, leg in enumerate(LEGS):
            fid = model.getFrameId(f"{leg}_foot")
            err = feet[leg] - data.oMf[fid].translation
            worst = max(worst, float(np.abs(err).max()))
            J = pin.computeFrameJacobian(model, data, q, fid, pin.LOCAL_WORLD_ALIGNED)[:3]
            cols = slice(6 + 3 * li, 9 + 3 * li)  # this leg's velocity columns
            Jl = J[:, cols]
            dq = Jl.T @ np.linalg.solve(Jl @ Jl.T + 1e-4 * np.eye(3), err)
            dq *= min(1.0, 0.2 / max(np.abs(dq).max(), 1e-12))  # clip steps near a straight leg
            sl = slice(7 + 3 * li, 10 + 3 * li)
            # Stay inside the joint ranges; the knee range encodes the knee direction, so this
            # also stops the solution flipping to the other knee branch.
            q[sl] = np.clip(q[sl] + dq, model.lowerPositionLimit[sl], model.upperPositionLimit[sl])
        if worst < 1e-5:
            break
    return q


# ---------------------------------------------------------------- problem

@dataclass
class Weights:
    base_track: float = 1e4
    state_reg: float = 1e1
    control_reg: float = 1e-2
    friction: float = 1e4
    swing_track: float = 1e5
    swing_end: float = 1e6
    touchdown_vel: float = 1e4


def build_problem(p: RobotParams, model: pin.Model, phases: list[Phase], dt: float = 0.02, mu: float = 0.5,
                  w: Weights = Weights()):
    state = crocoddyl.StateMultibody(model)
    actuation = crocoddyl.ActuationModelFloatingBase(state)
    nu = actuation.nu
    fid = {leg: model.getFrameId(f"{leg}_foot") for leg in LEGS}
    trunk = model.getFrameId("trunk")
    limits = np.array([ja.peak_torque for ja in joint_actuation(p)])

    # State regularisation: base position/orientation free-ish (tracked separately), joints held
    # near the reference posture, velocities damped.
    x_w = np.concatenate([np.zeros(6), np.full(model.nv - 6, 1.0), np.full(6, 1.0), np.full(model.nv - 6, 0.5)])

    def knot(stance, feet, base, q_ref, extra):
        contacts = crocoddyl.ContactModelMultiple(state, nu)
        costs = crocoddyl.CostModelSum(state, nu)
        for leg in stance:
            contacts.addContact(f"{leg}_contact", crocoddyl.ContactModel3D(
                state, fid[leg], feet[leg], pin.LOCAL_WORLD_ALIGNED, nu, np.array([100.0, 20.0])))  # Baumgarte (kp, kd)
            # Inscribed 4-sided pyramid: guarantees the true circular cone |f_t| <= mu f_n.
            cone = crocoddyl.FrictionCone(np.eye(3), mu, 4, True)
            res = crocoddyl.ResidualModelContactFrictionCone(state, fid[leg], cone, nu)
            act = crocoddyl.ActivationModelQuadraticBarrier(crocoddyl.ActivationBounds(cone.lb, cone.ub))
            costs.addCost(f"{leg}_friction", crocoddyl.CostModelResidual(state, act, res), w.friction)
        R = pin.utils.rpyToMatrix(0.0, -base[3], 0.0)
        res = crocoddyl.ResidualModelFramePlacement(state, trunk, pin.SE3(R, base[:3].copy()), nu)
        costs.addCost("base", crocoddyl.CostModelResidual(state, res), w.base_track)
        x_ref = np.concatenate([q_ref, np.zeros(model.nv)])
        res = crocoddyl.ResidualModelState(state, x_ref, nu)
        costs.addCost("state", crocoddyl.CostModelResidual(state, crocoddyl.ActivationModelWeightedQuad(x_w**2), res),
                      w.state_reg)
        costs.addCost("control", crocoddyl.CostModelResidual(state, crocoddyl.ResidualModelControl(state, nu)),
                      w.control_reg)
        for name, cost, weight in extra:
            costs.addCost(name, cost, weight)
        dmodel = crocoddyl.DifferentialActionModelContactFwdDynamics(state, actuation, contacts, costs, 0.0, True)
        dmodel.u_lb, dmodel.u_ub = -limits, limits
        return crocoddyl.IntegratedActionModelEuler(dmodel, dt)

    models, q_refs, meta = [], [], []
    for ph_i, ph in enumerate(phases):
        n = max(2, int(round(ph.duration / dt)))
        for k in range(n):
            t = (k + 1) / n
            base = ph.base_start + t * (ph.base_end - ph.base_start)
            feet = dict(ph.feet)
            extra = []
            if ph.kind == "swing":
                leg = ph.swing_leg
                f0, f1 = ph.swing_from, ph.feet[leg]
                xz = np.array([np.interp(t, np.linspace(0, 1, len(ph.swing_ref)), ph.swing_ref[:, c]) for c in (0, 1)])
                feet[leg] = np.array([xz[0], f0[1] + t * (f1[1] - f0[1]), xz[1]])
                if k < n - 1:
                    res = crocoddyl.ResidualModelFrameTranslation(state, fid[leg], feet[leg].copy(), nu)
                    extra.append(("swing_track", crocoddyl.CostModelResidual(state, res), w.swing_track))
                if k == n - 1:
                    res = crocoddyl.ResidualModelFrameTranslation(state, fid[leg], f1.copy(), nu)
                    extra.append(("swing_end", crocoddyl.CostModelResidual(state, res), w.swing_end))
                    res = crocoddyl.ResidualModelFrameVelocity(state, fid[leg], pin.Motion.Zero(),
                                                               pin.LOCAL_WORLD_ALIGNED, nu)
                    extra.append(("touchdown", crocoddyl.CostModelResidual(state, res), w.touchdown_vel))
            q_ref = reference_q(p, model, base, feet)
            models.append(knot(ph.stance, ph.feet, base, q_ref, extra))
            q_refs.append(q_ref)
            meta.append((ph_i, ph.kind, ph.stance))
    x0 = np.concatenate([reference_q(p, model, phases[0].base_start, phases[0].feet), np.zeros(model.nv)])
    terminal = models[-1]
    problem = crocoddyl.ShootingProblem(x0, models[:-1], terminal)
    return problem, q_refs, meta


@dataclass
class DynResult:
    converged: bool
    iterations: int
    cost: float
    t: np.ndarray  # (N,)
    tau: np.ndarray  # (N, nu) joint torques
    qd: np.ndarray  # (N+1, nv-6) joint speeds
    q: np.ndarray  # (N+1, nq)
    forces: dict  # leg -> (N, 3) contact forces (nan when not in contact)
    friction_used: np.ndarray  # (N,) max |f_t| / f_n over stance feet
    joint_names: list[str]
    meta: list
    foot_err: float  # worst swing-end foot position error, m


def solve(p: RobotParams, pl: Planner, path: list, dt: float = 0.02, mu: float = 0.5, max_iter: int = 200,
          T_shift: float = 1.0, T_swing: float = 0.8, verbose: bool = False) -> DynResult:
    model = pin_model(p)
    phases = plan_phases(pl, path, T_shift, T_swing)
    problem, q_refs, meta = build_problem(p, model, phases, dt, mu)
    solver = crocoddyl.SolverBoxFDDP(problem)
    if verbose:
        solver.setCallbacks([crocoddyl.CallbackVerbose()])
    xs0 = [problem.x0] + [np.concatenate([q, np.zeros(model.nv)]) for q in q_refs[:-1]]
    us0 = problem.quasiStatic(xs0[:-1])
    converged = solver.solve(xs0, us0, max_iter, False)

    N = len(solver.us)
    weight = p.total_mass_target * 9.81
    tau = np.array(solver.us)
    xs = np.array(solver.xs)
    q, v = xs[:, :model.nq], xs[:, model.nq:]
    forces = {leg: np.full((N, 3), np.nan) for leg in LEGS}
    fric = np.zeros(N)
    for k, d in enumerate(problem.runningDatas):
        contacts = d.differential.multibody.contacts.contacts
        for leg in LEGS:
            key = f"{leg}_contact"
            if key in contacts.todict():
                f = contacts[key].f.linear  # in the contact frame (LOCAL_WORLD_ALIGNED)
                forces[leg][k] = f
                if f[2] > 0.02 * weight:  # ignore a foot that is unloading before it lifts
                    fric[k] = max(fric[k], math.hypot(f[0], f[1]) / f[2])
    # Swing landing accuracy.
    data = model.createData()
    err = 0.0
    k = 0
    for ph in phases:
        n = max(2, int(round(ph.duration / dt)))
        k += n
        if ph.kind == "swing" and k < len(q):
            pin.framesForwardKinematics(model, data, q[k])
            err = max(err, float(np.linalg.norm(data.oMf[model.getFrameId(f"{ph.swing_leg}_foot")].translation
                                                - ph.feet[ph.swing_leg])))
    return DynResult(bool(converged), solver.iter, solver.cost, np.arange(N) * dt, tau, v[:, 6:], q, forces, fric,
                     [ja.joint for ja in joint_actuation(p)], meta, err)
