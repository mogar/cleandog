"""RL tracking of the round-trip reference in MJX (Warp backend) — M0 step 3.

The policy is a residual on the kinematic reference (reference.py), DeepMimic
style. Every control step (50 Hz) it outputs joint-angle offsets added to the
reference pose. A PD loop, run at the physics rate, turns those into motor
torques clipped to each actuator's torque-speed envelope. The reference fixes
which foot moves when and roughly where. The policy supplies everything the
reference leaves out: balance, contact forces, and recovery from errors.

What the policy observes is what the robot could estimate:

- IMU: gravity direction and gyro;
- joint angles and speeds;
- body velocity;
- the reference ahead of it, relative to its own pose.

The last item assumes the robot localizes itself against the stairwell, to
about a centimetre. The critic additionally sees privileged simulator state
(true foot errors, contact, friction, mass).

Domain randomization covers:

- foot and skid friction;
- link masses, payload and centre-of-mass offset;
- joint friction and rotor inertia;
- PD gains and motor strength;
- 0-12 ms actuation latency;
- observation noise;
- pushes.

Not modelled: stair geometry variation, sensor bias drift, cable drag, and
motor heating (RMS torque is reported instead).
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground._src import mjx_env

from .params import RobotParams, StairParams
from .reference import Reference
from .robot import LEGS, joint_actuation
from .scene import build_scene_xml

CTRL_DT = 0.02
SIM_DT = 0.004
N_FORCED = 4  # scheduled shoves an episode can hold: one per flight in a round trip


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt=CTRL_DT,
        sim_dt=SIM_DT,
        action_scale=0.25,
        kp_leg=50.0,
        kd_leg=1.2,
        kp_neck=20.0,
        kd_neck=0.5,
        reward=config_dict.create(
            joint=0.3, base_pos=0.2, base_rot=0.15, feet=0.25, joint_vel=0.1,
            torque=-5e-5, action_rate=-0.005, touchdown=-0.1, body_contact=-0.2,
            sigma_joint=0.15, sigma_pos=0.05, sigma_rot=0.15, sigma_feet=0.03, sigma_vel=1.5,
        ),
        term_pos=0.20,  # m, trunk from reference
        term_rot=0.6,  # rad
        noise=config_dict.create(level=1.0, joint_pos=0.01, joint_vel=0.5, gyro=0.1, gravity=0.03,
                                 linvel=0.05, ref_pos=0.01),
        push=config_dict.create(enable=True, interval=(4.0, 10.0), speed=(0.0, 0.25)),
        randomize_gains=True,
        latency_substeps=(0, 3),  # actuation delay, in physics steps (0-12 ms)
        init_noise=config_dict.create(joint=0.03, xy=0.01, yaw=0.03),
        # Start more episodes where the trunk is steep: shoves tip the robot there (M0 step 3d).
        steep_start_deg=40.0,
        steep_start_weight=3.0,
        impl="warp",
        # Who records CUDA graphs for the Warp physics: "warp" (MJX's GPU default), "jax" or "none".
        graph_mode="warp",
        naconmax_per_env=32,
        njmax=160,
    )


# ---------------------------------------------------------------- model and reference arrays

def rl_scene_xml(p: RobotParams, s: StairParams, timestep: float = SIM_DT) -> str:
    """The stairwell scene plus the sensors the env reads: a touch zone at each foot, foot
    velocities, and a touch zone around the trunk and skid plate (any contact there is
    unplanned: the reference never rests on the belly)."""
    root = ET.fromstring(build_scene_xml(p, s, timestep=timestep))
    bodies = {b.get("name"): b for b in root.iter("body")}
    sensor = root.find("sensor")
    for leg in LEGS:
        ET.SubElement(bodies[f"{leg}_shin"], "site", name=f"{leg}_touch", type="sphere",
                      pos=f"0 0 {-p.shin_length:g}", size=f"{p.foot_radius + 0.004:g}", rgba="0 0 0 0")
        ET.SubElement(sensor, "touch", name=f"{leg}_touch", site=f"{leg}_touch")
        ET.SubElement(sensor, "framelinvel", name=f"{leg}_vel", objtype="site", objname=f"{leg}_foot")
    hz = p.body_height / 2 + p.skid_thickness
    ET.SubElement(bodies["trunk"], "site", name="trunk_touch", type="box", pos=f"0 0 {-p.skid_thickness / 2:g}",
                  size=f"{p.body_length / 2 + 0.006:g} {p.body_width / 2 + 0.006:g} {hz + 0.006:g}", rgba="0 0 0 0")
    ET.SubElement(sensor, "touch", name="trunk_touch", site="trunk_touch")
    return ET.tostring(root, encoding="unicode")


@dataclass
class RefArrays:
    qpos: np.ndarray  # (N, nq)
    qvel: np.ndarray  # (N, nv)
    contact: np.ndarray  # (N, 4)
    feet: np.ndarray  # (N, 4, 3) foot-site positions
    segment: np.ndarray


def ref_arrays(m: mujoco.MjModel, ref: Reference) -> RefArrays:
    d = mujoco.MjData(m)
    n = len(ref.qpos)
    qvel = np.zeros((n, m.nv))
    for i in range(n - 1):
        mujoco.mj_differentiatePos(m, qvel[i], ref.dt, ref.qpos[i], ref.qpos[i + 1])
    sites = [m.site(f"{leg}_foot").id for leg in LEGS]
    feet = np.zeros((n, 4, 3))
    for i, q in enumerate(ref.qpos):
        d.qpos[:] = q
        mujoco.mj_kinematics(m, d)
        feet[i] = d.site_xpos[sites]
    return RefArrays(ref.qpos, qvel, ref.contact.astype(np.float32), feet, ref.segment)


# ---------------------------------------------------------------- small maths

def quat_to_mat(q):
    w, x, y, z = q
    return jp.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def rot_angle(R):
    return jp.arccos(jp.clip((jp.trace(R) - 1.0) / 2.0, -1.0, 1.0))


def yaw_quat(yaw):
    return jp.array([jp.cos(yaw / 2), 0.0, 0.0, jp.sin(yaw / 2)])


def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return jp.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


# ---------------------------------------------------------------- environment

class StairTrack(mjx_env.MjxEnv):
    """Track the round-trip reference. Episodes start anywhere on it (`t0`); the reference
    index is t0 + steps, wrapping (a round trip ends where it starts).

    `info["steps"]` (steps taken so far) belongs to the caller: Brax's EpisodeWrapper
    advances it and zeroes it on reset. Code driving the env directly must do the same.
    """

    def __init__(self, p: RobotParams, s: StairParams, ref: Reference, batch: int,
                 config: config_dict.ConfigDict | None = None, episode_length: int = 1000):
        config = config or default_config()
        super().__init__(config)
        self.p, self.s = p, s
        self._mj_model = mujoco.MjModel.from_xml_string(rl_scene_xml(p, s, config.sim_dt))
        kw = {}
        if config.impl == "warp":
            import warp as wp

            kw["graph_mode"] = getattr(wp.JaxCallableGraphMode, config.graph_mode.upper())
        self._mjx_model = mjx.put_model(self._mj_model, impl=config.impl, **kw)
        self.batch = batch
        self.naconmax = config.naconmax_per_env * batch
        self.episode_length = episode_length
        m = self._mj_model
        self.ref = ref_arrays(m, ref)
        self.segment_names = list(ref.segment_names)
        self.N = len(ref.qpos)
        self._rq = jp.array(self.ref.qpos, jp.float32)
        self._rv = jp.array(self.ref.qvel, jp.float32)
        self._rc = jp.array(self.ref.contact, jp.float32)
        self._rf = jp.array(self.ref.feet, jp.float32)

        acts = joint_actuation(p)
        self._peak = jp.array([a.peak_torque for a in acts])
        self._max_speed = jp.array([a.max_speed for a in acts])
        leg = np.array([a.joint.split("_")[0] in LEGS for a in acts])
        self._kp = jp.where(leg, config.kp_leg, config.kp_neck)
        self._kd = jp.where(leg, config.kd_leg, config.kd_neck)
        self._lo, self._hi = (jp.array(x) for x in m.jnt_range[1:].T)

        sens = lambda name: (m.sensor_adr[m.sensor(name).id], m.sensor_dim[m.sensor(name).id])
        self._touch = [sens(f"{leg}_touch")[0] for leg in LEGS]
        self._fvel = [sens(f"{leg}_vel")[0] for leg in LEGS]
        self._trunk_touch = sens("trunk_touch")[0]
        self._gyro = sens("trunk_gyro")[0]
        self._foot_sites = np.array([m.site(f"{leg}_foot").id for leg in LEGS])
        self._foot_geoms = np.array([m.geom(f"{leg}_foot").id for leg in LEGS])
        self._skid_geoms = np.array([m.geom(n).id for n in ("skid", "skid_lip_front", "skid_lip_rear")])
        self._trunk_body = m.body("trunk").id
        self.start_index: int | None = None  # set to start every episode at one reference frame
        # Episode start frames, weighted toward steep trunk pitch.
        w, x, y, z = self.ref.qpos[:, 3:7].T
        self.ref_pitch = np.degrees(np.abs(np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))))
        weights = np.where(self.ref_pitch >= config.steep_start_deg, config.steep_start_weight, 1.0)
        self._start_logits = jp.array(np.log(weights / weights.sum()), jp.float32)

    # -- required properties
    @property
    def xml_path(self) -> str:
        return ""

    @property
    def action_size(self) -> int:
        return self._mj_model.nu

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model

    # -- reset
    def reset(self, rng: jax.Array) -> mjx_env.State:
        rng, k = jax.random.split(rng)
        t0 = jax.random.categorical(k, self._start_logits) if self.start_index is None else self.start_index
        return self.reset_at(rng, t0)

    def reset_at(self, rng: jax.Array, t0) -> mjx_env.State:
        cfg = self._config
        n = cfg.init_noise
        rng, k1, k2, k3, k4, k5, k6, k7 = jax.random.split(rng, 8)
        t0 = jp.asarray(t0, jp.int32) % self.N
        qpos = self._rq[t0]
        qpos = qpos.at[0:2].add(jax.random.uniform(k1, (2,), minval=-n.xy, maxval=n.xy))
        qpos = qpos.at[2].add(0.003)
        qpos = qpos.at[3:7].set(quat_mul(yaw_quat(jax.random.uniform(k2, (), minval=-n.yaw, maxval=n.yaw)),
                                         qpos[3:7]))
        qpos = qpos.at[7:].add(jax.random.uniform(k3, (self.action_size,), minval=-n.joint, maxval=n.joint))
        qpos = qpos.at[7:].set(jp.clip(qpos[7:], self._lo, self._hi))
        data = mjx_env.make_data(self._mj_model, qpos=qpos, qvel=self._rv[t0], impl=self._mjx_model.impl.value,
                                 naconmax=self.naconmax, njmax=cfg.njmax)
        data = mjx.forward(self._mjx_model, data)

        if cfg.randomize_gains:
            gain = jax.random.uniform(k4, (2,), minval=0.8, maxval=1.2)
            strength = jax.random.uniform(k5, (), minval=0.85, maxval=1.0)
        else:
            gain, strength = jp.ones(2), jp.ones(())
        lat = jax.random.randint(k6, (), cfg.latency_substeps[0], cfg.latency_substeps[1] + 1)
        info = dict(
            rng=rng, t0=t0, steps=jp.zeros(()),
            last_act=jp.zeros(self.action_size), last_target=qpos[7:],
            kp_scale=gain[0], kd_scale=gain[1], strength=strength, latency=lat,
            push_timer=jax.random.uniform(k7, (), minval=cfg.push.interval[0], maxval=cfg.push.interval[1]),
            last_touch=jp.ones(4, bool),
            # Per joint over the last control step, for the actuator check.
            tau_peak=jp.zeros(self.action_size), tau_ms=jp.zeros(self.action_size), qd=qpos[7:] * 0,
            # Scheduled shoves (evaluation and diagnostics): at step forced_push_step[k] (-1: never),
            # add forced_push[k] (m/s, world x-y) to the trunk velocity.
            forced_push_step=jp.full(N_FORCED, -1, jp.int32), forced_push=jp.zeros((N_FORCED, 2)),
        )
        obs = self._obs(data, info, t0)
        metrics = {k: jp.zeros(()) for k in ("pos_err", "rot_err", "feet_err", "joint_err", "torque_sq",
                                             "r_joint", "r_base_pos", "r_base_rot", "r_feet", "r_joint_vel",
                                             "r_torque", "r_action_rate", "r_touchdown", "r_body_contact")}
        return mjx_env.State(data, obs, jp.zeros(()), jp.zeros(()), metrics, info)

    # -- helpers
    def ref_index(self, info, ahead=0):
        return (info["t0"] + info["steps"].astype(jp.int32) + ahead) % self.N

    def motor_torque(self, data, target, info):
        q, qd = data.qpos[7:], data.qvel[6:]
        tau = self._kp * info["kp_scale"] * (target - q) - self._kd * info["kd_scale"] * qd
        lim = self._peak * info["strength"] * jp.clip(1.0 - jp.abs(qd) / self._max_speed, 0.0, 1.0)
        return jp.clip(tau, -lim, lim)

    # -- step
    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        cfg, info = self._config, dict(state.info)
        data = state.data
        i1 = self.ref_index(info, 1)
        target = jp.clip(self._rq[i1, 7:] + cfg.action_scale * action, self._lo, self._hi)

        # Pushes: a sudden change of trunk velocity, at random intervals.
        info["rng"], kp1, kp2, kp3 = jax.random.split(info["rng"], 4)
        info["push_timer"] = info["push_timer"] - cfg.ctrl_dt
        push = (info["push_timer"] <= 0) & cfg.push.enable
        ang = jax.random.uniform(kp1, (), minval=0, maxval=2 * jp.pi)
        spd = jax.random.uniform(kp2, (), minval=cfg.push.speed[0], maxval=cfg.push.speed[1])
        dv = jp.where(push, spd, 0.0) * jp.array([jp.cos(ang), jp.sin(ang)])
        forced = info["steps"].astype(jp.int32) == info["forced_push_step"]
        dv = dv + jp.sum(jp.where(forced[:, None], info["forced_push"], 0.0), 0)
        data = data.replace(qvel=data.qvel.at[0:2].add(dv))
        info["push_timer"] = jp.where(push, jax.random.uniform(kp3, (), minval=cfg.push.interval[0],
                                                               maxval=cfg.push.interval[1]), info["push_timer"])

        n_sub = int(round(cfg.ctrl_dt / cfg.sim_dt))

        def sub(d, k):
            tgt = jp.where(k < info["latency"], info["last_target"], target)
            tau = self.motor_torque(d, tgt, info)
            d = mjx.step(self._mjx_model, d.replace(ctrl=tau))
            return d, tau

        data, taus = jax.lax.scan(sub, data, jp.arange(n_sub))
        i = i1  # data is now at the reference's next frame
        rq, rv, rf = self._rq[i], self._rv[i], self._rf[i]
        R = quat_to_mat(data.qpos[3:7])
        Rr = quat_to_mat(rq[3:7])
        pos_err = jp.linalg.norm(data.qpos[:3] - rq[:3])
        rot_err = rot_angle(R.T @ Rr)
        feet = data.site_xpos[self._foot_sites]
        feet_err = jp.linalg.norm(feet - rf, axis=-1)
        jerr = data.qpos[7:] - rq[7:]
        verr = data.qvel[6:] - rv[6:]
        touch = jp.array([data.sensordata[a] for a in self._touch]) > 1e-3
        fvel = jp.stack([data.sensordata[a:a + 3] for a in self._fvel])
        touchdown = jp.sum((touch & ~info["last_touch"]) * jp.sum(fvel**2, -1))
        body_contact = (data.sensordata[self._trunk_touch] > 1e-3).astype(jp.float32)

        w = cfg.reward
        r = dict(
            r_joint=w.joint * jp.exp(-jp.mean(jerr**2) / w.sigma_joint**2),
            r_base_pos=w.base_pos * jp.exp(-pos_err**2 / w.sigma_pos**2),
            r_base_rot=w.base_rot * jp.exp(-rot_err**2 / w.sigma_rot**2),
            r_feet=w.feet * jp.exp(-jp.mean(feet_err**2) / w.sigma_feet**2),
            r_joint_vel=w.joint_vel * jp.exp(-jp.mean(verr**2) / w.sigma_vel**2),
            r_torque=w.torque * jp.sum(jp.mean(taus**2, 0)),
            r_action_rate=w.action_rate * jp.sum((action - info["last_act"]) ** 2),
            r_touchdown=w.touchdown * touchdown,
            r_body_contact=w.body_contact * body_contact,
        )
        # Clipped at zero, so ending an episode early can never pay (the penalties are
        # small next to the tracking terms once the policy tracks).
        reward = jp.maximum(sum(r.values()), 0.0)

        bad = ~jp.all(jp.isfinite(data.qpos))
        done = (pos_err > cfg.term_pos) | (rot_err > cfg.term_rot) | bad
        info["last_act"] = action
        info["last_target"] = target
        info["last_touch"] = touch
        info["tau_peak"] = jp.max(jp.abs(taus), 0)
        info["tau_ms"] = jp.mean(taus**2, 0)
        info["qd"] = data.qvel[6:]
        obs = self._obs(data, info, i)
        metrics = dict(state.metrics)  # wrappers may add their own keys
        metrics.update(pos_err=pos_err, rot_err=rot_err, feet_err=jp.max(feet_err),
                       joint_err=jp.sqrt(jp.mean(jerr**2)), torque_sq=jp.sum(jp.mean(taus**2, 0)), **r)
        reward = jp.where(bad, 0.0, reward)
        return state.replace(data=data, obs=obs, reward=reward, done=done.astype(jp.float32), metrics=metrics,
                             info=info)

    # -- observations
    def _obs(self, data, info, i0):
        cfg = self._config
        nz = cfg.noise
        info["rng"], *ks = jax.random.split(info["rng"], 8)

        def noise(k, shape, scale):
            return nz.level * scale * jax.random.uniform(k, shape, minval=-1.0, maxval=1.0)

        R = quat_to_mat(data.qpos[3:7])
        pos = data.qpos[:3]
        gravity = R.T @ jp.array([0.0, 0.0, -1.0])
        gyro = data.sensordata[self._gyro:self._gyro + 3]
        linvel = R.T @ data.qvel[:3]
        q, qd = data.qpos[7:], data.qvel[6:]
        pos_meas = pos + noise(ks[0], (3,), nz.ref_pos)

        preview = []
        for ahead in (1, 5, 25):
            j = (i0 + ahead) % self.N
            rq = self._rq[j]
            Rr = quat_to_mat(rq[3:7])
            rel = R.T @ Rr
            preview += [rq[7:] - q, R.T @ (rq[:3] - pos_meas), rel[:, 0], rel[:, 2], self._rc[j]]
        actor = jp.concatenate([
            gravity + noise(ks[1], (3,), nz.gravity),
            gyro + noise(ks[2], (3,), nz.gyro),
            linvel + noise(ks[3], (3,), nz.linvel),
            q - self._rq[i0, 7:] + noise(ks[4], q.shape, nz.joint_pos),
            qd + noise(ks[5], qd.shape, nz.joint_vel),
            info["last_act"],
            *preview,
        ])
        feet = data.site_xpos[self._foot_sites]
        touch = jp.array([data.sensordata[a] for a in self._touch])
        m = self._mjx_model
        critic = jp.concatenate([
            actor, gravity, gyro, linvel, R.T @ (self._rq[i0, :3] - pos),
            (feet - self._rf[i0]).ravel(), jp.tanh(touch / 20.0), data.sensordata[self._trunk_touch:self._trunk_touch + 1] / 20.0,
            m.geom_friction[self._foot_geoms, 0], m.body_mass[self._trunk_body:self._trunk_body + 1] / 5.0,
            jp.array([info["kp_scale"], info["kd_scale"], info["strength"], info["latency"] / 3.0]),
        ])
        return {"state": actor, "privileged_state": critic}


# ---------------------------------------------------------------- domain randomization

def domain_randomize(env: StairTrack, foot_friction: float | None = None):
    """Per-env model randomization (Brax `randomization_fn`). `foot_friction` fixes the foot
    friction instead of drawing it from U(0.4, 1.0), for the friction sweep."""
    foot, skid, trunk = env._foot_geoms, env._skid_geoms, env._trunk_body
    nleg = env.action_size

    def fn(model: mjx.Model, rng: jax.Array):
        @jax.vmap
        def rand(rng):
            k = jax.random.split(rng, 8)
            friction = model.geom_friction
            mu = jax.random.uniform(k[0], minval=0.4, maxval=1.0) if foot_friction is None else foot_friction
            friction = friction.at[foot, 0].set(mu)
            friction = friction.at[skid, 0].set(jax.random.uniform(k[1], minval=0.1, maxval=0.3))
            mass = model.body_mass * jax.random.uniform(k[2], (model.nbody,), minval=0.9, maxval=1.1)
            mass = mass.at[trunk].add(jax.random.uniform(k[3], minval=-0.3, maxval=1.0))
            ipos = model.body_ipos.at[trunk].add(jax.random.uniform(k[4], (3,), minval=-0.02, maxval=0.02))
            fl = model.dof_frictionloss.at[6:].multiply(jax.random.uniform(k[5], (nleg,), minval=0.5, maxval=2.0))
            arm = model.dof_armature.at[6:].multiply(jax.random.uniform(k[6], (nleg,), minval=0.9, maxval=1.2))
            return friction, mass, ipos, fl, arm

        friction, mass, ipos, fl, arm = rand(rng)
        in_axes = jax.tree_util.tree_map(lambda x: None, model)
        fields = dict(geom_friction=friction, body_mass=mass, body_ipos=ipos, dof_frictionloss=fl,
                      dof_armature=arm)
        in_axes = in_axes.tree_replace({k: 0 for k in fields})
        return model.tree_replace(fields), in_axes

    return fn
