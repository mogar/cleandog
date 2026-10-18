"""M0 step 3c: evaluate the tracking policy on whole round trips.

Usage:
  python scripts/rl_eval.py [--tag main] [--trips 200]

Every trip starts at the bottom of the stairs (reference frame 0) and runs the
whole round trip without resets. A trip succeeds if the trunk never strays more
than term_pos (20 cm) or term_rot (0.6 rad) from the reference, and ends within
5 cm of the start. Conditions:

- randomized: the training distribution (friction, masses, gains, latency,
  noise, pushes);
- nominal: no randomization, noise or pushes;
- a friction sweep: foot friction fixed at 0.3 to 0.8, everything else
  randomized, pushes off.

Cross-check: the nominal trip again in CPU MuJoCo (the C library, not MJX/Warp),
at the training timestep and at half of it. Agreement between the two engines,
and across timesteps, is reported.

Writes outputs/<rl dir>/eval_<tag>.json and eval_<tag>.npz.
"""

from __future__ import annotations

import argparse
import functools
import json
import math
import os
import pickle
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rl_train import make_env  # noqa: E402

CHUNK = 500


def load_policy(path: Path, env):
    import jax
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import networks as ppo_networks

    blob = pickle.loads(path.read_bytes())
    nets = ppo_networks.make_ppo_networks(blob["obs_size"], blob["action_size"],
                                          preprocess_observations_fn=running_statistics.normalize,
                                          **blob["network"])
    policy = ppo_networks.make_inference_fn(nets)(blob["params"], deterministic=True)
    return jax.jit(policy), blob


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def mjx_trips(env, policy, n_trips: int, seed: int, randomize: bool, foot_friction=None, keep_qpos: int = 0):
    """Batched round trips in MJX. Returns per-trip arrays plus traces."""
    import jax
    from mujoco_playground._src import wrapper

    from cleandog_sim.rl import domain_randomize

    env.start_index = 0
    keys = jax.random.split(jax.random.PRNGKey(seed), n_trips)
    if randomize:
        fn = functools.partial(domain_randomize(env, foot_friction), rng=keys)
        venv = wrapper.BraxDomainRandomizationVmapWrapper(env, fn)
        reset, step = venv.reset, venv.step
    else:
        reset, step = jax.vmap(env.reset), jax.vmap(env.step)

    @functools.partial(jax.jit, static_argnums=2)
    def chunk(state, key, n):
        def body(carry, _):
            st, k = carry
            k, sk = jax.random.split(k)
            a, _ = policy(st.obs, sk)
            st = step(st, a)
            info = dict(st.info)
            info["steps"] = info["steps"] + 1
            st = st.replace(info=info)
            m = st.metrics
            out = dict(pos_err=m["pos_err"], rot_err=m["rot_err"], feet_err=m["feet_err"],
                       tau_peak=info["tau_peak"], tau_ms=info["tau_ms"], qd=info["qd"],
                       body=m["r_body_contact"] < 0)
            if keep_qpos:
                out["qpos"] = st.data.qpos[:keep_qpos]
            return (st, k), out

        (state, key), out = jax.lax.scan(body, (state, key), None, n)
        return state, key, out

    state = jax.jit(reset)(keys)
    key = jax.random.PRNGKey(seed + 1)
    traces = []
    t0 = time.time()
    for start in range(0, env.N, CHUNK):
        state, key, out = chunk(state, key, min(CHUNK, env.N - start))
        traces.append(jax.device_get(out))
    tr = {k: np.concatenate([t[k] for t in traces]) for k in traces[0]}
    cfg = env._config
    failed_at = np.full(n_trips, -1)
    bad = (tr["pos_err"] > cfg.term_pos) | (tr["rot_err"] > cfg.term_rot) | ~np.isfinite(tr["pos_err"])
    any_bad = bad.any(0)
    failed_at[any_bad] = bad.argmax(0)[any_bad]
    end_ok = tr["pos_err"][-1] < 0.05
    success = (failed_at < 0) & end_ok
    return dict(success=success, failed_at=failed_at, end_err=tr["pos_err"][-1], seconds=time.time() - t0,
                traces=tr)


def cpu_trip(env_nominal, policy, timestep: float, record_every: int = 5, n_steps: int | None = None,
             start: int = 0, push: tuple[int, np.ndarray] | None = None, log_contacts: bool = False):
    """The nominal round trip in CPU MuJoCo with the same observation code and PD law.

    `start` is the reference frame to begin at; `push` = (step, (vx, vy)) adds a shove to the
    trunk velocity at that step. With `log_contacts`, every control step records the normal
    force on each robot geom (by name) and which joints sat at their torque limit.
    """
    import jax
    import mujoco

    from cleandog_sim.rl import rl_scene_xml

    env = env_nominal
    m = mujoco.MjModel.from_xml_string(rl_scene_xml(env.p, env.s, timestep))
    d = mujoco.MjData(m)
    d.qpos[:] = env.ref.qpos[start]
    d.qpos[2] += 0.003
    d.qvel[:] = env.ref.qvel[start]
    mujoco.mj_forward(m, d)

    class D:  # the fields StairTrack._obs reads
        pass

    @jax.jit
    def obs_fn(qpos, qvel, sens, sites, last_act, rng, i0):
        dd = D()
        dd.qpos, dd.qvel, dd.sensordata, dd.site_xpos = qpos, qvel, sens, sites
        info = dict(rng=rng, last_act=last_act, kp_scale=1.0, kd_scale=1.0, strength=1.0, latency=0)
        return env._obs(dd, info, i0)

    acts_peak = np.asarray(env._peak)
    max_speed = np.asarray(env._max_speed)
    kp, kd = np.asarray(env._kp), np.asarray(env._kd)
    lo, hi = np.asarray(env._lo), np.asarray(env._hi)
    n_sub = int(round(env._config.ctrl_dt / timestep))
    last_act = np.zeros(env.action_size, np.float32)
    rng = jax.random.PRNGKey(0)
    n_steps = env.N if n_steps is None else n_steps
    pos_err = np.zeros(n_steps)
    tau_peak = np.zeros((n_steps, env.action_size))
    tau_ms = np.zeros((n_steps, env.action_size))
    qd_log = np.zeros((n_steps, env.action_size))
    saturated = np.zeros((n_steps, env.action_size), bool)
    contacts = []
    qpos_log = []
    robot_geoms = {g for g in range(m.ngeom) if m.body_rootid[m.geom_bodyid[g]] == m.body("trunk").id}
    gname = [m.geom(g).name or f"world{g}" for g in range(m.ngeom)]
    force = np.zeros(6)
    t0 = time.time()
    for i in range(n_steps):
        k = (start + i) % env.N
        if push is not None and i == push[0]:
            d.qvel[0:2] += push[1]
        obs = obs_fn(d.qpos.astype(np.float32), d.qvel.astype(np.float32), d.sensordata.astype(np.float32),
                     d.site_xpos.astype(np.float32), last_act, rng, k)
        a, _ = policy(jax.tree.map(lambda x: x[None], obs), rng)
        a = np.asarray(a[0])
        target = np.clip(env.ref.qpos[(k + 1) % env.N, 7:] + env._config.action_scale * a, lo, hi)
        taus = []
        for _ in range(n_sub):
            q, qd = d.qpos[7:], d.qvel[6:]
            tau = kp * (target - q) - kd * qd
            lim = acts_peak * np.clip(1 - np.abs(qd) / max_speed, 0, 1)
            saturated[i] |= np.abs(tau) >= lim * 0.999
            tau = np.clip(tau, -lim, lim)
            d.ctrl[:] = tau
            mujoco.mj_step(m, d)
            taus.append(tau)
        taus = np.array(taus)
        tau_peak[i], tau_ms[i], qd_log[i] = np.abs(taus).max(0), (taus**2).mean(0), d.qvel[6:]
        if log_contacts:
            step_c = {}
            for c_i in range(d.ncon):
                c = d.contact[c_i]
                g = c.geom1 if c.geom1 in robot_geoms else c.geom2
                other = c.geom2 if g == c.geom1 else c.geom1
                if g not in robot_geoms:
                    continue
                mujoco.mj_contactForce(m, d, c_i, force)
                key = f"{gname[g]}|{gname[other]}"
                step_c[key] = max(step_c.get(key, 0.0), float(force[0]))
            contacts.append(step_c)
        last_act = a.astype(np.float32)
        j = (k + 1) % env.N
        pos_err[i] = np.linalg.norm(d.qpos[:3] - env.ref.qpos[j, :3])
        if i % record_every == 0:
            qpos_log.append(d.qpos.copy())
        if not np.isfinite(pos_err[i]):
            break
    cfg = env._config
    pos_err = pos_err[:n_steps]
    failed = np.flatnonzero(pos_err > cfg.term_pos)
    return dict(pos_err=pos_err, tau_peak=tau_peak, tau_ms=tau_ms, qd=qd_log, qpos=np.array(qpos_log),
                saturated=saturated, contacts=contacts, record_every=record_every, failed_at=int(failed[0]) if len(failed) else -1,
                success=bool(len(failed) == 0 and pos_err[-1] < 0.05), seconds=time.time() - t0)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="main")
    ap.add_argument("--trips", type=int, default=200)
    ap.add_argument("--sweep-trips", type=int, default=64)
    ap.add_argument("--no-cpu", action="store_true")
    args = ap.parse_args(argv)

    from cleandog_sim.rl import default_config

    env, out = make_env(args.trips, episode_length=10**9)
    policy, blob = load_policy(out / f"policy_{args.tag}.pkl", env)
    print(f"policy {args.tag} at {blob['step'] / 1e6:.0f}M steps; round trip {env.N} control steps "
          f"({env.N * env._config.ctrl_dt:.0f} s)", flush=True)

    results, arrays = {}, {}

    def record(name, r, n):
        k = int(r["success"].sum())
        lo, hi = wilson(k, n)
        fails = r["failed_at"][r["failed_at"] >= 0]
        results[name] = dict(trips=n, success=k, rate=k / n, ci95=[lo, hi], failed_at=fails.tolist(),
                             end_err_median=float(np.median(r["end_err"])), seconds=round(r["seconds"], 1))
        print(f"  {name}: {k}/{n} = {k / n:.1%} (95% CI {lo:.1%}-{hi:.1%}), {r['seconds']:.0f} s", flush=True)

    r = mjx_trips(env, policy, args.trips, seed=1, randomize=True)
    record("randomized", r, args.trips)
    tr = r["traces"]
    # Actuator demand from the successful trips only: a failed trip flails once it has failed.
    ok = r["success"] if r["success"].any() else np.ones_like(r["success"])
    arrays.update(rand_pos_err=tr["pos_err"][::5].astype(np.float32), rand_failed_at=r["failed_at"],
                  rand_tau_peak=tr["tau_peak"][:, ok].max((0, 1)),
                  rand_tau_rms=np.sqrt(tr["tau_ms"][:, ok].mean(0)).max(0),
                  rand_qd_peak=np.abs(tr["qd"][:, ok]).max((0, 1)), rand_body=tr["body"].mean(0))
    del r, tr

    cfg = default_config()
    cfg.push.enable, cfg.noise.level, cfg.randomize_gains, cfg.latency_substeps = False, 0.0, False, (0, 0)
    env_nom, _ = make_env(4, cfg, episode_length=10**9)
    r = mjx_trips(env_nom, policy, 4, seed=2, randomize=False, keep_qpos=21)
    record("nominal", r, 4)
    tr = r["traces"]
    arrays.update(nom_pos_err=tr["pos_err"][:, 0], nom_qpos=tr["qpos"][::5, 0], nom_tau_peak=tr["tau_peak"][:, 0],
                  nom_tau_ms=tr["tau_ms"][:, 0], nom_qd=tr["qd"][:, 0])
    del r, tr

    cfg_s = default_config()
    cfg_s.push.enable = False
    env_s, _ = make_env(args.sweep_trips, cfg_s, episode_length=10**9)
    sweep = {}
    for mu in (0.3, 0.4, 0.5, 0.6, 0.8):
        r = mjx_trips(env_s, policy, args.sweep_trips, seed=10 + int(mu * 10), randomize=True, foot_friction=mu)
        record(f"friction {mu:.1f}", r, args.sweep_trips)
        sweep[mu] = results[f"friction {mu:.1f}"]
        arrays[f"sweep_failed_at_{mu:.1f}"] = r["failed_at"]
        del r

    if not args.no_cpu:
        for ts in (env._config.sim_dt, env._config.sim_dt / 2):
            c = cpu_trip(env_nom, policy, ts)
            name = f"cpu mujoco dt={ts * 1000:g} ms"
            results[name] = dict(trips=1, success=int(c["success"]), rate=float(c["success"]),
                                 failed_at=[c["failed_at"]] if c["failed_at"] >= 0 else [],
                                 end_err_median=float(c["pos_err"][-1]), seconds=round(c["seconds"], 1))
            print(f"  {name}: {'ok' if c['success'] else 'FAILED at ' + str(c['failed_at'])}, "
                  f"{c['seconds']:.0f} s", flush=True)
            tag = f"cpu{ts * 1000:g}".replace(".", "p")
            arrays.update({f"{tag}_{k}": v for k, v in c.items() if isinstance(v, np.ndarray) and k != "saturated"})

    np.savez_compressed(out / f"eval_{args.tag}.npz", **arrays)
    (out / f"eval_{args.tag}.json").write_text(json.dumps(dict(
        tag=args.tag, policy_step=blob["step"], ctrl_dt=env._config.ctrl_dt, n_steps=env.N,
        term_pos=env._config.term_pos, term_rot=env._config.term_rot, results=results,
        joints=_joint_names(env),
        segments=json.loads((out / "reference.json").read_text())["segments"])))
    print(f"-> {out / f'eval_{args.tag}.json'}")
    return 0


def _joint_names(env):
    m = env.mj_model
    return [m.joint(i).name for i in range(1, m.njnt)]


if __name__ == "__main__":
    sys.exit(main())
