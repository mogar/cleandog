"""M0 step 3d: where along the round trip does a shove make the policy fail, and why?

Usage:
  python scripts/rl_diagnose.py [--tag main] [--every 0.5] [--speeds 0.15 0.25]

1. **Shove map (MJX).** A nominal robot runs from 2 s before each moment on a grid
   along the round trip. At that moment it gets one shove (each speed, in each of
   8 directions), then runs 5 s more. A trial fails by the same rule as the
   round-trip evaluation (trunk more than 20 cm or 0.6 rad from the reference).
2. **Forensics (CPU MuJoCo).** The worst moments are replayed with every contact
   force and every torque-limit hit logged, so the failure can be read off.

Writes outputs/<rl dir>/diagnose_<tag>.npz and diagnose_<tag>.json. Then run
`python scripts/visualize.py rl_failures`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rl_eval import cpu_trip, load_policy  # noqa: E402
from rl_train import make_env  # noqa: E402

LEAD, AFTER = 100, 250  # control steps before and after the shove (2 s, 5 s)
N_DIRS = 8
BATCH = 4096


def nominal_config():
    from cleandog_sim.rl import default_config

    cfg = default_config()
    cfg.push.enable, cfg.noise.level, cfg.randomize_gains, cfg.latency_substeps = False, 0.0, False, (0, 0)
    return cfg


def shove_map(env, policy, push_idx: np.ndarray, speeds, seed: int = 0):
    """One trial per (push frame, speed, direction). Returns dict of arrays over trials."""
    import jax
    import jax.numpy as jp

    from cleandog_sim.rl import N_FORCED

    angles = np.arange(N_DIRS) * 2 * np.pi / N_DIRS
    trials = [(i, v, a) for i in push_idx for v in speeds for a in angles]
    reset = jax.jit(jax.vmap(env.reset_at))
    step = jax.vmap(env.step)
    cfg = env._config

    @jax.jit
    def run(state, key):
        def body(carry, _):
            st, k = carry
            k, sk = jax.random.split(k)
            a, _ = policy(st.obs, sk)
            st = step(st, a)
            info = dict(st.info)
            info["steps"] = info["steps"] + 1
            st = st.replace(info=info)
            return (st, k), (st.metrics["pos_err"], st.metrics["rot_err"], jp.any(info["tau_peak"] > 0.98 * env._peak, -1))

        (_, _), out = jax.lax.scan(body, (state, key), None, LEAD + AFTER)
        return out

    res = {k: [] for k in ("frame", "speed", "angle", "failed", "pre_failed", "peak_err", "fail_step", "sat_frac")}
    for b0 in range(0, len(trials), BATCH):
        chunk = trials[b0:b0 + BATCH]
        n = len(chunk)
        pad = BATCH - n  # keep one compiled shape
        fr = np.array([c[0] for c in chunk] + [chunk[0][0]] * pad)
        sp = np.array([c[1] for c in chunk] + [0.0] * pad)
        an = np.array([c[2] for c in chunk] + [0.0] * pad)
        keys = jax.random.split(jax.random.PRNGKey(seed + b0), BATCH)
        st = reset(keys, jp.array(fr - LEAD))
        info = dict(st.info)
        steps = np.full((BATCH, N_FORCED), -1)
        steps[:, 0] = LEAD
        pushes = np.zeros((BATCH, N_FORCED, 2))
        pushes[:, 0] = np.stack([sp * np.cos(an), sp * np.sin(an)], 1)
        info["forced_push_step"] = jp.array(steps, jp.int32)
        info["forced_push"] = jp.array(pushes, jp.float32)
        st = st.replace(info=info)
        pos, rot, sat = (np.asarray(x)[:, :n] for x in run(st, jax.random.PRNGKey(seed + 7 + b0)))
        bad = (pos > cfg.term_pos) | (rot > cfg.term_rot) | ~np.isfinite(pos)
        res["frame"] += fr[:n].tolist()
        res["speed"] += sp[:n].tolist()
        res["angle"] += an[:n].tolist()
        res["pre_failed"] += bad[:LEAD].any(0).tolist()
        res["failed"] += bad[LEAD:].any(0).tolist()
        res["fail_step"] += np.where(bad.any(0), bad.argmax(0) - LEAD, 10**6).tolist()
        res["peak_err"] += np.nan_to_num(pos[LEAD:], nan=1.0).max(0).tolist()
        res["sat_frac"] += sat[LEAD:].mean(0).tolist()
        print(f"  {b0 + n}/{len(trials)} trials, {np.mean(res['failed']):.1%} failed so far", flush=True)
    return {k: np.array(v) for k, v in res.items()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="main")
    ap.add_argument("--every", type=float, default=0.5, help="push-time grid spacing (s)")
    ap.add_argument("--speeds", type=float, nargs="+", default=[0.15, 0.25])
    ap.add_argument("--forensics", type=int, default=4, help="worst moments to replay on CPU")
    args = ap.parse_args(argv)

    env, out = make_env(BATCH, nominal_config(), episode_length=10**9)
    policy, blob = load_policy(out / f"policy_{args.tag}.pkl", env)
    dt = env._config.ctrl_dt
    step = int(round(args.every / dt))
    push_idx = np.arange(LEAD, env.N - AFTER, step)
    print(f"shove map: {len(push_idx)} moments x {len(args.speeds)} speeds x {N_DIRS} directions", flush=True)
    t0 = time.time()
    sm = shove_map(env, policy, push_idx, args.speeds)
    print(f"  done in {time.time() - t0:.0f} s; {sm['pre_failed'].sum()} trials failed before the shove", flush=True)

    # Worst moments at the largest speed: highest failure fraction, then highest error.
    vmax = max(args.speeds)
    sel = sm["speed"] == vmax
    frames = sm["frame"][sel].reshape(-1, N_DIRS)[:, 0]
    frac = sm["failed"][sel].reshape(-1, N_DIRS).mean(1)
    peak = sm["peak_err"][sel].reshape(-1, N_DIRS).max(1)
    order = np.lexsort((-peak, -frac))
    worst = []
    for i in order:  # one per cluster: at least 5 s apart
        if all(abs(frames[i] - frames[j]) * dt > 5 for j in worst):
            worst.append(i)
        if len(worst) == args.forensics:
            break

    forensics = []
    for i in worst:
        f = int(frames[i])
        rows = np.flatnonzero(sel & (sm["frame"] == f))
        r = rows[np.argmax(sm["peak_err"][rows] + 10 * sm["failed"][rows])]
        ang = float(sm["angle"][r])
        vec = vmax * np.array([np.cos(ang), np.sin(ang)])
        c = cpu_trip(env, policy, env._config.sim_dt, record_every=1, n_steps=LEAD + AFTER, start=f - LEAD,
                     push=(LEAD, vec), log_contacts=True)
        names = sorted({k for step_c in c["contacts"] for k in step_c})
        forces = np.array([[sc.get(k, 0.0) for k in names] for sc in c["contacts"]])
        forensics.append(dict(frame=f, time=f * dt, angle=ang, speed=vmax, fail_frac=float(frac[i]),
                              cpu_failed_at=c["failed_at"], contact_names=names))
        np.savez_compressed(out / f"diagnose_{args.tag}_forensic{len(forensics) - 1}.npz", pos_err=c["pos_err"],
                            qpos=c["qpos"], saturated=c["saturated"], forces=forces, tau_peak=c["tau_peak"])
        print(f"  forensic t={f * dt:.1f} s, shove {np.degrees(ang):.0f} deg: MJX {frac[i]:.0%} of directions fail; "
              f"CPU {'fails at +' + str(c['failed_at'] - LEAD) + ' steps' if c['failed_at'] >= 0 else 'recovers'}",
              flush=True)

    np.savez_compressed(out / f"diagnose_{args.tag}.npz", **sm)
    (out / f"diagnose_{args.tag}.json").write_text(json.dumps(dict(
        tag=args.tag, policy_step=blob["step"], dt=dt, lead=LEAD, after=AFTER, n_dirs=N_DIRS,
        speeds=args.speeds, every=args.every, forensics=forensics)))
    print(f"-> {out / f'diagnose_{args.tag}.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
