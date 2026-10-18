"""M0 step 3b: train the round-trip tracking policy with PPO (Brax) on MJX/Warp.

Usage:
  python scripts/rl_train.py [--steps 300e6] [--envs 4096] [--tag NAME]

Reads the latest outputs/*_rl/reference.npz. Writes into that folder:
policy_<tag>.pkl (normalizer + policy params, network sizes, config) and
train_<tag>.json (the learning curve). Then run scripts/rl_eval.py.
"""

from __future__ import annotations

import argparse
import functools
import json
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from visualize import latest  # noqa: E402

NETWORK = dict(policy_hidden_layer_sizes=(512, 256, 128), value_hidden_layer_sizes=(512, 256, 128),
               policy_obs_key="state", value_obs_key="privileged_state", init_noise_std=0.4)


def make_env(batch: int, config=None, episode_length: int = 1000):
    import json as _json

    from cleandog_sim import StairParams
    from cleandog_sim.reference import Reference
    from cleandog_sim.rl import StairTrack
    from cleandog_sim.transitions import morphology

    d = latest("*_rl/reference.npz").parent
    key = _json.loads((d / "reference.json").read_text())["morphology"]
    ref = Reference.load(d / "reference.npz")
    return StairTrack(morphology(*key), StairParams(), ref, batch, config, episode_length), d


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=float, default=300e6)
    ap.add_argument("--envs", type=int, default=4096)
    ap.add_argument("--tag", default="main")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    import jax
    from brax.training.agents.ppo import networks as ppo_networks
    from brax.training.agents.ppo import train as ppo
    from mujoco_playground import wrapper

    from cleandog_sim.rl import domain_randomize

    env, out = make_env(args.envs)
    eval_env, _ = make_env(128)
    curve = []
    t_start = time.time()

    def progress(step, metrics):
        row = dict(step=int(step), seconds=round(time.time() - t_start, 1),
                   **{k: float(v) for k, v in metrics.items() if k.startswith("eval/") and "std" not in k})
        curve.append(row)
        print(f"  {step / 1e6:6.1f}M  {row['seconds']:6.0f}s  reward {row.get('eval/episode_reward', 0):7.1f}  "
              f"length {row.get('eval/avg_episode_length', 0):6.1f}  pos_err "
              f"{row.get('eval/episode_pos_err', 0) / max(row.get('eval/avg_episode_length', 1), 1):.3f}",
              flush=True)
        (out / f"train_{args.tag}.json").write_text(json.dumps(dict(curve=curve, args=vars(args))))

    def save(step, make_policy, params):
        with open(out / f"policy_{args.tag}.pkl", "wb") as f:
            pickle.dump(dict(params=jax.device_get(params), network=NETWORK, step=int(step),
                             config=env._config.to_dict(), obs_size=env.observation_size,
                             action_size=env.action_size), f)

    print(f"training {args.steps / 1e6:.0f}M steps on {args.envs} envs -> {out}", flush=True)
    ppo.train(
        environment=env, eval_env=eval_env, num_timesteps=int(args.steps), num_envs=args.envs,
        episode_length=env.episode_length, wrap_env_fn=wrapper.wrap_for_brax_training,
        randomization_fn=domain_randomize(env), num_evals=31, num_resets_per_eval=1,
        normalize_observations=True, reward_scaling=1.0, action_repeat=1, unroll_length=20,
        num_minibatches=32, num_updates_per_batch=4, discounting=0.97, learning_rate=3e-4,
        entropy_cost=5e-3, batch_size=256, clipping_epsilon=0.2, max_grad_norm=1.0, seed=args.seed,
        num_eval_envs=128, network_factory=functools.partial(ppo_networks.make_ppo_networks, **NETWORK),
        progress_fn=progress, policy_params_fn=save,
    )
    print(f"done in {(time.time() - t_start) / 60:.0f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
