import os
import csv
import numpy as np
from metadrive.envs.metadrive_env import MetaDriveEnv


def make_env():
    # Simple single-agent driving env for lane-keeping
    config = dict(
        use_render=False,       # change to True if you want to watch it
        manual_control=False,
        traffic_density=0.0,    # start with no other cars: pure lane-keeping
        num_scenarios=1,
        start_seed=0,
        map="O",                # simple oval map; we can adjust later
    )
    return MetaDriveEnv(config)


def evaluate_policy(
    env,
    policy_fn,
    num_episodes=5,
    max_steps=1000,
    algo_name=None,
    seed=None,
    log_path=None,
):
    """
    env: MetaDriveEnv instance
    policy_fn: function (env, obs) -> action
    algo_name: string identifying the algorithm (e.g., "random", "ppo")
    seed: optional integer seed for logging
    log_path: if not None, save per-episode metrics to this CSV file

    Returns:
        episodes_data: list of dicts, one per episode
        summary: dict with mean/std of metrics
    """
    all_returns = []
    all_lengths = []
    all_collisions = []

    episodes_data = []

    for ep in range(num_episodes):
        obs, info = env.reset()
        terminated = False
        truncated = False
        ep_return = 0.0
        ep_length = 0
        ep_collisions = 0

        while not (terminated or truncated) and ep_length < max_steps:
            action = policy_fn(env, obs)
            obs, reward, terminated, truncated, info = env.step(action)

            ep_return += reward
            ep_length += 1

            # print info keys on first step of first episode for debugging
            if ep == 0 and ep_length == 1:
                print("Info keys example:", list(info.keys()))

            if ("crash_vehicle" in info and info["crash_vehicle"]) or \
               ("crash_object" in info and info["crash_object"]) or \
               ("crash_building" in info and info["crash_building"]):
                ep_collisions += 1

        all_returns.append(ep_return)
        all_lengths.append(ep_length)
        all_collisions.append(ep_collisions)

        ep_record = {
            "algo": algo_name if algo_name is not None else "",
            "seed": seed if seed is not None else "",
            "episode": ep,
            "return": ep_return,
            "length": ep_length,
            "collisions": ep_collisions,
        }
        episodes_data.append(ep_record)

        print(f"Episode {ep+1}: return={ep_return:.2f}, length={ep_length}, collisions={ep_collisions}")

    summary = {
        "return_mean": float(np.mean(all_returns)),
        "return_std": float(np.std(all_returns)),
        "length_mean": float(np.mean(all_lengths)),
        "length_std": float(np.std(all_lengths)),
        "collisions_mean": float(np.mean(all_collisions)),
    }

    print("\n=== Summary over", num_episodes, "episodes ===")
    print(f"Avg return: {summary['return_mean']:.2f} ± {summary['return_std']:.2f}")
    print(f"Avg length: {summary['length_mean']:.1f} ± {summary['length_std']:.1f}")
    print(f"Avg collisions: {summary['collisions_mean']:.2f}")

    # Optional: save to CSV
    if log_path is not None:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["algo", "seed", "episode", "return", "length", "collisions"],
            )
            writer.writeheader()
            for row in episodes_data:
                writer.writerow(row)
        print(f"Saved episode metrics to {log_path}")

    return episodes_data, summary


def random_policy(env, obs):
    """Baseline random policy: just sample from env.action_space."""
    return env.action_space.sample()


def run_random_policy(num_episodes=5, max_steps=1000, seed=0):
    env = make_env()
    log_path = f"results/random_lane_keeping_seed{seed}.csv"
    episodes_data, summary = evaluate_policy(
        env,
        random_policy,
        num_episodes=num_episodes,
        max_steps=max_steps,
        algo_name="random",
        seed=seed,
        log_path=log_path,
    )
    env.close()
    return episodes_data, summary


if __name__ == "__main__":
    import sys

    # Read seed from command line, default to 0
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    print(f"[RANDOM] Running random policy with seed={seed}")
    run_random_policy(num_episodes=5, max_steps=1000, seed=seed)