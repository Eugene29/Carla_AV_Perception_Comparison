import random
import sys

import numpy as np
import torch
from stable_baselines3 import SAC
from metadrive.envs.metadrive_env import MetaDriveEnv

from experiments.random_lane_keeping import evaluate_policy


def make_env_config():
    """
    Same config as PPO so comparison is fair.
    """
    return dict(
        use_render=False,
        manual_control=False,
        traffic_density=0.0,
        # match PPO: allow multiple scenarios so seeds change layouts
        num_scenarios=100,
        start_seed=0,
        map="O",
        horizon=1000,
    )


def make_train_env():
    config = make_env_config()
    return MetaDriveEnv(config)


def set_global_seeds(seed: int):
    """Set Python, NumPy, and Torch seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_sac(total_timesteps=200_000, model_path="sac_lane_keeping_seed0", seed=0):
    """
    Train a SAC agent on the lane-keeping task.
    Saves model to model_path (without .zip extension).
    """
    print(f"[SAC] Training with seed={seed}, saving to {model_path}.zip")

    set_global_seeds(seed)
    env = make_train_env()

    model = SAC(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        learning_rate=3e-4,
        buffer_size=100_000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        learning_starts=1_000,
        seed=seed,  # SB3 internal seeding
    )

    model.learn(total_timesteps=total_timesteps)
    model.save(model_path)
    env.close()
    print(f"[SAC] Saved model to {model_path}.zip")


def evaluate_sac(
    model_path="sac_lane_keeping_seed0",
    num_episodes=5,
    max_steps=1000,
    seed=0,
):
    """
    Load a trained SAC model and evaluate it with evaluate_policy(),
    logging to CSV just like PPO and random.
    """
    print(f"[SAC] Evaluating model {model_path}.zip with seed={seed}")
    set_global_seeds(seed)

    config = make_env_config()
    eval_env = MetaDriveEnv(config)
    model = SAC.load(model_path, env=eval_env)

    def sac_policy(env, obs):
        action, _ = model.predict(obs, deterministic=True)
        return action

    log_path = f"results/sac_lane_keeping_seed{seed}.csv"

    episodes_data, summary = evaluate_policy(
        eval_env,
        sac_policy,
        num_episodes=num_episodes,
        max_steps=max_steps,
        algo_name="sac",
        seed=seed,
        log_path=log_path,
    )
    eval_env.close()
    return episodes_data, summary


if __name__ == "__main__":
    # Read seed from command line, default to 0
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    model_path = f"sac_lane_keeping_seed{seed}"
    print(f"[MAIN] Training & evaluating SAC with seed={seed}")

    train_sac(total_timesteps=200_000, model_path=model_path, seed=seed)
    evaluate_sac(model_path=model_path, num_episodes=5, max_steps=1000, seed=seed)
