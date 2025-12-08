import random
import numpy as np
import torch
from stable_baselines3 import PPO

from metadrive.envs.metadrive_env import MetaDriveEnv

# Reuse the evaluation code from random_lane_keeping
from experiments.random_lane_keeping import evaluate_policy


def make_env_config():
    """
    Common config for training & evaluation.
    You can tweak this later (e.g., add traffic).
    """
    return dict(
        use_render=False,
        manual_control=False,
        traffic_density=0.0,
        # IMPORTANT: allow multiple scenario indices (0..99)
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


def train_ppo(total_timesteps=200_000, model_path="ppo_lane_keeping_seed0", seed=0):
    """
    Train a PPO agent on the lane-keeping task.
    Saves model to model_path (without .zip extension).
    """
    print(f"[PPO] Training with seed={seed}, saving to {model_path}.zip")

    set_global_seeds(seed)
    env = make_train_env()

    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        seed=seed,          # SB3 internal seeding
        # You can tweak hyperparams later
        n_steps=2048,
        batch_size=64,
        gae_lambda=0.95,
        gamma=0.99,
        learning_rate=3e-4,
        ent_coef=0.0,
    )

    model.learn(total_timesteps=total_timesteps)
    model.save(model_path)
    env.close()
    print(f"[PPO] Saved model to {model_path}.zip")


def evaluate_ppo(
    model_path="ppo_lane_keeping_seed0",
    num_episodes=5,
    max_steps=1000,
    seed=0,
):
    """
    Load a trained PPO model and evaluate it using the evaluate_policy()
    function from random_lane_keeping.py, with CSV logging.
    """
    print(f"[PPO] Evaluating model {model_path}.zip with seed={seed}")
    set_global_seeds(seed)

    config = make_env_config()
    eval_env = MetaDriveEnv(config)
    model = PPO.load(model_path, env=eval_env)

    def ppo_policy(env, obs):
        action, _ = model.predict(obs, deterministic=True)
        return action

    log_path = f"results/ppo_lane_keeping_seed{seed}.csv"

    episodes_data, summary = evaluate_policy(
        eval_env,
        ppo_policy,
        num_episodes=num_episodes,
        max_steps=max_steps,
        algo_name="ppo",
        seed=seed,
        log_path=log_path,
    )
    eval_env.close()
    return episodes_data, summary


if __name__ == "__main__":
    import sys

    # Read seed from command line, default to 0
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    model_path = f"ppo_lane_keeping_seed{seed}"
    print(f"[MAIN] Training & evaluating PPO with seed={seed}")

    train_ppo(total_timesteps=200_000, model_path=model_path, seed=seed)
    evaluate_ppo(model_path=model_path, num_episodes=5, max_steps=1000, seed=seed)
