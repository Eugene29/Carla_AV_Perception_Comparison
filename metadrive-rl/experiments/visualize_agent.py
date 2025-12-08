import sys
import time
import numpy as np

from stable_baselines3 import PPO, SAC
from metadrive.envs.metadrive_env import MetaDriveEnv


def make_render_env(seed: int = 0):
    """
    Environment config for visualization: enable rendering and relax some
    termination conditions so we can watch behavior even if the car goes off-road.
    """
    config = dict(
        use_render=True,          # show Panda3D window
        manual_control=False,     # agent controls the car
        traffic_density=0.0,
        num_scenarios=100,        # same as training, but you can set to 1 for determinism
        start_seed=0,
        map="O",
        horizon=1000,
        window_size=(1200, 800),

        # Relax done conditions for visualization (training can still use defaults)
        out_of_road_done=False,
        out_of_route_done=False,
        on_continuous_line_done=False,
        on_broken_line_done=False,
        crash_vehicle_done=False,
        crash_object_done=False,
        crash_human_done=False,
    )
    env = MetaDriveEnv(config)
    obs, info = env.reset(seed=seed)
    return env, obs


def run_random(seed: int = 0, max_steps: int = 1000):
    print(f"[VIS] Running RANDOM policy with seed={seed}")
    env, obs = make_render_env(seed)
    terminated = False
    truncated = False
    step = 0
    last_info = {}

    while not (terminated or truncated) and step < max_steps:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        last_info = info
        step += 1
        time.sleep(0.02)

    print("Episode finished after", step, "steps.")
    print("Last info dict:", last_info)
    print("Key termination flags:",
          "crash =", last_info.get("crash"),
          "out_of_road =", last_info.get("out_of_road"),
          "arrive_dest =", last_info.get("arrive_dest"),
          "max_step =", last_info.get("max_step"))
    print("Press ENTER to close the window.")
    input()
    env.close()
    print("[VIS] Finished RANDOM run.")


def run_ppo(seed: int = 0, model_seed: int = 0, max_steps: int = 1000):
    """
    Visualize a trained PPO agent.
    `model_seed` chooses which trained model to load (ppo_lane_keeping_seedX.zip).
    `seed` controls the evaluation environment scenario.
    """
    model_path = f"ppo_lane_keeping_seed{model_seed}"
    print(f"[VIS] Running PPO from {model_path}.zip on env seed={seed}")

    env, obs = make_render_env(seed)
    model = PPO.load(model_path, env=env)

    terminated = False
    truncated = False
    step = 0
    last_info = {}

    # Action smoothing
    last_action = np.zeros(env.action_space.shape, dtype=np.float32)
    alpha = 0.7  # 0 = no smoothing, closer to 1 = more smoothing

    while not (terminated or truncated) and step < max_steps:
        raw_action, _ = model.predict(obs, deterministic=True)
        raw_action = np.array(raw_action, dtype=np.float32)

        # Low-pass filter (smooth changes)
        smoothed_action = alpha * last_action + (1 - alpha) * raw_action

        # Conservative clipping to keep behavior gentle
        smoothed_action[0] = np.clip(smoothed_action[0], -0.5, 0.5)  # steering
        smoothed_action[1] = np.clip(smoothed_action[1],  -0.2,  0.5)  # throttle only

        #action[0] = np.clip(action[0], -0.5, 0.5)  # steering
        #action[1] = np.clip(action[1], -0.2, 0.5)  # throttle/brake

        last_action = smoothed_action

        obs, reward, terminated, truncated, info = env.step(smoothed_action)
        last_info = info
        step += 1
        time.sleep(0.02)

    print("Episode finished after", step, "steps.")
    print("Last info dict:", last_info)
    print("Key termination flags:",
          "crash =", last_info.get("crash"),
          "out_of_road =", last_info.get("out_of_road"),
          "arrive_dest =", last_info.get("arrive_dest"),
          "max_step =", last_info.get("max_step"))
    print("Press ENTER to close the window.")
    input()
    env.close()
    print("[VIS] Finished PPO run.")


def run_sac(seed: int = 0, model_seed: int = 0, max_steps: int = 1000):
    """
    Visualize a trained SAC agent.
    `model_seed` chooses which trained model to load (sac_lane_keeping_seedX.zip).
    `seed` controls the evaluation environment scenario.
    """
    model_path = f"sac_lane_keeping_seed{model_seed}"
    print(f"[VIS] Running SAC from {model_path}.zip on env seed={seed}")

    env, obs = make_render_env(seed)
    model = SAC.load(model_path, env=env)

    terminated = False
    truncated = False
    step = 0
    last_info = {}

    # Action smoothing (same as PPO)
    last_action = np.zeros(env.action_space.shape, dtype=np.float32)
    alpha = 0.7

    while not (terminated or truncated) and step < max_steps:
        raw_action, _ = model.predict(obs, deterministic=True)
        raw_action = np.array(raw_action, dtype=np.float32)

        smoothed_action = alpha * last_action + (1 - alpha) * raw_action
        smoothed_action[0] = np.clip(smoothed_action[0], -0.5, 0.5)
        smoothed_action[1] = np.clip(smoothed_action[1],  0.0,  0.4)

        last_action = smoothed_action

        obs, reward, terminated, truncated, info = env.step(smoothed_action)
        last_info = info
        step += 1
        time.sleep(0.02)

        # ✅ Stop immediately after a collision
        if info.get("crash", False):
            print("[VIS] Collision detected, stopping SAC episode.")
            break

    print("Episode finished after", step, "steps.")
    print("Last info dict:", last_info)
    print("Key termination flags:",
          "crash =", last_info.get("crash"),
          "out_of_road =", last_info.get("out_of_road"),
          "arrive_dest =", last_info.get("arrive_dest"),
          "max_step =", last_info.get("max_step"))
    print("Press ENTER to close the window.")
    input()
    env.close()
    print("[VIS] Finished SAC run.")


if __name__ == "__main__":
    """
    Usage:
        python experiments/visualize_agent.py random
        python experiments/visualize_agent.py ppo 0
        python experiments/visualize_agent.py sac 2

    First arg: algo = random | ppo | sac
    Second arg (optional): model seed (for PPO/SAC); default 0
    """
    if len(sys.argv) < 2:
        print("Usage: python experiments/visualize_agent.py [random|ppo|sac] [model_seed]")
        sys.exit(0)

    algo = sys.argv[1].lower()
    model_seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0

    if algo == "random":
        run_random(seed=0)
    elif algo == "ppo":
        run_ppo(seed=0, model_seed=model_seed)
    elif algo == "sac":
        run_sac(seed=0, model_seed=model_seed)
    else:
        print("Unknown algo. Use: random | ppo | sac")
