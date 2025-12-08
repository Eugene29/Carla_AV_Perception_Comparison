from experiments.random_lane_keeping import run_random_policy
from experiments.ppo_lane_keeping import train_ppo, evaluate_ppo


def run_random_multi_seed(seeds=(0, 1, 2), num_episodes=5, max_steps=1000):
    for seed in seeds:
        print(f"\n=== Running RANDOM policy, seed={seed} ===")
        run_random_policy(num_episodes=num_episodes, max_steps=max_steps, seed=seed)


def run_ppo_multi_seed(
    seeds=(0, 1, 2),
    total_timesteps=200_000,
    num_episodes=5,
    max_steps=1000,
):
    for seed in seeds:
        print(f"\n=== Training PPO, seed={seed} ===")
        model_path = f"ppo_lane_keeping_seed{seed}"
        train_ppo(total_timesteps=total_timesteps, model_path=model_path, seed=seed)

        print(f"\n=== Evaluating PPO, seed={seed} ===")
        evaluate_ppo(
            model_path=model_path,
            num_episodes=num_episodes,
            max_steps=max_steps,
            seed=seed,
        )


if __name__ == "__main__":
    seeds = (0, 1, 2)

    # 1) Random policy across seeds (just for completeness / baseline)
    run_random_multi_seed(seeds=seeds, num_episodes=5, max_steps=1000)

    # 2) PPO across seeds
    run_ppo_multi_seed(
        seeds=seeds,
        total_timesteps=200_000,
        num_episodes=5,
        max_steps=1000,
    )
