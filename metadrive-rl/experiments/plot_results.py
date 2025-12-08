import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = "results/"


def load_all_results():
    """Load all CSVs into one DataFrame."""
    dfs = []
    for fname in os.listdir(RESULTS_DIR):
        if fname.endswith(".csv"):
            df = pd.read_csv(os.path.join(RESULTS_DIR, fname))
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def aggregate_results(df):
    """
    Compute mean and std across seeds for each algorithm.
    Returns aggregated DataFrame:
    algo | metric | mean | std
    """
    grouped = df.groupby("algo").agg({
        "return": ["mean", "std"],
        "length": ["mean", "std"],
        "collisions": ["mean", "std"],
    })

    grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]
    grouped = grouped.reset_index()
    return grouped


def plot_bar(ax, x, y, yerr, title, ylabel):
    ax.bar(x, y, yerr=yerr, capsize=5)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(["Random", "PPO", "SAC"])


def main():
    df = load_all_results()
    agg = aggregate_results(df)

    # Order algorithms
    algo_order = ["random", "ppo", "sac"]
    agg = agg.set_index("algo").loc[algo_order].reset_index()

    # Build plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Episode Return
    plot_bar(
        axes[0],
        range(len(algo_order)),
        agg["return_mean"],
        agg["return_std"],
        "Episode Return",
        "Return",
    )

    # Episode Length
    plot_bar(
        axes[1],
        range(len(algo_order)),
        agg["length_mean"],
        agg["length_std"],
        "Episode Length",
        "Length (steps)",
    )

    # Collision Rate
    plot_bar(
        axes[2],
        range(len(algo_order)),
        agg["collisions_mean"],
        agg["collisions_std"],
        "Collision Rate",
        "Collisions per episode",
    )

    plt.tight_layout()
    plt.savefig("results/summary_plots.png")
    plt.show()


if __name__ == "__main__":
    main()
