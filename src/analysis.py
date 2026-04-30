"""
Statistical analysis and visualization for robustness experiments.

Produces:
  1. Summary table (mean ± std across seeds)
  2. Paired t-tests with Bonferroni correction
  3. Cohen's d effect sizes
  4. Robustness–accuracy tradeoff plots
  5. Per-degradation bar charts
  6. Radar plots

Usage:
  python -m src.analysis --results-dir results --out-dir results/analysis
"""

import argparse
import os
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


BONFERRONI_K = 7  # clean + 6 degradation types
ALPHA = 0.05
ADJUSTED_ALPHA = ALPHA / BONFERRONI_K


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────
def load_all_results(results_dir: str) -> pd.DataFrame:
    """Load and concatenate all robustness CSVs."""
    csvs = glob.glob(os.path.join(results_dir, "**", "*_robustness.csv"), recursive=True)
    if not csvs:
        raise FileNotFoundError(f"No robustness CSVs found in {results_dir}")
    dfs = [pd.read_csv(f) for f in csvs]
    df = pd.concat(dfs, ignore_index=True)
    # Drop duplicates (in case of re-runs appended)
    df = df.drop_duplicates(
        subset=["model", "strategy", "deg_type_trained", "seed",
                "eval_degradation", "eval_severity"],
        keep="last"
    )
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Summary statistics
# ──────────────────────────────────────────────────────────────────────────────
def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mean ± std accuracy across seeds, grouped by model × strategy × eval condition.
    """
    grouped = df.groupby(
        ["model", "strategy", "deg_type_trained", "eval_degradation", "eval_severity"]
    )["accuracy"]
    summary = grouped.agg(["mean", "std", "count"]).reset_index()
    summary.columns = [
        "model", "strategy", "deg_type_trained",
        "eval_degradation", "eval_severity",
        "mean_acc", "std_acc", "n_seeds",
    ]
    summary["formatted"] = summary.apply(
        lambda r: f"{r['mean_acc']:.2f} ± {r['std_acc']:.2f}", axis=1
    )
    return summary


def mean_robustness(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute mean robustness = average accuracy across all 18 degradation conditions.
    Per model × strategy × seed.
    """
    degraded = df[df["eval_degradation"] != "clean"]
    rob = degraded.groupby(
        ["model", "strategy", "deg_type_trained", "seed"]
    )["accuracy"].mean().reset_index()
    rob.columns = ["model", "strategy", "deg_type_trained", "seed", "mean_robustness"]
    return rob


# ──────────────────────────────────────────────────────────────────────────────
# Statistical tests
# ──────────────────────────────────────────────────────────────────────────────
def paired_tests(
    df: pd.DataFrame,
    baseline_strategy: str = "baseline",
    target_strategy: str = "mixed",
    target_deg: str = "none",
) -> pd.DataFrame:
    """
    Paired t-tests comparing augmented vs baseline, paired by seed.
    Tests clean accuracy and per-degradation-type mean accuracy.
    Returns DataFrame with t-stat, p-value, Cohen's d, significance.
    """
    results = []

    # Get unique models
    models = df["model"].unique()

    for model_name in models:
        # Baseline data
        base = df[(df["model"] == model_name) & (df["strategy"] == baseline_strategy)]
        # Augmented data
        aug = df[
            (df["model"] == model_name)
            & (df["strategy"] == target_strategy)
            & (df["deg_type_trained"] == target_deg)
        ]

        if base.empty or aug.empty:
            continue

        # Test conditions: clean + 6 degradation types (averaged over severities)
        conditions = ["clean"] + [
            d for d in df["eval_degradation"].unique() if d != "clean"
        ]

        for cond in conditions:
            if cond == "clean":
                base_scores = base[base["eval_degradation"] == "clean"].groupby("seed")["accuracy"].mean()
                aug_scores = aug[aug["eval_degradation"] == "clean"].groupby("seed")["accuracy"].mean()
            else:
                base_scores = base[base["eval_degradation"] == cond].groupby("seed")["accuracy"].mean()
                aug_scores = aug[aug["eval_degradation"] == cond].groupby("seed")["accuracy"].mean()

            # Align by seed
            common_seeds = sorted(set(base_scores.index) & set(aug_scores.index))
            if len(common_seeds) < 2:
                continue

            b = base_scores.loc[common_seeds].values
            a = aug_scores.loc[common_seeds].values
            diff = a - b

            t_stat, p_val = stats.ttest_rel(a, b)
            cohens_d = diff.mean() / diff.std() if diff.std() > 0 else 0.0

            results.append({
                "model": model_name,
                "comparison": f"{target_strategy} vs {baseline_strategy}",
                "deg_type_trained": target_deg,
                "condition": cond,
                "baseline_mean": b.mean(),
                "augmented_mean": a.mean(),
                "diff_mean": diff.mean(),
                "t_stat": t_stat,
                "p_value": p_val,
                "cohens_d": cohens_d,
                "significant": p_val < ADJUSTED_ALPHA,
            })

    return pd.DataFrame(results)


# ──────────────────────────────────────────────────────────────────────────────
# Visualization
# ──────────────────────────────────────────────────────────────────────────────
def plot_tradeoff(df: pd.DataFrame, out_dir: str):
    """
    Robustness–accuracy tradeoff scatter plot.
    X = clean accuracy, Y = mean robustness (avg over all degraded conditions).
    One point per (model, strategy, seed).
    """
    clean = df[df["eval_degradation"] == "clean"][
        ["model", "strategy", "deg_type_trained", "seed", "accuracy"]
    ].rename(columns={"accuracy": "clean_acc"})

    rob = mean_robustness(df)
    merged = clean.merge(rob, on=["model", "strategy", "deg_type_trained", "seed"])

    fig, ax = plt.subplots(figsize=(10, 7))
    strategies = merged["strategy"].unique()
    markers = {
        "baseline": "o", "mixed": "s", "single": "^", "curriculum": "D",
        "augmix": "P", "augmix_nojsd": "X", "curriculum_capped": "v",
        "curriculum_cosine": "h", "curriculum_clean50": "*",
    }
    colors = {"resnet18": "tab:blue", "mobilenetv2": "tab:orange"}

    for strat in strategies:
        for model_name in merged["model"].unique():
            subset = merged[(merged["strategy"] == strat) & (merged["model"] == model_name)]
            label = f"{model_name} / {strat}"
            if strat in ("single", "curriculum"):
                for deg in subset["deg_type_trained"].unique():
                    sub2 = subset[subset["deg_type_trained"] == deg]
                    ax.scatter(
                        sub2["clean_acc"], sub2["mean_robustness"],
                        marker=markers.get(strat, "x"),
                        color=colors.get(model_name, "grey"),
                        alpha=0.7, s=60,
                        label=f"{model_name}/{strat}/{deg}",
                    )
            else:
                ax.scatter(
                    subset["clean_acc"], subset["mean_robustness"],
                    marker=markers.get(strat, "x"),
                    color=colors.get(model_name, "grey"),
                    alpha=0.7, s=60,
                    label=label,
                )

    ax.set_xlabel("Clean Accuracy (%)")
    ax.set_ylabel("Mean Robustness (%)")
    ax.set_title("Robustness–Accuracy Tradeoff")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "tradeoff_scatter.png"), dpi=150)
    plt.close()
    print(f"Saved tradeoff_scatter.png")


def plot_degradation_bars(df: pd.DataFrame, out_dir: str):
    """Bar chart of accuracy per degradation type (averaged over severities and seeds)."""
    degraded = df[df["eval_degradation"] != "clean"].copy()
    agg = degraded.groupby(
        ["model", "strategy", "deg_type_trained", "eval_degradation"]
    )["accuracy"].mean().reset_index()

    # Create label column
    agg["label"] = agg.apply(
        lambda r: f"{r['strategy']}" + (f"/{r['deg_type_trained']}" if r["deg_type_trained"] != "none" else ""),
        axis=1,
    )

    for model_name in agg["model"].unique():
        sub = agg[agg["model"] == model_name]
        fig, ax = plt.subplots(figsize=(12, 6))
        pivot = sub.pivot_table(
            index="eval_degradation", columns="label", values="accuracy"
        )
        pivot.plot(kind="bar", ax=ax, width=0.8)
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(f"{model_name} — Per-Degradation Accuracy")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{model_name}_degradation_bars.png"), dpi=150)
        plt.close()
        print(f"Saved {model_name}_degradation_bars.png")


def plot_radar(df: pd.DataFrame, out_dir: str):
    """Radar plot of per-degradation mean accuracy for each strategy."""
    degraded = df[df["eval_degradation"] != "clean"].copy()
    agg = degraded.groupby(
        ["model", "strategy", "deg_type_trained", "eval_degradation"]
    )["accuracy"].mean().reset_index()

    deg_types = sorted(agg["eval_degradation"].unique())
    n = len(deg_types)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    for model_name in agg["model"].unique():
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        sub = agg[agg["model"] == model_name]

        labels = sub.apply(
            lambda r: f"{r['strategy']}" + (f"/{r['deg_type_trained']}" if r["deg_type_trained"] != "none" else ""),
            axis=1,
        )
        sub = sub.copy()
        sub["label"] = labels

        for label in sub["label"].unique():
            grp = sub[sub["label"] == label]
            values = []
            for dt in deg_types:
                row = grp[grp["eval_degradation"] == dt]
                values.append(row["accuracy"].values[0] if len(row) > 0 else 0)
            values += values[:1]
            ax.plot(angles, values, label=label, linewidth=1.5)
            ax.fill(angles, values, alpha=0.1)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(deg_types, size=8)
        ax.set_title(f"{model_name} — Robustness Radar", pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=7)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{model_name}_radar.png"), dpi=150)
        plt.close()
        print(f"Saved {model_name}_radar.png")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Analysis of robustness experiments")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--out-dir", default="results/analysis")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading results...")
    df = load_all_results(args.results_dir)
    print(f"Loaded {len(df)} rows")

    # Summary table
    summary = summary_table(df)
    summary.to_csv(os.path.join(args.out_dir, "summary_table.csv"), index=False)
    print(f"\nSummary table saved ({len(summary)} rows)")

    # Statistical tests: compare each augmented strategy to baseline
    all_tests = []
    strategies_to_test = []

    # Auto-discover all non-baseline strategies
    for strat in df["strategy"].unique():
        if strat == "baseline":
            continue
        for deg in df[df["strategy"] == strat]["deg_type_trained"].unique():
            strategies_to_test.append((strat, deg))

    for strat, deg in strategies_to_test:
        test_results = paired_tests(df, "baseline", strat, deg)
        all_tests.append(test_results)

    if all_tests:
        tests_df = pd.concat(all_tests, ignore_index=True)
        tests_df.to_csv(os.path.join(args.out_dir, "statistical_tests.csv"), index=False)
        print(f"Statistical tests saved ({len(tests_df)} comparisons)")

        # Print significant results
        sig = tests_df[tests_df["significant"]]
        if len(sig) > 0:
            print(f"\nSignificant results (p < {ADJUSTED_ALPHA:.4f}):")
            for _, row in sig.iterrows():
                direction = "+" if row["diff_mean"] > 0 else ""
                print(
                    f"  {row['model']} | {row['comparison']} | {row['condition']}: "
                    f"{direction}{row['diff_mean']:.2f}% (d={row['cohens_d']:.2f}, p={row['p_value']:.4f})"
                )

    # Plots
    print("\nGenerating plots...")
    plot_tradeoff(df, args.out_dir)
    plot_degradation_bars(df, args.out_dir)
    plot_radar(df, args.out_dir)

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
