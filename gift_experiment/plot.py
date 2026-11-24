import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
import numpy as np

COLORS = {"optimizer": "#0072B2", "rl": "#D55E00"}
LABELS = {"optimizer": "Optimizer", "rl": "RL"}
STAGE_LABELS = {
    1: "1. Static\npoint estimates",
    2: "2. Static\nuncertain volumes",
    3: "3. Sequential\noffers",
}


def style_axis(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#6b7280")
    ax.spines["bottom"].set_color("#6b7280")
    ax.tick_params(colors="#374151", labelsize=10)
    ax.grid(axis="y", color="#d1d5db", linewidth=0.8, alpha=0.6)
    ax.set_axisbelow(True)


def format_log_axis(ax) -> None:
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())


def episode_distributions(result: dict) -> tuple[np.ndarray, np.ndarray]:
    records = result["episode_results"]
    if "episode_decision_times_ms" not in result:
        raise ValueError(
            "Results do not include per-episode timings; rerun the experiment before plotting."
        )
    values = np.asarray([record["value"] if record["fits"] else 0 for record in records])
    times = np.asarray(result["episode_decision_times_ms"], dtype=float)
    if not len(values) or len(values) != len(times):
        raise ValueError("Each episode must have one gift value and one timing.")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(times)) or np.any(times <= 0):
        raise ValueError("Episode values must be finite and episode times must be finite and positive.")
    return values, times


def plot_experiment(data: dict, output: Path) -> None:
    stages = data["stages"]
    if not stages:
        raise ValueError("experiment has no stages to plot")
    positions = np.arange(len(stages), dtype=float)
    methods = ("optimizer", "rl")
    distributions = {
        (index, method): episode_distributions(stage["methods"][method])
        for index, stage in enumerate(stages)
        for method in methods
        if method in stage["methods"]
    }
    if not distributions:
        raise ValueError("No optimizer or RL results to plot.")
    time_values = [time for _, times in distributions.values() for time in times]
    positive_bounds = [
        float(times.mean() - times.std(ddof=1))
        for _, times in distributions.values()
        if len(times) > 1 and times.mean() > times.std(ddof=1)
    ]
    time_floor = min(time_values + positive_bounds) / 2
    fig, (value_ax, time_ax) = plt.subplots(1, 2, figsize=(15, 7), layout="constrained")
    width = 0.28
    clipped_time_error = False
    for index, stage in enumerate(stages):
        for offset, method in enumerate(methods):
            if method not in stage["methods"]:
                continue
            values, times = distributions[index, method]
            x = positions[index] + (offset - 0.5) * width
            for ax, samples in ((value_ax, values), (time_ax, times)):
                mean = float(samples.mean())
                variance = float(np.var(samples, ddof=1)) if len(samples) > 1 else None
                deviation = np.sqrt(variance) if variance is not None else None
                bottom = time_floor if ax is time_ax else 0
                ax.bar(x, mean - bottom, width=width * 0.85, bottom=bottom,
                       color=COLORS[method], alpha=0.85)
                if deviation is not None:
                    lower = deviation
                    if ax is time_ax and mean - deviation <= time_floor:
                        lower = mean - time_floor
                        clipped_time_error = True
                        ax.scatter(x, time_floor, marker="v", color="#111827", s=25,
                                   zorder=4, clip_on=False)
                    ax.errorbar(
                        x, mean, yerr=[[lower], [deviation]], fmt="none",
                        ecolor="#111827", capsize=3, linewidth=1.1,
                    )
                ax.annotate(
                    f"{mean:.3g}", (x, mean), xytext=(0, -12 if mean else 4),
                    textcoords="offset points", ha="center", fontsize=9,
                    color="white" if mean else "#374151", fontweight="bold",
                )
    for ax in (value_ax, time_ax):
        ax.set_xlabel("Problem setting", labelpad=10)
        ax.set_xticks(positions, [
            STAGE_LABELS.get(stage["stage"], f"Stage {stage['stage']}") for stage in stages
        ])
        ax.set_xlim(-0.5, len(stages) - 0.5)
        style_axis(ax)
        ax.margins(y=0.2)
    value_ax.set_title("Gift value", fontweight="bold", loc="left", pad=12)
    value_ax.set_ylabel("Mean feasible value (price units)\nOver 40 L counts as zero", labelpad=8)
    handles = [
        Patch(facecolor=COLORS[method], alpha=0.75, label=LABELS[method]) for method in methods
    ]
    value_ax.legend(handles=handles, loc="upper right", frameon=False)
    time_ax.set_title("Decision time", fontweight="bold", loc="left", pad=12)
    time_label = "Mean time per episode (ms, log scale)\nRL training excluded"
    if clipped_time_error:
        time_label += "\nTriangle: lower SD limit clipped"
    time_ax.set_ylabel(time_label, labelpad=8)
    format_log_axis(time_ax)
    time_ax.set_ylim(bottom=time_floor)
    config = data["config"]
    training = config.get("training_episodes")
    training_label = f"RL training: {training:,} episodes  |  " if training is not None else ""
    fig.suptitle(
        f"Optimizer vs RL  |  {config['evaluation_episodes']} simulated episodes  |  "
        f"seed {config['seed']}\n{training_label}error bars: +/- 1 SD",
        fontsize=14, fontweight="bold",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_sweep(rows: list[dict[str, str]], output: Path) -> None:
    if not rows:
        raise ValueError("sweep contains no rows to plot")
    horizons = sorted({int(row["horizon"]) for row in rows})
    fig, axes = plt.subplots(
        len(horizons), 2, figsize=(13, 4.5 * len(horizons)),
        squeeze=False, layout="constrained",
    )
    for index, horizon in enumerate(horizons):
        subset = [row for row in rows if int(row["horizon"]) == horizon]
        seeds = sorted({int(row["seed"]) for row in subset})
        scenarios = sorted({int(row["planner_scenarios"]) for row in subset})
        left, right = axes[index]
        for seed_index, seed in enumerate(seeds):
            group = sorted(
                (row for row in subset if int(row["seed"]) == seed),
                key=lambda row: int(row["planner_scenarios"]),
            )
            color = plt.get_cmap("tab10")(seed_index % 10)
            x = np.array([int(row["planner_scenarios"]) for row in group])
            delta = np.array([float(row["rl_minus_optimizer_feasible_value"]) for row in group])
            left.plot(x, delta, "o-", color=color, label=f"Seed {seed}")
            for x_value, value, row in zip(x, delta, group):
                if row["paired_value_difference_95_lower"] and row["paired_value_difference_95_upper"]:
                    low = float(row["paired_value_difference_95_lower"])
                    high = float(row["paired_value_difference_95_upper"])
                    left.errorbar(
                        x_value, value, yerr=[[value - low], [high - value]],
                        fmt="none", ecolor=color, capsize=3, linewidth=1,
                    )
            right.plot(
                x, [float(row["optimizer_episode_ms"]) for row in group],
                "o-", color=color, label=f"Optimizer, seed {seed}",
            )
            right.plot(
                x, [float(row["rl_episode_ms"]) for row in group],
                "s--", color=color, label=f"RL, seed {seed}",
            )
        left.axhline(0, color="#374151", linewidth=1)
        left.set_title(f"Value difference  |  {horizon} offers", fontweight="bold", loc="left", pad=12)
        left.set_ylabel("RL minus optimizer\n(mean feasible value, price units)", labelpad=8)
        right.set_title(f"Decision time  |  {horizon} offers", fontweight="bold", loc="left", pad=12)
        right.set_ylabel("Mean time per episode (ms, log scale)", labelpad=8)
        format_log_axis(right)
        for ax in (left, right):
            ax.set_xticks(scenarios)
            ax.set_xlabel("Sampled future scenarios per optimizer decision", labelpad=8)
            style_axis(ax)
            ax.legend(fontsize=9, frameon=False)
    fig.suptitle(
        "Sequential offers  |  optimizer planning budget",
        fontsize=14, fontweight="bold",
    )
    fig.text(
        0.5, -0.13,
        "Positive value difference favors RL; error bars are paired 95% t intervals. "
        "Time excludes RL training.\n"
        "Intervals describe episode variability, not volume-model or between-seed uncertainty.",
        ha="center", fontsize=9, color="#4b5563",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_training_sweep(data: dict, output: Path) -> None:
    rows = data["results"]
    if not rows:
        raise ValueError("training sweep contains no results to plot")
    seeds = sorted({row["training_seed"] for row in rows})
    budgets = sorted({row["stage"]["rl_training_episodes"] for row in rows})
    fig, (value_ax, time_ax) = plt.subplots(1, 2, figsize=(14, 6), layout="constrained")
    for seed_index, seed in enumerate(seeds):
        group = sorted(
            (row for row in rows if row["training_seed"] == seed),
            key=lambda row: row["stage"]["rl_training_episodes"],
        )
        x = [row["stage"]["rl_training_episodes"] for row in group]
        display_x = np.asarray(x, dtype=float) * np.exp(0.06 * (seed_index - (len(seeds) - 1) / 2))
        differences = np.array([row["rl_minus_optimizer_feasible_value"] for row in group])
        color = plt.get_cmap("tab10")(seed_index % 10)
        value_ax.plot(display_x, differences, "o-", color=color, label=f"Training seed {seed}")
        for budget, difference, row in zip(display_x, differences, group):
            low, high = row["paired_difference_95_lower"], row["paired_difference_95_upper"]
            if low is not None and high is not None:
                value_ax.errorbar(
                    budget, difference, yerr=[[difference - low], [high - difference]],
                    fmt="none", ecolor=color, capsize=3, linewidth=1,
                )
        time_ax.plot(
            display_x, [row["stage"]["rl_training_seconds"] for row in group],
            "o-", color=color, label=f"Training seed {seed}",
        )
    value_ax.axhline(0, color="#374151", linewidth=1)
    value_ax.set_title("Gift value vs optimizer", fontweight="bold", loc="left")
    value_ax.set_ylabel("RL minus optimizer mean feasible value\n95% paired episode intervals")
    time_ax.set_title("RL training time", fontweight="bold", loc="left")
    time_ax.set_ylabel("Training time (seconds)")
    for ax in (value_ax, time_ax):
        ax.set_xscale("log")
        ax.set_xticks(budgets, [f"{budget:,}" for budget in budgets])
        ax.set_xlabel("RL training episodes (seed markers offset)")
        style_axis(ax)
        ax.legend(fontsize=9, frameon=False)
    config = data["config"]
    fig.suptitle(
        f"Longer RL training  |  {config['evaluation_episodes']} shared evaluation episodes",
        fontsize=15, fontweight="bold",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot saved gift-selection experiment results.")
    parser.add_argument("--experiment", type=Path, default=Path("results/experiment.json"))
    parser.add_argument("--sweep", type=Path, help="Optional sequential sweep CSV to plot")
    parser.add_argument("--training-sweep", type=Path, help="Optional training-length sweep JSON")
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    with args.experiment.open(encoding="utf-8") as file:
        data = json.load(file)
    experiment_output = args.output_dir / "experiment.png"
    plot_experiment(data, experiment_output)
    print(f"Experiment plot: {experiment_output}")
    if args.sweep is not None:
        with args.sweep.open(newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        sweep_output = args.output_dir / "sweep.png"
        plot_sweep(rows, sweep_output)
        print(f"Sweep plot: {sweep_output}")
    if args.training_sweep is not None:
        with args.training_sweep.open(encoding="utf-8") as file:
            training_data = json.load(file)
        training_output = args.output_dir / "training_sweep.png"
        plot_training_sweep(training_data, training_output)
        print(f"Training sweep plot: {training_output}")


if __name__ == "__main__":
    main()
