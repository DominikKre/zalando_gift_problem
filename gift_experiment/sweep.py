import argparse
import csv
import math
from pathlib import Path

import numpy as np
from scipy.stats import norm, t

from .__main__ import positive_int
from .data import load_and_fit
from .experiment import run_stage


def paired_value_interval(first: list[dict], second: list[dict]) -> tuple[float, float | None, float | None]:
    if len(first) != len(second) or not first:
        raise ValueError("paired results must have the same positive length")
    differences = np.array([
        a["value"] * a["fits"] - b["value"] * b["fits"]
        for a, b in zip(first, second)
    ])
    mean = float(differences.mean())
    if len(differences) == 1:
        return mean, None, None
    half_width = float(
        t.ppf(0.975, len(differences) - 1)
        * differences.std(ddof=1) / np.sqrt(len(differences))
    )
    return mean, mean - half_width, mean + half_width


def wilson_interval(violations: int, episodes: int) -> tuple[float, float]:
    if episodes < 1 or not 0 <= violations <= episodes:
        raise ValueError("invalid violation count")
    z = norm.ppf(0.975)
    rate = violations / episodes
    denominator = 1 + z**2 / episodes
    center = (rate + z**2 / (2 * episodes)) / denominator
    margin = z * np.sqrt(rate * (1 - rate) / episodes + z**2 / (4 * episodes**2))
    return float((center - margin / denominator)), float((center + margin / denominator))


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep sequential offer horizons and planner budgets.")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--output", type=Path, default=Path("results/sweep.csv"))
    parser.add_argument("--horizons", nargs="+", type=positive_int, default=[6, 12])
    parser.add_argument("--scenario-counts", nargs="+", type=positive_int, default=[1, 4])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--episodes", type=positive_int, default=30)
    parser.add_argument("--training-episodes", type=positive_int, default=20000)
    parser.add_argument("--risk", type=float, default=0.05)
    args = parser.parse_args()
    if not 0 < args.risk < 1:
        parser.error("--risk must be between 0 and 1")
    if any(seed < 0 for seed in args.seeds):
        parser.error("--seeds must be nonnegative")

    rows = []
    for seed in args.seeds:
        model = load_and_fit(args.data_dir, seed=seed)
        if max(args.horizons) > len(model.names):
            parser.error("--horizons cannot exceed the number of items")
        for horizon in args.horizons:
            for scenarios in args.scenario_counts:
                print(f"seed={seed} horizon={horizon} scenarios={scenarios}", flush=True)
                result = run_stage(
                    model, 3, args.episodes, args.training_episodes, horizon,
                    scenarios, args.risk, seed + 3000,
                )
                opt = result["methods"]["optimizer"]
                rl = result["methods"]["rl"]
                clairvoyant = result["methods"]["clairvoyant"]
                difference, difference_low, difference_high = paired_value_interval(
                    rl["episode_results"], opt["episode_results"]
                )
                _, opt_violation_high = wilson_interval(opt["violations"], args.episodes)
                _, rl_violation_high = wilson_interval(rl["violations"], args.episodes)
                saved_seconds = (
                    opt["mean_episode_decision_ms"] - rl["mean_episode_decision_ms"]
                ) / 1000
                break_even = (
                    math.ceil(result["rl_training_seconds"] / saved_seconds)
                    if saved_seconds > 0 else None
                )
                rows.append({
                    "seed": seed,
                    "horizon": horizon,
                    "planner_scenarios": scenarios,
                    "evaluation_episodes": args.episodes,
                    "training_episodes": args.training_episodes,
                    "risk_target": args.risk,
                    "validation_rmse_liters": model.validation_rmse,
                    "optimizer_mean_feasible_value": opt["mean_feasible_value"],
                    "rl_mean_feasible_value": rl["mean_feasible_value"],
                    "clairvoyant_mean_feasible_value": clairvoyant["mean_feasible_value"],
                    "optimizer_violation_rate": opt["violation_rate"],
                    "rl_violation_rate": rl["violation_rate"],
                    "optimizer_violation_wilson_95_upper": opt_violation_high,
                    "rl_violation_wilson_95_upper": rl_violation_high,
                    "rl_minus_optimizer_feasible_value": difference,
                    "paired_value_difference_95_lower": difference_low,
                    "paired_value_difference_95_upper": difference_high,
                    "optimizer_episode_ms": opt["mean_episode_decision_ms"],
                    "rl_episode_ms": rl["mean_episode_decision_ms"],
                    "optimizer_p95_decision_ms": opt["p95_decision_ms"],
                    "rl_p95_decision_ms": rl["p95_decision_ms"],
                    "rl_training_seconds": result["rl_training_seconds"],
                    "time_only_break_even_episodes": break_even,
                })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Results: {args.output}")


if __name__ == "__main__":
    main()
