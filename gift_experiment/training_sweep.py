import argparse
import json
from pathlib import Path

from .__main__ import positive_int
from .data import load_and_fit
from .experiment import run_stage
from .sweep import paired_value_interval, wilson_interval


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare RL training lengths on sequential offers.")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--output", type=Path, default=Path("results/training_sweep.json"))
    parser.add_argument("--training-episodes", type=positive_int, nargs="+", default=[4000, 20000, 100000])
    parser.add_argument("--training-seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--episodes", type=positive_int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--horizon", type=positive_int, default=12)
    parser.add_argument("--scenarios", type=positive_int, default=4)
    parser.add_argument("--risk", type=float, default=0.05)
    args = parser.parse_args()
    if args.seed < 0 or any(seed < 0 for seed in args.training_seeds):
        parser.error("seeds must be nonnegative")
    if not 0 < args.risk < 1:
        parser.error("--risk must be between 0 and 1")
    if len(set(args.training_episodes)) != len(args.training_episodes):
        parser.error("--training-episodes cannot contain duplicates")
    if len(set(args.training_seeds)) != len(args.training_seeds):
        parser.error("--training-seeds cannot contain duplicates")
    model = load_and_fit(args.data_dir, seed=args.seed)
    if args.horizon > len(model.names):
        parser.error("--horizon cannot exceed the number of items")

    results = []
    for training_episodes in args.training_episodes:
        for training_seed in args.training_seeds:
            print(f"training episodes={training_episodes} seed={training_seed}", flush=True)
            stage = run_stage(
                model, 3, args.episodes, training_episodes, args.horizon,
                args.scenarios, args.risk, args.seed + 3000, training_seed=training_seed,
            )
            rl = stage["methods"]["rl"]
            optimizer = stage["methods"]["optimizer"]
            difference, lower, upper = paired_value_interval(
                rl["episode_results"], optimizer["episode_results"]
            )
            _, violation_upper = wilson_interval(rl["violations"], args.episodes)
            results.append({
                "training_seed": training_seed,
                "rl_minus_optimizer_feasible_value": difference,
                "paired_difference_95_lower": lower,
                "paired_difference_95_upper": upper,
                "rl_violation_wilson_95_upper": violation_upper,
                "stage": stage,
            })
            print(
                f"  RL={rl['mean_feasible_value']:.1f} "
                f"optimizer={optimizer['mean_feasible_value']:.1f} "
                f"training={stage['rl_training_seconds']:.2f}s", flush=True,
            )
    output = {
        "config": {
            "model_split_seed": args.seed,
            "evaluation_rng_seed": args.seed + 3200,
            "planner_rng_seed": args.seed + 3300,
            "training_seeds": args.training_seeds,
            "training_episodes": args.training_episodes,
            "evaluation_episodes": args.episodes,
            "horizon": args.horizon,
            "planner_scenarios": args.scenarios,
            "risk": args.risk,
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"Results: {args.output}")


if __name__ == "__main__":
    main()
