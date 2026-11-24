import argparse
import json
from pathlib import Path
import sys

from .data import load_and_fit
from .experiment import run_stage


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare optimization and RL on gift selection.")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--output", type=Path, default=Path("results/experiment.json"))
    parser.add_argument("--stages", type=int, nargs="+", choices=(1, 2, 3), default=[1, 2, 3])
    parser.add_argument("--episodes", type=positive_int, default=30)
    parser.add_argument("--training-episodes", type=positive_int, default=20000)
    parser.add_argument("--horizon", type=positive_int, default=12)
    parser.add_argument("--scenarios", type=positive_int, default=4)
    parser.add_argument("--risk", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 0 < args.risk < 1:
        parser.error("--risk must be between 0 and 1")
    if args.seed < 0:
        parser.error("--seed must be nonnegative")
    model = load_and_fit(args.data_dir, seed=args.seed)
    if args.horizon > len(model.names):
        parser.error("--horizon cannot exceed the number of items")
    output = {
        "config": {
            "seed": args.seed, "evaluation_episodes": args.episodes,
            "training_episodes": args.training_episodes, "horizon": args.horizon,
            "planner_scenarios": args.scenarios, "risk": args.risk,
        },
        "volume_model": {
            "item_count": len(model.names),
            "training_packages": model.training_packages,
            "validation_packages": model.validation_packages,
            "training_rmse_liters": model.training_rmse,
            "validation_rmse_liters": model.validation_rmse,
            "assumed_measurement_noise_std_liters": 2 ** 0.5,
            "item_order": list(model.names),
            "estimated_volumes_liters": dict(zip(model.names, model.means.tolist())),
            "covariance_liters_squared": model.covariance.tolist(),
        },
        "stages": [],
    }
    for stage in args.stages:
        print(f"Running stage {stage}...", flush=True)
        result = run_stage(
            model, stage, args.episodes, args.training_episodes, args.horizon,
            args.scenarios, args.risk, args.seed + stage * 1000,
        )
        output["stages"].append(result)
        for method, summary in result["methods"].items():
            print(
                f"  {method:11} value={summary['mean_value']:.1f} "
                f"violations={summary['violations']}/{args.episodes} "
                f"episode_ms={summary['mean_episode_decision_ms']:.3f}"
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"Results: {args.output}", file=sys.stdout)


if __name__ == "__main__":
    main()
