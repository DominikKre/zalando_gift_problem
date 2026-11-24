import time

import numpy as np

from .data import VolumeModel
from .optimizer import online_decision, solve_knapsack
from .rl import FeatureBuilder, train_policy


def _selection_record(
    selected: list[int], model: VolumeModel, weights: np.ndarray, true_volumes: np.ndarray
) -> dict:
    indices = np.asarray(selected, dtype=int)
    volume = float(true_volumes[indices].sum())
    return {
        "items": [model.names[index] for index in selected],
        "value": float(model.prices[indices].sum()),
        "constraint_volume": float(weights[indices].sum()),
        "simulated_volume": volume,
        "fits": bool(volume <= 40.0),
    }


def _summary(records: list[dict], episode_times: list[float], decision_times: list[float]) -> dict:
    values = np.asarray([record["value"] for record in records])
    fits = np.asarray([record["fits"] for record in records])
    return {
        "mean_value": float(values.mean()),
        "mean_value_if_fits": float(values[fits].mean()) if fits.any() else None,
        "mean_feasible_value": float(np.mean(values * fits)),
        "violation_rate": float(1 - fits.mean()),
        "violations": int((~fits).sum()),
        "episodes": len(records),
        "mean_episode_decision_ms": float(np.mean(episode_times) * 1000),
        "mean_decision_ms": float(np.mean(decision_times) * 1000),
        "p95_decision_ms": float(np.quantile(decision_times, 0.95) * 1000),
    }


def run_stage(
    model: VolumeModel,
    stage: int,
    episodes: int,
    training_episodes: int,
    horizon: int,
    scenarios: int,
    risk: float,
    seed: int,
    *,
    training_seed: int | None = None,
) -> dict:
    if stage not in (1, 2, 3):
        raise ValueError("stage must be 1, 2, or 3")
    if episodes < 1 or training_episodes < 1 or scenarios < 1:
        raise ValueError("episode and scenario counts must be positive")
    if not 1 <= horizon <= len(model.names):
        raise ValueError("horizon must be between 1 and the number of items")
    capacity = 40.0
    weights = model.means if stage == 1 else model.safe_weights(risk)
    offer_count = len(model.names) if stage < 3 else horizon
    start = time.perf_counter()
    policy = train_policy(
        model.prices, weights, capacity, offer_count, training_episodes,
        seed + 100 if training_seed is None else training_seed,
    )
    training_seconds = time.perf_counter() - start

    evaluation_rng = np.random.default_rng(seed + 200)
    planner_rng = np.random.default_rng(seed + 300)
    feature_builder = FeatureBuilder(model.prices, weights, capacity)
    methods = ["optimizer", "rl"] + (["clairvoyant"] if stage == 3 else [])
    records: dict[str, list[dict]] = {method: [] for method in methods}
    episode_times: dict[str, list[float]] = {method: [] for method in methods}
    decision_times: dict[str, list[float]] = {method: [] for method in methods}
    all_items = np.arange(len(model.names))

    for _ in range(episodes):
        offers = evaluation_rng.choice(all_items, size=offer_count, replace=False)
        true_volumes = model.sample(evaluation_rng)
        if stage < 3:
            start = time.perf_counter()
            mask = solve_knapsack(model.prices, weights, capacity)
            elapsed = time.perf_counter() - start
            selected = list(np.flatnonzero(mask))
            records["optimizer"].append(_selection_record(selected, model, weights, true_volumes))
            decision_times["optimizer"].append(elapsed)
            episode_times["optimizer"].append(elapsed)
        else:
            start = time.perf_counter()
            mask = solve_knapsack(model.prices[offers], weights[offers], capacity)
            elapsed = time.perf_counter() - start
            selected = list(offers[mask])
            records["clairvoyant"].append(_selection_record(selected, model, weights, true_volumes))
            decision_times["clairvoyant"].append(elapsed)
            episode_times["clairvoyant"].append(elapsed)

        for method in (["optimizer", "rl"] if stage == 3 else ["rl"]):
            selected = []
            remaining = list(all_items)
            remaining_capacity = capacity
            total_time = 0.0
            for step, item in enumerate(offers):
                remaining.remove(item)
                start = time.perf_counter()
                if weights[item] > remaining_capacity + 1e-9:
                    accept = False
                elif method == "rl":
                    state = feature_builder.build(
                        item, np.asarray(remaining, dtype=int), offer_count - step - 1,
                        remaining_capacity,
                    )
                    accept = policy.decide(state)
                else:
                    accept = online_decision(
                        item, np.asarray(remaining, dtype=int), offer_count - step - 1,
                        remaining_capacity, model.prices, weights, scenarios, planner_rng,
                    )
                elapsed = time.perf_counter() - start
                decision_times[method].append(elapsed)
                total_time += elapsed
                if accept:
                    selected.append(int(item))
                    remaining_capacity -= weights[item]
            episode_times[method].append(total_time)
            records[method].append(_selection_record(selected, model, weights, true_volumes))

    return {
        "stage": stage,
        "horizon": offer_count,
        "risk_target": risk if stage > 1 else None,
        "planner_scenarios": scenarios if stage == 3 else None,
        "rl_training_episodes": training_episodes,
        "rl_training_seconds": training_seconds,
        "rl_parameters": policy.parameters.tolist(),
        "methods": {
            method: {
                **_summary(records[method], episode_times[method], decision_times[method]),
                "episode_results": records[method],
                "episode_decision_times_ms": [
                    elapsed * 1000 for elapsed in episode_times[method]
                ],
            }
            for method in methods
        },
    }
