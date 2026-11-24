from dataclasses import dataclass

import numpy as np


class FeatureBuilder:
    def __init__(self, prices: np.ndarray, weights: np.ndarray, backpack: float):
        self.prices = prices
        self.weights = weights
        self.backpack = backpack
        self.price_scale = max(float(prices.max()), 1.0)
        self.ratio = prices / np.maximum(weights, 1e-8)
        self.ratio_scale = max(float(np.quantile(self.ratio, 0.9)), 1.0)

    def build(
        self, item: int, remaining: np.ndarray, remaining_offers: int, capacity: float
    ) -> np.ndarray:
        average_ratio = float(self.ratio[remaining].mean()) if len(remaining) else 0.0
        average_price = float(self.prices[remaining].mean()) if len(remaining) else 0.0
        summary = np.array([
            1.0,
            self.prices[item] / self.price_scale,
            self.weights[item] / self.backpack,
            min(self.ratio[item] / self.ratio_scale, 2.0),
            capacity / self.backpack,
            remaining_offers / len(self.prices),
            (capacity - self.weights[item]) / self.backpack,
            average_price / self.price_scale,
            np.clip((self.ratio[item] - average_ratio) / self.ratio_scale, -2.0, 2.0),
            self.prices[item] / self.price_scale * capacity / self.backpack,
        ])
        remaining_mask = np.zeros(len(self.prices))
        remaining_mask[remaining] = 1.0 / np.sqrt(len(self.prices))
        current_mask = np.zeros(len(self.prices))
        current_mask[item] = 1.0
        return np.concatenate((summary, remaining_mask, current_mask))


def _probability(parameters: np.ndarray, state: np.ndarray) -> float:
    logit = float(np.clip(parameters @ state, -30, 30))
    return 1.0 / (1.0 + np.exp(-logit))


@dataclass
class Policy:
    parameters: np.ndarray

    def decide(self, state: np.ndarray) -> bool:
        return _probability(self.parameters, state) >= 0.5


def train_policy(
    prices: np.ndarray,
    weights: np.ndarray,
    backpack: float,
    horizon: int,
    episodes: int,
    seed: int,
    batch_size: int = 32,
    learning_rate: float = 3.0,
) -> Policy:
    """Train a linear stochastic policy with REINFORCE and a fitted baseline."""
    if not 1 <= horizon <= len(prices) or episodes < 1 or batch_size < 1:
        raise ValueError("invalid training horizon, episode count, or batch size")
    rng = np.random.default_rng(seed)
    parameters = np.zeros(10 + 2 * len(prices))
    price_scale = max(float(prices.max()), 1.0)
    all_items = np.arange(len(prices))
    feature_builder = FeatureBuilder(prices, weights, backpack)
    for batch_start in range(0, episodes, batch_size):
        states, scores, returns = [], [], []
        for _ in range(min(batch_size, episodes - batch_start)):
            offers = rng.choice(all_items, size=horizon, replace=False)
            remaining = list(all_items)
            capacity = backpack
            trajectory = []
            for step, item in enumerate(offers):
                remaining.remove(item)
                if weights[item] > capacity + 1e-9:
                    continue
                state = feature_builder.build(
                    item, np.asarray(remaining, dtype=int), horizon - step - 1, capacity
                )
                probability = _probability(parameters, state)
                action = int(rng.random() < probability)
                reward = action * prices[item] / price_scale
                trajectory.append((state, action - probability, reward))
                capacity -= action * weights[item]
            future_return = 0.0
            for state, score, reward in reversed(trajectory):
                future_return += reward
                states.append(state)
                scores.append(score)
                returns.append(future_return)
        if not states:
            continue
        x = np.asarray(states)
        y = np.asarray(returns)
        # A regularized linear value baseline reduces policy-gradient variance.
        baseline_features = x[:, :10]
        baseline = np.linalg.solve(
            baseline_features.T @ baseline_features + 0.01 * np.eye(10),
            baseline_features.T @ y,
        )
        advantage = y - baseline_features @ baseline
        advantage /= max(float(advantage.std()), 1e-8)
        gradient = np.mean(x * (np.asarray(scores) * advantage)[:, None], axis=0)
        parameters += learning_rate * gradient
    return Policy(parameters)
