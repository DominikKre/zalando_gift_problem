import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp


def solve_knapsack(
    prices: np.ndarray, weights: np.ndarray, capacity: float
) -> np.ndarray:
    """Return an exact 0/1 solution, or fail explicitly if no optimum is found."""
    if len(prices) != len(weights):
        raise ValueError("prices and weights must have the same length")
    if capacity < -1e-9:
        raise ValueError("capacity cannot be negative")
    if not len(prices):
        return np.zeros(0, dtype=bool)
    result = milp(
        c=-prices,
        integrality=np.ones(len(prices)),
        bounds=Bounds(0, 1),
        constraints=LinearConstraint(weights.reshape(1, -1), -np.inf, max(0, capacity)),
        options={"mip_rel_gap": 0.0},
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"knapsack optimization failed: {result.message}")
    selection = result.x > 0.5
    if weights[selection].sum() > capacity + 1e-7:
        raise RuntimeError("optimizer returned an infeasible selection")
    return selection


def online_decision(
    item: int,
    remaining: np.ndarray,
    remaining_offers: int,
    capacity: float,
    prices: np.ndarray,
    weights: np.ndarray,
    scenarios: int,
    rng: np.random.Generator,
) -> bool:
    """Compare accept/reject using exact recourse for sampled future offers."""
    if scenarios < 1:
        raise ValueError("scenarios must be positive")
    if remaining_offers < 0 or remaining_offers > len(remaining):
        raise ValueError("invalid number of remaining offers")
    if weights[item] > capacity + 1e-9:
        return False
    accept_values = []
    reject_values = []
    for _ in range(scenarios):
        future = rng.choice(remaining, size=remaining_offers, replace=False)
        future_prices, future_weights = prices[future], weights[future]
        accept = solve_knapsack(future_prices, future_weights, capacity - weights[item])
        reject = solve_knapsack(future_prices, future_weights, capacity)
        accept_values.append(prices[item] + future_prices[accept].sum())
        reject_values.append(future_prices[reject].sum())
    return bool(np.mean(accept_values) >= np.mean(reject_values))
