from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from scipy.optimize import lsq_linear
from scipy.stats import norm


@dataclass(frozen=True)
class VolumeModel:
    names: tuple[str, ...]
    prices: np.ndarray
    means: np.ndarray
    covariance: np.ndarray
    training_rmse: float
    validation_rmse: float
    training_packages: int
    validation_packages: int

    def safe_weights(self, risk: float) -> np.ndarray:
        if not 0 < risk < 1:
            raise ValueError("risk must be between 0 and 1")
        # A union bound makes this safe for any subset, without assuming
        # independence between the estimated item volumes.
        return np.maximum(
            0, self.means + norm.isf(risk / len(self.names)) * np.sqrt(np.diag(self.covariance))
        )

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        return np.maximum(0, rng.multivariate_normal(self.means, self.covariance))


def load_and_fit(
    directory: Path, seed: int = 0, validation_fraction: float = 0.2
) -> VolumeModel:
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    with (directory / "items.json").open(encoding="utf-8-sig") as file:
        items = json.load(file)
    with (directory / "packages.json").open(encoding="utf-8-sig") as file:
        packages = json.load(file)

    names = tuple(item["name"] for item in items)
    if not names or len(set(names)) != len(names):
        raise ValueError("items must have unique, nonempty names")
    prices = np.asarray([item["price"] for item in items], dtype=float)
    if not np.all(np.isfinite(prices)) or np.any(prices < 0):
        raise ValueError("prices must be finite and nonnegative")
    indices = {name: index for index, name in enumerate(names)}
    matrix = np.zeros((len(packages), len(names)), dtype=float)
    observed = np.empty(len(packages))
    for row, package in enumerate(packages):
        package_items = package["items"]
        if len(set(package_items)) != len(package_items):
            raise ValueError(f"duplicate item in package {row}")
        for name in package_items:
            if name not in indices:
                raise ValueError(f"unknown item {name!r} in package {row}")
            matrix[row, indices[name]] = 1
        observed[row] = package["total_volume"]
    if not np.all(np.isfinite(observed)):
        raise ValueError("package volumes must be finite")

    order = np.random.default_rng(seed).permutation(len(packages))
    validation_count = max(1, int(len(packages) * validation_fraction))
    validation, training = order[:validation_count], order[validation_count:]
    design = matrix[training]
    if np.linalg.matrix_rank(design) != len(names):
        raise ValueError("training package data do not identify all item volumes")
    fit = lsq_linear(design, observed[training], bounds=(0, np.inf))
    if not fit.success:
        raise RuntimeError(f"volume estimation failed: {fit.message}")
    residuals = observed[validation] - matrix[validation] @ fit.x
    training_residuals = observed[training] - design @ fit.x
    # The stated measurement noise variance is 2. This is a Gaussian
    # approximation around the constrained fit, not a fully Bayesian posterior.
    covariance = 2.0 * np.linalg.inv(design.T @ design)
    return VolumeModel(
        names, prices, fit.x, covariance, float(np.sqrt(np.mean(training_residuals**2))),
        float(np.sqrt(np.mean(residuals**2))),
        len(training), len(validation),
    )
