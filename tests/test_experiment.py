import contextlib
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from gift_experiment.data import VolumeModel, load_and_fit
from gift_experiment.experiment import run_stage
from gift_experiment.optimizer import online_decision, solve_knapsack
from gift_experiment.rl import FeatureBuilder, train_policy
from gift_experiment.sweep import main as sweep_main, paired_value_interval, wilson_interval
from gift_experiment.training_sweep import main as training_sweep_main


class VolumeModelTests(unittest.TestCase):
    def test_fit_and_validate_nonnegative_volumes(self):
        items = [{"name": "a", "price": 4}, {"name": "b", "price": 5}]
        packages = (
            [{"items": ["a"], "total_volume": 2.0}] * 20
            + [{"items": ["b"], "total_volume": 3.0}] * 20
            + [{"items": ["a", "b"], "total_volume": 5.0}] * 20
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "items.json").write_text(json.dumps(items))
            (path / "packages.json").write_text(json.dumps(packages))
            model = load_and_fit(path, seed=2)
        np.testing.assert_allclose(model.means, [2, 3], atol=1e-10)
        self.assertAlmostEqual(model.validation_rmse, 0)
        self.assertEqual(model.training_packages + model.validation_packages, 60)
        self.assertTrue(np.all(model.safe_weights(0.05) >= model.means))
        self.assertTrue(np.all(model.safe_weights(0.01) >= model.safe_weights(0.05)))

    def test_duplicate_package_item_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "items.json").write_text(json.dumps([{"name": "a", "price": 1}]))
            (path / "packages.json").write_text(
                json.dumps([{"items": ["a", "a"], "total_volume": 2}])
            )
            with self.assertRaisesRegex(ValueError, "duplicate item"):
                load_and_fit(path)


class OptimizerTests(unittest.TestCase):
    def test_knapsack_finds_optimal_not_greedy_selection(self):
        prices = np.array([10.0, 7.0, 7.0])
        weights = np.array([4.0, 3.0, 3.0])
        selected = solve_knapsack(prices, weights, 6)
        self.assertEqual(prices[selected].sum(), 14)
        self.assertLessEqual(weights[selected].sum(), 6)

    def test_online_planner_accounts_for_future_offers(self):
        prices = np.array([9.0, 20.0])
        weights = np.array([2.0, 3.0])
        rng = np.random.default_rng(1)
        self.assertFalse(online_decision(0, np.array([1]), 1, 3, prices, weights, 2, rng))
        self.assertTrue(online_decision(0, np.array([1]), 0, 3, prices, weights, 2, rng))
        self.assertFalse(online_decision(1, np.array([0]), 1, 2, prices, weights, 2, rng))


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.model = VolumeModel(
            names=("a", "b", "c"),
            prices=np.array([9.0, 20.0, 7.0]),
            means=np.array([20.0, 30.0, 10.0]),
            covariance=np.zeros((3, 3)),
            training_rmse=0.0,
            validation_rmse=0.0,
            training_packages=10,
            validation_packages=2,
        )

    def test_training_is_reproducible_and_updates_parameters(self):
        args = (self.model.prices, self.model.means, 40.0, 2, 128, 7)
        first = train_policy(*args)
        second = train_policy(*args)
        np.testing.assert_allclose(first.parameters, second.parameters)
        self.assertGreater(np.linalg.norm(first.parameters), 0)

    def test_policy_learns_to_save_capacity_for_valuable_offer(self):
        prices = np.array([1.0, 10.0])
        weights = np.array([1.0, 1.0])
        policy = train_policy(prices, weights, 1.0, 2, 4000, 1)
        builder = FeatureBuilder(prices, weights, 1.0)
        self.assertFalse(policy.decide(builder.build(0, np.array([1]), 1, 1.0)))
        self.assertTrue(policy.decide(builder.build(1, np.array([0]), 1, 1.0)))

    def test_all_stages_are_feasible_and_paired(self):
        for stage in (1, 2, 3):
            with self.subTest(stage=stage):
                result = run_stage(self.model, stage, 4, 32, 2, 2, 0.05, 17)
                expected = {"optimizer", "rl"} | ({"clairvoyant"} if stage == 3 else set())
                self.assertEqual(set(result["methods"]), expected)
                for method in expected:
                    records = result["methods"][method]["episode_results"]
                    self.assertEqual(len(records), 4)
                    timings = result["methods"][method]["episode_decision_times_ms"]
                    self.assertEqual(len(timings), len(records))
                    self.assertTrue(all(value >= 0 for value in timings))
                    self.assertAlmostEqual(
                        np.mean(timings), result["methods"][method]["mean_episode_decision_ms"]
                    )
                    for record in records:
                        self.assertTrue(record["fits"])
                        self.assertLessEqual(record["constraint_volume"], 40)
                        self.assertEqual(len(record["items"]), len(set(record["items"])))
                if stage == 3:
                    upper = result["methods"]["clairvoyant"]["episode_results"]
                    for method in ("optimizer", "rl"):
                        for record, bound in zip(
                            result["methods"][method]["episode_results"], upper
                        ):
                            self.assertLessEqual(record["value"], bound["value"])

    def test_seed_reproduces_episode_results(self):
        args = (self.model, 3, 3, 32, 2, 2, 0.05, 19)
        first, second = run_stage(*args), run_stage(*args)
        for method in first["methods"]:
            self.assertEqual(
                first["methods"][method]["episode_results"],
                second["methods"][method]["episode_results"],
            )

    def test_training_seed_does_not_change_evaluation_episodes(self):
        args = (self.model, 3, 5, 64, 2, 2, 0.05, 19)
        first = run_stage(*args, training_seed=1)
        second = run_stage(*args, training_seed=2)
        self.assertNotEqual(first["rl_parameters"], second["rl_parameters"])
        for method in ("optimizer", "clairvoyant"):
            self.assertEqual(
                first["methods"][method]["episode_results"],
                second["methods"][method]["episode_results"],
            )

    def test_training_sweep_evaluates_same_episodes_across_lengths(self):
        project_dir = Path(__file__).resolve().parent.parent
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "training.json"
            args = [
                "training_sweep", "--data-dir", str(project_dir), "--output", str(output),
                "--training-episodes", "4", "8", "--training-seeds", "1", "2",
                "--episodes", "3", "--horizon", "3", "--scenarios", "1",
            ]
            with patch.object(sys, "argv", args), contextlib.redirect_stdout(io.StringIO()):
                training_sweep_main()
            data = json.loads(output.read_text())
        self.assertEqual(len(data["results"]), 4)
        runs = [row["stage"] for row in data["results"]]
        self.assertEqual([run["rl_training_episodes"] for run in runs], [4, 4, 8, 8])
        first = runs[0]["methods"]["optimizer"]["episode_results"]
        for run in runs[1:]:
            self.assertEqual(first, run["methods"]["optimizer"]["episode_results"])

    def test_sweep_writes_one_row_per_planning_budget(self):
        project_dir = Path(__file__).resolve().parent.parent
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "sweep.csv"
            args = [
                "sweep", "--data-dir", str(project_dir), "--output", str(output),
                "--horizons", "3", "--scenario-counts", "1", "2",
                "--episodes", "2", "--training-episodes", "16",
            ]
            with patch.object(sys, "argv", args), contextlib.redirect_stdout(io.StringIO()):
                sweep_main()
            with output.open(newline="") as file:
                rows = list(csv.DictReader(file))
        self.assertEqual([row["planner_scenarios"] for row in rows], ["1", "2"])
        self.assertTrue(all(row["evaluation_episodes"] == "2" for row in rows))
        self.assertTrue(all(row["rl_mean_feasible_value"] for row in rows))

    def test_sweep_reports_paired_value_and_violation_uncertainty(self):
        first = [{"value": 10, "fits": True}, {"value": 50, "fits": False}]
        second = [{"value": 8, "fits": True}, {"value": 3, "fits": True}]
        mean, low, high = paired_value_interval(first, second)
        self.assertAlmostEqual(mean, -0.5)
        self.assertLess(low, mean)
        self.assertGreater(high, mean)
        _, upper = wilson_interval(0, 50)
        self.assertGreater(upper, 0.05)


if __name__ == "__main__":
    unittest.main()
