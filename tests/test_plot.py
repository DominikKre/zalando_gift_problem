from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np

from gift_experiment.plot import (
    episode_distributions, plot_experiment, plot_sweep, plot_training_sweep,
)


class PlotTests(unittest.TestCase):
    def setUp(self):
        self.data = {
            "config": {"seed": 42, "evaluation_episodes": 5},
            "stages": [{
                "stage": 1,
                "methods": {
                    "optimizer": {
                        "mean_feasible_value": 80, "mean_value": 100,
                        "violation_rate": 0.2, "violations": 2, "episodes": 10,
                        "mean_episode_decision_ms": 5,
                        "episode_results": [
                            {"value": 100, "fits": fits} for fits in (True, False, True, True, True)
                        ],
                        "episode_decision_times_ms": [5, 4, 6, 5, 5],
                    },
                    "rl": {
                        "mean_feasible_value": 70, "mean_value": 70,
                        "violation_rate": 0, "violations": 0, "episodes": 10,
                        "mean_episode_decision_ms": 1,
                        "episode_results": [{"value": 70, "fits": True}] * 5,
                        "episode_decision_times_ms": [1, 1, 1, 1, 1],
                    },
                    "clairvoyant": {
                        "mean_feasible_value": 150,
                    },
                },
            }],
        }
        self.rows = [{
            "horizon": "12", "seed": "42", "planner_scenarios": "1",
            "rl_minus_optimizer_feasible_value": "-10",
            "paired_value_difference_95_lower": "-20",
            "paired_value_difference_95_upper": "0",
            "optimizer_episode_ms": "10", "rl_episode_ms": "0.1",
        }]

    def test_experiment_shows_only_optimizer_and_rl_values_and_times(self):
        with patch("matplotlib.figure.Figure.savefig"), patch(
            "gift_experiment.plot.plt.close"
        ):
            plot_experiment(self.data, Path("unused.png"))
            figure = plt.gcf()
            self.assertEqual(len(figure.axes), 2)
            values, times = figure.axes
            optimizer = self.data["stages"][0]["methods"]["optimizer"]
            rl = self.data["stages"][0]["methods"]["rl"]
            self.assertEqual(len(values.patches), 2)
            self.assertEqual([text.get_text() for text in values.get_legend().get_texts()], ["Optimizer", "RL"])
            self.assertAlmostEqual(values.patches[0].get_height(), optimizer["mean_feasible_value"])
            self.assertAlmostEqual(values.patches[1].get_height(), rl["mean_feasible_value"])
            self.assertEqual(len(times.patches), 2)
            self.assertAlmostEqual(
                times.patches[0].get_y() + times.patches[0].get_height(),
                optimizer["mean_episode_decision_ms"],
            )
            samples = episode_distributions(optimizer)[0]
            expected = np.array([samples.mean() - samples.std(ddof=1),
                                 samples.mean() + samples.std(ddof=1)])
            np.testing.assert_allclose(values.collections[0].get_segments()[0][:, 1], expected)
            for axis in figure.axes:
                self.assertFalse(any("Optimizer:" in text.get_text() or "RL:" in text.get_text()
                                     for text in axis.texts))
            self.assertEqual(times.get_yscale(), "log")
            for axis in figure.axes:
                self.assertTrue(axis.get_xlabel())
                self.assertTrue(axis.get_ylabel())
        plt.close(figure)

    def test_log_time_error_bar_clips_nonpositive_lower_bound(self):
        result = self.data["stages"][0]["methods"]["optimizer"]
        result["episode_decision_times_ms"] = [0.01, 0.01, 0.01, 0.01, 10.0]
        with patch("matplotlib.figure.Figure.savefig"), patch("gift_experiment.plot.plt.close"):
            plot_experiment(self.data, Path("unused.png"))
            figure = plt.gcf()
            times = figure.axes[1]
            self.assertIn("lower SD limit clipped", times.get_ylabel())
            segment = times.collections[1].get_segments()[0]
            self.assertGreater(segment[0, 1], 0)
            self.assertAlmostEqual(segment[0, 1], times.get_ylim()[0])
        plt.close(figure)

    def test_missing_episode_timings_requires_rerun(self):
        with self.assertRaisesRegex(ValueError, "rerun the experiment"):
            episode_distributions({"episode_results": [{"value": 1, "fits": True}]})

    def test_sweep_uses_actual_scenario_counts_on_horizontal_axis(self):
        rows = [dict(self.rows[0], planner_scenarios=str(count)) for count in (1, 4, 8)]
        with patch("matplotlib.figure.Figure.savefig"), patch("gift_experiment.plot.plt.close"):
            plot_sweep(rows, Path("unused.png"))
            figure = plt.gcf()
            for axis in figure.axes:
                self.assertEqual(list(axis.lines[0].get_xdata()), [1, 4, 8])
                self.assertTrue(axis.get_xlabel())
                self.assertTrue(axis.get_ylabel())
        plt.close(figure)

    def test_plots_render_and_empty_sweep_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "nested" / "experiment.png"
            second = Path(directory) / "nested" / "sweep.png"
            plot_experiment(self.data, first)
            plot_sweep(self.rows, second)
            for path in (first, second):
                self.assertEqual(path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            with self.assertRaisesRegex(ValueError, "no rows"):
                plot_sweep([], second)

    def test_training_sweep_plots_paired_differences_and_training_cost(self):
        data = {
            "config": {"evaluation_episodes": 5},
            "results": [
                {
                    "training_seed": 1,
                    "rl_minus_optimizer_feasible_value": difference,
                    "paired_difference_95_lower": difference - 2,
                    "paired_difference_95_upper": difference + 2,
                    "stage": {"rl_training_episodes": budget, "rl_training_seconds": seconds},
                }
                for budget, difference, seconds in [(4000, -10, 1), (20000, 5, 4)]
            ],
        }
        with patch("matplotlib.figure.Figure.savefig"), patch("gift_experiment.plot.plt.close"):
            plot_training_sweep(data, Path("unused.png"))
            figure = plt.gcf()
            values, times = figure.axes
            self.assertEqual(list(values.lines[0].get_ydata()), [-10, 5])
            self.assertEqual(list(times.lines[0].get_ydata()), [1, 4])
            self.assertEqual(values.get_xscale(), "log")
        plt.close(figure)


if __name__ == "__main__":
    unittest.main()
