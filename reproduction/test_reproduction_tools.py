#!/usr/bin/env python3
"""Pure-Python regression tests for the reproduction tooling."""

import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

import run_benchmark
from compare_benchmarks import calculate_metrics


class ReproductionToolTests(unittest.TestCase):
    def make_spec(self, method="mpc", **changes):
        values = {
            "method": method,
            "speed": 0.75,
            "seed": 0,
            "n_rf": 50,
            "learning_rate": 0.25,
            "kernel": "Gaussian",
            "kernel_std": 0.01,
            "heuristic": False,
            "radius": 1.0,
            "altitude": 1.0,
            "acceleration": 0.25,
            "run_id": "test_001",
            "results_root": Path("/tmp/results"),
            "flight_timeout": 900.0,
        }
        values.update(changes)
        return run_benchmark.BenchmarkSpec(**values)

    def test_mpc_and_ssi_plans_differ_only_where_expected(self):
        repository = Path("/tmp/repository")
        mpc = run_benchmark.build_plan(self.make_spec("mpc"), repository)
        ssi = run_benchmark.build_plan(self.make_spec("ssi-mpc"), repository)

        self.assertEqual(mpc.generated_file.name, "MPC_circle_0.75.mat")
        self.assertEqual(ssi.generated_file.name, "OLMPC_circle_0.75.mat")
        self.assertIn("n_rf:=0", mpc.controller_command)
        self.assertIn("n_rf:=50", ssi.controller_command)
        self.assertEqual(mpc.reference_command, ssi.reference_command)

    def test_unsafe_run_identifier_is_rejected(self):
        for run_id in ("../escape", "/absolute", "contains space", ""):
            with self.subTest(run_id=run_id), self.assertRaises(ValueError):
                run_benchmark.validate_specification(
                    self.make_spec(run_id=run_id)
                )

    def test_loop_speed_boundary_is_rejected_before_ros_starts(self):
        with self.assertRaisesRegex(ValueError, "greater than 2"):
            run_benchmark.validate_specification(
                self.make_spec(speed=0.5, acceleration=0.25)
            )

    def test_metadata_verification_includes_heuristic(self):
        spec = self.make_spec(heuristic=False)
        metadata = {
            "project_mean_distance_m": 0.1,
            "mean_optimization_time_s": 0.002,
            "n_random_features": 0,
            "learning_rate": 0.25,
            "kernel": "Gaussian",
            "kernel_std": 0.01,
            "heuristic": False,
            "random_seed": 0,
            "trajectory_name": "circle",
            "reference_speed_mps": 0.75,
            "metric_definition": "mean_euclidean_position_error",
        }
        run_benchmark.verify_metadata(spec, metadata, {"mean_distance_m": 0.1})
        metadata["heuristic"] = True
        with self.assertRaises(ValueError):
            run_benchmark.verify_metadata(
                spec, metadata, {"mean_distance_m": 0.1}
            )

    def test_metric_calculation(self):
        sample_count = 4
        ref_time = np.arange(sample_count, dtype=float) * 0.02
        ref_x = np.zeros((sample_count, 13), dtype=float)
        ref_x[:, 3] = 1.0
        ref_x[:, 0] = np.linspace(0.0, 0.3, sample_count)
        x = ref_x.copy()
        x[:, 0] -= 0.01
        ref_u = np.full((sample_count, 4), 0.5)
        u = np.full((sample_count, 4), 0.5)
        w_control = np.zeros((sample_count, 3), dtype=float)
        result = {
            "arrays": {
                "ref_time": ref_time,
                "ref_x": ref_x,
                "ref_u": ref_u,
                "x": x,
                "u": u,
                "w_control": w_control,
            }
        }
        metrics = calculate_metrics(result)

        self.assertEqual(metrics["sample_count"], sample_count)
        self.assertAlmostEqual(metrics["mean_distance_m"], 0.01)
        self.assertAlmostEqual(metrics["vector_rmse_m"], 0.01)

    def test_dry_run_does_not_import_ros_runtime(self):
        output = io.StringIO()
        with redirect_stdout(output):
            exit_code = run_benchmark.main(
                [
                    "--method",
                    "mpc",
                    "--speed",
                    "0.75",
                    "--seed",
                    "0",
                    "--run-id",
                    "test_001",
                    "--dry-run",
                ]
            )
        self.assertEqual(exit_code, 0)
        self.assertIn("DRY RUN PASSED", output.getvalue())


if __name__ == "__main__":
    unittest.main(verbosity=2)
