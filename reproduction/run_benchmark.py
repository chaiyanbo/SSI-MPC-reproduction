#!/usr/bin/env python3
"""Run one safely monitored MPC or SSI-MPC Gazebo benchmark."""

import argparse
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from compare_benchmarks import calculate_metrics, load_and_validate


@dataclass(frozen=True)
class BenchmarkSpec:
    method: str
    speed: float
    seed: int
    n_rf: int
    learning_rate: float
    kernel: str
    kernel_std: float
    heuristic: bool
    radius: float
    altitude: float
    acceleration: float
    run_id: str
    results_root: Path
    flight_timeout: float

    @property
    def n_random_features(self):
        return 0 if self.method == "mpc" else self.n_rf

    def manifest_configuration(self):
        data = asdict(self)
        data["results_root"] = str(self.results_root)
        data["speed_mps"] = data.pop("speed")
        data.pop("n_rf")
        data["radius_m"] = data.pop("radius")
        data["altitude_m"] = data.pop("altitude")
        data["acceleration_mps2"] = data.pop("acceleration")
        data["flight_timeout_s"] = data.pop("flight_timeout")
        data["n_random_features"] = self.n_random_features
        return data


@dataclass(frozen=True)
class BenchmarkPlan:
    repository: Path
    generated_file: Path
    run_directory: Path
    result_file: Path
    manifest_file: Path
    controller_log: Path
    reference_log: Path
    controller_command: tuple
    reference_command: tuple


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=("mpc", "ssi-mpc"))
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-rf", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=0.25)
    parser.add_argument(
        "--kernel", default="Gaussian", choices=("Gaussian", "Laplace", "Cauchy")
    )
    parser.add_argument("--kernel-std", type=float, default=0.01)
    parser.add_argument("--heuristic", action="store_true")
    parser.add_argument("--radius", type=float, default=1.0)
    parser.add_argument("--altitude", type=float, default=1.0)
    parser.add_argument("--acceleration", type=float, default=0.25)
    parser.add_argument("--run-id", default="run_001")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("/workspaces/ssi_mpc_artifacts/formal_runs"),
    )
    parser.add_argument("--flight-timeout", type=float, default=900.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def specification_from_arguments(args):
    spec = BenchmarkSpec(
        method=args.method,
        speed=args.speed,
        seed=args.seed,
        n_rf=args.n_rf,
        learning_rate=args.learning_rate,
        kernel=args.kernel,
        kernel_std=args.kernel_std,
        heuristic=args.heuristic,
        radius=args.radius,
        altitude=args.altitude,
        acceleration=args.acceleration,
        run_id=args.run_id,
        results_root=args.results_root,
        flight_timeout=args.flight_timeout,
    )
    validate_specification(spec)
    return spec


def validate_specification(spec):
    positive_values = {
        "--speed": spec.speed,
        "--learning-rate": spec.learning_rate,
        "--kernel-std": spec.kernel_std,
        "--radius": spec.radius,
        "--altitude": spec.altitude,
        "--acceleration": spec.acceleration,
        "--flight-timeout": spec.flight_timeout,
    }
    for name, value in positive_values.items():
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    if spec.speed <= 2.0 * spec.acceleration:
        raise ValueError(
            "--speed must be greater than 2 * --acceleration for the "
            "repository's loop trajectory generator"
        )
    if spec.method == "ssi-mpc" and spec.n_rf <= 0:
        raise ValueError("SSI-MPC requires --n-rf greater than zero")
    if spec.n_rf < 0:
        raise ValueError("--n-rf must not be negative")
    if spec.seed < 0:
        raise ValueError("--seed must not be negative")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}", spec.run_id):
        raise ValueError(
            "--run-id must start with an alphanumeric character and contain "
            "only letters, digits, '.', '_' or '-' (maximum 80 characters)"
        )


def _float_token(value):
    token = format(value, ".12g")
    return token if "." in token else token + ".0"


def build_plan(spec, repository=None):
    repository = repository or Path(__file__).resolve().parents[1]
    speed_token = _float_token(spec.speed)
    method_tag = "mpc" if spec.method == "mpc" else "ssi_mpc"
    prefix = "MPC" if spec.method == "mpc" else "OLMPC"
    generated_file = (
        repository / "ros_mpc" / "benchmark" / f"{prefix}_circle_{speed_token}.mat"
    )
    run_directory = (
        spec.results_root
        / f"circle_v{speed_token}"
        / spec.run_id
        / f"{method_tag}_seed{spec.seed}"
    )
    controller_command = (
        "roslaunch",
        "ros_mpc",
        "mpc_wrapper.launch",
        "plot:=false",
        "recording:=false",
        "save_data:=true",
        f"n_rf:={spec.n_random_features}",
        f"lr:={spec.learning_rate}",
        f"heuristic:={str(spec.heuristic).lower()}",
        f"kernel:={spec.kernel}",
        f"kernel_std:={spec.kernel_std}",
        f"random_seed:={spec.seed}",
        "run_reference_generator:=false",
    )
    reference_command = (
        "rosrun",
        "ros_mpc",
        "reference_publisher_node.py",
        "__name:=safe_ref_gen",
        "_plot:=false",
        "_quad_name:=hummingbird",
        "_mode:=loop",
        "_n_seeds:=1",
        f"_loop_z:={spec.altitude}",
        f"_loop_r:={spec.radius}",
        f"_loop_v_max:={spec.speed}",
        f"_loop_lin_a:={spec.acceleration}",
        "_loop_clockwise:=false",
        "_loop_yawing:=false",
        "_t_horizon:=1.0",
        "_n_nodes:=10",
        "_control_freq_factor:=5",
    )
    return BenchmarkPlan(
        repository=repository,
        generated_file=generated_file,
        run_directory=run_directory,
        result_file=run_directory / "result.mat",
        manifest_file=run_directory / "manifest.json",
        controller_log=run_directory / "controller.log",
        reference_log=run_directory / "reference.log",
        controller_command=controller_command,
        reference_command=reference_command,
    )


def printable_configuration(spec, plan):
    configuration = spec.manifest_configuration()
    configuration.update(
        {
            "generated_file": str(plan.generated_file),
            "run_directory": str(plan.run_directory),
            "controller_command": shlex.join(plan.controller_command),
            "reference_command": shlex.join(plan.reference_command),
        }
    )
    return configuration


def print_configuration(spec, plan):
    print("===== BENCHMARK CONFIGURATION =====")
    print(json.dumps(printable_configuration(spec, plan), indent=2, sort_keys=True))
    print("===================================")


def _saved_value(metadata, key):
    if key not in metadata:
        raise ValueError(f"Saved result is missing metadata: {key}")
    return metadata[key]


def _saved_bool(value):
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("true", "1"):
            return True
        if lowered in ("false", "0"):
            return False
        raise ValueError(f"Invalid saved boolean value: {value}")
    return bool(value)


def verify_metadata(spec, metadata, metrics):
    required = (
        "project_mean_distance_m",
        "mean_optimization_time_s",
        "n_random_features",
        "learning_rate",
        "kernel",
        "kernel_std",
        "heuristic",
        "random_seed",
        "trajectory_name",
        "reference_speed_mps",
        "metric_definition",
    )
    missing = [key for key in required if key not in metadata]
    if missing:
        raise ValueError(f"Saved result is missing metadata: {missing}")

    exact_checks = {
        "n_random_features": (int, spec.n_random_features),
        "random_seed": (int, spec.seed),
        "kernel": (str, spec.kernel),
        "trajectory_name": (str, "circle"),
        "metric_definition": (str, "mean_euclidean_position_error"),
    }
    for key, (conversion, expected) in exact_checks.items():
        actual = conversion(_saved_value(metadata, key))
        if actual != expected:
            raise ValueError(f"Saved {key} is {actual!r}, expected {expected!r}")

    numeric_checks = {
        "learning_rate": spec.learning_rate,
        "kernel_std": spec.kernel_std,
        "reference_speed_mps": spec.speed,
        "project_mean_distance_m": metrics["mean_distance_m"],
    }
    for key, expected in numeric_checks.items():
        actual = float(_saved_value(metadata, key))
        if not np.isclose(actual, expected, atol=1e-12, rtol=1e-9):
            raise ValueError(f"Saved {key} is {actual}, expected {expected}")

    if _saved_bool(_saved_value(metadata, "heuristic")) != spec.heuristic:
        raise ValueError("Saved heuristic does not match the requested value")
    optimization_time = float(_saved_value(metadata, "mean_optimization_time_s"))
    if not math.isfinite(optimization_time) or optimization_time <= 0.0:
        raise ValueError("Saved mean_optimization_time_s is invalid")


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _git_snapshot(repository):
    def run(*arguments):
        return subprocess.check_output(
            ("git", "-C", str(repository), *arguments),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()

    try:
        return {
            "commit": run("rev-parse", "HEAD"),
            "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "status_porcelain": run("status", "--porcelain").splitlines(),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "branch": None, "status_porcelain": []}


def write_manifest(path, manifest):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, sort_keys=True)
        file.write("\n")
    temporary.replace(path)


def _base_manifest(spec, plan):
    return {
        "schema_version": 2,
        "started_utc": _utc_now(),
        "status": "STARTING",
        "configuration": spec.manifest_configuration(),
        "commands": {
            "controller": shlex.join(plan.controller_command),
            "reference": shlex.join(plan.reference_command),
        },
        "software": {
            "git": _git_snapshot(plan.repository),
            "python": sys.version,
            "ros_distro": os.environ.get("ROS_DISTRO"),
            "acados_source_dir": os.environ.get("ACADOS_SOURCE_DIR"),
        },
        "logs": {
            "controller": str(plan.controller_log),
            "reference": str(plan.reference_log),
        },
    }


def run_benchmark(spec, plan):
    if plan.generated_file.exists():
        raise ValueError(f"Refusing to overwrite generated file: {plan.generated_file}")
    if plan.run_directory.exists():
        raise ValueError(f"Refusing to reuse run directory: {plan.run_directory}")
    plan.run_directory.mkdir(parents=True, exist_ok=False)
    manifest = _base_manifest(spec, plan)
    write_manifest(plan.manifest_file, manifest)

    from ros_runtime import run_flight

    try:
        outcome = run_flight(
            controller_command=plan.controller_command,
            reference_command=plan.reference_command,
            controller_log=plan.controller_log,
            reference_log=plan.reference_log,
            radius=spec.radius,
            altitude=spec.altitude,
            reference_speed=spec.speed,
            flight_timeout=spec.flight_timeout,
        )
    except KeyboardInterrupt:
        manifest.update(
            {"status": "INTERRUPTED", "finished_utc": _utc_now(), "exit_code": 130}
        )
        write_manifest(plan.manifest_file, manifest)
        raise

    manifest["flight"] = asdict(outcome)
    status = outcome.status
    error = outcome.error
    exit_code = 1

    if outcome.completed:
        if not plan.generated_file.exists():
            status = "DATA VALIDATION FAILED"
            error = f"Expected result file is missing: {plan.generated_file}"
        else:
            try:
                validated = load_and_validate(plan.generated_file)
                metrics = calculate_metrics(validated)
                verify_metadata(spec, validated["metadata"], metrics)
                shutil.move(str(plan.generated_file), str(plan.result_file))
                moved = load_and_validate(plan.result_file)
                manifest["result"] = {
                    "path": str(plan.result_file),
                    "sha256": moved["sha256"],
                    "metadata": moved["metadata"],
                    "metrics": calculate_metrics(moved),
                }
                status = "COMPLETE"
                error = ""
                exit_code = 0
            except Exception as exception:
                status = "DATA VALIDATION FAILED"
                error = str(exception)

    if plan.generated_file.exists():
        manifest["diagnostic_generated_file"] = str(plan.generated_file)
    manifest.update(
        {
            "status": status,
            "error": error,
            "finished_utc": _utc_now(),
            "exit_code": exit_code,
        }
    )
    write_manifest(plan.manifest_file, manifest)

    print("========================================")
    print("FINAL RESULT:", status)
    print("INTERFACE FINAL STATE:", "LOCKED" if outcome.final_lock_confirmed else "UNCONFIRMED")
    print("GAZEBO FINAL STATE:", "PAUSED" if outcome.paused else "RUNNING")
    if exit_code == 0:
        metrics = manifest["result"]["metrics"]
        metadata = manifest["result"]["metadata"]
        print("DATA VALIDATION: PASSED")
        print("RESULT FILE:", plan.result_file)
        print("RESULT SHA256:", manifest["result"]["sha256"])
        print(f"PROJECT MEAN DISTANCE: {metrics['mean_distance_m']:.9f} m")
        print(f"CONVENTIONAL VECTOR RMSE: {metrics['vector_rmse_m']:.9f} m")
        optimization_ms = float(metadata["mean_optimization_time_s"]) * 1000.0
        print(f"MEAN OPTIMIZATION TIME: {optimization_ms:.6f} ms")
    else:
        print("ERROR:", error or status)
    print("MANIFEST:", plan.manifest_file)
    print("========================================")
    return exit_code


def _require_environment():
    required = ("ROS_MASTER_URI", "ACADOS_SOURCE_DIR")
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise RuntimeError(
            f"Missing environment variables: {missing}. Source setup_reproduction.sh first."
        )


def main(argv=None):
    args = parse_arguments(argv)
    spec = specification_from_arguments(args)
    plan = build_plan(spec)
    print_configuration(spec, plan)
    if args.dry_run:
        print("DRY RUN PASSED - NO FLIGHT STARTED")
        return 0
    _require_environment()
    return run_benchmark(spec, plan)


if __name__ == "__main__":
    try:
        code = main()
    except KeyboardInterrupt:
        print("BENCHMARK INTERRUPTED", file=sys.stderr)
        code = 130
    except Exception as exception:
        print("BENCHMARK FAILED:", exception, file=sys.stderr)
        code = 1
    raise SystemExit(code)
