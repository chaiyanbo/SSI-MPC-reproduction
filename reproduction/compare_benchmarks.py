#!/usr/bin/env python3
"""Validate and compare paired MPC and SSI-MPC MATLAB result files."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ARRAY_KEYS = ("ref_time", "ref_x", "ref_u", "x", "u", "w_control")
METADATA_KEYS = (
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
ERROR_METRICS = (
    ("Project mean distance", "mean_distance_m"),
    ("Conventional vector RMSE", "vector_rmse_m"),
    ("X-axis RMSE", "x_rmse_m"),
    ("Y-axis RMSE", "y_rmse_m"),
    ("Z-axis RMSE", "z_rmse_m"),
    ("Median position error", "median_error_m"),
    ("95th-percentile error", "p95_error_m"),
    ("Maximum position error", "max_error_m"),
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def matlab_value(value):
    """Convert a scipy.io.loadmat value to JSON-safe Python data."""
    array = np.asarray(value)
    if array.dtype.kind in ("U", "S"):
        return "".join(str(item) for item in array.reshape(-1).tolist())
    if array.size == 1:
        item = array.item()
        return item.item() if isinstance(item, np.generic) else item
    return array.tolist()


def _require_shape(path, name, array, expected):
    if array.shape != expected:
        raise ValueError(
            f"{path}: {name} shape is {array.shape}, expected {expected}"
        )


def load_and_validate(path):
    """Load one result file and reject incomplete or non-physical arrays."""
    from scipy.io import loadmat

    path = Path(path)
    if not path.is_file():
        raise ValueError(f"Result file does not exist: {path}")

    data = loadmat(str(path))
    missing = [key for key in ARRAY_KEYS if key not in data]
    if missing:
        raise ValueError(f"{path} is missing arrays: {missing}")

    arrays = {
        "ref_time": np.asarray(data["ref_time"]).reshape(-1),
        "ref_x": np.asarray(data["ref_x"]),
        "ref_u": np.asarray(data["ref_u"]),
        "x": np.asarray(data["x"]),
        "u": np.asarray(data["u"]),
        "w_control": np.asarray(data["w_control"]),
    }
    sample_count = arrays["ref_time"].shape[0]
    if sample_count < 2:
        raise ValueError(f"{path}: not enough samples")

    expected_shapes = {
        "ref_time": (sample_count,),
        "ref_x": (sample_count, 13),
        "ref_u": (sample_count, 4),
        "x": (sample_count, 13),
        "u": (sample_count, 4),
        "w_control": (sample_count, 3),
    }
    for name, expected in expected_shapes.items():
        _require_shape(path, name, arrays[name], expected)
        if not np.isfinite(arrays[name]).all():
            raise ValueError(f"{path}: {name} contains NaN or Inf")

    dt = np.diff(arrays["ref_time"])
    if not np.all(dt > 0.0):
        raise ValueError(f"{path}: reference time is not strictly increasing")

    zero_state_rows = np.flatnonzero(
        np.all(np.isclose(arrays["x"], 0.0, atol=1e-12), axis=1)
    )
    zero_input_rows = np.flatnonzero(
        np.all(np.isclose(arrays["u"], 0.0, atol=1e-12), axis=1)
    )
    if zero_state_rows.size:
        raise ValueError(f"{path}: zero state rows {zero_state_rows.tolist()}")
    if zero_input_rows.size:
        raise ValueError(f"{path}: zero input rows {zero_input_rows.tolist()}")

    quaternion_norm = np.linalg.norm(arrays["x"][:, 3:7], axis=1)
    quaternion_deviation = float(np.max(np.abs(quaternion_norm - 1.0)))
    if quaternion_deviation > 1e-5:
        raise ValueError(
            f"{path}: quaternion deviation is {quaternion_deviation}"
        )

    metadata = {
        key: matlab_value(data[key]) for key in METADATA_KEYS if key in data
    }
    return {
        "path": path,
        "sha256": sha256(path),
        "arrays": arrays,
        "metadata": metadata,
    }


def calculate_metrics(result):
    arrays = result["arrays"]
    ref_time = arrays["ref_time"]
    position_error = arrays["ref_x"][:, :3] - arrays["x"][:, :3]
    distance_error = np.linalg.norm(position_error, axis=1)
    axis_rmse = np.sqrt(np.mean(position_error ** 2, axis=0))
    speed = np.linalg.norm(arrays["x"][:, 7:10], axis=1)
    position_step = np.linalg.norm(np.diff(arrays["x"][:, :3], axis=0), axis=1)
    quaternion_norm = np.linalg.norm(arrays["x"][:, 3:7], axis=1)

    return {
        "sample_count": int(ref_time.shape[0]),
        "duration_s": float(ref_time[-1] - ref_time[0]),
        "mean_dt_s": float(np.mean(np.diff(ref_time))),
        "mean_distance_m": float(np.mean(distance_error)),
        "vector_rmse_m": float(np.sqrt(np.mean(distance_error ** 2))),
        "x_rmse_m": float(axis_rmse[0]),
        "y_rmse_m": float(axis_rmse[1]),
        "z_rmse_m": float(axis_rmse[2]),
        "median_error_m": float(np.median(distance_error)),
        "p95_error_m": float(np.percentile(distance_error, 95)),
        "max_error_m": float(np.max(distance_error)),
        "max_speed_mps": float(np.max(speed)),
        "mean_speed_mps": float(np.mean(speed)),
        "max_position_step_m": float(np.max(position_step)),
        "rotor_input_min": float(np.min(arrays["u"])),
        "rotor_input_max": float(np.max(arrays["u"])),
        "rotor_input_rms": float(np.sqrt(np.mean(arrays["u"] ** 2))),
        "body_rate_rms": float(np.sqrt(np.mean(arrays["w_control"] ** 2))),
        "quaternion_max_deviation": float(
            np.max(np.abs(quaternion_norm - 1.0))
        ),
    }


def compare_results(mpc_result, ssi_result):
    """Compare validated runs and require bit-identical references."""
    for key in ("ref_time", "ref_x", "ref_u"):
        mpc_array = mpc_result["arrays"][key]
        ssi_array = ssi_result["arrays"][key]
        if mpc_array.shape != ssi_array.shape:
            raise ValueError(
                f"Reference {key} shape differs: "
                f"{mpc_array.shape} vs {ssi_array.shape}"
            )
        if not np.array_equal(mpc_array, ssi_array):
            maximum = float(np.max(np.abs(mpc_array - ssi_array)))
            raise ValueError(
                f"Reference {key} differs; maximum difference is {maximum}"
            )

    results = {"MPC": mpc_result, "SSI-MPC": ssi_result}
    metrics = {name: calculate_metrics(result) for name, result in results.items()}
    reductions = {}
    for _, key in ERROR_METRICS:
        baseline = metrics["MPC"][key]
        if baseline <= 0.0:
            raise ValueError(f"Cannot calculate reduction for zero baseline: {key}")
        reductions[key] = (
            (baseline - metrics["SSI-MPC"][key]) / baseline * 100.0
        )

    return {
        "schema_version": 1,
        "reference_arrays_identical": True,
        "results": {
            name: {
                "path": str(result["path"]),
                "sha256": result["sha256"],
                "metadata": result["metadata"],
                "metrics": metrics[name],
            }
            for name, result in results.items()
        },
        "reductions_pct": reductions,
    }


def write_json(path, comparison):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(comparison, file, indent=2, sort_keys=True)
        file.write("\n")
    temporary.replace(path)


def write_csv(path, comparison):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    mpc_metrics = comparison["results"]["MPC"]["metrics"]
    ssi_metrics = comparison["results"]["SSI-MPC"]["metrics"]
    reductions = comparison["reductions_pct"]
    with temporary.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(("metric", "mpc", "ssi_mpc", "reduction_pct"))
        for key in mpc_metrics:
            writer.writerow(
                (key, mpc_metrics[key], ssi_metrics[key], reductions.get(key, ""))
            )
    temporary.replace(path)


def _print_comparison(comparison):
    metrics = {
        name: result["metrics"]
        for name, result in comparison["results"].items()
    }
    reductions = comparison["reductions_pct"]
    print("REFERENCE ARRAYS IDENTICAL\n")
    print(f"{'Metric':<30}{'MPC (m)':>14}{'SSI-MPC (m)':>16}{'Reduction':>14}")
    for label, key in ERROR_METRICS:
        print(
            f"{label:<30}{metrics['MPC'][key]:>14.9f}"
            f"{metrics['SSI-MPC'][key]:>16.9f}{reductions[key]:>13.2f}%"
        )
    print(f"\nMPC maximum speed: {metrics['MPC']['max_speed_mps']:.9f} m/s")
    print(
        "SSI-MPC maximum speed: "
        f"{metrics['SSI-MPC']['max_speed_mps']:.9f} m/s"
    )
    print(f"Mean-distance reduction: {reductions['mean_distance_m']:.3f}%")
    print(
        "Conventional-RMSE reduction: "
        f"{reductions['vector_rmse_m']:.3f}%"
    )


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mpc_file", type=Path)
    parser.add_argument("ssi_mpc_file", type=Path)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--csv-out", type=Path)
    return parser.parse_args()


def main():
    args = parse_arguments()
    comparison = compare_results(
        load_and_validate(args.mpc_file),
        load_and_validate(args.ssi_mpc_file),
    )
    _print_comparison(comparison)
    if args.json_out is not None:
        write_json(args.json_out, comparison)
        print("JSON:", args.json_out)
    if args.csv_out is not None:
        write_csv(args.csv_out, comparison)
        print("CSV:", args.csv_out)
    print("PAIRWISE DATA COMPARISON PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
