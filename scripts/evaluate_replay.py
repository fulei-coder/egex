#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def evaluate_raw_hdf5(input_path: Path) -> dict:
    import h5py

    files = [input_path] if input_path.is_file() else sorted(input_path.glob("*.hdf5"))
    if not files:
        raise FileNotFoundError(f"No HDF5 episodes found under: {input_path}")

    action_shapes = Counter()
    grounding_valid = []
    phase_values = []
    for path in files:
        with h5py.File(path, "r") as f:
            qpos = np.asarray(f["observations/qpos"])
            action = np.asarray(f["action"])
            action_shapes[str(tuple(action.shape[1:]))] += 1
            if "observations/grounding/valid" in f:
                grounding_valid.append(np.asarray(f["observations/grounding/valid"]).reshape(-1))
            if "observations/phase" in f:
                phase_values.append(np.asarray(f["observations/phase"]).reshape(-1))
            elif "observations/ee_pose" in f and "observations/grounding/target_3d_base" in f:
                ee = np.asarray(f["observations/ee_pose"])
                tgt = np.asarray(f["observations/grounding/target_3d_base"])
                valid = np.asarray(f["observations/grounding/valid"]).reshape(-1) if "observations/grounding/valid" in f else np.zeros((ee.shape[0],), dtype=np.float32)
                dist = np.linalg.norm(ee[:, :3] - tgt[:, :3], axis=-1)
                phase = np.where((valid > 0.5) & (dist <= 0.08), 1.0, 0.0)
                phase_values.append(phase.astype(np.float32))
            if qpos.shape[-1] != 7 or action.shape[-1] != 7:
                raise ValueError(f"{path} does not keep the required 7D qpos/action interface.")

    grounding = np.concatenate(grounding_valid) if grounding_valid else np.zeros((0,), dtype=np.float32)
    phase = np.concatenate(phase_values) if phase_values else np.zeros((0,), dtype=np.float32)
    return {
        "mode": "raw_hdf5",
        "episodes": len(files),
        "action_shape_histogram": dict(action_shapes),
        "grounding_valid_rate": float(np.mean(grounding > 0.5)) if grounding.size else 0.0,
        "phase_distribution": {
            "transport_0": float(np.mean(phase < 0.5)) if phase.size else 0.0,
            "contact_1": float(np.mean(phase >= 0.5)) if phase.size else 0.0,
        },
        "action_dim_ok": True,
    }


def evaluate_lerobot_dataset(root: Path) -> dict:
    meta_dir = root / "meta"
    info_path = meta_dir / "info.json"
    episodes_path = meta_dir / "episodes.jsonl"
    stats_path = meta_dir / "stats.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Missing {info_path}")

    info = json.loads(info_path.read_text(encoding="utf-8"))
    features = info.get("features", {})
    action_feature = features.get("action", {})
    grounding_feature = features.get("observation.grounding.valid", {})
    phase_feature = features.get("observation.phase", {})

    result = {
        "mode": "lerobot_dataset",
        "dataset_root": str(root),
        "episodes_index_present": episodes_path.exists(),
        "stats_present": stats_path.exists(),
        "action_shape": action_feature.get("shape"),
        "action_dim_ok": action_feature.get("shape") == [7] or tuple(action_feature.get("shape", [])) == (7,),
        "grounding_feature_present": bool(grounding_feature),
        "phase_feature_present": bool(phase_feature),
    }

    try:
        import pyarrow.parquet as pq
    except Exception:
        result["note"] = "pyarrow not available; skipped parquet-level replay inspection."
        return result

    parquet_files = sorted((root / "data").glob("*.parquet"))
    if not parquet_files:
        result["note"] = "No parquet files found; skipped row-level inspection."
        return result

    grounding_values = []
    phase_values = []
    for path in parquet_files:
        table = pq.read_table(path, columns=[c for c in ("observation.grounding.valid", "observation.phase") if c in table_column_names(path)])
        if "observation.grounding.valid" in table.column_names:
            grounding_values.append(np.asarray(table["observation.grounding.valid"]).reshape(-1))
        if "observation.phase" in table.column_names:
            phase_values.append(np.asarray(table["observation.phase"]).reshape(-1))

    grounding = np.concatenate(grounding_values) if grounding_values else np.zeros((0,), dtype=np.float32)
    phase = np.concatenate(phase_values) if phase_values else np.zeros((0,), dtype=np.float32)
    result["grounding_valid_rate"] = float(np.mean(grounding > 0.5)) if grounding.size else 0.0
    result["phase_distribution"] = {
        "transport_0": float(np.mean(phase < 0.5)) if phase.size else 0.0,
        "contact_1": float(np.mean(phase >= 0.5)) if phase.size else 0.0,
    }
    return result


def table_column_names(path: Path) -> list[str]:
    import pyarrow.parquet as pq

    return pq.read_schema(path).names


def main():
    parser = argparse.ArgumentParser(description="Offline replay sanity check for Ego-Exo datasets")
    parser.add_argument("--input", type=Path, required=True, help="Raw HDF5 file/dir or LeRobot dataset root")
    args = parser.parse_args()

    input_path = args.input.resolve()
    if input_path.is_file() and input_path.suffix == ".hdf5":
        result = evaluate_raw_hdf5(input_path)
    elif input_path.is_dir() and (input_path / "meta" / "info.json").exists():
        result = evaluate_lerobot_dataset(input_path)
    elif input_path.is_dir():
        result = evaluate_raw_hdf5(input_path)
    else:
        raise FileNotFoundError(f"Unsupported input path: {input_path}")

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
