from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation


def load_calibration_config(path: str | Path) -> dict:
    cfg_path = Path(path).expanduser().resolve()
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid calibration config: {cfg_path}")
    return cfg


def intrinsics_from_config(cam_cfg: dict) -> dict:
    intr = dict((cam_cfg or {}).get("intrinsics") or {})
    return {
        "fx": float(intr.get("fx", 0.0)),
        "fy": float(intr.get("fy", 0.0)),
        "cx": float(intr.get("cx", 0.0)),
        "cy": float(intr.get("cy", 0.0)),
        "width": int(intr.get("width", cam_cfg.get("width", 0))),
        "height": int(intr.get("height", cam_cfg.get("height", 0))),
        "depth_scale": float(intr.get("depth_scale", 0.0)),
    }


def intrinsics_from_array(values: np.ndarray | list | tuple) -> dict:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    padded = np.zeros(7, dtype=np.float32)
    padded[: min(arr.size, 7)] = arr[:7]
    return {
        "fx": float(padded[0]),
        "fy": float(padded[1]),
        "cx": float(padded[2]),
        "cy": float(padded[3]),
        "width": int(round(float(padded[4]))),
        "height": int(round(float(padded[5]))),
        "depth_scale": float(padded[6]),
    }


def transform_from_config(cam_cfg: dict, key: str) -> np.ndarray:
    raw = ((cam_cfg or {}).get(key) or {}).get("data")
    if raw is None:
        print(f"[WARN] Missing calibration matrix {key}; falling back to identity.")
        return np.eye(4, dtype=np.float32)
    return np.asarray(raw, dtype=np.float32).reshape(4, 4)


def has_valid_intrinsics(intrinsics: dict) -> bool:
    intrinsics = intrinsics or {}
    return (
        abs(float(intrinsics.get("fx", 0.0))) > 1e-8
        and abs(float(intrinsics.get("fy", 0.0))) > 1e-8
        and int(intrinsics.get("width", 0)) > 0
        and int(intrinsics.get("height", 0)) > 0
    )


def ee_pose_to_T_base_ee(ee_pose: np.ndarray) -> np.ndarray:
    ee_pose = np.asarray(ee_pose, dtype=np.float32).reshape(-1)
    if ee_pose.size < 6:
        return np.eye(4, dtype=np.float32)
    rot = Rotation.from_euler("xyz", ee_pose[3:6], degrees=False).as_matrix().astype(np.float32)
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = rot
    T[:3, 3] = ee_pose[:3]
    return T
