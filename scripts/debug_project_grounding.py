#!/usr/bin/env python3

import argparse
import csv
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from realman_vla.geometry.calibration import (  # noqa: E402
    ee_pose_to_T_base_ee,
    has_valid_intrinsics,
    intrinsics_from_array,
    intrinsics_from_config,
    load_calibration_config,
    transform_from_config,
)
from realman_vla.geometry.crossview import project_exo_roi_to_ego  # noqa: E402
from realman_vla.vision.target_locator import TargetLocator  # noqa: E402


def save_roi_image(image, roi, out_path, label=""):
    canvas = image.copy()
    roi = np.asarray(roi, dtype=np.int32).reshape(4)
    if roi[2] > roi[0] and roi[3] > roi[1]:
        cv2.rectangle(canvas, (roi[0], roi[1]), (roi[2], roi[3]), (0, 255, 0), 2)
    if label:
        cv2.putText(canvas, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.imwrite(str(out_path), canvas)


def median_depth_in_roi(depth, roi_xyxy, depth_scale):
    roi = np.asarray(roi_xyxy, dtype=np.int32).reshape(4)
    if roi[2] <= roi[0] or roi[3] <= roi[1]:
        return 0.0
    patch = depth[max(0, roi[1]):max(0, roi[3]), max(0, roi[0]):max(0, roi[2])]
    if patch.size == 0:
        return 0.0
    valid_depth = patch[patch > 0]
    if valid_depth.size == 0:
        return 0.0
    return float(np.median(valid_depth.astype(np.float32) * depth_scale))


def main():
    parser = argparse.ArgumentParser(description="调试 Ego-Exo grounding 投影")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--calib", type=Path, default=Path("configs/calibration_realman.yaml"))
    parser.add_argument("--runtime-config", type=Path, default=Path("configs/egexo_runtime.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=50)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    calib_cfg = load_calibration_config(args.calib)
    with args.runtime_config.open("r", encoding="utf-8") as f:
        runtime_cfg = yaml.safe_load(f) or {}

    exo_intrinsics = intrinsics_from_config(calib_cfg.get("cameras", {}).get("cam_high", {}))
    ego_intrinsics = intrinsics_from_config(calib_cfg.get("cameras", {}).get("cam_wrist", {}))
    T_base_exo = transform_from_config(calib_cfg.get("cameras", {}).get("cam_high", {}), "T_base_cam")
    T_ee_ego = transform_from_config(calib_cfg.get("cameras", {}).get("cam_wrist", {}), "T_ee_cam")

    rows = []
    valid_count = 0
    area_list = []
    depth_list = []
    out_of_view = 0
    locator = TargetLocator(runtime_cfg.get("target_locator", {}))

    with h5py.File(args.input, "r") as f:
        cam_high = np.asarray(f["observations/images/cam_high"])
        cam_wrist = np.asarray(f["observations/images/cam_wrist"])
        depth_high = np.asarray(f["observations/depth/cam_high"])
        ee_pose = np.asarray(f["observations/ee_pose"])
        stored_roi = np.asarray(f["observations/grounding/target_roi_exo"]) if "observations/grounding/target_roi_exo" in f else None
        stored_valid = np.asarray(f["observations/grounding/valid"]) if "observations/grounding/valid" in f else None

        if not has_valid_intrinsics(exo_intrinsics) and "metadata/cameras/cam_high/intrinsics" in f:
            exo_intrinsics = intrinsics_from_array(np.asarray(f["metadata/cameras/cam_high/intrinsics"]))
        if not has_valid_intrinsics(ego_intrinsics) and "metadata/cameras/cam_wrist/intrinsics" in f:
            ego_intrinsics = intrinsics_from_array(np.asarray(f["metadata/cameras/cam_wrist/intrinsics"]))

        max_frames = min(args.max_frames, cam_high.shape[0])
        for idx in range(max_frames):
            locator_result = locator.locate(cam_high[idx], depth=depth_high[idx])
            exo_roi = np.asarray(locator_result.get("roi_xyxy", np.zeros(4, dtype=np.float32)), dtype=np.float32)
            roi_source = "runtime"
            if stored_roi is not None and idx < stored_roi.shape[0]:
                candidate_roi = np.asarray(stored_roi[idx], dtype=np.float32).reshape(-1)[:4]
                candidate_valid = float(np.asarray(stored_valid[idx]).reshape(-1)[0]) > 0.5 if stored_valid is not None else np.any(candidate_roi > 0)
                if candidate_valid and np.any(candidate_roi > 0):
                    exo_roi = candidate_roi
                    locator_result = {"valid": True, "reason": "stored_hdf5_roi"}
                    roi_source = "hdf5"

            result = project_exo_roi_to_ego(
                exo_roi_xyxy=exo_roi,
                exo_depth=depth_high[idx],
                exo_intrinsics=exo_intrinsics,
                ego_intrinsics=ego_intrinsics,
                T_base_exo=T_base_exo,
                T_base_ee=ee_pose_to_T_base_ee(ee_pose[idx]),
                T_ee_ego=T_ee_ego,
                image_size=cam_wrist[idx].shape[:2],
                cfg=calib_cfg.get("geometry", {}),
            )

            ego_roi = np.asarray(result["ego_roi_xyxy"], dtype=np.float32)
            is_valid = bool(result["valid"])
            valid_count += int(is_valid)
            roi_out_of_view = bool(result.get("roi_out_of_view", False))
            if is_valid:
                area = float(max(0.0, ego_roi[2] - ego_roi[0]) * max(0.0, ego_roi[3] - ego_roi[1]))
                area_list.append(area)
            if roi_out_of_view:
                out_of_view += 1

            depth_scale = float(exo_intrinsics.get("depth_scale", 0.001) or 0.001)
            depth_median = median_depth_in_roi(depth_high[idx], exo_roi, depth_scale)
            if depth_median > 0:
                depth_list.append(depth_median)

            exo_label = f"exo:{locator_result.get('reason', 'unknown')} src={roi_source}"
            ego_label = f"ego:{result.get('reason', 'unknown')}"
            save_roi_image(cam_high[idx], exo_roi, args.output / f"frame_{idx:03d}_exo_roi.jpg", label=exo_label)
            save_roi_image(cam_wrist[idx], ego_roi, args.output / f"frame_{idx:03d}_ego_projected_roi.jpg", label=ego_label)

            rows.append(
                {
                    "frame": idx,
                    "valid": int(is_valid),
                    "roi_source": roi_source,
                    "locator_reason": locator_result.get("reason", "unknown"),
                    "projection_reason": result.get("reason", "unknown"),
                    "roi_out_of_view": int(roi_out_of_view),
                    "median_depth_m": depth_median,
                    "target_x": float(np.asarray(result.get("target_3d_base", np.zeros(3)))[0]),
                    "target_y": float(np.asarray(result.get("target_3d_base", np.zeros(3)))[1]),
                    "target_z": float(np.asarray(result.get("target_3d_base", np.zeros(3)))[2]),
                    "exo_roi_x1": float(exo_roi[0]) if exo_roi.size == 4 else 0.0,
                    "exo_roi_y1": float(exo_roi[1]) if exo_roi.size == 4 else 0.0,
                    "exo_roi_x2": float(exo_roi[2]) if exo_roi.size == 4 else 0.0,
                    "exo_roi_y2": float(exo_roi[3]) if exo_roi.size == 4 else 0.0,
                    "ego_roi_x1": float(ego_roi[0]) if ego_roi.size == 4 else 0.0,
                    "ego_roi_y1": float(ego_roi[1]) if ego_roi.size == 4 else 0.0,
                    "ego_roi_x2": float(ego_roi[2]) if ego_roi.size == 4 else 0.0,
                    "ego_roi_y2": float(ego_roi[3]) if ego_roi.size == 4 else 0.0,
                }
            )

    summary_path = args.output / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["frame", "valid"])
        writer.writeheader()
        writer.writerows(rows)

    total = max(1, len(rows))
    print(f"valid_rate={valid_count / total:.3f}")
    print(f"roi_out_of_view_rate={out_of_view / total:.3f}")
    print(f"mean_roi_area={float(np.mean(area_list)) if area_list else 0.0:.3f}")
    print(f"median_depth={float(np.median(depth_list)) if depth_list else 0.0:.3f}")
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
