#!/usr/bin/env python3

import argparse
from pathlib import Path

import h5py
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="检查 Ego-Exo HDF5 完整性")
    parser.add_argument("--input", type=Path, required=True, help="输入 HDF5 文件")
    args = parser.parse_args()

    required = [
        "observations/qpos",
        "action",
        "observations/images/cam_high",
        "observations/images/cam_wrist",
        "observations/depth/cam_high",
        "observations/ee_pose",
        "metadata/cameras/cam_high/intrinsics",
        "metadata/cameras/cam_high/T_base_cam",
        "metadata/cameras/cam_wrist/intrinsics",
        "metadata/cameras/cam_wrist/T_ee_cam",
    ]

    with h5py.File(args.input, "r") as f:
        print(f"文件: {args.input}")
        for key in required:
            print(f"{key}: {'OK' if key in f else 'MISSING'}")

        qpos = np.asarray(f["observations/qpos"])
        action = np.asarray(f["action"])
        cam_high = np.asarray(f["observations/images/cam_high"])
        cam_wrist = np.asarray(f["observations/images/cam_wrist"])
        depth = np.asarray(f["observations/depth/cam_high"])
        ee_pose = np.asarray(f["observations/ee_pose"])

        print(f"qpos shape: {qpos.shape}")
        print(f"action shape: {action.shape}")
        print(f"cam_high shape: {cam_high.shape}")
        print(f"cam_wrist shape: {cam_wrist.shape}")
        print(f"depth shape: {depth.shape}")
        print(f"ee_pose shape: {ee_pose.shape}")

        consistent = len({qpos.shape[0], action.shape[0], cam_high.shape[0], cam_wrist.shape[0], depth.shape[0], ee_pose.shape[0]}) == 1
        print(f"frames consistent: {consistent}")
        print(f"qpos/action 7D: {qpos.shape[-1] == 7 and action.shape[-1] == 7}")
        print(f"depth all zero: {bool(np.all(depth == 0))}")
        print(f"wrist dark ratio: {float(np.mean(cam_wrist.mean(axis=(1,2,3)) < 3.0)):.3f}")


if __name__ == "__main__":
    main()
