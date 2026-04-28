#!/usr/bin/env python3
"""
HDF5 → LeRobot v3 数据集转换

将采集脚本生成的 HDF5 文件转换为 LeRobot 训练所需的标准格式。
转换过程包括：
  1. 读取 HDF5 中的关节角、夹爪状态、双相机图像
  2. 创建 LeRobot Dataset（含视频编码、归一化统计）
  3. 逐帧写入并保存

⚠️ 重要：--fps 必须与采集时的帧率一致！

第一次需要  pip install lerobot

用法:

cd ~/lerobot-realman-vla
python scripts/convert_to_lerobot.py \
  --input-dir data/raw_hdf5 \
  --output-dir data/pick_cube_15fps \
  --repo-id lerobot/pick_cube_15fps \
  --fps 15 \
  --task "pick up the cube and place it in the basket"

  

python scripts/convert_to_lerobot.py \
  --input-dir data/raw_hdf5 \
  --output-dir data/pick_yellow_flower_15fps \
  --repo-id tony/pick_yellow_flower_15fps \
  --fps 15 \
  --task "reach out, bow to pick the yellow flower and return to the original position"

  
如果原始 hdf5 存到这个盘，转换时也用这个路径：

python scripts/convert_to_lerobot.py \
  --input-dir /media/a104/1252BAD252BABA35/raw_hdf5/pick_cube \
  --output-dir /media/a104/1252BAD252BABA35/pick_cube_15fps \
  --repo-id local/pick_cube_15fps \
  --fps 15 \
  --task "pick up the cube and place it in the basket"
"""

import h5py
import numpy as np
from pathlib import Path
import argparse
import sys
import shutil
import torch
from PIL import Image

# 确保 LeRobot 可以被导入
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from realman_vla.geometry.calibration import (
    ee_pose_to_T_base_ee,
    intrinsics_from_config,
    load_calibration_config,
    transform_from_config,
)
from realman_vla.geometry.crossview import project_exo_roi_to_ego
from realman_vla.vision.target_locator import locate_target_roi


def convert_hdf5_episode(hdf5_path: Path):
    """读取单个 HDF5 文件，返回字典"""
    with h5py.File(hdf5_path, 'r') as f:
        data = {
            'qpos': np.array(f['observations/qpos']),        # (N, 6) 或 (N, 7)
            'action': np.array(f['action']),                  # (N, 6) 或 (N, 7)
            'cam_high': np.array(f['observations/images/cam_high']),    # (N, H, W, 3)
            'cam_wrist': np.array(f['observations/images/cam_wrist']),  # (N, H, W, 3)
        }
        optional_keys = {
            'depth_high': 'observations/depth/cam_high',
            'ee_pose': 'observations/ee_pose',
            'target_roi_exo': 'observations/grounding/target_roi_exo',
            'target_3d_base': 'observations/grounding/target_3d_base',
            'grounding_valid': 'observations/grounding/valid',
        }
        for out_key, hdf5_key in optional_keys.items():
            if hdf5_key in f:
                data[out_key] = np.array(f[hdf5_key])
    return data


def make_joint_feature_names(dim: int):
    """根据 6D/7D 数据维度生成 LeRobot feature names。"""
    if dim == 6:
        return [f"joint_{i}" for i in range(1, 7)]
    if dim == 7:
        return [f"joint_{i}" for i in range(1, 7)] + ["gripper"]
    raise ValueError(
        f"仅支持 6D 或 7D qpos/action，当前维度为 {dim}。"
        "6D 来自 collect_data.py，7D 来自 collect_data_uarm.py。"
    )


def main():
    parser = argparse.ArgumentParser(description='HDF5 → LeRobot 数据集转换')
    parser.add_argument("--input-dir", type=Path, required=True,
                        help="HDF5 文件所在目录")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="LeRobot 数据集输出目录")
    parser.add_argument("--repo-id", type=str, required=True,
                        help="数据集标识符 (如 lerobot/pick_cube_30fps)")
    parser.add_argument("--fps", type=int, default=30,
                        help="帧率，应与采集时一致（默认30）")
    parser.add_argument("--task", type=str, default="pick up the cube and place it in the basket",
                        help="任务描述（VLA策略需要，必须与推理时完全一致）")
    parser.add_argument("--enable-egexo", action="store_true", help="启用 Ego-Exo 附加特征转换")
    parser.add_argument("--calib", type=Path, default=Path("configs/calibration_realman.yaml"), help="标定配置文件")
    parser.add_argument("--runtime-config", type=Path, default=Path("configs/egexo_runtime.yaml"), help="Ego-Exo 运行时配置")
    args = parser.parse_args()

    # 检查输出目录
    if args.output_dir.exists():
        print(f"输出目录已存在: {args.output_dir}")
        resp = input("是否删除并重新创建？(y/n): ")
        if resp.lower() == 'y':
            shutil.rmtree(args.output_dir)
        else:
            print("取消转换")
            return

    # 获取所有 HDF5 文件
    hdf5_files = sorted(args.input_dir.glob("*.hdf5"))
    print(f"找到 {len(hdf5_files)} 个 episodes")

    if len(hdf5_files) == 0:
        print("错误：未找到 HDF5 文件")
        return

    # 检查第一个文件获取维度信息
    sample_data = convert_hdf5_episode(hdf5_files[0])
    state_dim = sample_data['qpos'].shape[1]    # 6D: collect_data.py, 7D: collect_data_uarm.py
    action_dim = sample_data['action'].shape[1]
    state_names = make_joint_feature_names(state_dim)
    action_names = make_joint_feature_names(action_dim)
    img_h, img_w = sample_data['cam_high'].shape[1:3]

    print(f"状态维度: {state_dim}, 动作维度: {action_dim}")
    print(f"图像尺寸: {img_h}x{img_w}")
    print(f"目标帧率: {args.fps} Hz")
    print(f"任务描述: \"{args.task}\"")

    # 定义数据集特征
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": state_names,
        },
        "observation.images.cam_high": {
            "dtype": "video",
            "shape": (img_h, img_w, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.images.cam_wrist": {
            "dtype": "video",
            "shape": (img_h, img_w, 3),
            "names": ["height", "width", "channels"],
        },
        "action": {
            "dtype": "float32",
            "shape": (action_dim,),
            "names": action_names,
        },
    }

    calibration_cfg = None
    runtime_cfg = {}
    exo_intrinsics = {}
    ego_intrinsics = {}
    T_base_exo = np.eye(4, dtype=np.float32)
    T_ee_ego = np.eye(4, dtype=np.float32)
    if args.enable_egexo:
        calibration_cfg = load_calibration_config(args.calib)
        runtime_cfg = load_calibration_config(args.runtime_config)
        exo_intrinsics = intrinsics_from_config(calibration_cfg.get("cameras", {}).get("cam_high", {}))
        ego_intrinsics = intrinsics_from_config(calibration_cfg.get("cameras", {}).get("cam_wrist", {}))
        T_base_exo = transform_from_config(calibration_cfg.get("cameras", {}).get("cam_high", {}), "T_base_cam")
        T_ee_ego = transform_from_config(calibration_cfg.get("cameras", {}).get("cam_wrist", {}), "T_ee_cam")

        features.update(
            {
                "observation.ee_pose": {
                    "dtype": "float32",
                    "shape": (6,),
                    "names": ["x", "y", "z", "rx", "ry", "rz"],
                },
                "observation.grounding.ego_roi": {
                    "dtype": "float32",
                    "shape": (4,),
                    "names": ["x1", "y1", "x2", "y2"],
                },
                "observation.grounding.valid": {
                    "dtype": "float32",
                    "shape": (1,),
                    "names": ["valid"],
                },
                "observation.target_3d_base": {
                    "dtype": "float32",
                    "shape": (3,),
                    "names": ["x", "y", "z"],
                },
                "observation.phase": {
                    "dtype": "float32",
                    "shape": (1,),
                    "names": ["phase"],
                },
            }
        )

    # 创建 LeRobot 数据集
    print(f"\n创建数据集: {args.output_dir}")
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.fps,
        features=features,
        root=args.output_dir,
        robot_type="realman",
        use_videos=True,
    )

    total_frames = 0

    # 逐 episode 转换
    for ep_idx, hdf5_path in enumerate(hdf5_files):
        data = convert_hdf5_episode(hdf5_path)
        num_frames = len(data['qpos'])
        total_frames += num_frames

        for frame_idx in range(num_frames):
            frame_data = {
                "observation.state": torch.from_numpy(
                    data['qpos'][frame_idx].astype(np.float32)),
                "observation.images.cam_high": Image.fromarray(
                    data['cam_high'][frame_idx]),
                "observation.images.cam_wrist": Image.fromarray(
                    data['cam_wrist'][frame_idx]),
                "action": torch.from_numpy(
                    data['action'][frame_idx].astype(np.float32)),
                "task": args.task,
            }

            if args.enable_egexo:
                ee_pose = np.asarray(
                    data.get("ee_pose", np.zeros((num_frames, 6), dtype=np.float32))[frame_idx],
                    dtype=np.float32,
                ).reshape(-1)[:6]
                depth_high = np.asarray(
                    data.get("depth_high", np.zeros((num_frames, img_h, img_w), dtype=np.uint16))[frame_idx],
                    dtype=np.uint16,
                )
                stored_roi = np.asarray(
                    data.get("target_roi_exo", np.zeros((num_frames, 4), dtype=np.float32))[frame_idx],
                    dtype=np.float32,
                ).reshape(-1)[:4]
                stored_valid = float(
                    np.asarray(data.get("grounding_valid", np.zeros((num_frames, 1), dtype=np.float32))[frame_idx]).reshape(-1)[0]
                )
                if np.any(stored_roi > 0):
                    exo_roi = stored_roi
                    grounding_valid = stored_valid > 0.5
                else:
                    locator_cfg = runtime_cfg.get("target_locator", {})
                    locator = locate_target_roi(data["cam_high"][frame_idx], depth_high, locator_cfg)
                    exo_roi = np.asarray(locator.get("roi_xyxy", np.zeros(4, dtype=np.float32)), dtype=np.float32)
                    grounding_valid = bool(locator.get("valid", False))

                if grounding_valid:
                    grounding = project_exo_roi_to_ego(
                        exo_roi_xyxy=exo_roi,
                        exo_depth=depth_high,
                        exo_intrinsics=exo_intrinsics,
                        ego_intrinsics=ego_intrinsics,
                        T_base_exo=T_base_exo,
                        T_base_ee=ee_pose_to_T_base_ee(ee_pose),
                        T_ee_ego=T_ee_ego,
                        image_size=data["cam_wrist"][frame_idx].shape[:2],
                        cfg=calibration_cfg.get("geometry", {}),
                    )
                else:
                    grounding = {
                        "valid": False,
                        "ego_roi_xyxy": np.zeros(4, dtype=np.float32),
                        "target_3d_base": np.zeros(3, dtype=np.float32),
                    }

                target_3d_base = np.asarray(grounding["target_3d_base"], dtype=np.float32).reshape(3)
                valid_value = float(grounding["valid"])
                if valid_value > 0.5:
                    distance = np.linalg.norm(ee_pose[:3] - target_3d_base)
                    threshold = float(runtime_cfg.get("phase", {}).get("distance_threshold_m", 0.08))
                    phase = 0.0 if distance > threshold else 1.0
                else:
                    phase = 0.0

                frame_data["observation.ee_pose"] = torch.from_numpy(ee_pose.astype(np.float32))
                frame_data["observation.grounding.ego_roi"] = torch.from_numpy(
                    np.asarray(grounding["ego_roi_xyxy"], dtype=np.float32)
                )
                frame_data["observation.grounding.valid"] = torch.tensor([valid_value], dtype=torch.float32)
                frame_data["observation.target_3d_base"] = torch.from_numpy(target_3d_base)
                frame_data["observation.phase"] = torch.tensor([phase], dtype=torch.float32)

            dataset.add_frame(frame_data)

        dataset.save_episode()

        if (ep_idx + 1) % 10 == 0 or ep_idx == len(hdf5_files) - 1:
            print(f"  已完成 {ep_idx + 1}/{len(hdf5_files)} episodes ({total_frames} frames)")

    # 整合数据集（视频编码 + 统计量计算）
    print("\n正在整合数据集（编码视频）...")
    dataset.consolidate()

    print(f"\n{'=' * 50}")
    print(f"✓ 数据集转换完成!")
    print(f"{'=' * 50}")
    print(f"  路径: {args.output_dir}")
    print(f"  FPS: {args.fps}")
    print(f"  Episodes: {len(hdf5_files)}")
    print(f"  总帧数: {total_frames}")
    print(f"  Task: \"{args.task}\"")
    print(f"\n下一步: 训练与推理时都使用 --freq {args.fps}")


if __name__ == "__main__":
    main()
