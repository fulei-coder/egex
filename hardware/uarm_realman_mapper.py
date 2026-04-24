#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


class UarmRealmanMapper:
    """Map U-arm leader offsets to RealMan joint-space targets.

    Runtime policy for the current stage:
    - Execute only the first 6 joints.
    - Always output a 7D action where index 6 is reserved for future gripper.
    """

    def __init__(self, config):
        cfg = config or {}

        self.active_dof = int(cfg.get("active_dof", 6))
        self.active_dof = max(1, min(6, self.active_dof))

        joint_map = cfg.get("joint_map", {})
        self.reorder = np.asarray(joint_map.get("reorder", [0, 1, 2, 3, 4, 5]), dtype=np.int32)
        self.sign = np.asarray(joint_map.get("sign", [1, 1, 1, 1, 1, 1]), dtype=np.float32)
        self.scale = np.asarray(joint_map.get("scale", [1, 1, 1, 1, 1, 1]), dtype=np.float32)
        self.offset_deg = np.asarray(joint_map.get("offset_deg", [0, 0, 0, 0, 0, 0]), dtype=np.float32)

        limits = cfg.get("joint_limits_deg", {})
        self.lower = np.asarray(limits.get("lower", [-170, -90, -170, -180, -120, -360]), dtype=np.float32)
        self.upper = np.asarray(limits.get("upper", [170, 90, 170, 180, 120, 360]), dtype=np.float32)

        control_cfg = cfg.get("control", {})
        self.use_ema = bool(control_cfg.get("use_ema", True))
        self.ema_alpha = float(control_cfg.get("ema_alpha", 0.3))
        self.max_step_deg = float(control_cfg.get("max_step_deg", 5.0))
        self.jump_reject_deg = float(control_cfg.get("jump_reject_deg", 20.0))

        gripper_cfg = cfg.get("gripper", {})
        self.gripper_enabled = bool(gripper_cfg.get("enabled", False))
        self.gripper_binary = bool(gripper_cfg.get("binary", True))
        self.gripper_placeholder = float(gripper_cfg.get("placeholder_value", 0.0))
        self.leader_gripper_index = int(gripper_cfg.get("leader_index", 6))
        self.leader_open_value = float(gripper_cfg.get("leader_open_value", 0.0))
        self.leader_close_value = float(gripper_cfg.get("leader_close_value", 0.48))
        self.robot_open_value = float(gripper_cfg.get("robot_open_value", 0.0))
        self.robot_close_value = float(gripper_cfg.get("robot_close_value", 1.0))
        self.gripper_threshold = float(gripper_cfg.get("threshold", 0.5))

        self._prev_joint_target = None

    def reset_state(self):
        self._prev_joint_target = None

    def get_gripper_placeholder(self):
        return float(self.gripper_placeholder)

    def _map_gripper(self, leader):
        if not self.gripper_enabled:
            return self.gripper_placeholder

        if leader.size <= self.leader_gripper_index:
            return self.gripper_placeholder

        leader_value = float(leader[self.leader_gripper_index])
        denom = self.leader_close_value - self.leader_open_value
        if abs(denom) < 1e-6:
            norm = 0.0
        else:
            norm = (leader_value - self.leader_open_value) / denom
        norm = float(np.clip(norm, 0.0, 1.0))

        if self.gripper_binary:
            return self.robot_close_value if norm >= self.gripper_threshold else self.robot_open_value

        return self.robot_open_value + norm * (self.robot_close_value - self.robot_open_value)

    def map(self, leader, zero_leader, robot_init_qpos):
        """Return 7D action where first 6 dims are joint targets in degrees."""
        leader = np.asarray(leader, dtype=np.float32).reshape(-1)
        zero_leader = np.asarray(zero_leader, dtype=np.float32).reshape(-1)
        robot_init_qpos = np.asarray(robot_init_qpos, dtype=np.float32).reshape(-1)

        if leader.size < 6 or zero_leader.size < 6 or robot_init_qpos.size < 6:
            raise ValueError("leader/zero_leader/robot_init_qpos must have at least 6 dims")

        leader_delta = leader[:6] - zero_leader[:6]
        leader_delta = leader_delta[self.reorder]
        raw_target = robot_init_qpos[:6] + leader_delta * self.sign * self.scale + self.offset_deg
        raw_target = np.clip(raw_target, self.lower, self.upper)

        target = raw_target.copy()
        if self._prev_joint_target is not None:
            delta = target - self._prev_joint_target

            if self.jump_reject_deg > 0:
                jump_mask = np.abs(delta) > self.jump_reject_deg
                target[jump_mask] = self._prev_joint_target[jump_mask]
                delta = target - self._prev_joint_target

            if self.max_step_deg > 0:
                delta = np.clip(delta, -self.max_step_deg, self.max_step_deg)
                target = self._prev_joint_target + delta

            if self.use_ema:
                target = self.ema_alpha * target + (1.0 - self.ema_alpha) * self._prev_joint_target

        target = np.clip(target, self.lower, self.upper)

        # Keep non-active joints fixed when active_dof < 6.
        if self.active_dof < 6:
            target[self.active_dof:] = robot_init_qpos[self.active_dof:6]

        self._prev_joint_target = target.copy()

        action = np.zeros(7, dtype=np.float32)
        action[:6] = target.astype(np.float32)
        action[6] = float(self._map_gripper(leader))
        return action
