#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import threading
import time

import numpy as np


class SharedCommandBuffer:
    """Thread-safe 7D action cache shared between control and recording threads."""

    def __init__(self, dim=7, default_action=None):
        self.dim = int(dim)
        self._lock = threading.Lock()

        if default_action is None:
            self._latest_action = np.zeros(self.dim, dtype=np.float32)
        else:
            arr = np.asarray(default_action, dtype=np.float32).reshape(-1)
            if arr.size < self.dim:
                arr = np.pad(arr, (0, self.dim - arr.size), mode="constant", constant_values=0.0)
            self._latest_action = arr[: self.dim].copy()

        self._last_update_time = 0.0

    def set(self, action):
        arr = np.asarray(action, dtype=np.float32).reshape(-1)
        if arr.size < self.dim:
            arr = np.pad(arr, (0, self.dim - arr.size), mode="constant", constant_values=0.0)

        with self._lock:
            self._latest_action = arr[: self.dim].copy()
            self._last_update_time = time.time()

    def get(self):
        with self._lock:
            return self._latest_action.copy(), self._last_update_time


class RealmanTeleopController:
    """Leader-follower controller for U-arm -> RealMan.

    Current debug mode:
    - execute only 6 joints via rm_movej
    - keep 7th dimension as reserved interface field
    """

    def __init__(
        self,
        arm,
        arm_lock,
        leader_subscriber,
        mapper,
        command_buffer,
        loop_hz=20,
        movej_speed=5,
        dry_run=False,
        execute_gripper=False,
        gripper_command_callback=None,
    ):
        self.arm = arm
        self.arm_lock = arm_lock
        self.leader_subscriber = leader_subscriber
        self.mapper = mapper
        self.command_buffer = command_buffer

        self.loop_hz = float(loop_hz)
        self.movej_speed = float(movej_speed)
        self.dry_run = bool(dry_run)
        self.execute_gripper = bool(execute_gripper)
        self.gripper_command_callback = gripper_command_callback

        self._running = True
        self._enabled = False
        self._calibrated = False
        self._last_gripper_binary = None

        self.zero_leader = np.zeros(7, dtype=np.float32)
        self.robot_init_qpos = np.zeros(7, dtype=np.float32)
        self.last_action = np.zeros(7, dtype=np.float32)
        self.last_error = ""

        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def calibrate(self, robot_init_qpos):
        leader, _, has_data = self.leader_subscriber.get()
        if not has_data:
            return False, "U-arm leader 当前无数据，无法校准"

        robot_init = np.asarray(robot_init_qpos, dtype=np.float32).reshape(-1)
        if robot_init.size < 7:
            robot_init = np.pad(robot_init, (0, 7 - robot_init.size), mode="constant", constant_values=0.0)

        self.zero_leader = leader.copy()
        self.robot_init_qpos = robot_init[:7].copy()
        self.mapper.reset_state()
        self.command_buffer.set(self.robot_init_qpos)
        self.last_action = self.robot_init_qpos.copy()
        self._calibrated = True
        return True, "U-arm / RealMan 零位标定完成"

    def enable(self):
        self._enabled = True

    def disable(self):
        self._enabled = False

    def status(self):
        return {
            "running": self._running,
            "enabled": self._enabled,
            "calibrated": self._calibrated,
            "dry_run": self.dry_run,
            "execute_gripper": self.execute_gripper,
            "last_error": self.last_error,
        }

    def _loop(self):
        interval = 1.0 / max(self.loop_hz, 1.0)

        while self._running:
            tick_start = time.time()

            try:
                if self._enabled and self._calibrated:
                    leader, _, has_data = self.leader_subscriber.get()
                    if has_data:
                        action = self.mapper.map(
                            leader=leader,
                            zero_leader=self.zero_leader,
                            robot_init_qpos=self.robot_init_qpos,
                        )

                        if not self.dry_run:
                            with self.arm_lock:
                                self.arm.rm_movej(action[:6].tolist(), self.movej_speed, 0, 0, 0)

                        if self.execute_gripper and self.gripper_command_callback is not None:
                            gripper_binary = 1 if float(action[6]) > 0.5 else 0
                            if gripper_binary != self._last_gripper_binary:
                                self._last_gripper_binary = gripper_binary
                                self.gripper_command_callback(gripper_binary)

                        self.last_action = action.copy()
                        self.command_buffer.set(action)
            except Exception as exc:
                self.last_error = str(exc)

            elapsed = time.time() - tick_start
            sleep_time = interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def shutdown(self):
        self._running = False
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)
