#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import threading
import time

import numpy as np
from std_msgs.msg import Float64MultiArray


class UarmLeaderSubscriber:
    """Subscribe to U-arm leader topic and cache the latest 7D command."""

    def __init__(self, node, topic="/servo_angles", leader_dim=7, qos_depth=10):
        if node is None:
            raise ValueError("ROS2 node is required for UarmLeaderSubscriber")

        self._node = node
        self.topic = topic
        self.leader_dim = int(leader_dim)
        self._lock = threading.Lock()

        self._latest = np.zeros(self.leader_dim, dtype=np.float32)
        self._last_time = 0.0
        self._has_data = False

        self._subscription = self._node.create_subscription(
            Float64MultiArray,
            self.topic,
            self._callback,
            qos_depth,
        )

    def _callback(self, msg):
        arr = np.asarray(msg.data, dtype=np.float32).reshape(-1)
        if arr.size < self.leader_dim:
            return

        with self._lock:
            self._latest[:] = arr[: self.leader_dim]
            self._last_time = time.time()
            self._has_data = True

    def get(self):
        """Return (latest, last_time, has_data)."""
        with self._lock:
            return self._latest.copy(), self._last_time, self._has_data

    def close(self):
        if self._subscription is not None:
            try:
                self._node.destroy_subscription(self._subscription)
            except Exception:
                pass
            self._subscription = None
