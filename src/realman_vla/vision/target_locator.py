from __future__ import annotations

import cv2
import numpy as np


def _clip_roi(roi_xyxy: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
    h, w = image_shape[:2]
    roi = np.asarray(roi_xyxy, dtype=np.float32).reshape(4)
    roi[0::2] = np.clip(roi[0::2], 0, max(0, w - 1))
    roi[1::2] = np.clip(roi[1::2], 0, max(0, h - 1))
    return roi


def _roi_is_valid(roi_xyxy: np.ndarray) -> bool:
    roi = np.asarray(roi_xyxy, dtype=np.float32).reshape(4)
    return bool(roi[2] > roi[0] and roi[3] > roi[1])


def locate_by_color(image_bgr, target_color, min_area=64):
    image = np.asarray(image_bgr, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        return {"roi_xyxy": np.zeros(4, dtype=np.float32), "valid": False, "reason": "invalid_image"}

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    color_cfg = target_color or {}
    lower = np.asarray(color_cfg.get("lower_hsv", [0, 80, 60]), dtype=np.uint8)
    upper = np.asarray(color_cfg.get("upper_hsv", [20, 255, 255]), dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    kernel = np.ones((5, 5), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {"roi_xyxy": np.zeros(4, dtype=np.float32), "valid": False, "reason": "no_color_contour"}

    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    if area < float(min_area):
        return {"roi_xyxy": np.zeros(4, dtype=np.float32), "valid": False, "reason": "color_area_too_small"}

    x, y, w, h = cv2.boundingRect(contour)
    roi = np.asarray([x, y, x + w, y + h], dtype=np.float32)
    return {"roi_xyxy": roi, "valid": True, "reason": "color_heuristic"}


class TargetLocator:
    def __init__(self, cfg):
        cfg = dict(cfg or {})
        self.cfg = cfg
        self.mode = str(cfg.get("mode", "manual_roi"))
        self.manual_roi_xyxy = np.asarray(cfg.get("manual_roi_xyxy", [0, 0, 0, 0]), dtype=np.float32)
        self.target_color = cfg.get("target_color")
        self.min_valid_depth_ratio = float(cfg.get("min_valid_depth_ratio", 0.02))
        self.color_min_area = float(cfg.get("color_min_area", 64))

    def locate(self, image_bgr, depth=None):
        image = np.asarray(image_bgr)
        h, w = image.shape[:2]

        if self.mode == "manual_roi":
            default_roi = self.manual_roi_xyxy.copy()
            if not _roi_is_valid(default_roi):
                default_roi = np.asarray([w * 0.3, h * 0.3, w * 0.7, h * 0.7], dtype=np.float32)
            roi = _clip_roi(default_roi, image.shape[:2])
            return {"roi_xyxy": roi, "valid": _roi_is_valid(roi), "reason": "manual_roi"}

        if self.mode == "color_heuristic":
            result = locate_by_color(image, self.target_color, min_area=self.color_min_area)
            if result["valid"]:
                result["roi_xyxy"] = _clip_roi(result["roi_xyxy"], image.shape[:2])
                result["valid"] = _roi_is_valid(result["roi_xyxy"])
            return result

        if self.mode == "external_detector":
            raise NotImplementedError("external_detector is reserved for a future detector integration.")

        if depth is None:
            return {"roi_xyxy": np.zeros(4, dtype=np.float32), "valid": False, "reason": "missing_depth"}

        depth = np.asarray(depth, dtype=np.float32)
        valid_mask = depth > 0
        valid_ratio = float(np.mean(valid_mask)) if valid_mask.size > 0 else 0.0
        if valid_ratio < self.min_valid_depth_ratio:
            return {"roi_xyxy": np.zeros(4, dtype=np.float32), "valid": False, "reason": "depth_valid_ratio_too_low"}

        ys, xs = np.where(valid_mask)
        if xs.size == 0 or ys.size == 0:
            return {"roi_xyxy": np.zeros(4, dtype=np.float32), "valid": False, "reason": "no_valid_depth_pixels"}
        x1, x2 = np.percentile(xs, [30, 70])
        y1, y2 = np.percentile(ys, [30, 70])
        roi = _clip_roi(np.asarray([x1, y1, x2, y2], dtype=np.float32), image.shape[:2])
        return {"roi_xyxy": roi, "valid": _roi_is_valid(roi), "reason": "depth_percentile"}


def locate_target_roi(image_bgr, depth=None, cfg=None):
    return TargetLocator(cfg).locate(image_bgr, depth=depth)
