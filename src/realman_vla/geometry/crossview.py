from __future__ import annotations

import numpy as np


def depth_roi_to_3d(depth, roi_xyxy, intrinsics, min_depth_m, max_depth_m):
    depth = np.asarray(depth)
    roi = np.asarray(roi_xyxy, dtype=np.float32).reshape(4)
    h, w = depth.shape[:2]
    x1, y1, x2, y2 = roi.astype(int)
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(x1 + 1, min(w, x2))
    y2 = max(y1 + 1, min(h, y2))
    patch = depth[y1:y2, x1:x2]
    if patch.size == 0:
        return np.zeros(3, dtype=np.float32), False

    depth_scale = float(intrinsics.get("depth_scale", 0.001) or 0.001)
    patch_m = patch.astype(np.float32) * depth_scale
    valid = patch_m[(patch_m >= min_depth_m) & (patch_m <= max_depth_m)]
    if valid.size == 0:
        return np.zeros(3, dtype=np.float32), False

    z = float(np.median(valid))
    u = 0.5 * (x1 + x2)
    v = 0.5 * (y1 + y2)
    fx = float(intrinsics.get("fx", 0.0))
    fy = float(intrinsics.get("fy", 0.0))
    cx = float(intrinsics.get("cx", 0.0))
    cy = float(intrinsics.get("cy", 0.0))
    if abs(fx) < 1e-8 or abs(fy) < 1e-8:
        return np.zeros(3, dtype=np.float32), False

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return np.asarray([x, y, z], dtype=np.float32), True


def transform_point(T_dst_src, p_src):
    T = np.asarray(T_dst_src, dtype=np.float32).reshape(4, 4)
    p = np.asarray(p_src, dtype=np.float32).reshape(3)
    homo = np.ones(4, dtype=np.float32)
    homo[:3] = p
    out = T @ homo
    return out[:3].astype(np.float32)


def project_point_to_image(p_cam, intrinsics):
    p = np.asarray(p_cam, dtype=np.float32).reshape(3)
    if p[2] <= 1e-8:
        return np.zeros(2, dtype=np.float32), False
    fx = float(intrinsics.get("fx", 0.0))
    fy = float(intrinsics.get("fy", 0.0))
    cx = float(intrinsics.get("cx", 0.0))
    cy = float(intrinsics.get("cy", 0.0))
    if abs(fx) < 1e-8 or abs(fy) < 1e-8:
        return np.zeros(2, dtype=np.float32), False
    u = fx * p[0] / p[2] + cx
    v = fy * p[1] / p[2] + cy
    return np.asarray([u, v], dtype=np.float32), True


def project_exo_roi_to_ego(
    exo_roi_xyxy,
    exo_depth,
    exo_intrinsics,
    ego_intrinsics,
    T_base_exo,
    T_base_ee,
    T_ee_ego,
    image_size,
    cfg,
):
    geom_cfg = dict(cfg or {})
    min_depth_m = float(geom_cfg.get("min_depth_m", 0.15))
    max_depth_m = float(geom_cfg.get("max_depth_m", 2.0))
    roi_expand_ratio = float(geom_cfg.get("roi_expand_ratio", 1.8))
    default_result = {
        "valid": False,
        "reason": "unknown",
        "target_3d_exo": np.zeros(3, dtype=np.float32),
        "target_3d_base": np.zeros(3, dtype=np.float32),
        "target_3d_ego": np.zeros(3, dtype=np.float32),
        "ego_roi_xyxy": np.zeros(4, dtype=np.float32),
    }

    exo_roi_xyxy = np.asarray(exo_roi_xyxy, dtype=np.float32).reshape(4)
    if exo_roi_xyxy[2] <= exo_roi_xyxy[0] or exo_roi_xyxy[3] <= exo_roi_xyxy[1]:
        result = dict(default_result)
        result["reason"] = "invalid_exo_roi"
        return result

    if abs(float(exo_intrinsics.get("fx", 0.0))) < 1e-8 or abs(float(exo_intrinsics.get("fy", 0.0))) < 1e-8:
        result = dict(default_result)
        result["reason"] = "invalid_exo_intrinsics"
        return result

    if abs(float(ego_intrinsics.get("fx", 0.0))) < 1e-8 or abs(float(ego_intrinsics.get("fy", 0.0))) < 1e-8:
        result = dict(default_result)
        result["reason"] = "invalid_ego_intrinsics"
        return result

    p_exo, valid = depth_roi_to_3d(exo_depth, exo_roi_xyxy, exo_intrinsics, min_depth_m, max_depth_m)
    if not valid:
        result = dict(default_result)
        result["reason"] = "invalid_depth"
        return result

    # T_base_exo is expected to map points from exo camera frame -> robot base frame.
    # So lifting an exo-frame 3D point into base frame should use it directly.
    T_base_exo = np.asarray(T_base_exo, dtype=np.float32).reshape(4, 4)
    target_3d_base = transform_point(T_base_exo, p_exo)

    # Wrist camera pose in base frame:
    #   T_base_ego = T_base_ee @ T_ee_ego
    # To project a base-frame point into ego frame, invert that chain.
    T_base_ego = (
        np.asarray(T_base_ee, dtype=np.float32).reshape(4, 4)
        @ np.asarray(T_ee_ego, dtype=np.float32).reshape(4, 4)
    )
    T_ego_base = np.linalg.inv(T_base_ego)
    target_3d_ego = transform_point(T_ego_base, target_3d_base)
    uv, valid = project_point_to_image(target_3d_ego, ego_intrinsics)
    if not valid:
        result = dict(default_result)
        result["reason"] = "projection_behind_camera"
        result["target_3d_exo"] = p_exo.astype(np.float32)
        result["target_3d_base"] = target_3d_base.astype(np.float32)
        result["target_3d_ego"] = target_3d_ego.astype(np.float32)
        return result

    img_h, img_w = int(image_size[0]), int(image_size[1])
    width = max(8.0, (exo_roi_xyxy[2] - exo_roi_xyxy[0]) * roi_expand_ratio)
    height = max(8.0, (exo_roi_xyxy[3] - exo_roi_xyxy[1]) * roi_expand_ratio)
    unclipped_roi = np.asarray(
        [
            uv[0] - width / 2.0,
            uv[1] - height / 2.0,
            uv[0] + width / 2.0,
            uv[1] + height / 2.0,
        ],
        dtype=np.float32,
    )
    roi_out_of_view = bool(
        unclipped_roi[0] < 0
        or unclipped_roi[1] < 0
        or unclipped_roi[2] > img_w - 1
        or unclipped_roi[3] > img_h - 1
    )
    ego_roi = np.asarray(
        [
            uv[0] - width / 2.0,
            uv[1] - height / 2.0,
            uv[0] + width / 2.0,
            uv[1] + height / 2.0,
        ],
        dtype=np.float32,
    )
    ego_roi[0::2] = np.clip(ego_roi[0::2], 0, img_w - 1)
    ego_roi[1::2] = np.clip(ego_roi[1::2], 0, img_h - 1)
    roi_valid = bool(ego_roi[2] > ego_roi[0] and ego_roi[3] > ego_roi[1])
    return {
        "valid": roi_valid,
        "reason": "ok" if roi_valid and not roi_out_of_view else ("roi_out_of_view" if roi_valid else "empty_ego_roi"),
        "target_3d_exo": p_exo,
        "target_3d_base": target_3d_base.astype(np.float32),
        "target_3d_ego": target_3d_ego.astype(np.float32),
        "ego_roi_xyxy": ego_roi.astype(np.float32) if roi_valid else np.zeros(4, dtype=np.float32),
        "projected_uv": uv.astype(np.float32),
        "roi_out_of_view": roi_out_of_view,
    }
