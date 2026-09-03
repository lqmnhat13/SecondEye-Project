"""Class-agnostic obstacle geometry from aligned metric depth."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from secondeye.multimodal.depth import DepthFusionConfig, metric_depth_band


@dataclass(frozen=True, slots=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def from_horizontal_fov(
        cls, width: int, height: int, horizontal_fov_degrees: float = 60.0
    ) -> "CameraIntrinsics":
        if width <= 0 or height <= 0:
            raise ValueError("kích thước ảnh phải dương")
        if not 10.0 <= horizontal_fov_degrees <= 150.0:
            raise ValueError("horizontal_fov_degrees nằm ngoài miền hợp lệ")
        fx = width / (2.0 * math.tan(math.radians(horizontal_fov_degrees) / 2.0))
        return cls(fx=fx, fy=fx, cx=(width - 1) / 2.0, cy=(height - 1) / 2.0)

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "CameraIntrinsics":
        return cls(
            fx=float(value["fx"]),
            fy=float(value["fy"]),
            cx=float(value["cx"]),
            cy=float(value["cy"]),
        )


@dataclass(frozen=True, slots=True)
class GeometryObstacleConfig:
    horizontal_fov_degrees: float = 60.0
    min_depth_m: float = 0.15
    max_depth_m: float = 5.0
    floor_region_top_fraction: float = 0.52
    corridor_top_fraction: float = 0.35
    corridor_top_width_fraction: float = 0.24
    corridor_bottom_width_fraction: float = 0.78
    min_obstacle_height_m: float = 0.10
    max_obstacle_height_m: float = 2.50
    floor_ransac_threshold_m: float = 0.06
    floor_min_inlier_ratio: float = 0.30
    floor_min_points: int = 180
    floor_max_samples: int = 6000
    min_component_pixels: int = 80
    morphology_kernel: int = 3

    def __post_init__(self) -> None:
        if not 0.0 < self.min_depth_m < self.max_depth_m:
            raise ValueError("depth geometry min/max không hợp lệ")
        for name, value in (
            ("floor_region_top_fraction", self.floor_region_top_fraction),
            ("corridor_top_fraction", self.corridor_top_fraction),
            ("corridor_top_width_fraction", self.corridor_top_width_fraction),
            ("corridor_bottom_width_fraction", self.corridor_bottom_width_fraction),
            ("floor_min_inlier_ratio", self.floor_min_inlier_ratio),
        ):
            if not 0.0 < value < 1.0:
                raise ValueError(f"{name} phải nằm trong (0, 1)")
        if self.corridor_top_width_fraction >= self.corridor_bottom_width_fraction:
            raise ValueError("corridor top phải hẹp hơn corridor bottom")
        if not 0.0 < self.min_obstacle_height_m < self.max_obstacle_height_m:
            raise ValueError("ngưỡng chiều cao vật cản không hợp lệ")
        if self.floor_min_points < 3 or self.floor_max_samples < self.floor_min_points:
            raise ValueError("số điểm RANSAC không hợp lệ")
        if self.min_component_pixels <= 0 or self.morphology_kernel <= 0:
            raise ValueError("component/morphology config phải dương")


DEFAULT_GEOMETRY_CONFIG = GeometryObstacleConfig()


def _project_depth(metric_depth_m: Any, intrinsics: CameraIntrinsics) -> Any:
    import numpy as np

    height, width = metric_depth_m.shape
    yy, xx = np.indices((height, width), dtype=np.float32)
    z = metric_depth_m.astype(np.float32, copy=False)
    x = (xx - intrinsics.cx) * z / intrinsics.fx
    y = (yy - intrinsics.cy) * z / intrinsics.fy
    return np.stack((x, y, z), axis=-1)


def _fit_floor_plane(
    points: Any,
    *,
    threshold_m: float,
    min_inlier_ratio: float,
    min_points: int,
    max_samples: int,
) -> tuple[Any | None, float]:
    """Fit a floor plane with deterministic RANSAC, returning [nx,ny,nz,d]."""
    import numpy as np

    if points.shape[0] < min_points:
        return None, 0.0
    rng = np.random.default_rng(0)
    if points.shape[0] > max_samples:
        points = points[rng.choice(points.shape[0], max_samples, replace=False)]
    best_mask = None
    best_count = 0
    for _ in range(80):
        sample = points[rng.choice(points.shape[0], 3, replace=False)]
        normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
        norm = float(np.linalg.norm(normal))
        if norm < 1e-6:
            continue
        normal = normal / norm
        if abs(float(normal[1])) < 0.55:
            continue
        if normal[1] < 0.0:
            normal = -normal
        offset = -float(np.dot(normal, sample[0]))
        residual = np.abs(points @ normal + offset)
        mask = residual <= threshold_m
        count = int(mask.sum())
        if count > best_count:
            best_count = count
            best_mask = mask
    if best_mask is None or best_count / points.shape[0] < min_inlier_ratio:
        return None, best_count / points.shape[0]
    inliers = points[best_mask]
    centroid = inliers.mean(axis=0)
    _, _, vh = np.linalg.svd(inliers - centroid, full_matrices=False)
    normal = vh[-1]
    if normal[1] < 0.0:
        normal = -normal
    if abs(float(normal[1])) < 0.55:
        return None, best_count / points.shape[0]
    offset = -float(np.dot(normal, centroid))
    plane = np.asarray([*normal, offset], dtype=np.float32)
    residual = np.abs(points @ plane[:3] + plane[3])
    ratio = float((residual <= threshold_m).mean())
    return plane, ratio


def _corridor_mask(height: int, width: int, config: GeometryObstacleConfig) -> Any:
    import numpy as np

    yy, xx = np.indices((height, width), dtype=np.float32)
    top_y = config.corridor_top_fraction * height
    progress = np.clip((yy - top_y) / max(1.0, height - top_y), 0.0, 1.0)
    width_fraction = config.corridor_top_width_fraction + progress * (
        config.corridor_bottom_width_fraction - config.corridor_top_width_fraction
    )
    half_width = width_fraction * width / 2.0
    return (yy >= top_y) & (np.abs(xx - (width - 1) / 2.0) <= half_width)


def detect_geometry_obstacles(
    metric_depth_m: Any,
    *,
    intrinsics: CameraIntrinsics | None = None,
    config: GeometryObstacleConfig = DEFAULT_GEOMETRY_CONFIG,
    depth_config: DepthFusionConfig = DepthFusionConfig(),
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return class-agnostic obstacle regions plus geometry diagnostics."""
    import cv2
    import numpy as np

    if not isinstance(metric_depth_m, np.ndarray) or metric_depth_m.ndim != 2:
        raise ValueError("metric_depth_m phải là ma trận HxW")
    height, width = metric_depth_m.shape
    camera = intrinsics or CameraIntrinsics.from_horizontal_fov(
        width, height, config.horizontal_fov_degrees
    )
    valid = (
        np.isfinite(metric_depth_m)
        & (metric_depth_m >= config.min_depth_m)
        & (metric_depth_m <= config.max_depth_m)
    )
    points = _project_depth(metric_depth_m, camera)
    floor_region = np.indices((height, width))[0] >= int(
        round(height * config.floor_region_top_fraction)
    )
    floor_points = points[valid & floor_region]
    plane, inlier_ratio = _fit_floor_plane(
        floor_points,
        threshold_m=config.floor_ransac_threshold_m,
        min_inlier_ratio=config.floor_min_inlier_ratio,
        min_points=config.floor_min_points,
        max_samples=config.floor_max_samples,
    )
    diagnostics: dict[str, Any] = {
        "usable": plane is not None,
        "floor_inlier_ratio": round(inlier_ratio, 4),
        "valid_depth_fraction": round(float(valid.mean()), 4),
        "intrinsics": {
            "fx": round(camera.fx, 4),
            "fy": round(camera.fy, 4),
            "cx": round(camera.cx, 4),
            "cy": round(camera.cy, 4),
        },
        "reason": None if plane is not None else "floor_plane_unavailable",
    }
    if plane is None:
        return [], diagnostics

    signed = points @ plane[:3] + plane[3]
    height_above_floor = -signed
    mask = (
        valid
        & _corridor_mask(height, width, config)
        & (height_above_floor >= config.min_obstacle_height_m)
        & (height_above_floor <= config.max_obstacle_height_m)
        & (metric_depth_m <= depth_config.medium_distance_m)
    )
    kernel_size = config.morphology_kernel
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    clean = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel)
    count, _, stats, centroids = cv2.connectedComponentsWithStats(clean, 8)
    obstacles: list[dict[str, Any]] = []
    for component in range(1, count):
        x, y, box_width, box_height, area = (int(value) for value in stats[component])
        if area < config.min_component_pixels:
            continue
        component_depth = metric_depth_m[y : y + box_height, x : x + box_width]
        component_mask = clean[y : y + box_height, x : x + box_width] > 0
        values = component_depth[
            component_mask & np.isfinite(component_depth) & (component_depth > 0.0)
        ]
        if values.size == 0:
            continue
        distance = float(np.percentile(values, depth_config.metric_percentile))
        band = metric_depth_band(distance, depth_config)
        center_x = float(centroids[component][0])
        direction = (
            "left"
            if center_x < width * 0.30
            else "right" if center_x > width * 0.70 else "center"
        )
        obstacles.append(
            {
                "label": "unknown_obstacle",
                "confidence": round(min(1.0, inlier_ratio), 4),
                "confidence_kind": "floor_inlier_ratio_heuristic",
                "bbox_xyxy": [
                    float(x),
                    float(y),
                    float(x + box_width),
                    float(y + box_height),
                ],
                "direction": direction,
                "distance_m": round(distance, 3),
                "depth_zone": band,
                "proximity_zone": band,
                "obstacle_candidate": True,
                "geometry_confirmed": True,
                "safety_evaluable": True,
                "proximity_reason": "metric_floor_geometry",
                "geometry_area_pixels": area,
            }
        )
    diagnostics["obstacle_count"] = len(obstacles)
    diagnostics["floor_plane"] = [round(float(value), 6) for value in plane]
    return obstacles, diagnostics


def _overlap_score(first: list[float], second: list[float]) -> float:
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0.0, min(ay2, by2) - max(ay1, by1)
    )
    if intersection <= 0.0:
        return 0.0
    area_a = max(1.0, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1.0, (bx2 - bx1) * (by2 - by1))
    union = area_a + area_b - intersection
    return max(intersection / union, intersection / min(area_a, area_b))


def fuse_geometry_with_detections(
    detections: list[dict[str, Any]],
    geometry_obstacles: list[dict[str, Any]],
    *,
    min_overlap: float = 0.25,
) -> list[dict[str, Any]]:
    """Give geometry regions semantic labels when available, retaining unknowns."""
    fused = [dict(item) for item in detections]
    used_semantic: set[int] = set()
    for obstacle in geometry_obstacles:
        best_index = None
        best_score = 0.0
        for index, semantic in enumerate(fused):
            if index in used_semantic:
                continue
            score = _overlap_score(
                [float(value) for value in obstacle["bbox_xyxy"]],
                [float(value) for value in semantic["bbox_xyxy"]],
            )
            if score > best_score:
                best_score = score
                best_index = index
        if best_index is None or best_score < min_overlap:
            fused.append(dict(obstacle))
            continue
        used_semantic.add(best_index)
        semantic = fused[best_index]
        semantic.update(
            {
                "geometry_bbox_xyxy": list(obstacle["bbox_xyxy"]),
                "geometry_overlap": round(best_score, 4),
                "distance_m": obstacle["distance_m"],
                "depth_zone": obstacle["depth_zone"],
                "proximity_zone": obstacle["proximity_zone"],
                "direction": obstacle["direction"],
                "obstacle_candidate": True,
                "geometry_confirmed": True,
                "safety_evaluable": True,
                "proximity_reason": "metric_floor_geometry_with_semantics",
            }
        )
    return fused
