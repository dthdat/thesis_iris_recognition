from __future__ import annotations

from math import pi
from typing import Any

import cv2
import numpy as np


def find_pupil_circle(gray: np.ndarray) -> np.ndarray | None:
    """Detect the pupillary boundary using HoughCircles on a grayscale image."""
    blurred = cv2.medianBlur(gray, 7)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=100,
        param1=100,
        param2=18,
        minRadius=15,
        maxRadius=80,
    )
    if circles is None:
        return None
    return circles[0][0]


def _iris_arc_means(gray_f: np.ndarray, cx: float, cy: float, radii: np.ndarray, angles: np.ndarray) -> np.ndarray:
    radii_grid = radii[:, None]
    angles_grid = angles[None, :]
    map_x = (cx + radii_grid * np.cos(angles_grid)).astype(np.float32)
    map_y = (cy + radii_grid * np.sin(angles_grid)).astype(np.float32)
    samples = cv2.remap(gray_f, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return np.median(samples, axis=1)


def valid_iris_geometry(
    gray: np.ndarray,
    pupil_xyr: np.ndarray | list[float] | tuple[float, float, float] | None,
    iris_xyr: np.ndarray | list[float] | tuple[float, float, float] | None,
    min_ratio: float = 1.8,
    max_ratio: float = 5.0,
) -> bool:
    if pupil_xyr is None or iris_xyr is None:
        return False
    h, w = gray.shape
    px, py, pr = map(float, pupil_xyr)
    ix, iy, ir = map(float, iris_xyr)
    if pr <= 0 or ir <= 0:
        return False
    ratio = ir / pr
    if not (min_ratio <= ratio <= max_ratio):
        return False
    center_offset = np.hypot(px - ix, py - iy)
    if center_offset > 0.35 * ir:
        return False
    margin = 2
    if ix - ir < -margin or ix + ir >= w + margin or iy - ir < -margin or iy + ir >= h + margin:
        return False
    return True


def find_iris_boundary_ido(
    gray: np.ndarray,
    pupil_xyr: np.ndarray,
    r_lo: float = 1.3,
    r_hi: float = 5.0,
    center_search: int = 4,
    sigma: float = 4.0,
    arc_deg: float = 28,
    skip_inner: float = 0.12,
) -> list[float] | None:
    """Daugman-style integro-differential limbus search over left/right arcs."""
    px, py, pr = float(pupil_xyr[0]), float(pupil_xyr[1]), float(pupil_xyr[2])
    h, w = gray.shape
    gray_f = gray.astype(np.float32)
    angles = np.concatenate(
        [np.linspace(-arc_deg, arc_deg, 50), np.linspace(180 - arc_deg, 180 + arc_deg, 50)]
    ) * (pi / 180.0)
    r_min = max(int(pr * r_lo), int(pr) + 5)
    r_max = min(int(pr * r_hi), int(0.97 * min(px, w - px, py, h - py)))
    if r_max <= r_min + 6:
        return None
    radii = np.arange(r_min, r_max, 1.0)
    kernel_size = max(3, int(sigma * 3) | 1)
    gaussian = cv2.getGaussianKernel(kernel_size, sigma).ravel()
    n_skip = max(2, int(len(radii) * skip_inner))
    best_score = -1e9
    best_circle: list[float] | None = None

    for dy in range(-center_search, center_search + 1, 2):
        for dx in range(-center_search, center_search + 1, 2):
            cx, cy = px + dx, py + dy
            intensities = _iris_arc_means(gray_f, cx, cy, radii, angles)
            smoothed = np.convolve(intensities, gaussian, mode="same")
            derivative = np.gradient(smoothed)
            derivative[:n_skip] = 0.0
            derivative[-2:] = 0.0
            idx = int(np.argmax(derivative))
            candidate = [cx, cy, int(round(float(radii[idx])))]
            if derivative[idx] > best_score and valid_iris_geometry(gray, pupil_xyr, candidate):
                best_score = float(derivative[idx])
                best_circle = candidate
    return best_circle


def _find_iris_hough(gray: np.ndarray, pupil_xyr: np.ndarray) -> np.ndarray | None:
    px, py, pr = pupil_xyr
    blurred = cv2.medianBlur(gray, 7)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=200,
        param1=100,
        param2=30,
        minRadius=int(pr * 1.5),
        maxRadius=int(pr * 4.0),
    )
    if circles is None:
        return None
    valid = []
    for circle in circles[0]:
        if ((circle[0] - px) ** 2 + (circle[1] - py) ** 2) < pr**2 and circle[2] > pr * 1.3:
            if valid_iris_geometry(gray, pupil_xyr, circle):
                valid.append(circle)
    if not valid:
        return None
    return max(valid, key=lambda c: c[2])


def find_iris_circle(
    gray: np.ndarray,
    pupil_xyr: np.ndarray | None,
    center_search: int = 4,
    return_method: bool = False,
) -> Any:
    if pupil_xyr is None:
        return (None, "no_pupil") if return_method else None

    ido = find_iris_boundary_ido(gray, pupil_xyr, center_search=center_search)
    if valid_iris_geometry(gray, pupil_xyr, ido):
        return (ido, "ido") if return_method else ido

    hough = _find_iris_hough(gray, pupil_xyr)
    if valid_iris_geometry(gray, pupil_xyr, hough):
        return (hough, "hough") if return_method else hough

    return (None, "fail") if return_method else None


def daugman_rubber_sheet(
    gray: np.ndarray,
    pupil_xyr: np.ndarray | list[float],
    iris_xyr: np.ndarray | list[float],
    polar_h: int = 64,
    polar_w: int = 512,
    r0: float = 0.10,
    r1: float = 0.87,
) -> np.ndarray:
    px, py, pr = pupil_xyr
    ix, iy, ir = iris_xyr
    theta = np.linspace(0, 2 * np.pi, polar_w, endpoint=False)
    px_c = px + pr * np.cos(theta)
    py_c = py + pr * np.sin(theta)
    ix_c = ix + ir * np.cos(theta)
    iy_c = iy + ir * np.sin(theta)
    radius = np.linspace(r0, r1, polar_h)[:, None]
    map_x = ((1 - radius) * px_c + radius * ix_c).astype(np.float32)
    map_y = ((1 - radius) * py_c + radius * iy_c).astype(np.float32)
    return cv2.remap(gray, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def preprocess_iris_to_polar(
    image_path: str,
    polar_h: int = 64,
    polar_w: int = 512,
    radial_inner: float = 0.10,
    radial_outer: float = 0.87,
    ido_center_search: int = 4,
    return_meta: bool = False,
) -> Any:
    meta: dict[str, Any] = {"path": image_path, "ok": False, "reason": None, "method": None}
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        meta["reason"] = "read_fail"
        return (None, meta) if return_meta else None

    pupil = find_pupil_circle(img)
    if pupil is None:
        meta["reason"] = "pupil_fail"
        return (None, meta) if return_meta else None

    iris, method = find_iris_circle(img, pupil, center_search=ido_center_search, return_method=True)
    if iris is None:
        meta["reason"] = method or "iris_fail"
        return (None, meta) if return_meta else None

    if not valid_iris_geometry(img, pupil, iris):
        meta["reason"] = "invalid_geometry"
        return (None, meta) if return_meta else None

    polar = daugman_rubber_sheet(img, pupil, iris, polar_h, polar_w, radial_inner, radial_outer)
    if polar is None or polar.shape != (polar_h, polar_w) or not np.isfinite(polar).all():
        meta["reason"] = "unwrap_fail"
        return (None, meta) if return_meta else None

    meta.update(
        {
            "ok": True,
            "reason": "ok",
            "method": method,
            "pupil": tuple(map(float, pupil)),
            "iris": tuple(map(float, iris)),
            "iris_pupil_ratio": float(float(iris[2]) / float(pupil[2])),
        }
    )
    return (polar.astype(np.uint8), meta) if return_meta else polar.astype(np.uint8)
