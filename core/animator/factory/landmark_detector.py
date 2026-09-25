"""Face landmark detection for autonomous puppet ingestion.

MediaPipe Face Mesh is preferred for humanoid art.  Stylised robots often do
not register as faces, so a deterministic OpenCV optic/jaw detector provides
the production fallback.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib.util
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


@dataclass(frozen=True, slots=True)
class Point3D:
    """A point in UV coordinates relative to the detected head bounds."""

    x: float
    y: float
    z: float = 0.0


@dataclass(frozen=True, slots=True)
class EyeLandmark:
    center: Point3D
    iris_radius: float


@dataclass(frozen=True, slots=True)
class FacialLandmarks:
    """Normalized geometry plus the pixel bounds needed to restore anchors."""

    left_eye: EyeLandmark
    right_eye: EyeLandmark
    mouth_center: Point3D
    mouth_width: float
    jawline: tuple[Point3D, ...]
    chin_base: Point3D
    neck_pivot: Point3D
    head_bbox: tuple[int, int, int, int]
    confidence: float
    backend: str
    facial_plate_bbox: tuple[float, float, float, float] | None = None
    forehead_plate_bbox: tuple[float, float, float, float] | None = None

    def pixel_point(self, point: Point3D) -> tuple[int, int]:
        x0, y0, x1, y1 = self.head_bbox
        return (
            int(round(x0 + point.x * max(1, x1 - x0))),
            int(round(y0 + point.y * max(1, y1 - y0))),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


ImageInput = Image.Image | np.ndarray | str | Path


def _as_rgba(image: ImageInput) -> np.ndarray:
    if isinstance(image, (str, Path)):
        with Image.open(image) as source:
            return np.asarray(source.convert("RGBA"), dtype=np.uint8)
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("RGBA"), dtype=np.uint8)
    array = np.asarray(image, dtype=np.uint8)
    if array.ndim != 3 or array.shape[2] not in (3, 4):
        raise ValueError("character image must be RGB or RGBA")
    if array.shape[2] == 3:
        alpha = np.full((*array.shape[:2], 1), 255, dtype=np.uint8)
        array = np.concatenate((array, alpha), axis=2)
    return array


def _content_bbox(rgba: np.ndarray) -> tuple[int, int, int, int]:
    alpha = rgba[..., 3]
    ys, xs = np.nonzero(alpha > 8)
    height, width = alpha.shape
    if not xs.size:
        raise ValueError("character image has no visible pixels")
    return (
        int(xs.min()),
        int(ys.min()),
        min(width, int(xs.max()) + 1),
        min(height, int(ys.max()) + 1),
    )


def _uv(
    x: float,
    y: float,
    z: float,
    bbox: tuple[int, int, int, int],
) -> Point3D:
    x0, y0, x1, y1 = bbox
    width, height = max(1, x1 - x0), max(1, y1 - y0)
    return Point3D(
        float(np.clip((x - x0) / width, 0.0, 1.0)),
        float(np.clip((y - y0) / height, 0.0, 1.0)),
        float(z / width),
    )


def _mediapipe_face_mesh(
    rgba: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> FacialLandmarks | None:
    """Run the legacy ``solutions.face_mesh`` API when the wheel provides it."""
    spec = importlib.util.find_spec("mediapipe")
    if spec is None or not spec.submodule_search_locations:
        return None
    package_root = Path(next(iter(spec.submodule_search_locations)))
    if not any(
        candidate.is_file()
        for candidate in (
            package_root / "solutions" / "face_mesh.py",
            package_root / "python" / "solutions" / "face_mesh.py",
        )
    ):
        # Current tasks-only MediaPipe wheels omit solutions.face_mesh and
        # take several seconds to import; skip that dead path immediately.
        return None
    try:
        import mediapipe as mp  # noqa: PLC0415

        solutions = getattr(mp, "solutions", None)
        face_mesh_api = getattr(solutions, "face_mesh", None)
        if face_mesh_api is None:
            return None
    except (ImportError, AttributeError):
        return None

    x0, y0, x1, y1 = bbox
    crop = np.ascontiguousarray(rgba[y0:y1, x0:x1, :3])
    height, width = crop.shape[:2]
    try:
        with face_mesh_api.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        ) as mesh:
            result = mesh.process(crop)
    except (RuntimeError, ValueError):
        return None
    if not result.multi_face_landmarks:
        return None

    points = result.multi_face_landmarks[0].landmark

    def mean_point(indices: tuple[int, ...]) -> tuple[float, float, float]:
        selected = [points[index] for index in indices if index < len(points)]
        return (
            x0 + float(np.mean([point.x for point in selected])) * width,
            y0 + float(np.mean([point.y for point in selected])) * height,
            float(np.mean([point.z for point in selected])) * width,
        )

    # Iris landmarks exist only when refine_landmarks=True. Eye-ring indices
    # keep the path compatible with older Face Mesh model files.
    left_indices = (468, 469, 470, 471, 472) if len(points) > 477 else (33, 133, 159, 145)
    right_indices = (473, 474, 475, 476, 477) if len(points) > 477 else (362, 263, 386, 374)
    eye_pixels = [mean_point(left_indices), mean_point(right_indices)]
    eye_pixels.sort(key=lambda point: point[0])

    def iris_radius(indices: tuple[int, ...], center: tuple[float, float, float]) -> float:
        ring = [points[index] for index in indices[1:] if index < len(points)]
        if not ring:
            return 0.035
        distances = [
            np.hypot(
                (point.x * width + x0) - center[0],
                (point.y * height + y0) - center[1],
            )
            for point in ring
        ]
        return float(np.clip(np.mean(distances) / max(1, width), 0.01, 0.12))

    mouth_left = mean_point((61,))
    mouth_right = mean_point((291,))
    mouth_top = mean_point((13,))
    mouth_bottom = mean_point((14,))
    mouth_center_px = (
        (mouth_top[0] + mouth_bottom[0]) / 2.0,
        (mouth_top[1] + mouth_bottom[1]) / 2.0,
        (mouth_top[2] + mouth_bottom[2]) / 2.0,
    )
    jaw_indices = (234, 93, 132, 58, 172, 136, 150, 176, 152, 400, 379, 365, 397, 288, 361, 323, 454)
    jawline = tuple(
        _uv(
            x0 + points[index].x * width,
            y0 + points[index].y * height,
            points[index].z * width,
            bbox,
        )
        for index in jaw_indices
    )
    chin_px = mean_point((152,))
    neck_px = (
        chin_px[0],
        min(float(y1), chin_px[1] + height * 0.075),
        chin_px[2],
    )
    return FacialLandmarks(
        left_eye=EyeLandmark(
            _uv(*eye_pixels[0], bbox),
            iris_radius(left_indices, eye_pixels[0]),
        ),
        right_eye=EyeLandmark(
            _uv(*eye_pixels[1], bbox),
            iris_radius(right_indices, eye_pixels[1]),
        ),
        mouth_center=_uv(*mouth_center_px, bbox),
        mouth_width=float(
            np.clip(abs(mouth_right[0] - mouth_left[0]) / max(1, width), 0.08, 0.7)
        ),
        jawline=jawline,
        chin_base=_uv(*chin_px, bbox),
        neck_pivot=_uv(*neck_px, bbox),
        head_bbox=bbox,
        confidence=0.95,
        backend="mediapipe_face_mesh",
    )


def _opencv_robot_fallback(
    rgba: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> FacialLandmarks:
    x0, y0, x1, y1 = bbox
    crop = rgba[y0:y1, x0:x1]
    rgb = crop[..., :3]
    alpha = crop[..., 3]
    height, width = alpha.shape
    luma = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    # Bright compact contours in the upper face are strong robotic-optic
    # candidates (cyan lenses, white irises, LED eyes).
    visible_luma = luma[alpha > 8]
    bright_threshold = max(150, int(np.percentile(visible_luma, 82)))
    bright = np.where((luma >= bright_threshold) & (alpha > 8), 255, 0).astype(np.uint8)
    bright[int(height * 0.68) :, :] = 0
    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    candidates: list[tuple[float, float, float, float]] = []
    for contour in cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        area = float(cv2.contourArea(contour))
        if area < max(5.0, width * height * 0.00008) or area > width * height * 0.06:
            continue
        perimeter = float(cv2.arcLength(contour, True))
        circularity = 4.0 * np.pi * area / max(1.0, perimeter * perimeter)
        (cx, cy), radius = cv2.minEnclosingCircle(contour)
        if circularity >= 0.35 and radius >= max(2.0, width * 0.008):
            candidates.append((cx, cy, radius, circularity))

    pair: tuple[tuple[float, float, float, float], tuple[float, float, float, float]] | None = None
    pair_score = float("-inf")
    for index, first in enumerate(candidates):
        for second in candidates[index + 1 :]:
            left, right = sorted((first, second), key=lambda item: item[0])
            separation = (right[0] - left[0]) / max(1, width)
            if not 0.12 <= separation <= 0.72:
                continue
            y_error = abs(left[1] - right[1]) / max(1, height)
            radius_error = abs(left[2] - right[2]) / max(1.0, max(left[2], right[2]))
            score = (
                left[3]
                + right[3]
                - 2.0 * y_error
                - 0.8 * radius_error
                - 1.2 * abs(separation - 0.32)
            )
            if score > pair_score:
                pair, pair_score = (left, right), score

    optics_detected = pair is not None
    if pair is None:
        eye_y = height * 0.34
        radius = max(3.0, width * 0.055)
        pair = (
            (width * 0.34, eye_y, radius, 0.0),
            (width * 0.66, eye_y, radius, 0.0),
        )

    # Search below the detected optics, but stop before chest plating. Using
    # fixed whole-bust percentages can mistake a dark torso seam for a mouth.
    eye_mid_x = (pair[0][0] + pair[1][0]) / 2.0
    eye_mid_y = (pair[0][1] + pair[1][1]) / 2.0
    eye_span = abs(pair[1][0] - pair[0][0])

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    green_pair_detected = False
    green_mask = np.where(
        (hsv[..., 0] >= 35)
        & (hsv[..., 0] <= 95)
        & (hsv[..., 1] >= 55)
        & (hsv[..., 2] >= 90)
        & (alpha > 8),
        255,
        0,
    ).astype(np.uint8)
    green_optics = sorted(
        (
            (float(cv2.contourArea(contour)), cv2.boundingRect(contour))
            for contour in cv2.findContours(
                green_mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )[0]
            if cv2.contourArea(contour) >= width * height * 0.0004
        ),
        reverse=True,
    )[:2]
    if len(green_optics) == 2:
        optic_pair = []
        for _area, (gx, gy, gw, gh) in green_optics:
            optic_pair.append(
                (
                    gx + gw * 0.5,
                    gy + gh * 0.5,
                    max(gw, gh) * 0.5,
                    1.0,
                )
            )
        pair = tuple(sorted(optic_pair, key=lambda item: item[0]))
        optics_detected = True
        green_pair_detected = True
        eye_mid_x = (pair[0][0] + pair[1][0]) / 2.0
        eye_mid_y = (pair[0][1] + pair[1][1]) / 2.0
        eye_span = abs(pair[1][0] - pair[0][0])
    plate_mask = np.where(
        (luma >= max(95, int(np.percentile(visible_luma, 55))))
        & (hsv[..., 1] <= 115)
        & (alpha > 8),
        255,
        0,
    ).astype(np.uint8)
    kernel_size = max(5, int(round(min(width, height) * 0.008)) | 1)
    plate_mask = cv2.morphologyEx(
        plate_mask,
        cv2.MORPH_CLOSE,
        np.ones((kernel_size, kernel_size), np.uint8),
    )
    facial_plate: tuple[int, int, int, int] | None = None
    facial_plate_score = 0.0
    for contour in cv2.findContours(
        plate_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )[0]:
        bx, by, bw, bh = cv2.boundingRect(contour)
        area = float(cv2.contourArea(contour))
        center_y = by + bh / 2.0
        if (
            area >= width * height * 0.012
            and width * 0.18 <= bw <= width * 0.65
            and height * 0.10 <= bh <= height * 0.38
            and eye_mid_y <= center_y <= eye_mid_y + height * 0.30
        ):
            score = area - abs((bx + bw / 2.0) - eye_mid_x) * 20.0
            if score > facial_plate_score:
                facial_plate = (bx, by, bw, bh)
                facial_plate_score = score

    if facial_plate is not None and not green_pair_detected:
        plate_x, plate_y, plate_w, plate_h = facial_plate
        plate_optics = [
            candidate
            for candidate in candidates
            if plate_x - width * 0.10
            <= candidate[0]
            <= plate_x + plate_w + width * 0.10
            and plate_y - height * 0.18
            <= candidate[1]
            <= plate_y + plate_h * 0.20
            and candidate[2] >= plate_w * 0.03
        ]
        refined_pair = None
        refined_score = float("-inf")
        for index, first in enumerate(plate_optics):
            for second in plate_optics[index + 1 :]:
                left, right = sorted((first, second), key=lambda item: item[0])
                separation = (right[0] - left[0]) / max(1, width)
                if not 0.12 <= separation <= 0.60:
                    continue
                y_error = abs(left[1] - right[1]) / max(1, height)
                score = (
                    left[3]
                    + right[3]
                    - 2.5 * y_error
                    - abs(separation - 0.27)
                )
                if score > refined_score:
                    refined_pair, refined_score = (left, right), score
        if refined_pair is not None:
            pair = refined_pair
            eye_mid_x = (pair[0][0] + pair[1][0]) / 2.0
            eye_mid_y = (pair[0][1] + pair[1][1]) / 2.0
            eye_span = abs(pair[1][0] - pair[0][0])

    # A compact quadrilateral above the optics is typically a robot's label
    # or forehead badge. It is calibration metadata rather than an animated
    # layer, but recording it lets QA prove that ingestion preserved it.
    edges = cv2.Canny(luma, 60, 160)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8))
    forehead_plate: tuple[int, int, int, int] | None = None
    forehead_score = float("-inf")
    for contour in cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]:
        bx, by, bw, bh = cv2.boundingRect(contour)
        area = float(cv2.contourArea(contour))
        aspect = bw / max(1.0, bh)
        center_x = bx + bw / 2.0
        if (
            area >= width * height * 0.0007
            and width * 0.18 <= bw <= width * 0.52
            and height * 0.04 <= bh <= height * 0.18
            and 1.4 <= aspect <= 4.5
            and by + bh < eye_mid_y
            and pair[0][0] - eye_span * 0.45 <= center_x <= pair[1][0] + eye_span * 0.30
        ):
            score = by + bh - abs(center_x - eye_mid_x) * 0.15
            if score > forehead_score:
                forehead_plate = (bx, by, bw, bh)
                forehead_score = score

    roi_x0 = max(0, int(eye_mid_x - width * 0.34))
    roi_x1 = min(width, int(eye_mid_x + width * 0.34))
    roi_y0 = max(0, int(eye_mid_y + height * 0.06))
    roi_y1 = min(height, int(eye_mid_y + height * 0.30))
    roi = luma[roi_y0:roi_y1, roi_x0:roi_x1]
    roi_alpha = alpha[roi_y0:roi_y1, roi_x0:roi_x1]
    dark_threshold = min(95, int(np.percentile(visible_luma, 24)))
    dark = np.where((roi <= dark_threshold) & (roi_alpha > 8), 255, 0).astype(np.uint8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((3, 7), np.uint8))
    mouth_box: tuple[int, int, int, int] | None = None
    mouth_score = 0.0
    for contour in cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        bx, by, bw, bh = cv2.boundingRect(contour)
        aspect = bw / max(1.0, bh)
        score = float(cv2.contourArea(contour)) * min(aspect, 8.0)
        center_x = bx + roi_x0 + bw / 2.0
        if (
            aspect >= 1.15
            and bw >= width * 0.08
            and abs(center_x - eye_mid_x) <= width * 0.20
            and score > mouth_score
        ):
            mouth_box, mouth_score = (bx + roi_x0, by + roi_y0, bw, bh), score

    mouth_detected = mouth_box is not None or facial_plate is not None
    if facial_plate is not None:
        bx, by, bw, bh = facial_plate
        mouth_x = bx + bw * 0.50
        mouth_y = by + bh * 0.82
        mouth_width = bw * 0.48
    elif mouth_box is None:
        mouth_x = eye_mid_x
        mouth_y = min(height * 0.78, eye_mid_y + height * 0.19)
        mouth_width = float(np.clip(eye_span * 0.54, width * 0.18, width * 0.34))
    else:
        bx, by, bw, bh = mouth_box
        mouth_x = bx + bw / 2.0
        mouth_y = by + bh * 0.82
        mouth_width = float(bw)

    # The lower silhouette is stable on both organic busts and mechanical
    # jaw plates, unlike a face classifier trained only on photographs.
    jaw_uv = (
        (0.12, 0.56),
        (0.18, 0.70),
        (0.30, 0.84),
        (0.50, 0.92),
        (0.70, 0.84),
        (0.82, 0.70),
        (0.88, 0.56),
    )
    jawline = tuple(Point3D(x, y) for x, y in jaw_uv)
    # Dock the viseme on the lower chin guard, below the optic line.
    mouth_x = eye_mid_x
    mouth_y = float(np.clip(eye_mid_y + height * 0.17, eye_mid_y + 64, height * 0.90))
    chin_y = min(0.94, mouth_y / max(1, height) + 0.035)
    neck_y = min(0.995, chin_y + 0.045)

    def normalized_box(
        box: tuple[int, int, int, int] | None,
    ) -> tuple[float, float, float, float] | None:
        if box is None:
            return None
        bx, by, bw, bh = box
        return (
            bx / max(1, width),
            by / max(1, height),
            (bx + bw) / max(1, width),
            (by + bh) / max(1, height),
        )

    left, right = pair
    return FacialLandmarks(
        left_eye=EyeLandmark(
            _uv(x0 + left[0], y0 + left[1], 0.0, bbox),
            float(np.clip(left[2] / max(1, width), 0.01, 0.15)),
        ),
        right_eye=EyeLandmark(
            _uv(x0 + right[0], y0 + right[1], 0.0, bbox),
            float(np.clip(right[2] / max(1, width), 0.01, 0.15)),
        ),
        mouth_center=_uv(x0 + mouth_x, y0 + mouth_y, 0.0, bbox),
        mouth_width=float(np.clip(mouth_width / max(1, width), 0.08, 0.36)),
        jawline=jawline,
        chin_base=Point3D(float(np.clip(mouth_x / max(1, width), 0.0, 1.0)), chin_y),
        neck_pivot=Point3D(float(np.clip(mouth_x / max(1, width), 0.0, 1.0)), neck_y),
        head_bbox=bbox,
        confidence=0.76 if optics_detected and mouth_detected else 0.52,
        backend="opencv_robotic_fallback",
        facial_plate_bbox=normalized_box(facial_plate),
        forehead_plate_bbox=normalized_box(forehead_plate),
    )


def detect_landmarks(image: ImageInput) -> FacialLandmarks:
    """Detect face geometry and return coordinates normalized to head bounds."""
    rgba = _as_rgba(image)
    bbox = _content_bbox(rgba)
    detected = _mediapipe_face_mesh(rgba, bbox)
    if detected is not None and detected.confidence >= 0.5:
        return detected
    return _opencv_robot_fallback(rgba, bbox)


__all__ = [
    "EyeLandmark",
    "FacialLandmarks",
    "Point3D",
    "detect_landmarks",
]
