"""Dock a tight-cropped head and chassis on the 1080x1920 stage."""
from __future__ import annotations

import cv2
import numpy as np


def _left_edge_flat_run(alpha: np.ndarray) -> tuple[int, int]:
    """Return (rows touching x=0, longest vertical run of that touch)."""
    touching = 0
    run = best = 0
    for row in alpha:
        if int(row[0]) > 0:
            touching += 1
            run += 1
            best = max(best, run)
        else:
            run = 0
    return touching, best


def diagnose_rotation_clip(raw_head: np.ndarray, label: str) -> None:
    """Log whether a flat left cut is already in the source or only after warp."""
    alpha = raw_head[:, :, 3]
    height = alpha.shape[0]
    touching, flat_run = _left_edge_flat_run(alpha)
    intact = flat_run < max(48, height // 10)
    cause = (
        "rounded contour intact; clip comes from warpAffine inside an unpadded box"
        if intact
        else "source already has a long vertical alpha cut at x=0"
    )
    print(
        f"rotation-clip diagnosis [{label}]: "
        f"nonzero_alpha_at_x0={touching}/{height} longest_flat_run={flat_run}px - {cause}"
    )


def dock_front_grounded(
    head: np.ndarray,
    body: np.ndarray,
    *,
    head_xy: tuple[int, int],
    stage_x: int = 540,
    collar_ratio: float = 0.50,
    chin_sink_px: int = 38,
    ground_y: int = 1920,
) -> tuple[np.ndarray, tuple[int, int]]:
    """Seat the collar under the chin and keep the base bleeding past y=1920."""
    body_x = stage_x - int(body.shape[1] * collar_ratio)
    collar_rim_y = int(body.shape[0] * 0.17)
    chin_y = int(head.shape[0] * 0.95)
    body_y = (int(head_xy[1]) + chin_y) - collar_rim_y - chin_sink_px
    bottom = body_y + int(body.shape[0])
    if bottom < ground_y:
        target_h = ground_y - body_y
        body = cv2.resize(body, (int(body.shape[1]), target_h), interpolation=cv2.INTER_LANCZOS4)
        bottom = body_y + target_h
    print(f"front-ground body_xy=({body_x},{body_y}) body_bottom_y={bottom}")
    return body, (int(body_x), int(body_y))


def mirror_approved_body(approved_body: np.ndarray) -> np.ndarray:
    """Flip the approved chassis on X. No generation and no rescale."""
    return cv2.flip(approved_body, 1)


def _read_unchanged(path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raw = np.fromfile(str(path), dtype=np.uint8)
        image = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"cannot read image: {path}")
    return image


def normalize_to_exact_ref(ref_head_path, raw_secondary_head: np.ndarray) -> tuple[np.ndarray, int]:
    """Scale a secondary head to the reference height. No boost and no multiplier."""
    ref_head = _read_unchanged(ref_head_path)
    exact_target_h = int(ref_head.shape[0])
    scale = exact_target_h / max(1, raw_secondary_head.shape[0])
    normalized_head = cv2.resize(
        raw_secondary_head,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_LANCZOS4,
    )
    print(
        f"exact-match ref_h={exact_target_h} source_h={raw_secondary_head.shape[0]} "
        f"scale={scale:.4f} out={normalized_head.shape[1]}x{normalized_head.shape[0]}"
    )
    return normalized_head, exact_target_h


def safe_rotate_head(head_img: np.ndarray, angle_deg: float, pivot_ratio: tuple[float, float] = (0.5, 0.95)) -> np.ndarray:
    """Rotate a head without clipping outer contours. Pad, warp, then tight-crop."""
    pad = 120
    padded = cv2.copyMakeBorder(
        head_img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0, 0)
    )
    h_orig, w_orig = head_img.shape[:2]
    pivot_x = int(w_orig * pivot_ratio[0]) + pad
    pivot_y = int(h_orig * pivot_ratio[1]) + pad
    matrix = cv2.getRotationMatrix2D((pivot_x, pivot_y), angle_deg, scale=1.0)
    rotated = cv2.warpAffine(
        padded,
        matrix,
        (padded.shape[1], padded.shape[0]),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )
    alpha = rotated[:, :, 3]
    pts = cv2.findNonZero(alpha)
    if pts is not None:
        x, y, w, h = cv2.boundingRect(pts)
        return rotated[y : y + h, x : x + w]
    return rotated


def crop_alpha(img: np.ndarray) -> np.ndarray:
    """Tight crop to pixels with any opacity."""
    ys, xs = np.nonzero(img[..., 3] > 0)
    if xs.size == 0:
        raise RuntimeError("crop_alpha found no opaque pixels")
    return img[int(ys.min()) : int(ys.max()) + 1, int(xs.min()) : int(xs.max()) + 1]


def safe_rotate(img: np.ndarray, angle: float) -> np.ndarray:
    """Rotate with a 120px pad so the contour is not clipped."""
    return safe_rotate_head(img, angle)


def scale_head(head_img: np.ndarray, target_h: int = 710) -> np.ndarray:
    """Uniform scale to the golden head envelope."""
    scale = int(target_h) / max(1, head_img.shape[0])
    return cv2.resize(head_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)


def scale_body(body_img: np.ndarray, head_w: int, ratio: float = 1.40) -> np.ndarray:
    """Uniform scale so the chassis width is ratio times the head width."""
    target_w = max(1, int(round(head_w * ratio)))
    scale = target_w / max(1, body_img.shape[1])
    return cv2.resize(body_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)


def dock_socket(
    head: np.ndarray,
    body: np.ndarray,
    view_name: str = "facing_left",
) -> tuple[np.ndarray, np.ndarray, tuple[int, int], tuple[int, int]]:
    """Place eye line, collar overlap, and a base that crosses y=1920."""
    stage_x = {"facing_left": 585, "facing_right": 495, "facing_front": 540}.get(view_name, 540)
    collar_ratio = {"facing_left": 0.46, "facing_right": 0.54}.get(view_name, 0.50)
    pivot_x = head.shape[1] // 2
    head_y = 600 - int(head.shape[0] * 0.42)
    head_x = stage_x - pivot_x
    collar_rim_y = int(body.shape[0] * 0.17)
    chin_y = int(head.shape[0] * 0.95)
    body_x = (head_x + pivot_x) - int(body.shape[1] * collar_ratio)
    body_y = (head_y + chin_y) - collar_rim_y - 25
    if body_y + body.shape[0] < 1920:
        body = cv2.resize(
            body,
            (int(body.shape[1]), 1920 - int(body_y)),
            interpolation=cv2.INTER_LANCZOS4,
        )
    return head, body, (int(head_x), int(head_y)), (int(body_x), int(body_y))


class ParametricPuppetContract:
    """Elastic studio framing. Broadcast lines stay fixed; the body follows the head."""

    CANVAS_W = 1080
    CANVAS_H = 1920
    CANONICAL_EYE_Y = 600
    STAGE_X_RIGHT = 585
    STAGE_X_LEFT = 495
    STAGE_X_CENTER = 540
    HEAD_HEIGHT_MIN_PCT = 0.34
    HEAD_HEIGHT_MAX_PCT = 0.46
    # Rough facing-left estimate from the approved Claude lock (about 42% of the canvas).
    HEAD_HEIGHT_DEFAULT_PCT = 0.423
    DEFAULT_BODY_TO_HEAD_RATIO = 1.44
    BODY_SCALE_Y_MULT = 1.15
    FACING_LEFT_JAW_OFFSET_X = -45
    CHIN_SINK_PX = 38
    APPROVED_UNIFORM_BOOST = 1.15
    APPROVED_BODY_SHIFT_X = 40
    APPROVED_BODY_SHIFT_Y = 50

    @classmethod
    def jaw_offset_x(cls, view_name: str) -> int:
        """Left-facing jaw sits 45px toward the neck column. Other views stay centered."""
        if view_name == "facing_left":
            return cls.FACING_LEFT_JAW_OFFSET_X
        return 0

    @classmethod
    def target_head_height(cls, pct: float | None = None) -> int:
        ratio = cls.HEAD_HEIGHT_DEFAULT_PCT if pct is None else pct
        height = int(cls.CANVAS_H * ratio)
        low = int(cls.CANVAS_H * cls.HEAD_HEIGHT_MIN_PCT)
        high = int(cls.CANVAS_H * cls.HEAD_HEIGHT_MAX_PCT)
        return max(low, min(high, height))

    @classmethod
    def fit_puppet(
        cls,
        head_img: np.ndarray,
        body_img: np.ndarray,
        view_name: str = "facing_left",
        body_bulk_ratio: float = 1.40,
        target_head_h: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, int, float]:
        """Scale one puppet inside the elastic envelope and choose its stage axis."""
        target_head_h = cls.target_head_height() if target_head_h is None else int(target_head_h)
        h_scale = target_head_h / max(1, head_img.shape[0])
        head = cv2.resize(head_img, None, fx=h_scale, fy=h_scale, interpolation=cv2.INTER_LANCZOS4)
        target_body_w = int(head.shape[1] * body_bulk_ratio)
        b_scale_x = target_body_w / max(1, body_img.shape[1])
        body = cv2.resize(
            body_img,
            None,
            fx=b_scale_x,
            fy=b_scale_x * cls.BODY_SCALE_Y_MULT,
            interpolation=cv2.INTER_LANCZOS4,
        )
        if view_name == "facing_left":
            stage_x, tilt = cls.STAGE_X_RIGHT, -3.5
        elif view_name == "facing_right":
            stage_x, tilt = cls.STAGE_X_LEFT, 3.5
        else:
            stage_x, tilt = cls.STAGE_X_CENTER, 0.0
        print(
            f"parametric-fit view={view_name} head={head.shape[1]}x{head.shape[0]} "
            f"body={body.shape[1]}x{body.shape[0]} stage_x={stage_x} tilt={tilt} "
            f"eye_y={cls.CANONICAL_EYE_Y} bulk={body_bulk_ratio}"
        )
        return head, body, stage_x, tilt


class CanonicalDebaterContract:
    """Permanent calibrated framing for 9:16 vertical debate avatars."""

    CANVAS_W = 1080
    CANVAS_H = 1920
    TARGET_HEAD_HEIGHT = 721
    REST_TILT_DEG = -3.5
    TARGET_EYE_Y = 600
    STAGE_CENTER_X = 585
    BODY_WIDTH = 876
    HEAD_SHIFT_X = -10
    HEAD_DROP_PX = 15
    BODY_SCALE_Y_MULT = 1.15
    BODY_DROP_PX = 18
    CHIN_SINK_OVERLAP = 75

    @classmethod
    def assemble(
        cls,
        raw_head_1024: np.ndarray,
        raw_body_rgba: np.ndarray,
        *,
        head_height: int | None = None,
        body_width: int | None = None,
        body_drop_px: int | None = None,
        head_shift_x: int | None = None,
        head_drop_px: int | None = None,
        body_shift_right: int = 0,
        head_rel_shift_left: int = 0,
        rotation_deg: float | None = None,
        stage_center_x: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, tuple[int, int], tuple[int, int], tuple[int, int]]:
        """Scale, tilt, and dock one puppet to the golden debate frame."""
        target_head = cls.TARGET_HEAD_HEIGHT if head_height is None else head_height
        target_body = cls.BODY_WIDTH if body_width is None else body_width
        drop_px = cls.BODY_DROP_PX if body_drop_px is None else body_drop_px
        shift_x = cls.HEAD_SHIFT_X if head_shift_x is None else head_shift_x
        head_drop = cls.HEAD_DROP_PX if head_drop_px is None else head_drop_px
        angle = abs(cls.REST_TILT_DEG) if rotation_deg is None else rotation_deg
        stage_x = cls.STAGE_CENTER_X if stage_center_x is None else stage_center_x
        h_scale = target_head / max(1, raw_head_1024.shape[0])
        head = cv2.resize(
            raw_head_1024, None, fx=h_scale, fy=h_scale, interpolation=cv2.INTER_LANCZOS4
        )
        diagnose_rotation_clip(raw_head_1024, "approved master before warp")
        head = safe_rotate_head(head, angle_deg=angle, pivot_ratio=(0.5, 0.95))
        touching, flat_run = _left_edge_flat_run(head[:, :, 3])
        print(
            f"rotation-padding: 120px pre-warp, post-crop left flat run={flat_run}px "
            f"({touching}/{head.shape[0]} rows at x=0)"
        )
        pivot_x = head.shape[1] // 2
        pivot_y = int(head.shape[0] * 0.95)
        b_scale_x = target_body / max(1, raw_body_rgba.shape[1])
        body = cv2.resize(
            raw_body_rgba,
            None,
            fx=b_scale_x,
            fy=b_scale_x * cls.BODY_SCALE_Y_MULT,
            interpolation=cv2.INTER_LANCZOS4,
        )
        eye_offset_y = int(head.shape[0] * 0.42)
        head_y = cls.TARGET_EYE_Y - eye_offset_y
        head_x = stage_x - pivot_x
        collar_center_x = int(body.shape[1] * 0.46)
        collar_rim_y = int(body.shape[0] * 0.17)
        chin_y = int(head.shape[0] * 0.95)
        body_x = (head_x + pivot_x) - collar_center_x + 20
        head_x += shift_x
        body_y = (head_y + chin_y) - collar_rim_y - cls.CHIN_SINK_OVERLAP + drop_px
        head_y += head_drop
        body_x += body_shift_right
        head_x += body_shift_right - head_rel_shift_left
        print(
            f"golden-contract head={head.shape[1]}x{head.shape[0]} "
            f"body={body.shape[1]}x{body.shape[0]} eye_y={cls.TARGET_EYE_Y} stage_x={stage_x} "
            f"head_xy=({head_x},{head_y}) body_xy=({body_x},{body_y}) "
            f"chin_sink={cls.CHIN_SINK_OVERLAP} body_drop={drop_px} "
            f"body_shift_right={body_shift_right} head_rel_left={head_rel_shift_left} "
            f"net_head_dx={body_shift_right - head_rel_shift_left}"
        )
        return head, body, (head_x, head_y), (body_x, body_y), (pivot_x, pivot_y)

    @classmethod
    def dock_screen_right(
        cls,
        raw_head: np.ndarray,
        mirrored_body: np.ndarray,
        *,
        head_height: int = 721,
        stage_center_x: int = 480,
        body_shift_left: int = 0,
        head_shift_right: int = 0,
        head_drop_px: int = 0,
    ) -> tuple[np.ndarray, np.ndarray, tuple[int, int], tuple[int, int], tuple[int, int]]:
        """Scale and clockwise-tilt only the head. The flipped body stays at its approved pixels."""
        h_scale = head_height / max(1, raw_head.shape[0])
        head = cv2.resize(raw_head, None, fx=h_scale, fy=h_scale, interpolation=cv2.INTER_LANCZOS4)
        diagnose_rotation_clip(raw_head, "screen-right head before warp")
        head = safe_rotate_head(head, angle_deg=-3.5, pivot_ratio=(0.5, 0.95))
        if head.shape[0] != head_height:
            fitted = head_height / max(1, head.shape[0])
            head = cv2.resize(head, None, fx=fitted, fy=fitted, interpolation=cv2.INTER_LANCZOS4)
        if mirrored_body.shape[1] == cls.BODY_WIDTH:
            body = mirrored_body
        else:
            b_scale_x = cls.BODY_WIDTH / max(1, mirrored_body.shape[1])
            body = cv2.resize(
                mirrored_body,
                None,
                fx=b_scale_x,
                fy=b_scale_x * cls.BODY_SCALE_Y_MULT,
                interpolation=cv2.INTER_LANCZOS4,
            )
        pivot_x = head.shape[1] // 2
        pivot_y = int(head.shape[0] * 0.95)
        eye_offset_y = int(head.shape[0] * 0.42)
        head_y = cls.TARGET_EYE_Y - eye_offset_y
        head_x = stage_center_x - pivot_x
        collar_center_x = int(body.shape[1] * 0.54)
        collar_rim_y = int(body.shape[0] * 0.17)
        chin_y = int(head.shape[0] * 0.95)
        body_x = (head_x + pivot_x) - collar_center_x + 20
        body_y = (head_y + chin_y) - collar_rim_y - cls.CHIN_SINK_OVERLAP + cls.BODY_DROP_PX
        body_x -= body_shift_left
        head_x += head_shift_right
        head_y += head_drop_px
        print(
            f"screen-right head={head.shape[1]}x{head.shape[0]} "
            f"body={body.shape[1]}x{body.shape[0]} eye_y={cls.TARGET_EYE_Y} "
            f"stage_x={stage_center_x} head_xy=({head_x},{head_y}) body_xy=({body_x},{body_y}) "
            f"body_bottom_y={body_y + body.shape[0]} "
            f"body_shift_left={body_shift_left} head_shift_right={head_shift_right} "
            f"head_drop_px={head_drop_px}"
        )
        return head, body, (head_x, head_y), (body_x, body_y), (pivot_x, pivot_y)


def dock_puppet_calibrated_final(
    head_img: np.ndarray,
    body_img: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int], tuple[int, int], tuple[int, int]]:
    """Lock the eyes at y=570 and sink the chin into the chest collar."""
    head_scale = 640.0 / max(1, head_img.shape[0])
    head = cv2.resize(head_img, None, fx=head_scale, fy=head_scale, interpolation=cv2.INTER_LANCZOS4)
    pivot_x = head.shape[1] // 2
    pivot_y = int(head.shape[0] * 0.95)
    matrix = cv2.getRotationMatrix2D((pivot_x, pivot_y), angle=3.5, scale=1.0)
    head = cv2.warpAffine(
        head,
        matrix,
        (head.shape[1], head.shape[0]),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )
    body_scale = 960.0 / max(1, body_img.shape[1])
    body = cv2.resize(
        body_img,
        None,
        fx=body_scale,
        fy=body_scale * 1.15,
        interpolation=cv2.INTER_LANCZOS4,
    )
    eye_offset_y = int(head.shape[0] * 0.42)
    head_y = 570 - eye_offset_y
    head_stage_x = 590
    head_x = head_stage_x - pivot_x
    collar_center_x = int(body.shape[1] * 0.46)
    collar_rim_y = int(body.shape[0] * 0.17)
    chin_y = int(head.shape[0] * 0.95)
    body_x = (head_x + pivot_x) - collar_center_x + 25
    body_y = (head_y + chin_y) - collar_rim_y - 70
    print(
        f"calibrated-final head={head.shape[1]}x{head.shape[0]} body={body.shape[1]}x{body.shape[0]} "
        f"eye_y=570 head_xy=({head_x},{head_y}) body_xy=({body_x},{body_y}) "
        f"body_bottom={body_y + body.shape[0]} chin_sink=70"
    )
    return head, body, (head_x, head_y), (body_x, body_y), (pivot_x, pivot_y)


def dock_puppet_calibrated_v2(
    head_img: np.ndarray,
    body_img: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int], tuple[int, int], tuple[int, int]]:
    """Widen the barrel, sink it past the floor, and hug the jaw 35px left."""
    canvas_h = 1920
    head_scale = 640.0 / max(1, head_img.shape[0])
    head = cv2.resize(head_img, None, fx=head_scale, fy=head_scale, interpolation=cv2.INTER_LANCZOS4)
    pivot_x = head.shape[1] // 2
    pivot_y = int(head.shape[0] * 0.95)
    matrix = cv2.getRotationMatrix2D((pivot_x, pivot_y), angle=3.5, scale=1.0)
    head = cv2.warpAffine(
        head,
        matrix,
        (head.shape[1], head.shape[0]),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )
    body_scale = 960.0 / max(1, body_img.shape[1])
    body = cv2.resize(
        body_img,
        None,
        fx=body_scale,
        fy=body_scale * 1.15,
        interpolation=cv2.INTER_LANCZOS4,
    )
    body_y = (canvas_h + 260) - body.shape[0]
    body_x = 180
    collar_center_x = int(body.shape[1] * 0.46)
    collar_rim_y = int(body.shape[0] * 0.17)
    chin_y = int(head.shape[0] * 0.95)
    head_x = (body_x + collar_center_x) - pivot_x - 35
    head_y = (body_y + collar_rim_y) - chin_y + 25
    print(
        f"calibrated-v2 head={head.shape[1]}x{head.shape[0]} body={body.shape[1]}x{body.shape[0]} "
        f"head_xy=({head_x},{head_y}) body_xy=({body_x},{body_y}) "
        f"body_bottom={body_y + body.shape[0]} jaw_shift=-35"
    )
    return head, body, (head_x, head_y), (body_x, body_y), (pivot_x, pivot_y)


def dock_puppet_calibrated(
    head_img: np.ndarray,
    body_img: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int], tuple[int, int], tuple[int, int]]:
    """Align the chin to the chest collar and sink the barrel past y=1920."""
    canvas_h = 1920
    head_scale = 640.0 / max(1, head_img.shape[0])
    head = cv2.resize(head_img, None, fx=head_scale, fy=head_scale, interpolation=cv2.INTER_LANCZOS4)
    pivot_x = head.shape[1] // 2
    pivot_y = int(head.shape[0] * 0.95)
    matrix = cv2.getRotationMatrix2D((pivot_x, pivot_y), angle=3.5, scale=1.0)
    head = cv2.warpAffine(
        head,
        matrix,
        (head.shape[1], head.shape[0]),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )
    body_scale = (1080 * 0.78) / max(1, body_img.shape[1])
    body = cv2.resize(
        body_img,
        None,
        fx=body_scale,
        fy=body_scale * 1.12,
        interpolation=cv2.INTER_LANCZOS4,
    )
    collar_center_x = int(body.shape[1] * 0.46)
    collar_rim_y = int(body.shape[0] * 0.17)
    body_x = 240
    body_y = canvas_h - body.shape[0] + 80
    chin_y = int(head.shape[0] * 0.95)
    head_x = (body_x + collar_center_x) - pivot_x
    head_y = (body_y + collar_rim_y) - chin_y + 25
    print(
        f"calibrated head={head.shape[1]}x{head.shape[0]} body={body.shape[1]}x{body.shape[0]} "
        f"collar_x={body_x + collar_center_x} head_xy=({head_x},{head_y}) "
        f"body_xy=({body_x},{body_y}) body_bottom={body_y + body.shape[0]}"
    )
    return head, body, (head_x, head_y), (body_x, body_y), (pivot_x, pivot_y)


def dock_puppet_procedural(
    head_img: np.ndarray,
    body_img: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int], tuple[int, int], tuple[int, int]]:
    """Seat the chin on the chest collar and lean the head 3.5 degrees."""
    stage_x = 690
    head_scale = 640.0 / max(1, head_img.shape[0])
    head = cv2.resize(head_img, None, fx=head_scale, fy=head_scale, interpolation=cv2.INTER_LANCZOS4)
    pivot_x = head.shape[1] // 2
    pivot_y = int(head.shape[0] * 0.95)
    matrix = cv2.getRotationMatrix2D((pivot_x, pivot_y), angle=3.5, scale=1.0)
    head = cv2.warpAffine(
        head,
        matrix,
        (head.shape[1], head.shape[0]),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )
    eye_offset_y = int(head.shape[0] * 0.42)
    head_y = 575 - eye_offset_y
    head_x = stage_x - (head.shape[1] // 2)
    body_scale_x = (head.shape[1] * 1.40) / max(1, body_img.shape[1])
    body = cv2.resize(
        body_img,
        None,
        fx=body_scale_x * 0.92,
        fy=body_scale_x * 1.05,
        interpolation=cv2.INTER_LANCZOS4,
    )
    chest_collar_y = int(body.shape[0] * 0.17)
    chin_y = int(head.shape[0] * 0.95)
    body_y = (head_y + chin_y) - chest_collar_y - 15
    body_x = stage_x - (body.shape[1] // 2)
    print(
        f"dock head={head.shape[1]}x{head.shape[0]} body={body.shape[1]}x{body.shape[0]} "
        f"stage_x={stage_x} head_xy=({head_x},{head_y}) body_xy=({body_x},{body_y}) "
        f"collar_y={chest_collar_y} tilt=3.5"
    )
    return head, body, (head_x, head_y), (body_x, body_y), (pivot_x, pivot_y)


def assemble_puppet(
    cropped_head: np.ndarray,
    cropped_body: np.ndarray,
    canvas_w: int = 1080,
    canvas_h: int = 1920,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int], tuple[int, int]]:
    """Scale the head to 36% of the canvas and dock its chin on the collar."""
    target_head_h = int(canvas_h * 0.36)
    head_scale = target_head_h / max(1, cropped_head.shape[0])
    head_scaled = cv2.resize(
        cropped_head, None, fx=head_scale, fy=head_scale, interpolation=cv2.INTER_LANCZOS4
    )
    target_body_w = int(canvas_w * 0.88)
    body_scale = target_body_w / max(1, cropped_body.shape[1])
    body_scaled = cv2.resize(
        cropped_body, None, fx=body_scale, fy=body_scale, interpolation=cv2.INTER_LANCZOS4
    )
    chin_y_offset = int(head_scaled.shape[0] * 0.92)
    collar_y_offset = int(body_scaled.shape[0] * 0.08)
    overlap_px = int(head_scaled.shape[0] * 0.02)
    body_y = canvas_h - body_scaled.shape[0]
    body_x = (canvas_w - body_scaled.shape[1]) // 2
    head_x = (canvas_w - head_scaled.shape[1]) // 2
    head_y = (body_y + collar_y_offset) - chin_y_offset + overlap_px
    print(
        f"decoupled head={head_scaled.shape[1]}x{head_scaled.shape[0]} "
        f"body={body_scaled.shape[1]}x{body_scaled.shape[0]} "
        f"head_y={head_y} body_y={body_y} overlap={overlap_px}"
    )
    return head_scaled, body_scaled, (head_x, head_y), (body_x, body_y)
