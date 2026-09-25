"""Canonical geometry shared by autonomous puppet rigs.

The matrix describes final 1080x1920 broadcast pixels. Every autonomous
puppet uses one harmonic scale for head and body; anatomy is never widened or
enlarged independently to chase a shoulder-width target.
"""
from __future__ import annotations

from dataclasses import dataclass


PUPPET_MATRIX = {
    "canvas_size": (1080, 1920),
    "target_eye_y": 770,
    "target_head_height": 550,
    "target_head_width": 480,
    "neck_pivot_y": 820,
    "shoulder_min_width": 850,
    "body_anchor_y": 1920,
}


@dataclass(frozen=True, slots=True)
class PuppetMatrixTransform:
    """One uniform transform shared by articulated head and static body."""

    head_scale: float
    body_scale: float
    head_offset_x: int
    head_offset_y: int
    body_offset_x: int
    body_offset_y: int


def solve_puppet_matrix(
    *,
    head_height: int,
    eye_center: tuple[float, float],
    neck_pivot: tuple[float, float],
    body_bbox: tuple[int, int, int, int],
    canvas_size: tuple[int, int],
) -> PuppetMatrixTransform:
    """Solve square-pixel transforms without stretching either anatomy plane."""
    source_w, _source_h = canvas_size
    body_x0, _body_y0, body_x1, body_y1 = body_bbox
    visible_body_width = max(1, body_x1 - body_x0)
    visible_body_bottom = max(1, body_y1)

    harmonic_scale = max(
        PUPPET_MATRIX["target_head_height"] / float(max(1, head_height)),
        PUPPET_MATRIX["shoulder_min_width"] / float(visible_body_width),
    )
    head_scale = harmonic_scale
    body_scale = harmonic_scale
    assembly_offset_x = int(
        round(
            (PUPPET_MATRIX["canvas_size"][0] - source_w * head_scale)
            * 0.5
        )
    )
    body_offset_y = int(
        round(PUPPET_MATRIX["body_anchor_y"] - visible_body_bottom * body_scale)
    )
    head_offset_y = int(
        round(PUPPET_MATRIX["target_eye_y"] - eye_center[1] * head_scale)
    )

    # A source canvas narrower than its visible alpha cannot occur, but this
    # assertion catches malformed bbox calibration before a render is started.
    if body_x0 < 0 or body_x1 > source_w:
        raise ValueError("body_bbox lies outside the puppet canvas")

    return PuppetMatrixTransform(
        head_scale=head_scale,
        body_scale=body_scale,
        head_offset_x=assembly_offset_x,
        head_offset_y=head_offset_y,
        body_offset_x=assembly_offset_x,
        body_offset_y=body_offset_y,
    )


__all__ = ["PUPPET_MATRIX", "PuppetMatrixTransform", "solve_puppet_matrix"]
