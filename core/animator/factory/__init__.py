"""V3 autonomous puppet ingestion and rigging.

The factory is deliberately separate from the V2 renderer.  It only turns
previously unrigged character art into the existing ``puppet.json`` contract.
"""
from .color_extractor import PuppetPalette, extract_palette
from .landmark_detector import FacialLandmarks, detect_landmarks
from .puppet_matrix import PUPPET_MATRIX, PuppetMatrixTransform, solve_puppet_matrix
from .rigger import auto_rig_character, has_character_imagery

__all__ = [
    "FacialLandmarks",
    "PuppetPalette",
    "PUPPET_MATRIX",
    "PuppetMatrixTransform",
    "auto_rig_character",
    "detect_landmarks",
    "extract_palette",
    "has_character_imagery",
    "solve_puppet_matrix",
]
