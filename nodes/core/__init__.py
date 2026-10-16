"""SoulFoundry core: pure-Python logic, zero ComfyUI dependency.

Everything in this package is importable and testable without ComfyUI,
torch, or a GPU. The ComfyUI node wrappers in ``nodes/`` are thin shells
over these modules.
"""

from .param_grid import (
    ARCHETYPES,
    BUDGET_PRESETS,
    DETAILS,
    ERAS,
    GENRES,
    OUTFITS,
    CharacterSpec,
    compile_bible,
    compile_prompt,
    spec_seed,
)
from .sheet import (
    LIGHTING_RULES,
    PANELS,
    SHEET_TEMPLATE,
    SheetQCResult,
    build_sheet_prompt,
    qc_sheet,
)
from .mentions import MentionError, expand_mentions, find_mentions
from .drift import (
    DRIFT_FLOOR,
    LOCK_CEILING,
    DriftReport,
    build_report,
    cosine_similarity,
    score_against_reference,
)
from .store import Element, ProjectStore, Shot

__all__ = [
    "ARCHETYPES",
    "BUDGET_PRESETS",
    "DETAILS",
    "DRIFT_FLOOR",
    "ERAS",
    "GENRES",
    "LOCK_CEILING",
    "OUTFITS",
    "LIGHTING_RULES",
    "PANELS",
    "SHEET_TEMPLATE",
    "CharacterSpec",
    "DriftReport",
    "Element",
    "MentionError",
    "ProjectStore",
    "SheetQCResult",
    "Shot",
    "build_report",
    "build_sheet_prompt",
    "compile_bible",
    "compile_prompt",
    "cosine_similarity",
    "expand_mentions",
    "find_mentions",
    "qc_sheet",
    "qc_sheet_metrics",
    "score_against_reference",
    "spec_seed",
]

__version__ = "0.1.0"
