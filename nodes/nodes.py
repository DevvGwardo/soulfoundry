"""ComfyUI node wrappers.

Thin shells over ``nodes/core``. Every node here should be readable in under
a minute: validate, call core, return. All real logic lives in core so it can
be tested without ComfyUI installed.

If ComfyUI is absent, importing this package still succeeds; the mapping is
built but the classes simply are not registered with the runtime.
"""

from __future__ import annotations

from .cast_param_grid import CastParamGrid
from .cast_sheet_export import CastSheetExport
from .character_lock import CharacterLock
from .consistency_audit import ConsistencyAudit
from .element_injector import ElementInjector, LocationElement, PropElement
from .element_library import ElementLibrary
from .face_embed import CharacterEmbed, EmbeddingFromFile
from .mention_resolver import MentionResolver
from .sheet_builder import SheetBuilder
from .sheet_qc import SheetQCBatch, SheetQC
from .shot_list_runner import ShotListRunner

NODE_CLASS_MAPPINGS: dict[str, type] = {
    "SoulFoundryCastParamGrid": CastParamGrid,
    "SoulFoundrySheetBuilder": SheetBuilder,
    "SoulFoundrySheetQC": SheetQC,
    "SoulFoundrySheetQCBatch": SheetQCBatch,
    "SoulFoundryCharacterEmbed": CharacterEmbed,
    "SoulFoundryEmbeddingFromFile": EmbeddingFromFile,
    "SoulFoundryCharacterLock": CharacterLock,
    "SoulFoundryElementLibrary": ElementLibrary,
    "SoulFoundryElementInjector": ElementInjector,
    "SoulFoundryLocationElement": LocationElement,
    "SoulFoundryPropElement": PropElement,
    "SoulFoundryMentionResolver": MentionResolver,
    "SoulFoundryConsistencyAudit": ConsistencyAudit,
    "SoulFoundryShotListRunner": ShotListRunner,
    "SoulFoundryCastSheetExport": CastSheetExport,
}

NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = {
    "SoulFoundryCastParamGrid": "Cast Parameter Grid (SoulFoundry)",
    "SoulFoundrySheetBuilder": "Character Sheet Builder (SoulFoundry)",
    "SoulFoundrySheetQC": "Sheet QC / Reject Test (SoulFoundry)",
    "SoulFoundrySheetQCBatch": "Sheet QC Batch Pick (SoulFoundry)",
    "SoulFoundryCharacterEmbed": "Character Embed / Face Vector (SoulFoundry)",
    "SoulFoundryEmbeddingFromFile": "Embedding From File (SoulFoundry)",
    "SoulFoundryCharacterLock": "Character Lock (SoulFoundry)",
    "SoulFoundryElementLibrary": "Element Library (SoulFoundry)",
    "SoulFoundryElementInjector": "Element Injector (SoulFoundry)",
    "SoulFoundryLocationElement": "Location Element (SoulFoundry)",
    "SoulFoundryPropElement": "Prop Element (SoulFoundry)",
    "SoulFoundryMentionResolver": "Mention Resolver (SoulFoundry)",
    "SoulFoundryConsistencyAudit": "Consistency Audit / Drift (SoulFoundry)",
    "SoulFoundryShotListRunner": "Shot List Runner (SoulFoundry)",
    "SoulFoundryCastSheetExport": "Cast Sheet Export (SoulFoundry)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
