"""CastParamGrid: the eight Soul Cast categories as widgets.

Emits a CHARACTER_SPEC. No prompting required, which is the whole point: the
user declares identity as parameters, and the prompt compiles deterministically
from them.
"""

from __future__ import annotations

from .core.param_grid import (
    ARCHETYPES,
    BUILDS,
    DETAILS,
    ERAS,
    EYE_COLORS,
    FACIAL_HAIR,
    GENRES,
    HAIR_COLORS,
    HAIR_STYLES,
    HAIR_TEXTURES,
    OUTFITS,
    CharacterSpec,
)
from .core.types import CHARACTER_SPEC

CATEGORY = "SoulFoundry/cast"


class CastParamGrid:
    """Declare a character once, as structured data."""

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "name": ("STRING", {"default": "santiago", "multiline": False}),
                "genre": (list(GENRES), {"default": "Drama"}),
                "budget": (
                    "INT",
                    {"default": 50, "min": 0, "max": 100, "step": 1},
                ),
                "era": (list(ERAS), {"default": "Contemporary"}),
                "archetype": (list(ARCHETYPES), {"default": "Everyman"}),
                "gender": (
                    "STRING",
                    {"default": "unspecified", "multiline": False},
                ),
                "race_ethnicity": (
                    "STRING",
                    {"default": "unspecified", "multiline": False},
                ),
                "age": ("STRING", {"default": "30s", "multiline": False}),
                "build": (list(BUILDS), {"default": "Average"}),
                "height": ("STRING", {"default": "average height", "multiline": False}),
                "eye_color": (list(EYE_COLORS), {"default": "Brown"}),
                "hair_style": (list(HAIR_STYLES), {"default": "Short"}),
                "hair_texture": (list(HAIR_TEXTURES), {"default": "Straight"}),
                "hair_color": (list(HAIR_COLORS), {"default": "Dark Brown"}),
                "facial_hair": (list(FACIAL_HAIR), {"default": "None"}),
                "details": (list(DETAILS), {"default": "None"}),
                "outfit": (list(OUTFITS), {"default": "Casual"}),
            },
            "optional": {
                "details_manual": ("STRING", {"default": "", "multiline": True}),
                "outfit_notes": ("STRING", {"default": "", "multiline": True}),
                "notes": ("STRING", {"default": "", "multiline": True}),
            },
        }

    RETURN_TYPES = (CHARACTER_SPEC,)
    RETURN_NAMES = ("spec",)
    FUNCTION = "build"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Build a character from the eight Soul Cast parameter categories. "
        "Outputs a CHARACTER_SPEC consumed by every other SoulFoundry node."
    )

    def build(self, name: str, genre: str, budget: int, era: str, archetype: str,
              gender: str, race_ethnicity: str, age: str, build: str, height: str,
              eye_color: str, hair_style: str, hair_texture: str, hair_color: str,
              facial_hair: str, details: str, outfit: str,
              details_manual: str = "", outfit_notes: str = "",
              notes: str = "") -> tuple[dict]:
        spec = CharacterSpec(
            name=name,
            genre=genre,
            budget=int(budget),
            era=era,
            archetype=archetype,
            gender=gender,
            race_ethnicity=race_ethnicity,
            age=age,
            build=build,
            height=height,
            eye_color=eye_color,
            hair_style=hair_style,
            hair_texture=hair_texture,
            hair_color=hair_color,
            facial_hair=facial_hair,
            details=details,
            details_manual=details_manual,
            outfit=outfit,
            outfit_notes=outfit_notes,
            notes=notes,
        )
        return (spec.to_dict(),)
