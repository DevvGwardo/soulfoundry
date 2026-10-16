"""SheetBuilder: compile a spec into the 3-panel turnaround prompt.

Emits a plain STRING prompt (and the compiled identity clause) so it can be
fed to any text-to-image node. The lighting rules are baked in here, not left
to the user, because most sheet failures are lighting failures.
"""

from __future__ import annotations

from .core.param_grid import CharacterSpec, compile_bible, compile_prompt, spec_seed
from .core.sheet import build_sheet_prompt
from .core.types import CHARACTER_SPEC

CATEGORY = "SoulFoundry/sheet"


class SheetBuilder:
    """Generate the lock artifact prompt from a spec."""

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "spec": (CHARACTER_SPEC,),
            },
            "optional": {
                "style": (
                    "STRING",
                    {"default": "studio reference photography", "multiline": False},
                ),
                "extra_negative": ("STRING", {"default": "", "multiline": True}),
                "use_bible": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "INT")
    RETURN_NAMES = ("sheet_prompt", "identity_prompt", "seed")
    FUNCTION = "build"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Compile a CHARACTER_SPEC into a 3-panel character sheet prompt with "
        "the lighting rules baked in. Also emits the identity clause and a "
        "deterministic seed so the same spec always produces the same sheet."
    )

    def build(self, spec: dict, style: str = "studio reference photography",
              extra_negative: str = "", use_bible: bool = False) -> tuple[str, str, int]:
        parsed = CharacterSpec.from_dict(spec)
        identity = compile_bible(parsed) if use_bible else compile_prompt(parsed)
        prompt = build_sheet_prompt(
            compile_prompt(parsed),
            style=style,
            extra_negative=extra_negative,
        )
        return (prompt, identity, spec_seed(parsed))
