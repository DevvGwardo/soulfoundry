"""MentionResolver: expand @name into the verbatim character bible.

Kept separate from ElementInjector so the expansion contract can be tested on
its own, and so a prompt-only workflow (no image, no store) can still use it.
"""

from __future__ import annotations

import json

from .core.mentions import MentionError, expand_mentions, find_mentions
from .core.types import ELEMENT

CATEGORY = "SoulFoundry/reuse"


class MentionResolver:
    """Expand @handles in a prompt from a library or a single Element."""

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {
                "element": (ELEMENT,),
                "library_json": ("STRING", {"default": "{}", "multiline": True}),
                "style": (["inline", "append", "strip"], {"default": "inline"}),
                "strict": ("BOOLEAN", {"default": True}),
                "max_inline_chars": (
                    "INT",
                    {"default": 600, "min": 80, "max": 8000, "step": 20},
                ),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "INT")
    RETURN_NAMES = ("expanded", "mentions", "count")
    FUNCTION = "resolve"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Expand @name mentions to their full character bible verbatim, so "
        "identity text is never paraphrased away between shots."
    )

    def resolve(self, prompt: str, element: dict | None = None,
                library_json: str = "{}", style: str = "inline",
                strict: bool = True,
                max_inline_chars: int = 600) -> tuple[str, str, int]:
        try:
            library: dict[str, str] = json.loads(library_json or "{}")
        except json.JSONDecodeError as exc:
            raise ValueError(f"library_json is not valid JSON: {exc}") from exc
        if not isinstance(library, dict):
            raise ValueError("library_json must decode to an object")

        if element:
            name = (element.get("name") or "").lstrip("@").lower()
            bible = element.get("bible_md") or ""
            if not name:
                raise ValueError("element has no name")
            library[name] = bible

        found = find_mentions(prompt)
        try:
            expanded = expand_mentions(
                prompt,
                library,
                strict=strict,
                style=style,
                max_inline_chars=int(max_inline_chars),
            )
        except MentionError as exc:
            raise ValueError(str(exc)) from exc

        return (expanded, ", ".join(f"@{m}" for m in found), len(found))
