"""ElementInjector: the core reuse primitive.

Takes a shot prompt plus bound Elements and produces the final prompt with
the locked identity injected. This is where drift is prevented: identity text
is inserted verbatim from the Element, never re-described.
"""

from __future__ import annotations

from typing import Iterable

from .core.mentions import MentionError, expand_mentions, find_mentions
from .core.store import Element
from .core.types import ELEMENT

CATEGORY = "SoulFoundry/reuse"

STYLES = ("inline", "append", "strip")


class ElementInjector:
    """Inject locked Elements into a shot prompt."""

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {
                "element": (ELEMENT,),
                "library_json": (
                    "STRING",
                    {"default": "{}", "multiline": True},
                ),
                "style": (list(STYLES), {"default": "inline"}),
                "strict": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("prompt", "report")
    FUNCTION = "inject"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Expand @mentions in a shot prompt from locked Elements. style="
        "'inline' substitutes the bible in place, 'append' adds a reference "
        "block, 'strip' removes mentions (the unlocked control condition)."
    )

    def inject(self, prompt: str, element: dict | None = None,
               library_json: str = "{}", style: str = "inline",
               strict: bool = True) -> tuple[str, str]:
        import json

        try:
            library: dict[str, str] = json.loads(library_json or "{}")
        except json.JSONDecodeError as exc:
            raise ValueError(f"library_json is not valid JSON: {exc}") from exc
        if not isinstance(library, dict):
            raise ValueError("library_json must decode to an object")

        if element:
            name = element.get("name", "")
            bible = element.get("bible_md") or element.get("backstory_md") or ""
            if not bible:
                raise ValueError(
                    f"element {name!r} has no bible_md to inject"
                )
            library[name.lstrip("@").lower()] = bible

        mentioned = find_mentions(prompt)
        try:
            expanded = expand_mentions(prompt, library, strict=strict, style=style)
        except MentionError as exc:
            raise ValueError(str(exc)) from exc

        if not mentioned:
            report = "no @mentions in prompt; passed through unchanged"
        else:
            report = (
                f"expanded {len(mentioned)} element(s) "
                f"[{', '.join('@' + m for m in mentioned)}] style={style}; "
                f"{len(prompt)} -> {len(expanded)} chars"
            )
        return (expanded, report)


class LocationElement:
    """Build a location Element from a reference image path."""

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "name": ("STRING", {"default": "location", "multiline": False}),
                "description": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {
                "image_path": ("STRING", {"default": "", "multiline": False}),
            },
        }

    RETURN_TYPES = (ELEMENT,)
    RETURN_NAMES = ("element",)
    FUNCTION = "build"
    CATEGORY = CATEGORY
    DESCRIPTION = "Create a location Element (environment, setting, atmosphere)."

    def build(self, name: str, description: str = "",
              image_path: str = "") -> tuple[dict]:
        el = Element(
            name=name,
            kind="location",
            sheet_path=image_path or None,
            bible_md=description or None,
        )
        return (el.to_dict(),)


class PropElement:
    """Build a prop Element from a reference image path."""

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "name": ("STRING", {"default": "prop", "multiline": False}),
                "description": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {
                "image_path": ("STRING", {"default": "", "multiline": False}),
            },
        }

    RETURN_TYPES = (ELEMENT,)
    RETURN_NAMES = ("element",)
    FUNCTION = "build"
    CATEGORY = CATEGORY
    DESCRIPTION = "Create a prop Element (object, vehicle, set piece)."

    def build(self, name: str, description: str = "",
              image_path: str = "") -> tuple[dict]:
        el = Element(
            name=name,
            kind="prop",
            sheet_path=image_path or None,
            bible_md=description or None,
        )
        return (el.to_dict(),)
