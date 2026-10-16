"""CastSheetExport: export the character bible so it is portable and diffable.

The counter-position to Higgsfield's $5 lock fee and non-exportable identity.
Everything this node writes is plain markdown or JSON, so it can live in git.
"""

from __future__ import annotations

import json

from .core.param_grid import CharacterSpec, compile_bible, compile_prompt
from .core.types import ELEMENT

CATEGORY = "SoulFoundry/export"

FORMATS = ("markdown", "json", "prompt")


class CastSheetExport:
    """Export an Element or a spec as markdown, JSON, or a bare prompt."""

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "format": (list(FORMATS), {"default": "markdown"}),
            },
            "optional": {
                "spec": ("SOULFOUNDRY_SPEC",),
                "element": (ELEMENT,),
                "include_backstory": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("export", "filename_suggestion")
    FUNCTION = "export"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Export a character as portable markdown, JSON, or a bare identity "
        "prompt. Free and diffable, unlike the closed equivalent."
    )

    def export(self, format: str = "markdown", spec: dict | None = None,
               element: dict | None = None,
               include_backstory: bool = True) -> tuple[str, str]:
        if spec is None and element is None:
            raise ValueError("give at least one of spec or element")

        parsed = (
            CharacterSpec.from_dict(spec)
            if spec is not None
            else _spec_from_element(element or {})
        )

        if format == "json":
            payload = parsed.to_dict()
            if not include_backstory:
                for key in ("backstory", "motivation", "fear", "flaw", "strength"):
                    payload.pop(key, None)
            payload["_compile_prompt"] = compile_prompt(parsed)
            payload["_digest"] = parsed.digest()
            body = json.dumps(payload, indent=2, sort_keys=True)
            name = f"{parsed.name}.json"
        elif format == "prompt":
            body = compile_prompt(parsed)
            name = f"{parsed.name}-prompt.txt"
        else:
            body = (
                compile_bible(parsed)
                if include_backstory
                else _bible_without_story(parsed)
            )
            name = f"{parsed.name}-bible.md"

        return (body, name)


def _bible_without_story(spec: CharacterSpec) -> str:
    full = compile_bible(spec)
    marker = "\n## Story\n"
    if marker in full:
        head, _, tail = full.partition(marker)
        tail = tail.split("\n---\n", 1)
        remainder = "\n---\n" + tail[1] if len(tail) > 1 else ""
        return head + remainder
    return full


def _spec_from_element(element: dict) -> CharacterSpec:
    raw = element.get("spec_json")
    if not raw:
        raise ValueError(
            "element has no spec_json; pass a spec instead, or re-lock the "
            "character from a spec so it is reproducible"
        )
    if isinstance(raw, str):
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"element spec_json is not valid JSON: {exc}") from exc
    else:
        data = raw
    if not isinstance(data, dict):
        raise ValueError("element spec_json must decode to an object")
    return CharacterSpec.from_dict(data)
