"""CharacterLock: freeze a reference so identity survives the whole project.

Three routes, in order of cost:
  A. Embedding (IP-Adapter / PuLID / InstantID) - zero training, the default.
  B. LoRA - highest fidelity, needs 10-30 varied images and GPU time.
  C. Reference latent - for reference-to-video (R2V), the 2026 default for
     multi-shot video.

The node does not run any of those models itself. It records which route was
used, where the artifact lives, and the spec digest it was locked from, so the
lock can be verified later. That record is the Element.
"""

from __future__ import annotations

from .core.param_grid import CharacterSpec
from .core.store import Element
from .core.types import CHARACTER_SPEC, ELEMENT, SHEET

CATEGORY = "SoulFoundry/lock"

ROUTES = ("embedding", "lora", "reference_latent")


class CharacterLock:
    """Turn an approved sheet into a persistent, reusable Element."""

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "spec": (CHARACTER_SPEC,),
                "route": (list(ROUTES), {"default": "embedding"}),
            },
            "optional": {
                "sheet": (SHEET,),
                "sheet_path": ("STRING", {"default": "", "multiline": False}),
                "embed_path": ("STRING", {"default": "", "multiline": False}),
                "lora_path": ("STRING", {"default": "", "multiline": False}),
                "trigger_token": ("STRING", {"default": "", "multiline": False}),
                "bible_md": ("STRING", {"default": "", "multiline": True}),
                "project_path": (
                    "STRING",
                    {"default": "", "multiline": False},
                ),
            },
        }

    RETURN_TYPES = (ELEMENT, "STRING")
    RETURN_NAMES = ("element", "status")
    FUNCTION = "lock"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Freeze a character reference and record how it was locked. Route A "
        "(embedding) needs no training; Route B (LoRA) is the high-fidelity "
        "upgrade. Writes to the project store when project_path is set."
    )

    def lock(self, spec: dict, route: str = "embedding", sheet=None,
             sheet_path: str = "", embed_path: str = "", lora_path: str = "",
             trigger_token: str = "", bible_md: str = "",
             project_path: str = "") -> tuple[dict, str]:
        parsed = CharacterSpec.from_dict(spec)

        if route == "embedding" and not (embed_path or sheet_path or sheet):
            raise ValueError(
                "route='embedding' needs an embed_path, sheet_path or sheet input"
            )
        if route == "lora" and not lora_path:
            raise ValueError("route='lora' needs lora_path")
        if route == "reference_latent" and not (sheet_path or sheet):
            raise ValueError("route='reference_latent' needs a sheet")

        element = Element(
            name=parsed.name,
            kind="character",
            spec_json=parsed.to_json(),
            spec_digest=parsed.digest(),
            sheet_path=sheet_path or None,
            embed_path=embed_path or None,
            lora_path=lora_path or None,
            trigger_token=trigger_token or None,
            bible_md=bible_md or None,
        )

        status = f"locked @{element.name} via {route} (spec {parsed.digest()})"

        if project_path:
            from .core.store import ProjectStore

            with ProjectStore(project_path) as store:
                store.save_element(element)
                store.pin(element.name, True)
            status += f"; saved to {project_path}"

        return (element.to_dict(), status)
