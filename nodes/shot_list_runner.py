"""ShotListRunner: run a whole sequence with locks applied.

The old way is re-roll each shot until the face looks right, which is both
slow and how sequences drift. This node resolves every shot's mentions,
binds Elements, and emits a run plan plus the store records, so one sequence
runs as a unit.

It deliberately does not execute the video model itself. It produces the plan
and the records; execution is the workflow's job.
"""

from __future__ import annotations

import json

from .core.mentions import MentionError, expand_mentions, find_mentions
from .core.types import SHOT_LIST

CATEGORY = "SoulFoundry/reuse"


class ShotListRunner:
    """Resolve and bind a whole shot list in one pass."""

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "shots_json": (
                    "STRING",
                    {"default": "[]", "multiline": True},
                ),
            },
            "optional": {
                "library_json": ("STRING", {"default": "{}", "multiline": True}),
                "style": (["inline", "append", "strip"], {"default": "inline"}),
                "strict": ("BOOLEAN", {"default": True}),
                "project_path": ("STRING", {"default": "", "multiline": False}),
            },
        }

    RETURN_TYPES = (SHOT_LIST, "STRING")
    RETURN_NAMES = ("run_plan", "report")
    FUNCTION = "run"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Resolve @mentions across an entire shot list in one pass and emit a "
        "run plan. shots_json is [{seq, prompt}, ...]."
    )

    def run(self, shots_json: str, library_json: str = "{}",
            style: str = "inline", strict: bool = True,
            project_path: str = "") -> tuple[dict, str]:
        shots = _load_shots(shots_json)
        try:
            library: dict[str, str] = json.loads(library_json or "{}")
        except json.JSONDecodeError as exc:
            raise ValueError(f"library_json is not valid JSON: {exc}") from exc
        if not isinstance(library, dict):
            raise ValueError("library_json must decode to an object")

        plan: list[dict] = []
        bound: dict[str, int] = {}
        for shot in shots:
            prompt = shot.get("prompt", "")
            mentioned = find_mentions(prompt)
            try:
                expanded = expand_mentions(prompt, library, strict=strict, style=style)
            except MentionError as exc:
                raise ValueError(f"shot {shot.get('seq')}: {exc}") from exc
            for handle in mentioned:
                bound[handle] = bound.get(handle, 0) + 1
            plan.append(
                {
                    "seq": shot.get("seq"),
                    "prompt": expanded,
                    "elements": mentioned,
                    "original_chars": len(prompt),
                    "expanded_chars": len(expanded),
                }
            )

        if project_path:
            from .core.store import ProjectStore, Shot

            with ProjectStore(project_path) as store:
                for entry in plan:
                    store.save_shot(
                        Shot(
                            seq=int(entry["seq"] or 0),
                            prompt=entry["prompt"],
                            bound_json=json.dumps(entry["elements"]),
                        )
                    )

        report_lines = [
            f"{len(plan)} shot(s) resolved, style={style}",
            f"elements bound: {', '.join(f'@{k} ({v}x)' for k, v in sorted(bound.items())) or 'none'}",
        ]
        if project_path:
            report_lines.append(f"persisted to {project_path}")
        if not plan:
            report_lines.append("WARNING: shot list was empty")

        return ({"shots": plan, "count": len(plan)}, "\n".join(report_lines))


def _load_shots(raw: str) -> list[dict]:
    try:
        payload = json.loads(raw or "[]")
    except json.JSONDecodeError as exc:
        raise ValueError(f"shots_json is not valid JSON: {exc}") from exc
    if not isinstance(payload, list):
        raise ValueError("shots_json must decode to an array")
    out: list[dict] = []
    for i, item in enumerate(payload):
        if isinstance(item, str):
            out.append({"seq": i + 1, "prompt": item})
            continue
        if not isinstance(item, dict):
            raise ValueError(f"shots_json[{i}] must be an object or a string")
        out.append(
            {
                "seq": item.get("seq", i + 1),
                "prompt": item.get("prompt", ""),
            }
        )
    return out
