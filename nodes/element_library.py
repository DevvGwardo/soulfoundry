"""ElementLibrary: project-scoped save, load, list and cross-project import.

This node is the persistence claim. A plain uploaded reference anchors for one
generation and expires; an Element lives in the store and comes back next
session.
"""

from __future__ import annotations

from .core.store import ELEMENT_KINDS, Element, ProjectStore
from .core.types import ELEMENT

CATEGORY = "SoulFoundry/library"

ACTIONS = ("save", "load", "list", "import", "delete", "pin", "summary")


class ElementLibrary:
    """Read and write the project store."""

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "project_path": ("STRING", {"default": "projects/default", "multiline": False}),
                "action": (list(ACTIONS), {"default": "list"}),
            },
            "optional": {
                "element": (ELEMENT,),
                "name": ("STRING", {"default": "", "multiline": False}),
                "kind": (["", *ELEMENT_KINDS], {"default": ""}),
                "source_project_path": ("STRING", {"default": "", "multiline": False}),
                "pinned_only": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = (ELEMENT, "STRING", "INT")
    RETURN_NAMES = ("element", "report", "count")
    FUNCTION = "run"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Project-scoped Element library. Save, load, list, pin, delete, or "
        "import an Element from another project."
    )

    def run(self, project_path: str, action: str = "list", element: dict | None = None,
            name: str = "", kind: str = "", source_project_path: str = "",
            pinned_only: bool = False) -> tuple[dict | None, str, int]:
        with ProjectStore(project_path) as store:
            if action == "save":
                if not element:
                    raise ValueError("action='save' needs an element input")
                parsed = Element(
                    **{k: v for k, v in element.items()
                       if k in Element.__dataclass_fields__}
                )
                saved = store.save_element(parsed)
                return (
                    saved.to_dict(),
                    f"saved {saved.handle} ({saved.kind}) to {store.path}",
                    1,
                )

            if action == "load":
                if not name:
                    raise ValueError("action='load' needs a name")
                found = store.get_element(name)
                if found is None:
                    raise KeyError(f"{name!r} not in {store.project_id!r}")
                return (found.to_dict(), f"loaded {found.handle}", 1)

            if action == "import":
                if not name:
                    raise ValueError("action='import' needs a name")
                if not source_project_path:
                    raise ValueError("action='import' needs source_project_path")
                with ProjectStore(source_project_path) as src:
                    imported = store.import_element(src, name)
                return (
                    imported.to_dict(),
                    f"imported {imported.handle} from {src.project_id}",
                    1,
                )

            if action == "delete":
                if not name:
                    raise ValueError("action='delete' needs a name")
                removed = store.delete_element(name)
                return (None, f"deleted {name}" if removed else f"{name} not found",
                        int(removed))

            if action == "pin":
                if not name:
                    raise ValueError("action='pin' needs a name")
                ok = store.pin(name, True)
                return (None, f"pinned {name}" if ok else f"{name} not found", int(ok))

            if action == "summary":
                summary = store.summary()
                return (None, str(summary), summary["total_elements"])

            # list
            elements = store.list_elements(kind or None, pinned_only=pinned_only)
            if not elements:
                return (None, f"no elements in {store.project_id}", 0)
            lines = [
                f"{el.handle:<20} {el.kind:<10} "
                f"{'pinned' if el.pinned else '      '} "
                f"{el.spec_digest or '-'}"
                for el in elements
            ]
            return (None, "\n".join(lines), len(elements))
