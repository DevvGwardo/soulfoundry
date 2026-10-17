"""CharacterLock: freeze a reference so identity survives the whole project.

Three routes, in order of cost:
  A. Embedding (IP-Adapter / PuLID / InstantID) - zero training, the default.
  B. LoRA - highest fidelity, needs 10-30 varied images and GPU time.
  C. Reference latent - for reference-to-video (R2V), the 2026 default for
     multi-shot video.

Route A now resolves to an actual vector in one of three ways, and the status
string always says which:

``computed``
    An ``image`` was passed, so this node ran SCRFD + ArcFace and wrote the
    vector to ``<project>/embeddings/<name>_reference.npy``.

``loaded``
    ``embed_path`` pointed at a ``.npy`` this pack wrote (or an ``embedding``
    socket carried one), and the vector was read back so the lock holds the
    vector itself, not a pointer.

``external``
    ``embed_path`` pointed at an artifact another pack owns (``.pt``,
    ``.safetensors``). The path is recorded and the status says the vector was
    not loaded, because this pack cannot read someone else's format. That is
    honest rather than pretending the lock is complete.

Routes B and C remain records: training a LoRA and extracting a reference
latent belong to other node packs. This node records where those artifacts
live and which spec digest they were locked from, so the lock is verifiable.
"""

from __future__ import annotations

from pathlib import Path

from .core.embed import EmbedderUnavailable, embed_image, load_embedding, save_embedding
from .core.param_grid import CharacterSpec
from .core.store import Element
from .core.types import CHARACTER_SPEC, ELEMENT, EMBEDDING, SHEET

CATEGORY = "SoulFoundry/lock"

ROUTES = ("embedding", "lora", "reference_latent")

#: Extensions this pack can read as a vector. Anything else on ``embed_path``
#: is treated as another pack's artifact and only recorded.
_VECTOR_SUFFIXES = (".npy",)


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
                "image": ("IMAGE",),
                "embedding": (EMBEDDING,),
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
                "backend": (["auto", "arcface", "perceptual"], {"default": "auto"}),
                "allow_perceptual": ("BOOLEAN", {"default": False}),
                "model_dir": ("STRING", {"default": "", "multiline": False}),
            },
        }

    RETURN_TYPES = (ELEMENT, EMBEDDING, "STRING")
    RETURN_NAMES = ("element", "embedding", "status")
    FUNCTION = "lock"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Freeze a character reference. Route A (embedding) computes a real "
        "ArcFace vector from an image and saves it, or loads a .npy vector; an "
        "external .pt/.safetensors path is recorded but cannot be read here and "
        "the status says so. Route B (LoRA) is the high-fidelity upgrade."
    )

    def lock(self, spec: dict, route: str = "embedding", image=None,
             embedding: dict | None = None, sheet=None, sheet_path: str = "",
             embed_path: str = "", lora_path: str = "", trigger_token: str = "",
             bible_md: str = "", project_path: str = "", backend: str = "auto",
             allow_perceptual: bool = False,
             model_dir: str = "") -> tuple[dict, dict, str]:
        parsed = CharacterSpec.from_dict(spec)
        notes: list[str] = []

        if route == "lora" and not lora_path:
            raise ValueError("route='lora' needs lora_path")
        if route == "reference_latent" and not (sheet_path or sheet):
            raise ValueError("route='reference_latent' needs a sheet")

        # -- route A ---------------------------------------------------------
        embed_payload: dict = {}
        state = "none"
        if route == "embedding":
            embed_payload, embed_path, state = self._resolve_embedding(
                image=image,
                embedding=embedding,
                embed_path=embed_path,
                project_path=project_path,
                name=parsed.name,
                backend=backend,
                allow_perceptual=allow_perceptual,
                model_dir=model_dir,
                notes=notes,
            )
            if state == "none":
                raise ValueError(
                    "route='embedding' needs an image, an embedding socket, or an "
                    "embed_path. Without one there is nothing to lock: the node "
                    "would only be recording a path to nothing."
                )

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

        status = f"locked @{element.name} via {route}"
        if state == "computed":
            used = embed_payload.get("backend", "?")
            status += f" [{embed_payload.get('dim')}-d {used}, computed]"
            if used != "arcface":
                status += " (NOT face identity)"
        elif state == "loaded":
            status += f" [{embed_payload.get('dim')}-d loaded from vector]"
        elif state == "external":
            status += (
                " [external artifact recorded, vector NOT loaded: this pack "
                "cannot read that format, so the audit will need the artifact "
                "or an embedding socket]"
            )
        status += f" (spec {parsed.digest()})"

        if project_path:
            from .core.store import ProjectStore

            with ProjectStore(project_path) as store:
                store.save_element(element)
                store.pin(element.name, True)
            status += f"; saved to {project_path}"

        if notes:
            status += "; " + "; ".join(notes)

        return (element.to_dict(), embed_payload, status)

    # -- helpers -----------------------------------------------------------

    def _resolve_embedding(
        self,
        *,
        image,
        embedding: dict | None,
        embed_path: str,
        project_path: str,
        name: str,
        backend: str,
        allow_perceptual: bool,
        model_dir: str,
        notes: list[str],
    ) -> tuple[dict, str, str]:
        """Return (payload, embed_path, state).

        state is one of ``computed``, ``loaded``, ``external`` or ``none``.
        """

        def _target() -> Path:
            base = (
                Path(project_path).parent / "embeddings"
                if project_path
                else Path("soulfoundry_embeddings")
            )
            return base / f"{_slug(name)}_reference.npy"

        # 1. an upstream CharacterEmbed already did the work
        if embedding and embedding.get("vector"):
            vector = list(embedding["vector"])
            used = embedding.get("backend", "unknown")
            is_face = bool(embedding.get("is_face_embedding", used == "arcface"))
            if not is_face:
                notes.append("reference is a perceptual descriptor, not face identity")
            if not embed_path:
                embed_path = str(
                    save_embedding(
                        vector,
                        _target(),
                        meta={"backend": used, "is_face_embedding": is_face, "label": name},
                    )
                )
            return (
                {
                    "vector": vector,
                    "dim": len(vector),
                    "backend": used,
                    "is_face_embedding": is_face,
                    "path": embed_path,
                    "state": "computed",
                },
                embed_path,
                "computed",
            )

        # 2. compute from the image we were handed
        if image is not None:
            try:
                result = embed_image(
                    image,
                    backend=backend,
                    model_dir=model_dir or None,
                    allow_perceptual=allow_perceptual,
                    max_faces=1,
                )
            except EmbedderUnavailable as exc:
                raise ValueError(
                    f"no face embedder available: {exc}. Run "
                    f"scripts/fetch_models.py, or set allow_perceptual=True."
                ) from exc
            if not result.ok:
                detail = "; ".join(result.notes) or "no face detected"
                raise ValueError(f"no face found in the reference image ({detail})")

            vector = result.primary
            used = result.backend
            if used != "arcface":
                notes.append("reference is a perceptual descriptor, not face identity")

            if not embed_path:
                embed_path = str(
                    save_embedding(
                        vector,
                        _target(),
                        meta={
                            "backend": used,
                            "is_face_embedding": used == "arcface",
                            "label": name,
                            "dim": result.dim,
                        },
                    )
                )
            return (
                {
                    "vector": vector,
                    "dim": result.dim,
                    "backend": used,
                    "is_face_embedding": used == "arcface",
                    "path": embed_path,
                    "faces": [f.to_dict() for f in result.faces],
                    "state": "computed",
                },
                embed_path,
                "computed",
            )

        # 3. an existing path: read it if it is our format, record it if not
        if embed_path:
            suffix = Path(embed_path).suffix.lower()
            if suffix in _VECTOR_SUFFIXES and Path(embed_path).is_file():
                vector, meta = load_embedding(embed_path)
                used = meta.get("backend", "unknown")
                if used != "arcface":
                    notes.append(f"loaded reference backend is {used}, not arcface")
                return (
                    {
                        "vector": vector,
                        "dim": len(vector),
                        "backend": used,
                        "is_face_embedding": used == "arcface",
                        "path": embed_path,
                        "state": "loaded",
                    },
                    embed_path,
                    "loaded",
                )

            # someone else's artifact (.pt, .safetensors, a missing file)
            reason = (
                "file does not exist"
                if not Path(embed_path).exists()
                else f"unsupported format {suffix or '(none)'}"
            )
            notes.append(f"recorded external artifact only: {reason}")
            return (
                {
                    "vector": [],
                    "dim": 0,
                    "backend": "external",
                    "is_face_embedding": False,
                    "path": embed_path,
                    "state": "external",
                },
                embed_path,
                "external",
            )

        return ({}, "", "none")


def _slug(name: str) -> str:
    keep = [c if (c.isalnum() or c in "-_") else "-" for c in name.strip().lower()]
    slug = "".join(keep).strip("-")
    return slug or "character"


__all__ = ["CharacterLock", "ROUTES"]
