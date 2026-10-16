"""Mention resolution: expand ``@name`` into the verbatim character bible.

The failure mode this exists to prevent: an agent paraphrases the character
description between shots, and the face drifts because the prompt drifted.
Expansion here is a straight text substitution of a locked artifact. It never
rewrites the bible.

Pure Python.
"""

from __future__ import annotations

import re
from typing import Callable, Iterable, Mapping

# @handle: letters, digits, underscore, hyphen. Must not be preceded by a word
# character (so an email like a@b does not match) and must not be followed by
# a word character (so @santiago2 does not half-match @santiago).
#
# Backticks are excluded on both sides on purpose: the compiled character bible
# documents its own handle as `@santiago`, and that is prose about the handle,
# not a reference to resolve. Without this the bible would re-inject a mention
# into its own expansion and break idempotency.
MENTION_RE = re.compile(r"(?<![\w@`])@([A-Za-z][A-Za-z0-9_-]*)(?!`)")


class MentionError(ValueError):
    """Raised when a prompt references a handle that is not in the library."""


def find_mentions(text: str) -> list[str]:
    """Return the distinct handles referenced in ``text``, in order."""
    seen: dict[str, None] = {}
    for match in MENTION_RE.finditer(text or ""):
        seen.setdefault(match.group(1).lower(), None)
    return list(seen)


def expand_mentions(
    text: str,
    library: Mapping[str, str],
    *,
    strict: bool = True,
    style: str = "inline",
    max_inline_chars: int = 600,
) -> str:
    """Expand every ``@handle`` in ``text`` from ``library``.

    ``library`` maps a bare handle (``"santiago"``, no ``@``) to its bible
    text. Keys are matched case-insensitively.

    ``style``:
      * ``"inline"`` - substitute the bible in place, flattened to one line.
        Use when the prompt has a length budget.
      * ``"append"`` - replace the mention with the handle and append a
        delimited reference block at the end. Use for video models that
        weight early tokens more heavily.
      * ``"strip"``  - remove the mention entirely, leaving the bible out.
        Use to measure drift *without* the lock (the control condition).

    ``strict`` raises ``MentionError`` on an unknown handle. Set ``False`` to
    leave unknown mentions untouched, which is useful while drafting.
    """
    if style not in {"inline", "append", "strip"}:
        raise ValueError(f"unknown style {style!r}")

    lookup = {k.lower().lstrip("@"): v for k, v in library.items()}
    appended: list[tuple[str, str]] = []

    def _sub(match: re.Match[str]) -> str:
        raw = match.group(1)
        key = raw.lower()
        if key not in lookup:
            if strict:
                raise MentionError(
                    f"@{raw} is not in the library (have: "
                    f"{sorted(lookup) or 'nothing'})"
                )
            return match.group(0)
        if style == "strip":
            return ""
        bible = lookup[key]
        if style == "append":
            if key not in [k for k, _ in appended]:
                appended.append((key, bible))
            return f"@{raw}"
        flat = " ".join(bible.split())
        if len(flat) > max_inline_chars:
            flat = flat[: max_inline_chars - 1].rstrip() + "…"
        return flat

    expanded = MENTION_RE.sub(_sub, text or "")

    if style == "append" and appended:
        block = ["", "---", "REFERENCE ELEMENTS (verbatim, do not paraphrase):"]
        for key, bible in appended:
            block.append("")
            block.append(f"[@{key}]")
            block.append(bible.strip())
        expanded = expanded.rstrip() + "\n" + "\n".join(block)

    return re.sub(r"[ \t]{2,}", " ", expanded).strip() if style == "inline" else expanded


def library_from_specs(specs: Iterable[object], render: Callable[[object], str]) -> dict[str, str]:
    """Build an expansion library from CharacterSpec objects.

    Small helper so the node wrappers do not each re-implement it.
    """
    out: dict[str, str] = {}
    for spec in specs:
        handle = getattr(spec, "handle", None)
        if not handle:
            continue
        out[handle.lstrip("@").lower()] = render(spec)
    return out
