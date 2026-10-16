"""Project store: the thing that makes this a product instead of a demo.

SQLite, one file per project, portable. Elements are project-scoped but can
be imported from another project, which is the persistence behaviour being
cloned: a plain uploaded reference anchors for one generation and expires,
an Element persists.

Pure Python (stdlib sqlite3 only).
"""

from __future__ import annotations

import os
import shutil
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Literal

ElementKind = Literal["character", "location", "prop"]
ELEMENT_KINDS: tuple[str, ...] = ("character", "location", "prop")

SCHEMA_VERSION = 1

SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS meta (
  key   TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS elements (
  id            TEXT PRIMARY KEY,
  project_id    TEXT NOT NULL,
  name          TEXT NOT NULL,
  kind          TEXT NOT NULL CHECK (kind IN ('character','location','prop')),
  spec_json     TEXT,
  spec_digest   TEXT,
  sheet_path    TEXT,
  embed_path    TEXT,
  lora_path     TEXT,
  trigger_token TEXT,
  bible_md      TEXT,
  backstory_md  TEXT,
  pinned        INTEGER NOT NULL DEFAULT 0,
  origin_project TEXT,
  created_at    TEXT NOT NULL,
  updated_at    TEXT NOT NULL,
  UNIQUE(project_id, name)
);

CREATE TABLE IF NOT EXISTS shots (
  id           TEXT PRIMARY KEY,
  project_id   TEXT NOT NULL,
  seq          INTEGER NOT NULL,
  prompt       TEXT,
  bound_json   TEXT,
  output_path  TEXT,
  drift_score  REAL,
  created_at   TEXT NOT NULL,
  UNIQUE(project_id, seq)
);

CREATE INDEX IF NOT EXISTS idx_elements_project ON elements(project_id);
CREATE INDEX IF NOT EXISTS idx_elements_kind ON elements(project_id, kind);
CREATE INDEX IF NOT EXISTS idx_shots_project ON shots(project_id, seq);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _new_id() -> str:
    return str(uuid.uuid4())


@dataclass
class Element:
    """A saved, reusable reference. Character, location, or prop."""

    name: str
    kind: ElementKind = "character"
    id: str = field(default_factory=_new_id)
    project_id: str = ""
    spec_json: str | None = None
    spec_digest: str | None = None
    sheet_path: str | None = None
    embed_path: str | None = None
    lora_path: str | None = None
    trigger_token: str | None = None
    bible_md: str | None = None
    backstory_md: str | None = None
    pinned: bool = False
    origin_project: str | None = None
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)

    def __post_init__(self) -> None:
        if self.kind not in ELEMENT_KINDS:
            raise ValueError(f"kind must be one of {ELEMENT_KINDS}, got {self.kind!r}")
        self.name = self.name.strip().lstrip("@")
        if not self.name:
            raise ValueError("Element name cannot be empty")
        if any(ch.isspace() for ch in self.name):
            raise ValueError(
                f"Element name {self.name!r} cannot contain whitespace "
                "(it has to be usable as @handle)"
            )

    @property
    def handle(self) -> str:
        return f"@{self.name}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "project_id": self.project_id,
            "name": self.name,
            "kind": self.kind,
            "spec_json": self.spec_json,
            "spec_digest": self.spec_digest,
            "sheet_path": self.sheet_path,
            "embed_path": self.embed_path,
            "lora_path": self.lora_path,
            "trigger_token": self.trigger_token,
            "bible_md": self.bible_md,
            "backstory_md": self.backstory_md,
            "pinned": bool(self.pinned),
            "origin_project": self.origin_project,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> Element:
        data = dict(row)
        data["pinned"] = bool(data.get("pinned"))
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class Shot:
    seq: int
    prompt: str = ""
    id: str = field(default_factory=_new_id)
    project_id: str = ""
    bound_json: str | None = None
    output_path: str | None = None
    drift_score: float | None = None
    created_at: str = field(default_factory=_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "project_id": self.project_id,
            "seq": self.seq,
            "prompt": self.prompt,
            "bound_json": self.bound_json,
            "output_path": self.output_path,
            "drift_score": self.drift_score,
            "created_at": self.created_at,
        }

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> Shot:
        data = dict(row)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class ProjectStore:
    """One SQLite file per project.

    ``ProjectStore("/projects/noir")`` creates the directory if needed. Use
    ``:memory:`` in tests.
    """

    def __init__(self, path: str | os.PathLike[str] = ":memory:") -> None:
        self.path = str(path)
        if self.path != ":memory:":
            p = Path(self.path)
            if p.suffix != ".sqlite":
                p = p / "project.sqlite"
            p.parent.mkdir(parents=True, exist_ok=True)
            self.path = str(p)
        self._conn = sqlite3.connect(self.path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA_SQL)
        self._conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES('schema_version', ?)",
            (str(SCHEMA_VERSION),),
        )
        self._conn.commit()

    # -- lifecycle ----------------------------------------------------------

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> ProjectStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @property
    def project_id(self) -> str:
        if self.path == ":memory:":
            return "memory"
        return Path(self.path).parent.name

    # -- elements -----------------------------------------------------------

    def save_element(self, element: Element) -> Element:
        """Insert or update by (project, name).

        The id of an existing row is preserved so that shots already bound to
        it keep pointing at the same Element.
        """
        element.project_id = element.project_id or self.project_id
        element.updated_at = _now()
        existing = self.get_element(element.name)
        if existing is not None:
            element.id = existing.id
            element.created_at = existing.created_at
            if element.origin_project is None:
                element.origin_project = existing.origin_project
        row = element.to_dict()
        self._conn.execute(
            """
            INSERT INTO elements (
              id, project_id, name, kind, spec_json, spec_digest, sheet_path,
              embed_path, lora_path, trigger_token, bible_md, backstory_md,
              pinned, origin_project, created_at, updated_at
            ) VALUES (
              :id, :project_id, :name, :kind, :spec_json, :spec_digest,
              :sheet_path, :embed_path, :lora_path, :trigger_token, :bible_md,
              :backstory_md, :pinned, :origin_project, :created_at, :updated_at
            )
            ON CONFLICT(project_id, name) DO UPDATE SET
              kind=excluded.kind,
              spec_json=excluded.spec_json,
              spec_digest=excluded.spec_digest,
              sheet_path=excluded.sheet_path,
              embed_path=excluded.embed_path,
              lora_path=excluded.lora_path,
              trigger_token=excluded.trigger_token,
              bible_md=excluded.bible_md,
              backstory_md=excluded.backstory_md,
              pinned=excluded.pinned,
              updated_at=excluded.updated_at
            """,
            {**row, "pinned": int(row["pinned"])},
        )
        self._conn.commit()
        return element

    def get_element(self, name: str) -> Element | None:
        row = self._conn.execute(
            "SELECT * FROM elements WHERE project_id=? AND lower(name)=lower(?)",
            (self.project_id, name.strip().lstrip("@")),
        ).fetchone()
        return Element.from_row(row) if row else None

    def list_elements(
        self,
        kind: str | None = None,
        *,
        pinned_only: bool = False,
    ) -> list[Element]:
        sql = "SELECT * FROM elements WHERE project_id=?"
        params: list[Any] = [self.project_id]
        if kind:
            if kind not in ELEMENT_KINDS:
                raise ValueError(f"kind must be one of {ELEMENT_KINDS}")
            sql += " AND kind=?"
            params.append(kind)
        if pinned_only:
            sql += " AND pinned=1"
        sql += " ORDER BY kind, name COLLATE NOCASE"
        return [Element.from_row(r) for r in self._conn.execute(sql, params)]

    def delete_element(self, name: str) -> bool:
        cur = self._conn.execute(
            "DELETE FROM elements WHERE project_id=? AND lower(name)=lower(?)",
            (self.project_id, name.strip().lstrip("@")),
        )
        self._conn.commit()
        return cur.rowcount > 0

    def pin(self, name: str, pinned: bool = True) -> bool:
        cur = self._conn.execute(
            "UPDATE elements SET pinned=?, updated_at=? "
            "WHERE project_id=? AND lower(name)=lower(?)",
            (int(pinned), _now(), self.project_id, name.strip().lstrip("@")),
        )
        self._conn.commit()
        return cur.rowcount > 0

    def library(self, kind: str | None = None) -> dict[str, str]:
        """Handle -> bible text, for ``expand_mentions``."""
        return {
            el.name.lower(): (el.bible_md or "")
            for el in self.list_elements(kind)
            if el.bible_md
        }

    # -- cross-project import ----------------------------------------------

    def import_element(self, source: ProjectStore, name: str) -> Element:
        """Copy an Element in from another project.

        This is the "pull assets from other projects" behaviour. The copy
        records where it came from so provenance is never lost.
        """
        found = source.get_element(name)
        if found is None:
            raise KeyError(f"{name!r} not found in {source.project_id!r}")
        clone = Element(
            name=found.name,
            kind=found.kind,
            spec_json=found.spec_json,
            spec_digest=found.spec_digest,
            sheet_path=found.sheet_path,
            embed_path=found.embed_path,
            lora_path=found.lora_path,
            trigger_token=found.trigger_token,
            bible_md=found.bible_md,
            backstory_md=found.backstory_md,
            pinned=False,
            origin_project=found.project_id,
        )
        return self.save_element(clone)

    # -- shots --------------------------------------------------------------

    def save_shot(self, shot: Shot) -> Shot:
        shot.project_id = shot.project_id or self.project_id
        self._conn.execute(
            """
            INSERT INTO shots (
              id, project_id, seq, prompt, bound_json, output_path,
              drift_score, created_at
            ) VALUES (
              :id, :project_id, :seq, :prompt, :bound_json, :output_path,
              :drift_score, :created_at
            )
            ON CONFLICT(project_id, seq) DO UPDATE SET
              prompt=excluded.prompt,
              bound_json=excluded.bound_json,
              output_path=excluded.output_path,
              drift_score=excluded.drift_score
            """,
            shot.to_dict(),
        )
        self._conn.commit()
        return shot

    def list_shots(self) -> list[Shot]:
        return [
            Shot.from_row(r)
            for r in self._conn.execute(
                "SELECT * FROM shots WHERE project_id=? ORDER BY seq",
                (self.project_id,),
            )
        ]

    def record_drift(self, seq: int, score: float) -> bool:
        cur = self._conn.execute(
            "UPDATE shots SET drift_score=? WHERE project_id=? AND seq=?",
            (float(score), self.project_id, seq),
        )
        self._conn.commit()
        return cur.rowcount > 0

    def drift_curve(self) -> list[tuple[int, float]]:
        """``[(seq, drift_score), ...]`` for shots that have a score."""
        return [
            (int(r["seq"]), float(r["drift_score"]))
            for r in self._conn.execute(
                "SELECT seq, drift_score FROM shots "
                "WHERE project_id=? AND drift_score IS NOT NULL ORDER BY seq",
                (self.project_id,),
            )
        ]

    # -- utility ------------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        counts = {
            r["kind"]: r["n"]
            for r in self._conn.execute(
                "SELECT kind, COUNT(*) AS n FROM elements WHERE project_id=? "
                "GROUP BY kind",
                (self.project_id,),
            )
        }
        return {
            "project_id": self.project_id,
            "path": self.path,
            "elements": counts,
            "total_elements": sum(counts.values()),
            "shots": len(self.list_shots()),
            "scored_shots": len(self.drift_curve()),
        }

    def export(self, dest: str | os.PathLike[str]) -> str:
        """Copy the store to ``dest`` (WAL checkpointed first)."""
        dest_path = Path(dest)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        if self.path == ":memory:":
            target = sqlite3.connect(str(dest_path))
            with target:
                self._conn.backup(target)
            target.close()
            return str(dest_path)
        self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        self._conn.commit()
        shutil.copy2(self.path, dest_path)
        return str(dest_path)

    def __iter__(self) -> Iterator[Element]:
        return iter(self.list_elements())
