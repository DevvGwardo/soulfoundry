# Project store schema

One SQLite file per project. `nodes/core/store.py` creates this schema on
open; this file is the canonical copy for readers.

```sql
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS meta (
  key   TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS elements (
  id             TEXT PRIMARY KEY,   -- uuid
  project_id     TEXT NOT NULL,
  name           TEXT NOT NULL,      -- the @mention handle
  kind           TEXT NOT NULL CHECK (kind IN ('character','location','prop')),
  spec_json      TEXT,               -- the 8-category params, for characters
  spec_digest    TEXT,               -- sha256 of canonical spec json, 16 hex
  sheet_path     TEXT,               -- 3-panel turnaround
  embed_path     TEXT,               -- IP-Adapter / PuLID / face embedding
  lora_path      TEXT,               -- optional trained LoRA
  trigger_token  TEXT,               -- LoRA trigger word
  bible_md       TEXT,               -- verbatim description block
  backstory_md   TEXT,
  pinned         INTEGER NOT NULL DEFAULT 0,
  origin_project TEXT,               -- set when imported from another project
  created_at     TEXT NOT NULL,
  updated_at     TEXT NOT NULL,
  UNIQUE(project_id, name)
);

CREATE TABLE IF NOT EXISTS shots (
  id          TEXT PRIMARY KEY,
  project_id  TEXT NOT NULL,
  seq         INTEGER NOT NULL,
  prompt      TEXT,
  bound_json  TEXT,                  -- element names bound to this shot
  output_path TEXT,
  drift_score REAL,                  -- from ConsistencyAudit
  created_at  TEXT NOT NULL,
  UNIQUE(project_id, seq)
);

CREATE INDEX IF NOT EXISTS idx_elements_project ON elements(project_id);
CREATE INDEX IF NOT EXISTS idx_elements_kind    ON elements(project_id, kind);
CREATE INDEX IF NOT EXISTS idx_shots_project    ON shots(project_id, seq);
```

## Notes

**Two tables is the whole data model.** Deliberate. A cast and a shot list is
all a sequence needs, and it is the part that makes this a product rather
than a demo.

**`kind` is constrained in the database.** Three kinds, matching the real
Element taxonomy: character, location, prop.

**Upsert preserves identity.** Saving an Element that already exists keeps
its original `id`, so shots already bound to it keep pointing at the same
row.

**Cross-project import is a copy, not a reference.** The imported row gets a
fresh `id` and records `origin_project`, so provenance survives and editing
one project's copy cannot silently change another's.

**WAL is on.** `export()` checkpoints the WAL before copying, so a backup is
never taken mid-write.

## Migrations

`meta.schema_version` is `1`. Migrations are additive `ALTER TABLE` steps
applied on open when the stored version is behind. There are none yet.
