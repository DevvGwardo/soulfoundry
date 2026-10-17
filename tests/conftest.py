"""Make the repo root importable so tests can `import nodes...`.

Also guards against the one environment trap that produces a confusing
failure: pytest's import machinery walks up from ``tests/``, finds the root
``__init__.py`` (which ComfyUI requires), and under some rootdir layouts tries
to import the repo root as a top-level module. If the checkout directory is
not a valid Python identifier (``verify-clone``, ``soul-foundry``, anything
with a hyphen), that import fails with "attempted relative import with no
known parent package" and every test errors at collection.

The fix is a valid directory name, which is also what ComfyUI requires, since
it imports the pack by directory name. This asserts it up front so the error
says what to do.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if not ROOT.name.isidentifier():
    raise RuntimeError(
        f"the checkout directory is named {ROOT.name!r}, which is not a valid "
        f"Python identifier. ComfyUI imports a custom node pack by directory "
        f"name, so the pack directory must be a valid identifier anyway "
        f"(clone as `soulfoundry`). Under pytest this name also breaks "
        f"collection with a misleading relative-import error. Rename the "
        f"directory to `soulfoundry` and re-run."
    )
