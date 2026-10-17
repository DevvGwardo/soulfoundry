#!/usr/bin/env python3
"""Download the ArcFace face models (SCRFD detector + MobileFaceNet embedder).

    python scripts/fetch_models.py

Writes into ``models/buffalo_sc`` by default:

    det_500m.onnx    2.5 MB   SCRFD face detector
    w600k_mbf.onnx    13 MB   ArcFace MobileFaceNet, 512-d embeddings

Total ~16 MB, which is why this is buffalo_sc and not buffalo_l. The larger
packs are 100-330 MB and score a little better on hard angles; for measuring
drift on a character sequence the small pack is the right trade.

Set ``SOULFOUNDRY_MODEL_DIR`` to put them somewhere else, or pass
``--dest``. ``--force`` re-downloads.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_sc.zip"
WANTED = ("det_500m.onnx", "w600k_mbf.onnx")
ROOT = Path(__file__).resolve().parent.parent


def default_dest() -> Path:
    env = os.environ.get("SOULFOUNDRY_MODEL_DIR")
    if env:
        return Path(env)
    return ROOT / "models" / "buffalo_sc"


def _download(url: str, target: Path) -> str:
    """Download to ``target``, returning the sha256 of what arrived."""
    digest = hashlib.sha256()
    with urllib.request.urlopen(url, timeout=120) as response, target.open("wb") as out:
        while True:
            chunk = response.read(1 << 16)
            if not chunk:
                break
            digest.update(chunk)
            out.write(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", default=None, help="target directory")
    parser.add_argument("--force", action="store_true", help="re-download")
    parser.add_argument("--url", default=URL, help="override the archive URL")
    args = parser.parse_args(argv)

    dest = Path(args.dest) if args.dest else default_dest()
    dest.mkdir(parents=True, exist_ok=True)

    present = [name for name in WANTED if (dest / name).is_file()]
    if len(present) == len(WANTED) and not args.force:
        print(f"already present in {dest}:")
        for name in WANTED:
            print(f"  {name}  {(dest / name).stat().st_size / 1e6:.1f} MB")
        print("use --force to re-download")
        return 0

    print(f"downloading {args.url}")
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "buffalo_sc.zip"
        sha = _download(args.url, archive)
        size = archive.stat().st_size
        print(f"  {size / 1e6:.1f} MB  sha256 {sha[:16]}...")

        with zipfile.ZipFile(archive) as zf:
            names = set(zf.namelist())
            missing = [name for name in WANTED if name not in names]
            if missing:
                print(
                    f"error: archive is missing {', '.join(missing)}; "
                    f"it contains {sorted(names)}",
                    file=sys.stderr,
                )
                return 2
            for name in WANTED:
                zf.extract(name, dest)
                print(f"  wrote {dest / name}")

    print(f"\ndone. models in {dest}")
    print("verify with: python -c \"from nodes.core.embed import FaceEmbedder;"
          " print(FaceEmbedder().available())\"")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
