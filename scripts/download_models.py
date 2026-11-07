#!/usr/bin/env python3
"""Download every model the app uses (app.ml.models.REGISTRY), pinned and checksummed.

scripts/run.sh runs this on every start: with every model present it does nothing and exits 0.
Missing files are streamed to '.part' files, checked (size and sha256) and renamed into place. A model
that fails does not stop the others; the exit status is 1 and stderr names each failed model.

Usage: download_models.py [--only KEY ...] [--verify] [--quiet]
  --only KEY   download only these models (faces, siglip, nsfw, persons, ocr)
  --verify     also hash every present file and report mismatches (slow: about 1.4 GB of reads)
  --quiet      print only failures

Models go to MODELS_DIR (default <repo>/models). Run it with the backend environment's python.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

log = logging.getLogger("download_models")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download the app's models (no-op when present).")
    parser.add_argument("--only", nargs="+", metavar="KEY", help="model keys to download")
    parser.add_argument("--verify", action="store_true", help="hash present files and report mismatches")
    parser.add_argument("--quiet", action="store_true", help="print only failures")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    from app.ml import models

    def say(msg: str) -> None:
        if not args.quiet:
            sys.stdout.write(msg + "\n")
            sys.stdout.flush()

    keys = args.only or list(models.REGISTRY)
    unknown = [k for k in keys if k not in models.REGISTRY]
    if unknown:
        sys.stderr.write(f"unknown model key(s): {', '.join(unknown)}; known: {', '.join(models.REGISTRY)}\n")
        return 2

    status = 0
    missing = [k for k in keys if not models.present(k)]
    if not missing:
        say(f"models present: {', '.join(keys)}")
    else:
        say(f"downloading models: {', '.join(missing)}")
        try:
            models.download(missing, progress=say)
        except models.DownloadError as e:
            for key, reason in e.failed.items():
                sys.stderr.write(f"model {key} failed: {reason}\n")
            status = 1
        except Exception as e:  # noqa: BLE001 - reported, exit 1
            sys.stderr.write(f"model download failed: {e}\n")
            status = 1
        else:
            say("models ready")

    if args.verify:
        for key in keys:
            if not models.present(key):
                continue
            bad = models.verify(key)
            if bad:
                sys.stderr.write(f"model {key}: checksum mismatch in {', '.join(bad)}\n")
                status = 1
            else:
                say(f"model {key}: checksums match")
    return status


if __name__ == "__main__":
    sys.exit(main())
