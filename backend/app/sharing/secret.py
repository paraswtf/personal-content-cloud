"""The app secret: DATA_DIR/secret.key, 32 random bytes with mode 0600, made on first use
(docs/ARCHITECTURE.md section 3). Signs the share unlock cookies (section 26)."""
from __future__ import annotations

import os
import secrets
import threading
from pathlib import Path

from app.config import settings

FILE = "secret.key"
_lock = threading.Lock()


def secret_key() -> bytes:
    path = Path(settings.data_dir) / FILE
    try:
        data = path.read_bytes()
        if len(data) >= 32:
            return data
    except OSError:
        pass
    with _lock:
        try:
            data = path.read_bytes()
            if len(data) >= 32:
                return data
        except OSError:
            pass
        path.parent.mkdir(parents=True, exist_ok=True)
        data = secrets.token_bytes(32)
        tmp = path.with_name(f".{FILE}.{secrets.token_hex(4)}.tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            os.write(fd, data)
        finally:
            os.close(fd)
        os.replace(tmp, path)
        return data
