"""Share links, the public side (docs/ARCHITECTURE.md section 26): /api/s/{token} and /files/s/{token}/...
No session: every route checks the share token itself and never answers 401."""
from __future__ import annotations

from fastapi import APIRouter

router = APIRouter()
