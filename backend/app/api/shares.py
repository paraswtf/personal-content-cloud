"""Share links, the owner side (docs/ARCHITECTURE.md section 26): /api/shares."""
from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.deps import require_user

router = APIRouter(dependencies=[Depends(require_user)])
