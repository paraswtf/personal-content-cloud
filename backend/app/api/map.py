"""Map and places (docs/ARCHITECTURE.md section 25): /api/map."""
from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.deps import require_user

router = APIRouter(dependencies=[Depends(require_user)])
