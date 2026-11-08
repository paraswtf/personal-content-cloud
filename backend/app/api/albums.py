"""Albums (docs/ARCHITECTURE.md section 24): /api/albums."""
from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.deps import require_user

router = APIRouter(dependencies=[Depends(require_user)])
