"""Timeline buckets (docs/ARCHITECTURE.md section 23): GET /api/timeline/buckets and /bucket."""
from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.deps import require_user

router = APIRouter(dependencies=[Depends(require_user)])
