"""Admin API key authentication.

/admin/* and /verify-ownership expose operational internals (traffic stats,
the ability to disable defense entirely, and watermark event lookups) that a
real deployment should not leave open to any caller who can reach /predict.
If ADMIN_API_KEY is set in the environment, these routes require a matching
`X-API-Key` header. If it's unset, the gateway logs a loud warning and allows
all requests through -- convenient for local development, but callers should
never treat an unset key as production-safe.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import Header, HTTPException

from modelvault.utils.logging import get_logger

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

_warned = False


def require_admin_key(x_api_key: str | None = Header(default=None)) -> None:
    global _warned
    admin_key = os.environ.get("ADMIN_API_KEY")

    if not admin_key:
        if not _warned:
            logger.warning(
                "ADMIN_API_KEY is not set -- admin/verify-ownership endpoints are UNAUTHENTICATED. "
                "Set ADMIN_API_KEY in .env before exposing this gateway beyond local development."
            )
            _warned = True
        return

    if x_api_key != admin_key:
        raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key header")
