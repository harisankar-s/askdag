"""The API key check on /api/* (ADR 0022). Off while API_KEY is unset."""

import hmac
from typing import Annotated

from fastapi import Depends, HTTPException, Security
from fastapi.security import APIKeyHeader

from asklineage.config import Settings, get_settings

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(
    settings: Annotated[Settings, Depends(get_settings)],
    given: Annotated[str | None, Security(API_KEY_HEADER)],
) -> None:
    if settings.api_key is None:
        return
    expected = settings.api_key.get_secret_value().encode()
    if given is None or not hmac.compare_digest(given.encode(), expected):
        raise HTTPException(
            401,
            "Missing or wrong API key; send it in the X-API-Key header.",
            headers={"WWW-Authenticate": "APIKey"},
        )
