"""Pydantic models used by the FastAPI layer."""
from __future__ import annotations

from pydantic import BaseModel, Field


class RecommendationResponse(BaseModel):
    """Successful recommendation payload returned by the API."""

    user_id: str = Field(..., description="Identifier of the requested user")
    recommendation: str = Field(..., description="Generated recommendation text")


class ErrorResponse(BaseModel):
    """Error payload returned by the API in case of failures."""

    error_code: str = Field(..., description="Stable machine-readable error code")
    detail: str = Field(..., description="Human-readable error description")
