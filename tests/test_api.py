"""Integration-style tests for the FastAPI endpoint.

These tests override the dependency that constructs the
:class:`RecommendationService` so that no real LLM nor real CSV files
are required.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import pytest
from fastapi.testclient import TestClient
from langchain.llms.base import LLM

from app.main import app, get_recommendation_service
from app.recommender import RecommendationService, UserRepository


class ApiFakeLLM(LLM):
    """Simple LLM for exercising the HTTP layer in tests."""

    def _call(self, prompt: str, stop: Optional[list[str]] = None) -> str:  # type: ignore[override]
        # Return a very short, recognizable payload.
        return "API FAKE RECOMMENDATION"

    @property
    def _identifying_params(self) -> Dict[str, object]:  # type: ignore[override]
        return {}

    @property
    def _llm_type(self) -> str:  # type: ignore[override]
        return "api_fake_llm_for_tests"


@pytest.fixture()
def api_client(tmp_path: Path) -> TestClient:
    """Return a TestClient with a mocked RecommendationService dependency."""

    # Prepare a small CSV file for the API tests.
    csv_path = tmp_path / "users.csv"
    csv_path.write_text(
        "user_id,name,skills\n"
        "api-user-1,Charlie,Python|FastAPI\n",
        encoding="utf-8",
    )

    user_repo = UserRepository(csv_path)
    llm = ApiFakeLLM()
    service = RecommendationService(user_repository=user_repo, llm=llm)

    def override_service() -> RecommendationService:
        return service

    app.dependency_overrides[get_recommendation_service] = override_service

    client = TestClient(app)

    yield client

    # Cleanup dependency overrides after tests to avoid cross-test
    # contamination if this module is imported elsewhere.
    app.dependency_overrides.clear()


def test_get_recommendation_success(api_client: TestClient) -> None:
    """The happy path returns a structured JSON recommendation payload."""

    resp = api_client.get("/shop/v1/recommendations/api-user-1")

    assert resp.status_code == 200
    data = resp.json()

    assert data["user_id"] == "api-user-1"
    assert data["recommendation"] == "API FAKE RECOMMENDATION"


def test_get_recommendation_unknown_user_returns_404(api_client: TestClient) -> None:
    """Unknown users should yield a 404 with a stable error code."""

    resp = api_client.get("/shop/v1/recommendations/does-not-exist")

    assert resp.status_code == 404
    data = resp.json()

    # The detail itself is a dict because we set it so in HTTPException.
    assert data["detail"]["error_code"] == "USER_NOT_FOUND"
    assert "does-not-exist" in data["detail"]["detail"]


def test_get_recommendation_invalid_user_id_returns_400(api_client: TestClient) -> None:
    """Syntactically invalid user IDs should yield a 400 error."""

    # Spaces will be URL-encoded by the HTTP client.
    resp = api_client.get("/shop/v1/recommendations/%20%20%20")

    assert resp.status_code == 400
    data = resp.json()

    assert data["detail"]["error_code"] == "INVALID_USER_ID"
