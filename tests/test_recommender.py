"""Unit tests for the LangChain-based recommendation component.

These tests avoid real external LLM calls by providing fake in-process
LLM implementations.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import pytest
from langchain.llms.base import LLM

from app.exceptions import (
    InvalidUserIdError,
    RecommendationError,
    UserNotFoundError,
)
from app.recommender import RecommendationService, UserRepository


class FakeLLM(LLM):
    """Deterministic LLM implementation for tests.

    It captures the last prompt it was called with and returns a
    predictable string so tests can assert on model behaviour without
    external dependencies.
    """

    def __init__(self) -> None:
        super().__init__()
        self.last_prompt: Optional[str] = None

    def _call(self, prompt: str, stop: Optional[list[str]] = None) -> str:  # type: ignore[override]
        self.last_prompt = prompt
        # Produce a short, deterministic response to simplify assertions.
        return f"FAKE RECOMMENDATION FOR PROMPT HASH {hash(prompt) % 10_000}"

    @property
    def _identifying_params(self) -> Dict[str, object]:  # type: ignore[override]
        return {}

    @property
    def _llm_type(self) -> str:  # type: ignore[override]
        return "fake_llm_for_tests"


class ErrorLLM(LLM):
    """LLM that intentionally fails, used to exercise error handling."""

    def _call(self, prompt: str, stop: Optional[list[str]] = None) -> str:  # type: ignore[override]
        raise RuntimeError("Simulated LLM failure")

    @property
    def _identifying_params(self) -> Dict[str, object]:  # type: ignore[override]
        return {}

    @property
    def _llm_type(self) -> str:  # type: ignore[override]
        return "error_llm_for_tests"


@pytest.fixture()
def sample_csv(tmp_path: Path) -> Path:
    """Create a small temporary CSV file with user data for tests."""

    csv_path = tmp_path / "users.csv"
    csv_path.write_text(
        "user_id,name,skills,level\n"
        "user-1,Alice,Python|SQL,intermediate\n"
        "user-2,Bob,Java|Kotlin,advanced\n",
        encoding="utf-8",
    )
    return csv_path


@pytest.fixture()
def user_repo(sample_csv: Path) -> UserRepository:
    return UserRepository(sample_csv)


@pytest.fixture()
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture()
def recommendation_service(user_repo: UserRepository, fake_llm: FakeLLM) -> RecommendationService:
    return RecommendationService(user_repository=user_repo, llm=fake_llm)


def test_recommendation_success(recommendation_service: RecommendationService, fake_llm: FakeLLM) -> None:
    """A valid user_id yields a non-empty recommendation and uses user data in the prompt."""

    result = recommendation_service.recommend("user-1")

    assert isinstance(result, str)
    assert result.startswith("FAKE RECOMMENDATION")

    # Ensure the LLM saw a JSON representation of the user record that
    # includes the correct user_id and name.
    assert fake_llm.last_prompt is not None
    assert "user_profile" in fake_llm.last_prompt  # part of the template

    # The prompt is built from JSON; double-check it contains key fields.
    # We don't parse the whole template here—just look for fragments.
    assert "\"user_id\": \"user-1\"" in fake_llm.last_prompt
    assert "\"name\": \"Alice\"" in fake_llm.last_prompt


def test_invalid_user_id_raises() -> None:
    """Empty or whitespace-only user IDs should raise InvalidUserIdError."""

    # Use an in-memory CSV for this small test.
    tmp = Path("test_users.csv")
    try:
        tmp.write_text("user_id,name\nuser-1,Alice\n", encoding="utf-8")
        repo = UserRepository(tmp)
        service = RecommendationService(user_repository=repo, llm=FakeLLM())

        with pytest.raises(InvalidUserIdError):
            service.recommend("")

        with pytest.raises(InvalidUserIdError):
            service.recommend("   ")
    finally:
        if tmp.exists():
            tmp.unlink()


def test_unknown_user_id_raises(user_repo: UserRepository, fake_llm: FakeLLM) -> None:
    """Unknown but syntactically valid user IDs raise UserNotFoundError."""

    service = RecommendationService(user_repository=user_repo, llm=fake_llm)
    with pytest.raises(UserNotFoundError):
        service.recommend("non-existent-user")


def test_llm_error_is_wrapped_as_recommendation_error(sample_csv: Path) -> None:
    """Low-level LLM failures are converted into RecommendationError."""

    repo = UserRepository(sample_csv)
    service = RecommendationService(user_repository=repo, llm=ErrorLLM())

    with pytest.raises(RecommendationError) as exc_info:
        service.recommend("user-1")

    # Provide a reasonably clear error message.
    assert "language model error" in str(exc_info.value).lower()
