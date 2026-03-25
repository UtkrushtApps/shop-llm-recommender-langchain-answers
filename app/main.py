"""FastAPI entrypoint for the shop LLM recommender service.

The main HTTP surface is::

    GET /shop/v1/recommendations/{user_id}

which returns a JSON payload like::

    {
      "user_id": "123",
      "recommendation": "..."
    }

This module wires together the FastAPI app, the CSV-backed user
repository, and the LangChain-based recommendation service. Tests
override the dependency that constructs :class:`RecommendationService`
so no external LLM calls are ever made.
"""
from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, status

from .exceptions import (
    InvalidUserIdError,
    RecommendationError,
    ShopRecommenderError,
    UserDataError,
    UserNotFoundError,
)
from .models import ErrorResponse, RecommendationResponse
from .recommender import EchoLLM, RecommendationService, UserRepository


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _default_csv_path() -> Path:
    """Resolve the default user CSV path.

    The path can be overridden using the ``USER_CSV_PATH`` environment
    variable. If not set, it defaults to ``data/users.csv`` relative to
    the project root.
    """

    env_path = os.getenv("USER_CSV_PATH")
    if env_path:
        return Path(env_path)

    # Assume a typical project layout where this file lives in
    # ``app/main.py`` and user data is in ``data/users.csv``.
    return Path(__file__).resolve().parent.parent / "data" / "users.csv"


@lru_cache(maxsize=1)
def get_recommendation_service() -> RecommendationService:
    """Create and cache a :class:`RecommendationService` instance.

    This function is used as a FastAPI dependency and is overridden by
    tests to provide mocked services.
    """

    csv_path = _default_csv_path()
    logger.info("Initializing RecommendationService with CSV at %s", csv_path)
    user_repo = UserRepository(csv_path)

    # Use EchoLLM as the default in-process LLM; production deployments
    # are expected to plug in a real LLM.
    llm = EchoLLM()

    return RecommendationService(user_repository=user_repo, llm=llm)


app = FastAPI(title="Shop LLM Recommender", version="1.0.0")


@app.get(
    "/shop/v1/recommendations/{user_id}",
    response_model=RecommendationResponse,
    responses={
        400: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
def get_recommendation(
    user_id: str,
    service: RecommendationService = Depends(get_recommendation_service),
) -> RecommendationResponse:
    """Return a personalized recommendation for ``user_id``.

    Error conditions are mapped to explicit HTTP status codes and a
    consistent JSON error format.
    """

    try:
        recommendation_text = service.recommend(user_id=user_id)
    except InvalidUserIdError as exc:
        # 400 – the caller sent an invalid identifier.
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error_code": "INVALID_USER_ID",
                "detail": str(exc),
            },
        ) from exc
    except UserNotFoundError as exc:
        # 404 – the identifier is syntactically valid but unknown.
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error_code": "USER_NOT_FOUND",
                "detail": str(exc),
            },
        ) from exc
    except UserDataError as exc:
        # 500 – internal configuration/data issue.
        logger.exception("User data error while serving recommendation", exc_info=exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error_code": "USER_DATA_ERROR",
                "detail": str(exc),
            },
        ) from exc
    except RecommendationError as exc:
        # 500 – LLM error.
        logger.exception("Recommendation generation failed", exc_info=exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error_code": "RECOMMENDATION_ERROR",
                "detail": str(exc),
            },
        ) from exc
    except ShopRecommenderError as exc:  # pragma: no cover - defensive
        # A generic internal error that we did not classify explicitly.
        logger.exception("Unhandled shop recommender error", exc_info=exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error_code": "INTERNAL_ERROR",
                "detail": str(exc),
            },
        ) from exc

    # When FastAPI receives a dict in HTTPException.detail, it bypasses
    # the response_model. For successful responses we just return the
    # Pydantic model and FastAPI handles serialization.
    return RecommendationResponse(user_id=user_id, recommendation=recommendation_text)
