"""LangChain-based recommendation component.

This module contains two main pieces:

* :class:`UserRepository` – loads and caches user profiles from a CSV file.
* :class:`RecommendationService` – uses LangChain to generate
  personalized recommendation text for a given ``user_id``.

The LangChain integration is deliberately simple and synchronous. It uses
``PromptTemplate`` and ``LLMChain`` to drive an ``LLM`` instance. Tests
provide fake LLM implementations so no real external calls are made.
"""
from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Dict, Iterable, Optional, Union

from langchain.chains import LLMChain
from langchain.llms.base import LLM
from langchain.prompts import PromptTemplate

from .exceptions import (
    InvalidUserIdError,
    RecommendationError,
    ShopRecommenderError,
    UserDataError,
    UserNotFoundError,
)

logger = logging.getLogger(__name__)


class UserRepository:
    """Repository that loads user data from a CSV file.

    The CSV is expected to contain at least a ``user_id`` column. All
    remaining columns are treated as arbitrary user attributes and kept
    as strings.
    """

    def __init__(self, csv_path: Union[str, Path], encoding: str = "utf-8") -> None:
        self._csv_path = Path(csv_path)
        self._encoding = encoding
        self._users: Dict[str, Dict[str, str]] = {}
        self._load_users()

    @property
    def path(self) -> Path:
        """Return the underlying CSV path (useful for observability/tests)."""

        return self._csv_path

    def _load_users(self) -> None:
        """Load all users from the CSV file into memory.

        Raises
        ------
        UserDataError
            If the CSV file is missing, unreadable, or malformed.
        """

        if not self._csv_path.is_file():
            raise UserDataError(f"User data CSV file not found at '{self._csv_path}'.")

        try:
            with self._csv_path.open("r", encoding=self._encoding, newline="") as f:
                reader = csv.DictReader(f)

                if not reader.fieldnames:
                    raise UserDataError(
                        f"User data CSV at '{self._csv_path}' has no header row."
                    )

                if "user_id" not in reader.fieldnames:
                    raise UserDataError(
                        "User data CSV must contain a 'user_id' column; "
                        f"found columns: {', '.join(reader.fieldnames)}."
                    )

                count = 0
                for row in reader:
                    user_id = (row.get("user_id") or "").strip()
                    if not user_id:
                        logger.warning(
                            "Skipping row without user_id in %s: %r", self._csv_path, row
                        )
                        continue

                    if user_id in self._users:
                        logger.warning(
                            "Duplicate user_id '%s' encountered in %s; overwriting previous entry.",
                            user_id,
                            self._csv_path,
                        )

                    # Normalize all values to strings (DictReader already returns str-or-None).
                    normalized: Dict[str, str] = {
                        key: (value if value is not None else "") for key, value in row.items()
                    }
                    self._users[user_id] = normalized
                    count += 1

                logger.info(
                    "Loaded %d user records from %s", count, self._csv_path.resolve()
                )

        except OSError as exc:  # I/O-related problems
            raise UserDataError(
                f"Failed to read user data CSV at '{self._csv_path}': {exc}"
            ) from exc
        except csv.Error as exc:
            raise UserDataError(
                f"Failed to parse user data CSV at '{self._csv_path}': {exc}"
            ) from exc

    def get_user(self, user_id: str) -> Dict[str, str]:
        """Return the user row for ``user_id``.

        Raises
        ------
        InvalidUserIdError
            If ``user_id`` is empty or only whitespace.
        UserNotFoundError
            If the user is not present in the loaded CSV data.
        """

        if user_id is None or not str(user_id).strip():
            raise InvalidUserIdError("user_id must be a non-empty string.")

        key = str(user_id).strip()
        try:
            return self._users[key]
        except KeyError as exc:
            raise UserNotFoundError(f"No user found for user_id='{key}'.") from exc

    def all_users(self) -> Iterable[Dict[str, str]]:
        """Return an iterable of all user records.

        Primarily intended for debugging or administrative use.
        """

        return self._users.values()


class EchoLLM(LLM):
    """A very small LangChain LLM used as a safe default.

    This implementation does not make any external API calls. It simply
    echoes a short, generic recommendation snippet based on the prompt.

    It is primarily meant as a placeholder for local development and for
    wiring the application without leaking real provider keys. In
    production, replace this with a proper LLM implementation (e.g.
    OpenAI, Anthropic, etc.).
    """

    def _call(self, prompt: str, stop: Optional[list[str]] = None) -> str:  # type: ignore[override]
        # Provide a bounded echo for safety/observability.
        snippet = prompt.strip().replace("\n", " ")[:200]
        return (
            "This is a stubbed recommendation generated by EchoLLM. "
            f"Prompt summary: {snippet}"
        )

    @property
    def _llm_type(self) -> str:  # type: ignore[override]
        return "echo_llm"

    @property
    def _identifying_params(self) -> Dict[str, object]:  # type: ignore[override]
        # No special parameters; this is a pure in-process implementation.
        return {}


class RecommendationService:
    """Service that generates personalized recommendations via LangChain."""

    def __init__(
        self,
        user_repository: UserRepository,
        llm: LLM,
        logger_: Optional[logging.Logger] = None,
    ) -> None:
        self._user_repository = user_repository
        self._llm = llm
        self._logger = logger_ or logging.getLogger(f"{__name__}.RecommendationService")

        # Define a simple prompt template that expects a JSON user profile.
        template = (
            "You are an assistant helping users pick the best skill assessments.\n"\
            "Given this user profile as JSON: {user_profile}\n"\
            "Draft 2-3 concise sentences recommending which skill assessments "
            "they should take next and why. Focus on clarity and usefulness."
        )

        prompt = PromptTemplate(
            input_variables=["user_profile"],
            template=template,
        )

        # A basic synchronous LLMChain – tests will inject fake LLM
        # implementations so no real external traffic occurs.
        self._chain = LLMChain(llm=self._llm, prompt=prompt)

    def recommend(self, user_id: str) -> str:
        """Return recommendation text for the given ``user_id``.

        This function is deterministic relative to the underlying LLM –
        we simply pass the raw user row (as JSON) into the prompt and
        return the resulting text, validating non-emptiness.

        Raises
        ------
        InvalidUserIdError
            If ``user_id`` is syntactically invalid.
        UserNotFoundError
            If the user is unknown.
        RecommendationError
            If the LLM fails or returns an empty response.
        """

        # These exceptions propagate directly; the API layer will map
        # them to appropriate HTTP statuses.
        user_record = self._user_repository.get_user(user_id)

        # Ensure stable order for reproducible prompts (useful for
        # logging, debugging, and tests).
        user_json = json.dumps(user_record, sort_keys=True, ensure_ascii=False)

        self._logger.info(
            "Generating recommendation for user_id='%s' using %s", user_id, type(self._llm).__name__
        )

        try:
            # LLMChain.run returns a plain string in the common case.
            raw_output = self._chain.run(user_profile=user_json)
        except (ShopRecommenderError, InvalidUserIdError, UserNotFoundError):
            # Domain errors should not be wrapped again.
            raise
        except Exception as exc:  # pragma: no cover - defensive broad catch
            self._logger.exception(
                "LLM chain execution failed for user_id='%s'", user_id, exc_info=exc
            )
            raise RecommendationError(
                "Failed to generate recommendation due to a language model error."
            ) from exc

        if not isinstance(raw_output, str):  # highly defensive
            self._logger.error(
                "Unexpected non-string output from LLM chain for user_id='%s': %r",
                user_id,
                raw_output,
            )
            raise RecommendationError(
                "Language model returned an unexpected response type."
            )

        recommendation = raw_output.strip()
        if not recommendation:
            self._logger.error(
                "Empty recommendation returned by LLM for user_id='%s'", user_id
            )
            raise RecommendationError(
                "Language model returned an empty recommendation."
            )

        return recommendation
