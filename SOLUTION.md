# Solution Steps

1. Create the core package structure: an `app` package for the service code and a `tests` package for automated tests. Add an empty `__init__.py` in `app/` so it is treated as a module.

2. Implement domain-specific exceptions in `app/exceptions.py`:
- Define a `ShopRecommenderError` base class inheriting from `Exception`.
- Derive `UserDataError`, `InvalidUserIdError`, `UserNotFoundError`, and `RecommendationError` from this base class.
- Document each exception with when it should be raised (CSV issues, bad input, unknown user, LLM failures).

3. Implement Pydantic API models in `app/models.py`:
- Create `RecommendationResponse` with `user_id: str` and `recommendation: str` fields and brief descriptions.
- Create `ErrorResponse` with `error_code: str` and `detail: str` fields to standardize error payloads.

4. Implement the CSV-backed user repository and LangChain-based service in `app/recommender.py`:
- Import `csv`, `json`, `logging`, `Path`, `LLM`, `LLMChain`, and `PromptTemplate`.
- Implement `UserRepository`:
  - Accept a `csv_path` in the constructor, store it, and call a private `_load_users` method.
  - In `_load_users`, verify the file exists; if not, raise `UserDataError`.
  - Open the CSV with `csv.DictReader` and ensure there is a header row.
  - Verify that a `user_id` column is present; otherwise raise `UserDataError`.
  - Iterate rows, normalize `user_id` (strip whitespace); skip rows with missing IDs, logging a warning.
  - Store each row as a `dict[str, str]` in an internal mapping `self._users[user_id]`, logging and overwriting on duplicates.
  - Handle `OSError` and `csv.Error` by raising `UserDataError` with descriptive messages.
  - Implement `get_user(user_id)` to validate non-empty IDs, then return the corresponding row or raise `UserNotFoundError`.
  - Implement `all_users()` to return an iterable of all stored user dicts.
- Implement a small in-process `EchoLLM` class that subclasses `langchain.llms.base.LLM`:
  - Implement `_call(self, prompt, stop=None)` to return a stub recommendation that includes a truncated prompt snippet.
  - Implement `_llm_type` and `_identifying_params` properties.
- Implement `RecommendationService`:
  - Accept a `UserRepository`, an `LLM` instance, and an optional logger in the constructor.
  - Create a `PromptTemplate` that takes `user_profile` and instructs the model to write 2–3 sentence assessment recommendations.
  - Construct an `LLMChain` with the provided `llm` and this prompt template.
  - Implement `recommend(user_id)` to:
    - Use `user_repository.get_user(user_id)` to retrieve the record (propagate its `InvalidUserIdError`/`UserNotFoundError`).
    - Convert the record to a stable JSON string `user_json = json.dumps(..., sort_keys=True)`.
    - Log that a recommendation is being generated.
    - Call `self._chain.run(user_profile=user_json)` inside a `try` block.
    - On unexpected exceptions, log them and raise `RecommendationError` with a clear message.
    - Validate that the output is a non-empty string; if not, raise `RecommendationError`.
    - Return the stripped recommendation text.

5. Implement the FastAPI app and dependency wiring in `app/main.py`:
- Configure module-level logging via `logging.basicConfig` and get a logger.
- Implement `_default_csv_path()` to resolve the user CSV path using the `USER_CSV_PATH` environment variable or falling back to `data/users.csv` at the project root.
- Implement a cached dependency function `get_recommendation_service()` using `@lru_cache`:
  - Use `_default_csv_path()` to construct a `UserRepository`.
  - Instantiate the default in-process `EchoLLM`.
  - Return a `RecommendationService` wired to the repository and LLM.
- Create a `FastAPI` app instance.
- Add a `GET /shop/v1/recommendations/{user_id}` route:
  - Depend on `get_recommendation_service`.
  - Call `service.recommend(user_id)` in a `try` block.
  - Map `InvalidUserIdError` to HTTP 400 with `error_code="INVALID_USER_ID"`.
  - Map `UserNotFoundError` to HTTP 404 with `error_code="USER_NOT_FOUND"`.
  - Map `UserDataError` to HTTP 500 with `error_code="USER_DATA_ERROR"`.
  - Map `RecommendationError` to HTTP 500 with `error_code="RECOMMENDATION_ERROR"`.
  - For each error, raise `HTTPException` with `detail` set to a dict containing `error_code` and `detail` message.
  - On success, return a `RecommendationResponse` instance; FastAPI will serialize it to JSON.

6. Expose the FastAPI app from the package root in `app/__init__.py` so ASGI servers can import `app.app` directly.

7. Write unit tests for the recommendation component in `tests/test_recommender.py`:
- Import `pytest`, `Path`, and `LLM`.
- Implement `FakeLLM` subclassing `LLM`:
  - Track the last prompt in `self.last_prompt`.
  - Return a deterministic string like `"FAKE RECOMMENDATION FOR PROMPT HASH ..."` from `_call`.
- Implement `ErrorLLM` subclassing `LLM` whose `_call` method always raises `RuntimeError`.
- Add a `sample_csv` fixture that creates a temp CSV file with a `user_id` column and a couple of rows.
- Add `user_repo`, `fake_llm`, and `recommendation_service` fixtures to construct a `UserRepository` and `RecommendationService` using `FakeLLM`.
- Test `recommendation_success`:
  - Call `recommendation_service.recommend("user-1")`.
  - Assert the returned string starts with the fake prefix.
  - Assert that `fake_llm.last_prompt` is not `None` and contains both the `user_id` and `name` JSON keys from the row.
- Test `invalid_user_id_raises`:
  - Create a minimal CSV, build repository and service, and assert that passing `""` or whitespace user IDs raises `InvalidUserIdError`.
- Test `unknown_user_id_raises`:
  - With the sample repository, call `recommend("non-existent-user")` and assert `UserNotFoundError` is raised.
- Test `llm_error_is_wrapped_as_recommendation_error`:
  - Build a service with `ErrorLLM` and assert that `RecommendationError` is raised with a message mentioning a language model error.

8. Write integration-style API tests in `tests/test_api.py` that exercise the FastAPI layer with dependency overrides:
- Define an `ApiFakeLLM` subclass of `LLM` that always returns a short, fixed string like `"API FAKE RECOMMENDATION"`.
- Create an `api_client` fixture:
  - Use `tmp_path` to create a temporary CSV file with a single user (e.g., `api-user-1`).
  - Build a `UserRepository` for this file and a `RecommendationService` using `ApiFakeLLM`.
  - Override `get_recommendation_service` via `app.dependency_overrides` to return this pre-built service.
  - Instantiate and yield a `TestClient(app)`.
  - After yielding, clear `app.dependency_overrides`.
- Test `get_recommendation_success`:
  - Call `GET /shop/v1/recommendations/api-user-1`.
  - Assert HTTP 200, `user_id` in the response matches, and `recommendation` equals the fixed fake text.
- Test `get_recommendation_unknown_user_returns_404`:
  - Call the endpoint with an unknown ID and assert HTTP 404 and that the JSON body’s `detail.error_code` is `"USER_NOT_FOUND"`.
- Test `get_recommendation_invalid_user_id_returns_400`:
  - Call the endpoint with a URL-encoded whitespace ID (e.g., `%20%20%20`).
  - Assert HTTP 400 and that `detail.error_code` is `"INVALID_USER_ID"`.

9. Run the test suite with `pytest` to ensure all unit and API tests pass and that no real external LLM calls are made (they should all use the fake LLM implementations). Optionally, start the FastAPI app (e.g., via `uvicorn app.main:app`) and manually call the `/shop/v1/recommendations/{user_id}` endpoint to observe the stubbed recommendations and error handling.

