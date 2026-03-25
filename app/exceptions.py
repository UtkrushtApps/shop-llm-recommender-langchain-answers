"""Domain-specific exceptions for the shop LLM recommender service."""


class ShopRecommenderError(Exception):
    """Base exception for all recommender-related errors."""


class UserDataError(ShopRecommenderError):
    """Raised when the user data CSV is missing or malformed."""


class InvalidUserIdError(ShopRecommenderError):
    """Raised when a provided user_id is syntactically invalid."""


class UserNotFoundError(ShopRecommenderError):
    """Raised when no user record exists for the given user_id."""


class RecommendationError(ShopRecommenderError):
    """Raised when the LLM-based recommendation cannot be generated."""
