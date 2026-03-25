"""Application package for the shop LLM recommender service.

This module exposes the FastAPI app instance so that ASGI servers
(Uvicorn, Gunicorn, etc.) can run `app:app`.
"""
from .main import app

__all__ = ["app"]
