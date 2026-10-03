"""Streaming / API-native data access (real sources only)."""
from .remote_client import RemoteDataClient, StreamConfig

__all__ = ["RemoteDataClient", "StreamConfig"]
