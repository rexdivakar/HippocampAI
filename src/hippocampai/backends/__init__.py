"""Backend abstraction layer for local and remote modes."""

from hippocampai.backends.base import BaseBackend
from hippocampai.backends.local import LocalBackend
from hippocampai.backends.remote import RemoteBackend

__all__ = ["BaseBackend", "LocalBackend", "RemoteBackend"]
