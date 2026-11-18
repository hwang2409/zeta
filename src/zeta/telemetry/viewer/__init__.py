"""standalone otlp receiver and viewer."""

from .app import run_viewer
from .store import ViewerStore

__all__ = ["ViewerStore", "run_viewer"]
