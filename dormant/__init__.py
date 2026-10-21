"""Dormant: an open-weight model that reads what macOS already recorded about your
files and tells you which of them you are never going to open again.

Local only. No telemetry leaves the machine, and the repository contains none.
"""

from .config import APP_NAME, PATHS, VERSION

__all__ = ["APP_NAME", "PATHS", "VERSION"]
