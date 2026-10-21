"""Dormant configuration.

Everything Dormant learns is written under DATA_DIR, which holds real file paths,
real open timestamps and real sizes. That directory is gitignored by design: the
repository ships the model, never the telemetry it was fitted on.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

APP_NAME = "Dormant"
VERSION = "0.1.0"

REPO_ROOT = Path(__file__).resolve().parent.parent


def _path_from_env(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    return Path(raw).expanduser().resolve() if raw else default


# ---------------------------------------------------------------- prediction

#: Horizon W. "Will this file be opened again within this many days?"
#: Ninety, not thirty. Asking about a month forces the model to be certain about the
#: least predictable stretch of a file's life, and it makes the honest answer "no".
#: A quarter is the question you can actually act on.
REOPEN_WINDOW_DAYS = 90

#: Dormant only asks that question about something already stale. Asking it at
#: creation time is a different problem (does this file matter?) and it floods the
#: positive class, because of course a fresh download gets opened.
MIN_DORMANT_AGE_DAYS = 21

#: A file whose only recorded open is younger than this is censored, not negative.
CENSOR_MIN_DAYS = REOPEN_WINDOW_DAYS

#: What a human would do without a model, used as the rule baseline.
RULE_DORMANT_AFTER_DAYS = 90

#: Target precision on the archive decision, measured on the held-out slice.
PRECISION_TARGET = 0.95

DEFAULT_TOP_K = 200


# ---------------------------------------------------------------------- data

@dataclass(frozen=True)
class Paths:
    data_dir: Path = field(default_factory=lambda: _path_from_env("DORMANT_DATA", REPO_ROOT / "data"))
    archive_dir: Path = field(default_factory=lambda: _path_from_env("DORMANT_ARCHIVE", Path.home() / "Dormant Archive"))

    @property
    def db_path(self) -> Path:
        return self.data_dir / "dormant.sqlite"

    @property
    def manifest_dir(self) -> Path:
        return self.data_dir / "manifests"

    @property
    def checkpoint_dir(self) -> Path:
        return _path_from_env("DORMANT_CHECKPOINTS", REPO_ROOT / "checkpoints")

    @property
    def tabpfn_v2_classifier(self) -> Path:
        return self.checkpoint_dir / "tabpfn-v2-classifier.ckpt"

    @property
    def tabpfn_v2_regressor(self) -> Path:
        return self.checkpoint_dir / "tabpfn-v2-regressor.ckpt"


PATHS = Paths()

# ------------------------------------------------------------------- scanning

#: Directories that are build noise: enormous, machine-generated, never "opened".
SCAN_EXCLUDE_DIRS = {
    ".git", ".svn", ".hg", "node_modules", ".venv", "venv", "env", "__pycache__",
    ".cache", ".npm", ".yarn", ".pnpm-store", ".gradle", ".m2", ".cargo", ".rustup",
    "build", "dist", "target", "out", ".next", ".nuxt", ".parcel-cache", "Pods",
    "DerivedData", ".terraform", "site-packages", "node-gyp", ".motif",
}

#: Bundles are directories that look like files and are never user-opened.
SCAN_EXCLUDE_SUFFIXES = (".app", ".framework", ".bundle", ".xpc", ".dSYM", ".plugin", ".pkg")

#: Only file types a person can actually open have a meaningful "reopened" label.
#: Source files are deliberately absent: nobody double-clicks a .php in Finder, and
#: leaving them in let the model recommend archiving a checked-out dependency.
OPENABLE_EXTS = {
    ".pdf", ".doc", ".docx", ".rtf", ".txt", ".md", ".markdown", ".pages", ".key",
    ".numbers", ".xls", ".xlsx", ".csv", ".ppt", ".pptx", ".pptm", ".epub", ".mobi",
    ".png", ".jpg", ".jpeg", ".gif", ".heic", ".heif", ".tiff", ".tif", ".webp",
    ".bmp", ".svg", ".psd", ".sketch", ".ai",
    ".mp3", ".m4a", ".wav", ".aac", ".flac", ".aiff",
    ".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm",
    ".zip", ".tar", ".gz", ".dmg", ".iso", ".7z", ".rar",
    ".json", ".yaml", ".yml", ".toml", ".ini", ".log", ".ipynb", ".tex", ".bib",
    ".ics", ".vcf", ".html", ".py", ".sql",
}

#: Files inside these trees are plumbing, not documents nobody double-clicks.
PROJECT_MARKERS = (".git", "node_modules", "Package.swift", "Podfile", "Podfile.lock", "Cargo.toml",
                   "go.mod", "pubspec.yaml", "package.json", "package-lock.json", "pnpm-lock.yaml",
                   "yarn.lock", "pyproject.toml", "requirements.txt", "setup.py", "Pipfile",
                   "composer.json", "composer.lock", "Gemfile", "pom.xml", "build.gradle",
                   "build.gradle.kts", "settings.gradle", "gradlew", "CMakeLists.txt", "Gopkg.toml")

#: Directory names that are build furniture wherever they appear. A checked-out
#: dependency is not a document that went to sleep, and the download-zip repos that
#: have no .git at all still contain these.
PROJECT_SEGMENTS = ("node_modules", "vendor", "DerivedData", "Pods", "site-packages", ".gradle",
                    ".terraform", ".cargo", ".pnpm-store", "bower_components", ".venv", "venv")

#: Files smaller than this carry no usable signal and are noise to archive.
MIN_FILE_BYTES = 2_048

#: Cap on files kept in the index.
MAX_SCAN_FILES = 40_000

#: Cap on files walked. Traversal is cheap, mdls is not, so these are separate budgets.
MAX_TRAVERSAL_FILES = 400_000

#: Cap on training rows. TabPFN puts the whole context through the model at once,
#: so a bigger number buys seconds, not accuracy.
MAX_TRAIN_ROWS = 9_000

#: mdls is a subprocess. Batching amortises its start-up cost across many files.
MDLS_BATCH = 150

DEFAULT_ROOTS: tuple[str, ...] = ("~/Downloads", "~/Desktop", "~/Documents")

API_HOST = "127.0.0.1"
API_PORT = int(os.environ.get("DORMANT_PORT", "8712"))
