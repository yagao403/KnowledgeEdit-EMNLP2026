"""Path helpers for XML distillation datasets."""

from __future__ import annotations

import glob
from pathlib import Path

from core import DATA_PATH


def resolve_data_path(path: str | Path) -> Path:
    """Resolve string paths relative to ``DATA_PATH``; preserve ``Path`` inputs."""

    return DATA_PATH / path if isinstance(path, str) else path


def get_exercises_glob(path: str | Path) -> list[Path]:
    """Expand an XML file, directory, or recursive glob into sorted files."""

    resolved = resolve_data_path(path)
    pattern = str(resolved)
    if any(character in pattern for character in "*?["):
        return sorted(Path(match) for match in glob.glob(pattern, recursive=True))
    if resolved.is_dir():
        return sorted(resolved.glob("*.xml"))
    if resolved.suffix == ".xml" and resolved.exists():
        return [resolved]
    return []


def get_exercise_files(
    base_model: str,
    split: str,
    model: str,
    dataset: str,
    folder: str = "data",
) -> list[Path]:
    return sorted((DATA_PATH / base_model / split / model / dataset / folder).glob("*.xml"))
