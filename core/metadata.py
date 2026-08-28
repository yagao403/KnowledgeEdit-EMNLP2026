"""JSON-serializable metadata stored alongside message trajectories."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class Metadata(dict[str, Any]):
    """Small compatibility wrapper used by the XML message format."""

    def to_json(self) -> str:
        return json.dumps(self, indent=2, default=_json_default)

    @classmethod
    def from_json(cls, value: str) -> "Metadata":
        return cls(json.loads(value))


def _json_default(value: Any) -> str:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")
