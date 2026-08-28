"""Small shared utilities used by inference, message serialization, and training."""

from __future__ import annotations

from enum import Enum
import io
import os
from pathlib import Path
import re
import sys
import tokenize

from core import ADAPTER_PATH, BASE_PATH, LOGBOOK_PATH


class Colors:
    DEFAULT = "\033[0m"
    MAGENTA = "\033[95m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[33m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    ITALIC = "\033[3m"


class DualOutput:
    """Write stdout to both the terminal and a log file."""

    def __init__(self, filename: str | Path, mode: str = "a"):
        self.terminal = sys.stdout
        self.log = open(filename, mode, encoding="utf-8")

    def write(self, message: str) -> None:
        self.terminal.write(message)
        self.log.write(message)

    def flush(self) -> None:
        self.terminal.flush()
        self.log.flush()


def find_runs(path: os.PathLike, pattern: str) -> list[Path]:
    return list(Path(path).glob(f"*/*{pattern}"))


def get_adapter_path(adapter_id: str | Path) -> str | Path:
    """Resolve an adapter path or unique run-name suffix."""

    if not adapter_id or Path(adapter_id).exists():
        return adapter_id

    matches = find_runs(BASE_PATH / "checkpoints", str(adapter_id))
    matches.extend(find_runs(ADAPTER_PATH, str(adapter_id)))
    matches.extend(find_runs(LOGBOOK_PATH / "training", str(adapter_id)))
    if len(matches) > 1:
        raise ValueError(f"Multiple adapters found: {matches}")
    if not matches:
        raise ValueError(f"Adapter not found: {adapter_id}")
    print(f"Adapter found: {matches[0]}")
    return matches[0]


def xml_encoder(input_string: str) -> str:
    """Encode XML-disallowed control characters without losing information."""

    def replace_control_chars(match: re.Match) -> str:
        return f"__{ord(match.group()):02x}"

    escaped = input_string.replace("__", "__US")
    return re.sub(r"[\x00-\x09\x0b-\x1f\x7f]", replace_control_chars, escaped)


def xml_decoder(encoded_string: str) -> str:
    """Reverse :func:`xml_encoder`."""

    decoded = re.sub(r"__([0-9a-f]{2})", lambda match: chr(int(match.group(1), 16)), encoded_string)
    return decoded.replace("__US", "__")


class MyEnum(Enum):
    @classmethod
    def from_value(cls, value):
        for member in cls:
            if member.value == value:
                return member
        raise ValueError(f"{value} is not a valid value in {cls.__name__}")

    def __eq__(self, other):
        return hasattr(other, "value") and self.value == other.value

    def __hash__(self):
        return hash(self.value)


def remove_comments(code: str, remove_empty_lines: bool = False) -> str:
    """Remove Python comments while preserving strings and line structure."""

    lines = code.splitlines(keepends=True)
    for token in tokenize.generate_tokens(io.StringIO(code).readline):
        if token.type == tokenize.COMMENT:
            row, column = token.start
            line = lines[row - 1]
            newline = "\n" if line.endswith("\n") else ""
            lines[row - 1] = line[:column].rstrip() + newline
    result = "".join(lines)
    if remove_empty_lines:
        result = "\n".join(line for line in result.splitlines() if line.strip())
    return result
