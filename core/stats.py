"""Lightweight request statistics used by the vLLM client."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from statistics import fmean
from typing import DefaultDict

from core.usage import Usage


@dataclass
class Stats:
    start_time: datetime = field(default_factory=datetime.now)
    duration: timedelta | None = None
    call_times: list[float] = field(default_factory=list)
    tool_calls: DefaultDict[str, int] = field(default_factory=lambda: defaultdict(int))
    retry_count: int = 0
    usages: list[Usage] = field(default_factory=list)
    costs: list[float] = field(default_factory=list)
    parent_stats: "Stats | None" = None
    n_steps: int = 0

    @classmethod
    def from_status(cls, status: dict | None = None) -> "Stats":
        if status and (start_time := status.get("start_time")):
            return cls(start_time=datetime.fromisoformat(start_time))
        return cls()

    @property
    def n_calls(self) -> int:
        return len(self.call_times)

    @property
    def usage(self) -> Usage:
        return sum(self.usages, Usage())

    @property
    def cost(self) -> float:
        return sum(self.costs)

    def add_tool_call(self, tool_name: str) -> None:
        self.tool_calls[tool_name] += 1
        if self.parent_stats:
            self.parent_stats.add_tool_call(tool_name)

    def set_duration(self) -> None:
        self.duration = datetime.now() - self.start_time

    def update(
        self,
        usage: Usage | None = None,
        call_time: float = 1.0,
        retry_count: int = 0,
        model: str | None = None,
    ) -> None:
        self.call_times.append(call_time)
        self.retry_count += retry_count
        if usage:
            self.usages.append(usage)
        if model is not None:
            if usage is None:
                raise ValueError("usage is required when model is provided")
            self.costs.append(usage.to_cost(model) or 0.0)
        if self.parent_stats:
            self.parent_stats.update(usage, call_time, retry_count, model)

    def to_dict(self) -> dict:
        return {
            "start_time": self.start_time.isoformat(),
            "duration": int(self.duration.total_seconds()) if self.duration else None,
            "n_calls": self.n_calls,
            "n_steps": self.n_steps,
            "mean_call_time": fmean(self.call_times) if self.call_times else None,
            "tool_calls": dict(self.tool_calls),
            "retry_count": self.retry_count,
            "total_usage": self.usage.to_dict(),
            "total_cost": round(self.cost, 2),
        }
