"""Small transport-neutral request and response models."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AgentRequest:
    """A single user text request from any channel."""

    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentResponse:
    """A single agent response that can be rendered by any channel."""

    output: str | None
    session_id: str
    cwd: Path
    steps: int
    messages: list[dict[str, Any]]
    observations: list[dict[str, Any]] = field(default_factory=list)
