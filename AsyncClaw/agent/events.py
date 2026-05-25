"""智能体事件日志协议。"""

from __future__ import annotations

from typing import Any, Protocol


class EventLogger(Protocol):
    """运行时使用的最小日志接口。"""

    def log(self, event: str, data: dict[str, Any]) -> Any:
        ...
