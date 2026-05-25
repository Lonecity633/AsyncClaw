"""时间相关工具。"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from AsyncClaw.tools.spec import Tool


def _current_time(arguments: dict[str, Any]) -> dict[str, str]:
    now = datetime.now().astimezone()
    return {
        "timezone": str(now.tzinfo),
        "iso_time": now.isoformat(timespec="seconds"),
        "date": now.date().isoformat(),
        "time": now.time().replace(microsecond=0).isoformat(),
    }


current_time_tool = Tool(
    name="current_time",
    description="返回当前本地日期和时间。",
    schema={
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    },
    handler=_current_time,
)
