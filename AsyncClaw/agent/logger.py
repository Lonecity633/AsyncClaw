"""智能体运行事件的 JSONL 日志。"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class JsonlEventLogger:
    """将智能体事件追加写入 JSONL 文件。"""

    def __init__(self, path: str | Path = "logs/events.jsonl"):
        self.path = Path(path)

    def log(self, event: str, data: dict[str, Any]) -> None:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "data": data,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False, default=str))
            file.write("\n")
