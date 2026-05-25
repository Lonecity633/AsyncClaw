"""简单数学工具。"""

from __future__ import annotations

from typing import Any

from AsyncClaw.tools.spec import Tool


def _multiply(arguments: dict[str, Any]) -> dict[str, float | int]:
    a = arguments["a"]
    b = arguments["b"]
    return {"product": a * b}


multiply_tool = Tool(
    name="multiply",
    description="将两个数字相乘并返回乘积。",
    schema={
        "type": "object",
        "properties": {
            "a": {"type": "number", "description": "第一个因数。"},
            "b": {"type": "number", "description": "第二个因数。"},
        },
        "required": ["a", "b"],
        "additionalProperties": False,
    },
    handler=_multiply,
)
