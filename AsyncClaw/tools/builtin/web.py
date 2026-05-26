"""Tavily-backed web search and fetch tools."""

from __future__ import annotations

import os
from typing import Any

from AsyncClaw.tools.spec import Tool


SEARCH_DEPTHS = {"basic", "advanced"}
SEARCH_TOPICS = {"general", "news", "finance"}
EXTRACT_DEPTHS = {"basic", "advanced"}
FETCH_FORMATS = {"markdown", "text"}


def _web_search(arguments: dict[str, Any]) -> dict[str, Any]:
    query = _required_string(arguments, "query")
    max_results = _int_in_range(arguments.get("max_results", 5), "max_results", 1, 20)
    search_depth = _enum_value(
        arguments.get("search_depth", "basic"),
        "search_depth",
        SEARCH_DEPTHS,
    )
    topic = _enum_value(arguments.get("topic", "general"), "topic", SEARCH_TOPICS)

    payload: dict[str, Any] = {
        "query": query,
        "max_results": max_results,
        "search_depth": search_depth,
        "topic": topic,
    }
    _copy_optional(arguments, payload, "time_range")
    _copy_optional(arguments, payload, "include_answer")
    _copy_optional(arguments, payload, "include_raw_content")
    _copy_string_list(arguments, payload, "include_domains")
    _copy_string_list(arguments, payload, "exclude_domains")

    return _tavily_client().search(**payload)


def _web_fetch(arguments: dict[str, Any]) -> dict[str, Any]:
    url = _required_string(arguments, "url")
    extract_depth = _enum_value(
        arguments.get("extract_depth", "basic"),
        "extract_depth",
        EXTRACT_DEPTHS,
    )
    content_format = _enum_value(arguments.get("format", "markdown"), "format", FETCH_FORMATS)
    include_images = bool(arguments.get("include_images", False))

    return _tavily_client().extract(
        urls=url,
        extract_depth=extract_depth,
        format=content_format,
        include_images=include_images,
    )


def _tavily_client() -> Any:
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError("缺少 TAVILY_API_KEY，无法使用 Tavily web 工具。")
    try:
        from tavily import TavilyClient
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "缺少 tavily-python 依赖。请安装项目依赖后再使用 Tavily web 工具。"
        ) from exc
    return TavilyClient(api_key=api_key)


def _required_string(arguments: dict[str, Any], name: str) -> str:
    value = arguments.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} 必须是非空字符串。")
    return value.strip()


def _int_in_range(value: Any, name: str, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 必须是整数。") from exc
    if parsed < minimum or parsed > maximum:
        raise ValueError(f"{name} 必须在 {minimum} 到 {maximum} 之间。")
    return parsed


def _enum_value(value: Any, name: str, allowed: set[str]) -> str:
    if not isinstance(value, str) or value not in allowed:
        choices = ", ".join(sorted(allowed))
        raise ValueError(f"{name} 必须是以下值之一：{choices}。")
    return value


def _copy_optional(source: dict[str, Any], target: dict[str, Any], name: str) -> None:
    if name in source and source[name] is not None:
        target[name] = source[name]


def _copy_string_list(source: dict[str, Any], target: dict[str, Any], name: str) -> None:
    if name not in source or source[name] is None:
        return
    value = source[name]
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{name} 必须是字符串数组。")
    target[name] = [item for item in value if item.strip()]


web_search_tool = Tool(
    name="web_search",
    description=(
        "使用 Tavily 执行实时网页搜索，适合获取最新信息、新闻、资料来源和网页摘要。"
    ),
    schema={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "搜索查询。"},
            "max_results": {
                "type": "integer",
                "minimum": 1,
                "maximum": 20,
                "default": 5,
                "description": "返回结果数量。默认 5，范围 1-20。",
            },
            "search_depth": {
                "type": "string",
                "enum": ["basic", "advanced"],
                "default": "basic",
                "description": "搜索深度。",
            },
            "topic": {
                "type": "string",
                "enum": ["general", "news", "finance"],
                "default": "general",
                "description": "搜索主题。",
            },
            "time_range": {
                "type": "string",
                "enum": ["day", "week", "month", "year"],
                "description": "限制搜索结果时间范围。",
            },
            "include_answer": {
                "oneOf": [
                    {"type": "boolean"},
                    {"type": "string", "enum": ["basic", "advanced"]},
                ],
                "description": "是否包含 Tavily 生成的回答。",
            },
            "include_raw_content": {
                "oneOf": [
                    {"type": "boolean"},
                    {"type": "string", "enum": ["markdown", "text"]},
                ],
                "description": "是否包含搜索结果页面正文。",
            },
            "include_domains": {
                "type": "array",
                "items": {"type": "string"},
                "description": "只搜索这些域名。",
            },
            "exclude_domains": {
                "type": "array",
                "items": {"type": "string"},
                "description": "排除这些域名。",
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    },
    handler=_web_search,
)


web_fetch_tool = Tool(
    name="web_fetch",
    description="使用 Tavily 抽取单个 URL 的网页正文，可返回 markdown 或 text。",
    schema={
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "要抓取的网页 URL。"},
            "extract_depth": {
                "type": "string",
                "enum": ["basic", "advanced"],
                "default": "basic",
                "description": "抽取深度。",
            },
            "format": {
                "type": "string",
                "enum": ["markdown", "text"],
                "default": "markdown",
                "description": "返回正文格式。",
            },
            "include_images": {
                "type": "boolean",
                "default": False,
                "description": "是否包含图片 URL。",
            },
        },
        "required": ["url"],
        "additionalProperties": False,
    },
    handler=_web_fetch,
)
