"""使用真实 OpenAI 兼容 API 运行交互式 CLI。"""

from __future__ import annotations

from pathlib import Path

from AsyncClaw.agent.llm import create_openai_llm
from AsyncClaw.agent.runtime import AgentLoop
from AsyncClaw.config import load_llm_config
from AsyncClaw.tools import ToolContext, build_tool_registry


def main() -> None:
    config = load_llm_config()
    llm = create_openai_llm(config)
    tool_context = ToolContext(cwd=Path.cwd(), allow_shell_exec=True)
    agent = AgentLoop(
        llm=llm,
        tools=build_tool_registry(tool_context),
        max_steps=config.agent_max_steps,
        tool_context=tool_context,
    )
    messages = []

    print("AsyncClaw 交互式智能体。输入 'exit' 或 'quit' 退出。")
    while True:
        user_text = input("用户：").strip()
        if not user_text:
            continue
        if user_text.lower() in {"exit", "quit"}:
            break

        messages.append({"role": "user", "content": user_text})
        try:
            result = agent.run(messages)
        except Exception as exc:
            print(f"错误：{exc}")
            messages.pop()
            continue

        messages = result.messages
        print(f"助手：{result.output}")


if __name__ == "__main__":
    main()
