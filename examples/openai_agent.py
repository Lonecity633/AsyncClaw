"""使用真实 OpenAI 兼容 API 运行交互式 CLI。"""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from AsyncClaw.agent.llm import create_openai_llm
from AsyncClaw.agent.runtime import AgentLoop
from AsyncClaw.config import load_llm_config
from AsyncClaw.tools import ToolContext, build_tool_registry
from AsyncClaw.workspace import WorkspaceStore


def main() -> None:
    config = load_llm_config()
    llm = create_openai_llm(config)
    tool_context = ToolContext(cwd=Path.cwd(), allow_shell_exec=True)
    workspace = WorkspaceStore(root=Path.cwd() / "workspace")
    agent = AgentLoop(
        llm=llm,
        tools=build_tool_registry(tool_context, workspace=workspace),
        max_steps=config.agent_max_steps,
        tool_context=tool_context,
        workspace=workspace,
    )

    print("AsyncClaw 交互式智能体。输入 'exit' 或 'quit' 退出。")
    print(f"session_id: {workspace.session_id}")
    while True:
        user_text = input("用户：").strip()
        if not user_text:
            continue
        if user_text.lower() in {"exit", "quit"}:
            break

        messages = [{"role": "user", "content": user_text}]
        try:
            result = agent.run(messages)
        except Exception as exc:
            print(f"错误：{exc}")
            continue

        print(f"助手：{result.output}")


if __name__ == "__main__":
    main()
