"""使用真实 OpenAI 兼容 API 运行交互式 CLI。"""

from __future__ import annotations

from AsyncClaw.cli.agent import run_agent_cli


def main() -> None:
    raise SystemExit(run_agent_cli())


if __name__ == "__main__":
    main()
