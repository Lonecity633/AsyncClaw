"""Top-level AsyncClaw command line parser."""

from __future__ import annotations

import argparse
from pathlib import Path

from AsyncClaw.cli.agent import run_agent_cli


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="asyncclaw")
    subparsers = parser.add_subparsers(dest="command", required=True)

    agent_parser = subparsers.add_parser("agent", help="run the interactive agent")
    agent_parser.add_argument(
        "--cwd",
        type=Path,
        default=None,
        help="task working directory used by tools and channel context",
    )
    agent_parser.add_argument(
        "--env-file",
        default=None,
        help="dotenv file path, resolved relative to --cwd when not absolute",
    )
    agent_parser.add_argument(
        "--workspace-root",
        type=Path,
        default=None,
        help="workspace path for sessions, history, memory, and shell sandbox",
    )
    agent_parser.add_argument(
        "--no-shell",
        action="store_true",
        help="do not expose the shell_exec tool",
    )
    agent_parser.add_argument(
        "--no-cron",
        action="store_true",
        help="do not run the workspace cron heartbeat service",
    )
    agent_parser.add_argument(
        "--cron-max-concurrent-jobs",
        type=int,
        default=2,
        help="maximum number of cron jobs that may run concurrently",
    )
    agent_parser.set_defaults(func=_run_agent)
    return parser


def _run_agent(args: argparse.Namespace) -> int:
    return run_agent_cli(
        cwd=args.cwd,
        env_file=args.env_file or ".env",
        env_file_explicit=args.env_file is not None,
        workspace_root=args.workspace_root,
        allow_shell_exec=not args.no_shell,
        allow_cron=not args.no_cron,
        cron_max_concurrent_jobs=args.cron_max_concurrent_jobs,
    )


if __name__ == "__main__":
    raise SystemExit(main())
