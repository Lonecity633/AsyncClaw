from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from AsyncClaw.agent.workspace import DEFAULT_SYSTEM_PROMPT, WorkspaceStore


class WorkspaceStoreTest(unittest.TestCase):
    def test_creates_workspace_directories_and_session_id(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = WorkspaceStore(root=Path(directory) / "workspace")

            self.assertTrue(workspace.session_id)
            self.assertTrue(workspace.session_dir.is_dir())
            self.assertTrue(workspace.history_dir.is_dir())
            self.assertTrue(workspace.memory_dir.is_dir())

    def test_session_reads_all_turns_below_summary_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = WorkspaceStore(
                root=Path(directory) / "workspace",
                session_id="session-a",
            )
            for index in range(12):
                workspace.append_session_turn(
                    [
                        {"role": "user", "content": f"user-{index}"},
                        {"role": "assistant", "content": f"assistant-{index}"},
                    ]
                )

            messages = workspace.load_context_messages()

        self.assertEqual(len(messages), 24)
        self.assertEqual(messages[0], {"role": "user", "content": "user-0"})
        self.assertEqual(messages[-1], {"role": "assistant", "content": "assistant-11"})

    def test_session_compaction_keeps_recent_complete_turns(self) -> None:
        async def summarize(previous_summary, discarded_turns):
            self.assertEqual(previous_summary, "旧摘要")
            self.assertEqual(len(discarded_turns), 2)
            return "新摘要"

        with tempfile.TemporaryDirectory() as directory:
            workspace = WorkspaceStore(
                root=Path(directory) / "workspace",
                session_id="session-a",
                summary_threshold=3,
                recent_turn_limit=1,
            )
            workspace.save_short_term_summary("旧摘要")
            for index in range(3):
                workspace.append_session_turn(
                    [
                        {"role": "user", "content": f"user-{index}"},
                        {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": f"call-{index}",
                                    "type": "function",
                                    "function": {
                                        "name": "multiply",
                                        "arguments": '{"a": 2, "b": 3}',
                                    },
                                }
                            ],
                        },
                        {
                            "role": "tool",
                            "tool_call_id": f"call-{index}",
                            "name": "multiply",
                            "content": '{"product": 6}',
                        },
                        {"role": "assistant", "content": f"done-{index}"},
                    ]
                )

            result = asyncio.run(workspace.compact_session_if_needed(summarize))
            turns = workspace.load_session_turns()
            messages = workspace.load_context_messages()
            summary = workspace.load_short_term_summary()

        self.assertTrue(result["compacted"])
        self.assertEqual(summary, "新摘要")
        self.assertEqual(len(turns), 1)
        self.assertEqual([message["role"] for message in turns[0]["messages"]], [
            "user",
            "assistant",
            "tool",
            "assistant",
        ])
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("新摘要", messages[0]["content"])
        self.assertEqual(messages[1]["content"], "user-2")

    def test_history_records_user_inputs_with_session_id(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = WorkspaceStore(
                root=Path(directory) / "workspace",
                session_id="session-a",
            )

            workspace.append_user_input("你好")

            records = _read_jsonl(workspace.user_inputs_path)

        self.assertEqual(records[0]["session_id"], "session-a")
        self.assertEqual(records[0]["content"], "你好")
        self.assertIn("timestamp", records[0])

    def test_save_user_profile_overwrites_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = WorkspaceStore(root=Path(directory) / "workspace")

            first = workspace.save_user_profile("# 用户画像\n- 喜欢 Python")
            second = workspace.save_user_profile("# 用户画像\n- 喜欢测试")

            profile = workspace.load_user_profile()

        self.assertTrue(first["saved"])
        self.assertTrue(second["saved"])
        self.assertEqual(profile, "# 用户画像\n- 喜欢测试")

    def test_system_prompt_contains_memory_decision_policy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = WorkspaceStore(root=Path(directory) / "workspace")

            prompt = workspace.build_system_prompt()

        self.assertIn("save_user_profile", prompt)
        self.assertIn("稳定身份背景", prompt)
        self.assertIn("稳定偏好和兴趣", prompt)
        self.assertIn("长期目标和计划", prompt)
        self.assertIn("不要记录到 memory", prompt)
        self.assertIn("（暂无用户画像）", prompt)

    def test_default_prompt_does_not_embed_specific_user_facts(self) -> None:
        self.assertNotIn("打篮球", DEFAULT_SYSTEM_PROMPT)
        self.assertNotIn("玩游戏", DEFAULT_SYSTEM_PROMPT)
        self.assertNotIn("计算机的", DEFAULT_SYSTEM_PROMPT)


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


if __name__ == "__main__":
    unittest.main()
