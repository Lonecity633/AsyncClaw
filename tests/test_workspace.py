from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from AsyncClaw.workspace import DEFAULT_SYSTEM_PROMPT, WorkspaceStore


class WorkspaceStoreTest(unittest.TestCase):
    def test_creates_workspace_directories_and_session_id(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = WorkspaceStore(root=Path(directory) / "workspace")

            self.assertTrue(workspace.session_id)
            self.assertTrue(workspace.session_dir.is_dir())
            self.assertTrue(workspace.history_dir.is_dir())
            self.assertTrue(workspace.memory_dir.is_dir())

    def test_session_reads_recent_ten_messages(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = WorkspaceStore(
                root=Path(directory) / "workspace",
                session_id="session-a",
                short_term_limit=10,
            )
            for index in range(12):
                workspace.append_session_message("user", f"message-{index}")

            messages = workspace.load_recent_messages()

        self.assertEqual(len(messages), 10)
        self.assertEqual(messages[0], {"role": "user", "content": "message-2"})
        self.assertEqual(messages[-1], {"role": "user", "content": "message-11"})

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
