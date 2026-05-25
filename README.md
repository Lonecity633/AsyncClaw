# AsyncClaw

一个使用 OpenAI 兼容工具调用的最小 ReAct 风格智能体循环，不依赖 LangChain 或 LangGraph。

## 包含内容

- `AsyncClaw.agent.AgentLoop`：使用 OpenAI 兼容聊天补全客户端运行推理-行动-观察循环。
- `AsyncClaw.agent.runtime.AgentLoop`：推荐的新运行时入口，支持同步 `run()` 和异步 `arun()`。
- `AsyncClaw.agent.messages`：集中处理 OpenAI SDK 对象或字典响应归一化、工具参数解析和工具消息构造。
- `AsyncClaw.config.load_llm_config`：通过 `python-dotenv` 加载 `.env`。
- `AsyncClaw.tools.ToolRegistry`：注册工具，并暴露 OpenAI `tools` schema。
- `AsyncClaw.tools.ToolExecutor`：统一执行工具，并将工具异常转换成结构化观察结果。
- `AsyncClaw.tools.ToolContext`：描述单次运行的工具能力和执行边界。
- `AsyncClaw.tools.shell_exec_tool`：可选的本地 shell 工具，仅在 `ToolContext` 允许时暴露。
- `AsyncClaw.tools.multiply_tool`：最简单的示例工具，用于计算两个数字的乘积。
- `AsyncClaw.tools.current_time_tool`：返回当前本地日期和时间。

工具使用以下内部结构，定义在 `AsyncClaw.tools.spec`：

```python
{
    "name": "tool_name",
    "description": "...",
    "schema": {...},
    "handler": callable,
}
```

内置工具放在 `AsyncClaw.tools.builtin` 下。旧路径如 `AsyncClaw.tools.math_tools`、`AsyncClaw.tools.time_tools` 和 `AsyncClaw.tools.shell_exec` 仍保留为兼容导入。

循环期望传入符合 OpenAI Python SDK 形状的客户端对象：

```python
client.chat.completions.create(
    model="...",
    messages=[...],
    tools=[...],
    tool_choice="auto",
)
```

它也接受相同响应形状的普通字典，便于测试和本地演示保持无额外依赖。

如果在异步程序中使用，请调用 `arun()`：

```python
result = await agent.arun(messages)
```

同步 CLI 或脚本可以继续调用：

```python
result = agent.run(messages)
```

## 由上下文控制的 shell 执行

`shell_exec` 默认不会暴露。请基于 `ToolContext` 为每次运行构造工具集合：

```python
from pathlib import Path

from AsyncClaw.tools import ToolContext, build_tool_registry

tool_context = ToolContext(cwd=Path.cwd(), allow_shell_exec=True)
tools = build_tool_registry(tool_context)
```

将同一个上下文传入 `AgentLoop`：

```python
agent = AgentLoop(
    llm=llm,
    tools=tools,
    tool_context=tool_context,
)
```

当 `shell_exec` 被调用时，会先执行安全检查，拒绝明显破坏性的命令，对未知命令请求 CLI 审批，并在 `ToolContext.cwd` 中执行已批准命令。默认超时时间为 10 秒，stdout/stderr 分别截断到 16 KB。

## 配置真实 API

编辑 `.env`：

```bash
OPENAI_API_KEY=你的-api-key
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini
AGENT_MAX_STEPS=8
```

安装依赖：

```bash
pip install -r requirements.txt
```

## 运行示例

```bash
python -m examples.openai_agent
```

`examples.openai_agent` 会启动交互式 CLI。输入 `exit` 或 `quit` 即可退出。

## 智能体日志

智能体事件会以 JSONL 格式写入：

```bash
logs/events.jsonl
```

每一行都是一个事件，包括 `user_message`、`llm_request`、`llm_response`、`tool_call`、`tool_result`、`final_answer` 和 `error`。

运行 CLI 时可以用以下命令实时查看日志：

```bash
tail -f logs/events.jsonl
```

## 运行测试

```bash
python -m unittest discover -s tests
```
