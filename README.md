# AsyncClaw

一个使用 OpenAI 兼容工具调用的最小 ReAct 风格智能体循环，不依赖 LangChain 或 LangGraph。

## 包含内容

- `AsyncClaw.agent.AgentLoop`：使用 OpenAI 兼容聊天补全客户端运行推理-行动-观察循环。
- `AsyncClaw.agent.runtime.AgentLoop`：推荐的新运行时入口，支持同步 `run()` 和异步 `arun()`。
- `AsyncClaw.channels.AgentService`：可复用的文本请求服务层，CLI 和后续 channel 都可以接入。
- `AsyncClaw.agent.messages`：集中处理 OpenAI SDK 对象或字典响应归一化、工具参数解析和工具消息构造。
- `AsyncClaw.config.load_llm_config`：通过 `python-dotenv` 加载 `.env`，支持按 provider 切换模型服务商。
- `AsyncClaw.tools.ToolRegistry`：注册工具，并暴露 OpenAI `tools` schema。
- `AsyncClaw.tools.ToolExecutor`：统一执行工具，并将工具异常转换成结构化观察结果。
- `AsyncClaw.tools.ToolContext`：描述单次运行的工具能力和执行边界。
- `AsyncClaw.tools.resolve_sandbox_path`：解析并校验软沙箱内路径，禁止越界访问。
- `AsyncClaw.tools.shell_exec_tool`：可选的本地 shell 工具，仅在 `ToolContext` 允许时暴露。
- `AsyncClaw.agent.workspace.WorkspaceStore`：在 `workspace/` 中存储会话、用户输入历史和长期用户画像。
- `AsyncClaw.agent.cron.CronStore` / `CronService`：在 `workspace/cron/jobs.json` 中存储定时任务，并通过 heartbeat 定期触发。
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

内置工具放在 `AsyncClaw.tools.builtin` 下；工具基础类型、注册表和安全策略分别放在 `AsyncClaw.tools.spec`、`AsyncClaw.tools.registry` 和 `AsyncClaw.tools.safety`。

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

默认情况下，`ToolContext` 会把软沙箱根目录设为 `Path.cwd() / "workspace" / "office"`，并自动创建该目录。`shell_exec` 即使收到其他 `cwd`，也只会在这个 `sandbox_root` 中执行。

将同一个上下文传入 `AgentLoop`：

```python
agent = AgentLoop(
    llm=llm,
    tools=tools,
    tool_context=tool_context,
)
```

当 `shell_exec` 被调用时，会先执行软沙箱安全检查：

- `safe`：只读环境诊断命令可直接执行，例如 `pwd`、`ls`、`python --version`、`conda --version`、`conda env list`、`which python`、`which conda`，以及简单只读管道。
- `confirm`：删除、覆盖、安装依赖、网络访问、git 修改、长时间运行等命令需要 CLI 审批。
- `deny`：绝对路径、`..`、`~`、敏感文件、office 外路径、`python -c`、`node -e` 等绕过沙箱的方式会直接拒绝。

默认超时时间为 10 秒，stdout/stderr 分别截断到 16 KB。这个机制是应用层软限制，不等同于操作系统或容器级隔离。

## Workspace 记忆

CLI 默认启用 `WorkspaceStore`，数据写入：

```text
workspace/session/{session_id}.jsonl
workspace/session/{session_id}.summary.md
workspace/history/user_inputs.jsonl
workspace/memory/user_profile.md
workspace/cron/jobs.json
```

每次运行 CLI 会自动生成一个 `session_id`。同一会话内的完整回合写入 session；一轮包含用户消息、助手工具调用、工具结果和助手最终回复。所有用户输入同时写入全局 history。

短期记忆按完整回合裁剪，避免把工具调用链拆开：

- 少于 40 轮时，保留全部历史回合。
- 达到或超过 40 轮时，复用当前 LLM 将旧摘要和更早回合合并为新的近期对话摘要。
- 后续请求会注入系统提示词、当前 `user_profile.md`、近期对话摘要和最近 10 轮完整消息。

启用 workspace 时会暴露 `save_user_profile` 工具。模型判断用户消息包含长期信息时，可以调用该工具并传入完整 Markdown 用户画像，工具会覆盖写入 `workspace/memory/user_profile.md`。

## Cron 定时任务

CLI 默认启动一个轻量 heartbeat 服务，每 1 秒扫描 `workspace/cron/jobs.json`。任务到期时会把 `prompt` 作为独立的 cron agent turn 执行：原始任务只作为用户消息进入模型，调度器约束会作为系统提示注入，因此模型仍会根据可用工具自动决定是否调用 `current_time`、`shell_exec` 等工具，同时不会把内部调度提示混入用户可见输出。

默认最多同时执行 2 个定时任务；可以通过 CLI 调整：

```bash
asyncclaw agent --cron-max-concurrent-jobs 4
```

启用 workspace 时会暴露以下工具：

- `create_cron_job`：创建定时任务。
- `list_cron_jobs`：列出所有定时任务。
- `delete_cron_job`：按 `id` 删除定时任务。

调度格式支持：

```json
{"schedule": {"type": "at", "run_at": "2026-05-25T10:00:00+00:00"}}
{"schedule": {"type": "every", "seconds": 3600}}
```

一次性 `at` 任务执行后会自动禁用；周期性 `every` 任务执行后会计算下一次运行时间。运行中的任务会被标记为 `running`，同一个任务不会在上一次执行完成前重复触发。执行失败会记录在任务的 `last_error` 和 `failure_count` 中，后续周期仍会继续调度。

## 配置真实 API

编辑 `.env`：

```bash
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=你的-api-key
DEEPSEEK_MODEL=deepseek-v4-flash
AGENT_MAX_STEPS=8
```

内置 provider 定义在 `AsyncClaw.providers`，当前包含：

- `openai`
- `deepseek`
- `siliconflow`
- `xiaomi`
- `anthropic`

除 Anthropic 原生 API 外，当前入口使用 OpenAI-compatible Chat Completions 形状。Anthropic 如需使用，请通过 `LLM_BASE_URL` 指向兼容端点；原生 Anthropic SDK 可在后续新增适配器。

通用配置也可用于任意 provider：

```bash
LLM_PROVIDER=siliconflow
LLM_API_KEY=你的-api-key
LLM_MODEL=deepseek-ai/DeepSeek-V3
```

Provider 专属配置优先级高于通用配置，`LLM_BASE_URL` 可覆盖 provider 默认地址。旧的 OpenAI 配置仍兼容：

```bash
OPENAI_API_KEY=你的-api-key
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini
```

开发安装：

```bash
pip install -e .
```

## 运行示例

```bash
asyncclaw agent
```

安装后可在任意目录运行 `asyncclaw agent`。CLI 会把启动目录作为任务目录，工具上下文中的 `cwd` 会指向这里；会话和记忆默认写入 AsyncClaw 项目根目录：

- `workspace/`：会话、历史输入、长期记忆和 shell 软沙箱。
- `logs/events.jsonl`：智能体事件日志。

启动面板中的 `workspace` 字段会显示实际记忆存储路径。如果想把 session 和 memory 写到指定位置，可以运行：

```bash
asyncclaw agent --workspace-root /path/to/asyncclaw-workspace
```

默认配置会优先读取启动目录下的 `.env`；如果不存在，会回退到 AsyncClaw 项目根目录的 `.env`。启动面板中的 `env` 字段会显示实际加载路径。也可以用 `--env-file` 指定配置文件，相对路径会按 `--cwd` 解析。

输入 `exit` 或 `quit` 即可退出。

如果不想暴露 `shell_exec` 工具，可以运行：

```bash
asyncclaw agent --no-shell
```

如果不想启动 cron heartbeat，可以运行：

```bash
asyncclaw agent --no-cron
```

旧示例仍可运行，并会转到同一个 Rich CLI：

```bash
python -m examples.openai_agent
```

后续接入其他输入来源时，优先复用 `AgentService`：

```python
from AsyncClaw.channels import AgentService

service = AgentService()
response = service.handle_text("你好")
print(response.output)
```

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
