"""Run one dreaming agent. This is the only module that imports LangChain or DeepAgents."""

import json
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from deepagents.backends import CompositeBackend, FilesystemBackend
from deepagents.backends.protocol import EditResult, ReadResult, WriteResult
from deepagents.backends.utils import validate_path
from deepagents.middleware.filesystem import FilesystemMiddleware, FilesystemPermission
from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic
from langchain_anthropic.middleware import AnthropicPromptCachingMiddleware

# The official prompts hard-code the transcript path, so the tools expose the same virtual paths.
UPLOADS_PATH = "/mnt/session/uploads"
MEMORY_PATH = "/mnt/memory/dreaming"

# The official analyzer and organizer toolset: read, write, edit, glob and grep.
TOOLS = ["read_file", "write_file", "edit_file", "glob", "grep"]

# The first matching rule wins and a path no rule matches is allowed. These globs do not match hidden paths,
# so ReadOnlyBackend is what actually keeps agents from writing anywhere but memory.
PERMISSIONS = [
    FilesystemPermission(operations=["write"], paths=[f"{UPLOADS_PATH}/**"], mode="deny"),
    FilesystemPermission(operations=["read"], paths=[f"{UPLOADS_PATH}/**"], mode="allow"),
    FilesystemPermission(operations=["read", "write"], paths=[MEMORY_PATH, f"{MEMORY_PATH}/**"], mode="allow"),
    FilesystemPermission(operations=["read", "write"], paths=["/**"], mode="deny"),
]

# US dollars per million tokens at list price, from the cookbook's helper. Prices change: check the pricing
# page before relying on a figure. cache_write is the 5-minute rate, which the caching middleware uses.
PRICES = {
    "claude-opus-5-5": {"input": 4.00, "cache_write": 5.00, "cache_read": 0.20, "output": 20.00},
    "claude-sonnet-5-5": {"input": 2.00, "cache_write": 2.50, "cache_read": 0.10, "output": 10.00},
    "claude-haiku-5-5": {"input": 0.10, "cache_write": 0.125, "cache_read": 0.01, "output": 0.50},
}
# A request whose prompt is over this many tokens pays these rates instead.
LONG_PROMPT_PRICES = {
    "claude-haiku-5-5": {"over": 100_000, "input": 0.50, "cache_write": 0.625, "cache_read": 0.05, "output": 2.50},
}

# Tool arguments the tool log leaves out. It records where an agent read and wrote, not what, so file text goes.
# So does read_file's limit, which the backends replace with the run's own on a read of a file's start.
UNLOGGED_ARGUMENTS = {"content", "old_string", "new_string", "limit"}


class LimitedBackend(FilesystemBackend):
    """Returns up to the run's number of lines from a read of a file's start, whatever limit the agent asked for."""

    def __init__(self, read_limit: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.read_limit = read_limit

    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> ReadResult:
        # A later offset keeps the agent's limit: widening it would resend text the agent already has.
        return super().read(file_path, offset, self.read_limit if offset == 0 else min(limit, self.read_limit))


class ReadOnlyBackend(LimitedBackend):
    """Refuses every change, whatever the path looks like."""

    def write(self, file_path: str, *args, **kwargs) -> WriteResult:
        return WriteResult(error=f"Error: {file_path} is read-only")

    def edit(self, file_path: str, *args, **kwargs) -> EditResult:
        return EditResult(error=f"Error: {file_path} is read-only")


class MemoryBackend(LimitedBackend):
    """Keeps memory to Markdown pages: agents have no delete tool, so a stray file would stay forever."""

    def write(self, file_path: str, content: str, *args, **kwargs) -> WriteResult:
        if not file_path.endswith(".md"):
            return WriteResult(error=f"Error: memory files must end in .md: {file_path}")
        return super().write(file_path, content, *args, **kwargs)


@dataclass
class AgentRun:
    """What one agent run produced. error is set when the agent did not finish normally."""

    text: str = ""
    error: str | None = None
    usage: Counter = field(default_factory=Counter)
    cost: float | None = None  # estimated US dollars; None when the model has no entry in PRICES


def _count_usage(run: AgentRun, model: str, metadata: dict) -> None:
    details = metadata.get("input_token_details", {})
    prompt = metadata.get("input_tokens", 0)  # includes cached tokens
    output = metadata.get("output_tokens", 0)
    cache_read = details.get("cache_read") or 0
    # Cache writes are reported either as one total or split by TTL, never both.
    cache_write = sum(
        details.get(key) or 0
        for key in ("cache_creation", "ephemeral_5m_input_tokens", "ephemeral_1h_input_tokens")
    )
    run.usage.update(
        requests=1, input_including_cache=prompt, cache_read=cache_read, cache_write=cache_write, output=output
    )
    if run.cost is not None:
        price = PRICES[model]
        long_price = LONG_PROMPT_PRICES.get(model)
        if long_price and prompt > long_price["over"]:
            price = long_price
        run.cost += (
            (prompt - cache_read - cache_write) * price["input"]
            + cache_write * price["cache_write"]
            + cache_read * price["cache_read"]
            + output * price["output"]
        ) / 1_000_000


def _log_tool_call(tool_log: Path, label: str, call: dict, message) -> None:
    failed = message.status == "error"
    text = str(message.text)
    entry = {
        "time": datetime.now(UTC).isoformat(timespec="seconds"),
        "agent": label,
        "tool": call.get("name"),
        "args": {key: value for key, value in call.get("args", {}).items() if key not in UNLOGGED_ARGUMENTS},
        "ok": not failed,
        "result_chars": len(text),
    }
    if failed:
        entry["error"] = text[:500]
    with tool_log.open("a", encoding="utf-8") as log:
        log.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _reads_from_start(call: dict, path: str | None) -> bool:
    """Whether a tool call reads a path from its first line, however the agent wrote the path."""
    args = call.get("args", {})
    if path is None or call.get("name") != "read_file" or args.get("offset"):
        return False
    try:
        return validate_path(args.get("file_path", "")) == validate_path(path)
    except (TypeError, ValueError):
        return False


def run_agent(
    model: str,
    effort: str | None,
    system: str,
    task: str,
    transcripts: Path,
    memory: Path,
    tool_log: Path,
    label: str,
    read_limit: int,
    must_read: str | None = None,
) -> AgentRun:
    """Run the agent until it stops calling tools, logging each tool call as it returns.

    must_read is a path the agent has to read from its first line, or the run counts as failed.
    A failure is returned in the result rather than raised, so the tokens spent before it are still counted.
    """
    run = AgentRun(cost=0.0 if model in PRICES else None)
    try:
        backend = CompositeBackend(
            default=ReadOnlyBackend(read_limit, root_dir=transcripts, virtual_mode=True),  # only hidden paths get this far
            routes={
                f"{UPLOADS_PATH}/": ReadOnlyBackend(read_limit, root_dir=transcripts, virtual_mode=True),
                f"{MEMORY_PATH}/": MemoryBackend(read_limit, root_dir=memory, virtual_mode=True),
            },
        )
        agent = create_agent(
            model=ChatAnthropic(model=model, max_tokens=64000, streaming=True, reasoning_effort=effort),
            tools=[],
            system_prompt=system,
            middleware=[
                # No eviction: the analyzer has to read the whole transcript.
                FilesystemMiddleware(
                    backend=backend,
                    tools=TOOLS,
                    tool_token_limit_before_evict=None,
                    permissions=PERMISSIONS,
                ),
                AnthropicPromptCachingMiddleware(),
            ],
        )
        messages, calls, has_read = [], {}, False
        inputs = {"messages": [{"role": "user", "content": task}]}
        # Each step yields the whole conversation so far; only the messages not yet seen are new.
        for state in agent.stream(inputs, {"recursion_limit": 1000, "run_name": label}, stream_mode="values"):
            for message in state["messages"][len(messages) :]:
                if message.type == "ai":
                    calls.update({call["id"]: call for call in message.tool_calls})
                    if message.usage_metadata:
                        _count_usage(run, model, message.usage_metadata)
                elif message.type == "tool":
                    call = calls.get(message.tool_call_id, {})
                    _log_tool_call(tool_log, label, call, message)
                    if _reads_from_start(call, must_read):
                        has_read = has_read or message.status != "error"
            messages = state["messages"]

        stop_reason = messages[-1].response_metadata.get("stop_reason")
        if stop_reason != "end_turn":
            run.error = f"agent stopped early: {stop_reason}"
        elif must_read and not has_read:
            run.error = f"agent finished without reading {must_read} from its first line"
        else:
            run.text = str(messages[-1].text)
    except Exception as error:
        run.error = f"{type(error).__name__}: {error}"
    return run
