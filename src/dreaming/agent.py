"""Run one dreaming agent. This is the only module that imports LangChain or DeepAgents."""

from collections import Counter
from pathlib import Path

from deepagents.backends import CompositeBackend, FilesystemBackend
from deepagents.middleware.filesystem import FilesystemMiddleware, FilesystemPermission
from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic
from langchain_anthropic.middleware import AnthropicPromptCachingMiddleware

# The official prompts hard-code the transcript path, so the tools expose the same virtual paths.
UPLOADS_PATH = "/mnt/session/uploads"
MEMORY_PATH = "/mnt/memory/dreaming"

# The official analyzer and organizer toolset: read, write, edit, glob and grep.
TOOLS = ["read_file", "write_file", "edit_file", "glob", "grep"]

# The first matching rule wins: transcripts are read-only, memory is read-write, everything else is denied.
PERMISSIONS = [
    FilesystemPermission(operations=["write"], paths=[f"{UPLOADS_PATH}/**"], mode="deny"),
    FilesystemPermission(operations=["read"], paths=[f"{UPLOADS_PATH}/**"], mode="allow"),
    FilesystemPermission(operations=["read", "write"], paths=[MEMORY_PATH, f"{MEMORY_PATH}/**"], mode="allow"),
    FilesystemPermission(operations=["read", "write"], paths=["/**"], mode="deny"),
]


def run_agent(model: str, system: str, task: str, transcripts: Path, memory: Path) -> tuple[str, Counter]:
    """Run the agent until it stops calling tools. Returns its final text and token usage."""
    backend = CompositeBackend(
        default=FilesystemBackend(root_dir=transcripts, virtual_mode=True),  # never reached: denied above
        routes={
            f"{UPLOADS_PATH}/": FilesystemBackend(root_dir=transcripts, virtual_mode=True),
            f"{MEMORY_PATH}/": FilesystemBackend(root_dir=memory, virtual_mode=True),
        },
    )
    agent = create_agent(
        model=ChatAnthropic(model=model, max_tokens=64000, streaming=True),
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
    result = agent.invoke({"messages": [{"role": "user", "content": task}]}, {"recursion_limit": 1000})

    final = result["messages"][-1]
    stop_reason = final.response_metadata.get("stop_reason")
    if stop_reason != "end_turn":
        raise RuntimeError(f"agent stopped early: {stop_reason}")

    usage = Counter()
    for message in result["messages"]:
        metadata = getattr(message, "usage_metadata", None) or {}
        details = metadata.get("input_token_details", {})
        usage["input_including_cache"] += metadata.get("input_tokens", 0)
        usage["output"] += metadata.get("output_tokens", 0)
        usage["cache_read"] += details.get("cache_read") or 0
        # Cache writes are reported either as one total or split by TTL, never both.
        for key in ("cache_creation", "ephemeral_5m_input_tokens", "ephemeral_1h_input_tokens"):
            usage["cache_write"] += details.get(key) or 0
    return final.text, usage
