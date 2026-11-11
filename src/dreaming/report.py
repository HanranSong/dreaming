"""Write one run's report as Markdown."""

from collections import Counter
from pathlib import Path

USAGE_COLUMNS = {
    "Requests": "requests",
    "Input incl. cache": "input_including_cache",
    "Cache read": "cache_read",
    "Cache write": "cache_write",
    "Output": "output",
    "Compactions": "compactions",
}


def _window(tokens: int | None) -> str:
    return "the model's own" if tokens is None else f"{tokens:,} tokens"


def usd(cost: float | None) -> str:
    return "n/a" if cost is None else f"${cost:.3f}"


def _total_cost(agents: list) -> float | None:
    costs = [run.cost for _, run in agents]
    return None if None in costs else sum(costs)


def _quote(text: str) -> list[str]:
    return [f"> {line}".rstrip() for line in text.strip().splitlines()]


def _usage_row(label: str, usage: Counter, cost: float | None) -> str:
    numbers = " | ".join(f"{usage[key]:,}" for key in USAGE_COLUMNS.values())
    return f"| {label} | {numbers} | {usd(cost)} |"


def write_report(path: Path, record: dict) -> None:
    """Write the run as it stands. Called after every step, so a killed run still leaves a truthful report."""
    sessions, organizer, agents = record["sessions"], record["organizer"], record["agents"]
    counts = Counter(session["status"] for session in sessions)
    models = ", ".join(
        f"{agent} {model} (effort: {effort or 'API default'})" for agent, (model, effort) in record["models"].items()
    )
    compaction = record["compaction"]
    lines = [
        f"# Dreaming run {record['id']}",
        "",
        f"- Project: {record['project']}",
        f"- Models: {models}",
        f"- Compaction: at {compaction['percent']}% of the context window "
        f"(analyzer: {_window(compaction['analyzer'])}, organizer: {_window(compaction['organizer'])})",
        f"- Started: {record['started']}",
        f"- Finished: {record['finished'] or 'no: still running, or the run was interrupted'}",
        "- Sessions: " + (", ".join(f"{count} {status}" for status, count in counts.items()) or "none"),
        f"- Organizer: {organizer['status']}",
        f"- Estimated cost: {usd(_total_cost(agents))} at list price",
        "- Also in this folder: tool_calls.jsonl (each tool call an agent made), memory_before/ "
        "(memory as it was before this run, kept for the latest runs only) and, if an agent's context "
        "was compacted, conversation_history/ (the messages compaction replaced) and large_tool_results/ "
        "(tool results moved out of a request that was too long)",
        "",
        "## Sessions",
    ]
    for session in sessions:
        lines += ["", f"### {session['id']}: {session['status']}"]
        if session.get("title"):
            lines.append(f"- Title: {' '.join(session['title'].split())}")
        if session.get("previous_error"):
            lines.append(f"- Earlier attempt: {session['previous_error']}")
        if session.get("error"):
            lines.append(f"- Error: {session['error']}")
        if session.get("note"):
            lines.append(f"- {session['note']}")
        if session.get("summary"):
            lines += ["", *_quote(session["summary"])]
    if not sessions:
        lines += ["", "No session was analyzed in this run."]

    if record["not_run"]:
        lines += ["", "## Not run", ""]
        lines += [f"- {session_id}: {reason}" for session_id, reason in record["not_run"]]

    lines += ["", f"## Organizer: {organizer['status']}"]
    if organizer.get("error"):
        lines.append(f"- Error: {organizer['error']}")
    if organizer.get("note"):
        lines.append(f"- {organizer['note']}")
    if organizer.get("result"):
        lines += ["", *_quote(organizer["result"])]

    changes = [
        f"- {kind.capitalize()}: {', '.join(names)}" for kind, names in record["memory_changes"].items() if names
    ]
    lines += ["", "## Memory changes", "", *(changes or ["No file in memory/ changed."])]

    lines += ["", "## Usage", ""]
    if agents:
        lines.append("| Agent | " + " | ".join(USAGE_COLUMNS) + " | Est. cost |")
        lines.append("|---|" + "---:|" * (len(USAGE_COLUMNS) + 1))
        total = Counter()
        for label, run in agents:
            total.update(run.usage)
            lines.append(_usage_row(label, run.usage, run.cost))
        lines.append(_usage_row("Total", total, _total_cost(agents)))
    else:
        lines.append("No agent ran.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
