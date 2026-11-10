"""Plain-text transcript rendering.

Copied unchanged from anthropics/claude-cookbooks, managed_agents/memory_distill_utils.py (MIT License,
Copyright (c) 2023 Anthropic), so transcripts match the format the official analyzer prompt describes.
"""

import json
import re

# Nothing taken from a session may pass for a label in the transcript file: a value on a label's own line is
# collapsed to one line, and a body line that starts like a label is indented by one space (the file's head
# then says so).
_TEXT_LABEL = re.compile(
    r"(SESSION |NOTE ABOUT THIS FILE:|=== |USER:|AGENT:|TOOL CALL |TOOL RESULT \(|THREAD MESSAGE |EVENT )"
)
_INDENTED_LABEL = re.compile(r" " + _TEXT_LABEL.pattern)
INDENTED_LINE_NOTE = (
    "NOTE ABOUT THIS FILE: A line indented by one space that starts like a label is part of the text above "
    "it, not a label."
)
# With ensure_ascii=False, json.dumps leaves these line-breaking code points raw (U+0085, U+2028, U+2029);
# escaped, a value written as one line of JSON stays one line for any reader.
_JSON_LINE_BREAKS = {0x85: "\\u0085", 0x2028: "\\u2028", 0x2029: "\\u2029"}


def _compact_json(value):
    """value as one line of JSON."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).translate(_JSON_LINE_BREAKS)


def _event_type(ev):
    """An event's type in dotted spelling; an underscored type from another exporter ("agent_tool_use") gives
    the dotted one ("agent.tool_use")."""
    kind = str(ev.get("type") or "")
    if "." in kind or "_" not in kind:
        return kind
    return kind.replace("_", ".", 1)


def _body(text):
    # splitlines also breaks at \r, \v, \f, U+0085, U+2028, U+2029 and the other code points a reader may
    # treat as a line break, so a label after any of them is indented too; each is written back as \n.
    return "\n".join(
        " " + line if _TEXT_LABEL.match(line) else line for line in (text.splitlines() or [""])
    )


def _one_line(value):
    """A value for a label line: its text with every whitespace run made a single space, so that nothing
    taken from the transcript can start a new line inside a label."""
    return " ".join(str(value).split())


def _tool_label(ev):
    """A tool call's name as the text labels it: server/name for an MCP tool."""
    name = _one_line(ev.get("name") or "") or "unknown tool"
    server = _one_line(ev.get("mcp_server_name") or ev.get("server_name") or "")
    return f"{server}/{name}" if server else name


def _blocks_text(content):
    """The text of a message's or tool result's content. A string stays as it is and None gives "". A block
    or a list gives each string or text block's text, a text document's title and data, a search result's
    title, source and text, and any other item as one line of JSON, joined by newlines."""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        content = [content]
    if not isinstance(content, list):
        return "" if content is None else _compact_json(content)
    parts = []
    for block in content:
        kind = block.get("type", "text") if isinstance(block, dict) else None
        source = block.get("source") if isinstance(block, dict) else None
        if isinstance(block, str):
            parts.append(block)
        elif kind == "text" and isinstance(block.get("text"), str):
            parts.append(block["text"])
        elif (
            kind == "document"
            and isinstance(source, dict)
            and source.get("type") == "text"
            and isinstance(source.get("data"), str)
        ):
            title = block.get("title")
            parts.append((f"[document: {title}]\n" if title else "") + source["data"])
        elif kind == "search_result":
            inner = _blocks_text(block.get("content"))
            head = " ".join(str(block[k]) for k in ("title", "source") if block.get(k))
            parts.append((f"[search result: {head}]\n" if head else "") + inner)
        else:
            parts.append(_compact_json(block))
    return "\n".join(parts)


def render_text(export):
    """One session's export (export_session's dict) as the plain text of its transcript file: a line for the
    session, a header per thread and a labeled block per event, separated by blank lines. An event with no
    label of its own keeps its other fields as one line of JSON. Texts are written as they are, except that
    every kind of line break becomes a newline and a line that would read as a label is indented by one
    space."""
    session = export.get("session") if isinstance(export.get("session"), dict) else {}
    agent = export.get("agent") if isinstance(export.get("agent"), dict) else {}
    about = "; ".join(
        f"{label} {value}"
        for label, value in (
            ("agent", agent.get("name")),
            ("created", session.get("created_at")),
        )
        if value
    )
    title = _one_line(session.get("title") or "")
    out = [
        f"SESSION {_one_line(session.get('id') or '?')}: {title}".rstrip()
        + (f" ({_one_line(about)})" if about else "")
    ]
    if export.get("export_note"):
        out.append(f"NOTE ABOUT THIS FILE: {_one_line(export['export_note'])}")
    threads = export.get("threads")
    for thread in threads if isinstance(threads, list) else []:
        thread = thread if isinstance(thread, dict) else {}
        tid = _one_line(thread.get("id") or "")
        tid = "" if tid in ("", "primary") else f" {tid}"
        header = "primary thread" if thread.get("parent_thread_id") is None else "sub-thread"
        agent_name = _one_line(thread.get("agent_name") or "") or "agent"
        out.append(f"=== {header}{tid}: {agent_name} ===")
        names = {}  # tool call id: tool name, for the results' labels
        events = thread.get("events")
        for ev in events if isinstance(events, list) else []:
            if not isinstance(ev, dict):
                out.append(f"EVENT {_compact_json(ev)}")
                continue
            kind = _one_line(_event_type(ev))
            if kind == "user.message":
                out.append("USER:\n" + _body(_blocks_text(ev.get("content"))))
            elif kind == "agent.message":
                out.append("AGENT:\n" + _body(_blocks_text(ev.get("content"))))
            elif kind.endswith("tool_use"):
                name = _tool_label(ev)
                if ev.get("id") is not None:
                    names[str(ev["id"])] = name
                fields = ev.get("input")
                if isinstance(fields, dict):
                    body = "\n".join(
                        f"{k}:\n{v}"
                        if isinstance(v, str) and "\n" in v
                        else f"{k}: {v if isinstance(v, str) else _compact_json(v)}"
                        for k, v in fields.items()
                    )
                else:
                    body = _blocks_text(fields)
                out.append(f"TOOL CALL {name}:\n{_body(body)}")
            elif kind.endswith("tool_result"):
                ref = (
                    ev.get("tool_use_id")
                    or ev.get("mcp_tool_use_id")
                    or ev.get("custom_tool_use_id")
                )
                name = names.get(str(ref), "unknown tool") if ref is not None else "unknown tool"
                flag = " [error]" if ev.get("is_error") else ""
                out.append(
                    f"TOOL RESULT ({name}){flag}:\n" + _body(_blocks_text(ev.get("content")))
                )
            elif kind == "agent.thread_message_received":
                who = (
                    ev.get("from_agent_name")
                    or ev.get("from_display_name")
                    or ev.get("from_session_thread_id")
                )
                label = (
                    f"THREAD MESSAGE RECEIVED from {_one_line(who)}"
                    if who
                    else "THREAD MESSAGE RECEIVED"
                )
                out.append(f"{label}:\n" + _body(_blocks_text(ev.get("content"))))
            elif kind == "agent.thread_message_sent":
                who = (
                    ev.get("to_agent_name")
                    or ev.get("to_display_name")
                    or ev.get("to_session_thread_id")
                )
                label = f"THREAD MESSAGE SENT to {_one_line(who)}" if who else "THREAD MESSAGE SENT"
                out.append(f"{label}:\n" + _body(_blocks_text(ev.get("content"))))
            else:
                rest = {k: v for k, v in ev.items() if k not in ("type", "id")}
                out.append(f"EVENT {kind}: {_compact_json(rest)}" if rest else f"EVENT {kind}")
    text = "\n\n".join(out) + "\n"
    if any(_INDENTED_LABEL.match(line) for line in text.splitlines()):
        out.insert(1, INDENTED_LINE_NOTE)
        text = "\n\n".join(out) + "\n"
    return text
