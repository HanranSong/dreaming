"""Session logs built by hand, in the shapes Claude Code and Codex write them, under a throwaway home folder."""

import json
from pathlib import Path

import pytest

from dreaming import codex, formats

PROJECT = Path("/work/shop")


def write_jsonl(path: Path, records: list[dict], *, partial_line: str = "") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record) + "\n" for record in records) + partial_line, encoding="utf-8")
    return path


@pytest.fixture(autouse=True)
def no_notes_from_other_tests():
    """Format notes are kept for a whole run, so each test starts with none."""
    formats.clear()


@pytest.fixture
def home(tmp_path, monkeypatch) -> Path:
    """The home folder Claude Code's logs are looked up under."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    return tmp_path


@pytest.fixture
def codex_home(tmp_path, monkeypatch) -> Path:
    """Codex's folder. The reader lists its logs once per run, so that list is dropped around each test."""
    folder = tmp_path / "codex"
    folder.mkdir()
    monkeypatch.setenv("CODEX_HOME", str(folder))
    codex._threads.cache_clear()
    yield folder
    codex._threads.cache_clear()
