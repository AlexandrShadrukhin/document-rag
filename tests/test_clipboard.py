from __future__ import annotations

import re

from app.desktop.clipboard import (
    copy_selection,
    cut_selection,
    paste_clipboard,
    select_all,
    shortcut_sequences,
)


class FakeEntry:
    def __init__(self, text: str = "", state: str = "normal") -> None:
        self.text = text
        self.state = state
        self.cursor = len(text)
        self.selection: tuple[int, int] | None = None
        self.clipboard = ""
        self.focused = False

    def winfo_class(self) -> str:
        return "TEntry"

    def cget(self, name: str) -> str:
        assert name == "state"
        return self.state

    def instate(self, states: tuple[str, ...]) -> bool:
        return self.state in states

    def selection_present(self) -> bool:
        return self.selection is not None

    def index(self, index: str | int) -> int:
        if isinstance(index, int):
            return index
        if index == "insert":
            return self.cursor
        if index == "end":
            return len(self.text)
        if index == "sel.first" and self.selection:
            return self.selection[0]
        if index == "sel.last" and self.selection:
            return self.selection[1]
        raise ValueError(index)

    def get(self, start: int, end: int) -> str:
        return self.text[start:end]

    def delete(self, start: int, end: int) -> None:
        self.text = self.text[:start] + self.text[end:]
        self.cursor = start
        self.selection = None

    def insert(self, index: int | str, value: str) -> None:
        position = self.index(index)
        self.text = self.text[:position] + value + self.text[position:]

    def icursor(self, index: int | str) -> None:
        self.cursor = self.index(index)

    def selection_range(self, start: int, end: str) -> None:
        self.selection = (start, self.index(end))

    def clipboard_get(self) -> str:
        return self.clipboard

    def clipboard_clear(self) -> None:
        self.clipboard = ""

    def clipboard_append(self, value: str) -> None:
        self.clipboard += value

    def focus_set(self) -> None:
        self.focused = True


class FakeText:
    def __init__(self, text: str = "", state: str = "normal") -> None:
        self.text = text
        self.state = state
        self.cursor = len(text)
        self.selection: tuple[int, int] | None = None
        self.clipboard = ""
        self.focused = False

    def winfo_class(self) -> str:
        return "Text"

    def cget(self, name: str) -> str:
        assert name == "state"
        return self.state

    def _offset(self, index: object) -> int:
        value = str(index)
        if value == "insert":
            return self.cursor
        if value == "end-1c":
            return len(self.text)
        if value.startswith("1.") and "+" not in value:
            return int(value.split(".", 1)[1])
        match = re.fullmatch(r"1\.(\d+)\+(\d+)c", value)
        if match:
            return int(match.group(1)) + int(match.group(2))
        raise ValueError(value)

    def tag_ranges(self, name: str) -> tuple[str, str] | tuple[()]:
        assert name == "sel"
        if self.selection is None:
            return ()
        return f"1.{self.selection[0]}", f"1.{self.selection[1]}"

    def get(self, start: object, end: object) -> str:
        return self.text[self._offset(start) : self._offset(end)]

    def delete(self, start: object, end: object) -> None:
        first, last = self._offset(start), self._offset(end)
        self.text = self.text[:first] + self.text[last:]
        self.cursor = first
        self.selection = None

    def insert(self, index: object, value: str) -> None:
        position = self._offset(index)
        self.text = self.text[:position] + value + self.text[position:]

    def mark_set(self, name: str, index: object) -> None:
        assert name == "insert"
        self.cursor = self._offset(index)

    def tag_add(self, name: str, start: object, end: object) -> None:
        assert name == "sel"
        self.selection = self._offset(start), self._offset(end)

    def see(self, index: object) -> None:
        self._offset(index)

    def index(self, index: str) -> str:
        if index == "insert":
            return f"1.{self.cursor}"
        raise ValueError(index)

    def clipboard_get(self) -> str:
        return self.clipboard

    def clipboard_clear(self) -> None:
        self.clipboard = ""

    def clipboard_append(self, value: str) -> None:
        self.clipboard += value

    def focus_set(self) -> None:
        self.focused = True


def test_paste_into_empty_entry() -> None:
    widget = FakeEntry()
    widget.clipboard = "вставка"
    assert paste_clipboard(widget) == "break"
    assert widget.text == "вставка"
    assert widget.cursor == len("вставка")
    assert widget.focused is True


def test_paste_replaces_entry_selection() -> None:
    widget = FakeEntry("до старое после")
    widget.selection = (3, 9)
    widget.cursor = len(widget.text)
    widget.clipboard = "новое"
    paste_clipboard(widget)
    assert widget.text == "до новое после"
    assert widget.cursor == 3 + len("новое")


def test_paste_into_text_at_cursor() -> None:
    widget = FakeText("Начало конец")
    widget.cursor = 7
    widget.clipboard = "середина "
    paste_clipboard(widget)
    assert widget.text == "Начало середина конец"
    assert widget.cursor == 7 + len("середина ")


def test_select_all_entry_and_text() -> None:
    entry = FakeEntry("entry")
    text = FakeText("text")
    assert select_all(entry) == "break"
    assert select_all(text) == "break"
    assert entry.selection == (0, 5)
    assert text.selection == (0, 4)


def test_read_only_widget_rejects_paste_and_cut() -> None:
    widget = FakeText("read only", state="disabled")
    widget.selection = (0, 4)
    widget.clipboard = "replacement"
    assert paste_clipboard(widget) == "break"
    assert cut_selection(widget) == "break"
    assert widget.text == "read only"
    assert widget.clipboard == "replacement"


def test_read_only_widget_allows_select_all_and_copy() -> None:
    widget = FakeText("read only", state="disabled")
    select_all(widget)
    assert copy_selection(widget) == "break"
    assert widget.clipboard == "read only"
    assert widget.text == "read only"


def test_platform_shortcuts_are_explicit() -> None:
    mac = shortcut_sequences("Darwin")
    other = shortcut_sequences("Windows")
    assert "<Command-KeyPress-v>" in mac["paste"]
    assert "<Command-KeyPress-V>" in mac["paste"]
    assert "<Meta-KeyPress-v>" in mac["paste"]
    assert "<Control-KeyPress-v>" in other["paste"]
    assert "<Control-KeyPress-V>" in other["paste"]
