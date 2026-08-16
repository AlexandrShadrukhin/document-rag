from __future__ import annotations

import platform
import tkinter as tk
from collections.abc import Callable, Iterator
from typing import Any

ClipboardHandler = Callable[[Any], str]


def _widget_class(widget: Any) -> str:
    return str(widget.winfo_class())


def _is_text(widget: Any) -> bool:
    return _widget_class(widget) == "Text"


def _is_entry(widget: Any) -> bool:
    return _widget_class(widget) in {"Entry", "TEntry", "TCombobox"}


def is_editable(widget: Any) -> bool:
    if _is_text(widget):
        return str(widget.cget("state")) != "disabled"
    if _is_entry(widget):
        if hasattr(widget, "instate"):
            return not widget.instate(("disabled", "readonly"))
        return str(widget.cget("state")) not in {"disabled", "readonly"}
    return False


def _selection(widget: Any) -> tuple[Any, Any] | None:
    if _is_text(widget):
        ranges = widget.tag_ranges("sel")
        return (ranges[0], ranges[1]) if len(ranges) == 2 else None
    if _is_entry(widget) and widget.selection_present():
        return widget.index("sel.first"), widget.index("sel.last")
    return None


def copy_selection(widget: Any) -> str:
    selection = _selection(widget)
    if selection is None:
        return "break"
    selected = widget.get(*selection)
    widget.clipboard_clear()
    widget.clipboard_append(selected)
    return "break"


def cut_selection(widget: Any) -> str:
    selection = _selection(widget)
    if selection is None or not is_editable(widget):
        return "break"
    copy_selection(widget)
    widget.delete(*selection)
    return "break"


def paste_clipboard(widget: Any) -> str:
    if not is_editable(widget):
        return "break"
    try:
        value = widget.clipboard_get()
    except tk.TclError:
        return "break"
    selection = _selection(widget)
    insert_at = selection[0] if selection is not None else widget.index("insert")
    if selection is not None:
        widget.delete(*selection)
    widget.insert(insert_at, value)
    if _is_text(widget):
        widget.mark_set("insert", f"{insert_at}+{len(value)}c")
        widget.see("insert")
    else:
        widget.icursor(int(insert_at) + len(value))
    widget.focus_set()
    return "break"


def select_all(widget: Any) -> str:
    if _is_text(widget):
        widget.tag_add("sel", "1.0", "end-1c")
        widget.mark_set("insert", "end-1c")
        widget.see("insert")
    elif _is_entry(widget):
        widget.selection_range(0, "end")
        widget.icursor("end")
    return "break"


def shortcut_sequences(system: str | None = None) -> dict[str, tuple[str, ...]]:
    current = system or platform.system()
    modifiers = ("Command", "Meta", "Mod1") if current == "Darwin" else ("Control",)
    keys = {"paste": "v", "copy": "c", "cut": "x", "select_all": "a"}
    return {
        action: tuple(
            f"<{modifier}-KeyPress-{letter}>"
            for modifier in modifiers
            for letter in (key, key.upper())
        )
        for action, key in keys.items()
    }


def install_clipboard_support(widget: Any, system: str | None = None) -> None:
    handlers: dict[str, ClipboardHandler] = {
        "paste": paste_clipboard,
        "copy": copy_selection,
        "cut": cut_selection,
        "select_all": select_all,
    }
    for action, sequences in shortcut_sequences(system).items():
        handler = handlers[action]
        for sequence in sequences:
            widget.bind(sequence, lambda event, callback=handler: callback(event.widget), add="+")

    menu = tk.Menu(widget, tearoff=False)
    menu.add_command(label="Cut", command=lambda: cut_selection(widget))
    menu.add_command(label="Copy", command=lambda: copy_selection(widget))
    menu.add_command(label="Paste", command=lambda: paste_clipboard(widget))
    menu.add_separator()
    menu.add_command(label="Select All", command=lambda: select_all(widget))

    def show_menu(event: Any) -> str:
        widget.focus_set()
        editable = is_editable(widget)
        menu.entryconfigure("Cut", state="normal" if editable else "disabled")
        menu.entryconfigure("Paste", state="normal" if editable else "disabled")
        menu.tk_popup(event.x_root, event.y_root)
        return "break"

    for sequence in ("<Button-2>", "<Button-3>", "<Control-Button-1>"):
        widget.bind(sequence, show_menu, add="+")
    widget._clipboard_context_menu = menu


def editable_widgets(root: Any) -> Iterator[Any]:
    for child in root.winfo_children():
        if _is_text(child) or _is_entry(child):
            yield child
        yield from editable_widgets(child)


def install_clipboard_support_tree(root: tk.Misc, system: str | None = None) -> None:
    for widget in editable_widgets(root):
        install_clipboard_support(widget, system)
