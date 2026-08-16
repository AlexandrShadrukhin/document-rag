from __future__ import annotations

from pathlib import Path
from typing import Any

import app.lab.environment as environment_module
from app.lab.environment import collect_environment


def test_platform_collector_returns_serializable_snapshot(
    tmp_path: Path, monkeypatch: Any
) -> None:
    monkeypatch.setattr(
        environment_module,
        "_torch_info",
        lambda: ("test-torch", False, None, None, True),
    )
    monkeypatch.setattr(
        environment_module,
        "ollama_status",
        lambda base_url: (True, ["qwen3:4b-instruct"]),
    )
    snapshot = collect_environment("http://localhost:11434", tmp_path)
    payload = snapshot.as_dict()
    assert payload["os"]
    assert payload["python_version"]
    assert payload["logical_cpu_cores"]
    assert payload["total_ram_bytes"] > 0
    assert payload["disk_free_bytes"] > 0
    assert payload["pytorch_version"] == "test-torch"
    assert payload["mps_available"] is True
    assert payload["ollama_models"] == ["qwen3:4b-instruct"]

