from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DeviceSelection:
    requested: str
    selected: str
    fallback_reason: str | None = None


def _device_works(torch_module: object, device: str) -> tuple[bool, str | None]:
    try:
        tensor = torch_module.ones(1, device=device)  # type: ignore[attr-defined]
        _ = tensor.cpu()
        return True, None
    except Exception as error:
        return False, str(error)


def resolve_torch_device(requested: str | None) -> DeviceSelection:
    requested_name = (requested or "auto").lower()
    if requested_name not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError("device must be one of: auto, cpu, cuda, mps")
    if requested_name == "cpu":
        return DeviceSelection(requested="cpu", selected="cpu")

    try:
        import torch
    except Exception as error:
        reason = f"PyTorch unavailable: {error}"
        logger.warning("Falling back to CPU: %s", reason)
        return DeviceSelection(requested=requested_name, selected="cpu", fallback_reason=reason)

    candidates = [requested_name] if requested_name != "auto" else ["cuda", "mps"]
    failures: list[str] = []
    for candidate in candidates:
        if candidate == "cuda" and not torch.cuda.is_available():
            failures.append("CUDA is not available to the current PyTorch build")
            continue
        if candidate == "mps":
            backend = getattr(torch.backends, "mps", None)
            if backend is None or not backend.is_available():
                failures.append("MPS is not available to the current PyTorch build")
                continue
        works, error = _device_works(torch, candidate)
        if works:
            selection = DeviceSelection(requested=requested_name, selected=candidate)
            logger.info("Selected ML device: %s (requested: %s)", candidate, requested_name)
            return selection
        failures.append(f"{candidate.upper()} runtime check failed: {error}")

    reason = "; ".join(failures) or "No accelerator is available"
    logger.warning("Falling back to CPU for requested device '%s': %s", requested_name, reason)
    return DeviceSelection(requested=requested_name, selected="cpu", fallback_reason=reason)
