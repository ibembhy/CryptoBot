from __future__ import annotations

import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any


def write_json_atomic(path: str | Path, payload: Any, *, indent: int = 2) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", dir=target.parent, delete=False) as handle:
        temp_path = Path(handle.name)
        json.dump(payload, handle, indent=indent)
        handle.flush()
    temp_path.replace(target)


def read_json_strict(
    path: str | Path,
    *,
    missing_default: Any = None,
    label: str = "JSON file",
) -> Any:
    target = Path(path)
    if not target.exists():
        return missing_default
    try:
        return json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{label} is unreadable or corrupt: {target}") from exc
