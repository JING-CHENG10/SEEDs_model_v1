from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


def _resolve_env_path(name: str) -> Optional[Path]:
    raw = os.environ.get(name)
    if not raw:
        return None
    return Path(raw).expanduser().resolve()


def get_code_root() -> str:
    """Return the project Code root, dynamically located from this file.

    Override with `NZF_CODE_ROOT` when running from a non-standard layout.
    """
    env = _resolve_env_path("NZF_CODE_ROOT")
    if env:
        return str(env)

    here = Path(__file__).resolve()
    for parent in here.parents:
        if parent.name.lower() == "code":
            return str(parent)
        if (parent / "input").is_dir() and (parent / "src").is_dir():
            return str(parent)

    # Repository layout is normally Code/bin/new/config_paths.py.
    return str(here.parents[2])


def get_input_base() -> str:
    """Return the base directory for FAOSTAT and runtime input files.

    Order of precedence:
    1) Environment variable `NZF_INPUT_DIR`
    2) `<Code>/input`, where `<Code>` is located from this file
    """
    env = _resolve_env_path("NZF_INPUT_DIR")
    if env:
        return str(env)
    return str(Path(get_code_root()) / "input")


def get_src_base() -> str:
    """Return the base directory for configuration/dictionary files (Code/src).

    Order of precedence:
    1) Environment variable `NZF_SRC_DIR`
    2) `<Code>/src`, where `<Code>` is located from this file
    """
    env = _resolve_env_path("NZF_SRC_DIR")
    if env:
        return str(env)
    return str(Path(get_code_root()) / "src")


def get_results_base(scenario: Optional[str] = None) -> str:
    r"""Return the output directory, optionally for a specific scenario.

    Preference order:
      1) Environment variable `NZF_OUTPUT_DIR`
      2) `<Code>/output`, where `<Code>` is located from this file
    """
    env = _resolve_env_path("NZF_OUTPUT_DIR")
    base = env if env else Path(get_code_root()) / "output"
    if scenario:
        base = base / scenario
    return str(base)


def get_luh2_data_base() -> str:
    """Return the LUH2 data directory used by land preprocessing scripts."""
    env = _resolve_env_path("NZF_LUH2_DIR")
    if env:
        return str(env)
    return str(Path(get_input_base()) / "Land" / "LUH2_GCB2019" / "data")
