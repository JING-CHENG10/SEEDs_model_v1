# -*- coding: utf-8 -*-
"""Process-local and node-local caches for repeated tabular reads."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import hashlib
import os
import pickle
import sys
import time
import uuid
import pandas as pd

_EXCEL_CACHE: Dict[Tuple[Any, ...], pd.DataFrame] = {}
_CSV_CACHE: Dict[Tuple[Any, ...], pd.DataFrame] = {}
_PICKLE_CACHE: Dict[Tuple[Any, ...], pd.DataFrame] = {}
_CACHE_FORMAT_VERSION = 2


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple(sorted((str(k), _freeze(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    if isinstance(value, set):
        return tuple(sorted((_freeze(v) for v in value), key=repr))
    if callable(value):
        code = getattr(value, "__code__", None)
        code_hash = hashlib.sha256(code.co_code).hexdigest() if code is not None else ""
        return (
            "callable",
            getattr(value, "__module__", ""),
            getattr(value, "__qualname__", getattr(value, "__name__", "")),
            code_hash,
        )
    try:
        hash(value)
        return value
    except Exception:
        try:
            payload = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
            return ("pickle", hashlib.sha256(payload).hexdigest())
        except Exception:
            return ("repr", repr(value))


def _file_signature(path: str) -> Tuple[int, int]:
    st = os.stat(path)
    return int(st.st_mtime_ns), int(st.st_size)


def _contains_callable(value: Any) -> bool:
    if callable(value):
        return True
    if isinstance(value, dict):
        return any(_contains_callable(k) or _contains_callable(v) for k, v in value.items())
    if isinstance(value, (list, tuple, set)):
        return any(_contains_callable(v) for v in value)
    return False


def _persistent_cache_enabled() -> bool:
    raw = os.environ.get("NZF_PERSISTENT_CACHE", "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def get_runtime_cache_dir() -> Path:
    override = os.environ.get("NZF_RUNTIME_CACHE_DIR", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Caches"
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    return base / "nzf_model" / f"runtime_cache_v{_CACHE_FORMAT_VERSION}"


def _persistent_excel_cache_path(
    abs_path: str,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
) -> Path:
    payload = (
        _CACHE_FORMAT_VERSION,
        sys.version_info[:2],
        pd.__version__,
        abs_path,
        _file_signature(abs_path),
        _freeze(args),
        _freeze(kwargs),
    )
    digest = hashlib.sha256(
        pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    ).hexdigest()
    return get_runtime_cache_dir() / "excel" / f"{digest}.pkl"


def _read_pickle_entry(cache_path: Path) -> Tuple[bool, Any]:
    if not cache_path.exists():
        return False, None
    try:
        return True, pd.read_pickle(cache_path)
    except Exception:
        try:
            cache_path.unlink(missing_ok=True)
        except OSError:
            pass
        return False, None


def _write_pickle_atomic(value: Any, cache_path: Path) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_name(
        f".{cache_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        pd.to_pickle(value, tmp_path)
        os.replace(tmp_path, cache_path)
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass


def _lock_timeout_seconds() -> float:
    try:
        return max(1.0, float(os.environ.get("NZF_RUNTIME_CACHE_LOCK_TIMEOUT", "900")))
    except Exception:
        return 900.0


def _lock_stale_seconds() -> float:
    try:
        return max(60.0, float(os.environ.get("NZF_RUNTIME_CACHE_LOCK_STALE", "3600")))
    except Exception:
        return 3600.0


def _read_excel_persistent(
    abs_path: str,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
) -> Any:
    if not _persistent_cache_enabled() or _contains_callable((args, kwargs)):
        return pd.read_excel(abs_path, *args, **kwargs)

    cache_path = _persistent_excel_cache_path(abs_path, args, kwargs)
    found, value = _read_pickle_entry(cache_path)
    if found:
        return value

    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        return pd.read_excel(abs_path, *args, **kwargs)
    lock_path = cache_path.with_suffix(cache_path.suffix + ".lock")
    deadline = time.monotonic() + _lock_timeout_seconds()
    lock_fd: Optional[int] = None
    while lock_fd is None:
        try:
            candidate_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            try:
                os.write(
                    candidate_fd,
                    f"{os.getpid()} {time.time():.6f}".encode("ascii"),
                )
            except OSError:
                os.close(candidate_fd)
                try:
                    lock_path.unlink(missing_ok=True)
                except OSError:
                    pass
                return pd.read_excel(abs_path, *args, **kwargs)
            lock_fd = candidate_fd
        except FileExistsError:
            found, value = _read_pickle_entry(cache_path)
            if found:
                return value
            try:
                if time.time() - lock_path.stat().st_mtime > _lock_stale_seconds():
                    lock_path.unlink(missing_ok=True)
                    continue
            except OSError:
                continue
            if time.monotonic() >= deadline:
                return pd.read_excel(abs_path, *args, **kwargs)
            time.sleep(0.2)
        except OSError:
            return pd.read_excel(abs_path, *args, **kwargs)

    try:
        found, value = _read_pickle_entry(cache_path)
        if found:
            return value
        value = pd.read_excel(abs_path, *args, **kwargs)
        try:
            _write_pickle_atomic(value, cache_path)
        except Exception:
            pass
        return value
    finally:
        if lock_fd is not None:
            os.close(lock_fd)
        try:
            lock_path.unlink(missing_ok=True)
        except OSError:
            pass


def read_excel_cached(
    path: str,
    *args: Any,
    copy: bool = True,
    **kwargs: Any,
) -> Any:
    abs_path = str(Path(path).resolve())
    if _contains_callable((args, kwargs)):
        value = pd.read_excel(abs_path, *args, **kwargs)
    else:
        key = (
            "excel",
            abs_path,
            _file_signature(abs_path),
            _freeze(args),
            _freeze(kwargs),
        )
        if key not in _EXCEL_CACHE:
            _EXCEL_CACHE[key] = _read_excel_persistent(abs_path, args, kwargs)
        value = _EXCEL_CACHE[key]
    if not copy:
        return value
    if isinstance(value, pd.DataFrame):
        return value.copy(deep=True)
    if isinstance(value, dict):
        return {
            key: item.copy(deep=True) if isinstance(item, pd.DataFrame) else item
            for key, item in value.items()
        }
    return value


def read_csv_cached(path: str, *, copy: bool = True, **kwargs) -> pd.DataFrame:
    abs_path = str(Path(path).resolve())
    key = ("csv", abs_path, _file_signature(abs_path), _freeze(kwargs))
    if key not in _CSV_CACHE:
        _CSV_CACHE[key] = pd.read_csv(abs_path, **kwargs)
    df = _CSV_CACHE[key]
    return df.copy(deep=True) if copy else df


def read_pickle_cached(path: str, *, copy: bool = True, **kwargs) -> pd.DataFrame:
    abs_path = str(Path(path).resolve())
    key = ("pickle", abs_path, _file_signature(abs_path), _freeze(kwargs))
    if key not in _PICKLE_CACHE:
        _PICKLE_CACHE[key] = pd.read_pickle(abs_path, **kwargs)
    df = _PICKLE_CACHE[key]
    return df.copy(deep=True) if copy else df


def read_excel_sidecar_cached(
    path: str,
    *,
    sheet_name: Any = 0,
    copy: bool = True,
    **kwargs: Any,
) -> Any:
    """Backward-compatible name; cache files now live in the node-local cache directory."""
    return read_excel_cached(
        path,
        sheet_name=sheet_name,
        copy=copy,
        **kwargs,
    )


def read_tabular_cached(path: str, *, sheet_name: Any = 0, copy: bool = True, **kwargs) -> pd.DataFrame:
    abs_path = str(Path(path).resolve())
    suffix = Path(abs_path).suffix.lower()
    if suffix in {".csv", ".txt"}:
        return read_csv_cached(abs_path, copy=copy, **kwargs)
    if suffix in {".pkl", ".pickle"}:
        return read_pickle_cached(abs_path, copy=copy, **kwargs)
    if suffix in {".xlsx", ".xls", ".xlsm", ".xlsb"}:
        return read_excel_sidecar_cached(abs_path, sheet_name=sheet_name, copy=copy, **kwargs)
    raise ValueError(f"Unsupported tabular file type: {path}")


def clear_memory_caches() -> None:
    _EXCEL_CACHE.clear()
    _CSV_CACHE.clear()
    _PICKLE_CACHE.clear()
