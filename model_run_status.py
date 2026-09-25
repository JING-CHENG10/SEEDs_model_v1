# -*- coding: utf-8 -*-
"""Structured run-status helpers shared by the model pipeline and smoke tests.

The helpers deliberately avoid importing Gurobi so status artifacts remain
readable in lightweight validation environments.
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union


RUN_STATUS_FILENAME = "run_status.json"

_TERMINAL_SUCCESS_PIPELINE_STATUSES = {
    "completed",
    "completed_with_module_failures",
}

_EMISSION_MODULE_SUCCESS_STATUSES = {
    "completed",
    "completed_empty",
    "not_applicable",
}

_SOLVER_STATUS_NAMES = {
    1: "loaded",
    2: "optimal",
    3: "infeasible",
    4: "inf_or_unbd",
    5: "unbounded",
    6: "cutoff",
    7: "iteration_limit",
    8: "node_limit",
    9: "time_limit",
    10: "solution_limit",
    11: "interrupted",
    12: "numeric",
    13: "suboptimal",
    14: "in_progress",
    15: "user_obj_limit",
    16: "work_limit",
    17: "memory_limit",
}

_STATUS_ALIASES = {
    "inf_or_unbounded": "inf_or_unbd",
    "infeasible_or_unbounded": "inf_or_unbd",
    "infeasible_or_iis": "infeasible",
    "inprogress": "in_progress",
    "mem_limit": "memory_limit",
    "not run": "not_run",
    "not-run": "not_run",
    "user_objective_limit": "user_obj_limit",
}


PathLike = Union[str, os.PathLike[str], Path]


@dataclass(frozen=True)
class ResumeValidation:
    """Result of the strict provenance gate used before reusing run outputs."""

    allowed: bool
    reason: str
    payload: Optional[Dict[str, Any]] = None
    run_id: str = ""
    scenario_id: str = ""
    started_at_epoch: Optional[float] = None
    finished_at_epoch: Optional[float] = None
    manifest_path: Optional[Path] = None

    def __bool__(self) -> bool:
        return bool(self.allowed)


@dataclass(frozen=True)
class ArtifactIntegrityValidation:
    """Result of validating a declared output artifact against its contents."""

    allowed: bool
    reason: str
    path: Optional[Path] = None
    declaration: Optional[Dict[str, Any]] = None

    def __bool__(self) -> bool:
        return bool(self.allowed)


def build_resume_fingerprint(payload: Any) -> str:
    """Return a deterministic digest for a runner's effective scenario inputs.

    Callers should pass JSON-compatible primitives.  Rejecting a changed digest
    prevents an old directory with the same scenario/sample ID from being
    attached to a newly generated draw.
    """
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def solver_status_name(status: Any) -> str:
    """Return a stable lowercase solver status name."""
    if status is None:
        return "not_run"
    if isinstance(status, bool):
        return "unknown"
    try:
        code = int(status)
    except (TypeError, ValueError, OverflowError):
        text = str(status).strip().lower()
        if not text:
            return "unknown"
        text = text.replace("grb.", "").replace(" ", "_").replace("-", "_")
        return _STATUS_ALIASES.get(text, text)
    return _SOLVER_STATUS_NAMES.get(code, f"status_{code}")


def solver_status_has_solution(
    status: Any,
    *,
    sol_count: Optional[Any] = None,
    explicit: Optional[Any] = None,
) -> bool:
    """Determine whether a status artifact represents an available incumbent."""
    if explicit is not None:
        return bool(explicit)
    name = solver_status_name(status)
    if name == "optimal":
        return True
    try:
        count = int(sol_count) if sol_count is not None else 0
    except (TypeError, ValueError, OverflowError):
        count = 0
    return count > 0 and name in {
        "suboptimal",
        "time_limit",
        "solution_limit",
        "interrupted",
        "user_obj_limit",
        "work_limit",
        "memory_limit",
        "numeric",
    }


def solver_result_is_extractable(status: Any, sol_count: Optional[Any]) -> bool:
    """Return whether solver variable values may safely be read."""
    try:
        count = int(sol_count) if sol_count is not None else 0
    except (TypeError, ValueError, OverflowError):
        count = 0
    if count <= 0:
        return False
    return solver_status_name(status) in {
        "optimal",
        "suboptimal",
        "time_limit",
        "solution_limit",
        "interrupted",
        "user_obj_limit",
        "work_limit",
        "memory_limit",
        "numeric",
    }


def normalize_solver_status(
    status: Any,
    *,
    sol_count: Optional[Any] = None,
    has_solution: Optional[Any] = None,
    **metadata: Any,
) -> Dict[str, Any]:
    """Build the canonical solver block stored in ``run_status.json``."""
    name = solver_status_name(status)
    try:
        status_code = int(status) if status is not None and not isinstance(status, bool) else None
    except (TypeError, ValueError, OverflowError):
        status_code = None
    try:
        solution_count = int(sol_count) if sol_count is not None else None
    except (TypeError, ValueError, OverflowError):
        solution_count = None
    block: Dict[str, Any] = {
        "status_code": status_code,
        "status_name": name,
        "sol_count": solution_count,
        "has_solution": solver_status_has_solution(
            status,
            sol_count=solution_count,
            explicit=has_solution,
        ),
        "optimal": name == "optimal",
    }
    block.update({key: value for key, value in metadata.items() if value is not None})
    return block


def _status_path(path: PathLike) -> Path:
    candidate = Path(path)
    if candidate.name.lower() == RUN_STATUS_FILENAME:
        return candidate
    return candidate / RUN_STATUS_FILENAME


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_run_status(path: PathLike, payload: Mapping[str, Any]) -> Path:
    """Atomically write a structured run-status artifact."""
    out_path = _status_path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(f".{out_path.name}.tmp")
    text = json.dumps(
        dict(payload),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        default=_json_default,
    )
    tmp_path.write_text(text + "\n", encoding="utf-8")
    os.replace(tmp_path, out_path)
    return out_path


def read_run_status(path: PathLike) -> Optional[Dict[str, Any]]:
    """Read ``run_status.json``; return ``None`` for missing/invalid artifacts."""
    status_path = _status_path(path)
    if not status_path.exists():
        return None
    try:
        payload = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _optional_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n", ""}:
        return False
    return None


def _emission_module_records(
    payload: Optional[Mapping[str, Any]],
) -> Optional[List[Mapping[str, Any]]]:
    if not isinstance(payload, Mapping):
        return None
    emissions = payload.get("emissions")
    if not isinstance(emissions, Mapping):
        return None
    modules = emissions.get("modules")
    if isinstance(modules, Mapping):
        return [record for record in modules.values() if isinstance(record, Mapping)]
    if isinstance(modules, list):
        return [record for record in modules if isinstance(record, Mapping)]
    return None


def _module_records_complete(records: Optional[List[Mapping[str, Any]]]) -> Optional[bool]:
    if records is None:
        return None
    required_records = [
        record
        for record in records
        if _optional_bool(record.get("required")) is True
    ]
    if not required_records:
        return True
    return all(
        str(record.get("status", "")).strip().lower()
        in _EMISSION_MODULE_SUCCESS_STATUSES
        for record in required_records
    )


def emission_modules_complete(payload: Optional[Mapping[str, Any]]) -> Optional[bool]:
    """Extract emission-module completeness from a structured status payload."""
    if not isinstance(payload, Mapping):
        return None
    emissions = payload.get("emissions")
    if not isinstance(emissions, Mapping):
        return None
    for key in ("completeness", "modules_complete", "complete"):
        if key in emissions and emissions[key] is not None:
            return _optional_bool(emissions[key])
    return _module_records_complete(_emission_module_records(payload))


def _parse_utc_timestamp(value: Any) -> Optional[float]:
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = dt.datetime.fromisoformat(text)
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        return None
    return float(parsed.timestamp())


def _normalized_module_map(
    records: Optional[List[Mapping[str, Any]]],
) -> Optional[Dict[str, Mapping[str, Any]]]:
    if records is None:
        return None
    normalized: Dict[str, Mapping[str, Any]] = {}
    for record in records:
        name = str(record.get("module", "") or "").strip().upper()
        if not name or name in normalized:
            return None
        normalized[name] = record
    return normalized


def _rejected_resume(
    reason: str,
    *,
    payload: Optional[Dict[str, Any]] = None,
) -> ResumeValidation:
    return ResumeValidation(False, str(reason), payload=payload)


def validate_run_for_resume(
    path: PathLike,
    *,
    expected_scenario_id: Optional[str] = None,
    expected_resume_fingerprint: Optional[str] = None,
) -> ResumeValidation:
    """Validate the terminal status and emissions manifest before reuse.

    This is deliberately strict.  Missing or legacy status files, incomplete
    emissions metadata, and manifests from another run all make the output
    ineligible for automatic resume.
    """
    status_path = _status_path(path)
    run_dir = status_path.parent
    payload = read_run_status(status_path)
    if payload is None:
        return _rejected_resume("missing_or_invalid_run_status")

    try:
        schema_version = int(payload.get("schema_version"))
    except (TypeError, ValueError, OverflowError):
        schema_version = 0
    if schema_version < 1:
        return _rejected_resume("legacy_or_missing_status_schema", payload=payload)

    pipeline_status = str(payload.get("pipeline_status", "") or "").strip().lower()
    if pipeline_status not in _TERMINAL_SUCCESS_PIPELINE_STATUSES:
        return _rejected_resume(
            f"pipeline_not_terminal_success:{pipeline_status or 'missing'}",
            payload=payload,
        )

    run_id = str(payload.get("run_id", "") or "").strip()
    scenario_id = str(payload.get("scenario_id", "") or "").strip()
    if not run_id:
        return _rejected_resume("missing_run_id", payload=payload)
    if not scenario_id:
        return _rejected_resume("missing_scenario_id", payload=payload)
    if expected_scenario_id is not None and scenario_id != str(expected_scenario_id):
        return _rejected_resume("scenario_id_mismatch", payload=payload)
    if expected_resume_fingerprint is not None:
        run_config = payload.get("run_config")
        actual_fingerprint = (
            str(run_config.get("resume_fingerprint", "") or "").strip()
            if isinstance(run_config, Mapping)
            else ""
        )
        if not actual_fingerprint:
            return _rejected_resume("missing_resume_fingerprint", payload=payload)
        if actual_fingerprint != str(expected_resume_fingerprint).strip():
            return _rejected_resume("resume_fingerprint_mismatch", payload=payload)

    started_at_epoch = _parse_utc_timestamp(payload.get("started_at_utc"))
    finished_at_epoch = _parse_utc_timestamp(payload.get("finished_at_utc"))
    if started_at_epoch is None or finished_at_epoch is None:
        return _rejected_resume("missing_or_invalid_terminal_timestamps", payload=payload)
    if finished_at_epoch < started_at_epoch:
        return _rejected_resume("terminal_timestamp_precedes_start", payload=payload)

    solver = payload.get("solver")
    if not isinstance(solver, Mapping) or _optional_bool(solver.get("has_solution")) is not True:
        return _rejected_resume("solver_solution_not_confirmed", payload=payload)
    solver_name = solver_status_name(
        solver.get("status_code")
        if solver.get("status_code") is not None
        else solver.get("status_name")
    )
    if solver_name != "optimal":
        return _rejected_resume(
            f"solver_not_optimal:{solver_name or 'unknown'}",
            payload=payload,
        )

    error = payload.get("error")
    if isinstance(error, Mapping):
        error_present = any(str(value or "").strip() for value in error.values())
    else:
        error_present = bool(str(error or "").strip())
    if error_present:
        return _rejected_resume("terminal_status_contains_error", payload=payload)

    emissions = payload.get("emissions")
    if not isinstance(emissions, Mapping):
        return _rejected_resume("missing_emissions_status", payload=payload)
    if _optional_bool(emissions.get("completeness")) is not True:
        return _rejected_resume("emissions_not_declared_complete", payload=payload)
    status_records = _emission_module_records(payload)
    if _module_records_complete(status_records) is not True:
        return _rejected_resume("required_status_modules_incomplete", payload=payload)
    status_modules = _normalized_module_map(status_records)
    if not status_modules:
        return _rejected_resume("missing_or_invalid_status_modules", payload=payload)

    manifest_ref = str(emissions.get("manifest", "") or "").strip()
    if not manifest_ref:
        return _rejected_resume("missing_emissions_manifest_reference", payload=payload)
    manifest_path = Path(manifest_ref)
    if not manifest_path.is_absolute():
        manifest_path = run_dir / manifest_path
    try:
        run_dir_resolved = run_dir.resolve()
        manifest_path = manifest_path.resolve()
        manifest_path.relative_to(run_dir_resolved)
    except (OSError, RuntimeError, ValueError):
        return _rejected_resume("manifest_outside_run_directory", payload=payload)
    if not manifest_path.is_file() or manifest_path.stat().st_size <= 0:
        return _rejected_resume("missing_or_empty_emissions_manifest", payload=payload)
    # A manifest older than this run is necessarily stale.  The small tolerance
    # accommodates filesystems with coarse timestamp resolution.
    if manifest_path.stat().st_mtime + 2.0 < started_at_epoch:
        return _rejected_resume("stale_emissions_manifest", payload=payload)

    try:
        with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            manifest_rows = [dict(row) for row in reader]
    except (OSError, UnicodeError, csv.Error):
        return _rejected_resume("unreadable_emissions_manifest", payload=payload)
    required_columns = {"run_id", "scenario_id", "module", "required", "status"}
    if not manifest_rows or not required_columns.issubset(set(reader.fieldnames or [])):
        return _rejected_resume("invalid_emissions_manifest_schema", payload=payload)

    if {str(row.get("run_id", "") or "").strip() for row in manifest_rows} != {run_id}:
        return _rejected_resume("manifest_run_id_mismatch", payload=payload)
    if {str(row.get("scenario_id", "") or "").strip() for row in manifest_rows} != {scenario_id}:
        return _rejected_resume("manifest_scenario_id_mismatch", payload=payload)

    manifest_modules = _normalized_module_map(manifest_rows)
    if not manifest_modules or set(manifest_modules) != set(status_modules):
        return _rejected_resume("manifest_status_module_set_mismatch", payload=payload)
    for name, manifest_record in manifest_modules.items():
        status_record = status_modules[name]
        manifest_required = _optional_bool(manifest_record.get("required"))
        status_required = _optional_bool(status_record.get("required"))
        manifest_status = str(manifest_record.get("status", "") or "").strip().lower()
        status_status = str(status_record.get("status", "") or "").strip().lower()
        if manifest_required is None or status_required is None:
            return _rejected_resume("invalid_module_required_flag", payload=payload)
        if manifest_required != status_required or manifest_status != status_status:
            return _rejected_resume("manifest_status_module_mismatch", payload=payload)
        if manifest_required and manifest_status not in _EMISSION_MODULE_SUCCESS_STATUSES:
            return _rejected_resume("required_manifest_module_incomplete", payload=payload)

    return ResumeValidation(
        True,
        "ok",
        payload=payload,
        run_id=run_id,
        scenario_id=scenario_id,
        started_at_epoch=started_at_epoch,
        finished_at_epoch=finished_at_epoch,
        manifest_path=manifest_path,
    )


def artifact_matches_validated_run(
    path: PathLike,
    validation: ResumeValidation,
    *,
    require_nonempty: bool = True,
) -> bool:
    """Return whether an output exists and was created during the validated run."""
    if not validation.allowed or validation.started_at_epoch is None:
        return False
    artifact = Path(path)
    try:
        if not artifact.is_file():
            return False
        stat = artifact.stat()
    except OSError:
        return False
    if require_nonempty and stat.st_size <= 0:
        return False
    return stat.st_mtime + 2.0 >= validation.started_at_epoch


def _artifact_path_within_run(run_dir: PathLike, relative_path: Any) -> Path:
    """Resolve an artifact path while enforcing confinement to ``run_dir``."""
    root = Path(run_dir).resolve()
    candidate_ref = Path(str(relative_path or ""))
    if not str(relative_path or "").strip() or candidate_ref.is_absolute():
        raise ValueError("artifact path must be a non-empty relative path")
    try:
        candidate = (root / candidate_ref).resolve()
        candidate.relative_to(root)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError("artifact path escapes the run directory") from exc
    return candidate


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_required_columns(columns: Sequence[Any]) -> Tuple[str, ...]:
    normalized = tuple(str(column).strip() for column in columns)
    if any(not column for column in normalized):
        raise ValueError("required_columns must contain non-empty names")
    if len(set(normalized)) != len(normalized):
        raise ValueError("required_columns must not contain duplicates")
    return normalized


def _inspect_delimited_artifact(path: Path) -> Tuple[int, Tuple[str, ...]]:
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, delimiter=delimiter)
            columns = tuple(str(column or "").strip() for column in (reader.fieldnames or ()))
            if not columns or any(not column for column in columns):
                raise ValueError("artifact has a missing or invalid header")
            row_count = sum(1 for _ in reader)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ValueError("artifact is not a readable delimited table") from exc
    return row_count, columns


def build_artifact_integrity_declaration(
    run_dir: PathLike,
    artifact_path: PathLike,
    *,
    required_columns: Sequence[str] = (),
) -> Dict[str, Any]:
    """Describe a CSV/TSV output so later resume checks verify its contents.

    ``artifact_path`` may be absolute or relative when creating the declaration,
    but its resolved target must remain within ``run_dir``.  The stored path is
    always POSIX-style and relative, making the declaration relocatable with the
    run directory.
    """
    root = Path(run_dir).resolve()
    supplied_path = Path(artifact_path)
    candidate = supplied_path.resolve() if supplied_path.is_absolute() else (root / supplied_path).resolve()
    try:
        relative_path = candidate.relative_to(root)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError("artifact path escapes the run directory") from exc
    if candidate.suffix.lower() not in {".csv", ".tsv"}:
        raise ValueError("artifact integrity declarations currently require CSV or TSV")
    if not candidate.is_file():
        raise FileNotFoundError(candidate)

    normalized_required = _normalize_required_columns(required_columns)
    row_count, actual_columns = _inspect_delimited_artifact(candidate)
    missing = [column for column in normalized_required if column not in actual_columns]
    if missing:
        raise ValueError(f"artifact is missing required columns: {', '.join(missing)}")
    stat = candidate.stat()
    return {
        "relative_path": relative_path.as_posix(),
        "bytes": int(stat.st_size),
        "sha256": _file_sha256(candidate),
        "row_count": int(row_count),
        "required_columns": list(normalized_required),
    }


def _rejected_artifact(
    reason: str,
    *,
    declaration: Optional[Mapping[str, Any]] = None,
    path: Optional[Path] = None,
) -> ArtifactIntegrityValidation:
    return ArtifactIntegrityValidation(
        False,
        str(reason),
        path=path,
        declaration=dict(declaration) if isinstance(declaration, Mapping) else None,
    )


def validate_artifact_integrity_declaration(
    run_dir: PathLike,
    declaration: Mapping[str, Any],
    validation: ResumeValidation,
) -> ArtifactIntegrityValidation:
    """Strictly validate a declared CSV/TSV output for automatic resume."""
    if not isinstance(declaration, Mapping):
        return _rejected_artifact("invalid_artifact_declaration")
    if not validation.allowed:
        return _rejected_artifact(
            "run_not_validated_for_resume",
            declaration=declaration,
        )

    try:
        artifact = _artifact_path_within_run(run_dir, declaration.get("relative_path"))
    except ValueError:
        return _rejected_artifact("artifact_outside_run_directory", declaration=declaration)
    if artifact.suffix.lower() not in {".csv", ".tsv"}:
        return _rejected_artifact(
            "unsupported_artifact_format",
            declaration=declaration,
            path=artifact,
        )
    if not artifact_matches_validated_run(artifact, validation, require_nonempty=False):
        return _rejected_artifact(
            "artifact_missing_empty_or_stale",
            declaration=declaration,
            path=artifact,
        )

    expected_bytes = declaration.get("bytes")
    expected_rows = declaration.get("row_count")
    expected_sha256 = str(declaration.get("sha256", "") or "").strip().lower()
    try:
        if isinstance(expected_bytes, bool) or int(expected_bytes) < 0:
            raise ValueError
        expected_bytes = int(expected_bytes)
        if isinstance(expected_rows, bool) or int(expected_rows) < 0:
            raise ValueError
        expected_rows = int(expected_rows)
        required_columns_value = declaration.get("required_columns")
        if not isinstance(required_columns_value, (list, tuple)):
            raise ValueError
        required_columns = _normalize_required_columns(required_columns_value)
    except (TypeError, ValueError, OverflowError):
        return _rejected_artifact(
            "invalid_artifact_declaration",
            declaration=declaration,
            path=artifact,
        )
    if len(expected_sha256) != 64 or any(char not in "0123456789abcdef" for char in expected_sha256):
        return _rejected_artifact(
            "invalid_artifact_declaration",
            declaration=declaration,
            path=artifact,
        )

    try:
        actual_bytes = artifact.stat().st_size
    except OSError:
        return _rejected_artifact("unreadable_artifact", declaration=declaration, path=artifact)
    if actual_bytes != expected_bytes:
        return _rejected_artifact("artifact_byte_count_mismatch", declaration=declaration, path=artifact)
    try:
        actual_sha256 = _file_sha256(artifact)
    except OSError:
        return _rejected_artifact("unreadable_artifact", declaration=declaration, path=artifact)
    if actual_sha256 != expected_sha256:
        return _rejected_artifact("artifact_sha256_mismatch", declaration=declaration, path=artifact)

    try:
        actual_rows, actual_columns = _inspect_delimited_artifact(artifact)
    except ValueError:
        return _rejected_artifact("unreadable_artifact", declaration=declaration, path=artifact)
    if actual_rows != expected_rows:
        return _rejected_artifact("artifact_row_count_mismatch", declaration=declaration, path=artifact)
    if any(column not in actual_columns for column in required_columns):
        return _rejected_artifact("artifact_required_columns_missing", declaration=declaration, path=artifact)

    return ArtifactIntegrityValidation(
        True,
        "ok",
        path=artifact,
        declaration=dict(declaration),
    )
