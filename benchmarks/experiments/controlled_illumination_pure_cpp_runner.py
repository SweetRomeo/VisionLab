from __future__ import annotations

from collections.abc import Callable, Mapping
import csv
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

from benchmarks.experiments.controlled_illumination_run_artifacts import (
    write_completed_run_artifacts_atomic,
)
from benchmarks.experiments.controlled_illumination_runner_context import (
    load_runner_context_from_environment,
)
from benchmarks.realtime.realtime_config import (
    load_realtime_config,
)
from benchmarks.realtime.realtime_records import (
    FrameStatus,
    RealtimeFrameRecord,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PURE_CPP_ARCHITECTURE = "pure_cpp"

EXECUTABLE_ENVIRONMENT_VARIABLE = (
    "VISIONLAB_CPP_REALTIME_EXECUTABLE"
)
RAW_RESULTS_ENVIRONMENT_VARIABLE = (
    "VISIONLAB_CPP_FRAME_RESULTS_PATH"
)
RUN_TIMEOUT_SECONDS = 300

CSV_FIELDS = [
    "architecture",
    "algorithm",
    "resolution",
    "trial",
    "frame_index",
    "scheduled_timestamp_ms",
    "enqueued_timestamp_ms",
    "processing_start_timestamp_ms",
    "processing_end_timestamp_ms",
    "drop_timestamp_ms",
    "source_delay_ms",
    "queue_wait_time_ms",
    "processing_time_ms",
    "end_to_end_latency_ms",
    "deadline_ms",
    "deadline_missed",
    "frame_status",
]


class ControlledIlluminationPureCppRunnerError(
    RuntimeError
):
    """Raised when the Pure C++ runner cannot execute."""


def current_utc_timestamp() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


def is_debug_path(candidate: Path) -> bool:
    return any(
        "debug" in path_part.lower()
        for path_part in candidate.parts
    )


def validate_executable_path(
    executable_path: Path,
) -> Path:
    resolved_path = executable_path.resolve()

    if not resolved_path.is_file():
        raise ControlledIlluminationPureCppRunnerError(
            "Pure C++ real-time executable was not "
            f"found: {resolved_path}"
        )

    if is_debug_path(resolved_path):
        raise ControlledIlluminationPureCppRunnerError(
            "Pure C++ controlled-illumination runs "
            "must use a Release executable."
        )

    return resolved_path


def find_pure_cpp_realtime_executable(
    environment: Mapping[str, str] | None = None,
    *,
    project_root: Path = PROJECT_ROOT,
) -> Path:
    active_environment = (
        os.environ
        if environment is None
        else environment
    )

    configured_path = active_environment.get(
        EXECUTABLE_ENVIRONMENT_VARIABLE
    )

    if configured_path:
        return validate_executable_path(
            Path(configured_path)
        )

    executable_name = (
        "VisionLabCppRealtime.exe"
        if os.name == "nt"
        else "VisionLabCppRealtime"
    )
    build_root = (
        project_root
        / "cpp-opencv-core"
        / "build"
    )

    if not build_root.is_dir():
        raise ControlledIlluminationPureCppRunnerError(
            "Pure C++ build directory was not found: "
            f"{build_root}"
        )

    candidates = sorted(
        (
            candidate.resolve()
            for candidate in build_root.rglob(
                executable_name
            )
            if (
                candidate.is_file()
                and not is_debug_path(candidate)
            )
        ),
        key=lambda candidate: str(candidate),
    )

    if not candidates:
        raise ControlledIlluminationPureCppRunnerError(
            "A Release VisionLabCppRealtime "
            "executable could not be found."
        )

    if len(candidates) > 1:
        candidate_list = "\n".join(
            f"- {candidate}"
            for candidate in candidates
        )

        raise ControlledIlluminationPureCppRunnerError(
            "Multiple Pure C++ Release executables "
            "were found. Set "
            f"{EXECUTABLE_ENVIRONMENT_VARIABLE}.\n"
            f"{candidate_list}"
        )

    return candidates[0]


def require_csv_value(
    row: dict[str, str | None],
    field_name: str,
) -> str:
    value = row.get(field_name)

    if value is None or value == "":
        raise ControlledIlluminationPureCppRunnerError(
            "Required C++ frame-result value is "
            f"missing: {field_name}"
        )

    return value


def parse_positive_integer(
    row: dict[str, str | None],
    field_name: str,
) -> int:
    text = require_csv_value(
        row,
        field_name,
    )

    try:
        value = int(text)
    except ValueError as error:
        raise ControlledIlluminationPureCppRunnerError(
            "Invalid integer in C++ frame results: "
            f"{field_name}={text}"
        ) from error

    if value <= 0:
        raise ControlledIlluminationPureCppRunnerError(
            "C++ frame-result integer must be "
            f"positive: {field_name}"
        )

    return value


def parse_finite_number(
    row: dict[str, str | None],
    field_name: str,
) -> float:
    text = require_csv_value(
        row,
        field_name,
    )

    try:
        value = float(text)
    except ValueError as error:
        raise ControlledIlluminationPureCppRunnerError(
            "Invalid numeric C++ frame result: "
            f"{field_name}={text}"
        ) from error

    if not math.isfinite(value):
        raise ControlledIlluminationPureCppRunnerError(
            "C++ frame-result number must be finite: "
            f"{field_name}"
        )

    return value


def parse_optional_number(
    row: dict[str, str | None],
    field_name: str,
) -> float | None:
    text = row.get(field_name)

    if text is None:
        raise ControlledIlluminationPureCppRunnerError(
            "C++ frame-result field is missing: "
            f"{field_name}"
        )

    if text == "":
        return None

    try:
        value = float(text)
    except ValueError as error:
        raise ControlledIlluminationPureCppRunnerError(
            "Invalid optional numeric C++ frame "
            f"result: {field_name}={text}"
        ) from error

    if not math.isfinite(value):
        raise ControlledIlluminationPureCppRunnerError(
            "C++ frame-result number must be finite: "
            f"{field_name}"
        )

    return value


def parse_optional_boolean(
    row: dict[str, str | None],
    field_name: str,
) -> bool | None:
    text = row.get(field_name)

    if text is None:
        raise ControlledIlluminationPureCppRunnerError(
            "C++ frame-result field is missing: "
            f"{field_name}"
        )

    if text == "":
        return None

    if text == "true":
        return True

    if text == "false":
        return False

    raise ControlledIlluminationPureCppRunnerError(
        "Invalid boolean C++ frame result: "
        f"{field_name}={text}"
    )


def frame_record_from_csv_row(
    row: dict[str, str | None],
) -> RealtimeFrameRecord:
    frame_status_text = require_csv_value(
        row,
        "frame_status",
    )

    try:
        frame_status = FrameStatus(
            frame_status_text
        )
    except ValueError as error:
        raise ControlledIlluminationPureCppRunnerError(
            "Unsupported C++ frame status: "
            f"{frame_status_text}"
        ) from error

    return RealtimeFrameRecord(
        architecture=require_csv_value(
            row,
            "architecture",
        ),
        algorithm=require_csv_value(
            row,
            "algorithm",
        ),
        resolution=require_csv_value(
            row,
            "resolution",
        ),
        trial=parse_positive_integer(
            row,
            "trial",
        ),
        frame_index=parse_positive_integer(
            row,
            "frame_index",
        ),
        scheduled_timestamp_ms=(
            parse_finite_number(
                row,
                "scheduled_timestamp_ms",
            )
        ),
        enqueued_timestamp_ms=(
            parse_optional_number(
                row,
                "enqueued_timestamp_ms",
            )
        ),
        processing_start_timestamp_ms=(
            parse_optional_number(
                row,
                "processing_start_timestamp_ms",
            )
        ),
        processing_end_timestamp_ms=(
            parse_optional_number(
                row,
                "processing_end_timestamp_ms",
            )
        ),
        drop_timestamp_ms=(
            parse_optional_number(
                row,
                "drop_timestamp_ms",
            )
        ),
        source_delay_ms=(
            parse_optional_number(
                row,
                "source_delay_ms",
            )
        ),
        queue_wait_time_ms=(
            parse_optional_number(
                row,
                "queue_wait_time_ms",
            )
        ),
        processing_time_ms=(
            parse_optional_number(
                row,
                "processing_time_ms",
            )
        ),
        end_to_end_latency_ms=(
            parse_optional_number(
                row,
                "end_to_end_latency_ms",
            )
        ),
        deadline_ms=parse_finite_number(
            row,
            "deadline_ms",
        ),
        deadline_missed=parse_optional_boolean(
            row,
            "deadline_missed",
        ),
        frame_status=frame_status,
    )


def load_cpp_frame_records(
    input_path: Path,
) -> tuple[RealtimeFrameRecord, ...]:
    if not input_path.is_file():
        raise ControlledIlluminationPureCppRunnerError(
            "Pure C++ frame-result file was not "
            f"created: {input_path}"
        )

    with input_path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as input_file:
        reader = csv.DictReader(input_file)

        if reader.fieldnames != CSV_FIELDS:
            raise ControlledIlluminationPureCppRunnerError(
                "Pure C++ frame-result CSV header "
                "does not match the shared schema."
            )

        records = tuple(
            frame_record_from_csv_row(row)
            for row in reader
        )

    if not records:
        raise ControlledIlluminationPureCppRunnerError(
            "Pure C++ frame-result CSV is empty."
        )

    return records


def execute_pure_cpp_run(
    environment: Mapping[str, str] | None = None,
    *,
    now_provider: Callable[[], str] = (
        current_utc_timestamp
    ),
    project_root: Path = PROJECT_ROOT,
) -> tuple[Path, Path]:
    context = load_runner_context_from_environment(
        environment,
        expected_architecture=PURE_CPP_ARCHITECTURE,
    )
    realtime_config = load_realtime_config()

    executable_path = (
        find_pure_cpp_realtime_executable(
            environment,
            project_root=project_root,
        )
    )

    raw_results_path = (
        context.output_directory
        / ".pure_cpp_frame_results.raw.csv"
    ).resolve()
    raw_results_path.unlink(
        missing_ok=True
    )

    child_environment = dict(os.environ)

    if environment is not None:
        child_environment.update(environment)

    child_environment[
        RAW_RESULTS_ENVIRONMENT_VARIABLE
    ] = str(raw_results_path)

    started_at_utc = now_provider()

    try:
        completed_process = subprocess.run(
            [str(executable_path)],
            cwd=project_root,
            env=child_environment,
            capture_output=True,
            text=True,
            timeout=RUN_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise ControlledIlluminationPureCppRunnerError(
            "Pure C++ controlled-illumination run "
            f"exceeded {RUN_TIMEOUT_SECONDS} seconds."
        ) from error
    except OSError as error:
        raise ControlledIlluminationPureCppRunnerError(
            "Pure C++ real-time executable could "
            f"not be started: {error}"
        ) from error

    finished_at_utc = now_provider()

    if completed_process.returncode != 0:
        error_output = (
            completed_process.stderr.strip()
            or completed_process.stdout.strip()
            or "No process output was captured."
        )

        raise ControlledIlluminationPureCppRunnerError(
            "Pure C++ controlled-illumination "
            "executable failed with exit code "
            f"{completed_process.returncode}: "
            f"{error_output}"
        )

    records = load_cpp_frame_records(
        raw_results_path
    )

    artifact_paths = (
        write_completed_run_artifacts_atomic(
            context,
            records,
            started_at_utc=started_at_utc,
            finished_at_utc=finished_at_utc,
            warmup_frame_count=(
                realtime_config.warmup_frames
            ),
        )
    )

    raw_results_path.unlink(
        missing_ok=True
    )

    return artifact_paths


def run_cli() -> int:
    try:
        frame_results_path, summary_path = (
            execute_pure_cpp_run()
        )
    except Exception as error:
        print(
            "Pure C++ controlled-illumination "
            f"run failed: {error}",
            file=sys.stderr,
        )
        return 1

    print(
        "Pure C++ controlled-illumination "
        "run completed."
    )
    print(
        f"Frame results: {frame_results_path}"
    )
    print(
        f"Execution summary: {summary_path}"
    )

    return 0


def main() -> None:
    raise SystemExit(run_cli())


if __name__ == "__main__":
    main()
