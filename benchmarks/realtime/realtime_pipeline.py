from collections.abc import (
    Callable,
    Iterable,
    Iterator,
)
from pathlib import Path
from threading import Event, Lock, Thread
from time import perf_counter_ns

import cv2
import numpy as np

from benchmarks.realtime.latest_frame_queue import (
    LatestFrameQueue,
    ScheduledFrame,
)
from benchmarks.realtime.realtime_config import (
    RealtimeConfig,
)
from benchmarks.realtime.realtime_records import (
    RealtimeFrameRecord,
    RealtimeRunContext,
    create_dropped_record,
    create_processed_record,
)


FrameProcessor = Callable[
    [np.ndarray],
    np.ndarray,
]


def iter_video_frames(
    video_path: Path,
) -> Iterator[np.ndarray]:
    if not video_path.is_file():
        raise FileNotFoundError(
            f"Input video was not found: {video_path}"
        )

    capture = cv2.VideoCapture(
        str(video_path)
    )

    if not capture.isOpened():
        raise RuntimeError(
            f"Input video could not be opened: "
            f"{video_path}"
        )

    try:
        while True:
            frame_received, frame = capture.read()

            if not frame_received:
                break

            if frame is None or frame.size == 0:
                raise RuntimeError(
                    "An empty frame was read from "
                    f"the input video: {video_path}"
                )

            yield frame
    finally:
        capture.release()


def iter_camera_frames(
    camera_index: int,
) -> Iterator[np.ndarray]:
    if (
        isinstance(camera_index, bool)
        or not isinstance(camera_index, int)
        or camera_index < 0
    ):
        raise ValueError(
            "camera_index must be a "
            "non-negative integer."
        )

    capture = cv2.VideoCapture(
        camera_index
    )

    try:
        if not capture.isOpened():
            raise RuntimeError(
                f"Camera {camera_index} "
                "could not be opened."
            )

        while True:
            frame_received, frame = (
                capture.read()
            )

            if not frame_received:
                raise RuntimeError(
                    f"A frame could not be read "
                    f"from camera {camera_index}."
                )

            if (
                frame is None
                or frame.size == 0
            ):
                raise RuntimeError(
                    "An empty frame was read "
                    f"from camera {camera_index}."
                )

            yield frame
    finally:
        capture.release()


def validate_input_frame(
    frame: np.ndarray,
) -> None:
    if not isinstance(frame, np.ndarray):
        raise TypeError(
            "Input frame must be a NumPy array."
        )

    if (
        frame.size == 0
        or frame.ndim != 3
        or frame.shape[2] != 3
        or frame.dtype != np.uint8
    ):
        raise ValueError(
            "Input frame must be a non-empty "
            "H x W x 3 uint8 BGR image."
        )


def validate_processed_frame(
    source: np.ndarray,
    processed: np.ndarray,
) -> None:
    if not isinstance(processed, np.ndarray):
        raise TypeError(
            "Frame processor must return "
            "a NumPy array."
        )

    if processed.size == 0:
        raise ValueError(
            "Frame processor returned "
            "an empty image."
        )

    if processed.shape != source.shape:
        raise ValueError(
            "Processed frame shape does not "
            "match the source frame."
        )

    if processed.dtype != source.dtype:
        raise ValueError(
            "Processed frame data type does not "
            "match the source frame."
        )


def wait_until(
    target_timestamp_ns: int,
    stop_event: Event,
) -> bool:
    while True:
        if stop_event.is_set():
            return False

        remaining_ns = (
            target_timestamp_ns
            - perf_counter_ns()
        )

        if remaining_ns <= 0:
            return True

        stop_requested = stop_event.wait(
            remaining_ns / 1_000_000_000.0
        )

        if stop_requested:
            return False


def run_realtime_trial(
    *,
    frame_source: Iterable[np.ndarray],
    processor: FrameProcessor,
    config: RealtimeConfig,
    architecture: str,
    algorithm: str,
    width: int,
    height: int,
    trial: int,
) -> list[RealtimeFrameRecord]:
    if not callable(processor):
        raise TypeError(
            "processor must be callable."
        )

    for field_name, value in (
        ("width", width),
        ("height", height),
        ("trial", trial),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
        ):
            raise ValueError(
                f"{field_name} must be a "
                "positive integer."
            )

    frame_period_ns = round(
        1_000_000_000.0
        / config.target_fps
    )
    total_frame_count = (
        config.warmup_frames
        + config.measured_frames
    )

    frame_queue = LatestFrameQueue[np.ndarray](
        capacity=config.queue_capacity
    )
    stop_event = Event()

    record_lock = Lock()
    error_lock = Lock()

    records: list[RealtimeFrameRecord] = []
    errors: list[Exception] = []

    origin_timestamp_ns = perf_counter_ns()

    run_context = RealtimeRunContext(
        architecture=architecture,
        algorithm=algorithm,
        resolution=f"{width}x{height}",
        trial=trial,
        origin_timestamp_ns=(
            origin_timestamp_ns
        ),
        deadline_ms=config.deadline_ms,
    )

    def register_error(
        error: Exception,
    ) -> None:
        with error_lock:
            if not errors:
                errors.append(error)

        stop_event.set()
        frame_queue.close()

    def append_record(
        record: RealtimeFrameRecord,
    ) -> None:
        with record_lock:
            records.append(record)

    def measured_frame_index(
        sequence_index: int,
    ) -> int | None:
        if sequence_index <= config.warmup_frames:
            return None

        return (
            sequence_index
            - config.warmup_frames
        )

    def producer() -> None:
        frame_iterator = iter(frame_source)

        try:
            for sequence_index in range(
                1,
                total_frame_count + 1,
            ):
                if stop_event.is_set():
                    break

                scheduled_timestamp_ns = (
                    origin_timestamp_ns
                    + (
                        sequence_index - 1
                    ) * frame_period_ns
                )

                try:
                    source_frame = next(
                        frame_iterator
                    )
                except StopIteration as error:
                    raise RuntimeError(
                        "Input frame source ended before "
                        "the configured real-time trial "
                        f"completed. Required frames: "
                        f"{total_frame_count}; received: "
                        f"{sequence_index - 1}."
                    ) from error

                validate_input_frame(
                    source_frame
                )

                # Video reading and resizing belong to
                # the source side of the pipeline.
                resized_frame = cv2.resize(
                    source_frame,
                    (width, height),
                    interpolation=cv2.INTER_LINEAR,
                )

                if not wait_until(
                    scheduled_timestamp_ns,
                    stop_event,
                ):
                    break

                if stop_event.is_set():
                    break

                enqueued_timestamp_ns = (
                    perf_counter_ns()
                )

                scheduled_frame = ScheduledFrame(
                    frame_index=sequence_index,
                    scheduled_timestamp_ns=(
                        scheduled_timestamp_ns
                    ),
                    enqueued_timestamp_ns=(
                        enqueued_timestamp_ns
                    ),
                    payload=resized_frame,
                )

                dropped_frame = frame_queue.put(
                    scheduled_frame
                )

                if dropped_frame is not None:
                    dropped_index = (
                        measured_frame_index(
                            dropped_frame.frame_index
                        )
                    )

                    if dropped_index is not None:
                        append_record(
                            create_dropped_record(
                                run_context,
                                frame_index=(
                                    dropped_index
                                ),
                                scheduled_timestamp_ns=(
                                    dropped_frame
                                    .scheduled_timestamp_ns
                                ),
                                enqueued_timestamp_ns=(
                                    dropped_frame
                                    .enqueued_timestamp_ns
                                ),
                                drop_timestamp_ns=(
                                    enqueued_timestamp_ns
                                ),
                            )
                        )
        except Exception as error:
            register_error(error)
        finally:
            close_method = getattr(
                frame_iterator,
                "close",
                None,
            )

            if callable(close_method):
                close_method()

            frame_queue.close()

    def consumer() -> None:
        try:
            while True:
                scheduled_frame = (
                    frame_queue.get()
                )

                if scheduled_frame is None:
                    break

                processing_start_timestamp_ns = (
                    perf_counter_ns()
                )

                processed_frame = processor(
                    scheduled_frame.payload
                )

                processing_end_timestamp_ns = (
                    perf_counter_ns()
                )

                validate_processed_frame(
                    scheduled_frame.payload,
                    processed_frame,
                )

                result_frame_index = (
                    measured_frame_index(
                        scheduled_frame.frame_index
                    )
                )

                if result_frame_index is not None:
                    append_record(
                        create_processed_record(
                            run_context,
                            frame_index=(
                                result_frame_index
                            ),
                            scheduled_timestamp_ns=(
                                scheduled_frame
                                .scheduled_timestamp_ns
                            ),
                            enqueued_timestamp_ns=(
                                scheduled_frame
                                .enqueued_timestamp_ns
                            ),
                            processing_start_timestamp_ns=(
                                processing_start_timestamp_ns
                            ),
                            processing_end_timestamp_ns=(
                                processing_end_timestamp_ns
                            ),
                        )
                    )
        except Exception as error:
            register_error(error)

    consumer_thread = Thread(
        target=consumer,
        name="realtime-consumer",
    )
    producer_thread = Thread(
        target=producer,
        name="realtime-producer",
    )

    consumer_thread.start()
    producer_thread.start()

    producer_thread.join()
    consumer_thread.join()

    if errors:
        raise RuntimeError(
            "Real-time trial failed: "
            f"{errors[0]}"
        ) from errors[0]

    records.sort(
        key=lambda record: record.frame_index
    )

    expected_indices = set(
        range(
            1,
            config.measured_frames + 1,
        )
    )
    observed_indices = [
        record.frame_index
        for record in records
    ]
    observed_index_set = set(
        observed_indices
    )

    duplicate_indices = sorted(
        frame_index
        for frame_index
        in observed_index_set
        if observed_indices.count(frame_index) > 1
    )
    missing_indices = sorted(
        expected_indices - observed_index_set
    )
    unexpected_indices = sorted(
        observed_index_set - expected_indices
    )

    if (
        duplicate_indices
        or missing_indices
        or unexpected_indices
        or len(records)
        != config.measured_frames
    ):
        raise RuntimeError(
            "Real-time frame-result integrity "
            "validation failed. "
            f"Missing: {missing_indices}; "
            f"duplicates: {duplicate_indices}; "
            f"unexpected: {unexpected_indices}."
        )

    return records