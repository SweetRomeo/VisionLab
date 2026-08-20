from dataclasses import dataclass
from threading import Condition
from typing import Generic, TypeVar


PayloadType = TypeVar("PayloadType")


@dataclass(frozen=True)
class ScheduledFrame(Generic[PayloadType]):
    frame_index: int
    scheduled_timestamp_ns: int
    enqueued_timestamp_ns: int
    payload: PayloadType

    def __post_init__(self) -> None:
        if self.frame_index <= 0:
            raise ValueError(
                "frame_index must be positive."
            )

        if self.scheduled_timestamp_ns < 0:
            raise ValueError(
                "scheduled_timestamp_ns cannot "
                "be negative."
            )

        if self.enqueued_timestamp_ns < 0:
            raise ValueError(
                "enqueued_timestamp_ns cannot "
                "be negative."
            )


class LatestFrameQueue(Generic[PayloadType]):
    def __init__(
        self,
        capacity: int = 1,
    ) -> None:
        if capacity != 1:
            raise ValueError(
                "LatestFrameQueue currently "
                "requires capacity to be 1."
            )

        self._condition = Condition()
        self._pending_frame: (
            ScheduledFrame[PayloadType] | None
        ) = None
        self._closed = False

    def put(
        self,
        frame: ScheduledFrame[PayloadType],
    ) -> ScheduledFrame[PayloadType] | None:
        with self._condition:
            if self._closed:
                raise RuntimeError(
                    "Cannot add a frame to a "
                    "closed queue."
                )

            dropped_frame = self._pending_frame
            self._pending_frame = frame

            self._condition.notify()

            return dropped_frame

    def get(
        self,
    ) -> ScheduledFrame[PayloadType] | None:
        with self._condition:
            self._condition.wait_for(
                lambda: (
                    self._pending_frame is not None
                    or self._closed
                )
            )

            if self._pending_frame is None:
                return None

            frame = self._pending_frame
            self._pending_frame = None

            return frame

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()

    @property
    def is_closed(self) -> bool:
        with self._condition:
            return self._closed

    @property
    def has_pending_frame(self) -> bool:
        with self._condition:
            return self._pending_frame is not None