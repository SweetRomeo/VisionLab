import unittest
from threading import Event, Thread

from benchmarks.realtime.latest_frame_queue import (
    LatestFrameQueue,
    ScheduledFrame,
)


class ScheduledFrameTests(unittest.TestCase):
    def test_rejects_invalid_metadata(self) -> None:
        invalid_values = (
            ("frame_index", 0),
            ("scheduled_timestamp_ns", -1),
            ("enqueued_timestamp_ns", -1),
        )

        for field_name, invalid_value in invalid_values:
            with self.subTest(field_name=field_name):
                arguments = {
                    "frame_index": 1,
                    "scheduled_timestamp_ns": 10,
                    "enqueued_timestamp_ns": 20,
                    "payload": "frame",
                }
                arguments[field_name] = invalid_value

                with self.assertRaises(ValueError):
                    ScheduledFrame(**arguments)


class LatestFrameQueueTests(unittest.TestCase):
    def setUp(self) -> None:
        self.first_frame = ScheduledFrame(
            frame_index=1,
            scheduled_timestamp_ns=10,
            enqueued_timestamp_ns=20,
            payload="first",
        )
        self.second_frame = ScheduledFrame(
            frame_index=2,
            scheduled_timestamp_ns=30,
            enqueued_timestamp_ns=40,
            payload="second",
        )

    def test_rejects_unsupported_capacity(
        self,
    ) -> None:
        with self.assertRaises(ValueError):
            LatestFrameQueue(capacity=2)

    def test_put_and_get_returns_frame(
        self,
    ) -> None:
        frame_queue = LatestFrameQueue[str]()

        dropped_frame = frame_queue.put(
            self.first_frame
        )

        self.assertIsNone(dropped_frame)
        self.assertTrue(
            frame_queue.has_pending_frame
        )
        self.assertEqual(
            frame_queue.get(),
            self.first_frame,
        )
        self.assertFalse(
            frame_queue.has_pending_frame
        )

    def test_new_frame_replaces_pending_frame(
        self,
    ) -> None:
        frame_queue = LatestFrameQueue[str]()

        frame_queue.put(self.first_frame)
        dropped_frame = frame_queue.put(
            self.second_frame
        )

        self.assertEqual(
            dropped_frame,
            self.first_frame,
        )
        self.assertEqual(
            frame_queue.get(),
            self.second_frame,
        )

    def test_close_drains_pending_frame(
        self,
    ) -> None:
        frame_queue = LatestFrameQueue[str]()

        frame_queue.put(self.first_frame)
        frame_queue.close()

        self.assertTrue(frame_queue.is_closed)
        self.assertEqual(
            frame_queue.get(),
            self.first_frame,
        )
        self.assertIsNone(frame_queue.get())

    def test_close_unblocks_waiting_consumer(
        self,
    ) -> None:
        frame_queue = LatestFrameQueue[str]()
        consumer_started = Event()
        received_frames = []

        def consume_frame() -> None:
            consumer_started.set()
            received_frames.append(
                frame_queue.get()
            )

        consumer_thread = Thread(
            target=consume_frame,
            daemon=True,
        )
        consumer_thread.start()

        self.assertTrue(
            consumer_started.wait(timeout=1.0)
        )

        frame_queue.close()
        consumer_thread.join(timeout=2.0)

        self.assertFalse(
            consumer_thread.is_alive()
        )
        self.assertEqual(
            received_frames,
            [None],
        )

    def test_put_after_close_is_rejected(
        self,
    ) -> None:
        frame_queue = LatestFrameQueue[str]()
        frame_queue.close()

        with self.assertRaises(RuntimeError):
            frame_queue.put(self.first_frame)


if __name__ == "__main__":
    unittest.main()