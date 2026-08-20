import csv
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from benchmarks.realtime.realtime_records import (
    FrameStatus,
    RealtimeRunContext,
    create_dropped_record,
    create_processed_record,
    create_skipped_record,
    write_frame_records,
)


class RealtimeRecordTests(unittest.TestCase):
    def setUp(self) -> None:
        self.origin_timestamp_ns = 1_000_000_000

        self.context = RealtimeRunContext(
            architecture="pure_python",
            algorithm="clahe",
            resolution="640x480",
            trial=1,
            origin_timestamp_ns=(
                self.origin_timestamp_ns
            ),
            deadline_ms=33.333,
        )

    def timestamp_ns(
        self,
        milliseconds: float,
    ) -> int:
        return (
            self.origin_timestamp_ns
            + round(milliseconds * 1_000_000.0)
        )

    def create_processed(
        self,
        *,
        frame_index: int = 1,
        scheduled_ms: float = 0.0,
        enqueued_ms: float = 1.0,
        processing_start_ms: float = 2.0,
        processing_end_ms: float = 12.0,
    ):
        return create_processed_record(
            self.context,
            frame_index=frame_index,
            scheduled_timestamp_ns=(
                self.timestamp_ns(scheduled_ms)
            ),
            enqueued_timestamp_ns=(
                self.timestamp_ns(enqueued_ms)
            ),
            processing_start_timestamp_ns=(
                self.timestamp_ns(
                    processing_start_ms
                )
            ),
            processing_end_timestamp_ns=(
                self.timestamp_ns(
                    processing_end_ms
                )
            ),
        )

    def test_processed_record_calculates_timings(
        self,
    ) -> None:
        record = self.create_processed()

        self.assertEqual(
            record.frame_status,
            FrameStatus.PROCESSED,
        )
        self.assertAlmostEqual(
            record.source_delay_ms,
            1.0,
        )
        self.assertAlmostEqual(
            record.queue_wait_time_ms,
            1.0,
        )
        self.assertAlmostEqual(
            record.processing_time_ms,
            10.0,
        )
        self.assertAlmostEqual(
            record.end_to_end_latency_ms,
            12.0,
        )
        self.assertFalse(
            record.deadline_missed
        )

    def test_processed_record_detects_deadline_miss(
        self,
    ) -> None:
        record = self.create_processed(
            processing_end_ms=40.0,
        )

        self.assertTrue(
            record.deadline_missed
        )

    def test_dropped_record_calculates_drop_data(
        self,
    ) -> None:
        record = create_dropped_record(
            self.context,
            frame_index=2,
            scheduled_timestamp_ns=(
                self.timestamp_ns(33.0)
            ),
            enqueued_timestamp_ns=(
                self.timestamp_ns(34.0)
            ),
            drop_timestamp_ns=(
                self.timestamp_ns(36.0)
            ),
        )

        self.assertEqual(
            record.frame_status,
            FrameStatus.DROPPED,
        )
        self.assertAlmostEqual(
            record.source_delay_ms,
            1.0,
        )
        self.assertAlmostEqual(
            record.drop_timestamp_ms,
            36.0,
        )
        self.assertIsNone(
            record.processing_time_ms
        )
        self.assertIsNone(
            record.deadline_missed
        )

    def test_skipped_record_has_no_queue_data(
        self,
    ) -> None:
        record = create_skipped_record(
            self.context,
            frame_index=3,
            scheduled_timestamp_ns=(
                self.timestamp_ns(66.0)
            ),
        )

        self.assertEqual(
            record.frame_status,
            FrameStatus.SKIPPED,
        )
        self.assertIsNone(
            record.enqueued_timestamp_ms
        )
        self.assertIsNone(
            record.processing_start_timestamp_ms
        )
        self.assertIsNone(
            record.deadline_missed
        )

    def test_invalid_timestamp_order_is_rejected(
        self,
    ) -> None:
        with self.assertRaises(ValueError):
            self.create_processed(
                scheduled_ms=10.0,
                enqueued_ms=9.0,
                processing_start_ms=11.0,
                processing_end_ms=12.0,
            )

    def test_write_frame_records_creates_csv(
        self,
    ) -> None:
        processed_record = self.create_processed()

        dropped_record = create_dropped_record(
            self.context,
            frame_index=2,
            scheduled_timestamp_ns=(
                self.timestamp_ns(33.0)
            ),
            enqueued_timestamp_ns=(
                self.timestamp_ns(34.0)
            ),
            drop_timestamp_ns=(
                self.timestamp_ns(36.0)
            ),
        )

        skipped_record = create_skipped_record(
            self.context,
            frame_index=3,
            scheduled_timestamp_ns=(
                self.timestamp_ns(66.0)
            ),
        )

        with TemporaryDirectory() as directory:
            output_path = (
                Path(directory)
                / "frame_results.csv"
            )

            written_count = write_frame_records(
                [
                    skipped_record,
                    processed_record,
                    dropped_record,
                ],
                output_path,
            )

            self.assertEqual(written_count, 3)
            self.assertTrue(output_path.is_file())

            with output_path.open(
                "r",
                newline="",
                encoding="utf-8",
            ) as input_file:
                rows = list(
                    csv.DictReader(input_file)
                )

            self.assertEqual(
                [
                    row["frame_status"]
                    for row in rows
                ],
                [
                    "processed",
                    "dropped",
                    "skipped",
                ],
            )
            self.assertEqual(
                rows[0]["deadline_missed"],
                "false",
            )
            self.assertEqual(
                rows[1]["processing_time_ms"],
                "",
            )
            self.assertEqual(
                rows[2]["enqueued_timestamp_ms"],
                "",
            )

    def test_duplicate_records_are_rejected(
        self,
    ) -> None:
        record = self.create_processed()

        with TemporaryDirectory() as directory:
            output_path = (
                Path(directory)
                / "frame_results.csv"
            )

            with self.assertRaises(ValueError):
                write_frame_records(
                    [record, record],
                    output_path,
                )

    def test_non_csv_output_is_rejected(
        self,
    ) -> None:
        record = self.create_processed()

        with TemporaryDirectory() as directory:
            output_path = (
                Path(directory)
                / "frame_results.txt"
            )

            with self.assertRaises(ValueError):
                write_frame_records(
                    [record],
                    output_path,
                )


if __name__ == "__main__":
    unittest.main()