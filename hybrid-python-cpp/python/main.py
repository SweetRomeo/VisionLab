from time import perf_counter

import cv2

import visionlab_cpp


def main() -> None:
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        raise RuntimeError("Kamera acilamadi.")

    algorithms = {
        ord("1"): (
            "Original",
            visionlab_cpp.ProcessingAlgorithm.ORIGINAL,
        ),
        ord("2"): (
            "Gamma Correction",
            visionlab_cpp.ProcessingAlgorithm.GAMMA,
        ),
        ord("3"): (
            "Histogram Equalization",
            visionlab_cpp.ProcessingAlgorithm.HISTOGRAM,
        ),
        ord("4"): (
            "CLAHE",
            visionlab_cpp.ProcessingAlgorithm.CLAHE,
        ),
    }

    algorithm_name = "Original"
    algorithm = visionlab_cpp.ProcessingAlgorithm.ORIGINAL

    try:
        while True:
            frame_received, frame = camera.read()

            if not frame_received:
                continue

            start_time = perf_counter()

            processed_frame = visionlab_cpp.process_frame(
                frame,
                algorithm,

                # Gamma için 1.0 görüntüyü değiştirmez.
                # 0.6 görüntüyü aydınlatır.
                gamma_value=0.6,

                clahe_clip_limit=4.0,
                clahe_grid_size=8,
            )

            process_time_ms = (
                perf_counter() - start_time
            ) * 1000.0

            processing_fps = (
                1000.0 / process_time_ms
                if process_time_ms > 0.0
                else 0.0
            )

            cv2.putText(
                processed_frame,
                (
                    f"{algorithm_name} | "
                    f"{process_time_ms:.2f} ms | "
                    f"{processing_fps:.1f} FPS"
                ),
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                processed_frame,
                "1: Original  2: Gamma  3: Histogram  4: CLAHE",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            cv2.imshow(
                "VisionLab Hybrid - Python + C++",
                processed_frame,
            )

            pressed_key = cv2.waitKey(1) & 0xFF

            if pressed_key in (27, ord("q")):
                break

            if pressed_key in algorithms:
                algorithm_name, algorithm = algorithms[
                    pressed_key
                ]

                print(
                    f"Selected algorithm: {algorithm_name}"
                )

    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()