from dataclasses import dataclass
from enum import Enum, auto
import math

import cv2
import numpy as np
from numpy.typing import NDArray


Image = NDArray[np.uint8]

class ProcessingAlgorithm(Enum):
    ORIGINAL = auto()
    GAMMA = auto()
    HISTOGRAM = auto()
    CLAHE = auto()


@dataclass
class ProcessingParameters:
    gamma_value: float = 0.6
    clahe_clip_limit: float = 4.0
    clahe_grid_size: int = 8


class ImageProcess:
    def process(
        self,
        source: Image | None,
        algorithm: ProcessingAlgorithm,
        parameters: ProcessingParameters,
    ) -> Image:
        if source is None:
            return np.empty(
                (0, 0, 3),
                dtype=np.uint8,
            )

        self._validate_source(source)

        if source.size == 0:
            return source.copy()

        if algorithm == ProcessingAlgorithm.ORIGINAL:
            return source.copy()

        if algorithm == ProcessingAlgorithm.GAMMA:
            return self._apply_gamma(
                source,
                parameters.gamma_value,
            )

        if algorithm == ProcessingAlgorithm.HISTOGRAM:
            return self._apply_histogram_equalization(source)

        if algorithm == ProcessingAlgorithm.CLAHE:
            return self._apply_clahe(
                source,
                parameters.clahe_clip_limit,
                parameters.clahe_grid_size,
            )

        return source.copy()

    @staticmethod
    def _validate_source(source: Image) -> None:
        if not isinstance(source, np.ndarray):
            raise TypeError(
                "Girdi bir NumPy dizisi olmalıdır."
            )

        if source.dtype != np.uint8:
            raise ValueError(
                "Girdi uint8 veri tipinde olmalıdır."
            )

        if source.ndim != 3 or source.shape[2] != 3:
            raise ValueError(
                "Girdi H x W x 3 biçiminde bir BGR "
                "görüntü olmalıdır."
            )

    @staticmethod
    def _apply_gamma(
        source: Image,
        gamma: float,
    ) -> Image:
        if not math.isfinite(gamma) or gamma <= 0.0:
            raise ValueError(
                "Gamma değeri sıfırdan büyük olmalıdır."
            )

        normalized_values = (
            np.arange(256, dtype=np.float64) / 255.0
        )

        lookup_table = np.clip(
            np.rint(
                np.power(normalized_values, gamma) * 255.0
            ),
            0,
            255,
        ).astype(np.uint8)

        lab_image = cv2.cvtColor(
            source,
            cv2.COLOR_BGR2LAB,
        )

        lightness, channel_a, channel_b = cv2.split(
            lab_image
        )

        lightness = cv2.LUT(
            lightness,
            lookup_table,
        )

        processed_lab_image = cv2.merge(
            (
                lightness,
                channel_a,
                channel_b,
            )
        )

        return cv2.cvtColor(
            processed_lab_image,
            cv2.COLOR_LAB2BGR,
        )

    @staticmethod
    def _apply_histogram_equalization(
        source: Image,
    ) -> Image:
        lab_image = cv2.cvtColor(
            source,
            cv2.COLOR_BGR2LAB,
        )

        lightness, channel_a, channel_b = cv2.split(
            lab_image
        )

        lightness = cv2.equalizeHist(lightness)

        processed_lab_image = cv2.merge(
            (
                lightness,
                channel_a,
                channel_b,
            )
        )

        return cv2.cvtColor(
            processed_lab_image,
            cv2.COLOR_LAB2BGR,
        )

    @staticmethod
    def _apply_clahe(
        source: Image,
        clip_limit: float,
        grid_size: int,
    ) -> Image:
        if (
            not math.isfinite(clip_limit)
            or clip_limit <= 0.0
        ):
            raise ValueError(
                "CLAHE clip limit sıfırdan büyük olmalıdır."
            )

        if grid_size <= 0:
            raise ValueError(
                "CLAHE grid size sıfırdan büyük olmalıdır."
            )

        lab_image = cv2.cvtColor(
            source,
            cv2.COLOR_BGR2LAB,
        )

        lightness, channel_a, channel_b = cv2.split(
            lab_image
        )

        clahe = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=(grid_size, grid_size),
        )

        lightness = clahe.apply(lightness)

        processed_lab_image = cv2.merge(
            (
                lightness,
                channel_a,
                channel_b,
            )
        )

        return cv2.cvtColor(
            processed_lab_image,
            cv2.COLOR_LAB2BGR,
        )