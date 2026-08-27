import cv2
import numpy as np
from PIL import Image
from typing import Union
from pydantic import BaseModel, Field


class RedInkDetectionResult(BaseModel):
    """Output of Stage 0: Non-LLM Computer Vision Red-Ink Detection."""
    has_red_ink: bool = Field(..., description="Whether red-ink teacher markings are present.")
    red_pixel_count: int = Field(0, description="Total count of red pixels detected above noise threshold.")
    red_pixel_ratio: float = Field(0.0, description="Ratio of red pixels to total image pixels.")
    details: str = Field("", description="Diagnostic details on the detection.")


class RedInkDetector:
    """
    Stage 0: Fast OpenCV HSV color-thresholding red-ink detector.
    Runs on every script page without using LLM compute.
    """

    def __init__(
        self,
        min_pixel_threshold: int = 150,
        min_saturation: int = 60,
        min_value: int = 60
    ):
        self.min_pixel_threshold = min_pixel_threshold
        self.min_saturation = min_saturation
        self.min_value = min_value

    def detect(self, image_input: Union[Image.Image, np.ndarray, str]) -> RedInkDetectionResult:
        """
        Analyze image for presence of red ink teacher annotations.
        """
        # 1. Convert input to OpenCV BGR numpy array
        if isinstance(image_input, str):
            image_bgr = cv2.imread(image_input)
            if image_bgr is None:
                raise ValueError(f"Could not load image from path: {image_input}")
        elif isinstance(image_input, Image.Image):
            rgb_arr = np.array(image_input.convert("RGB"))
            image_bgr = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)
        elif isinstance(image_input, np.ndarray):
            if len(image_input.shape) == 2:
                image_bgr = cv2.cvtColor(image_input, cv2.COLOR_GRAY2BGR)
            elif image_input.shape[2] == 4:
                image_bgr = cv2.cvtColor(image_input, cv2.COLOR_RGBA2BGR)
            else:
                image_bgr = image_input
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        total_pixels = image_bgr.shape[0] * image_bgr.shape[1]

        # 2. Convert to HSV color space
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

        # 3. Dual-range red hue thresholding (wraps around 0 and 180 degrees in OpenCV)
        lower_red_1 = np.array([0, self.min_saturation, self.min_value], dtype=np.uint8)
        upper_red_1 = np.array([10, 255, 255], dtype=np.uint8)

        lower_red_2 = np.array([170, self.min_saturation, self.min_value], dtype=np.uint8)
        upper_red_2 = np.array([180, 255, 255], dtype=np.uint8)

        mask1 = cv2.inRange(hsv, lower_red_1, upper_red_1)
        mask2 = cv2.inRange(hsv, lower_red_2, upper_red_2)
        red_mask = cv2.bitwise_or(mask1, mask2)

        # 4. Morphological opening to filter isolated 1-2px compression noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        filtered_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)

        # 5. Count positive red pixels
        red_pixel_count = int(cv2.countNonZero(filtered_mask))
        red_pixel_ratio = float(red_pixel_count / max(1, total_pixels))

        has_red_ink = red_pixel_count >= self.min_pixel_threshold

        details = (
            f"Detected {red_pixel_count} red pixels ({red_pixel_ratio * 100:.3f}% of page). "
            f"Threshold={self.min_pixel_threshold} -> has_red_ink={has_red_ink}"
        )

        return RedInkDetectionResult(
            has_red_ink=has_red_ink,
            red_pixel_count=red_pixel_count,
            red_pixel_ratio=red_pixel_ratio,
            details=details
        )
