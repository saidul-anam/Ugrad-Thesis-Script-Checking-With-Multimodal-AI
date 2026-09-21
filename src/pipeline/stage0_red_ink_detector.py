import cv2
import numpy as np
from PIL import Image
from typing import Union, Optional, Any
from pydantic import BaseModel, Field, ConfigDict


class RedInkDetectionResult(BaseModel):
    """Output of Stage 0: Non-LLM Computer Vision Red-Ink Detection."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    has_red_ink: bool = Field(..., description="Whether red-ink teacher markings are present globally.")
    red_pixel_count: int = Field(0, description="Total count of red pixels detected above noise threshold.")
    red_pixel_ratio: float = Field(0.0, description="Ratio of red pixels to total image pixels.")
    margin_has_red_ink: bool = Field(False, description="Whether red ink numbers/scores are present in the left margin.")
    margin_red_pixel_count: int = Field(0, description="Count of red pixels in the left margin zone.")
    body_has_red_ink: bool = Field(False, description="Whether in-body red checkmarks/ticks are present.")
    body_red_pixel_count: int = Field(0, description="Count of red pixels in the student answer body zone.")
    details: str = Field("", description="Diagnostic details on the detection.")
    clean_image: Optional[Any] = Field(None, description="Inpainted clean canvas with in-body checkmarks removed.")


class RedInkDetector:
    """
    Stage 0: Fast OpenCV HSV color-thresholding red-ink detector.
    Runs on every script page without using LLM compute.
    Supports Dual-Zone (Margin Scores vs. In-Body Ticks) and Clean Canvas Inpainting.
    """

    def __init__(
        self,
        min_pixel_threshold: int = 1200,
        margin_pixel_threshold: int = 800,
        margin_width_ratio: float = 0.18,
        min_saturation: int = 60,
        min_value: int = 60,
        enable_inpainting: bool = True
    ):
        self.min_pixel_threshold = min_pixel_threshold
        self.margin_pixel_threshold = margin_pixel_threshold
        self.margin_width_ratio = margin_width_ratio
        self.min_saturation = min_saturation
        self.min_value = min_value
        self.enable_inpainting = enable_inpainting

    def detect(self, image_input: Union[Image.Image, np.ndarray, str]) -> RedInkDetectionResult:
        """
        Analyze image for presence of red ink teacher annotations,
        separating marginal score numbers from in-body checkmarks/ticks.
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

        h, w = image_bgr.shape[:2]
        total_pixels = h * w
        margin_w = int(w * self.margin_width_ratio)

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
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        filtered_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)

        # 5. Dual-zone count: Margin (0 to margin_w) vs Body (margin_w to w)
        margin_mask = filtered_mask[:, :margin_w]
        body_mask = filtered_mask[:, margin_w:]

        margin_red_pixel_count = int(cv2.countNonZero(margin_mask))
        body_red_pixel_count = int(cv2.countNonZero(body_mask))
        red_pixel_count = margin_red_pixel_count + body_red_pixel_count
        red_pixel_ratio = float(red_pixel_count / max(1, total_pixels))

        has_red_ink = red_pixel_count >= self.min_pixel_threshold
        margin_has_red_ink = margin_red_pixel_count >= self.margin_pixel_threshold
        body_has_red_ink = body_red_pixel_count >= 400

        # 6. Luminance-preserving suppression of body checkmarks to generate a clean student handwriting canvas for Stage 1
        clean_pil_image = None
        if self.enable_inpainting and body_has_red_ink:
            try:
                # Dilate body mask slightly to cover stroke fringes
                dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                dilated_body = cv2.dilate(body_mask, dilate_kernel, iterations=1)

                # Estimate background paper luminance from high-value non-text pixels
                gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
                paper_val = int(np.median(gray[gray > 180])) if np.any(gray > 180) else 235

                # Suppress red ticks on blank paper, but preserve dark student handwriting strokes underneath
                clean_bgr = image_bgr.copy()
                body_region = clean_bgr[:, margin_w:]
                body_gray = gray[:, margin_w:]

                # Pure red marks on paper have higher luminance; overlapping student strokes are dark (gray < 35)
                pure_teacher_mask = (dilated_body > 0) & (body_gray >= 35)
                body_region[pure_teacher_mask] = [paper_val, paper_val, paper_val]
                clean_bgr[:, margin_w:] = body_region

                clean_pil_image = Image.fromarray(cv2.cvtColor(clean_bgr, cv2.COLOR_BGR2RGB))
            except Exception:
                clean_pil_image = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        else:
            clean_pil_image = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))

        details = (
            f"Detected {red_pixel_count} red px ({red_pixel_ratio * 100:.3f}%). "
            f"Margin={margin_red_pixel_count} px (Score: {margin_has_red_ink}) | "
            f"Body={body_red_pixel_count} px (Ticks: {body_has_red_ink})"
        )

        return RedInkDetectionResult(
            has_red_ink=has_red_ink,
            red_pixel_count=red_pixel_count,
            red_pixel_ratio=red_pixel_ratio,
            margin_has_red_ink=margin_has_red_ink,
            margin_red_pixel_count=margin_red_pixel_count,
            body_has_red_ink=body_has_red_ink,
            body_red_pixel_count=body_red_pixel_count,
            details=details,
            clean_image=clean_pil_image
        )
