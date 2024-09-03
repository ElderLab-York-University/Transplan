from transformers import pipeline
from PIL import Image
import requests
import cv2
import numpy as np


class DepthAnythingV2Inference:
    def __init__(self) -> None:
        self.pipe = pipeline(
            task="depth-estimation",
            model="depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf",
            device="cuda",
        )

    def run_inference(self, cv2_img: np.ndarray):
        rgb_image = cv2.cvtColor(
            cv2_img,
            cv2.COLOR_BGR2RGB,
        )

        pil_image = Image.fromarray(rgb_image)

        depth = self.pipe(pil_image)["depth"]

        return np.array(depth)
