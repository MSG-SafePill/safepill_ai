from pathlib import Path

import cv2
import numpy as np


def read_image(path: Path) -> np.ndarray:
    # cv2.imread does not reliably support Korean Windows paths.
    if not path.is_file():
        raise FileNotFoundError(f"Image file not found: {path}")
    encoded = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR) if encoded.size else None
    if image is None:
        raise ValueError("Unable to decode image. Use a valid JPEG or PNG image.")
    return image
