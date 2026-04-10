import numpy as np
from skimage.filters import threshold_multiotsu
from skimage.color import rgb2gray


def threshold_segment(image_rgb: np.ndarray, n_classes: int = 2,
                      channel: str = "Grayscale") -> tuple:
    """
    Segment using multi-Otsu thresholding on a selected channel.

    Returns (labels_array, metadata).
    """
    n_classes = max(2, min(5, int(n_classes)))

    ch = _extract_channel(image_rgb, channel)

    thresholds = threshold_multiotsu(ch, classes=n_classes)
    labels = np.digitize(ch, bins=thresholds).astype(np.int32)

    return labels, {
        "n_segments": n_classes,
        "thresholds": [round(float(t), 1) for t in thresholds],
    }


def _extract_channel(image_rgb: np.ndarray, channel: str) -> np.ndarray:
    if channel == "R":
        return image_rgb[:, :, 0].astype(np.float64)
    elif channel == "G":
        return image_rgb[:, :, 1].astype(np.float64)
    elif channel == "B":
        return image_rgb[:, :, 2].astype(np.float64)
    elif channel == "HSV-V":
        import cv2
        hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
        return hsv[:, :, 2].astype(np.float64)
    else:  # Grayscale
        gray = rgb2gray(image_rgb)
        return gray * 255.0
