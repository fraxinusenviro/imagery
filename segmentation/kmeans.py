import cv2
import numpy as np


def kmeans_segment(image_rgb: np.ndarray, n_clusters: int = 5, attempts: int = 3,
                   color_space: str = "LAB") -> tuple:
    """
    Segment using K-Means color clustering.

    Returns (labels_array, metadata).
    """
    n_clusters = max(2, int(n_clusters))
    attempts = max(1, int(attempts))

    if color_space == "LAB":
        work = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    else:
        work = image_rgb.copy()

    h, w = work.shape[:2]
    pixels = work.reshape(-1, 3).astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels_flat, _ = cv2.kmeans(
        pixels, n_clusters, None, criteria, attempts, cv2.KMEANS_PP_CENTERS
    )

    labels = labels_flat.flatten().reshape(h, w).astype(np.int32)
    return labels, {"n_segments": n_clusters}
