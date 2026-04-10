import numpy as np
from skimage.segmentation import felzenszwalb


def felzenszwalb_segment(image_rgb: np.ndarray, scale: float = 100.0,
                         sigma: float = 0.8, min_size: int = 50) -> tuple:
    """
    Segment using Felzenszwalb graph-based segmentation.

    Returns (labels_array, metadata).
    """
    scale = max(1.0, float(scale))
    sigma = max(0.0, float(sigma))
    min_size = max(1, int(min_size))

    labels = felzenszwalb(image_rgb, scale=scale, sigma=sigma, min_size=min_size)

    actual_n = int(np.unique(labels).size)
    return labels.astype(np.int32), {"n_segments": actual_n}
