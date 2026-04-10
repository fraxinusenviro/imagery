import numpy as np
from skimage.segmentation import slic


def slic_segment(image_rgb: np.ndarray, n_segments: int = 100,
                 compactness: float = 10.0, sigma: float = 1.0) -> tuple:
    """
    Segment using SLIC superpixels.

    Returns (labels_array, metadata).
    """
    n_segments = max(10, int(n_segments))
    compactness = max(0.1, float(compactness))
    sigma = max(0.0, float(sigma))

    labels = slic(
        image_rgb,
        n_segments=n_segments,
        compactness=compactness,
        sigma=sigma,
        convert2lab=True,
        start_label=0,
    )

    actual_n = int(np.unique(labels).size)
    return labels.astype(np.int32), {"n_segments": actual_n}
