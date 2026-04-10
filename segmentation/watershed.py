import numpy as np
import cv2
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage.color import rgb2gray
from scipy import ndimage as ndi


def watershed_segment(image_rgb: np.ndarray, gradient_blur: int = 5,
                      min_distance: int = 20, compactness: float = 0.0) -> tuple:
    """
    Segment using Watershed algorithm driven by gradient magnitude.

    Pipeline: grayscale → blur → Sobel gradient → distance-based markers → watershed.

    Returns (labels_array, metadata).
    """
    gradient_blur = max(1, int(gradient_blur))
    if gradient_blur % 2 == 0:
        gradient_blur += 1
    min_distance = max(1, int(min_distance))
    compactness = max(0.0, float(compactness))

    gray = (rgb2gray(image_rgb) * 255).astype(np.uint8)

    blurred = cv2.GaussianBlur(gray, (gradient_blur, gradient_blur), 0)

    sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    gradient = np.sqrt(sobelx ** 2 + sobely ** 2)

    # Threshold gradient to identify "flat" (low-gradient) foreground regions
    # Normalize to uint8 for Otsu (which requires CV_8UC1)
    grad_norm = cv2.normalize(gradient, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, fg_mask = cv2.threshold(grad_norm, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    fg_mask = fg_mask.astype(bool)

    dist = ndi.distance_transform_edt(fg_mask)
    coords = peak_local_max(dist, min_distance=min_distance, labels=fg_mask)

    markers = np.zeros(gray.shape, dtype=np.int32)
    for i, (r, c) in enumerate(coords, start=1):
        markers[r, c] = i

    if len(coords) == 0:
        r, c = np.unravel_index(np.argmax(dist), dist.shape)
        markers[r, c] = 1

    labels = watershed(gradient, markers, compactness=compactness)

    actual_n = int(np.unique(labels[labels > 0]).size)
    return labels.astype(np.int32), {"n_segments": actual_n}
