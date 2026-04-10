import base64
import io
import numpy as np
from PIL import Image


def encode_to_base64(image_rgb: np.ndarray) -> str:
    """Encode an RGB numpy array to a base64-encoded PNG string."""
    img = Image.fromarray(image_rgb.astype(np.uint8), "RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=False)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def labels_to_color_image(labels: np.ndarray) -> np.ndarray:
    """Map an integer label array to an RGB image using a high-contrast HSV colormap."""
    unique = np.unique(labels)
    n = len(unique)
    label_to_idx = {lbl: i for i, lbl in enumerate(unique)}

    colors = np.array([
        _hsv_to_rgb(i / max(n, 1), 0.75, 0.95)
        for i in range(n)
    ], dtype=np.uint8)

    idx_image = np.vectorize(label_to_idx.__getitem__)(labels)
    return colors[idx_image]


def _hsv_to_rgb(h: float, s: float, v: float):
    """Convert HSV (0-1 floats) to (R, G, B) uint8."""
    import colorsys
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return int(r * 255), int(g * 255), int(b * 255)


def overlay_boundaries(original_rgb: np.ndarray, labels: np.ndarray, color=(1.0, 0.1, 0.1)) -> np.ndarray:
    """Draw segment boundaries on the original image."""
    from skimage.segmentation import mark_boundaries
    # mark_boundaries expects float [0,1] image
    img_float = original_rgb.astype(np.float64) / 255.0
    marked = mark_boundaries(img_float, labels, color=color, mode="thick")
    return (marked * 255).astype(np.uint8)


def resize_for_display(image_rgb: np.ndarray, max_dim: int = 800) -> np.ndarray:
    """Downscale image so longest side <= max_dim, preserving aspect ratio."""
    h, w = image_rgb.shape[:2]
    if max(h, w) <= max_dim:
        return image_rgb
    scale = max_dim / max(h, w)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    img = Image.fromarray(image_rgb.astype(np.uint8))
    img = img.resize((new_w, new_h), Image.LANCZOS)
    return np.array(img)
