from .kmeans import kmeans_segment
from .threshold import threshold_segment
from .slic import slic_segment
from .felzenszwalb import felzenszwalb_segment
from .watershed import watershed_segment

ALGORITHMS = {
    "kmeans": {
        "label": "K-Means Color Clustering",
        "fn": kmeans_segment,
        "params": [
            {"name": "n_clusters", "label": "Clusters", "min": 2, "max": 20, "default": 5, "step": 1},
            {"name": "attempts",   "label": "Attempts", "min": 1, "max": 10, "default": 3, "step": 1},
        ],
        "selects": [
            {"name": "color_space", "label": "Color Space", "options": ["LAB", "RGB"], "default": "LAB"}
        ],
    },
    "threshold": {
        "label": "Multi-Otsu Threshold",
        "fn": threshold_segment,
        "params": [
            {"name": "n_classes", "label": "Classes", "min": 2, "max": 5, "default": 2, "step": 1},
        ],
        "selects": [
            {"name": "channel", "label": "Channel", "options": ["Grayscale", "R", "G", "B", "HSV-V"], "default": "Grayscale"}
        ],
    },
    "slic": {
        "label": "SLIC Superpixels",
        "fn": slic_segment,
        "params": [
            {"name": "n_segments",   "label": "Segments",    "min": 10,  "max": 500, "default": 100, "step": 10},
            {"name": "compactness",  "label": "Compactness", "min": 1,   "max": 50,  "default": 10,  "step": 1},
            {"name": "sigma",        "label": "Sigma",       "min": 0,   "max": 5,   "default": 1,   "step": 0.5},
        ],
        "selects": [],
    },
    "felzenszwalb": {
        "label": "Felzenszwalb Graph-Based",
        "fn": felzenszwalb_segment,
        "params": [
            {"name": "scale",    "label": "Scale",    "min": 1,   "max": 500, "default": 100, "step": 1},
            {"name": "sigma",    "label": "Sigma",    "min": 0,   "max": 5,   "default": 0.8, "step": 0.1},
            {"name": "min_size", "label": "Min Size", "min": 1,   "max": 500, "default": 50,  "step": 1},
        ],
        "selects": [],
    },
    "watershed": {
        "label": "Watershed",
        "fn": watershed_segment,
        "params": [
            {"name": "gradient_blur",  "label": "Gradient Blur",  "min": 1,  "max": 21, "default": 5,  "step": 2},
            {"name": "min_distance",   "label": "Min Distance",   "min": 5,  "max": 50, "default": 20, "step": 1},
            {"name": "compactness",    "label": "Compactness",    "min": 0,  "max": 5,  "default": 0,  "step": 0.1},
        ],
        "selects": [],
    },
}


def get_schema():
    """Return algorithm schema without function references (for JSON serialization)."""
    schema = {}
    for key, algo in ALGORITHMS.items():
        schema[key] = {
            "label": algo["label"],
            "params": algo["params"],
            "selects": algo["selects"],
        }
    return schema
