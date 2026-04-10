import os
import io
import uuid
import time
import base64
import tempfile

import numpy as np
from flask import Flask, request, jsonify, send_from_directory, session
from PIL import Image, UnidentifiedImageError

from segmentation import ALGORITHMS, get_schema
from segmentation.utils import encode_to_base64, resize_for_display, labels_to_color_image, overlay_boundaries

app = Flask(__name__, static_folder="static")
app.secret_key = os.environ.get("SECRET_KEY", os.urandom(24))
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB

TMPDIR = tempfile.gettempdir()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tmp_path(uid: str) -> str:
    return os.path.join(TMPDIR, f"imagery_{uid}.npy")


def _cleanup(uid: str) -> None:
    try:
        os.remove(_tmp_path(uid))
    except FileNotFoundError:
        pass


def _load_image() -> np.ndarray | None:
    uid = session.get("image_uid")
    if not uid:
        return None
    path = _tmp_path(uid)
    if not os.path.exists(path):
        return None
    return np.load(path)


def _preview_base64(image_rgb: np.ndarray) -> str:
    preview = resize_for_display(image_rgb, max_dim=800)
    img = Image.fromarray(preview.astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/algorithms", methods=["GET"])
def algorithms():
    return jsonify(get_schema())


@app.route("/upload", methods=["POST"])
def upload():
    if "image" not in request.files:
        return jsonify({"error": "No image field in request"}), 400

    file = request.files["image"]
    if not file.filename:
        return jsonify({"error": "No file selected"}), 400

    raw = file.read()
    try:
        pil_img = Image.open(io.BytesIO(raw))
        pil_img.verify()
        pil_img = Image.open(io.BytesIO(raw))
        pil_img = pil_img.convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Invalid image: {e}"}), 400

    image_rgb = np.array(pil_img)
    h, w = image_rgb.shape[:2]

    old_uid = session.get("image_uid")
    if old_uid:
        _cleanup(old_uid)

    uid = str(uuid.uuid4())
    np.save(_tmp_path(uid), image_rgb)
    session["image_uid"] = uid

    return jsonify({
        "status": "ok",
        "width": w,
        "height": h,
        "preview": _preview_base64(image_rgb),
    })


@app.route("/segment", methods=["POST"])
def segment():
    image_rgb = _load_image()
    if image_rgb is None:
        return jsonify({"error": "No image uploaded. Please upload an image first."}), 400

    body = request.get_json(silent=True) or {}
    algo_key = body.get("algorithm", "kmeans")
    params = body.get("params", {})
    display_mode = body.get("display_mode", "color")  # "color" | "boundary"

    if algo_key not in ALGORITHMS:
        return jsonify({"error": f"Unknown algorithm: {algo_key}"}), 400

    work_img = resize_for_display(image_rgb, max_dim=800)

    algo = ALGORITHMS[algo_key]
    fn = algo["fn"]

    # Build kwargs from schema, coercing types correctly
    kwargs = {}
    for p in algo["params"]:
        name = p["name"]
        raw_val = params.get(name, p["default"])
        # If step has a fractional component, treat as float; otherwise int
        step = p["step"]
        if isinstance(step, float) and step % 1 != 0:
            kwargs[name] = float(raw_val)
        else:
            kwargs[name] = int(float(raw_val))
    for s in algo.get("selects", []):
        kwargs[s["name"]] = str(params.get(s["name"], s["default"]))

    t0 = time.time()
    try:
        labels, metadata = fn(work_img, **kwargs)
    except Exception as e:
        return jsonify({"error": f"Segmentation failed: {str(e)}"}), 500
    elapsed_ms = int((time.time() - t0) * 1000)

    if display_mode == "boundary":
        result_rgb = overlay_boundaries(work_img, labels)
    else:
        result_rgb = labels_to_color_image(labels)

    metadata["elapsed_ms"] = elapsed_ms
    return jsonify({
        "result_image": encode_to_base64(result_rgb),
        "metadata": metadata,
    })


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------

@app.errorhandler(413)
def too_large(_):
    return jsonify({"error": "File too large. Maximum size is 10 MB."}), 413


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
