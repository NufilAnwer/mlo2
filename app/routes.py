from io import BytesIO

from flask import (
    Blueprint,
    current_app,
    jsonify,
    redirect,
    render_template,
    request,
    url_for
)

from PIL import Image, UnidentifiedImageError
from werkzeug.utils import secure_filename

import tensorflow as tf

from .inference import predict_image
from .database import save_prediction, get_prediction_history


main = Blueprint("main", __name__)

ALLOWED_EXTENSIONS = {
    "jpg": "JPEG",
    "jpeg": "JPEG",
    "png": "PNG",
    "webp": "WEBP"
}

MAX_IMAGE_PIXELS = 20_000_000


def validate_image(filename, image_bytes):

    if not filename or "." not in filename:
        raise ValueError("Please upload a valid image file.")

    extension = filename.rsplit(".", 1)[1].lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise ValueError(
            "Unsupported file type. Upload JPG, JPEG, PNG, or WEBP."
        )

    if not image_bytes:
        raise ValueError("The uploaded file is empty.")

    try:
        with Image.open(BytesIO(image_bytes)) as image:

            actual_format = image.format
            width, height = image.size

            if actual_format != ALLOWED_EXTENSIONS[extension]:
                raise ValueError(
                    "The file extension does not match its image format."
                )

            if width <= 0 or height <= 0:
                raise ValueError("The image dimensions are invalid.")

            if width * height > MAX_IMAGE_PIXELS:
                raise ValueError("The image dimensions are too large.")

            image.verify()

    except ValueError:
        raise

    except (
        UnidentifiedImageError,
        OSError,
        SyntaxError
    ) as error:

        raise ValueError(
            "The uploaded file is not a valid or readable image."
        ) from error


@main.route("/", methods=["GET"])
def index():

    return render_template("index.html")


@main.route("/result", methods=["POST"])
def result():

    uploaded_file = request.files.get("image")

    if uploaded_file is None or not uploaded_file.filename:
        return render_template(
            "error.html",
            message="Please select an image before submitting."
        ), 400

    original_filename = uploaded_file.filename

    safe_filename = secure_filename(original_filename)

    if not safe_filename:
        return render_template(
            "error.html",
            message="The uploaded filename is invalid."
        ), 400

    image_bytes = uploaded_file.read()

    try:

        validate_image(
            original_filename,
            image_bytes
        )

        prediction = predict_image(image_bytes)

        save_prediction(
            safe_filename,
            prediction
        )

        return render_template(
            "result.html",
            filename=safe_filename,
            prediction=prediction
        )

    except (
        ValueError,
        UnidentifiedImageError,
        OSError,
        tf.errors.InvalidArgumentError
    ) as error:

        return render_template(
            "error.html",
            message=str(error)
        ), 400


@main.route("/history", methods=["GET"])
def history():

    records = get_prediction_history()

    return render_template(
        "history.html",
        records=records
    )


@main.route("/health", methods=["GET"])
def health():

    return jsonify({
        "status": "healthy",
        "model": "loaded",
        "model_path": current_app.config["MODEL_PATH"]
    }), 200


@main.route("/upload", methods=["GET"])
def upload():

    return redirect(url_for("main.index"))
