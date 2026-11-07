import os
import hmac

from flask import Flask, request, jsonify
from werkzeug.exceptions import RequestEntityTooLarge

from app.inference import predict_image, MODEL_NAME
from app.routes import validate_image


app = Flask(__name__)

app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024

API_KEY = os.getenv("VISIONGUARD_API_KEY", "")


@app.route("/health", methods=["GET"])
def health():

    return jsonify({
        "status": "healthy",
        "model": "loaded",
        "model_name": MODEL_NAME
    }), 200


@app.route("/predict", methods=["POST"])
def predict():

    supplied_key = request.headers.get("X-API-Key", "")

    if not API_KEY:
        return jsonify({
            "error": "API authentication is not configured."
        }), 503

    if not hmac.compare_digest(supplied_key, API_KEY):
        return jsonify({
            "error": "Invalid or missing API key."
        }), 401

    uploaded_file = request.files.get("image")

    if uploaded_file is None or not uploaded_file.filename:
        return jsonify({
            "error": "Please provide an image using the image field."
        }), 400

    image_bytes = uploaded_file.read()

    try:

        validate_image(
            uploaded_file.filename,
            image_bytes
        )

        prediction = predict_image(image_bytes)

        return jsonify(prediction), 200

    except ValueError as error:

        return jsonify({
            "error": str(error)
        }), 400

    except Exception:

        app.logger.exception("Prediction failed.")

        return jsonify({
            "error": "The prediction service encountered an internal error."
        }), 500


@app.errorhandler(RequestEntityTooLarge)
def file_too_large(error):

    return jsonify({
        "error": "Image exceeds the 5 MB upload limit."
    }), 413


if __name__ == "__main__":
    from waitress import serve
    serve(app, host="0.0.0.0", port=8000)
