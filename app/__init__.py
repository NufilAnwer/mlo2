import os
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, render_template

from .database import init_db


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def create_app():

    load_dotenv(PROJECT_ROOT / ".env")
    load_dotenv(PROJECT_ROOT / ".env.api")

    app = Flask(__name__)

    app.config["SECRET_KEY"] = os.getenv(
        "SECRET_KEY",
        "visionguard-development-key"
    )

    app.config["MODEL_PATH"] = os.getenv(
        "MODEL_PATH",
        str(PROJECT_ROOT / "models" / "task_3_5_champion.keras")
    )

    app.config["THRESHOLD"] = float(
        os.getenv("THRESHOLD", "0.4980708956718445")
    )

    app.config["DATABASE_PATH"] = os.getenv(
        "DATABASE_PATH",
        str(PROJECT_ROOT / "instance" / "visionguard.db")
    )

    app.config["DATABASE_URL"] = os.getenv("DATABASE_URL")

    app.config["INFERENCE_API_URL"] = os.getenv(
        "VISIONGUARD_API_URL",
        "http://3.26.146.78:8000"
    )

    app.config["INFERENCE_API_KEY"] = os.getenv(
        "VISIONGUARD_API_KEY",
        ""
    )

    max_upload_mb = int(
        os.getenv("MAX_UPLOAD_MB", "5")
    )

    app.config["MAX_UPLOAD_MB"] = max_upload_mb

    app.config["MAX_CONTENT_LENGTH"] = (
        max_upload_mb * 1024 * 1024
    )

    Path(app.instance_path).mkdir(
        parents=True,
        exist_ok=True
    )

    init_db(app)

    from .routes import main
    app.register_blueprint(main)

    @app.errorhandler(413)
    def file_too_large(error):

        return render_template(
            "error.html",
            message=f"File is too large. Maximum upload size is {app.config['MAX_UPLOAD_MB']} MB."
        ), 413

    @app.errorhandler(404)
    def page_not_found(error):

        return render_template(
            "error.html",
            message="The requested page could not be found."
        ), 404

    return app
