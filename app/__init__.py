from flask import Flask
from config import Config
from app.extensions import db, bp
from dotenv import load_dotenv
import os


def create_app():
    flask_app = Flask(__name__)

    # Creates a new Flask application instance.
    # The __name__ argument tells Flask the name of the current Python module,
    # which is used to locate resources such as templates and static files.
    flask_app.config.from_object(Config)
    db.init_app(flask_app)
    from app import routes

    load_dotenv()
    MODEL_DIR = os.getenv("MODEL_DIR", "trained_models")  # Default to trained_models
    flask_app.register_blueprint(bp)

    with flask_app.app_context():
        from app import models

        db.create_all()

    return flask_app
