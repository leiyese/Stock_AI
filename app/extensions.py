from flask_sqlalchemy import SQLAlchemy
from flask import Blueprint


# Initiera db
db = SQLAlchemy()

# Initate blueprint bp
bp = Blueprint("bp", __name__)
