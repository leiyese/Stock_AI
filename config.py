import os
from dotenv import load_dotenv

load_dotenv


class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "default_secret_key")
    SQLALCHEMY_DATABASE_URI = os.getenv("DATABASE_URI", "sqlite:///stocks.db")
    SQLALCHEMY_TRACK_MODIFICATIONS = False
