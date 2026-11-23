# config.py
import os
from dotenv import load_dotenv

load_dotenv()


# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Database
DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_USER = os.getenv("DB_USER")
URL_LINK = os.getenv("URL_LINK")
