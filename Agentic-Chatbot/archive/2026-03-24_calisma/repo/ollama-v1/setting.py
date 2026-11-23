# setting.py
import os
from dotenv import load_dotenv
load_dotenv()

# Database
DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

# Qdrant
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION")

# Embed Model
DENSE_MODEL = os.getenv("DENSE_MODEL")
SPARSE_MODEL = os.getenv("SPARSE_MODEL")
RERANKER_MODEL = os.getenv("RERANKER_MODEL")
