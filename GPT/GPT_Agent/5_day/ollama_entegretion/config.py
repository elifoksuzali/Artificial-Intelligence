# config.py
import os
from dotenv import load_dotenv

load_dotenv()

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Qdrant
QDRANT_HOST= os.getenv("QDRANT_HOST")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")

# Host ve port'u temizle
def clean_host_port(host_raw, port_raw):
    host = host_raw
    port = port_raw
    
    # Protokolü temizle
    if host.startswith(('http://', 'https://')):
        host = host.split('://')[1]
    
    # Port bilgisini host'tan ayır
    if ':' in host:
        host, port_str = host.split(':', 1)
        port = port_str
    
    return host, int(port)

QDRANT_HOST = clean_host_port(QDRANT_HOST)

# Collection
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "mng_01_09_25")

# Model
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")

# Data paths
JSON_DATA_PATH = os.getenv("JSON_DATA_PATH", "sata/birlesik.json")
VECTOR_OUTPUT_DIR = os.getenv("VECTOR_OUTPUT_DIR", "vectors")

# Vector dimensions
VECTOR_DIMENSION = int(os.getenv("VECTOR_DIMENSION", "1024"))  # BAAI/bge-m3 boyutu
