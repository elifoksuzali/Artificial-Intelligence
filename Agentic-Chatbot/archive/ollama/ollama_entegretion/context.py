import os, sys
from qdrantDB import QdrantDatabase
import openai
from typing import List, Dict, Any
from config import OPENAI_API_KEY
from openai import OpenAI

openai_api_key = OPENAI_API_KEY
client = OpenAI(api_key=openai_api_key)

# Qdrant bağlantısı için global değişken
qdrant_client = None

def get_qdrant_client():
    global qdrant_client
    if qdrant_client is None:
        qdrant_client = QdrantDatabase("turizm_turlari")
        qdrant_client.connect()
    return qdrant_client
