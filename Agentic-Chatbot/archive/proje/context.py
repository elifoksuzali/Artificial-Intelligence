# region Context
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from qdrantDb import QdrantDatabase
import openai
from typing import List, Dict, Any

from database.db import KeyManager


km = KeyManager()
api_key = km.load_api_key()
openai_api_key = api_key['gpt_api_key']

# OpenAI istemcisi
openai_api_key = openai_api_key
os.environ["OPENAI_API_KEY"] = openai_api_key
client = openai.OpenAI(api_key=openai_api_key)

# Qdrant bağlantısı için global değişken
qdrant_client = None

def get_qdrant_client():
    global qdrant_client
    if qdrant_client is None:
        qdrant_client = QdrantDatabase("mng-cosine")
        qdrant_client.connect()
    return qdrant_client

def get_embedding(text: str) -> List[float]:
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

async def get_context(question: str, q_client: QdrantDatabase = None, num_results: int = 10, offset: int = 0) -> List[Dict[str, Any]]:
    """
    Soru için Qdrant'tan bağlam alır.
    
    Args:
        question: Kullanıcının sorusu
        q_client: Qdrant veritabanı istemcisi (opsiyonel)
        num_results: Alınacak sonuç sayısı
        offset: Başlangıç offset'i (sayfalama için)
    
    Returns:
        List[Dict[str, Any]]: Bulunan sonuçlar
    """
    try:
        # Eğer q_client verilmemişse, global client'ı kullan
        if q_client is None:
            q_client = get_qdrant_client()
        
        # Soruyu vektöre dönüştür
        question_vector = get_embedding(question)
        
        # Qdrant'tan sonuçları al
        results = q_client.query_points(
            vector=question_vector,
            top_k=num_results + offset  # Offset'i hesaba kat
        )
        
        # Offset'ten sonraki sonuçları al
        results = results[offset:offset + num_results]
        
        # Sonuçları dönüştür
        formatted_results = []
        for result in results:
            formatted_results.append({
                "id": result.id,
                "score": result.score,
                "payload": result.payload,
                "text": result.payload.get("text", "")
            })
        
        print(f"✅ Qdrant'tan {len(formatted_results)} sonuç alındı.")
        return formatted_results
        
    except Exception as e:
        print(f"❌ Qdrant sorgu hatası: {str(e)}")
        return []