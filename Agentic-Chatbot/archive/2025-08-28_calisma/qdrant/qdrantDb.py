import os
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, ScoredPoint, PointIdsList, VectorParams, Distance, Filter, FieldCondition, MatchValue, Prefetch, QueryRequest
from typing import List, Dict, Any, Optional
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from db_key import KeyManager

key_manager = KeyManager()
api_keys = key_manager.load_api_key()
QDRANT_HOST = api_keys.get('qdrant_host', '')
QDRANT_API_KEY = api_keys.get('qdrant_api_key', '')


class QdrantDatabase:
    def __init__(self, collection_name: str):
        self.collection_name = collection_name
        self.client: Optional[QdrantClient] = None

    def connect(self):
        if not self.client:
            try:
                self.client = QdrantClient(
                    url=QDRANT_HOST,
                    api_key=QDRANT_API_KEY,
                    timeout=20,  # int olarak değiştirildi
                    verify=False  # SSL doğrulamasını devre dışı bırak
                )
                # Bağlantıyı test et
                self.client.get_collections()
                print("✅ Qdrant bağlantısı başarılı!")
            except Exception as e:
                print(f"❌ Qdrant bağlantı hatası: {str(e)}")
                raise

    def _ensure_client(self):
        """Client'ın bağlı olduğundan emin ol"""
        if not self.client:
            self.connect()

    def get_client(self) -> QdrantClient:
        """Qdrant client'ı döndür"""
        self._ensure_client()
        return self.client
