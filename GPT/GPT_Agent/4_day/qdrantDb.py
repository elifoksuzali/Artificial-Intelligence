# region QdrantDb
import os
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, ScoredPoint, PointIdsList, VectorParams, Distance
from typing import List, Dict, Any, Optional

# Embedding işlemlerini açık kaynaklı model ile de yapabiliriz
os.environ['QDRANT_HOST'] = 'http://QDRANT_HOST'  # HTTP kullan
os.environ['QDRANT_API_KEY'] = 'QDRANT_API_KEY'

class QdrantDatabase:
    def __init__(self, collection_name: str):
        self.collection_name = collection_name
        self.client: Optional[QdrantClient] = None

    def connect(self):
        if not self.client:
            try:
                self.client = QdrantClient(
                    url=os.getenv('QDRANT_HOST'),
                    api_key=os.getenv('QDRANT_API_KEY'),
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

    def create_collection(self, vector_size: int):
        self._ensure_client()
        if self.client:
            self.client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
            )

    def insert_point(self, point_id: int, vector: List[float], payload: Optional[Dict[str, Any]] = None):
        self._ensure_client()
        if self.client:
            point = PointStruct(
                id=point_id,
                vector=vector,
                payload=payload
            )
            self.client.upsert(collection_name=self.collection_name, points=[point])

    def insert_chat_message(self, point_id: int, message: str, response: str, embedding: list[float]):
        self._ensure_client()
        if self.client:
            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    PointStruct(
                        id=point_id,
                        vector=embedding,
                        payload={
                            "type": "chat",
                            "user": message,
                            "response": response
                        }
                    )
                ]
            )

    def delete_point(self, point_id: int):
        self._ensure_client()
        if self.client:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=PointIdsList(points=[point_id])
            )

    def update_point(self, point_id: int, vector: Optional[List[float]] = None, payload: Optional[Dict[str, Any]] = None):
        if vector is not None:  # vector None değilse güncelle
            self.insert_point(point_id, vector, payload)

    def query_points(self, vector: List[float], top_k: int = 5) -> List[ScoredPoint]:
        self._ensure_client()
        if not self.client:
            return []
        return self.client.search(
            collection_name=self.collection_name,
            query_vector=vector,
            limit=top_k,
            with_payload=True,
        )

    def delete_collection(self):
        self._ensure_client()
        if self.client:
            self.client.delete_collection(collection_name=self.collection_name)

    def get_collection_info(self):
        self._ensure_client()
        if not self.client:
            return None
        return self.client.get_collection(collection_name=self.collection_name)
