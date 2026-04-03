import os
import logging
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, ScoredPoint, PointIdsList, VectorParams, Distance
from typing import List, Dict, Any, Optional
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.conversions import common_types as types

# Embedding işlemlerini açık kaynaklı model ile de yapabiliriz
os.environ['QDRANT_HOST'] = 'QDRANT_HOST'
os.environ['QDRANT_API_KEY'] = 'QDRANT_API_KEY'

class QdrantDatabase:
    def __init__(self, collection_name: str):
        self.collection_name = collection_name
        self.client = QdrantClient(
            url=os.getenv('QDRANT_HOST'),
            api_key=os.getenv('QDRANT_API_KEY'),
            timeout=20.0
        )
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    def connect(self):
        if not self.client:
            self.client = QdrantClient(
                url=os.getenv('QDRANT_HOST'),
                api_key=os.getenv('QDRANT_API_KEY'),
                timeout=20.0
            )

    def create_collection(self, vector_size: int):
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )

    def insert_point(self, point_id: int, vector: List[float], payload: Optional[Dict[str, Any]] = None):
        point = PointStruct(
            id=point_id,
            vector=vector,
            payload=payload
        )
        self.client.upsert(collection_name=self.collection_name, points=[point])
    # QdrantDatabase sınıfının içine EKLE
    def insert_chat_message(self, point_id: int, message: str, response: str, embedding: list[float]):
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
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=PointIdsList(points=[point_id])
        )

    def update_point(self, point_id: int, vector: Optional[List[float]] = None, payload: Optional[Dict[str, Any]] = None):
        self.insert_point(point_id, vector, payload)

    def query_points(self, vector: List[float], top_k: int = 5) -> List[ScoredPoint]:
        self.connect() # sadece gerektiğinde bağlan 
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=vector,
            limit=top_k,
            with_payload=True,
        )
        return search_result

    def delete_collection(self):
        self.client.delete_collection(collection_name=self.collection_name)

    def search(self, query_vector: List[float], limit: int = 5) -> List[ScoredPoint]:
        try:
            logging.info("-------------------------------")
            logging.info("Qdrant Search İsteği:")
            logging.info(f"Collection: {self.collection_name}")
            logging.info(f"Limit: {limit}")
            logging.info(f"Query Vector: {query_vector[:5]}...")  # İlk 5 elemanı göster

            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit
            )

            logging.info("-------------------------------")
            logging.info("Qdrant Search Sonucu (Ham):")
            logging.info(f"Toplam sonuç: {len(search_results)}")
            
            for i, result in enumerate(search_results, 1):
                logging.info(f"\nSonuç {i}:")
                logging.info(f"Score: {result.score}")
                logging.info(f"Payload: {result.payload}")
                if result.vector is not None:  # Vector kontrolü eklendi
                    logging.info(f"Vector: {result.vector[:5]}...")  # İlk 5 elemanı göster
                else:
                    logging.info("Vector: None")

            return search_results

        except Exception as e:
            logging.error(f"Qdrant search hatası: {str(e)}")
            raise e
