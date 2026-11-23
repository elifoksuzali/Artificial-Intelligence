# config.py
import json
import uuid
import time
from datetime import datetime, timezone
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
import logging
from dotenv import load_dotenv
import os
import torch
from transformers import AutoTokenizer, AutoModel

# Logging ayarları
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables
load_dotenv()
QDRANT_HOST = os.getenv("QDRANT_HOST", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

class TurMultiVectorProcessor:
    def __init__(self, qdrant_url=QDRANT_HOST, collection_name="tur_multivector"):
        # Qdrant client'ı API key ile başlat
        if QDRANT_API_KEY:
            self.client = QdrantClient(url=qdrant_url, api_key=QDRANT_API_KEY)
        else:
            self.client = QdrantClient(qdrant_url)
            
        self.collection_name = collection_name
        
        # Dense embedding modeli (BGE-M3)
        self.dense_model = SentenceTransformer('BAAI/bge-m3')
        
        # Sparse embedding modeli (ColBERT)
        self.sparse_tokenizer = AutoTokenizer.from_pretrained('colbert-ir/colbertv2.0')
        self.sparse_model = AutoModel.from_pretrained('colbert-ir/colbertv2.0')
        self.sparse_model.eval()
        
    def create_collection(self):
        """Multi-vector collection oluştur - Dense ve Sparse vektörlerle"""
        try:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    # Dense vectors (BGE-M3)
                    "isim_dense": models.VectorParams(
                        size=1024,
                        distance=models.Distance.COSINE
                    ),
                    "metin_dense": models.VectorParams(
                        size=1024,
                        distance=models.Distance.COSINE
                    ),
                    "ziyaret_dense": models.VectorParams(
                        size=1024,
                        distance=models.Distance.COSINE
                    ),
                    "konaklama_dense": models.VectorParams(
                        size=1024,
                        distance=models.Distance.COSINE
                    ),
                    "ulasim_dense": models.VectorParams(
                        size=1024,
                        distance=models.Distance.COSINE
                    ),
                    "kategori_dense": models.VectorParams(
                        size=1024,
                        distance=models.Distance.COSINE
                    ),
                    # Sparse vectors (ColBERT)
                    "isim_sparse": models.VectorParams(
                        size=30522,  # ColBERT vocabulary size
                        distance=models.Distance.DOT
                    ),
                    "metin_sparse": models.VectorParams(
                        size=30522,
                        distance=models.Distance.DOT
                    ),
                    "ziyaret_sparse": models.VectorParams(
                        size=30522,
                        distance=models.Distance.DOT
                    ),
                    "konaklama_sparse": models.VectorParams(
                        size=30522,
                        distance=models.Distance.DOT
                    ),
                    "ulasim_sparse": models.VectorParams(
                        size=30522,
                        distance=models.Distance.DOT
                    ),
                    "kategori_sparse": models.VectorParams(
                        size=30522,
                        distance=models.Distance.DOT
                    )
                }
            )
            logger.info(f"Collection '{self.collection_name}' başarıyla oluşturuldu")
        except Exception as e:
            logger.info(f"Collection zaten mevcut veya oluşturma hatası: {e}")
    
    def encode_dense_text(self, text):
        """Metni dense vektörize et (BGE-M3)"""
        if not text or text.strip() == "":
            return [0.0] * 1024
        
        try:
            embeddings = self.dense_model.encode(text)
            if hasattr(embeddings, 'tolist'):
                return embeddings.tolist()
            return embeddings
        except Exception as e:
            logger.error(f"Dense encoding hatası: {e}")
            return [0.0] * 1024
    
    def encode_sparse_text(self, text):
        """Metni sparse vektörize et (ColBERT)"""
        if not text or text.strip() == "":
            return [0.0] * 30522
        
        try:
            # Tokenize
            inputs = self.sparse_tokenizer(
                text,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True
            )
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.sparse_model(**inputs)
                # Max pooling over tokens
                embeddings = torch.max(outputs.last_hidden_state, dim=1)[0]
                # Convert to sparse format (top-k values)
                sparse_emb = self._dense_to_sparse(embeddings.squeeze(), k=100)
                return sparse_emb
                
        except Exception as e:
            logger.error(f"Sparse encoding hatası: {e}")
            return [0.0] * 30522
    
    def _dense_to_sparse(self, dense_vector, k=100):
        """Dense vektörü sparse formata çevir (top-k values)"""
        # En büyük k değeri al
        values, indices = torch.topk(dense_vector, k)
        
        # Sparse vektör oluştur
        sparse_vector = [0.0] * 30522
        for i, idx in enumerate(indices):
            sparse_vector[idx.item()] = values[i].item()
        
        return sparse_vector
    
    def process_kategori_text(self, kategoriler):
        """Kategori listesini metin olarak birleştir"""
        if not kategoriler:
            return ""
        
        kategori_isimleri = [k.get('isim', '') for k in kategoriler if k.get('isim')]
        return " ".join(kategori_isimleri)
    
    def process_tur_data(self, json_file_path):
        """JSON dosyasından tur verilerini işle ve Qdrant'a kaydet"""
        
        # Collection'ı oluştur
        self.create_collection()
        
        # JSON dosyasını oku
        with open(json_file_path, 'r', encoding='utf-8') as f:
            tur_data = json.load(f)
        
        logger.info(f"Toplam {len(tur_data)} tur verisi işlenecek")
        
        processed_count = 0
        failed_count = 0
        
        for tur in tur_data:
            try:
                # Her tur için dense ve sparse vektörleri oluştur
                vectors = {
                    # Dense vectors
                    "isim_dense": self.encode_dense_text(tur.get('isim', '')),
                    "metin_dense": self.encode_dense_text(tur.get('metin', '')),
                    "ziyaret_dense": self.encode_dense_text(tur.get('ziyaretedilecekyerler', '')),
                    "konaklama_dense": self.encode_dense_text(tur.get('konaklama', '')),
                    "ulasim_dense": self.encode_dense_text(tur.get('ulasim', '')),
                    "kategori_dense": self.encode_dense_text(self.process_kategori_text(tur.get('turKategori', []))),
                    
                    # Sparse vectors
                    "isim_sparse": self.encode_sparse_text(tur.get('isim', '')),
                    "metin_sparse": self.encode_sparse_text(tur.get('metin', '')),
                    "ziyaret_sparse": self.encode_sparse_text(tur.get('ziyaretedilecekyerler', '')),
                    "konaklama_sparse": self.encode_sparse_text(tur.get('konaklama', '')),
                    "ulasim_sparse": self.encode_sparse_text(tur.get('ulasim', '')),
                    "kategori_sparse": self.encode_sparse_text(self.process_kategori_text(tur.get('turKategori', [])))
                }
                
                # Point oluştur - vectors parametresini doğru şekilde kullan
                point = models.PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_DNS, f"tur_{tur.get('id', 'unknown')}")),
                    vector=vectors,  # vectors yerine vector kullan
                    payload={
                        "id": tur.get('id'),
                        "isim": tur.get('isim'),
                        "turkodu": tur.get('turkodu'),
                        "gecesayisi": tur.get('gecesayisi'),
                        "metin": tur.get('metin'),
                        "geceleme": tur.get('geceleme'),
                        "konaklama": tur.get('konaklama'),
                        "ulasim": tur.get('ulasim'),
                        "ziyaretedilecekyerler": tur.get('ziyaretedilecekyerler'),
                        "vizesiz": tur.get('vizesiz'),
                        "kesinkalkis": tur.get('kesinkalkis'),
                        "ulasimtipi": tur.get('ulasimtipi'),
                        "turtipi": tur.get('turtipi'),
                        "turKategori": tur.get('turKategori', []),
                        "url": tur.get('url'),
                        "processing_metadata": {
                            "upload_datetime": datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
                            "vector_count": len(vectors),
                            "embedding_types": ["dense", "sparse"]
                        }
                    }
                )
                
                # Qdrant'a kaydet
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=[point]
                )
                
                processed_count += 1
                if processed_count % 100 == 0:
                    logger.info(f"İşlenen tur sayısı: {processed_count}")
                    
            except Exception as e:
                failed_count += 1
                logger.error(f"Tur işlenirken hata: {str(e)}")
                logger.error(f"Hatalı tur ID: {tur.get('id', 'unknown')}")
        
        logger.info(f"İşlem tamamlandı. Başarılı: {processed_count}, Başarısız: {failed_count}")
    
    def search_similar_turs(self, query_text, limit=10, vector_type="dense", field="isim"):
        """Benzer turları ara - Dense veya Sparse"""
        try:
            vector_name = f"{field}_{vector_type}"
            
            if vector_type == "dense":
                query_vector = self.encode_dense_text(query_text)
            else:  # sparse
                query_vector = self.encode_sparse_text(query_text)
            
            # Arama yap
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector={vector_name: query_vector},
                limit=limit
            )
            
            return search_result
            
        except Exception as e:
            logger.error(f"Arama hatası: {e}")
            return []
    
    def hybrid_search(self, query_text, limit=10, field="isim", dense_weight=0.7, sparse_weight=0.3):
        """Hybrid arama - Dense ve Sparse kombinasyonu"""
        try:
            dense_vector = self.encode_dense_text(query_text)
            sparse_vector = self.encode_sparse_text(query_text)
            
            # Hybrid arama
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector={
                    f"{field}_dense": dense_vector,
                    f"{field}_sparse": sparse_vector
                },
                query_filter=None,
                limit=limit,
                score_threshold=None,
                with_payload=True,
                with_vectors=False,
                search_params=models.SearchParams(
                    hnsw_ef=128,
                    exact=False
                )
            )
            
            return search_result
            
        except Exception as e:
            logger.error(f"Hybrid arama hatası: {e}")
            return []

# Kullanım örneği
if __name__ == "__main__":
    processor = TurMultiVectorProcessor()
    
    # Verileri işle ve Qdrant'a kaydet
    processor.process_tur_data("birlesik.json")
    
    # Örnek aramalar
    print("\n--- Benzer Tur Arama Örnekleri ---")
    
    # Dense arama
    results = processor.search_similar_turs("Kapadokya turu", limit=5, vector_type="dense", field="isim")
    print(f"\nKapadokya benzeri turlar (Dense):")
    for result in results:
        print(f"- {result.payload['isim']} (Skor: {result.score:.3f})")
    
    # Sparse arama
    results = processor.search_similar_turs("Antalya Fethiye", limit=5, vector_type="sparse", field="ziyaret")
    print(f"\nAntalya/Fethiye benzeri turlar (Sparse):")
    for result in results:
        print(f"- {result.payload['isim']} (Skor: {result.score:.3f})")
    
    # Hybrid arama
    results = processor.hybrid_search("İstanbul turu", limit=5, field="isim")
    print(f"\nİstanbul benzeri turlar (Hybrid):")
    for result in results:
        print(f"- {result.payload['isim']} (Skor: {result.score:.3f})")
