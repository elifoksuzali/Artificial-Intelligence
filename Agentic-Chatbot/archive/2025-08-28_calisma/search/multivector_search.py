from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding, LateInteractionTextEmbedding
import numpy as np
from config import URL_LINK
from result.display_result import display_multivector_results

# ========================
# 1. Qdrant Bağlantısı
# ========================
COLLECTION_NAME = "mng"
client = QdrantClient(URL_LINK)

# ========================
# 2. Embed Modelleri
# ========================
dense_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")  # Dense embedding
colbert_model = LateInteractionTextEmbedding(model_name="colbert-ir/colbertv2.0")  # ColBERT

# ========================
# 3. Arama Fonksiyonu
# ========================
def search_tours(query: str, top_k: int = 5):  # Varsayılan değer 5'e çıkarıldı
    # Embed sorguyu al
    dense_query = list(dense_model.embed([query]))[0]  # numpy array
    colbert_query = list(colbert_model.embed([query]))[0]  # numpy array

    # NumPy array -> list çevir ve formatı düzelt
    dense_query = dense_query.tolist()

    # ColBERT vektörünü düzleştir - nested list olabilir
    if isinstance(colbert_query, np.ndarray):
        colbert_query = colbert_query.tolist()

    # Eğer hala nested list ise, düzleştir
    if colbert_query and isinstance(colbert_query[0], list):
        # İlk vektörü al (genellikle ColBERT'te ilk vektör kullanılır)
        colbert_query = colbert_query[0]

    # Vektörlerin float olduğundan emin ol
    dense_query = [float(x) for x in dense_query]
    colbert_query = [float(x) for x in colbert_query]

    # Multi-vector arama yap
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=models.Prefetch(
            query=dense_query,
            using="text-dense",
        ),
        query=colbert_query,
        using="colbert",  # Mevcut collection'da colbert alanı kullanılıyor
        limit=top_k,  # Artık top_k kullanılıyor
        with_payload=True,
    )

    # Sistemin beklediği formatta döndür
    points_data = []
    for r in results.points:
        point_data = {
            "id": r.id,
            "score": r.score,
            "payload": r.payload
        }
        points_data.append(point_data)

    result = {
        "success": True,
        "method": "multivector",
        "data": points_data,
        "count": len(points_data)
    }
    
    # Display fonksiyonunu çağır
    display_multivector_results(result)
    
    return result

def search_tours_by_field(query: str, field: str = "colbert", top_k: int = 5):  # Varsayılan değer 5'e çıkarıldı
    """
    Belirli bir alanda arama yapar (şu anda sadece colbert alanı destekleniyor)
    """
    # Embed sorguyu al
    colbert_query = list(colbert_model.embed([query]))[0]
    
    # NumPy array -> list çevir
    if isinstance(colbert_query, np.ndarray):
        colbert_query = colbert_query.tolist()
    
    # Eğer hala nested list ise, düzleştir
    if colbert_query and isinstance(colbert_query[0], list):
        colbert_query = colbert_query[0]
    
    # Vektörün float olduğundan emin ol
    colbert_query = [float(x) for x in colbert_query]
    
    # Belirli alanda arama yap
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=colbert_query,
        using=field,  # Hangi alanda arama yapılacağı
        limit=top_k,  # Artık top_k kullanılıyor
        with_payload=True,
    )
    
    # Sonuçları yazdır
    print(f"\n🔎 Sorgu: {query}")
    print(f"🎯 Arama Alanı: {field}")
    print()
    
    # Sistemin beklediği formatta döndür
    points_data = []
    for r in results.points:
        point_data = {
            "id": r.id,
            "score": r.score,
            "payload": r.payload
        }
        points_data.append(point_data)
    
    result = {
        "success": True,
        "method": f"field_search_{field}",
        "data": points_data,
        "count": len(points_data)
    }
    
    # Display fonksiyonunu çağır
    display_multivector_results(result)
    
    return result