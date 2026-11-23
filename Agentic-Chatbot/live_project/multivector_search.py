# region library
from qdrant_client import AsyncQdrantClient, models
from fastembed import (
    TextEmbedding,
    SparseTextEmbedding
    )
from setting import DENSE_MODEL,SPARSE_MODEL,RERANKER_MODEL,QDRANT_COLLECTION
from fastembed.rerank.cross_encoder import TextCrossEncoder

import os,sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from connection_sql import KeyManager  
# 🔑 API anahtarlarını yükle
key_manager = KeyManager()
api_keys = key_manager.load_api_key()
QDRANT_HOST = api_keys.get('qdrant_host', '')
QDRANT_API_KEY = api_keys.get('qdrant_api_key', '')


# =====================
# 3. Embed Modelleri
# =====================
dense_model = TextEmbedding(model_name=DENSE_MODEL)
sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL)

# =====================
# 2. Qdrant Bağlantısı
# =====================
client = AsyncQdrantClient(url=QDRANT_HOST, api_key=QDRANT_API_KEY, check_compatibility=False,timeout=60)

# Lazy load için global değişken
_reranker = None

# endregion


def get_reranker():
    global _reranker
    if _reranker is None:
        print("🔄 Reranker yükleniyor...")
        _reranker = TextCrossEncoder(
            model_name=RERANKER_MODEL,
        )
    return _reranker

# =====================
# RERANK
# =====================
def rerank_results(query: str, results: list, top_k: int = 10, 
                rerank_fields: list = None):
    """
    Sonuçları rerank eder.
    
    Args:
        query: Kullanıcı sorgusu
        results: Qdrant'dan dönen sonuçlar
        top_k: Döndürülecek maksimum sonuç sayısı
        rerank_fields: Rerank için kullanılacak alanlar (None ise default)
    """
    if not results:
        return []

    reranker = get_reranker()
    
    # Default rerank alanları
    if rerank_fields is None:
        rerank_fields = ['isim', 'turtipi', 'gecesayisi', 'konaklama.tam_metin',
                        'ulasim', 'ziyaretedilecekyerler', 'metin']

    documents = []
    for r in results:
        payload = r.payload or {}
        
        doc_parts = []
        for field in rerank_fields:
            if '.' in field:
                # Nested alan (örn: konaklama.tam_metin)
                parts = field.split('.')
                value = payload
                for part in parts:
                    if isinstance(value, dict):
                        value = value.get(part, '')
                    else:
                        value = ''
                        break
                if value:
                    doc_parts.append(str(value))
            else:
                value = payload.get(field, '')
                if isinstance(value, list):
                    # Liste ise virgülle birleştir
                    value = ', '.join(str(v) for v in value if v)
                if value:
                    doc_parts.append(str(value))
        
        doc_text = " ".join(doc_parts).strip()
        documents.append(doc_text if doc_text else "Bilgi yok")

    scores = reranker.rerank(
        query=query,
        documents=documents
    )

    # ✅ ScoredPoint'e dokunmuyoruz
    scored_results = list(zip(results, scores))

    # rerank skoruna göre sırala
    scored_results.sort(key=lambda x: x[1], reverse=True)

    # sadece top_k al
    top_results = scored_results[:top_k]
    
    
    # 🔥 BURASI ÖNEMLİ: payload filtreleme
    filtered_results = []
    for r, score in top_results:
        payload = r.payload or {}

        filtered_results.append({
            "id": payload.get("id"),
            "isim": payload.get("isim"),
            "gecesayisi": payload.get("gecesayisi"),
            "turtipi": payload.get("turtipi"),
            "konaklama": payload.get("konaklama", {}).get("tam_metin"),
            "score": score
        })


    return filtered_results


async def hybrid_search(collection_name: str, query: str, limit: int = 30):
    try:
        
        # ---------- Dense Query ----------
        dense_raw = list(dense_model.embed([query]))[0]
        dense_vector = dense_raw.tolist()

        # ---------- Sparse Query ----------
        sparse_raw = list(sparse_model.embed([query]))[0]
        sparse_vector = models.SparseVector(
            indices=sparse_raw.indices.tolist(),
            values=sparse_raw.values.tolist()
        )

        # ---------- Prefetch ----------
        prefetch = [
            models.Prefetch(query=dense_vector, using="dense", limit=limit),
            models.Prefetch(query=sparse_vector, using="sparse", limit=limit),
        ]

        # ---------- Hybrid ----------
        response = await client.query_points(
            collection_name=collection_name,
            prefetch=prefetch,
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            with_payload=True,
            limit=limit
        )

        results = response.points

        print(f"✅ {len(results)} aday bulundu (Hybrid)\n")

        # ---------- Rerank ----------
        final_results = rerank_results(
            query=query,
            results=results,
            top_k=10
        )

        print(f"🏆 Rerank Sonuçları:\n {final_results}")
        return final_results

    except Exception as e:
        print(f"❌ Arama hatası: {str(e)}")
        return []


# =====================
# ASYNC HYBRID SEARCH (Filtresiz)
# =====================
async def async_hybrid_search(
    collection_name: str, 
    query: str, 
    limit: int = 30,
    top_k: int = 10,
    rerank_fields: list = None
) -> list:
    """
    Async hybrid search - sadece vektör araması (filtre yok).
    multivector_only stratejisi için kullanılır.
    
    Args:
        collection_name: Qdrant koleksiyon adı
        query: Kullanıcı sorgusu
        limit: Prefetch limit
        top_k: Rerank sonrası döndürülecek sonuç sayısı
        rerank_fields: Rerank için kullanılacak alanlar
    
    Returns:
        Rerank edilmiş sonuç listesi
    """
    try:
        # ---------- Dense Query ----------
        dense_raw = list(dense_model.embed([query]))[0]
        dense_vector = dense_raw.tolist()

        # ---------- Sparse Query ----------
        sparse_raw = list(sparse_model.embed([query]))[0]
        sparse_vector = models.SparseVector(
            indices=sparse_raw.indices.tolist(),
            values=sparse_raw.values.tolist()
        )

        # ---------- Prefetch ----------
        prefetch = [
            models.Prefetch(query=dense_vector, using="dense", limit=limit),
            models.Prefetch(query=sparse_vector, using="sparse", limit=limit),
        ]

        # ---------- Hybrid Query ----------
        response = await client.query_points(
            collection_name=collection_name,
            prefetch=prefetch,
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            with_payload=True,
            limit=limit,
            # ✅ HNSW arama parametreleri
            search_params=models.SearchParams(
                hnsw_ef=128,
                exact=False,
                quantization=models.QuantizationSearchParams(
                    rescore=True,
                    oversampling=1.5,
                )
            )
        )

        results = response.points
        print(f"✅ {len(results)} aday bulundu (Async Hybrid)\n")

        # ---------- Rerank ----------
        final_results = rerank_results(
            query=query,
            results=results,
            top_k=top_k,
            rerank_fields=rerank_fields
        )

        return final_results

    except Exception as e:
        print(f"❌ Async hybrid search hatası: {str(e)}")
        return []


# =====================
# ASYNC FILTERED HYBRID SEARCH (Hem vektör hem filtre)
# =====================
async def async_filtered_hybrid_search(
    collection_name: str, 
    query: str,
    query_filter: models.Filter,
    limit: int = 30,
    top_k: int = 10,
    rerank_fields: list = None
) -> list:
    """
    Async filtered hybrid search - hem vektör hem filtre ile arama.
    hybrid_combined stratejisi için kullanılır.
    
    Bu fonksiyon client.search() yerine query_points + filter kullanır.
    Qdrant'ın prefetch + filter kombinasyonunu destekler.
    
    Args:
        collection_name: Qdrant koleksiyon adı
        query: Kullanıcı sorgusu
        query_filter: Qdrant Filter objesi
        limit: Prefetch limit
        top_k: Rerank sonrası döndürülecek sonuç sayısı
        rerank_fields: Rerank için kullanılacak alanlar
    
    Returns:
        Rerank edilmiş sonuç listesi
    """
    try:
        # ---------- Dense Query ----------
        dense_raw = list(dense_model.embed([query]))[0]
        dense_vector = dense_raw.tolist()

        # ---------- Sparse Query ----------
        sparse_raw = list(sparse_model.embed([query]))[0]
        sparse_vector = models.SparseVector(
            indices=sparse_raw.indices.tolist(),
            values=sparse_raw.values.tolist()
        )

        # ---------- Prefetch with Filter ----------
        # Her bir prefetch'e filter ekliyoruz
        prefetch = [
            models.Prefetch(
                query=dense_vector, 
                using="dense", 
                limit=limit,
                filter=query_filter
            ),
            models.Prefetch(
                query=sparse_vector, 
                using="sparse", 
                limit=limit,
                filter=query_filter
            ),
        ]

        # ---------- Hybrid Query with Filter ----------
        response = await client.query_points(
            collection_name=collection_name,
            prefetch=prefetch,
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            query_filter=query_filter,  # Ana query'ye de filter ekle
            with_payload=True,
            limit=limit,
            # ✅ HNSW arama parametreleri
            search_params=models.SearchParams(
                hnsw_ef=128,
                exact=False,
                quantization=models.QuantizationSearchParams(
                    rescore=True,
                    oversampling=1.5,
                )
            )
        )

        results = response.points
        print(f"✅ {len(results)} aday bulundu (Async Filtered Hybrid)\n")

        # ---------- Rerank ----------
        final_results = rerank_results(
            query=query,
            results=results,
            top_k=top_k,
            rerank_fields=rerank_fields
        )

        return final_results

    except Exception as e:
        print(f"❌ Async filtered hybrid search hatası: {str(e)}")
        return []


# =====================
# ASYNC DENSE SEARCH WITH FILTER (Basit vektör + filtre)
# =====================
async def async_dense_search_with_filter(
    collection_name: str, 
    query: str,
    query_filter: models.Filter,
    limit: int = 10
) -> list:
    """
    Basit dense vektör araması + filtre.
    query_points ile hem vektör hem filtre birlikte kullanılır.
    
    Args:
        collection_name: Qdrant koleksiyon adı
        query: Kullanıcı sorgusu
        query_filter: Qdrant Filter objesi
        limit: Döndürülecek sonuç sayısı
    
    Returns:
        ScoredPoint listesi
    """
    try:
        # ---------- Dense Query ----------
        dense_raw = list(dense_model.embed([query]))[0]
        dense_vector = dense_raw.tolist()

        # ---------- Query Points with Filter ----------
        # AsyncQdrantClient'ta search yerine query_points kullanılır
        response = await client.query_points(
            collection_name=collection_name,
            query=dense_vector,
            using="dense",
            query_filter=query_filter,
            with_payload=True,
            limit=limit,
            # ✅ HNSW arama parametreleri
            search_params=models.SearchParams(
                hnsw_ef=128,
                exact=False,
                quantization=models.QuantizationSearchParams(
                    rescore=True,
                    oversampling=1.5,
                )
            )
        )

        results = response.points
        print(f"✅ {len(results)} sonuç bulundu (Dense + Filter)\n")
        return results

    except Exception as e:
        print(f"❌ Dense search with filter hatası: {str(e)}")
        return []


