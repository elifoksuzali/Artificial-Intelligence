"""
🚀 Qdrant HNSW + Payload Index Performans Testi
================================================
Bu script, optimizasyon öncesi/sonrası arama hızlarını karşılaştırır.

Kullanım:
    python benchmark_test.py

NOT: Önce .env dosyasında QDRANT_COLLECTION değerinin doğru olduğundan emin ol.
"""

import asyncio
import time
import statistics
from qdrant_client import AsyncQdrantClient, models
from fastembed import TextEmbedding, SparseTextEmbedding
from setting import DENSE_MODEL, SPARSE_MODEL, QDRANT_COLLECTION

import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from connection_sql import KeyManager

# 🔑 API anahtarlarını yükle
key_manager = KeyManager()
api_keys = key_manager.load_api_key()
QDRANT_HOST = api_keys.get('qdrant_host', '')
QDRANT_API_KEY = api_keys.get('qdrant_api_key', '')

# =====================
# Embed Modelleri
# =====================
print("🔄 Embed modelleri yükleniyor...")
dense_model = TextEmbedding(model_name=DENSE_MODEL)
sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL)
print("✅ Embed modelleri hazır!\n")

# =====================
# Qdrant Client
# =====================
client = AsyncQdrantClient(url=QDRANT_HOST, api_key=QDRANT_API_KEY, check_compatibility=False, timeout=60)

# =====================
# Test Sorguları (Gerçekçi senaryolar)
# =====================
TEST_QUERIES = [
    "7 günlük yurt dışı turları",
    "Kapadokya turu otobüslü",
    "3 yıldızlı otel konaklamalı tur",
    "vizesiz yurt dışı turlar",
    "Antalya deniz tatili uygun fiyatlı",
]

# =====================
# Test Filtreleri
# =====================
TEST_FILTERS = [
    # Basit filtre
    models.Filter(
        must=[
            models.FieldCondition(key="gecesayisi", match=models.MatchValue(value=7))
        ]
    ),
    # Keyword filtre
    models.Filter(
        must=[
            models.FieldCondition(key="turtipi", match=models.MatchAny(any=["Yurt Dışı", "yurt dışı"]))
        ]
    ),
    # Kombine filtre
    models.Filter(
        must=[
            models.FieldCondition(key="gecesayisi", match=models.MatchValue(value=5)),
            models.FieldCondition(key="kesinkalkis", match=models.MatchValue(value="Bu tur kesin kalkışlıdır.")),
        ]
    ),
]


def embed_dense(query: str) -> list:
    """Dense embedding oluştur"""
    dense_raw = list(dense_model.embed([query]))[0]
    return dense_raw.tolist()


def embed_sparse(query: str) -> models.SparseVector:
    """Sparse embedding oluştur"""
    sparse_raw = list(sparse_model.embed([query]))[0]
    return models.SparseVector(
        indices=sparse_raw.indices.tolist(),
        values=sparse_raw.values.tolist()
    )


async def benchmark_dense_search(collection_name: str, query: str, query_filter: models.Filter = None) -> float:
    """Dense search süresini ölç (sadece Qdrant sorgu süresi, embedding hariç)"""
    dense_vector = embed_dense(query)
    
    start = time.perf_counter()
    response = await client.query_points(
        collection_name=collection_name,
        query=dense_vector,
        using="dense",
        query_filter=query_filter,
        with_payload=True,
        limit=20,
        search_params=models.SearchParams(
            hnsw_ef=128,
            exact=False,
            quantization=models.QuantizationSearchParams(
                rescore=True,
                oversampling=1.5,
            )
        )
    )
    elapsed = time.perf_counter() - start
    return elapsed, len(response.points)


async def benchmark_dense_search_exact(collection_name: str, query: str, query_filter: models.Filter = None) -> float:
    """Brute-force (exact) search süresini ölç — karşılaştırma için"""
    dense_vector = embed_dense(query)
    
    start = time.perf_counter()
    response = await client.query_points(
        collection_name=collection_name,
        query=dense_vector,
        using="dense",
        query_filter=query_filter,
        with_payload=True,
        limit=20,
        search_params=models.SearchParams(
            exact=True,  # Brute-force — HNSW kullanmaz
        )
    )
    elapsed = time.perf_counter() - start
    return elapsed, len(response.points)


async def benchmark_hybrid_search(collection_name: str, query: str, query_filter: models.Filter = None) -> float:
    """Hybrid search süresini ölç"""
    dense_vector = embed_dense(query)
    sparse_vector = embed_sparse(query)
    
    if query_filter:
        prefetch = [
            models.Prefetch(query=dense_vector, using="dense", limit=30, filter=query_filter),
            models.Prefetch(query=sparse_vector, using="sparse", limit=30, filter=query_filter),
        ]
    else:
        prefetch = [
            models.Prefetch(query=dense_vector, using="dense", limit=30),
            models.Prefetch(query=sparse_vector, using="sparse", limit=30),
        ]

    start = time.perf_counter()
    response = await client.query_points(
        collection_name=collection_name,
        prefetch=prefetch,
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        query_filter=query_filter,
        with_payload=True,
        limit=20,
        search_params=models.SearchParams(
            hnsw_ef=128,
            exact=False,
            quantization=models.QuantizationSearchParams(
                rescore=True,
                oversampling=1.5,
            )
        )
    )
    elapsed = time.perf_counter() - start
    return elapsed, len(response.points)


async def check_collection_info(collection_name: str):
    """Collection bilgilerini göster"""
    try:
        info = await client.get_collection(collection_name)
        print("=" * 60)
        print(f"📊 Collection Bilgisi: {collection_name}")
        print("=" * 60)
        print(f"  📍 Vektör sayısı  : {info.points_count}")
        print(f"  📍 Durum          : {info.status}")
        print(f"  📍 HNSW config    : {info.config.hnsw_config}")
        
        # Quantization kontrolü
        if info.config.quantization_config:
            print(f"  📍 Quantization   : ✅ Aktif")
        else:
            print(f"  📍 Quantization   : ❌ Yok")
        
        # Payload index kontrolü
        if info.payload_schema:
            indexed_fields = list(info.payload_schema.keys())
            print(f"  📍 Payload index  : ✅ {len(indexed_fields)} alan indexli")
            for field in indexed_fields:
                schema = info.payload_schema[field]
                print(f"      - {field}: {schema}")
        else:
            print(f"  📍 Payload index  : ❌ Yok")
        
        print("=" * 60)
        return info
    except Exception as e:
        print(f"❌ Collection bilgisi alınamadı: {e}")
        return None


async def run_benchmark(collection_name: str, num_runs: int = 3):
    """Ana benchmark fonksiyonu"""
    
    print("\n" + "🚀" * 30)
    print("   QDRANT PERFORMANS TESTİ BAŞLIYOR")
    print("🚀" * 30 + "\n")
    
    # 1) Collection bilgisi
    await check_collection_info(collection_name)
    
    # =============================================
    # TEST 1: Dense Search (Filtresiz)
    # =============================================
    print("\n" + "=" * 60)
    print("🔍 TEST 1: Dense Search (Filtresiz)")
    print("=" * 60)
    
    hnsw_times = []
    exact_times = []
    
    for query in TEST_QUERIES:
        # HNSW search
        hnsw_run_times = []
        for _ in range(num_runs):
            elapsed, count = await benchmark_dense_search(collection_name, query)
            hnsw_run_times.append(elapsed)
        avg_hnsw = statistics.mean(hnsw_run_times)
        hnsw_times.append(avg_hnsw)
        
        # Exact (brute-force) search
        exact_run_times = []
        for _ in range(num_runs):
            elapsed, count = await benchmark_dense_search_exact(collection_name, query)
            exact_run_times.append(elapsed)
        avg_exact = statistics.mean(exact_run_times)
        exact_times.append(avg_exact)
        
        speedup = avg_exact / avg_hnsw if avg_hnsw > 0 else 0
        print(f"  📝 \"{query[:40]}...\"")
        print(f"     HNSW: {avg_hnsw*1000:.1f}ms | Exact: {avg_exact*1000:.1f}ms | Hızlanma: {speedup:.1f}x")
    
    avg_hnsw_all = statistics.mean(hnsw_times)
    avg_exact_all = statistics.mean(exact_times)
    overall_speedup = avg_exact_all / avg_hnsw_all if avg_hnsw_all > 0 else 0
    print(f"\n  🏆 Ortalama: HNSW={avg_hnsw_all*1000:.1f}ms | Exact={avg_exact_all*1000:.1f}ms | Hızlanma: {overall_speedup:.1f}x")
    
    # =============================================
    # TEST 2: Dense Search + Filter
    # =============================================
    print("\n" + "=" * 60)
    print("🔍 TEST 2: Dense Search + Filter (Payload Index Testi)")
    print("=" * 60)
    
    for i, qf in enumerate(TEST_FILTERS):
        query = TEST_QUERIES[i % len(TEST_QUERIES)]
        
        run_times = []
        result_count = 0
        for _ in range(num_runs):
            elapsed, count = await benchmark_dense_search(collection_name, query, query_filter=qf)
            run_times.append(elapsed)
            result_count = count
        avg_time = statistics.mean(run_times)
        
        print(f"  📝 Filtre {i+1}: {avg_time*1000:.1f}ms ({result_count} sonuç)")
    
    # =============================================
    # TEST 3: Hybrid Search (Dense + Sparse + RRF)
    # =============================================
    print("\n" + "=" * 60)
    print("🔍 TEST 3: Hybrid Search (Dense + Sparse + RRF Fusion)")
    print("=" * 60)
    
    hybrid_times = []
    for query in TEST_QUERIES:
        run_times = []
        result_count = 0
        for _ in range(num_runs):
            elapsed, count = await benchmark_hybrid_search(collection_name, query)
            run_times.append(elapsed)
            result_count = count
        avg_time = statistics.mean(run_times)
        hybrid_times.append(avg_time)
        
        print(f"  📝 \"{query[:40]}...\": {avg_time*1000:.1f}ms ({result_count} sonuç)")
    
    avg_hybrid = statistics.mean(hybrid_times)
    print(f"\n  🏆 Ortalama Hybrid: {avg_hybrid*1000:.1f}ms")
    
    # =============================================
    # TEST 4: Hybrid Search + Filter
    # =============================================
    print("\n" + "=" * 60)
    print("🔍 TEST 4: Hybrid Search + Filter")
    print("=" * 60)
    
    for i, qf in enumerate(TEST_FILTERS):
        query = TEST_QUERIES[i % len(TEST_QUERIES)]
        
        run_times = []
        result_count = 0
        for _ in range(num_runs):
            elapsed, count = await benchmark_hybrid_search(collection_name, query, query_filter=qf)
            run_times.append(elapsed)
            result_count = count
        avg_time = statistics.mean(run_times)
        
        print(f"  📝 Filtre {i+1}: {avg_time*1000:.1f}ms ({result_count} sonuç)")
    
    # =============================================
    # ÖZET
    # =============================================
    print("\n" + "=" * 60)
    print("📊 SONUÇ ÖZETİ")
    print("=" * 60)
    print(f"  Dense HNSW (ort.)    : {avg_hnsw_all*1000:.1f}ms")
    print(f"  Dense Exact (ort.)   : {avg_exact_all*1000:.1f}ms")
    print(f"  HNSW Hızlanma        : {overall_speedup:.1f}x")
    print(f"  Hybrid (ort.)        : {avg_hybrid*1000:.1f}ms")
    print(f"  Her test {num_runs}x tekrarlandı (ortalama alındı)")
    print("=" * 60)


if __name__ == "__main__":
    collection = QDRANT_COLLECTION
    if not collection:
        collection = input("Collection adı girin: ").strip()
    
    print(f"\n🎯 Test edilecek collection: {collection}")
    print(f"🔧 Dense model: {DENSE_MODEL}")
    print(f"🔧 Sparse model: {SPARSE_MODEL}")
    print(f"🔧 Qdrant host: {QDRANT_HOST}")
    
    asyncio.run(run_benchmark(collection, num_runs=3))
