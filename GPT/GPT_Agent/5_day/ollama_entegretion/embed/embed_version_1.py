import json
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
import qdrant_client
from qdrant_client import models
import gc
import os
from dotenv import load_dotenv

# .env dosyasını yükle (hata olursa devam et)
try:
    load_dotenv()
    print(".env dosyası yüklendi")
except Exception as e:
    print(f".env dosyası yüklenemedi: {e}")
    print("Varsayılan ayarlar kullanılıyor")

# Qdrant konfigürasyonu
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_HOST = os.getenv("QDRANT_HOST", "http://QDRANT_HOST:6333")

# Qdrant istemcisi oluştur
if QDRANT_API_KEY and QDRANT_API_KEY != "your_api_key_here":
    print(f"Qdrant'a bağlanılıyor: {QDRANT_HOST}")
    client = qdrant_client.QdrantClient(
        url=QDRANT_HOST,
        api_key=QDRANT_API_KEY
    )
else:
    print("Bellek içi Qdrant kullanılıyor")
    client = qdrant_client.QdrantClient(":memory:")

# Konfigürasyon
class CFG:
    embedding_size = 1024  # BGE-M3 model boyutu
    collection_name = "turizm_turlari"

def create_qdrant_collection(collection_name: str, client=client):
    """Turizm verileri için Qdrant koleksiyonu oluştur"""
    try:
        # Önce koleksiyonun var olup olmadığını kontrol et
        collections = client.get_collections()
        collection_exists = any(col.name == collection_name for col in collections.collections)
        
        if collection_exists:
            print(f"Koleksiyon '{collection_name}' zaten mevcut, kullanılıyor...")
            info = client.get_collection(collection_name=collection_name)
            print(f"Mevcut koleksiyon bilgisi - {info}")
            return True
        else:
            # Koleksiyonu oluştur
            client.create_collection(
                collection_name=collection_name,
                # Dense vector index
                vectors_config={
                    "text-dense": models.VectorParams(
                        size=CFG.embedding_size, 
                        distance=qdrant_client.models.Distance.COSINE
                    )
                },
                # Sparse vector index - her alan için ayrı sparse vector
                sparse_vectors_config={
                    "isim_sparse": models.SparseVectorParams(
                        index=qdrant_client.models.SparseIndexParams(on_disk=False)
                    ),
                    "metin_sparse": models.SparseVectorParams(
                        index=qdrant_client.models.SparseIndexParams(on_disk=False)
                    ),
                    "geceleme_sparse": models.SparseVectorParams(
                        index=qdrant_client.models.SparseIndexParams(on_disk=False)
                    ),
                    "konaklama_sparse": models.SparseVectorParams(
                        index=qdrant_client.models.SparseIndexParams(on_disk=False)
                    ),
                    "ulasim_sparse": models.SparseVectorParams(
                        index=qdrant_client.models.SparseIndexParams(on_disk=False)
                    ),
                    "ziyaretedilecekyerler_sparse": models.SparseVectorParams(
                        index=qdrant_client.models.SparseIndexParams(on_disk=False)
                    ),
                    "vizesiz_sparse": models.SparseVectorParams(
                        index=qdrant_client.models.SparseIndexParams(on_disk=False)
                    ),
                    "kesinkalkis_sparse": models.SparseVectorParams(
                        index=qdrant_client.models.SparseIndexParams(on_disk=False)
                    ),
                    "ulasimtipi_sparse": models.SparseVectorParams(
                        index=qdrant_client.models.SparseIndexParams(on_disk=False)
                    ),
                    "turtipi_sparse": models.SparseVectorParams(
                        index=qdrant_client.models.SparseIndexParams(on_disk=False)
                    ),
                    "turKategori_sparse": models.SparseVectorParams(
                        index=qdrant_client.models.SparseIndexParams(on_disk=False)
                    )
                }
            )
            info = client.get_collection(collection_name=collection_name)
            print(f"Yeni koleksiyon oluşturuldu - {info}")
            return True
            
    except Exception as e:
        print(f"Koleksiyon oluşturma hatası: {e}")
        print("Mevcut koleksiyon kullanılmaya çalışılıyor...")
        try:
            info = client.get_collection(collection_name=collection_name)
            print(f"Mevcut koleksiyon bilgisi - {info}")
            return True
        except Exception as e2:
            print(f"Koleksiyon erişim hatası: {e2}")
            return False

# Koleksiyon oluştur
create_qdrant_collection(CFG.collection_name)

# Payload index oluştur (hata olursa devam et)
try:
    client.create_payload_index(
        collection_name=CFG.collection_name,
        field_name="tur_id",
        field_schema="keyword",
    )
    print("Payload index oluşturuldu")
except Exception as e:
    print(f"Payload index oluşturma hatası (muhtemelen zaten mevcut): {e}")

# Modelleri yükle
print("Modeller yükleniyor...")

# CUDA kontrolü
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Kullanılan cihaz: {device}")

# CPU için optimize edilmiş ayarlar
if device == "cpu":
    print("CPU modu: Bellek ve hız optimizasyonları uygulanıyor...")
    # CPU için daha küçük batch size
    BATCH_SIZE = 8
    # CPU için daha kısa max_length
    MAX_LENGTH = 256
else:
    BATCH_SIZE = 32
    MAX_LENGTH = 512

# Dense embedding modeli (BGE-M3)
print("BGE-M3 modeli yükleniyor...")
tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")
model = AutoModel.from_pretrained(
    "BAAI/bge-m3",
    torch_dtype=torch.float32,  # CPU için float32 kullan
    attn_implementation="eager"  # CPU için eager kullan
).to(device)

# ColBERT modeli yükle (AutoTokenizer ve AutoModel ile)
print("ColBERT modeli yükleniyor...")
sparse_tokenizer = AutoTokenizer.from_pretrained('colbert-ir/colbertv2.0')
sparse_model = AutoModel.from_pretrained('colbert-ir/colbertv2.0')
sparse_model.eval()
sparse_model.to(device)

def get_embedding(input_texts: [str], model=model):
    """BGE-M3 ile dense embedding hesaplama"""
    if isinstance(input_texts, str):
        input_texts = [input_texts]
    
    # BGE-M3 için prefix ekle
    prefixed_texts = [f"Represent this sentence: {text}" for text in input_texts]
    
    batch_dict = tokenizer(
        prefixed_texts, 
        max_length=MAX_LENGTH,  # CPU için optimize edilmiş length
        padding=True, 
        truncation=True, 
        return_tensors='pt'
    ).to(device)
    
    with torch.no_grad():
        outputs = model(**batch_dict)
    
    # [CLS] token'ı kullan
    embeddings = outputs.last_hidden_state[:, 0]
    embeddings = F.normalize(embeddings, p=2, dim=1)
    
    return embeddings.detach().cpu().tolist()

def compute_colbert_sparse_vector(text: str, tokenizer=sparse_tokenizer, model=sparse_model):
    """ColBERT ile sparse embedding hesaplama"""
    if not text.strip():
        return {"indices": [], "values": []}
    
    try:
        # Metni tokenize et
        inputs = tokenizer(
            text, 
            max_length=MAX_LENGTH,  # CPU için optimize edilmiş length
            padding=True, 
            truncation=True, 
            return_tensors='pt'
        ).to(device)
        
        # ColBERT embedding hesapla
        with torch.no_grad():
            # Model çıktısını al
            outputs = model(**inputs)
            
            # Token-level embeddings'leri al
            token_embeddings = outputs.last_hidden_state  # [1, seq_len, hidden_dim]
            
            # Her token için pooling yaparak sparse vector oluştur
            # ColBERT'ın sparse representation'ına benzer yaklaşım
            sparse_embeddings = torch.mean(token_embeddings, dim=2)  # [1, seq_len]
            sparse_vec = sparse_embeddings.squeeze(0)  # [seq_len]
            
            # Sıfır olmayan değerleri al
            indices = sparse_vec.nonzero().squeeze()
            values = sparse_vec[indices]
            
            # Eğer hiç sıfır olmayan değer yoksa, en yüksek değerleri al
            if len(indices) == 0:
                # En yüksek 10 değeri al
                top_k = min(10, len(sparse_vec))
                values, indices = torch.topk(sparse_vec, top_k)
        
        if device == "cuda:0":
            torch.cuda.empty_cache()
            
        return {
            "indices": indices.cpu().numpy(),
            "values": values.cpu().numpy()
        }
        
    except Exception as e:
        print(f"Sparse vector hesaplama hatası: {e}")
        print(f"Problemli metin: {text[:100]}...")
        # Hata durumunda boş sparse vector döndür
        return {
            "indices": [],
            "values": []
        }

def load_turizm_data(file_path: str = "birlesik.json"):
    """Turizm verilerini yükle"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def create_dense_text(tur_data):
    """Dense embedding için tüm alanları birleştir"""
    dense_text = f"{tur_data.get('isim', '')} "
    dense_text += f"{tur_data.get('metin', '')} "
    dense_text += f"{tur_data.get('geceleme', '')} "
    dense_text += f"{tur_data.get('konaklama', '')} "
    dense_text += f"{tur_data.get('ulasim', '')} "
    dense_text += f"{tur_data.get('ziyaretedilecekyerler', '')} "
    dense_text += f"{tur_data.get('vizesiz', '')} "
    dense_text += f"{tur_data.get('kesinkalkis', '')} "
    dense_text += f"{tur_data.get('ulasimtipi', '')} "
    dense_text += f"{tur_data.get('turtipi', '')} "
    
    # Kategorileri ekle
    if 'turKategori' in tur_data:
        for kategori in tur_data['turKategori']:
            dense_text += f"{kategori.get('isim', '')} "
    
    return dense_text.strip()

def create_sparse_texts(tur_data):
    """Her alan için ayrı sparse text oluştur"""
    sparse_texts = {
        "isim_sparse": tur_data.get('isim', ''),
        "metin_sparse": tur_data.get('metin', ''),
        "geceleme_sparse": tur_data.get('geceleme', ''),
        "konaklama_sparse": tur_data.get('konaklama', ''),
        "ulasim_sparse": tur_data.get('ulasim', ''),
        "ziyaretedilecekyerler_sparse": tur_data.get('ziyaretedilecekyerler', ''),
        "vizesiz_sparse": tur_data.get('vizesiz', ''),
        "kesinkalkis_sparse": tur_data.get('kesinkalkis', ''),
        "ulasimtipi_sparse": tur_data.get('ulasimtipi', ''),
        "turtipi_sparse": tur_data.get('turtipi', ''),
        "turKategori_sparse": ""
    }
    
    # Kategorileri birleştir
    if 'turKategori' in tur_data:
        kategori_texts = []
        for kategori in tur_data['turKategori']:
            kategori_texts.append(kategori.get('isim', ''))
        sparse_texts["turKategori_sparse"] = " ".join(kategori_texts)
    
    return sparse_texts

def process_turizm_data():
    """Turizm verilerini işle ve Qdrant'a yükle"""
    print("Turizm verileri yükleniyor...")
    
    # Önce koleksiyondaki mevcut veri sayısını kontrol et
    try:
        count_result = client.count(collection_name=CFG.collection_name, exact=True)
        count = count_result.count if hasattr(count_result, 'count') else count_result
        print(f"Mevcut koleksiyonda {count} tur bulunuyor")
        
        if count > 0:
            print("Koleksiyon zaten dolu, veri yükleme atlanıyor...")
            return
    except Exception as e:
        print(f"Koleksiyon sayım hatası: {e}")
    
    turizm_data = load_turizm_data()
    print(f"Toplam {len(turizm_data)} tur bulundu")
    
    # Dense ve sparse metinleri oluştur
    dense_texts = []
    sparse_texts_list = []
    tur_payloads = []
    
    for tur in turizm_data:
        # Dense text oluştur
        dense_text = create_dense_text(tur)
        dense_texts.append(dense_text)
        
        # Sparse texts oluştur
        sparse_texts = create_sparse_texts(tur)
        sparse_texts_list.append(sparse_texts)
        
        # Payload oluştur (tüm alanları dahil et)
        payload = {
            "tur_id": tur.get('id'),
            "tur_kodu": tur.get('turkodu'),
            "isim": tur.get('isim'),
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
            "dense_text": dense_text
        }
        tur_payloads.append(payload)
    
    print("BGE-M3 dense embedding'ler hesaplanıyor...")
    # Dense embedding'ler
    dense_vectors = []
    
    for i in range(0, len(dense_texts), BATCH_SIZE):
        batch_texts = dense_texts[i:i + BATCH_SIZE]
        batch_embeddings = get_embedding(batch_texts)
        dense_vectors.extend(batch_embeddings)
        
        if i % (BATCH_SIZE * 10) == 0:
            print(f"Dense embedding {i}/{len(dense_texts)} tamamlandı")
    
    print("ColBERT sparse embedding'ler hesaplanıyor...")
    # Sparse embedding'ler - her alan için ayrı ayrı
    sparse_vectors = []
    
    for i, sparse_texts in enumerate(sparse_texts_list):
        if i % 50 == 0:  # CPU için daha sık progress
            print(f"ColBERT sparse embedding {i}/{len(sparse_texts_list)}")
        
        sparse_vec_dict = {}
        for field_name, text in sparse_texts.items():
            try:
                sparse_vec = compute_colbert_sparse_vector(text)
                sparse_vec_dict[field_name] = sparse_vec
            except Exception as e:
                print(f"Alan '{field_name}' için sparse vector hesaplama hatası: {e}")
                sparse_vec_dict[field_name] = {"indices": [], "values": []}
        
        sparse_vectors.append(sparse_vec_dict)
    
    print("Qdrant'a veriler yükleniyor...")
    # Qdrant'a noktaları yükle
    points = []
    for i in range(len(turizm_data)):
        # Sparse vectors dictionary'sini oluştur
        sparse_vectors_dict = {}
        for field_name in sparse_vectors[i].keys():
            # indices ve values'ları güvenli şekilde list'e çevir
            indices = sparse_vectors[i][field_name]["indices"]
            values = sparse_vectors[i][field_name]["values"]
            
            # numpy array ise tolist() çağır, değilse olduğu gibi kullan
            if hasattr(indices, 'tolist'):
                indices = indices.tolist()
            if hasattr(values, 'tolist'):
                values = values.tolist()
            
            sparse_vectors_dict[field_name] = qdrant_client.models.SparseVector(
                indices=indices,
                values=values
            )
        
        point = qdrant_client.models.PointStruct(
            id=int(turizm_data[i]['id']),
            vector={
                "text-dense": dense_vectors[i],
                **sparse_vectors_dict  # Tüm sparse vector'ları ekle
            },
            payload=tur_payloads[i]
        )
        points.append(point)
    
    # Batch halinde yükle
    batch_size = 1000
    for i in range(0, len(points), batch_size):
        batch = points[i:i + batch_size]
        client.upsert(
            collection_name=CFG.collection_name,
            points=batch
        )
        print(f"Batch {i//batch_size + 1} yüklendi ({len(batch)} tur)")
    
    # Belleği temizle
    del points, dense_vectors, sparse_vectors, turizm_data
    gc.collect()
    
    print("Veri yükleme tamamlandı!")

def search_tours(query: str, limit: int = 10):
    """Hybrid arama fonksiyonu"""
    # Query için dense embedding
    query_dense = get_embedding([query])[0]
    
    # Query için ColBERT sparse embedding
    query_sparse = compute_colbert_sparse_vector(query)
    
    # indices ve values'ları güvenli şekilde list'e çevir
    indices = query_sparse["indices"]
    values = query_sparse["values"]
    
    if hasattr(indices, 'tolist'):
        indices = indices.tolist()
    if hasattr(values, 'tolist'):
        values = values.tolist()
    
    query_sparse_vector = qdrant_client.models.SparseVector(
        indices=indices,
        values=values
    )
    
    # Tüm sparse alanlar için aynı query vector'ı kullan
    sparse_vectors_dict = {}
    sparse_fields = [
        "isim_sparse", "metin_sparse", "geceleme_sparse", "konaklama_sparse",
        "ulasim_sparse", "ziyaretedilecekyerler_sparse", "vizesiz_sparse",
        "kesinkalkis_sparse", "ulasimtipi_sparse", "turtipi_sparse", "turKategori_sparse"
    ]
    
    for field in sparse_fields:
        sparse_vectors_dict[field] = query_sparse_vector
    
    # Hybrid arama
    search_result = client.search(
        collection_name=CFG.collection_name,
        query_vector={
            "text-dense": query_dense,
            **sparse_vectors_dict
        },
        limit=limit,
        with_payload=True,
        with_vectors=False
    )
    
    return search_result

def search_tours_dense_only(query: str, limit: int = 10):
    """Sadece dense embedding ile arama"""
    query_dense = get_embedding([query])[0]
    
    search_result = client.search(
        collection_name=CFG.collection_name,
        query_vector={"text-dense": query_dense},
        limit=limit,
        with_payload=True,
        with_vectors=False
    )
    
    return search_result

def search_tours_sparse_only(query: str, limit: int = 10):
    """Sadece sparse embedding ile arama"""
    query_sparse = compute_colbert_sparse_vector(query)
    
    # indices ve values'ları güvenli şekilde list'e çevir
    indices = query_sparse["indices"]
    values = query_sparse["values"]
    
    if hasattr(indices, 'tolist'):
        indices = indices.tolist()
    if hasattr(values, 'tolist'):
        values = values.tolist()
    
    query_sparse_vector = qdrant_client.models.SparseVector(
        indices=indices,
        values=values
    )
    
    # Tüm sparse alanlar için aynı query vector'ı kullan
    sparse_vectors_dict = {}
    sparse_fields = [
        "isim_sparse", "metin_sparse", "geceleme_sparse", "konaklama_sparse",
        "ulasim_sparse", "ziyaretedilecekyerler_sparse", "vizesiz_sparse",
        "kesinkalkis_sparse", "ulasimtipi_sparse", "turtipi_sparse", "turKategori_sparse"
    ]
    
    for field in sparse_fields:
        sparse_vectors_dict[field] = query_sparse_vector
    
    search_result = client.search(
        collection_name=CFG.collection_name,
        query_vector=sparse_vectors_dict,
        limit=limit,
        with_payload=True,
        with_vectors=False
    )
    
    return search_result

def search_tours_with_filters(query: str, ulasim_tipi: str = None, gece_sayisi: str = None, limit: int = 10):
    """Filtreli arama"""
    query_dense = get_embedding([query])[0]
    
    # Query için ColBERT sparse embedding
    query_sparse = compute_colbert_sparse_vector(query)
    query_sparse_vector = qdrant_client.models.SparseVector(
        indices=query_sparse["indices"].tolist(),
        values=query_sparse["values"].tolist()
    )
    
    # Filtre oluştur
    must_conditions = []
    if ulasim_tipi:
        must_conditions.append(
            models.FieldCondition(
                key="ulasimtipi",
                match=models.MatchValue(value=ulasim_tipi)
            )
        )
    if gece_sayisi:
        must_conditions.append(
            models.FieldCondition(
                key="gecesayisi",
                match=models.MatchValue(value=gece_sayisi)
            )
        )
    
    # Tüm sparse alanlar için aynı query vector'ı kullan
    sparse_vectors_dict = {}
    sparse_fields = [
        "isim_sparse", "metin_sparse", "geceleme_sparse", "konaklama_sparse",
        "ulasim_sparse", "ziyaretedilecekyerler_sparse", "vizesiz_sparse",
        "kesinkalkis_sparse", "ulasimtipi_sparse", "turtipi_sparse", "turKategori_sparse"
    ]
    
    for field in sparse_fields:
        sparse_vectors_dict[field] = query_sparse_vector
    
    search_result = client.search(
        collection_name=CFG.collection_name,
        query_vector={
            "text-dense": query_dense,
            **sparse_vectors_dict
        },
        query_filter=models.Filter(must=must_conditions) if must_conditions else None,
        limit=limit,
        with_payload=True,
        with_vectors=False
    )
    
    return search_result

# Ana işlem
if __name__ == "__main__":
    # Verileri işle ve yükle
    process_turizm_data()
    
    # Koleksiyon istatistikleri
    try:
        count_result = client.count(
            collection_name=CFG.collection_name,
            exact=True
        )
        count = count_result.count if hasattr(count_result, 'count') else count_result
        print(f"Toplam {count} tur yüklendi")
    except Exception as e:
        print(f"Koleksiyon sayım hatası: {e}")
        print("Arama örnekleri gösteriliyor...")
    
    print("\nVeri yükleme işlemi tamamlandı!")
    print("Sistem kullanıma hazır!") 