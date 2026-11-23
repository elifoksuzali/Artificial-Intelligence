import os
# region requirements

from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding, SparseTextEmbedding
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
from typing import List
import logging
from datetime import datetime
from setting import DENSE_MODEL,SPARSE_MODEL
from connection_sql import KeyManager
import hmac
import hashlib
import base64
import re
# ==========================================
# Logging Ayarları
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# ===================================

# FastAPI Server Başlatma
# =======================
app = FastAPI(title="Qdrant Veri Yükleme API")

# region 1. Qdrant Url ve Api Key Bilgileri Çekme

#key_manager = KeyManager()
#api_keys = key_manager.load_qdrant_api_key()
#QDRANT_HOST = api_keys.get('qdrant_host', '')
#QDRANT_API_KEY = api_keys.get('qdrant_api_key', '')

QDRANT_HOST = "http://localhost:6333"
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# endregion

# =====================
# 2. Qdrant Bağlantısı
# =====================
client = QdrantClient(url=QDRANT_HOST, api_key=QDRANT_API_KEY, check_compatibility=False,timeout=60)

# =====================
# 3. Embed Modelleri
# =====================
dense_model = TextEmbedding(model_name=DENSE_MODEL)
sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL)

# =====================
# 4. Pydantic Model
# =====================
class FirmaLinksRequest(BaseModel):
    firma_kodu: str
    data: List[str]
# =====================


# endregion

# region Step 1: Take new data from url and firm_code 
def fetch_data_from_url(url: str) -> List[dict]:
    """URL'den JSON veri çeker"""
    try:
        logger.info(f"🌐 URL'den veri çekiliyor: {url}")
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        logger.info(f"📥 {len(data)} adet tur verisi çekildi")
        return data
        
    except requests.exceptions.HTTPError as he:
        status_code = he.response.status_code if getattr(he, 'response', None) is not None else None
        if status_code in (400, 404):
            logger.warning(f"⚠️ URL'den veri çekilemedi (skip): {url}, HTTP {status_code}")
            return []
        logger.error(f"❌ URL'den veri çekilemedi: {url}, HTTP Hata: {str(he)}")
        return []
    except Exception as e:
        logger.error(f"❌ URL'den veri çekilemedi: {url}, Hata: {str(e)}")
        return []

# endregion

# region Step 2: Create date version 
def create_date_version():
    """Bugünün tarihini alır ve versiyon formatında döndürür"""
    try:
        today = datetime.now()
        date_version = today.strftime("%d.%m.%y")  # 29.08.25 formatında
        
        logger.info(f"📅 Tarih versiyonu oluşturuldu: {date_version}")
        
        return date_version
        
    except Exception as e:
        logger.error(f"❌ Tarih versiyonu oluşturulamadı: {str(e)}")
        return None

# endregion

# region Step 3: List collection
def list_collection():
    """
    Qdrant'ta bulunan collectionları listeler
    """
    try:
        # GET /collections endpoint'ini kullanarak collection listesini al
        collections = client.get_collections()
        logger.info(f"📋 Qdrant'ta bulunan collectionlar:")
        for collection in collections.collections:
            logger.info(f"   - {collection.name}")
        return collections.collections
        
    except Exception as e:
        logger.error(f"❌ Collection listesi alınamadı: {str(e)}")
        return None
# endregion

# region Step 4: Compare firma_kodu with collection names
def compare_firma_with_collections(firm_code: str):
    """
    Firma kodu ile collection isimlerini karşılaştırır
    İlk _ karakterinden önceki kısmı alarak eşleşme arar
    """
    try:
        logger.info("🔍 Firma kodu ile collection isimleri karşılaştırılıyor...")
        
        # Step 3: Collection listesini al
        collection_list = list_collection()
        
        matching_collections = []
        
        if collection_list:
            for collection in collection_list:
                collection_name = collection.name
                # İlk _ karakterinden önceki kısmı al
                base_name = collection_name.split('_')[0] if '_' in collection_name else collection_name
                
                logger.info(f"📋 Collection: {collection_name} -> Base name: {base_name}")
                
                if base_name.lower() == firm_code.lower():
                    matching_collections.append(collection_name)
                    logger.info(f"✅ Eşleşme bulundu: {firm_code} = {collection_name}")
                else:
                    logger.info(f"❌ Eşleşme yok: {firm_code} != {collection_name}")
        
        logger.info(f"🎯 Toplam {len(matching_collections)} adet eşleşen collection bulundu")
        return matching_collections, collection_list
        
    except Exception as e:
        logger.error(f"❌ Collection karşılaştırma hatası: {str(e)}")
        return [], None

# endregion

# region Step 5: firm_code + date version create new collection
def parse_konaklama(raw: str) -> dict:
    raw_lower = raw.lower()
    result = {
        "tam_metin": raw,
        "otel_yildiz": [],
        "konaklama_tipi": [],
        "pansiyon_tipi": [],
        "konum_tipi": [],
        "otel_markasi": []
    }
    # -----------------
    # Yıldız
    # -----------------
    stars = re.findall(r'(\d)\s*\*', raw)
    result["otel_yildiz"] = sorted(set(int(s) for s in stars))
    # -----------------
    # Konaklama tipi
    # -----------------
    if "otel" in raw_lower:
        result["konaklama_tipi"].append("otel")
    if "bungalov" in raw_lower:
        result["konaklama_tipi"].append("bungalov")
    if "villa" in raw_lower:
        result["konaklama_tipi"].append("villa")
    if "gemi" in raw_lower:
        result["konaklama_tipi"].append("gemi")
    # -----------------
    # Pansiyon tipi
    # -----------------
    if "yarım pansiyon" in raw_lower:
        result["pansiyon_tipi"].append("yarim_pansiyon")
    if "tam pansiyon" in raw_lower:
        result["pansiyon_tipi"].append("tam_pansiyon")
    if "her şey dahil" in raw_lower:
        result["pansiyon_tipi"].append("her_sey_dahil")
    # -----------------
    # Konum tipi
    # -----------------
    if "şehir dışı" in raw_lower:
        result["konum_tipi"].append("sehir_disi")
    if "merkez" in raw_lower:
        result["konum_tipi"].append("merkez")
    return result

def normalize_list_field(raw: str) -> list:

    """
    Ziyaret edilecek yerleri normalize eder ve dizi olarak döndürür.
    - Ayırıcılar: &, -, –, ;, :, ,
    - Parantez içindeki sayıları kaldırır
    - Baş ve sondaki boşlukları temizler
    - Fazla boşlukları tek boşluk yapar
    """
    if not raw or not isinstance(raw, str):
        return []

    # Tüm ayırıcıları ortak bir karaktere çevir
    separators = ['&', '–', '-', ';', ':', ',']
    for sep in separators:
        raw = raw.replace(sep, '|')

    # "|" ile ayır
    items = raw.split('|')

    # Temizle ve parantez içindeki sayıları kaldır, boş olanları at
    result = []
    for item in items:
        clean_item = re.sub(r'\(\d+\)', '', item).strip()  # parantez sayıları kaldır
        clean_item = re.sub(r'\s+', ' ', clean_item)        # fazla boşlukları tek boşluk yap
        if clean_item:
            result.append(clean_item)

    return result

def transform_tur_payload(tur: dict) -> dict:
    """Tur payload'unu normalize eder: tip dönüşümleri, tarih formatlama ve ekstra alanlar."""
    try:
        normalized = dict(tur)
        # ----------------- ID ve sayısal alanlar -----------------
        if 'id' in normalized:
            try:
                normalized['id'] = int(str(normalized['id']).strip())
            except Exception:
                pass
        if 'gecesayisi' in normalized:
            try:
                normalized['gecesayisi'] = int(str(normalized['gecesayisi']).strip())
            except Exception:
                pass
        # ----------------- turKategori puanları -----------------
        if isinstance(normalized.get('turKategori'), list):
            for kategori in normalized['turKategori']:
                if isinstance(kategori, dict) and 'puan' in kategori:
                    try:
                        kategori['puan'] = int(str(kategori['puan']).strip())
                    except Exception:
                        kategori['puan'] = 0
        # ----------------- konaklama detayları -----------------
        if 'konaklama' in normalized and isinstance(normalized['konaklama'], str):
            try:
                normalized['konaklama'] = parse_konaklama(normalized['konaklama'])
            except Exception:
                normalized['konaklama'] = {}
        # ----------------- tarihFiyat -----------------
        tr_months = {
            1: "Ocak", 2: "Şubat", 3: "Mart", 4: "Nisan", 5: "Mayıs", 6: "Haziran",
            7: "Temmuz", 8: "Ağustos", 9: "Eylül", 10: "Ekim", 11: "Kasım", 12: "Aralık"
        }
        if isinstance(normalized.get('tarihFiyat'), list):
            new_tarih_fiyat = []
            for item in normalized['tarihFiyat']:
                if not isinstance(item, dict):
                    continue
                new_item = dict(item)
                if 'kisibasi' in new_item:
                    try:
                        new_item['kisibasi'] = float(str(new_item['kisibasi']).replace(',', '.').strip())
                    except Exception:
                        new_item['kisibasi'] = 0.0
                if 'kalan' in new_item:
                    try:
                        kalan_val = int(str(new_item['kalan']).strip())
                        new_item['kalan'] = max(kalan_val, 0)
                    except Exception:
                        new_item['kalan'] = 0
                if 'tarih' in new_item and isinstance(new_item['tarih'], str):
                    try:
                        year_str, month_str, day_str = new_item['tarih'].split('-')
                        year = int(year_str)
                        month = int(month_str)
                        day = int(day_str)
                        new_item['tarih'] = {
                            "gun": day,
                            "ay": tr_months.get(month, month_str),
                            "yil": year
                        }
                    except Exception:
                        pass
                new_tarih_fiyat.append(new_item)
            normalized['tarihFiyat'] = new_tarih_fiyat
            
        # ----------------- turtipi ve ziyaretedilecekyerler -----------------
        if 'turtipi' in normalized:
            normalized['turtipi'] = normalize_list_field(normalized['turtipi'])
        if 'ziyaretedilecekyerler' in normalized:
            normalized['ziyaretedilecekyerler'] = normalize_list_field(normalized['ziyaretedilecekyerler'])
            
        # ----------------- kesinkalkis ve vizesiz -----------------
        # 0 veya "0" → negatif ifade
        if 'kesinkalkis' in normalized:
            val = str(normalized['kesinkalkis']).strip()
            if val == "0":
                normalized['kesinkalkis'] = "Bu tur kesin kalkışlı değildir."
            else:
                normalized['kesinkalkis'] = "Bu tur kesin kalkışlıdır."
        if 'vizesiz' in normalized:
            val = str(normalized['vizesiz']).strip()
            if val == "0":
                normalized['vizesiz'] = "Bu tur vizesizdir."
            else:
                normalized['vizesiz'] = "Bu tur vizelidir."
        return normalized
    except Exception as e:
        logger.warning(f"⚠️ Payload dönüştürmede sorun: {str(e)}")
        return tur

def build_embedding_text(tur: dict) -> str:
    kategori_isimleri = [
        k.get("isim", "")
        for k in tur.get("turKategori", [])
        if isinstance(k, dict)
    ]

    ziyaretedilecek = tur.get("ziyaretedilecekyerler", [])
    if isinstance(ziyaretedilecek, list):
        ziyaretedilecek = " ".join(ziyaretedilecek)

    turtipi = tur.get("turtipi", [])
    if isinstance(turtipi, list):
        turtipi = " ".join(turtipi)

    konaklama_text = tur.get("konaklama", "")

    combined_text = f"""
    {tur.get("isim", "")}
    {tur.get("metin", "")}
    {konaklama_text}
    {tur.get("ulasim", "")}
    {" ".join(kategori_isimleri)}
    {ziyaretedilecek}
    {turtipi}
    """

    return combined_text.strip()

def upload_tur_data(collection_name: str, tur: dict):

    try:
        tur = transform_tur_payload(tur)

        embedding_text = build_embedding_text(tur)

        # -------- Dense --------
        dense_raw = list(dense_model.embed([embedding_text]))[0]
        dense_vector = dense_raw.tolist()  # numpy -> list

        # -------- Sparse --------
        sparse_raw = list(sparse_model.embed([embedding_text]))[0]

        sparse_vector = models.SparseVector(
            indices=sparse_raw.indices.tolist(),
            values=sparse_raw.values.tolist()
        )

        # -------- Point --------
        point = models.PointStruct(
            id=int(tur.get("id", 0)),
            vector={
                "dense": dense_vector,
                "sparse": sparse_vector
            },
            payload=tur
        )

        client.upsert(
            collection_name=collection_name,
            points=[point]
        )

        logger.info(f"📤 Tur verisi yüklendi: ID {tur.get('id')}")

    except Exception as e:
        logger.error(f"❌ Tur verisi yüklenirken hata: ID {tur.get('id')}, Hata: {str(e)}")
        raise e

def create_payload_indexes(collection_name: str):
    """
    FILTER_SEARCH_FIELDS kurallarına göre tüm filtrelenebilir alanlar için
    Qdrant payload index'leri oluşturur.
    Index'ler filtre sorgularını 10-100x hızlandırır.
    """
    # ═══════════════════════════════════════════════════════
    # FILTER_SEARCH_FIELDS → Payload Index Type Mapping
    # Her alan, Qdrant'taki gerçek veri tipine göre index'lenir
    # ═══════════════════════════════════════════════════════
    PAYLOAD_INDEX_MAP = {
        # ========================
        # Top Level - KEYWORD (string exact match)
        # ========================
        "geceleme":     models.PayloadSchemaType.KEYWORD,
        "turtipi":      models.PayloadSchemaType.KEYWORD,    # list[str] → KEYWORD (Qdrant array'leri otomatik handle eder)
        "ulasimtipi":   models.PayloadSchemaType.KEYWORD,
        "ulasim":       models.PayloadSchemaType.KEYWORD,
        "vizesiz":      models.PayloadSchemaType.KEYWORD,    # str ("Bu tur vizesizdir." vb.)
        "kalkissehir":  models.PayloadSchemaType.KEYWORD,
        "kesinkalkis":  models.PayloadSchemaType.KEYWORD,    # str ("Bu tur kesin kalkışlıdır." vb.)
        "turkodu":      models.PayloadSchemaType.KEYWORD,
        "vizeulke":     models.PayloadSchemaType.KEYWORD,
        "url":          models.PayloadSchemaType.KEYWORD,

        # ========================
        # Top Level - INTEGER
        # ========================
        "gecesayisi":   models.PayloadSchemaType.INTEGER,
        "id":           models.PayloadSchemaType.INTEGER,

        # ========================
        # TarihFiyat (Array of Object) — nested
        # ========================
        "tarihFiyat[].tarih.gun":   models.PayloadSchemaType.INTEGER,
        "tarihFiyat[].tarih.ay":    models.PayloadSchemaType.KEYWORD,   # "Ocak", "Şubat" vb.
        "tarihFiyat[].tarih.yil":   models.PayloadSchemaType.INTEGER,
        "tarihFiyat[].kisibasi":    models.PayloadSchemaType.FLOAT,
        "tarihFiyat[].kur":         models.PayloadSchemaType.KEYWORD,
        "tarihFiyat[].kalan":       models.PayloadSchemaType.INTEGER,

        # ========================
        # Konaklama Nested Alanlar
        # ========================
        "konaklama.konaklama_tipi": models.PayloadSchemaType.KEYWORD,   # list[str]
        "konaklama.konum_tipi":     models.PayloadSchemaType.KEYWORD,   # list[str]
        "konaklama.otel_markasi":   models.PayloadSchemaType.KEYWORD,   # list[str]
        "konaklama.otel_yildiz":    models.PayloadSchemaType.INTEGER,   # list[int]
        "konaklama.pansiyon_tipi":  models.PayloadSchemaType.KEYWORD,   # list[str]

        # ========================
        # Ziyaret
        # ========================
        "ziyaretedilecekyerler":    models.PayloadSchemaType.KEYWORD,   # list[str]
    }

    created_count = 0
    failed_count = 0

    for field_name, schema_type in PAYLOAD_INDEX_MAP.items():
        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field_name,
                field_schema=schema_type,
            )
            logger.info(f"✅ Payload index oluşturuldu: {field_name} ({schema_type})")
            created_count += 1
        except Exception as e:
            # Index zaten varsa veya başka bir hata olursa devam et
            logger.warning(f"⚠️ Payload index oluşturulamadı ({field_name}): {e}")
            failed_count += 1

    logger.info(f"🎯 Payload index sonucu: {created_count} başarılı, {failed_count} başarısız")

def create_new_collection(firm_code: str, date_info: str, url_data: List[dict]):
    """
    Firma kodu + bugün tarihi ile yeni collection oluşturur
    Format: firma_kodu_bugun_tarih (örn: dilek_29_08_25)
    Eğer collection zaten varsa ve veri içeriyorsa, mevcut olanı kullanır
    URL verilerini embed yapıp collection'a kaydeder
    """
    try:
        # Collection adını oluştur: firma_kodu_bugun_tarih
        collection_name = f"{firm_code}_{date_info.replace('.', '_')}"
        logger.info(f"🏗️ Collection kontrol ediliyor: {collection_name}")
        
        # Önce collection'ın var olup olmadığını kontrol et
        collection_exists = False
        collection_has_data = False
        
        try:
            collection_info = client.get_collection(collection_name)
            collection_exists = True
            collection_has_data = collection_info.points_count > 0
            logger.info(f"✅ Collection mevcut: {collection_name}")
            logger.info(f"📊 Collection'da {collection_info.points_count} adet veri var")
            
            # Eğer collection varsa ve veri içeriyorsa, veri yükleme atlanır
            if collection_has_data:
                logger.info(f"⏭️ Collection '{collection_name}' zaten veri içeriyor. Veri yükleme ATLANDI.")
                return collection_name
                
        except Exception:
            # Collection yoksa oluştur
            logger.info(f"🏗️ Yeni collection oluşturuluyor: {collection_name}")
            
            client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    "dense": models.VectorParams(
                        size=384,
                        distance=models.Distance.COSINE,
                        # ✅ Dense vektör için HNSW konfigürasyonu
                        hnsw_config=models.HnswConfigDiff(
                            m=16,
                            ef_construct=128,
                        ),
                    ),
                },
                sparse_vectors_config={
                    "sparse": models.SparseVectorParams(
                        index=models.SparseIndexParams(
                            on_disk=False,
                        ),
                        modifier=models.Modifier.IDF,  # ✅ BM25-like ağırlıklandırma
                    )
                },
                # ✅ Genel HNSW config
                hnsw_config=models.HnswConfigDiff(
                    m=16,
                    ef_construct=128,
                    payload_m=16,       # Payload filtreli aramalarda HNSW performansı
                ),
                # ✅ Optimizasyon ayarları
                optimizers_config=models.OptimizersConfigDiff(
                    indexing_threshold=10000,  # 10K noktadan sonra HNSW indeksi oluşturulsun
                ),
                # ✅ Scalar Quantization — Collection seviyesinde (RAM %75 tasarruf + hız)
                quantization_config=models.ScalarQuantization(
                    scalar=models.ScalarQuantizationConfig(
                        type=models.ScalarType.INT8,
                        quantile=0.99,
                        always_ram=True,
                    )
                ),
            )
            
            logger.info(f"✅ Yeni collection başarıyla oluşturuldu: {collection_name}")
            
            # ✅ Payload Index'leri oluştur (FILTER_SEARCH_FIELDS kurallarına göre)
            logger.info(f"🔧 Payload index'leri oluşturuluyor: {collection_name}")
            create_payload_indexes(collection_name)
        
        # URL verilerini embed yapıp collection'a kaydet (sadece yeni collection veya boş koleksiyon için)
        if url_data:
            logger.info(f"🔍 {len(url_data)} adet veri embed yapılıyor...")
            
            for i, tur_data in enumerate(url_data):
                try:
                    upload_tur_data(collection_name, tur_data)
                    logger.info(f"📝 Veri {i+1}/{len(url_data)} başarıyla yüklendi")
                except Exception as e:
                    logger.error(f"❌ Veri {i+1} yüklenemedi: {str(e)}")
                    continue
            
            logger.info(f"💾 Toplam {len(url_data)} adet veri collection'a kaydedildi")
        
        return collection_name
        
    except Exception as e:
        logger.error(f"❌ Collection işlem hatası: {str(e)}")
        return None
# endregion

# region Step 6: Delete old collections
def delete_old_collections(firm_code: str, current_date: str, collection_list):
    """
    Firma koduna ait eski collection'ları siler
    Sadece güncel collection'ı tutar, diğerlerini siler
    """
    try:
        logger.info("🗑️ Eski collection'lar kontrol ediliyor...")
        
        # Firma koduna ait collection'ları filtrele
        firm_collections = []
        for collection in collection_list:
            collection_name = collection.name
            base_name = collection_name.split('_')[0] if '_' in collection_name else collection_name
            
            if base_name.lower() == firm_code.lower():
                firm_collections.append(collection_name)
        
        logger.info(f"📋 Firma '{firm_code}' için bulunan collection'lar: {firm_collections}")
        
        # Güncel collection adı
        current_collection = f"{firm_code}_{current_date.replace('.', '_')}"
        logger.info(f"🎯 Güncel collection: {current_collection}")
        
        # Eski collection'ları sil
        deleted_collections = []
        for collection_name in firm_collections:
            if collection_name != current_collection:
                try:
                    logger.info(f"🗑️ Eski collection siliniyor: {collection_name}")
                    client.delete_collection(collection_name)
                    logger.info(f"✅ Eski collection silindi: {collection_name}")
                    deleted_collections.append(collection_name)
                except Exception as e:
                    logger.error(f"❌ Collection silinemedi {collection_name}: {str(e)}")
        
        logger.info(f"🗑️ Toplam {len(deleted_collections)} adet eski collection silindi: {deleted_collections}")
        return deleted_collections
        
    except Exception as e:
        logger.error(f"❌ Eski collection silme hatası: {str(e)}")
        return []

@app.post("/load_firma_links")
def load_firma_links(request: FirmaLinksRequest):
    """
    Firma koduna göre collection oluşturur veya mevcut olanı kullanır
    URL'lerden veri çeker ve Qdrant'a yükler
    """

    try:
        # Hash doğrulama
        firm_code = request.firma_kodu
        urls = request.data
        logger.info(f"🏢 Firma kodu: {firm_code} işleniyor...")
        logger.info(f"🔗 Alınan URL'ler: {urls}")

        # Step 2: Tarih versiyonu oluştur
        date_info = create_date_version()
        if date_info:
            logger.info(f"📅 İşlem tarihi: {date_info} ({date_info})")

        # Her URL'den veri çek ve JSON'a kaydet
        all_data = []
        for url in urls:
            logger.info(f"📡 URL işleniyor: {url}")
            data = fetch_data_from_url(url)
            if not data:
                logger.warning(f"⚠️ Bu URL için veri yok veya çekilemedi, atlanıyor: {url}")
                continue
            all_data.extend(data)

        if not all_data:
            logger.warning("⚠️ Hiçbir URL'den veri alınamadı. Devam ediliyor (collection oluşturma/silme adımları çalışabilir).")

        # Step 4: Firma kodu ile collection isimlerini karşılaştır
        matching_collections, collection_list = compare_firma_with_collections(firm_code)

        # Step 5: Collection oluştur veya mevcut olanı kullan
        collection_name = create_new_collection(firm_code, date_info, all_data)
        if collection_name:
            # Collection adından bugünlük olup olmadığını kontrol et
            today_date = f"{date_info.replace('.', '_')}"
            if collection_name.endswith(today_date):
                logger.info(f"🎯 Bugünlük collection kullanılıyor: {collection_name}")
            else:
                logger.info(f"🎉 Yeni collection hazır: {collection_name}")

            # Step 6: Eski collection'ları sil (sadece yeni collection başarıyla oluşturulduktan sonra)
            logger.info("✅ Collection başarıyla oluşturuldu, eski collection'lar siliniyor...")
            deleted_collections = delete_old_collections(firm_code, date_info, collection_list)
        else:
            logger.warning("⚠️ Collection işlemi başarısız")
            logger.warning("⚠️ Eski collection'lar korunuyor (veri güvenliği için)")
            deleted_collections = []
            raise HTTPException(status_code=500, detail="Collection oluşturulamadı")

        return {
            "firma_kodu": firm_code,
            "islenen_url_sayisi": len(urls),
            "collection_isimleri": [c.name for c in collection_list] if collection_list else [],
            "eslesen_collections": matching_collections,
            "eslesen_sayisi": len(matching_collections),
            "collection_durumu": "mevcut" if collection_name in [c.name for c in collection_list] else "yeni_olusturuldu",
            "yeni_collection_adi": collection_name,
            "silinen_eski_collection_adi": deleted_collections,
            "silinen_eski_collection_sayisi": len(deleted_collections)
        }

    except Exception as e:
        logger.error(f"❌ Hata: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# =====================
# Mevcut Collection Optimizasyonu (Bir kez çalıştır)
# =====================
@app.post("/optimize_collection")
def optimize_existing_collection(collection_name: str):
    """
    Mevcut collection'a HNSW + Quantization + Payload Index uygular.
    BİR KEZ çalıştırılır. Veri kaybı OLMAZ.
    Qdrant arka planda re-index yapar, status 'green' olunca biter.
    """
    try:
        # 1) HNSW + Quantization güncelle
        logger.info(f"🔧 HNSW + Quantization güncelleniyor: {collection_name}")
        client.update_collection(
            collection_name=collection_name,
            hnsw_config=models.HnswConfigDiff(
                m=16,
                ef_construct=128,
                payload_m=16,
            ),
            quantization_config=models.ScalarQuantization(
                scalar=models.ScalarQuantizationConfig(
                    type=models.ScalarType.INT8,
                    quantile=0.99,
                    always_ram=True,
                )
            ),
        )
        logger.info(f"✅ HNSW + Quantization güncellendi: {collection_name}")

        # 2) Payload Index'leri oluştur
        logger.info(f"🔧 Payload index'leri oluşturuluyor: {collection_name}")
        create_payload_indexes(collection_name)

        # 3) Durum kontrolü
        info = client.get_collection(collection_name)
        
        return {
            "collection": collection_name,
            "status": str(info.status),
            "points_count": info.points_count,
            "hnsw_config": str(info.config.hnsw_config),
            "message": "✅ Optimizasyon başlatıldı. Status 'green' olunca tamamlanmış demektir."
        }

    except Exception as e:
        logger.error(f"❌ Optimizasyon hatası: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =====================
# 5. FastAPI Server Başlat
# =====================
if __name__ == "__main__":
    import uvicorn
    print(" FastAPI Server başlatılıyor...")
    print("📡 Endpoint: http://0.0.0.0:8007/load_firma_links")
    print("🌐 Network erişimi: http://[CIHAZ_IP]:8007/load_firma_links")
    print("📖 API Docs: http://0.0.0.0:8007/docs")
    uvicorn.run(app, host="0.0.0.0", port=8007)
# endregion






