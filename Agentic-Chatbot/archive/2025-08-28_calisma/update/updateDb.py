# =====================
# Kütüphaneler
import sys,os
from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding, LateInteractionTextEmbedding
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
from typing import List
import logging
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import URL_LINK

# =====================
# Logging Ayarları
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# =====================


# FastAPI Server Başlatma
# =====================
app = FastAPI(title="Qdrant Veri Yükleme API")


# =====================
# 1. Qdrant Bağlantısı
# =====================
client = QdrantClient(URL_LINK, check_compatibility=False)

# =====================
# 2. Embed Modelleri
# =====================
dense_model = TextEmbedding(model_name="BAAI/bge-small-en")
colbert_model = LateInteractionTextEmbedding(model_name="colbert-ir/colbertv2.0")

# =====================
# 3. Pydantic Model
# =====================
class FirmaLinksRequest(BaseModel):
    firma_kodu: str
    data: List[str]

# =====================
# 4. API Endpoint - Firma Verilerini Yükle
# =====================
@app.post("/load_firma_links")
async def load_firma_links(request: FirmaLinksRequest):
    """
    Firma koduna göre collection oluşturur veya mevcut olanı kullanır
    URL'lerden veri çeker ve Qdrant'a yükler
    """
    try:
        firma_kodu = request.firma_kodu
        data = request.data
        collection_name = firma_kodu
        logger.info(f" Firma kodu: {firma_kodu} işleniyor...")
        
        # 1. Collection kontrolü ve oluşturma
        collection_exists = await check_and_create_collection(collection_name)
        
        # 2. Her URL'den veri çekme ve işleme
        total_processed = 0
        total_updated = 0
        total_new = 0
        
        for url in data:
            logger.info(f"📡 URL işleniyor: {url}")
            
            # URL'den veri çek
            tur_data = await fetch_data_from_url(url)
            if not tur_data:
                logger.warning(f"⚠️ URL'den veri çekilemedi: {url}")
                continue
            
            logger.info(f"📊 Bu URL'den {len(tur_data)} adet tur verisi geldi")
            
            # Her tur için işlem yap
            for tur in tur_data:
                logger.info(f"🔄 Tur işleniyor: ID {tur.get('id', 'N/A')} - {tur.get('isim', 'N/A')}")
                
                # Mevcut kayıt kontrolü
                existing_record = await check_existing_record(collection_name, tur.get('id'))
                
                if existing_record:
                    logger.info(f"🔄 Mevcut kayıt bulundu, güncelleniyor: ID {tur.get('id')}")
                    await delete_existing_record(collection_name, tur.get('id'))
                    total_updated += 1
                else:
                    logger.info(f" Yeni kayıt ekleniyor: ID {tur.get('id')}")
                    total_new += 1
                
                # Yeni veriyi yükle
                await upload_tur_data(collection_name, tur)
                total_processed += 1
                
                logger.info(f"✅ Tur başarıyla işlendi: ID {tur.get('id')}")
            
            logger.info(f"📈 URL tamamlandı: {url} - Toplam işlenen: {total_processed}")
        
        # 3. Sonuç raporu
        collection_info = client.get_collection(collection_name)
        final_count = collection_info.points_count
        
        result = {
            "firma_kodu": firma_kodu,
            "collection_name": collection_name,
            "total_processed": total_processed,
            "total_updated": total_updated,
            "total_new": total_new,
            "final_vector_count": final_count,
            "status": "success"
        }
        
        logger.info(f"🎉 İşlem tamamlandı! Sonuç: {result}")
        return result
        
    except Exception as e:
        logger.error(f"❌ Hata oluştu: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# =====================
# 4. Yardımcı Fonksiyonlar
# =====================
async def check_and_create_collection(collection_name: str) -> bool:
    """Collection var mı kontrol eder, yoksa oluşturur"""
    try:
        collection_info = client.get_collection(collection_name)
        logger.info(f"✅ Collection mevcut: {collection_name}")
        logger.info(f"📊 Mevcut vektör sayısı: {collection_info.points_count}")
        return True
    except Exception as e:
        if "404" in str(e) or "Not found" in str(e):
            logger.info(f"ℹ️ Collection '{collection_name}' bulunamadı, yeni oluşturuluyor...")
            
            client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    "text-dense": models.VectorParams(size=384, distance=models.Distance.COSINE),
                    "colbert": models.VectorParams(
                        size=128,
                        distance=models.Distance.COSINE,
                        multivector_config=models.MultiVectorConfig(
                            comparator=models.MultiVectorComparator.MAX_SIM
                        ),
                        hnsw_config=models.HnswConfigDiff(m=0)
                    )
                }
            )
            logger.info(f"✅ Yeni collection başarıyla oluşturuldu: {collection_name}")
            return False
        else:
            raise e

async def fetch_data_from_url(url: str) -> List[dict]:
    """URL'den JSON veri çeker"""
    try:
        logger.info(f"🌐 URL'den veri çekiliyor: {url}")
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        logger.info(f"📥 {len(data)} adet tur verisi çekildi")
        return data
        
    except Exception as e:
        logger.error(f"❌ URL'den veri çekilemedi: {url}, Hata: {str(e)}")
        return []

async def check_existing_record(collection_name: str, tur_id: str) -> bool:
    """Tur ID'sine göre mevcut kayıt kontrolü"""
    try:
        results = client.query_points(
            collection_name=collection_name,
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="id",
                        match=models.MatchValue(value=tur_id)
                    )
                ]
            ),
            limit=1
        )
        return len(results.points) > 0
    except Exception as e:
        logger.warning(f"⚠️ Kayıt kontrolünde hata: {str(e)}")
        return False

async def delete_existing_record(collection_name: str, tur_id: str):
    """Mevcut kaydı siler"""
    try:
        point_id = int(tur_id) if tur_id.isdigit() else tur_id
        
        # En basit format
        client.delete(
            collection_name=collection_name,
            points_selector=[point_id]
        )
        logger.info(f"🗑️ Mevcut kayıt silindi: ID {tur_id}")
    except Exception as e:
        logger.warning(f"⚠️ Kayıt silinirken hata: {str(e)}")

async def upload_tur_data(collection_name: str, tur: dict):
    """Tur verisini Qdrant'a yükler"""
    try:
        # turKategori'den sadece isim alanlarını al
        kategori_isimleri = [kategori.get('isim', '') for kategori in tur.get('turKategori', []) if isinstance(kategori, dict)]
        
        # Embed edilecek alanları birleştir
        searchable_text = f"{tur.get('ulasimtipi', '')} {tur.get('turtipi', '')} {tur.get('isim', '')} {tur.get('metin', '')} {tur.get('geceleme', '')} {tur.get('konaklama', '')} {tur.get('ulasim', '')} {tur.get('ziyaretedilecekyerler', '')} {' '.join(kategori_isimleri)}"
        
        # Vektörler oluştur
        dense_vec = list(dense_model.embed([searchable_text]))[0]
        colbert_vec = list(colbert_model.embed([searchable_text]))[0]
        
        # Point oluştur
        point = models.PointStruct(
            id=int(tur.get("id", 0)),
            vector={
                "text-dense": dense_vec,
                "colbert": colbert_vec
            },
            payload=tur
        )
        
        # Qdrant'a yükle
        client.upload_points(collection_name=collection_name, points=[point])
        logger.info(f"📤 Tur verisi yüklendi: ID {tur.get('id')}")
        
    except Exception as e:
        logger.error(f"❌ Tur verisi yüklenirken hata: ID {tur.get('id')}, Hata: {str(e)}")
        raise e

# =====================
# 5. FastAPI Server Başlat
# =====================
if __name__ == "__main__":
    import uvicorn
    print(" FastAPI Server başlatılıyor...")
    print("📡 Endpoint: http://0.0.0.0:8000/load_firma_links")
    print("🌐 Network erişimi: http://[CIHAZ_IP]:8000/load_firma_links")
    print("📖 API Docs: http://0.0.0.0:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
