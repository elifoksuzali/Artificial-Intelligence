# Qdrant Hybrid Search - Tur Verileri

Bu proje, `birlesik.json` dosyasındaki tur verilerini Qdrant vektör veritabanına yükleyerek hybrid search (semantic + keyword) özelliği sağlar.

## 🚀 Özellikler

- **Semantic Search**: Metin benzerliği bazlı arama
- **Keyword Search**: Tam kelime eşleşmesi bazlı arama
- **Hybrid Search**: Semantic ve keyword aramalarını birleştirme
- **Filtreleme**: Tur tipi, ulaşım tipi, kategori gibi alanlara göre filtreleme
- **Türkçe Desteği**: Türkçe metinler için optimize edilmiş embedding modelleri

## 📋 Gereksinimler

- Python 3.8+
- Docker
- 4GB+ RAM (embedding işlemleri için)

## 🛠️ Kurulum

### 1. Gerekli Paketleri Yükle

```bash
pip install -r requirements.txt
```

### 2. Qdrant'ı Başlat

```bash
python qdrant_setup.py
```

Veya manuel olarak:

```bash
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 -v qdrant_storage:/qdrant/storage qdrant/qdrant
```

### 3. Verileri Yükle

```bash
python qdrant_hybrid_search.py
```

## 📊 Veri Yapısı

`birlesik.json` dosyası şu alanları içerir:

```json
{
    "id": "77",
    "isim": "Tur adı",
    "turkodu": "2036",
    "gecesayisi": "6",
    "metin": "Açıklama metni",
    "geceleme": "6 Gece 7 Gün",
    "konaklama": "3* Otellerde 5 Gece Yarım Pansiyon",
    "ulasim": "Otobüs ile İstanbul - Karadeniz - Batum - İstanbul",
    "ziyaretedilecekyerler": "Ziyaret edilecek yerler listesi",
    "vizesiz": "Bu tur vizesizdir",
    "kesinkalkis": "Bu tur kesin kalkışlı değildir",
    "ulasimtipi": "Otobüslü",
    "turtipi": "yurt içi",
    "turKategori": [
        {
            "isim": "Kategori adı",
            "puan": "0"
        }
    ],
    "url": "Tur URL'si"
}
```

## 🔍 Kullanım Örnekleri

### Temel Arama

```python
from qdrant_hybrid_search import QdrantHybridSearch

# Qdrant search sınıfını başlat
qdrant_search = QdrantHybridSearch()

# Semantic search
results = qdrant_search.hybrid_search("Karadeniz yayla turu", limit=10)

# Sonuçları göster
for result in results:
    print(f"Tur: {result['payload']['isim']}")
    print(f"Score: {result['score']:.3f}")
    print("---")
```

### Filtreli Arama

```python
# Yurt içi otobüslü kültür turları
results = qdrant_search.hybrid_search(
    "kültür turu",
    limit=10,
    filters={
        "turtipi": "yurt içi",
        "ulasimtipi": "Otobüslü"
    }
)
```

### Kategori Bazlı Arama

```python
# Belirli kategorideki turlar
results = qdrant_search.hybrid_search(
    "tur",
    limit=10,
    filters={"turKategori.isim": "Kapadokya Turları"}
)
```

### Gece Sayısına Göre Filtreleme

```python
# 3 gece turları
results = qdrant_search.hybrid_search(
    "tur",
    limit=10,
    filters={"gecesayisi": "3"}
)
```

## 🎯 Arama Örnekleri

Projeyi test etmek için:

```bash
python search_examples.py
```

Bu script şu örnekleri çalıştırır:

1. **Semantic Search**: "Karadeniz yayla turu"
2. **Filtreli Arama**: "kültür turu" + yurt içi + otobüslü
3. **Kapadokya Turları**: "Kapadokya turu"
4. **Ege Akdeniz**: "Ege Akdeniz turu"
5. **Kategori Filtreli**: "Batum Turları"
6. **Gece Sayısı**: "3 gece turları"

## 🔧 Konfigürasyon

### Embedding Modelleri

Farklı embedding modelleri kullanabilirsiniz:

```python
# Varsayılan model (İngilizce odaklı)
qdrant_search = QdrantHybridSearch(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Türkçe için daha uygun model
qdrant_search_tr = QdrantHybridSearch(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
```

### Koleksiyon Adı

```python
qdrant_search = QdrantHybridSearch(collection_name="tur_verileri")
```

## 🌐 Qdrant UI

Qdrant'ın web arayüzüne erişmek için:

- **URL**: http://localhost:6333/dashboard
- **API**: http://localhost:6333

## 📈 Performans

- **Embedding Boyutu**: 384 (all-MiniLM-L6-v2) veya 512 (multilingual)
- **Vektör Mesafesi**: Cosine Similarity
- **Batch Yükleme**: 1000 adet kayıt
- **Arama Hızı**: ~10ms (1000 kayıt için)

## 🔍 Hybrid Search Detayları

Hybrid search iki tür aramayı birleştirir:

1. **Semantic Search**: Metin benzerliği bazlı
2. **Keyword Search**: Tam kelime eşleşmesi

Alpha parametresi ile ağırlık ayarlanabilir:
- `alpha=0`: Sadece keyword search
- `alpha=1`: Sadece semantic search
- `alpha=0.5`: Eşit ağırlık (varsayılan)

## 🛠️ Geliştirme

### Yeni Alan Ekleme

`prepare_text_for_embedding` fonksiyonunu güncelleyin:

```python
def prepare_text_for_embedding(self, tur_data: Dict[str, Any]) -> str:
    text_parts = []
    
    # Mevcut alanlar...
    
    # Yeni alan ekle
    if tur_data.get("yeni_alan"):
        text_parts.append(f"Yeni alan: {tur_data['yeni_alan']}")
    
    return " | ".join(text_parts)
```

### Yeni Filtre Ekleme

```python
results = qdrant_search.hybrid_search(
    "tur",
    filters={"yeni_alan": "değer"}
)
```

## 📝 Loglar

Loglar `logging` modülü ile yönetilir:

```python
import logging
logging.basicConfig(level=logging.INFO)
```

## 🚨 Hata Giderme

### Qdrant Bağlantı Hatası

```bash
# Qdrant'ı yeniden başlat
docker stop qdrant
docker rm qdrant
docker run -d --name qdrant -p 6333:6333 qdrant/qdrant
```

### Embedding Model Hatası

```bash
# Model'i yeniden indir
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"
```

### Bellek Hatası

Büyük veri setleri için:
- Batch boyutunu küçültün (1000 → 500)
- Daha küçük embedding modeli kullanın

## 📄 Lisans

Bu proje MIT lisansı altında lisanslanmıştır.

## 🤝 Katkıda Bulunma

1. Fork yapın
2. Feature branch oluşturun (`git checkout -b feature/amazing-feature`)
3. Commit yapın (`git commit -m 'Add amazing feature'`)
4. Push yapın (`git push origin feature/amazing-feature`)
5. Pull Request oluşturun 