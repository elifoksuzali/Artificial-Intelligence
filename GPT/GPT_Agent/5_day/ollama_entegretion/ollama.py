import sys
import os

# Kendi agents modülünüzü import edin
sys.path.insert(0, os.path.dirname(__file__))
from agents import Agent,Runner # local_agents.py olarak kaydedin
from dotenv import load_dotenv
import json
import requests
from typing import Dict, List, Any
import re
from config import QDRANT_HOST, QDRANT_PORT, QDRANT_API_KEY, OLLAMA_BASE_URL
model="gpt-oss:20b"
use_ollama=True,  # Ollama kullanımını aktif et
ollama_base_url=OLLAMA_BASE_URL

# Load environment variables
load_dotenv()

# Sorgu düzeltme agenti oluştur
query_correction_agent = Agent(
    name="QueryCorrectionAgent",
    instructions="""
Sen bir Türkçe sorgu düzeltme uzmanısın. Kullanıcıların yazım hataları, eksik harfler ve anlamsal belirsizliklerle gelen tur sorgularını düzeltiyorsun.

GÖREV:
1. Kullanıcının yazdığı sorguyu analiz et
2. Yazım hatalarını düzelt
3. Eksik harfleri tamamla
4. Anlamsal olarak en yakın düzgün cümleye dönüştür
5. Sayıları standart formata çevir (yedi → 7, iki → 2)

DÜZELTME KURALLARI:
- "gnlk" → "günlük"
- "Frnsw" → "Fransa"
- "ydlzlı" → "yıldızlı"
- "tm" → "tam"
- "pansyn" → "pansiyon"
- "iki" → "2"
- "yedi" → "7"
- "üç" → "3"
- "dört" → "4"
- "beş" → "5"
- "altı" → "6"
- "sekiz" → "8"
- "dokuz" → "9"
- "on" → "10"

ÇIKTI FORMATI:
Sadece düzeltilmiş sorguyu ver, başka açıklama yapma.

ÖRNEKLER:
Giriş: " iki gnlk Frnsw 5 ydlzlı tm pansyn turlar"
Çıkış: "2 günlük Fransa 5 yıldızlı tam pansiyon turlar"

Giriş: "yedi gecelik İstanbul turları"
Çıkış: "7 gecelik İstanbul turları"
""",
    model=model,
    use_ollama=use_ollama,
    ollama_base_url=ollama_base_url
)

# Tur analiz agenti oluştur
analysis_agent = Agent(
    name="TourAnalysisAgent", 
    instructions="""
Sen bir tur analiz uzmanısın. Kullanıcıların tur sorularını analiz edip, tur veri setindeki alanlarla eşleştirme yapıyorsun.

Tur veri seti alanları:
- id: Tur kimlik numarası
- isim: Tur adı
- turkodu: Tur kodu (sayısal değer)
- gecesayisi: Kaç gece (sayı olarak)
- geceleme: "X Gece Y Gün" formatında
- konaklama: Konaklama tipi (örn: "5 Gece Yarım Pansiyon", "Oteller", "Tam Pansiyon", "Tren Konaklaması")
- ulasim: Ulaşım detayı (örn: "Otobüs ile İstanbul - Ege & Akdeniz - İstanbul")
- ulasimtipi: Ulaşım tipi ("Uçaklı", "Trenli", "Otobüslü", "Gemi", "Gemi ile Gidiş Dönüş")
- ziyaretedilecekyerler: Gezilecek yerler listesi (Ankara, Sarajevo, İstanbul, Patmos vb.)
- vizesiz: Vize durumu ("Bu tur vizesizdir", "Bu tur vizelidir")
- kesinkalkis: Kesin kalkış durumu ("Bu tur kesin kalkışlıdır", "Bu tur kesin kalkışlı değildir")
- turtipi: "yurt içi", "yurt dışı", "gemi"
- turKategori: Tur kategorileri array (her birinin "isim" ve "puan" alanı var)
- url: Tur detay linki

Kullanıcı sorusunu analiz et ve şu formatta eşleştirme yap:

ÖRNEK:
Soru: "7 günlük yurt içi otobüslü turlar neler"
Analiz: gecesayisi:7, turtipi:yurt içi, ulasimtipi:Otobüslü

KURALLAR:
1. Sadece sorudan çıkarılabilecek bilgileri eşleştir
2. GÜN/GECE HESAPLAMA:
   - Kullanıcı "X günlük" diyorsa: gecesayisi:X - TAM EŞLEŞME ARANACAK
   - Kullanıcı "X gecelik" diyorsa: gecesayisi:X - TAM EŞLEŞME ARANACAK
   - Örnek: "7 günlük" = gecesayisi:7 (tam eşleşme)
   - Örnek: "5 gecelik" = gecesayisi:5 (tam eşleşme)

3. KONAKLAMA ARAMA KURALLARI:
   - "Tam pansiyon konaklamalı" → konaklama:Tam Pansiyon
   - "Yarım pansiyon" → konaklama:Yarım Pansiyon
   - "Oteller" → konaklama:Oteller
   - "Tren konaklaması" → konaklama:Tren Konaklaması
   - Sadece ana terimi al, "konaklamalı", "olan" gibi ekleri alma

4. ULAŞIM ARAMA KURALLARI:
   - "Uçaklı" → ulasimtipi:Uçaklı
   - "Trenli" → ulasimtipi:Trenli
   - "Otobüslü" → ulasimtipi:Otobüslü
   - "Gemi" → ulasimtipi:Gemi
   - "Gemi ile gidiş dönüş" → ulasimtipi:Gemi ile Gidiş Dönüş

5. VİZE ARAMA KURALLARI:
   - "vizesiz" → vizesiz:Bu tur vizesizdir
   - "vizeli" → vizesiz:Bu tur vizelidir
   - Tam cümle formatını kullan, sadece "vizesiz" değil

6. KESİN KALKİŞ ARAMA KURALLARI:
   - "kesin kalkışlı" → kesinkalkis:Bu tur kesin kalkışlıdır
   - "kesin kalkışlı değil" → kesinkalkis:Bu tur kesin kalkışlı değildir
   - Tam cümle formatını kullan

7. ZİYARET YERLERİ VE KATEGORİ ARAMA KURALLARI:
   - Şehir/yer adları için HEM ziyaretedilecekyerler HEM de turKategori[].isim alanlarında arama yap
   - "İstanbul turları" → ziyaretedilecekyerler:İstanbul VE turKategori[].isim:İstanbul
   - "Ankara turları" → ziyaretedilecekyerler:Ankara VE turKategori[].isim:Ankara
   - "Karadeniz turları" → turKategori[].isim:Karadeniz
   - "Yunan Adaları" → turKategori[].isim:Yunan Adaları
   - "Santorini" → turKategori[].isim:Santorini
   - "Balayı Turları" → turKategori[].isim:Balayı Turları
   - "Avrupa turları" → turKategori[].isim:Avrupa Turları
   - "Elit Turlar" → turKategori[].isim:Elit Turlar
   - "İstanbul Çıkışlı" → turKategori[].isim:İstanbul Çıkışlı
   - "Yurtdışı Turları" → turKategori[].isim:Yurtdışı Turları
   - Sadece şehir/yer adını al, "güzergahlı", "turları" gibi ekleri alma

8. TUR KODU ARAMA:
   - Sayısal değer verilirse → turkodu:sayı

9. SIRALAMA KURALI:
   - Eğer turKategori[].isim eşleşmesi varsa, sonuçlar turKategori[].puan'a göre büyükten küçüğe sıralanmalı

10. Array alanları için [] notasyonu kullan:
    - turKategori[].isim (kategori ismi için)
    - turKategori[].puan (kategori puanı için)

11. Sadece eşleşen alanları döndür, boş bir alan varsa belirtme
12. geceleme alanını ASLA kullanma, sadece gecesayisi kullan

Cevabını sadece eşleştirme formatında ver, başka açıklama yapma.
""",
    model=model,
    use_ollama=use_ollama,
    ollama_base_url=ollama_base_url
)

def correct_user_query(user_query):
    """
    Kullanıcı sorgusunu düzeltir ve anlamsal olarak en yakın düzgün cümleye dönüştürür
    """
    try:
        result = Runner.run_sync(query_correction_agent, user_query)
        corrected_query = result.final_output.strip()
        
        # Eğer düzeltme yapıldıysa kullanıcıya bilgi ver
        if corrected_query != user_query:
            print(f"🔧 Sorgu düzeltildi:")
            print(f"   Orijinal: '{user_query}'")
            print(f"   Düzeltilmiş: '{corrected_query}'")
            print(f"   Merhabalar! Kast ettiğiniz tur içeriği '{corrected_query}' şeklinde anlaşılmıştır. Sizler için kontrol ediyorum...")
        
        return corrected_query
    except Exception as e:
        print(f"⚠️ Sorgu düzeltme hatası: {str(e)}")
        return user_query

def analyze_tour_query(user_query):
    """
    Kullanıcı sorgusunu analiz eder ve tur alanlarıyla eşleştirir
    """
    try:
        result = Runner.run_sync(analysis_agent, user_query)
        return result.final_output
    except Exception as e:
        return f"Analiz hatası: {str(e)}"

def create_flexible_filter(analysis_result):
    """
    Daha esnek filter oluşturur - Qdrant'ın doğru formatına uygun
    """
    field_mappings = {
        "gecesayisi": {
            "type": "exact_match",
            "key": "gecesayisi",
            "description": "Gece sayısı (tam eşleşme)"
        },
        "turtipi": {
            "type": "match_value",
            "key": "text_fields.turtipi",
            "description": "Tur tipi (yurt içi/yurt dışı/gemi)"
        },
        "ulasimtipi": {
            "type": "match_value",
            "key": "text_fields.ulasimtipi",
            "description": "Ulaşım tipi (Uçaklı/Trenli/Otobüslü/Gemi)"
        },
        "turKategori[].isim": {
            "type": "match_nested",
            "key": "turKategori",
            "nested_key": "isim",
            "description": "Tur kategorisi"
        },
        "konaklama": {
            "type": "match_substring",
            "key": "text_fields.konaklama",
            "description": "Konaklama tipi - substring eşleştirme"
        },
        "ziyaretedilecekyerler": {
            "type": "match_text",
            "key": "text_fields.ziyaretedilecekyerler",
            "description": "Ziyaret edilecek yerler"
        },
        "vizesiz": {
            "type": "match_value",
            "key": "text_fields.vizesiz",
            "description": "Vize durumu (tam cümle formatı)"
        },
        "kesinkalkis": {
            "type": "match_value",
            "key": "text_fields.kesinkalkis",
            "description": "Kesin kalkış durumu (tam cümle formatı)"
        },
        "turkodu": {
            "type": "exact_match",
            "key": "turkodu",
            "description": "Tur kodu (sayısal)"
        }
    }
    
    must_conditions = []
    should_conditions = []
    detected_fields = []
    field_details = []
    has_category_match = False
    
    if not analysis_result or "Analiz hatası" in analysis_result:
        return {
            "filter": {"must": []},
            "detected_fields": [],
            "field_details": [],
            "has_flexible_conditions": False,
            "has_category_match": False
        }
    
    criteria = analysis_result.split(", ")
    
    for criterion in criteria:
        if ":" in criterion:
            key, value = criterion.split(":", 1)
            key = key.strip()
            value = value.strip()
            
            if key in field_mappings:
                field_config = field_mappings[key]
                detected_fields.append(key)
                
                # Kategori eşleşmesi var mı kontrol et
                if key == "turKategori[].isim":
                    has_category_match = True
                
                try:
                    if field_config["type"] == "exact_match":
                        # Gecesayisi ve turkodu için TAM EŞLEŞME - STRING olarak gönder
                        if key in ["gecesayisi", "turkodu", "geceleme"]:
                            # String olarak bırak, int'e çevirme
                            condition = {
                                "key": field_config["key"],
                                "match": {"value": value}  # String olarak gönder
                            }
                            must_conditions.append(condition)
                            field_details.append({
                                "field": key,
                                "type": "exact_match",
                                "value": value,  # String olarak kaydet
                                "description": field_config["description"]
                            })
                    
                    elif field_config["type"] == "match_value":
                        # String değerler için
                        condition = {
                            "key": field_config["key"],
                            "match": {"value": value}
                        }
                        must_conditions.append(condition)
                        field_details.append({
                            "field": key,
                            "type": "exact_match",
                            "value": value,
                            "description": field_config["description"]
                        })
                    
                    elif field_config["type"] == "match_substring":
                        # Konaklama için substring arama - hem must hem should
                        must_condition = {
                            "key": field_config["key"],
                            "match": {"value": value}
                        }
                        must_conditions.append(must_condition)
                        
                        # Should için alternatif varyantlar
                        should_variants = generate_accommodation_variants(value)
                        for variant in should_variants:
                            should_condition = {
                                "key": field_config["key"],
                                "match": {"value": variant}
                            }
                            should_conditions.append(should_condition)
                        
                        field_details.append({
                            "field": key,
                            "type": "substring_match",
                            "value": value,
                            "variants": should_variants,
                            "description": field_config["description"]
                        })
                    
                    elif field_config["type"] == "match_nested":
                        # Nested array field'lar için
                        condition = {
                            "key": field_config["key"],
                            "match": {
                                "key": field_config["nested_key"],
                                "value": value
                            }
                        }
                        must_conditions.append(condition)
                        field_details.append({
                            "field": key,
                            "type": "nested_match",
                            "value": value,
                            "description": field_config["description"]
                        })
                    
                    elif field_config["type"] == "match_text":
                        # Ziyaret yerleri için hem must hem should koşulları ekle
                        # MUST: ziyaretedilecekyerler alanında arama
                        must_condition = {
                            "key": field_config["key"],
                            "match": {"value": value}
                        }
                        must_conditions.append(must_condition)
                        
                        # SHOULD: turKategori alanında da arama yap
                        should_condition = {
                            "key": "turKategori",
                            "match": {
                                "key": "isim",
                                "value": value
                            }
                        }
                        should_conditions.append(should_condition)
                        
                        field_details.append({
                            "field": key,
                            "type": "text_search_with_category",
                            "value": value,
                            "description": f"{field_config['description']} + turKategori arama"
                        })
                        
                except Exception as e:
                    print(f"⚠️ Alan işleme hatası - {key}: {value} - Hata: {e}")
                    continue
    
    # Qdrant'ın doğru formatına uygun filter yapısı oluştur
    filter_structure = {}
    
    if must_conditions:
        filter_structure["must"] = must_conditions  # Array olarak
    
    if should_conditions:
        filter_structure["should"] = should_conditions  # Array olarak
        filter_structure["minimum_should_match"] = 1
    
    return {
        "filter": filter_structure,
        "detected_fields": detected_fields,
        "field_details": field_details,
        "has_flexible_conditions": len(should_conditions) > 0,
        "has_category_match": has_category_match
    }

def generate_accommodation_variants(base_value):
    """
    Konaklama tipi için varyantlar oluşturur
    """
    variants = []
    base_lower = base_value.lower()
    
    if "tam pansiyon" in base_lower:
        variants.extend([
            "tam pansiyonlu",
            "tam pansiyon konaklama",
            "full board",
            "tam pansiyonlu konaklama",
            "tam pansiyon oteller"
        ])
    
    if "yarım pansiyon" in base_lower:
        variants.extend([
            "yarım pansiyonlu",
            "yarım pansiyon konaklama", 
            "half board",
            "yarım pansiyonlu konaklama"
        ])
    
    if "her şey dahil" in base_lower:
        variants.extend([
            "all inclusive",
            "herşey dahil",
            "ultra her şey dahil",
            "all inclusive konaklama"
        ])
    
    if "otel" in base_lower:
        variants.extend([
            "oteller",
            "otel konaklama",
            "hotel",
            "5* otel",
            "4* otel",
            "3* otel"
        ])
    
    return variants

def search_structured(qdrant_params, has_category_match=False):
    """
    Yapısal arama (filter-based) - Önce must, sonra should dener
    """
    try:
        qdrant_url = f"{QDRANT_HOST}/collections/turizm_turlari/points/scroll"
        
        # 1. Önce must ile dene
        must_filter = {"must": qdrant_params["filter"].get("must", [])}
        request_body = {
            "filter": must_filter,
            "limit": qdrant_params["limit"],
            "with_payload": qdrant_params["with_payload"],
            "with_vector": False
        }
        
        print(f"🔧 DEBUG - Qdrant Filter (MUST):")
        print(json.dumps(request_body, indent=2, ensure_ascii=False))
        
        response = requests.post(qdrant_url, json=request_body)
        
        if response.status_code == 200:
            result = response.json()
            points = result.get("result", {}).get("points", [])
            
            if points:
                print(f"✅ MUST ile {len(points)} sonuç bulundu")
                # Eğer kategori eşleşmesi varsa, puan'a göre sırala
                if has_category_match:
                    points = sorted(points, key=lambda x: 
                        int(x.get("payload", {}).get("turKategori", [{}])[0].get("puan", 0)), 
                        reverse=True
                    )
                    print(f"📊 Kategori puanına göre sıralandı (büyükten küçüğe)")
                
                return {
                    "success": True,
                    "method": "structured_must",
                    "data": points,
                    "count": len(points)
                }
        
        # 2. Must ile sonuç yoksa should ile dene
        if qdrant_params["filter"].get("should"):
            should_filter = {
                "should": qdrant_params["filter"]["should"],
                "minimum_should_match": qdrant_params["filter"].get("minimum_should_match", 1)
            }
            
            request_body = {
                "filter": should_filter,
                "limit": qdrant_params["limit"],
                "with_payload": qdrant_params["with_payload"],
                "with_vector": False
            }
            
            print(f"🔧 DEBUG - Qdrant Filter (SHOULD):")
            print(json.dumps(request_body, indent=2, ensure_ascii=False))
            
            response = requests.post(qdrant_url, json=request_body)
            
            if response.status_code == 200:
                result = response.json()
                points = result.get("result", {}).get("points", [])
                
                if points:
                    print(f"✅ SHOULD ile {len(points)} sonuç bulundu")
                    # Eğer kategori eşleşmesi varsa, puan'a göre sırala
                    if has_category_match:
                        points = sorted(points, key=lambda x: 
                            int(x.get("payload", {}).get("turKategori", [{}])[0].get("puan", 0)), 
                            reverse=True
                        )
                        print(f"📊 Kategori puanına göre sıralandı (büyükten küçüğe)")
                    
                    return {
                        "success": True,
                        "method": "structured_should",
                        "data": points,
                        "count": len(points)
                    }
        
        # 3. Hiçbir sonuç bulunamadı
        print(f"❌ MUST ve SHOULD ile sonuç bulunamadı")
        return {
            "success": True,
            "method": "no_results",
            "data": [],
            "count": 0
        }
            
    except Exception as e:
        return {
            "success": False,
            "method": "structured",
            "error": str(e)
        }

def display_results(structured_result):
    """
    Yapısal arama sonuçlarını gösterir
    """
    print("\n" + "="*80)
    print("🔍 YAPISAL ARAMA SONUÇLARI")
    print("="*80)
    
    # Structured results
    structured_points = structured_result.get("data", []) if structured_result.get("success") else []
    method = structured_result.get("method", "unknown")
    
    if structured_points:
        method_text = "MUST" if "must" in method else "SHOULD" if "should" in method else "UNKNOWN"
        print(f"\n📊 YAPISAL ARAMA SONUÇLARI ({len(structured_points)} adet) - {method_text} ile bulundu:")
        print("-" * 50)
        
        for i, point in enumerate(structured_points[:10]):  # İlk 10'u göster
            if isinstance(point, dict):
                point_id = point.get("id")
                payload = point.get("payload", {})
            else:
                point_id = getattr(point, 'id', None)
                payload = getattr(point, 'payload', {})
            
            # Debug: Payload yapısını göster
            print(f"🔍 Point {i+1} - ID: {point_id}")
            print(f"📋 Payload keys: {list(payload.keys())}")
            
            # Qdrant'ta veriler text_fields içinde saklanıyor
            text_fields = payload.get('text_fields', {})
            
            # Veri alanlarını doğru key'lerle al
            tur_kodu = payload.get('tur_kodu', 'N/A')  # ✅ Düzeltildi: turkodu -> tur_kodu
            isim = payload.get('isim', 'N/A')
            gece_sayisi = payload.get('gecesayisi', 'N/A')
            geceleme = payload.get('geceleme', 'N/A')
            konaklama = payload.get('konaklama', 'N/A')
            ulasim = payload.get('ulasimtipi', 'N/A')
            tur_tipi = payload.get('turtipi', 'N/A')
            guzergah = payload.get('ziyaretedilecekyerler', 'N/A')
            kesin_kalkis = payload.get('kesinkalkis', 'N/A')
            
            print(f" Tur Kodu: {tur_kodu}")
            print(f" Adı: {isim}")
            print(f" Gece Sayısı: {gece_sayisi}")
            print(f" Geceleme: {geceleme}")
            print(f" Konaklama: {konaklama}")
            print(f" Ulaşım: {ulasim}")
            print(f" Tur Tipi: {tur_tipi}")
            print(f" Tur Güzergahı: {guzergah}")
            print(f" Kesin Kalkışlı: {kesin_kalkis}")
            print()
            
            # İlk 3 point'te detaylı debug bilgisi göster
            if i < 3:
                print(f"🔍 DEBUG - Point {i+1} detayları:")
                print(f"   Raw payload: {payload}")
                print(f"   Text fields: {text_fields}")
                print("-" * 30)
    else:
        print(f"\n📊 YAPISAL ARAMA SONUÇLARI: Sonuç bulunamadı")
        if not structured_result.get("success"):
            print(f"   Hata: {structured_result.get('error')}")
        else:
            print(f"   MUST ve SHOULD ile sonuç bulunamadı")

def structured_search(user_query):
    """
    Sadece yapısal arama yapar
    """
    print(f"🔍 Kullanıcı Sorusu: {user_query}")
    print("=" * 80)
    
    # 1. Sorguyu düzelt
    print("🔧 ADIM 1: Sorgu Düzeltme")
    corrected_query = correct_user_query(user_query)
    
    # 2. Sorguyu analiz et
    print("\n📊 ADIM 2: Sorgu Analizi")
    analysis = analyze_tour_query(corrected_query)
    print(f"Analiz: {analysis}")
    
    # 3. Esnek filter oluştur
    print("\n🔧 ADIM 3: Esnek Filter Oluşturma")
    filter_result = create_flexible_filter(analysis)
    
    print(f"Tespit edilen alanlar: {filter_result['detected_fields']}")
    print(f"Esnek koşullar var mı: {filter_result['has_flexible_conditions']}")
    print(f"Kategori eşleşmesi var mı: {filter_result['has_category_match']}")
    
    if filter_result['field_details']:
        print("Alan detayları:")
        for detail in filter_result['field_details']:
            print(f"   • {detail['field']}: {detail['value']} ({detail['type']})")
            if 'variants' in detail:
                print(f"     Varyantlar: {detail['variants']}")
    
    # Debug: Filter yapısını göster
    print(f"\n🔧 Filter Yapısı:")
    print(json.dumps(filter_result["filter"], indent=2, ensure_ascii=False))
    
    # 4. Yapısal arama
    print("\n🏗️ ADIM 4: Yapısal Arama")
    qdrant_params = {
        "filter": filter_result["filter"],
        "limit": 100,
        "with_payload": True
    }
    
    structured_result = search_structured(qdrant_params, filter_result['has_category_match'])
    print(f"Yapısal arama sonucu: {structured_result['count'] if structured_result.get('success') else 'Hata'}")
    
    # 5. Sonuçları göster
    display_results(structured_result)
    
    return {
        "query": user_query,
        "corrected_query": corrected_query,
        "analysis": analysis,
        "structured_result": structured_result
    }

def main():
    # Test sorguları
    test_queries = [
        " yedi günlük turlar",

    ]
    
    print("=== YAPISAL TUR ARAMA SİSTEMİ ===\n")
    
    for query in test_queries:
        result = structured_search(query)
        print("\n" + "🔚 SONUÇ ÖZET".center(80, "="))
        print(f"Orijinal sorgu: '{query}'")
        print(f"Düzeltilmiş sorgu: '{result['corrected_query']}'")
        print(f"Bulunan sonuç sayısı: {result['structured_result']['count'] if result['structured_result'].get('success') else 0}")
        print("=" * 80)
        print()
    
    # Kullanıcıdan canlı soru alma
    print("\n🎯 CANLI TEST - Soru girin (çıkmak için 'exit'):")
    while True:
        user_input = input("\n💬 Sorunuz: ")
        if user_input.lower() == 'exit':
            break
        
        structured_search(user_input)

if __name__ == "__main__":
    main()