import sys
import os
import argparse
import json
import requests
from typing import Dict, List, Any
import re
from config import URL_LINK
from result.display_result import display_filter_results
from runner_sync.runner import correct_user_query,analyze_tour_query
# Bu satır kaldırıldı - circular import nedeniyle
from result.display_result import display_filter_results

# Filter search için kullanılacak ana alanlar
FILTER_SEARCH_FIELDS = {"turtipi", "vizesiz", "ulasimtipi", "turkodu", "gecesayisi", "kesinkalkis"}
def create_filter_only_search(filter_fields):
    """
    Sadece filter search için gerekli koşulları oluşturur
    """
    field_mappings = {
        "gecesayisi": "gecesayisi",
        "turtipi": "turtipi",  # text_fields. kaldırıldı
        "ulasimtipi": "ulasimtipi",  # text_fields. kaldırıldı
        "vizesiz": "vizesiz",  # text_fields. kaldırıldı
        "kesinkalkis": "kesinkalkis",  # text_fields. kaldırıldı
        "turkodu": "turkodu"
    }
    
    must_conditions = []
    should_conditions = []
    
    for field_info in filter_fields:
        field = field_info["field"]
        value = field_info["value"]
        
        if field in field_mappings:
            # MUST koşulu ekle
            must_condition = {
                "key": field_mappings[field],
                "match": {"value": value}
            }
            must_conditions.append(must_condition)
            
            # SHOULD koşulu da ekle (alternatif arama için)
            should_condition = {
                "key": field_mappings[field],
                "match": {"value": value}
            }
            should_conditions.append(should_condition)
    
    return {
        "must": must_conditions,
        "should": should_conditions
    }

def search_structured(qdrant_params, has_category_match=False):
    """
    Yapısal arama (filter-based) - Önce must, sonra should dener
    """
    try:
        qdrant_url = f"{URL_LINK}/collections/mng/points/scroll"
        
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
                
                result_data = {
                    "success": True,
                    "method": "structured_must",
                    "data": points,
                    "count": len(points)
                }
                
                # Display fonksiyonunu çağır
                display_filter_results(result_data)
                
                return result_data
            else:
                print(f"❌ MUST ile sonuç bulunamadı, SHOULD ile deneniyor...")
        else:
            print(f"❌ MUST araması HTTP hatası: {response.status_code}, SHOULD ile deneniyor...")
        
        # 2. Must ile sonuç yoksa should ile dene
        if qdrant_params["filter"].get("should") and len(qdrant_params["filter"]["should"]) > 0:
            should_filter = {
                "should": qdrant_params["filter"]["should"]
            }
            
            request_body = {
                "filter": should_filter,
                "limit": qdrant_params.get("limit", 2),  # limit parametresi eklendi
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
                    
                    result_data = {
                        "success": True,
                        "method": "structured_should",
                        "data": points,
                        "count": len(points)
                    }
                    
                    # Display fonksiyonunu çağır
                    display_filter_results(result_data)
                    
                    return result_data
                else:
                    print(f"❌ SHOULD ile de sonuç bulunamadı")
            else:
                print(f"❌ SHOULD araması HTTP hatası: {response.status_code}")
        else:
            print(f"❌ SHOULD koşulları bulunamadı")
        
        # 3. Hiçbir sonuç bulunamadı
        print(f"❌ MUST ve SHOULD ile sonuç bulunamadı")
        result_data = {
            "success": True,
            "method": "no_results",
            "data": [],
            "count": 0
        }
        
        # Display fonksiyonunu çağır
        display_filter_results(result_data)
        
        return result_data
            
    except Exception as e:
        return {
            "success": False,
            "method": "structured",
            "error": str(e)
        }

def filter_search(user_query):
    """
    Sadece filter search yapan basit arama sistemi
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
    
    # 3. Filter search yap
    print("\n🔍 ADIM 3: Filter Search")
    
    if not analysis or "Analiz hatası" in analysis:
        print("❌ Analiz hatası nedeniyle arama yapılamadı")
        return {
            "query": user_query,
            "corrected_query": corrected_query,
            "analysis": analysis,
            "filter_fields": [],
            "search_result": None
        }
    
    # Analiz sonucundan filter alanlarını çıkar ve FILTER_SEARCH_FIELDS kontrolü yap
    filter_fields = []
    non_filter_fields = []
    criteria = analysis.split(", ")
    
    for criterion in criteria:
        if ":" in criterion:
            key, value = criterion.split(":", 1)
            key = key.strip()
            value = value.strip()
            
            if key in FILTER_SEARCH_FIELDS:
                filter_fields.append({"field": key, "value": value})
            else:
                non_filter_fields.append({"field": key, "value": value})
    
    # Eğer FILTER_SEARCH_FIELDS'de olmayan alanlar varsa multivector search'e yönlendir
    if non_filter_fields:
        print(f"⚠️ FILTER_SEARCH_FIELDS'de olmayan alanlar tespit edildi: {[f['field'] for f in non_filter_fields]}")
        print(f"✅ Multivector search'e yönlendiriliyor - Soru: '{corrected_query}'")
        
        # Multivector search import et ve çalıştır
        from search.multivector_search import search_tours
        search_result = search_tours(corrected_query, top_k=5)
        
        return {
            "query": user_query,
            "corrected_query": corrected_query,
            "analysis": analysis,
            "filter_fields": [],
            "non_filter_fields": non_filter_fields,
            "search_result": search_result,
            "search_method": "multivector"
        }
    
    # Eğer sadece FILTER_SEARCH_FIELDS'de olan alanlar varsa filter search yap
    elif filter_fields:
        print(f"✅ Tespit edilen filter alanları: {[f['field'] for f in filter_fields]}")
        
        # Filter search yap
        filter_conditions = create_filter_only_search(filter_fields)
        qdrant_params = {
            "filter": filter_conditions,
            "limit": 2,  # SADECE 1 KAYIT!
            "with_payload": True
        }
        
        # Kategori eşleşmesi var mı kontrol et
        has_category_match = any(f["field"] == "turKategori[].isim" for f in filter_fields)
        
        search_result = search_structured(qdrant_params, has_category_match)
        print(f"✅ Filter Search sonucu: {search_result['count'] if search_result.get('success') else 'Hata'}")
        
        # 4. Sonuçları göster
        print("\n📊 ADIM 4: Sonuçları Görüntüleme")
        display_filter_results(search_result)
        
        return {
            "query": user_query,
            "corrected_query": corrected_query,
            "analysis": analysis,
            "filter_fields": filter_fields,
            "non_filter_fields": [],
            "search_result": search_result,
            "search_method": "filter"
        }
    
    # Hiç alan bulunamadıysa multivector search'e yönlendir
    else:
        print("⚠️ Hiç alan bulunamadı, multivector search'e yönlendiriliyor")
        from search.multivector_search import search_tours
        search_result = search_tours(corrected_query, top_k=5)
        
        return {
            "query": user_query,
            "corrected_query": corrected_query,
            "analysis": analysis,
            "filter_fields": [],
            "non_filter_fields": [],
            "search_result": search_result,
            "search_method": "multivector"
        }


