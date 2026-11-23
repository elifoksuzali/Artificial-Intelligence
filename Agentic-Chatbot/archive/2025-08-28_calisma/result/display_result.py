

def display_multivector_results(search_result):
    """
    Multivector search sonuçlarını gösterir
    """
    print("\n" + "="*80)
    print("🧠 MULTIVECTOR SEARCH SONUÇLARI")
    print("="*80)
    
    if search_result.get("success") and search_result.get("data"):
        points = search_result["data"]
        print(f"\n📊 MULTIVECTOR SEARCH SONUÇLARI ({len(points)} adet):")
        print("-" * 50)
        
        # Tüm sonuçları göster (limit kontrolü yok)
        for i, point in enumerate(points):
            if isinstance(point, dict):
                point_id = point.get("id")
                payload = point.get("payload", {})
            else:
                point_id = getattr(point, 'id', None)
                payload = getattr(point, 'payload', {})
            
            print(f"🔍 Sonuç #{i+1}:")
            
            # Tur bilgilerini göster
            turkodu = payload.get('turkodu', payload.get('turKodu', payload.get('id', 'N/A')))
            isim = payload.get('isim', payload.get('name', 'N/A'))
            gecesayisi = payload.get('gecesayisi', payload.get('geceSayisi', 'N/A'))
            konaklama = payload.get('konaklama', payload.get('konakLama', 'N/A'))
            ulasimtipi = payload.get('ulasimtipi', payload.get('ulasimTipi', 'N/A'))
            turtipi = payload.get('turtipi', payload.get('turTipi', 'N/A'))
            
            print(f"   Tur Kodu: {turkodu}")
            print(f"   Adı: {isim}")
            print(f"   Gece Sayısı: {gecesayisi}")
            print(f"   Konaklama: {konaklama}")
            print(f"   Ulaşım: {ulasimtipi}")
            print(f"   Tur Tipi: {turtipi}")
            
            # URL varsa göster
            if 'url' in payload:
                print(f"   URL: {payload['url']}")
            
            print()
    else:
        print(f"\n📊 MULTIVECTOR SEARCH SONUÇLARI: Sonuç bulunamadı")
        if not search_result.get("success"):
            print(f"   Hata: {search_result.get('error')}")


def display_filter_results(structured_result):
    """
    Yapısal arama sonuçlarını gösterir
    """
    print("\n" + "="*80)
    print("🔍 FILTER SEARCH SONUÇLARI")
    print("="*80)
    
    # Structured results
    structured_points = structured_result.get("data", []) if structured_result.get("success") else []
    method = structured_result.get("method", "unknown")
    
    if structured_points:
        method_text = "MUST" if "must" in method else "SHOULD" if "should" in method else "UNKNOWN"
        print(f"\n📊 FILTER SEARCH SONUÇLARI ({len(structured_points)} adet) - {method_text} ile bulundu:")
        print("-" * 50)
        
        print("-" * 50)
        
        # Tüm sonuçları göster (limit kontrolü yok)
        for i, point in enumerate(structured_points):
            if isinstance(point, dict):
                point_id = point.get("id")
                payload = point.get("payload", {})
            else:
                point_id = getattr(point, 'id', None)
                payload = getattr(point, 'payload', {})
            
            print(f"🔍 Sonuç #{i+1}:")
            
            # Try different possible data structures for turkodu
            # Check multiple possible field names for tur kodu
            turkodu = None
            if 'turkodu' in payload:
                turkodu = payload['turkodu']
            elif 'turKodu' in payload:
                turkodu = payload['turKodu']
            elif 'tur_kodu' in payload:
                turkodu = payload['tur_kodu']
            elif 'kod' in payload:
                turkodu = payload['kod']
            elif 'id' in payload:
                turkodu = payload['id']
            else:
                # Check in text_fields if exists
                if 'text_fields' in payload and isinstance(payload['text_fields'], dict):
                    text_fields = payload['text_fields']
                    if 'turkodu' in text_fields:
                        turkodu = text_fields['turkodu']
                    elif 'turKodu' in text_fields:
                        turkodu = text_fields['turKodu']
                    elif 'kod' in text_fields:
                        turkodu = text_fields['kod']
                
                # If still not found, try to extract from URL
                if not turkodu and 'url' in payload:
                    url = payload['url']
                    # Extract tour code from URL if possible
                    import re
                    url_match = re.search(r'/(\d+)(?:-|$)', url)
                    if url_match:
                        turkodu = f"URL-{url_match.group(1)}"
            
            # If still no turkodu, use point_id
            if not turkodu:
                turkodu = f"ID-{point_id}" if point_id else "N/A"
            
            # Try different possible data structures for other fields
            isim = payload.get('isim', payload.get('name', 'N/A'))
            gecesayisi = payload.get('gecesayisi', payload.get('geceSayisi', 'N/A'))
            geceleme = payload.get('geceleme', payload.get('geceLeme', 'N/A'))
            konaklama = payload.get('konaklama', payload.get('konakLama', 'N/A'))
            ulasimtipi = payload.get('ulasimtipi', payload.get('ulasimTipi', 'N/A'))
            turtipi = payload.get('turtipi', payload.get('turTipi', 'N/A'))
            ziyaretedilecekyerler = payload.get('ziyaretedilecekyerler', payload.get('ziyaretEdilecekYerler', 'N/A'))
            kesinkalkis = payload.get('kesinkalkis', payload.get('kesinKalkis', 'N/A'))
            
            # Check text_fields if exists
            if 'text_fields' in payload:
                text_fields = payload['text_fields']
                if not isim or isim == 'N/A':
                    isim = text_fields.get('isim', text_fields.get('name', 'N/A'))
                if not geceleme or geceleme == 'N/A':
                    geceleme = text_fields.get('geceleme', text_fields.get('geceLeme', 'N/A'))
                if not konaklama or konaklama == 'N/A':
                    konaklama = text_fields.get('konaklama', text_fields.get('konakLama', 'N/A'))
                if not ulasimtipi or ulasimtipi == 'N/A':
                    ulasimtipi = text_fields.get('ulasimtipi', text_fields.get('ulasimTipi', 'N/A'))
                if not turtipi or turtipi == 'N/A':
                    turtipi = text_fields.get('turtipi', text_fields.get('turTipi', 'N/A'))
                if not ziyaretedilecekyerler or ziyaretedilecekyerler == 'N/A':
                    ziyaretedilecekyerler = text_fields.get('ziyaretedilecekyerler', text_fields.get('ziyaretEdilecekYerler', 'N/A'))
                if not kesinkalkis or kesinkalkis == 'N/A':
                    kesinkalkis = text_fields.get('kesinkalkis', text_fields.get('kesinKalkis', 'N/A'))
            
            # Check for nested structures
            if isinstance(ziyaretedilecekyerler, list):
                ziyaretedilecekyerler = ", ".join(ziyaretedilecekyerler)
            
            # Display the results
            print(f"   Tur Kodu: {turkodu}")
            print(f"   Adı: {isim}")
            print(f"   Gece Sayısı: {gecesayisi}")
            print(f"   Geceleme: {geceleme}")
            print(f"   Konaklama: {konaklama}")
            print(f"   Ulaşım: {ulasimtipi}")
            print(f"   Tur Tipi: {turtipi}")
            print(f"   Tur Güzergahı: {ziyaretedilecekyerler}")
            print(f"   Kesin Kalkışlı: {kesinkalkis}")
            
            # Show additional fields if available
            if 'url' in payload:
                print(f"   URL: {payload['url']}")
            if 'turKategori' in payload and payload['turKategori']:
                categories = []
                for cat in payload['turKategori']:
                    if isinstance(cat, dict):
                        cat_name = cat.get('isim', 'N/A')
                        cat_puan = cat.get('puan', 'N/A')
                        categories.append(f"{cat_name}({cat_puan})")
                if categories:
                    print(f"   Kategoriler: {', '.join(categories)}")
            
            print()
    else:
        print(f"\n📊 FILTER SEARCH SONUÇLARI: Sonuç bulunamadı")
        if not structured_result.get("success"):
            print(f"   Hata: {structured_result.get('error')}")
        else:
            print(f"   MUST ve SHOULD ile sonuç bulunamadı")

