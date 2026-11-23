from qdrant_client import models
from typing import Optional,Dict,Any,Tuple,List
# =========================================================
# region QDRANT FILTERING SEARCH
# =======================================================

# Tüm filtrelenebilir alanlar (hem düz hem nested)
# Aynı zamanda bu liste, "label_field" içindeki hangi alanın
# filtre olarak kullanılacağını belirler.
# Bu alanlar EXACT MATCH ile filtreleme için kullanılır (scroll/filter)
FILTER_SEARCH_FIELDS = [
        
        # ========================
        # Top Level Alanlar
        # ========================
        "gecesayisi",
        "geceleme",
        "turtipi", 
        "ulasimtipi",
        "ulasim",
        "vizesiz",
        "kalkissehir",
        "kesinkalkis",
        "turkodu",
        "vizeulke",
        "url",
        "id",
        
        # ========================
        # TarihFiyat (Array of Object)
        # ========================
        "tarihFiyat[].tarih.gun",
        "tarihFiyat[].tarih.ay",
        "tarihFiyat[].tarih.yil",
        "tarihFiyat[].kisibasi", 
        "tarihFiyat[].kur",
        "tarihFiyat[].kalan",
        
        # ========================
        # Konaklama Nested Alanlar
        # ========================
        "konaklama.konaklama_tipi",
        "konaklama.konum_tipi",
        "konaklama.otel_markasi",
        "konaklama.otel_yildiz",
        "konaklama.pansiyon_tipi",

        # ========================
        # Ziyaret
        # ========================
        "ziyaretedilecekyerler",
        
        # ⚠️ turKategori[].isim ve turKategori[].puan BURADA OLMAZ!
        # Bu alanlar semantik/embedding ile aranır (MULTIVECTOR_SEARCH_FIELDS)
]

# =========================================================
# MULTIVECTOR / SEMANTIC SEARCH ALANLARI
# Bu alanlar SADECE vektör/semantik arama için kullanılır
# Soru embedding'e çevrilir ve Qdrant'ta vektör araması yapılır
# =========================================================
MULTIVECTOR_SEARCH_FIELDS = [
    "isim",                     # Tur adı - semantik benzerlik
    "metin",                    # Açıklama metni - semantik benzerlik
    "ulasim",                   # Ulaşım detayı - semantik benzerlik
    "ziyaretedilecekyerler",    # Gezilecek yerler - semantik benzerlik
    "turKategori[].isim",       # Kategori ismi - semantik benzerlik (ASLA filter değil!)
    "turKategori[].puan",       # Kategori puanı - semantik benzerlik
]


def classify_search_fields(
    label_field: Dict[str, Any],
    kume_list: Dict[str, List[str]]
) -> Tuple[Dict[str, Any], Dict[str, Any], str]:
    """
    Agent'ın çıktısındaki alanları filter ve multivector olarak sınıflandırır.
    
    Args:
        label_field: Agent'dan gelen {alan: değer} dict
        kume_list: Agent'dan gelen {"must": [...], "should": [...]} dict
    
    Returns:
        Tuple[filter_fields, multivector_fields, search_strategy]
        - filter_fields: Filtreleme için kullanılacak alanlar
        - multivector_fields: Semantik arama için kullanılacak alanlar  
        - search_strategy: "filter_only" | "multivector_only" | "hybrid_combined"
    """
    filter_fields: Dict[str, Any] = {}
    multivector_fields: Dict[str, Any] = {}
    
    # label_field içindeki alanları sınıflandır
    for field_name, field_value in label_field.items():
        # Boş değerleri atla
        if field_value is None or field_value == "" or field_value == []:
            continue
            
        is_filter = field_name in FILTER_SEARCH_FIELDS
        is_multivector = field_name in MULTIVECTOR_SEARCH_FIELDS
        
        if is_filter and is_multivector:
            # Her iki kategoride de varsa, her ikisine de ekle
            filter_fields[field_name] = field_value
            multivector_fields[field_name] = field_value
        elif is_filter:
            filter_fields[field_name] = field_value
        elif is_multivector:
            multivector_fields[field_name] = field_value
        else:
            # Bilinmeyen alan - multivector'a ekle (semantik arama denesin)
            multivector_fields[field_name] = field_value
    
    # Arama stratejisini belirle
    has_filter = bool(filter_fields)
    has_multivector = bool(multivector_fields)
    
    if has_filter and has_multivector:
        search_strategy = "hybrid_combined"
    elif has_filter:
        search_strategy = "filter_only"
    elif has_multivector:
        search_strategy = "multivector_only"
    else:
        search_strategy = "multivector_only"  # Default: semantik arama
    
    return filter_fields, multivector_fields, search_strategy


def build_kume_filter_dict(
    filter_fields: Dict[str, Any],
    kume_list: Dict[str, List[str]]
) -> Dict[str, Dict[str, Any]]:
    """
    Filter alanlarını kume_list'e göre gruplandırarak filter_dict oluşturur.
    
    Args:
        filter_fields: {alan: değer} dict
        kume_list: {"must": [alan1, alan2], "should": [alan3]} dict
    
    Returns:
        {"must": {alan: değer}, "should": {alan: değer}, "must_not": {alan: değer}}
    """
    filter_dict: Dict[str, Dict[str, Any]] = {}
    
    for group_name in ["must", "should", "must_not"]:
        group_fields = kume_list.get(group_name, [])
        group_data = {
            f: filter_fields[f]
            for f in group_fields
            if f in filter_fields
        }
        if group_data:
            filter_dict[group_name] = group_data
    
    # kume_list'te olmayan ama filter_fields'ta olan alanları must'a ekle
    all_kume_fields = set()
    for fields in kume_list.values():
        all_kume_fields.update(fields)
    
    orphan_fields = {
        f: v for f, v in filter_fields.items() 
        if f not in all_kume_fields
    }
    if orphan_fields:
        if "must" not in filter_dict:
            filter_dict["must"] = {}
        filter_dict["must"].update(orphan_fields)
    
    return filter_dict

# Payload olarak Qdrant'a gönderilebilecek top-level alan adları
# (nested [] notasyonlu alanlar Qdrant with_payload'da kullanılamaz,
#  bunun yerine üst alan adı kullanılır)
NESTED_FIELD_TO_PAYLOAD = {
    "tarihFiyat[].tarih.gun":  "tarihFiyat",
    "tarihFiyat[].tarih.ay":   "tarihFiyat",
    "tarihFiyat[].tarih.yil":  "tarihFiyat",
    "tarihFiyat[].kisibasi":   "tarihFiyat",
    "tarihFiyat[].kur":        "tarihFiyat",
    "tarihFiyat[].kalan":      "tarihFiyat",
    "konaklama.konaklama_tipi":"konaklama",
    "konaklama.konum_tipi":    "konaklama",
    "konaklama.otel_markasi":  "konaklama",
    "konaklama.otel_yildiz":   "konaklama",
    "konaklama.pansiyon_tipi": "konaklama",
    "turKategori[].isim":      "turKategori",
    "turKategori[].puan":      "turKategori",
    "ziyaretedilecekyerler":   "ziyaretedilecekyerler",
}


def resolve_payload_fields(field_names: list[str]) -> list[str]:
    """
    Agent'dan gelen [] veya noktalı notasyonlu alan adlarını
    Qdrant with_payload için geçerli üst alan adlarına dönüştürür.
    Örn: "tarihFiyat[].kisibasi" → "tarihFiyat"
    """
    resolved = []
    for f in field_names:
        top = NESTED_FIELD_TO_PAYLOAD.get(f, f)   # bilinmiyorsa olduğu gibi bırak
        if top not in resolved:
            resolved.append(top)
    return resolved


def build_advanced_filter(
    filter_dict: Optional[Dict[str, Any]],
    explode_lists: bool = False
) -> Optional[models.Filter]:
    """
    filter_dict örn:
    {
        "must":     {"turtipi": "yurt içi"},
        "should":   {"ulasimtipi": ["Uçaklı", "Otobüslü"]},
        "must_not": {"tarihFiyat[].tarih.ay": "Ocak"}
    }
    Boş string ("") veya None değerli alanlar otomatik olarak atlanır.
    """

    if not filter_dict:
        return None

    def build_group(group_data: Dict[str, Any]):
        conditions = []

        for key, value in group_data.items():

            # ---- Boş / None değerleri atla ----
            if value is None or value == "" or value == []:
                continue

            if isinstance(value, list):
                # Listeden None / "" temizle
                clean_list = [v for v in value if v is not None and v != ""]
                if not clean_list:
                    continue

                if explode_lists:
                    for v in clean_list:
                        conditions.append(
                            models.FieldCondition(
                                key=key,
                                match=models.MatchValue(value=v)
                            )
                        )
                else:
                    conditions.append(
                        models.FieldCondition(
                            key=key,
                            match=models.MatchAny(any=clean_list)
                        )
                    )

            else:
                conditions.append(
                    models.FieldCondition(
                        key=key,
                        match=models.MatchValue(value=value)
                    )
                )

        return conditions if conditions else None

    must     = build_group(filter_dict.get("must",     {}))
    should   = build_group(filter_dict.get("should",   {}))
    must_not = build_group(filter_dict.get("must_not", {}))

    if not any([must, should, must_not]):
        return None

    if should:
        return models.Filter(
            must=must or None,
            must_not=must_not or None,
            min_should=models.MinShould(
                conditions=should,
                min_count=1
            )
        )
    else:
        return models.Filter(
            must=must or None,
            must_not=must_not or None
        )

def build_raw_filter(filter_dict: Optional[Dict[str, Any]]) -> Optional[dict]:
    """
    filter_dict'i Qdrant REST API scroll endpointi için ham JSON filtre
    yapısına dönüştürür.  qdrant-client serileştirme sorunlarını bypass eder.

    Çıktı doğrudan Postman'daki çalışan formatla aynıdır:
    {
        "must": [{"key": "turtipi", "match": {"value": "yurt içi"}}],
        "must_not": [{"key": "id", "match": {"any": [677, 678]}}]
    }
    """
    if not filter_dict:
        return None

    result: dict = {}

    for group_name in ("must", "should", "must_not"):
        group_data = filter_dict.get(group_name, {})
        if not group_data:
            continue

        conditions: list[dict] = []
        for key, value in group_data.items():
            # Boş / None atla
            if value is None or value == "" or value == []:
                continue

            if isinstance(value, list):
                clean = [v for v in value if v is not None and v != ""]
                if clean:
                    conditions.append({"key": key, "match": {"any": clean}})
            else:
                conditions.append({"key": key, "match": {"value": value}})

        if conditions:
            result[group_name] = conditions

    return result if result else None


# endregion
# =========================================================
