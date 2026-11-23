from qdrant_client import QdrantClient,models
from typing import Optional,Dict,Any
# =========================================================
# region QDRANT FILTERING SEARCH
# =======================================================

# Tüm filtrelenebilir alanlar (hem düz hem nested)
# Aynı zamanda bu liste, "label_field" içindeki hangi alanın
# filtre olarak kullanılacağını belirler.
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
        # Kategori / Ziyaret
        # ========================
        "turKategori[].isim",
        "turKategori[].puan",
        "ziyaretedilecekyerler",
]

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
