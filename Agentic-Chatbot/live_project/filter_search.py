from qdrant_client import models
from typing import Optional,Dict,Any,Tuple,List
import asyncio
import httpx
from connection_qdrant import QdrantDatabase, qdrant_host, qdrant_api_key
client = QdrantDatabase().get_client()
import logging, re
logger = logging.getLogger(__name__)
from constants import ErrorCode
from setting import QDRANT_COLLECTION

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


# =========================================================
# region QDRANT HELPER (Retry ile)
# =========================================================

async def qdrant_scroll_with_retry(
    scroll_body: dict,
    max_retries: int = 3,
) -> dict | None:
    """
    Qdrant scroll endpoint'ine retry mekanizması ile sorgu yapar.
    
    Args:
        scroll_body: Qdrant scroll sorgu body'si
        max_retries: Maksimum deneme sayısı
    
    Returns:
        Qdrant yanıtı (dict) veya None
    """
    last_error = None
    
    for attempt in range(1, max_retries + 1):
        try:
            async with httpx.AsyncClient(verify=False, timeout=30) as http_qdrant:
                qdrant_resp = await http_qdrant.post(
                    f"{qdrant_host}/collections/{QDRANT_COLLECTION}/points/scroll",
                    json=scroll_body,
                    headers={
                        "api-key": qdrant_api_key,
                        "Content-Type": "application/json",
                    },
                )
                
                if qdrant_resp.status_code == 200:
                    if attempt > 1:
                        logger.info(
                            f"✅ Qdrant sorgusu başarılı (deneme {attempt}/{max_retries})"
                        )
                    return qdrant_resp.json()
                elif qdrant_resp.status_code >= 500:
                    last_error = f"HTTP {qdrant_resp.status_code}"
                    logger.warning(
                        f"[{ErrorCode.QDRANT_QUERY_ERROR.code}] "
                        f"Qdrant server hatası (deneme {attempt}/{max_retries}): "
                        f"{qdrant_resp.status_code}"
                    )
                else:
                    logger.error(
                        f"[{ErrorCode.QDRANT_QUERY_ERROR.code}] "
                        f"Qdrant sorgu hatası: {qdrant_resp.status_code} — "
                        f"{qdrant_resp.text[:200]}"
                    )
                    return None
                    
        except httpx.TimeoutException as e:
            last_error = e
            logger.warning(
                f"[{ErrorCode.QDRANT_CONNECTION_ERROR.code}] "
                f"Qdrant timeout (deneme {attempt}/{max_retries}): {e}"
            )
        except httpx.ConnectError as e:
            last_error = e
            logger.warning(
                f"[{ErrorCode.QDRANT_CONNECTION_ERROR.code}] "
                f"Qdrant bağlantı hatası (deneme {attempt}/{max_retries}): {e}"
            )
        except Exception as e:
            last_error = e
            logger.warning(
                f"[{ErrorCode.QDRANT_QUERY_ERROR.code}] "
                f"Qdrant hatası (deneme {attempt}/{max_retries}): "
                f"{type(e).__name__}: {e}"
            )
        
        if attempt < max_retries:
            delay = 2 ** attempt
            logger.info(f"⏳ Qdrant {delay}s sonra yeniden deneniyor...")
            await asyncio.sleep(delay)
    
    logger.error(
        f"[{ErrorCode.MAX_RETRIES_EXCEEDED.code}] "
        f"Qdrant tüm denemeler başarısız ({max_retries} deneme): {last_error}"
    )
    return None


# endregion
# =========================================================

# =========================================================
# region PAYLOAD → TOON DÖNÜŞÜM
# =========================================================

def flatten_tour_payload(item: dict) -> dict:
    """
    Qdrant'tan gelen nested payload'u düz (flat) ve okunabilir formata çevirir.
    Tüm alanlar handle edilir — nested/array yapılar tek string'e indirgenir.

    Kaynak (Qdrant payload)                 →  Hedef (flat dict)
    ──────────────────────────────────────────────────────────────
    id                                      →  id
    isim                                    →  isim
    geceleme / gecesayisi                   →  sure          "3 Gece 4 Gün"
    turtipi[]                               →  tur_tipi      "yurt dışı"
    konaklama{..., tam_metin}               →  konaklama     "3* - 4* Oteller"
    ulasim + ulasimtipi                     →  ulasim        "THY ile İstanbul - Malta"
    tarihFiyat[{tarih,kisibasi,kur,kalan}]  →  fiyat         ["9 Ekim 2025 (799.0 EUR) Kalan:20", ...]
    kalkissehir                             →  kalkis        "İstanbul"
    kesinkalkis                             →  kesin_kalkis  "Evet" / "Hayır"
    vizesiz                                 →  vize          "Vizesiz" / "Vizeli"
    vizeulke                                →  vize_ulke     "Yunanistan"
    ziyaretedilecekyerler[]                 →  yerler        "Malta, Gozo"
    turkodu                                 →  kod           "AVT323"
    url                                     →  url           "https://..."
    metin                                   →  aciklama      "MNG Turizm güvencesi ile..."
    """
    flat: dict = {}

    # ── id + isim (her zaman) ──
    flat["id"] = item.get("id")
    flat["isim"] = item.get("isim")

    # ── geceleme / gecesayisi → süre ──
    geceleme = item.get("geceleme")
    gecesayisi = item.get("gecesayisi")
    if geceleme:
        flat["sure"] = geceleme
    elif gecesayisi is not None:
        flat["sure"] = f"{gecesayisi} Gece"

    # ── turtipi → tek string ──
    turtipi = item.get("turtipi")
    if turtipi:
        if isinstance(turtipi, list):
            joined = ", ".join(str(t) for t in turtipi if t)
            if joined:
                flat["tur_tipi"] = joined
        else:
            flat["tur_tipi"] = str(turtipi)

    # ── konaklama → tam_metin ──
    konaklama = item.get("konaklama")
    if konaklama and isinstance(konaklama, dict):
        tam_metin = konaklama.get("tam_metin")
        if tam_metin:
            flat["konaklama"] = tam_metin

    # ── ulasim / ulasimtipi → tek alan (detaylı olan tercih edilir) ──
    ulasim = item.get("ulasim")
    ulasimtipi = item.get("ulasimtipi")
    if ulasim:
        flat["ulasim"] = ulasim
    elif ulasimtipi:
        flat["ulasim"] = ulasimtipi

    # ── tarihFiyat → okunabilir string listesi ──
    tarih_fiyat = item.get("tarihFiyat")
    if tarih_fiyat and isinstance(tarih_fiyat, list):
        fiyatlar = []
        for tf in tarih_fiyat:
            parts = []
            tarih = tf.get("tarih") or {}
            gun = tarih.get("gun", "")
            ay = tarih.get("ay", "")
            yil = tarih.get("yil", "")
            tarih_str = f"{gun} {ay} {yil}".strip()
            if tarih_str:
                parts.append(tarih_str)

            kisibasi = tf.get("kisibasi")
            kur = tf.get("kur", "")
            if kisibasi is not None:
                parts.append(f"({kisibasi} {kur})")

            kalan = tf.get("kalan")
            if kalan is not None:
                parts.append(f"Kalan:{kalan}")

            if parts:
                fiyatlar.append(" ".join(parts))
        if fiyatlar:
            flat["fiyat"] = fiyatlar

    # ── kalkissehir ──
    kalkissehir = item.get("kalkissehir")
    if kalkissehir:
        flat["kalkis"] = kalkissehir

    # ── kesinkalkis → kısa format ──
    kesinkalkis = item.get("kesinkalkis")
    if kesinkalkis:
        ks = str(kesinkalkis).lower()
        if "kesin kalkışlıdır" in ks or kesinkalkis == "1":
            flat["kesin_kalkis"] = "Evet"
        elif "kesin kalkışlı değildir" in ks or kesinkalkis == "0":
            flat["kesin_kalkis"] = "Hayır"
        else:
            flat["kesin_kalkis"] = kesinkalkis

    # ── vizesiz → kısa format ──
    vizesiz = item.get("vizesiz")
    if vizesiz:
        vz = str(vizesiz).lower()
        if "vizesizdir" in vz or vizesiz == "0":
            flat["vize"] = "Vizesiz"
        elif "vizelidir" in vz or vizesiz == "1":
            flat["vize"] = "Vizeli"
        else:
            flat["vize"] = vizesiz

    # ── vizeulke ──
    vizeulke = item.get("vizeulke")
    if vizeulke:
        flat["vize_ulke"] = vizeulke

    # ── ziyaretedilecekyerler → virgüllü string ──
    yerler = item.get("ziyaretedilecekyerler")
    if yerler:
        if isinstance(yerler, list):
            joined = ", ".join(str(y) for y in yerler if y)
            if joined:
                flat["yerler"] = joined
        else:
            flat["yerler"] = str(yerler)


    # ── turkodu ──
    turkodu = item.get("turkodu")
    if turkodu:
        flat["kod"] = turkodu

    # ── url ──
    url = item.get("url")
    if url:
        flat["url"] = str(url)

    # ── metin (açıklama) → tek satır ──
    metin = item.get("metin")
    if metin:
        flat["aciklama"] = " ".join(str(metin).split())

    return flat


# TOON çıktısında alan sırası ve etiketleri
_TOON_FIELD_ORDER = [
    ("sure",         "Süre"),
    ("tur_tipi",     "Tur Tipi"),
    ("konaklama",    "Konaklama"),
    ("ulasim",       "Ulaşım"),
    ("fiyat",        "Fiyat"),
    ("kalkis",       "Kalkış"),
    ("kesin_kalkis", "Kesin Kalkış"),
    ("vize",         "Vize"),
    ("vize_ulke",    "Vize Ülke"),
    ("yerler",       "Güzergah Yerleri"),
    ("kod",          "Tur Kodu"),
    ("url",          "URL"),
    ("aciklama",     "Açıklama"),
]

def to_toon(items: list) -> str:
    """
    Tur listesini TOON (Token-Optimized Notation) formatına çevirir.
    JSON'a kıyasla ~%60-70 daha az token kullanır.

    Çıktı örneği (fiyatsız):
    [#171] Malta Turu THY ile 3 Gece | Süre: 3 Gece 4 Gün | Tür: yurt dışı | Konaklama: 3*-4* Oteller | Ulaşım: THY ile İstanbul-Malta | Vize: Vizesiz | Yerler: Malta | Kod: AVT323

    Çıktı örneği (fiyatlı):
    [#171] Malta Turu THY ile 3 Gece | Konaklama: 3*-4* Oteller | Ulaşım: THY
        📊 Fiyat Bilgileri:
        - 9 Ekim 2025 (799.0 EUR) Kalan:20
        - 15 Kasım 2025 (899.0 EUR) Kalan:10
    """
    lines = []
    for item in items:
        parts = [f"[#{item.get('id', '?')}] {item.get('isim', '')}"]
        fiyat_entries: list[str] = []

        # Sabit sırayla tüm alanlar
        for key, label in _TOON_FIELD_ORDER:
            val = item.get(key)
            if val is None:
                continue
            # Fiyat: ayrı satırlarda tablo-uyumlu format
            if key == "fiyat" and isinstance(val, list):
                fiyat_entries = [str(v) for v in val]
            elif isinstance(val, list):
                parts.append(f"{label}: {', '.join(str(v) for v in val)}")
            else:
                parts.append(f"{label}: {val}")

        main_line = " | ".join(parts)
        if fiyat_entries:
            main_line += "\n  📊 Fiyat Bilgileri:"
            for entry in fiyat_entries:
                main_line += f"\n    - {entry}"
        lines.append(main_line)

    return "\n".join(lines)

# endregion
# =========================================================




# =========================================================
# region MORE TOURS — Geçmiş ID Çıkarma
# =========================================================
_TOON_ID_PATTERN = re.compile(r'\[#(\d+)\]')


def extract_shown_tour_ids(chat_history: list[dict]) -> list[int]:
    """
    Chat history'deki TOON çıktılarından daha önce gösterilmiş
    tur ID'lerini çıkarır.  [#677] → 677

    Sadece assistant mesajlarına bakar (kullanıcı mesajlarında
    ID olması durumunu hariç tutar).
    """
    ids: set[int] = set()
    for msg in chat_history:
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", "")
        for m in _TOON_ID_PATTERN.findall(content):
            ids.add(int(m))
    return sorted(ids)


def find_original_tour_query(chat_history: list[dict]) -> str | None:
    """
    more_tours: TOON yanıt zincirini geriye doğru takip ederek
    orijinal arama sorgusunu bulur.

    Zincir mantığı:
        user: "yurt içi turlar"      ← BU DÖNER (orijinal sorgu)
        assistant: [#677] ...         (TOON yanıt)
        user: "başka var mı"
        assistant: [#171] ...         (TOON yanıt)
        user: "daha var mı acaba"     ← lastMessage (more_tours)

    Algoritma:
    1. Sondan geriye doğru ilk TOON assistant mesajını bul
    2. Hemen öncesindeki user mesajını al
    3. Eğer o user mesajı da bir TOON'dan sonra gelmişse (yani
        kendisi de "more" tarzı), daha geriye git
    4. Önünde TOON olmayan ilk user mesajı = orijinal sorgu
    """
    idx = len(chat_history) - 1

    while idx >= 0:
        # TOON olmayan mesajları atla
        if chat_history[idx].get("role") != "assistant" or \
            not _TOON_ID_PATTERN.search(chat_history[idx].get("content", "")):
            idx -= 1
            continue

        # TOON bulundu (index `idx`). Hemen önceki user mesajını bul.
        user_idx = None
        for j in range(idx - 1, -1, -1):
            if chat_history[j].get("role") == "user":
                user_idx = j
                break

        if user_idx is None:
            return None  # TOON'dan önce user mesajı yok

        # Bu user mesajının hemen öncesinde de TOON var mı?
        # Varsa → bu user "more" zincirinin parçası, daha geriye git
        has_preceding_toon = False
        for k in range(user_idx - 1, -1, -1):
            if chat_history[k].get("role") == "assistant":
                if _TOON_ID_PATTERN.search(chat_history[k].get("content", "")):
                    has_preceding_toon = True
                    idx = k      # Önceki TOON'a atla
                break            # İlk assistant'ta dur

        if not has_preceding_toon:
            return chat_history[user_idx]["content"]
        # has_preceding_toon → while döngüsü idx=k ile devam eder

    return None


# endregion
# =========================================================
