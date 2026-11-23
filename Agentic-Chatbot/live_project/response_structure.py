from typing import Optional, List, Dict, Any, Tuple
from pydantic import BaseModel, HttpUrl, field_validator, ConfigDict
from difflib import get_close_matches
import logging

_logger = logging.getLogger(__name__)

# =========================================================
# region BASEMODEL
# =========================================================

# =========================================================
# 🔒 GEÇERLİ ALAN İSİMLERİ (SABİT)
# LLM bu alan isimlerinden başka bir şey dönemez.
# Hatalı dönerse fuzzy matching ile düzeltilir.
# =========================================================
VALID_QDRANT_FIELDS = frozenset({
    # ─── Top-level alanlar ───
    "id", "isim", "turkodu", "gecesayisi", "geceleme",
    "kalkissehir", "kesinkalkis", "turtipi",
    "ulasim", "ulasimtipi", "vizesiz", "vizeulke", "url",
    "ziyaretedilecekyerler", "metin",
    # ─── Konaklama (nested) ───
    "konaklama.konaklama_tipi", "konaklama.konum_tipi",
    "konaklama.otel_markasi", "konaklama.otel_yildiz",
    "konaklama.pansiyon_tipi",
    # ─── TarihFiyat (array of objects) ───
    "tarihFiyat[].tarih.gun", "tarihFiyat[].tarih.ay",
    "tarihFiyat[].tarih.yil", "tarihFiyat[].kisibasi",
    "tarihFiyat[].kur", "tarihFiyat[].kalan",
    # ─── TurKategori (array of objects) ───
    "turKategori[].isim", "turKategori[].puan",
})

# payload_field için ek olarak top-level parent isimler de kabul edilir
VALID_PAYLOAD_EXTRAS = frozenset({
    "konaklama", "tarihFiyat", "turKategori",
})


def _correct_field_name(name: str, allow_payload_extras: bool = False):
    """
    Alan adını VALID_QDRANT_FIELDS içinde arar.
    Bulamazsa fuzzy matching ile en yakın geçerli ismi döner.
    Eşleşme yoksa None döner (alan atılacak).

    Örnekler:
        'ulasi'     → 'ulasim'       (fuzzy düzeltme)
        'ulasim'    → 'ulasim'       (birebir eşleşme)
        'xyz123'    → None           (geçersiz → atılır)
    """
    pool = VALID_QDRANT_FIELDS | VALID_PAYLOAD_EXTRAS if allow_payload_extras else VALID_QDRANT_FIELDS
    if name in pool:
        return name
    matches = get_close_matches(name, list(pool), n=1, cutoff=0.70)
    return matches[0] if matches else None


class QuestionFieldResult(BaseModel):
    """
    question_field_agent'ın JSON çıktısını doğrulayan Pydantic modeli.
    Hatalı alan isimlerini fuzzy matching ile otomatik düzeltir.
    Geçersiz alanları atar (VALID_QDRANT_FIELDS dışı).
    """
    label_field: Dict[str, Any] = {}
    payload_field: List[str] = []
    search_field: List[str] = []
    kume_list: Dict[str, List[str]] = {}

    @classmethod
    def validate_agent_output(cls, raw: dict) -> Tuple["QuestionFieldResult", List[str]]:
        """
        Agent çıktısını doğrula ve düzelt.
        Returns: (QuestionFieldResult, corrections_log)

        Kullanım:
            validated, corrections = QuestionFieldResult.validate_agent_output(agent_dict)
        """
        corrections: List[str] = []

        # ── label_field: key'leri doğrula/düzelt ──
        raw_label = raw.get("label_field", {})
        corrected_label: Dict[str, Any] = {}
        if isinstance(raw_label, dict):
            for k, v in raw_label.items():
                fixed = _correct_field_name(k)
                if fixed:
                    if fixed != k:
                        corrections.append(f"label_field: '{k}' → '{fixed}'")
                    corrected_label[fixed] = v
                else:
                    corrections.append(f"label_field: '{k}' geçersiz → ATILDI")

        # ── payload_field: item'ları doğrula/düzelt ──
        raw_payload = raw.get("payload_field", [])
        corrected_payload: List[str] = []
        if isinstance(raw_payload, list):
            for f in raw_payload:
                if not isinstance(f, str):
                    continue
                fixed = _correct_field_name(f, allow_payload_extras=True)
                if fixed:
                    if fixed != f:
                        corrections.append(f"payload_field: '{f}' → '{fixed}'")
                    if fixed not in corrected_payload:
                        corrected_payload.append(fixed)
                else:
                    corrections.append(f"payload_field: '{f}' geçersiz → ATILDI")

        # ── search_field: sadece geçerli gruplar ──
        valid_groups = {"must", "should", "must_not"}
        raw_search = raw.get("search_field", [])
        corrected_search: List[str] = []
        if isinstance(raw_search, list):
            for s in raw_search:
                if s in valid_groups:
                    corrected_search.append(s)
                else:
                    corrections.append(f"search_field: '{s}' geçersiz → ATILDI")

        # ── kume_list: grup key'leri + alan listelerini doğrula ──
        raw_kume = raw.get("kume_list", {})
        corrected_kume: Dict[str, List[str]] = {}
        if isinstance(raw_kume, dict):
            for group, fields in raw_kume.items():
                if group not in valid_groups:
                    corrections.append(f"kume_list: '{group}' geçersiz grup → ATILDI")
                    continue
                if not isinstance(fields, list):
                    continue
                fixed_fields: List[str] = []
                for f in fields:
                    if not isinstance(f, str):
                        continue
                    fixed = _correct_field_name(f)
                    if fixed:
                        if fixed != f:
                            corrections.append(f"kume_list[{group}]: '{f}' → '{fixed}'")
                        if fixed not in fixed_fields:
                            fixed_fields.append(fixed)
                    else:
                        corrections.append(f"kume_list[{group}]: '{f}' geçersiz → ATILDI")
                if fixed_fields:
                    corrected_kume[group] = fixed_fields

        result = cls(
            label_field=corrected_label,
            payload_field=corrected_payload,
            search_field=corrected_search,
            kume_list=corrected_kume,
        )
        return result, corrections

class Customer(BaseModel):
    customerId: str
    firstName: str
    lastName: str
    gender: str
    phone: str
    email: str
    
class PreviousMessage(BaseModel):
    messageId: str
    date: str
    role: str
    content: str
    
class Prompt(BaseModel):
    type: str
    prompt: str

class AISettings(BaseModel):
    endpoint: str
    prompts: List[Prompt]
    
class SocialRequest(BaseModel):
    agent_parametre: str
    companyId: str
    customer: Customer
    chatId: str
    chatType: str
    previousMessages: List[PreviousMessage]
    lastMessage: str
    aiSettings: AISettings

    
# =========================================================
# 1️⃣ STORAGE MODELLER (QDRANT FULL PAYLOAD)
# =========================================================

class Tarih(BaseModel):
    ay: Optional[str] = None
    gun: Optional[int] = None
    yil: Optional[int] = None


class TarihFiyat(BaseModel):
    kalan: Optional[int] = None
    kisibasi: Optional[float] = None
    kur: Optional[str] = None
    tarih: Optional[Tarih] = None


class Konaklama(BaseModel):
    konaklama_tipi: Optional[List[str]] = None
    konum_tipi: Optional[List[str]] = None
    otel_markasi: Optional[List[str]] = None
    otel_yildiz: Optional[List[int]] = None
    pansiyon_tipi: Optional[List[str]] = None
    tam_metin: Optional[str] = None


class TurKategori(BaseModel):
    isim: Optional[str] = None
    puan: Optional[int] = None


class TourPayload(BaseModel):
    id: Optional[int] = None
    geceleme: Optional[str] = None
    gecesayisi: Optional[int] = None
    isim: Optional[str] = None
    kalkissehir: Optional[str] = None
    kesinkalkis: Optional[str] = None
    konaklama: Optional[Konaklama] = None
    metin: Optional[str] = None
    tarihFiyat: Optional[List[TarihFiyat]] = None
    turKategori: Optional[List[TurKategori]] = None
    turkodu: Optional[str] = None
    turtipi: Optional[List[str]] = None
    ulasim: Optional[str] = None
    ulasimtipi: Optional[str] = None
    url: Optional[HttpUrl] = None
    vizesiz: Optional[str] = None
    vizeulke: Optional[str] = None
    ziyaretedilecekyerler: Optional[List[str]] = None

    @field_validator("*", mode="before")
    @classmethod
    def empty_string_to_none(cls, v):
        if v == "" or v == []:
            return None
        return v


# =========================================================
# 2️⃣ VIEW MODELLER (INTENT’E GÖRE RESPONSE)
# =========================================================

# 🔹 tour_list
# extra="allow" → Qdrant'dan dönen ek alanlar (konaklama, ulasim vb.)
# otomatik olarak modele eklenir ve model_dump() ile serialize edilir.
class TourListItem(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)
    id: int
    isim: str

# 🔹 tour_program_detail
class TourProgramDetailView(BaseModel):
    id: int
    gun: int
    baslik: str
    program: str
    cruisevaris: Optional[str] = None


# 🔹 tour_info (full)
class TourInfoView(TourPayload):
    pass


    
class ResponseData(BaseModel):
    companyId: str
    chatId: str
    label: List[str]           # Örn: intent etiketleri (tour_info, customer_info vs.)
    content: str               # agent cevabı
    model: Optional[str] = None
    tokens_used: Optional[int] = None
    
# ---------------- Response Models ----------------

class SocialResponse(BaseModel):
    success: bool
    data: Optional[ResponseData] = None
    errorMessage: Optional[str] = None


# =========================================================
# 3️⃣ FİYAT HESAPLAMA MODELLERİ
# =========================================================

class PriceCalculateRequest(BaseModel):
    """WhatsApp / Postman'den gelen fiyat hesaplama form verisi"""
    tour_id: int
    tur_tarihi: str                             # "2025-09-27"
    oda_sayisi: int                             # >= 1
    yetiskin_sayisi: int                        # >= 1
    cocuk_sayisi: int = 0                       # >= 0
    cocuk_yaslari: List[int] = []               # Her çocuğun yaşı
    chatId: Optional[str] = None
    companyId: Optional[str] = None

class AlternatifTarih(BaseModel):
    """Kontenjan/tarih hatalarında önerilen alternatif"""
    value: str              # "2025-10-11"
    label: str              # "11 Ekim 2025"
    kisibasi: float
    kur: str
    kalan: int

class PriceCalculateResponse(BaseModel):
    """Fiyat hesaplama sonucu — validasyon uyarıları dahil"""
    success: bool
    step: str = "completed"                     # State: tour_id | date | rooms | quota | completed | error
    tour_id: int
    tour_name: str = ""
    tarih: str = ""                             # Seçilen tarih label
    kur: str = ""                               # "USD", "EUR", "TL"
    kisibasi_fiyat: float = 0.0                 # Birim fiyat
    yetiskin_sayisi: int = 0
    yetiskin_toplam: float = 0.0                # yetiskin_sayisi * kisibasi
    cocuk_detay: List[Dict[str, Any]] = []      # [{"yas": 8, "indirim": "%50", "ucret": 1299.5}, ...]
    cocuk_toplam: float = 0.0
    oda_sayisi: int = 0
    oda_guncellendi: bool = False               # Oda sayısı otomatik güncellendiyse True
    oda_uyari: str = ""                         # "Bir odada max 3 kişi. Oda sayısı 2 olarak güncellendi."
    oda_basi_toplam: float = 0.0                # Kişi bazlı toplam (yetişkin + çocuk)
    ekstra_oda_ucreti: float = 0.0              # ekstra_oda * EXTRA_ROOM_COST (2000)
    genel_toplam: float = 0.0                   # oda_basi_toplam + ekstra_oda_ucreti
    kalan_kontenjan: int = 0
    kontenjan_pirlekli: bool = False             # Kontenjan yetersiz uyarısı
    alternatif_tarihler: List[AlternatifTarih] = []  # Kontenjan/tarih hatası → öneriler
    mesaj: str = ""                             # Özet mesaj / uyarı