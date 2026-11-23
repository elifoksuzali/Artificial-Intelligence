from enum import Enum
class ErrorCode(Enum):
    """Sistem hata kodları"""
    # 1xxx - Genel hatalar
    UNKNOWN_ERROR = (1000, "Bilinmeyen bir hata oluştu")
    INVALID_REQUEST = (1001, "Geçersiz istek formatı")
    MISSING_REQUIRED_FIELD = (1002, "Zorunlu alan eksik")
    
    # 2xxx - LLM/Agent hataları
    LLM_CONNECTION_ERROR = (2001, "LLM servisi bağlantı hatası")
    LLM_TIMEOUT = (2002, "LLM servisi zaman aşımı")
    LLM_RESPONSE_PARSE_ERROR = (2003, "LLM yanıtı parse edilemedi")
    AGENT_EXECUTION_ERROR = (2004, "Agent çalıştırma hatası")
    MAX_RETRIES_EXCEEDED = (2005, "Maksimum yeniden deneme sayısı aşıldı")
    
    # 3xxx - Qdrant hataları
    QDRANT_CONNECTION_ERROR = (3001, "Qdrant veritabanı bağlantı hatası")
    QDRANT_QUERY_ERROR = (3002, "Qdrant sorgu hatası")
    QDRANT_NO_RESULTS = (3003, "Qdrant'tan sonuç bulunamadı")
    
    # 4xxx - MNG API hataları
    MNG_API_CONNECTION_ERROR = (4001, "MNG API bağlantı hatası")
    MNG_API_TIMEOUT = (4002, "MNG API zaman aşımı")
    MNG_API_RESPONSE_ERROR = (4003, "MNG API yanıt hatası")
    
    # 5xxx - Rezervasyon/Fiyat hataları
    TOUR_NOT_FOUND = (5001, "Tur bulunamadı")
    INVALID_DATE = (5002, "Geçersiz tarih")
    INSUFFICIENT_QUOTA = (5003, "Yetersiz kontenjan")
    PRICE_CALCULATION_ERROR = (5004, "Fiyat hesaplama hatası")
    
    # 6xxx - Email hataları
    EMAIL_SEND_ERROR = (6001, "E-posta gönderim hatası")
    SMTP_CONNECTION_ERROR = (6002, "SMTP bağlantı hatası")
    
    def __init__(self, code: int, message: str):
        self.code = code
        self.message = message



