import logging, asyncio
from constants import ErrorCode
from functools import wraps
from typing import Callable, Any
import httpx
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from response_structure import SocialRequest, SocialResponse

# =========================================================
# region ERROR CODES & MESSAGES
# =========================================================


class APIError(Exception):
    """Özel API hatası sınıfı"""
    def __init__(self, error_code: ErrorCode, detail: str = None):
        self.error_code = error_code
        self.code = error_code.code
        self.message = error_code.message
        self.detail = detail or ""
        super().__init__(f"[{self.code}] {self.message}: {self.detail}")
    
    def to_dict(self) -> dict:
        return {
            "error_code": self.code,
            "error_message": self.message,
            "detail": self.detail
        }


# =========================================================
# region RETRY CONFIGURATION
# =========================================================

# Retry ayarları
MAX_RETRIES = 3
RETRY_DELAY_BASE = 1.0  # saniye
RETRY_DELAY_MAX = 10.0  # maksimum bekleme süresi


async def retry_async(
    func: Callable,
    *args,
    max_retries: int = MAX_RETRIES,
    delay_base: float = RETRY_DELAY_BASE,
    delay_max: float = RETRY_DELAY_MAX,
    error_code: ErrorCode = ErrorCode.UNKNOWN_ERROR,
    **kwargs
) -> Any:
    """
    Async fonksiyonları retry mekanizması ile çalıştırır.
    
    Args:
        func: Çalıştırılacak async fonksiyon
        max_retries: Maksimum deneme sayısı
        delay_base: Temel bekleme süresi (exponential backoff için)
        delay_max: Maksimum bekleme süresi
        error_code: Başarısız olursa fırlatılacak hata kodu
    
    Returns:
        Fonksiyonun sonucu
    
    Raises:
        APIError: Tüm denemeler başarısız olursa
    """
    last_exception = None
    
    for attempt in range(1, max_retries + 1):
        try:
            result = await func(*args, **kwargs)
            if attempt > 1:
                logger.info(f"✅ Retry başarılı (deneme {attempt}/{max_retries})")
            return result
            
        except asyncio.TimeoutError as e:
            last_exception = e
            logger.warning(
                f"⏱️ Timeout hatası (deneme {attempt}/{max_retries}): {str(e)}"
            )
        except httpx.ConnectError as e:
            last_exception = e
            logger.warning(
                f"🔌 Bağlantı hatası (deneme {attempt}/{max_retries}): {str(e)}"
            )
        except httpx.HTTPStatusError as e:
            last_exception = e
            logger.warning(
                f"📡 HTTP hatası (deneme {attempt}/{max_retries}): {e.response.status_code}"
            )
        except Exception as e:
            last_exception = e
            logger.warning(
                f"❌ Hata (deneme {attempt}/{max_retries}): {type(e).__name__}: {str(e)}"
            )
        
        if attempt < max_retries:
            # Exponential backoff with jitter
            delay = min(delay_base * (2 ** (attempt - 1)), delay_max)
            delay += asyncio.get_running_loop().time() % 1  # Jitter
            logger.info(f"⏳ {delay:.1f}s sonra yeniden deneniyor...")
            await asyncio.sleep(delay)
    
    # Tüm denemeler başarısız
    logger.error(
        f"💥 Tüm denemeler başarısız ({max_retries} deneme): {type(last_exception).__name__}"
    )
    raise APIError(
        ErrorCode.MAX_RETRIES_EXCEEDED,
        f"{error_code.message} - {type(last_exception).__name__}: {str(last_exception)}"
    )


def with_retry(
    max_retries: int = MAX_RETRIES,
    error_code: ErrorCode = ErrorCode.UNKNOWN_ERROR
):
    """
    Async fonksiyonlara retry dekoratörü ekler.
    
    Usage:
        @with_retry(max_retries=3, error_code=ErrorCode.LLM_CONNECTION_ERROR)
        async def call_llm(...):
            ...
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await retry_async(
                func, *args,
                max_retries=max_retries,
                error_code=error_code,
                **kwargs
            )
        return wrapper
    return decorator


# =========================================================
# region CONCURRENCY & THRESHOLDS
# =========================================================
# vLLM max 7 eşzamanlı konuşma destekler — Semaphore ile kontrol
MAX_CONCURRENT = 7
_request_semaphore = asyncio.Semaphore(MAX_CONCURRENT)



async def handle_request(request: SocialRequest) -> SocialResponse:
    """
    Ana giriş noktası — Semaphore ile eşzamanlılık kontrolü sağlar.
    chatId bazında izolasyon: her istek bağımsız işlenir.
    """
    from main import _process_request  # lazy import — circular import'u önler
    async with _request_semaphore:
        return await _process_request(request)



