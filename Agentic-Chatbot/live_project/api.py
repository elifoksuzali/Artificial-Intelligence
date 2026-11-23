from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from response_structure import SocialRequest, SocialResponse, PriceCalculateRequest, PriceCalculateResponse
from connection_sql import KeyManager
from setting import DB_HOST, DB_USER, DB_PASSWORD, DB_NAME
import mysql.connector
from mysql.connector import Error
from error_handler import handle_request
from tool_calls import calculate_price

key_manager = KeyManager()
api_list = key_manager.load_api_key()
control_api_key = api_list.get('SocialChatbotWhiteList', '')

whitelist_ips = []

DB_CONFIG = {
    "host": DB_HOST,
    "user": DB_USER,
    "password": DB_PASSWORD,
    "database": DB_NAME
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global whitelist_ips
    whitelist_ips = await get_SocialChatbotWhiteList(control_api_key)
    print("🚀 API başlatılıyor...")
    print("📝 Agent'lar ilk mesaj geldiğinde parametreye göre initialize edilecek")
    yield

# FastAPI instance
app = FastAPI(title="HARBIAI API", lifespan=lifespan)

# CORS ayarları
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def get_SocialChatbotWhiteList(api_key=None):
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        if conn.is_connected():
            cursor = conn.cursor()
            cursor.execute("SELECT deger FROM ayarlar WHERE anahtar = 'SocialChatbotWhiteList'")
            result = cursor.fetchone()
            cursor.close()
            conn.close()
            if result and result[0]:
                return [ip.strip() for ip in result[0].split(',')]
            return []
    except Error as e:
        print(f"Veritabanı hatası: {e}")
        return []

# IP kontrol middleware
@app.middleware("http")
async def verify_ip(request: Request, call_next):
    # CORS preflight (OPTIONS) isteklerini IP kontrolünden muaf tut
    if request.method == "OPTIONS":
        return await call_next(request)

    forwarded_for = request.headers.get("X-Forwarded-For")
    client_ip = forwarded_for.split(',')[0].strip() if forwarded_for else request.client.host
    print(f"IP kontrolü: {client_ip}")
    whitelist_ips = await get_SocialChatbotWhiteList(control_api_key)

    if client_ip not in whitelist_ips:
        return JSONResponse(status_code=403, content={"success": False, "errorMessage": f"IP yasaklı: {client_ip}"})
    return await call_next(request)


# POST endpoint
@app.post("/social-chatbot/message",response_model=SocialResponse)
async def process_message(request: SocialRequest):
    try:
        print(f"[+] Parametre: {request.agent_parametre}")
        return await handle_request(request)

    except Exception as e:
        print(f"[!] Hata: {str(e)}")
        return SocialResponse(success=False, errorMessage=str(e))

# =========================================================
# FİYAT HESAPLAMA ENDPOINT
# WhatsApp frontend'ten gelen form verisini alır, hesaplar, sonuç döndürür.
# =========================================================
@app.post("/social-chatbot/calculate-price", response_model=PriceCalculateResponse)
async def calculate_price_endpoint(request: PriceCalculateRequest):
    """
    Fiyat hesaplama endpoint'i — STATE MACHINE tabanlı validasyon zinciri.
    
    Adımlar:
      1) tour_id  → Qdrant'tan tur var mı?
      2) date     → Tarih müsait mi? Yoksa → alternatif tarihler
      3) rooms    → Oda kapasitesi max 3 kişi. Aşarsa otomatik günceller.
      4) quota    → Kontenjan yeterli mi? Yoksa → alternatif tarihler
      5) pricing  → Hesapla ve sonucu döndür
    
    Postman test body:
    {
        "tour_id": 246,
        "tur_tarihi": "2025-09-27",
        "oda_sayisi": 1,
        "yetiskin_sayisi": 2,
        "cocuk_sayisi": 1,
        "cocuk_yaslari": [8]
    }
    """
    try:
        # ── Ön validasyon: yetişkin >= 1 ──
        if request.yetiskin_sayisi < 1:
            return PriceCalculateResponse(
                success=False, step="error", tour_id=request.tour_id,
                mesaj="Yetişkin sayısı en az 1 olmalıdır.",
            )

        # ── Ön validasyon: oda >= 1 ──
        if request.oda_sayisi < 1:
            return PriceCalculateResponse(
                success=False, step="error", tour_id=request.tour_id,
                mesaj="Oda sayısı en az 1 olmalıdır.",
            )

        # ── Ön validasyon: çocuk sayısı ↔ yaş listesi ──
        if len(request.cocuk_yaslari) != request.cocuk_sayisi:
            return PriceCalculateResponse(
                success=False, step="error", tour_id=request.tour_id,
                mesaj=f"Çocuk sayısı ({request.cocuk_sayisi}) ile çocuk yaşları listesi "
                      f"({len(request.cocuk_yaslari)} adet) uyuşmuyor. "
                      f"Her çocuk için yaş bilgisi girilmelidir.",
            )

        # ── Ana hesaplama (state machine içeride) ──
        result = await calculate_price(request)
        return result

    except Exception as e:
        print(f"[!] Fiyat hesaplama hatası: {str(e)}")
        return PriceCalculateResponse(
            success=False, step="error", tour_id=request.tour_id,
            mesaj=f"Hesaplama hatası: {str(e)}",
        )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8003)
