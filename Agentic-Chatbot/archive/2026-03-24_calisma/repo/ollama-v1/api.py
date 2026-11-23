from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from response_structure import SocialRequest,SocialResponse
from connection_sql import KeyManager  # <-- Python 3.9 için gerekli
from setting import DB_HOST, DB_USER, DB_PASSWORD, DB_NAME
import mysql.connector
from mysql.connector import Error
from main import handle_request

key_manager = KeyManager()
api_list = key_manager.load_api_key()
control_api_key = api_list.get('SocialChatbotWhiteList', '')

# FastAPI instance
app = FastAPI(title="HARBIAI API")

# CORS ayarları
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
whitelist_ips = []

DB_CONFIG = {
    "host": DB_HOST,
    "user": DB_USER,
    "password": DB_PASSWORD,
    "database": DB_NAME
}


@app.on_event("startup")
async def startup_event():
    global whitelist_ips
    whitelist_ips = await get_SocialChatbotWhiteList(control_api_key)
    print("🚀 API başlatılıyor...")
    print("📝 Agent'lar ilk mesaj geldiğinde parametreye göre initialize edilecek")

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
    forwarded_for = request.headers.get("X-Forwarded-For")
    client_ip = forwarded_for.split(',')[0].strip() if forwarded_for else request.client.host
    print(f"IP kontrolü: {client_ip}")
    whitelist_ips = await get_SocialChatbotWhiteList(control_api_key)

    if client_ip not in whitelist_ips:
        return JSONResponse(status_code=403,content={"success": False, "errorMessage": f"IP yasaklı: {client_ip}"})
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

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
