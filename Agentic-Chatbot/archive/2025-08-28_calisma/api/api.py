from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Dict
import uvicorn
import mysql.connector
from mysql.connector import Error
import logging
import os
from datetime import datetime

# === Doğru importlar ===

from core.logic import handle_social_message
from models.schema import SocialRequest, SocialResponse
from prompts.handle_db import PromptManager
from agent.agent import initialize_agents
from prompts.global_prompt_manager import global_prompt_manager
from db_key.db import DB_CONFIG

# Logging konfigürasyonu
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

log_filename = f"logs/api_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_filename, encoding='utf-8'),
        logging.StreamHandler()  # Console'a da yazdır
    ]
)

logger = logging.getLogger(__name__)

app = FastAPI(title="Qdrant Search API")

# CORS ayarları
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Startup event handler - artık agent'ları başlatmıyoruz
@app.on_event("startup")
async def startup_event():
    logger.info("🚀 API başlatılıyor...")
    logger.info("📝 Agent'lar ilk mesaj geldiğinde parametreye göre initialize edilecek")
    logger.info(f"📁 Log dosyası: {log_filename}")

def get_SocialChatbotWhiteList():
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
        logger.error(f"Veritabanı hatası: {e}")
        return []

# IP kontrol middleware
@app.middleware("http")
async def verify_ip(request: Request, call_next):
    forwarded_for = request.headers.get("X-Forwarded-For")
    client_ip = forwarded_for.split(',')[0].strip() if forwarded_for else request.client.host
    logger.info(f"IP kontrolü: {client_ip}")
    
    whitelist_ips = get_SocialChatbotWhiteList()
    if client_ip not in whitelist_ips:
        logger.warning(f"IP yasaklı: {client_ip}")
        return JSONResponse(status_code=403, content={"success": False, "errorMessage": f"IP yasaklı: {client_ip}"})
    
    return await call_next(request)

# POST endpoint
@app.post("/social-chatbot/message", response_model=SocialResponse)
async def process_message(request: SocialRequest):
    try:
        logger.info(f"[+] Parametre: {request.parametre}")
        logger.info(f"[+] Mesaj: {request.lastMessage}")
        logger.info(f"[+] Chat ID: {request.chatId}")
        
        # Global prompt manager'ı kullanarak prompt'ları yükle
        global global_prompt_manager
        await global_prompt_manager.load_prompts(request.parametre)
        
        if not global_prompt_manager.prompts:
            logger.error("Prompt yüklenemedi.")
            return SocialResponse(success=False, errorMessage="Prompt yüklenemedi.")
        
        # Agent'ları parametreye göre yeniden initialize et
        await initialize_agents(global_prompt_manager.prompts)
        
        # 2. Ana mesaj işleme
        response = await handle_social_message(request)
        logger.info(f"[+] Yanıt başarılı: {response.success}")
        return response
    except Exception as e:
        logger.error(f"[!] Hata: {str(e)}")
        return SocialResponse(success=False, errorMessage=str(e))

if __name__ == "__main__":
    logger.info("API başlatılıyor...")
    uvicorn.run(app, host="0.0.0.0", port=8001) 