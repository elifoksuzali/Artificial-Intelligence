import os
# api.py

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Dict
import uvicorn
import mysql.connector
from mysql.connector import Error

# === Doğru importlar ===

from core.logic import handle_social_message
from models.schema import SocialRequest, SocialResponse
from prompts.handle_db import PromptManager
from agent.agent import initialize_agents


app = FastAPI(title="Social Chatbot API")

# CORS ayarları
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Startup event handler
@app.on_event("startup")
async def startup_event():
    print("🚀 API başlatılıyor...")
    try:
        await initialize_agents()
        print("✅ Agent'lar başarıyla initialize edildi")
    except Exception as e:
        print(f"❌ Agent initialization hatası: {e}")

# Veritabanı IP kontrolü
DB_CONFIG = {
    'host': os.getenv("DB_HOST"),
    'user': os.getenv("DB_USER"),
    'password': os.getenv("DB_PASSWORD"),
    'database': os.getenv("DB_NAME")
}

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
        print(f"Veritabanı hatası: {e}")
        return []

# IP kontrol middleware
@app.middleware("http")
async def verify_ip(request: Request, call_next):
    forwarded_for = request.headers.get("X-Forwarded-For")
    client_ip = forwarded_for.split(',')[0].strip() if forwarded_for else request.client.host
    print(f"IP kontrolü: {client_ip}")
    
    whitelist_ips = get_SocialChatbotWhiteList()
    if client_ip not in whitelist_ips:
        return JSONResponse(status_code=403, content={"success": False, "errorMessage": f"IP yasaklı: {client_ip}"})
    
    return await call_next(request)

# POST endpoint
@app.post("/social-chatbot/message", response_model=SocialResponse)
async def process_message(request: SocialRequest):
    try:
        print(f"[+] Parametre: {request.parametre}")
        prompt_manager = PromptManager()
        # 1. Parametreye özel işlemler
        prompt = await prompt_manager.load_prompts(request.parametre)
        
        if not prompt:
            return SocialResponse(success=False, errorMessage="Prompt yüklenemedi.")
        
        # Agent'ları parametreye göre yeniden initialize et
        await initialize_agents(prompt)
        
        # 2. Ana mesaj işleme
        response = await handle_social_message(request)
        return response
    
    except Exception as e:
        print(f"[!] Hata: {str(e)}")
        return SocialResponse(success=False, errorMessage=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
