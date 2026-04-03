from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import json
from datetime import datetime
from pydict.model import ChatRequest
from handle.handle import process_chat_message
from prompt.prompt_db import PromptManager
from analysis.question_analysis import analyze_message

app = FastAPI(title="Social Chatbot API")

@app.post("/social-chatbot/message")
async def receive_chat_data(request: ChatRequest):
    try:
        print(f"🚀 [CHAT-{request.chatId}] Yeni mesaj isteği alındı")
        
        # Request verilerini dict'e çevir
        request_data = request.dict()
        
        # Promptları hemen al
        prompt_manager = PromptManager()
        parametre = request_data.get('parametre')
        prompts = prompt_manager.load_prompts(parametre)
        
        # lastMessage'ı analiz et
        last_message = request_data.get('lastMessage', '')
        analysis_result = analyze_message(last_message)
        
        print(f"✅ [CHAT-{request.chatId}] Mesaj analiz edildi")
        
        # Response döndür
        return {
            "status": "completed",
            "message": f"Chat {request.chatId} için mesaj işleme ve analiz tamamlandı",
            "chat_id": request.chatId,
            "parametre": parametre,
            #"prompts": prompts,
            "prompt_count": len(prompts) if prompts else 0,
            "analysis_result": analysis_result,
            "original_message": last_message,
            "processed_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"❌ [CHAT-{request.chatId}] HATA: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Veri alınırken hata oluştu: {str(e)}")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "message": "API çalışıyor!",
        "timestamp": datetime.now().isoformat(),
        "agent_system": "active"
    }

@app.get("/")
async def root():
    return {
        "message": "Social Chatbot API'ye Hoş Geldiniz!",
        "endpoints": {
            "chat": "/social-chatbot/message",
            "health": "/health",
            "docs": "/docs"
        },
        "agent_system": {
            "status": "active",
            "agents": ["greeting", "tour_list", "tour_detail", "price"],
            "features": ["question_analysis", "agent_routing", "category_detection"]
        }
    }

@app.get("/agents/status")
async def get_agent_status():
    """Agent sisteminin durumunu kontrol eder"""
    return {
        "status": "active",
        "agents": {
            "greeting": "active",
            "tour_list": "active", 
            "tour_detail": "active",
            "price": "active"
        },
        "features": {
            "question_analysis": "active",
            "agent_routing": "active",
            "category_detection": "active"
        },
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    print("🚀 API başlatılıyor...")
    print("📡 Endpoint: http://localhost:8000/social-chatbot/message")
    print("🏥 Health check: http://localhost:8000/health")
    print("📚 API Docs: http://localhost:8000/docs")
    print("🤖 Agent Status: http://localhost:8000/agents/status")
    print("\n🔄 Uvicorn ile başlatmak için:")
    print("uvicorn api:app --host 0.0.0.0 --port 8000 --reload")
    uvicorn.run(app, host="0.0.0.0", port=8000)
