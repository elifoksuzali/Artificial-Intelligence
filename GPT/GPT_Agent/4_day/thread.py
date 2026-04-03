# region Thread
import os
import time
import json
import openai
import re
import requests
import base64
import asyncio
import logging
import orjson  # Daha hızlı JSON işleme
from functools import lru_cache
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Literal, Dict, List, Any, Tuple
from datetime import datetime, timedelta
import aiohttp
from ratelimit import limits, sleep_and_retry
from context import get_context, get_qdrant_client, get_embedding
from qdrantDb import QdrantDatabase
from utils import (
    tour_assistant_system_prompt,
    message_analyzer_system_prompt,
    tour_detail_system_prompt,
    greeting_system_prompt,
    repl_agent_system_prompt,
    query_router_agent_system_prompt,
    follow_up_agent_system_prompt,
    determine_agent_and_extract_system_prompt,
    question_analyzer_agent_system_prompt
)
import uuid

# Logging ayarları
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('agent_logs.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Cache boyutu
CACHE_SIZE = 1000

# JSON işleme için orjson kullanımı
def json_dumps(obj: Any) -> str:
    return orjson.dumps(obj, option=orjson.OPT_NON_STR_KEYS | orjson.OPT_SERIALIZE_NUMPY).decode('utf-8')

def json_loads(s: str) -> Any:
    return orjson.loads(s)

# Cache decorator'ları
@lru_cache(maxsize=CACHE_SIZE)
def cache_tour_info(tour_data: str) -> Dict[str, str]:
    """Tur bilgilerini cache'le"""
    return json_loads(tour_data)

@lru_cache(maxsize=CACHE_SIZE)
def cache_context_results(query: str) -> List[Dict[str, Any]]:
    """Qdrant sonuçlarını cache'le"""
    return get_context(query)

class Timer:
    def __init__(self, name):
        self.name = name
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        self.start_time = time.time()
        logger.info(f"[TIMER] {self.name} başladı")
        return self

    def __exit__(self, *args):
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        logger.info(f"[TIMER] {self.name} tamamlandı - Süre: {duration:.2f} saniye")

class AgentLogger:
    @staticmethod
    def log_agent_action(agent_name: str, action: str, details: Dict = None):
        logger.info(f"[AGENT] {agent_name} - {action}")
        if details:
            logger.info(f"[DETAILS] {json_dumps(details)}")

app = FastAPI()

# CORS ayarları
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from database.db import KeyManager
km = KeyManager()
api_key = km.load_api_key()

openai.api_key = api_key['gpt_api_key']

# Ana tur asistanı
TOUR_ASSISTANT_ID = "TOUR_ASSISTANT_ID"

# Mesaj analiz asistanı
MESSAGE_ANALYZER_ASSISTANT_ID = "MESSAGE_ANALYZER_ASSISTANT_ID"

# Qdrant bağlantısı için global değişken
qdrant_client = None

async def search_tours(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    """Qdrant'ta tur ara"""
    client = get_qdrant_client()
    
    # OpenAI ile query embedding oluştur
    query_vector = get_embedding(query)
    
    # Qdrant'ta ara
    search_result = client.query_points(
        vector=query_vector,
        top_k=limit
    )
    
    return [hit.payload for hit in search_result]

# Thread havuzu ve session yönetimi
thread_pool: Dict[str, dict] = {}
session_threads: Dict[str, str] = {}  # session_id -> thread_id eşleştirmesi
MAX_THREADS = 100
THREAD_EXPIRY = 3600  # 1 saat

# Rate limiting
CALLS = 100
RATE_LIMIT_PERIOD = 60

# Konuşma geçmişi yönetimi
conversation_history: Dict[str, List[Dict[str, str]]] = {}
MAX_HISTORY_LENGTH = 10  # Son 10 mesajı tut

class MessageType(BaseModel):
    type: Literal["greeting", "tour_query", "other"]
    confidence: float

class TourQuery(BaseModel):
    query: str
    session_id: Optional[str] = None

class TourResponse(BaseModel):
    tour_id: Optional[str] = None
    tour_details: Optional[str] = None
    tour_program: Optional[str] = None
    error: Optional[str] = None
    message_type: Optional[str] = None
    session_id: Optional[str] = None

async def analyze_with_gpt(query: str, context: List[Dict[str, Any]]) -> Tuple[str, List[str]]:
    """GPT ile sorguyu analiz et ve hangi alanların çıkarılması gerektiğini belirle"""
    try:
        messages = [
            {"role": "system", "content": query_router_agent_system_prompt},
            {"role": "user", "content": json_dumps({
                "query": query,
                "context": context,
                "available_fields": ["id", "isim", "turkodu", "geceleme", "konaklama", "ulasim", 
                                   "ziyaretedilecekyerler", "vizesiz", "kesinkalkis", "url", 
                                   "turtipi", "ulasimtipi"]
            })}
        ]

        response = await asyncio.to_thread(
            openai.chat.completions.create,
            model="gpt-4o-mini",
            messages=messages
        )

        raw = response.choices[0].message.content
        result = json_loads(raw)
        return result.get("agent", "isim"), result.get("required_fields", [])
    except Exception as e:
        logger.error(f"❌ GPT agent/cevap hatası: {e}")
        # Hata durumunda varsayılan olarak tüm alanları döndür
        return "isim", ["id", "isim", "turkodu", "geceleme", "konaklama", "ulasim", 
                       "ziyaretedilecekyerler", "vizesiz", "kesinkalkis", "url", 
                       "turtipi", "ulasimtipi"]

async def extract_tour_info(text: str, fields: List[str]) -> Dict[str, str]:
    """Belirtilen alanları tur verisinden çıkar"""
    # Cache'den kontrol et
    cached_data = cache_tour_info(text)
    if cached_data:
        return {field: cached_data.get(field, "") for field in fields}
    
    # JSON olarak parse et
    try:
        data = json_loads(text)
        return {field: str(data.get(field, "")) for field in fields}
    except:
        return {field: "" for field in fields}

async def prepare_assistant_message(query: str, tour_info: Dict[str, Any], conversation_history: List[Dict[str, str]]) -> str:
    """Asistan için mesaj hazırla"""
    try:
        # Eğer tour_info boşsa, Qdrant'tan gelen verileri kullan
        if not tour_info and conversation_history:
            # Son Qdrant sonuçlarını bul
            for msg in reversed(conversation_history):
                if msg.get("role") == "assistant" and "tour_info" in msg.get("content", ""):
                    try:
                        content = json_loads(msg["content"])
                        if "tour_info" in content:
                            tour_info = content["tour_info"]
                            break
                    except:
                        continue

        # Tur bilgilerini formatla
        formatted_tours = []
        if isinstance(tour_info, list):
            for tour in tour_info:
                if isinstance(tour, dict):
                    formatted_tours.append({
                        "id": tour.get("id", ""),
                        "isim": tour.get("isim", ""),
                        "turkodu": tour.get("turkodu", ""),
                        "geceleme": tour.get("geceleme", ""),
                        "konaklama": tour.get("konaklama", ""),
                        "ulasim": tour.get("ulasim", ""),
                        "ziyaretedilecekyerler": tour.get("ziyaretedilecekyerler", ""),
                        "vizesiz": tour.get("vizesiz", ""),
                        "kesinkalkis": tour.get("kesinkalkis", ""),
                        "url": tour.get("url", "")
                    })
        elif isinstance(tour_info, dict):
            formatted_tours.append({
                "id": tour_info.get("id", ""),
                "isim": tour_info.get("isim", ""),
                "turkodu": tour_info.get("turkodu", ""),
                "geceleme": tour_info.get("geceleme", ""),
                "konaklama": tour_info.get("konaklama", ""),
                "ulasim": tour_info.get("ulasim", ""),
                "ziyaretedilecekyerler": tour_info.get("ziyaretedilecekyerler", ""),
                "vizesiz": tour_info.get("vizesiz", ""),
                "kesinkalkis": tour_info.get("kesinkalkis", ""),
                "url": tour_info.get("url", "")
            })

        # Yanıt formatını oluştur
        response = {
            "query": query,
            "tour_info": formatted_tours,
            "conversation_history": conversation_history
        }

        return json_dumps(response)
    except Exception as e:
        logger.error(f"❌ Asistan mesaj hazırlama hatası: {e}")
        return json_dumps({
            "query": query,
            "tour_info": [],
            "conversation_history": conversation_history,
            "error": str(e)
        })

def update_conversation_history(session_id: str, message: Dict[str, str]):
    """Konuşma geçmişini güncelle"""
    if session_id not in conversation_history:
        conversation_history[session_id] = []
    
    conversation_history[session_id].append(message)
    
    # Maksimum uzunluğu kontrol et
    if len(conversation_history[session_id]) > MAX_HISTORY_LENGTH:
        conversation_history[session_id] = conversation_history[session_id][-MAX_HISTORY_LENGTH:]

def get_conversation_history(session_id: str) -> List[Dict[str, str]]:
    """Belirli bir oturumun konuşma geçmişini getir"""
    return conversation_history.get(session_id, [])

def format_conversation_history(history: List[Dict[str, str]]) -> str:
    """Konuşma geçmişini formatla"""
    formatted = []
    for msg in history:
        role = "Kullanıcı" if msg["role"] == "user" else "Asistan"
        formatted.append(f"{role}: {msg['content']}")
    return "\n".join(formatted)

def get_thread_from_pool():
    """Thread havuzundan boş bir thread al veya yeni oluştur"""
    current_time = datetime.now()
    
    # Eski threadleri temizle
    for thread_id in list(thread_pool.keys()):
        if (current_time - thread_pool[thread_id]['last_used']) > timedelta(seconds=THREAD_EXPIRY):
            del thread_pool[thread_id]
    
    # Boş thread ara
    for thread_id, thread_data in thread_pool.items():
        if not thread_data['in_use']:
            thread_data['in_use'] = True
            thread_data['last_used'] = current_time
            return thread_id
    
    # Yeni thread oluştur
    if len(thread_pool) < MAX_THREADS:
        new_thread = openai.beta.threads.create()
        thread_pool[new_thread.id] = {
            'in_use': True,
            'last_used': current_time
        }
        return new_thread.id
    
    return None

def release_thread(thread_id: str):
    """Thread'i serbest bırak"""
    if thread_id in thread_pool:
        thread_pool[thread_id]['in_use'] = False
        thread_pool[thread_id]['last_used'] = datetime.now()

def get_or_create_thread(session_id: str) -> str:
    """Session ID için thread al veya yeni oluştur"""
    if session_id in session_threads:
        thread_id = session_threads[session_id]
        if thread_id in thread_pool:
            thread_pool[thread_id]['last_used'] = datetime.now()
            return thread_id
    
    # Yeni thread oluştur
    thread_id = get_thread_from_pool()
    if thread_id:
        session_threads[session_id] = thread_id
        thread_pool[thread_id]['last_used'] = datetime.now()
    return thread_id

async def analyze_message_with_thread(message: str, thread_id: str) -> MessageType:
    try:
        # Mesajı thread'e ekle
        await asyncio.to_thread(
            openai.beta.threads.messages.create,
            thread_id=thread_id,
            role="user",
            content=message
        )
        
        # Mesaj analiz asistanını çalıştır
        run = await asyncio.to_thread(
            openai.beta.threads.runs.create,
            thread_id=thread_id,
            assistant_id=MESSAGE_ANALYZER_ASSISTANT_ID,
            instructions=message_analyzer_system_prompt,
            response_format={"type": "json_object"}
        )
        
        # Run'ın tamamlanmasını bekle
        while True:
            run = await asyncio.to_thread(
                openai.beta.threads.runs.retrieve,
                thread_id=thread_id,
                run_id=run.id
            )
            if run.status == "completed":
                break
            elif run.status in ["failed", "cancelled"]:
                raise HTTPException(status_code=500, detail="Assistant run failed")
            await asyncio.sleep(2)
        
        # Yanıtı al
        messages = await asyncio.to_thread(
            openai.beta.threads.messages.list,
            thread_id=thread_id
        )
        response_content = messages.data[0].content[0].text.value
        
        try:
            result = json_loads(response_content)
            logger.info(f"[MESSAGE_ANALYSIS] Gelen mesaj: {message}")
            logger.info(f"[MESSAGE_ANALYSIS] Belirlenen tip: {result.get('type')}")
            logger.info(f"[MESSAGE_ANALYSIS] Güven skoru: {result.get('confidence')}")
            return MessageType(**result)
        except:
            logger.warning(f"[MESSAGE_ANALYSIS] Mesaj tipi belirlenemedi, varsayılan olarak 'other' kullanılıyor")
            return MessageType(type="other", confidence=0.5)
    except Exception as e:
        logger.error(f"[MESSAGE_ANALYSIS] Hata: {str(e)}")
        return MessageType(type="other", confidence=0.5)

async def get_tour_response_with_thread(message: str, tour_data: List[Dict[str, Any]], thread_id: str) -> str:
    try:
        # 1. Önce mesaj tipini analiz et
        message_type = await analyze_message_with_thread(message, thread_id)
        
        # 2. Mesaj tipine göre işlem yap
        if message_type.type == "greeting":
            # Selamlama ise greeting_system_prompt kullan
            await asyncio.to_thread(
                openai.beta.threads.messages.create,
                thread_id=thread_id,
                role="user",
                content=json_dumps({
                    "query": message,
                    "system_prompt": greeting_system_prompt
                })
            )
        elif message_type.type == "tour_query":
            # Tur sorgusu ise önce query_router ile analiz et
            await asyncio.to_thread(
                openai.beta.threads.messages.create,
                thread_id=thread_id,
                role="user",
                content=json_dumps({
                    "query": message,
                    "tour_data": tour_data,
                    "system_prompt": query_router_agent_system_prompt
                })
            )
            
            # Sonra determine_agent ile hangi bilgilerin çıkarılacağını belirle
            await asyncio.to_thread(
                openai.beta.threads.messages.create,
                thread_id=thread_id,
                role="user",
                content=json_dumps({
                    "query": message,
                    "tour_data": tour_data,
                    "system_prompt": determine_agent_and_extract_system_prompt
                })
            )
            
            # Eğer takip sorusu ise follow_up_agent kullan
            if "başka" in message.lower() or "daha" in message.lower():
                await asyncio.to_thread(
                    openai.beta.threads.messages.create,
                    thread_id=thread_id,
                    role="user",
                    content=json_dumps({
                        "query": message,
                        "tour_data": tour_data,
                        "system_prompt": follow_up_agent_system_prompt
                    })
                )
            else:
                # Normal tur sorgusu ise tour_detail_system_prompt kullan
                await asyncio.to_thread(
                    openai.beta.threads.messages.create,
                    thread_id=thread_id,
                    role="user",
                    content=json_dumps({
                        "query": message,
                        "tour_data": tour_data,
                        "system_prompt": tour_detail_system_prompt
                    })
                )
        else:
            # Diğer sorular için repl_agent kullan
            await asyncio.to_thread(
                openai.beta.threads.messages.create,
                thread_id=thread_id,
                role="user",
                content=json_dumps({
                    "query": message,
                    "system_prompt": repl_agent_system_prompt
                })
            )
        
        # Tur asistanını çalıştır
        run = await asyncio.to_thread(
            openai.beta.threads.runs.create,
            thread_id=thread_id,
            assistant_id=TOUR_ASSISTANT_ID
        )
        
        # Run'ın tamamlanmasını bekle
        max_retries = 30  # Maksimum 30 deneme
        retry_count = 0
        while retry_count < max_retries:
            run = await asyncio.to_thread(
                openai.beta.threads.runs.retrieve,
                thread_id=thread_id,
                run_id=run.id
            )
            
            if run.status == "completed":
                break
            elif run.status in ["failed", "cancelled", "expired"]:
                raise HTTPException(status_code=500, detail=f"Assistant run {run.status}")
            
            retry_count += 1
            await asyncio.sleep(2)  # 2 saniye bekle
        
        if retry_count >= max_retries:
            raise HTTPException(status_code=504, detail="Assistant run timeout")
        
        # Yanıtı al
        messages = await asyncio.to_thread(
            openai.beta.threads.messages.list,
            thread_id=thread_id
        )
        response = messages.data[0].content[0].text.value
        logger.info(f"[TOUR_RESPONSE] Gelen mesaj: {message}")
        logger.info(f"[TOUR_RESPONSE] Yanıt: {response}")
        return response
    except Exception as e:
        logger.error(f"[TOUR_RESPONSE] Hata: {str(e)}")
        return "❌ Üzgünüm, şu anda tur bilgilerini işleyemiyorum. Lütfen daha sonra tekrar deneyin."

async def get_greeting_response_with_thread(message: str, thread_id: str) -> str:
    try:
        # Mesajı thread'e ekle
        await asyncio.to_thread(
            openai.beta.threads.messages.create,
            thread_id=thread_id,
            role="user",
            content=message
        )
        
        # Tur asistanını çalıştır
        run = await asyncio.to_thread(
            openai.beta.threads.runs.create,
            thread_id=thread_id,
            assistant_id=TOUR_ASSISTANT_ID
        )
        
        # Run'ın tamamlanmasını bekle
        while True:
            run = await asyncio.to_thread(
                openai.beta.threads.runs.retrieve,
                thread_id=thread_id,
                run_id=run.id
            )
            if run.status == "completed":
                break
            elif run.status in ["failed", "cancelled"]:
                raise HTTPException(status_code=500, detail="Assistant run failed")
            await asyncio.sleep(2)
        
        # Yanıtı al
        messages = await asyncio.to_thread(
            openai.beta.threads.messages.list,
            thread_id=thread_id
        )
        response = messages.data[0].content[0].text.value
        logger.info(f"[GREETING_RESPONSE] Gelen mesaj: {message}")
        logger.info(f"[GREETING_RESPONSE] Yanıt: {response}")
        return response
    except Exception as e:
        logger.error(f"[GREETING_RESPONSE] Hata: {str(e)}")
        return "Merhaba! Size nasıl yardımcı olabilirim?"

@app.post("/query-tour", response_model=TourResponse)
async def query_tour(tour_query: TourQuery):
    session_start_time = time.time()
    session_id = tour_query.session_id or str(uuid.uuid4())
    
    try:
        # Session için thread al veya oluştur
        thread_id = get_or_create_thread(session_id)
        if not thread_id:
            raise HTTPException(status_code=429, detail="Too many concurrent requests")

        # 1. Mesaj tipini analiz et
        with Timer("Mesaj Tipi Analizi"):
            message_type = await analyze_message_with_thread(tour_query.query, thread_id)
            AgentLogger.log_agent_action("MessageAnalyzer", "Mesaj tipi belirlendi", {
                "type": message_type.type,
                "confidence": message_type.confidence
            })

        # 2. Mesaj tipine göre işlem yap
        if message_type.type == "greeting":
            # Selamlama ise direkt yanıt ver
            with Timer("Selamlama Yanıtı"):
                response = await get_greeting_response_with_thread(tour_query.query, thread_id)
                logger.info(f"[GREETING] Selamlama yanıtı oluşturuldu: {response}")
                
                # Konuşma geçmişini güncelle
                update_conversation_history(session_id, {
                    "role": "user",
                    "content": tour_query.query
                })
                update_conversation_history(session_id, {
                    "role": "assistant",
                    "content": response
                })
                
                return TourResponse(
                    tour_details=response,
                    message_type="greeting",
                    session_id=session_id
                )
        
        elif message_type.type == "tour_query":
            # Tur sorgusu ise Qdrant'tan veri al
            with Timer("Qdrant Sorgusu"):
                context_results = await search_tours(tour_query.query)
                AgentLogger.log_agent_action("QdrantAgent", "Bağlam alındı", {
                    "result_count": len(context_results)
                })

                if not context_results:
                    return TourResponse(
                        tour_details="Üzgünüm, aradığınız kriterlere uygun tur bulamadım. Lütfen farklı bir arama yapın.",
                        message_type="tour_query",
                        session_id=session_id
                    )

            # Tur yanıtını oluştur
            with Timer("Tur Yanıtı"):
                response = await get_tour_response_with_thread(tour_query.query, context_results, thread_id)
                AgentLogger.log_agent_action("TourAssistant", "Yanıt üretildi", {
                    "response_length": len(response)
                })

            # Konuşma geçmişini güncelle
            update_conversation_history(session_id, {
                "role": "user",
                "content": tour_query.query
            })
            update_conversation_history(session_id, {
                "role": "assistant",
                "content": response
            })

            return TourResponse(
                tour_details=response,
                message_type="tour_query",
                session_id=session_id
            )
        else:
            # Diğer sorular için genel yanıt
            return TourResponse(
                tour_details="Üzgünüm, bu konuda size yardımcı olamıyorum. Lütfen tur ile ilgili bir soru sorun.",
                message_type="other",
                session_id=session_id
            )

    except Exception as e:
        logger.error(f"❌ Hata: {str(e)}")
        return TourResponse(
            error=str(e),
            message_type="error",
            session_id=session_id
        )
    finally:
        session_duration = time.time() - session_start_time
        logger.info(f"[COMPLETE] İşlem tamamlandı - Toplam süre: {session_duration:.2f} saniye")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
