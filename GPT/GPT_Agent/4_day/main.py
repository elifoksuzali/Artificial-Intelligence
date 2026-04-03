# region Main
import os
import sys
import json
import asyncio
from typing import Dict, Any, List, Optional, Union, cast
import openai
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam, ChatCompletionAssistantMessageParam
from datetime import datetime
import re
import agent as agent
from utils import system_prompt as custom_rules,determine_agent_and_extract_system_prompt,repl_agent_system_prompt,query_router_agent_system_prompt,follow_up_agent_system_prompt,question_analyzer_agent_system_prompt,response_validator_agent_system_prompt
from pydantic import BaseModel

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
    type: str
    content: str
    
class Prompt(BaseModel):
    type: str
    prompt: str
    
class AISettings(BaseModel):
    endpoint: str
    prompts: List[Prompt]       
    
class SocialRequest(BaseModel):
    companyId: str
    customer: Customer
    chatId: str
    chatType: str
    previousMessages: List[PreviousMessage]
    lastMessage: str
    aiSettings: AISettings 
    
class ResponseData(BaseModel):
    companyId: str
    chatId: str
    type: str
    content: str
    
class SocialResponse(BaseModel):
    success: bool
    data: Optional[ResponseData] = None
    errorMessage: Optional[str] = None

class ConversationContext:
    def __init__(self):
        self.selected_tour: Optional[Dict[str, Any]] = None  # Seçilen tur
        self.tour_list: List[Dict[str, Any]] = []        # Mevcut tur listesi
        self.query_type: Optional[str] = None     # Tur tipi (yurt içi/dışı)
        self.tour_category: Optional[str] = None  # Tur kategorisi
        self.transport_type: Optional[str] = None # Ulaşım tipi
        self.last_query: Optional[str] = None     # Son soru
        self.is_more_requested: bool = False  # Daha fazla tur istenip istenmediği
        self.shown_result_ids: set = set()  # Gösterilen sonuçların ID'leri
        self.temp_tours: List[Dict[str, Any]] = []  # Geçici tur listesi
        self.last_shown_tours: List[Dict[str, Any]] = []  # Son gösterilen turlar

    def add_tour(self, tour: Dict[str, Any]) -> None:
        """Turu geçici listeye ekle"""
        tour_id = str(tour.get("id", ""))
        if tour_id and tour_id not in self.shown_result_ids:
            self.temp_tours.append(tour)
            self.shown_result_ids.add(tour_id)

    def get_tour_by_id(self, tour_id: str) -> Optional[Dict[str, Any]]:
        """ID'ye göre tur bilgisini getir"""
        # Önce geçici listede ara
        for tour in self.temp_tours:
            if str(tour.get("id", "")) == tour_id:
                return tour
        # Sonra input_data'da ara
        for tour in input_data:
            if str(tour.get("id", "")) == tour_id:
                return tour
        return None

    def get_tour_by_number(self, number: int) -> Optional[Dict[str, Any]]:
        """Sıra numarasına göre tur bilgisini getir"""
        if 0 < number <= len(self.last_shown_tours):
            return self.last_shown_tours[number-1]
        return None

    def clear_temp_tours(self) -> None:
        """Geçici tur listesini temizle"""
        self.temp_tours = []

def extract_tour_detail_question(question: str) -> Optional[str]:
    """Tur detayı sorulup sorulmadığını kontrol et"""
    detail_keywords = {
        "konaklama": "konaklama",
        "id": "id",
        "turkodu": "turkodu",
        "geceleme": "geceleme",
        "ziyaret": "ziyaretedilecekyerler",
        "ulaşım": "ulasim",
        "vize": "vizesiz",
        "kalkış": "kesinkalkis"
    }
    
    for keyword, field in detail_keywords.items():
        if keyword in question.lower():
            return field
    return None

def check_more_request(question: str) -> bool:
    """Daha fazla tur isteği mi kontrol et"""
    more_keywords = [
        "daha fazla", "başka tur", "farklı tur",
        "başka örnek", "diğer turlar", "evet"
    ]
    return any(keyword in question.lower() for keyword in more_keywords)

def extract_tour_category(question: str) -> Optional[str]:
    """Sorgudan tur kategorisini çıkar"""
    categories = {
        "yurt içi": "yurt içi",
        "yurt dışı": "yurt dışı",
        "otobüs": "otobüs",
        "uçak": "uçak",
        "gemi": "gemi"
    }
    
    for keyword, category in categories.items():
        if keyword in question.lower():
            return category
    return None

def get_tour_detail(tour: Dict[str, Any], detail_field: str) -> str:
    """Tur detayını getir"""
    if not tour:
        return "Tur bulunamadı."
    
    payload = tour.get("payload", {})
    return str(payload.get(detail_field, "Bu bilgi mevcut değil."))

def filter_tours_by_context(results: List[Dict], context: ConversationContext) -> List[Dict]:
    """Sonuçları bağlama göre filtrele"""
    filtered = []
    for result in results:
        payload = result.get("payload", {})
        result_id = str(result.get("id", ""))
        
        # ID kontrolü
        if result_id in context.shown_result_ids:
            continue
            
        # Tur tipi kontrolü
        if context.query_type and payload.get("turtipi") != context.query_type:
            continue
            
        # Kategori kontrolü
        if context.tour_category and payload.get("turKategori") != context.tour_category:
            continue
            
        # Ulaşım tipi kontrolü
        if context.transport_type and payload.get("ulasimtipi") != context.transport_type:
            continue
            
        filtered.append(result)
        context.shown_result_ids.add(result_id)
    
    return filtered

# Proje içi importlar
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from qdrantDb import QdrantDatabase
from context import get_context, get_qdrant_client
from database.db import KeyManager

km = KeyManager()
api_key = km.load_api_key()
openai_api_key = api_key['gpt_api_key']

# OpenAI istemcisi
openai_api_key = openai_api_key
os.environ["OPENAI_API_KEY"] = openai_api_key
client = OpenAI(api_key=openai_api_key)

# Global değişkenler
input_data: List[Dict[str, Any]] = []  # Qdrant'tan gelen veriler burada saklanacak
chat_history: List[Dict[str, str]] = []  # Sohbet geçmişi
last_query: Dict[str, Any] = {}  # Son yapılan sorguyu saklamak için
current_page: int = 0  # Mevcut sayfa numarası
PAGE_SIZE: int = 10  # Her sayfada gösterilecek tur sayısı
shown_result_ids: set = set()  # Daha önce gösterilen sonuçların ID'lerini tutacak set
last_query_context: Dict[str, Any] = {  # Son sorgu bağlamını tutacak
    "query_type": None,  # "yurt içi", "yurt dışı" veya None
    "shown_ids": [],  # Son sorguda gösterilen ID'ler
    "total_results": 0,  # Toplam bulunan sonuç sayısı
    "current_offset": 0  # Mevcut offset
}
agent_map = {
    "id": agent.ResearchPaperExtractionID,
    "isim": agent.ResearchPaperExtractionName,
    "turkodu": agent.ResearchPaperExtractionTurKodu,
    "geceleme": agent.ResearchPaperExtractionGeceSayisi,
    "konaklama": agent.ResearchPaperExtractionKonaklama,
    "ulasim": agent.ResearchPaperExtractionUlasim,
    "ziyaretedilecekyerler": agent.ResearchPaperExtractionZiyaretEdilecekYerler,
    "vizesiz": agent.ResearchPaperExtractionVizeDurumu,
    "kesinkalkis": agent.ResearchPaperExtractionKesinkalkis,
    "url": agent.ResearchPaperExtractionUrl
}

# Qdrant bağlantısı için global değişken
qdrant_client = None

class SimpleAgent:
    def __init__(self, name: str, model: str, instructions: str):
        self.name = name
        self.model = model
        self.instructions = instructions

    async def run(self, messages: List[Dict[str, str]], context: Optional[Dict[str, Any]] = None, input_data: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        try:
            # Context'i mesajlara ekle
            if context:
                # Set'leri list'e dönüştür
                if isinstance(context, dict):
                    for key, value in context.items():
                        if isinstance(value, set):
                            context[key] = list(value)
                context_message = f"Context: {json.dumps(context, ensure_ascii=False)}"
                messages.append({"role": "system", "content": context_message})
            
            # input_data'yı mesajlara ekle
            if input_data:
                input_data_message = f"input_data: {json.dumps(input_data, ensure_ascii=False)}"
                messages.append({"role": "system", "content": input_data_message})
            
            # Agent talimatlarını ekle
            messages.append({
                "role": "system", 
                "content": f"{self.instructions}\n\nLütfen cevabınızı JSON formatında verin. Örnek format: {{\"answer\": \"cevap\", \"source\": \"kaynak\", \"confidence\": 0.95}}"
            })
            
            # OpenAI'ya istek at
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,  # type: ignore
                response_format={"type": "json_object"}
            )
            
            # JSON cevabını parse et
            content = response.choices[0].message.content
            if not content:
                return {"answer": "Cevap alınamadı", "source": "gpt", "confidence": 0.0}
            
            try:
                result = json.loads(content)
                # Eğer answer anahtarı yoksa, tüm içeriği answer olarak kullan
                if "answer" not in result:
                    result = {"answer": content, "source": "gpt", "confidence": 0.8}
                return result
            except json.JSONDecodeError:
                return {"answer": content, "source": "gpt", "confidence": 0.8}
                
        except Exception as e:
            print(f"Agent hatası: {str(e)}")
            return {"answer": f"Bir hata oluştu: {str(e)}", "source": "error", "confidence": 0.0}

def is_about_listed_tour(question: str, chat_history: List[Dict[str, str]]) -> Optional[Dict[str, Any]]:
    """
    Soru, daha önce listelenmiş bir tur hakkında mı kontrol eder.
    Örnek: "3. turun id'si nedir?" -> True
    """
    # Son asistan mesajını bul (tur listesi olmalı)
    last_assistant_msg = None
    for msg in reversed(chat_history):
        if msg["role"] == "assistant":
            last_assistant_msg = msg["content"]
            break
    
    if not last_assistant_msg:
        return None
    
    # Tur numarasını bul (örn: "3. tur")
    tur_num_match = re.search(r'(\d+)\.\s*tur', question.lower())
    if not tur_num_match:
        return None
    
    tur_num = int(tur_num_match.group(1))
    if 0 < tur_num <= len(input_data):
        return input_data[tur_num-1]
    
    return None

async def get_context_for_question(question: str, conv_context: ConversationContext) -> Dict[str, Any]:
    """Soru için bağlam oluştur"""
    try:
        # Qdrant bağlantısını sadece gerektiğinde aç
        global qdrant_client
        if qdrant_client is None:
            qdrant_client = get_qdrant_client()
        
        # Soruyu analiz et
        detail_field = extract_tour_detail_question(question)
        is_more_request = check_more_request(question)
        tour_category = extract_tour_category(question)
        
        # Bağlam oluştur
        context = {
            "detail_field": detail_field,
            "is_more_request": is_more_request,
            "tour_category": tour_category,
            "shown_result_ids": list(conv_context.shown_result_ids),
            "last_shown_tours": conv_context.last_shown_tours
        }
        
        # Eğer tur detayı veya tur listesi isteği varsa Qdrant'tan veri al
        if detail_field or not is_more_request:
            results = await get_context(question, qdrant_client)
            context["results"] = results
        
        return context
        
    except Exception as e:
        print(f"❌ Bağlam oluşturma hatası: {str(e)}")
        return {}

# Agent'ları oluştur
question_analyzer_agent = SimpleAgent(
    name="Question Analyzer",
    model="gpt-4.1",
    instructions=question_analyzer_agent_system_prompt
)

# Reply agent'ı oluştur
reply_agent = SimpleAgent(
    name="Reply Agent",
    model="gpt-4.1",
    instructions=repl_agent_system_prompt
)

query_router_agent = SimpleAgent(
    name="Query Router",
    model="gpt-4.1",
    instructions=query_router_agent_system_prompt
)

follow_up_agent = SimpleAgent(
    name="Follow Up Agent",
    model="gpt-4.1",
    instructions=follow_up_agent_system_prompt
)

# Response validator agent'ı oluştur
response_validator_agent = SimpleAgent(
    name="Response Validator",
    model="gpt-4.1",
    instructions=response_validator_agent_system_prompt
)

async def determine_agent_and_extract(user_query: str, input_data: list, chat_history: Optional[List[Dict[str, str]]] = None) -> tuple[str, List[Dict[str, Any]]]:
    system_message = {
        "role": "system",
        "content": f"""
        {determine_agent_and_extract_system_prompt}
        {custom_rules}

        Tour context:
        {json.dumps(input_data, ensure_ascii=False)}

        Previous conversation:
        {json.dumps(chat_history, ensure_ascii=False) if chat_history else "[]"}
        """
    }

    try:
        messages = chat_history.copy() if chat_history else []
        messages.insert(0, system_message)  # system mesajı en başa eklenir
        messages.append({"role": "user", "content": user_query})

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,  # type: ignore
            response_format={"type": "json_object"}  # type: ignore
        )

        raw = response.choices[0].message.content
        print(f"\n🌟 GPT Raw Output:\n{raw}")

        data = clean_json(raw)
        if not data:
            return "isim", []

        agent_name = data.get("agent", "isim")
        items = data.get("items", [])

        if agent_name not in agent_map:
            print("⚠️ Bilinmeyen agent adı, varsayılan 'isim' kullanılıyor.")
            agent_name = "isim"

        return agent_name, items

    except Exception as e:
        print(f"❌ GPT agent/cevap hatası: {e}")
        return "isim", []

def clean_json(response_content: Optional[str]) -> Optional[Dict[str, Any]]:
    if not response_content:
        return None
    try:
        response_content = re.sub(r"```json\s*|\s*```", "", response_content).strip()
        return json.loads(response_content)
    except json.JSONDecodeError:
        return None

def safe_parse_json(model_class, json_data):
    try:
        return model_class.model_validate(json_data)
    except agent.ValidationError as e:
        return f"Veri eksik veya hatalı: {e}"

async def get_more_tours(conv_context: ConversationContext) -> Dict[str, Any]:
    """Mevcut bağlama göre daha fazla tur getir"""
    print("\n🌟 Sizin için yeni özel seçimler hazırlıyorum...")
    
    # input_data'daki mevcut ID'leri topla
    existing_ids = {str(tour.get("id", "")) for tour in input_data}
    print(f"\n📊 Koleksiyonumuzda {len(existing_ids)} özel seçim bulunuyor")
    
    # Qdrant'tan daha fazla sonuç al (100 kayıt)
    if conv_context.last_query:
        results = await get_context(
            question=conv_context.last_query,
            q_client=qdrant_client,
            num_results=25,
            offset=len(conv_context.tour_list)
        )
    else:
        results = []
    
    # Sadece yeni turları filtrele
    new_results = []
    for result in results:
        result_id = str(result.get("id", ""))
        # Eğer bu tur input_data'da yoksa ekle
        if result_id and result_id not in existing_ids:
            new_results.append(result)
            existing_ids.add(result_id)  # ID'yi ekle ki tekrar gelmesin
    
    # Yeni sonuçları input_data'ya ekle
    if new_results:
        input_data.extend(new_results)
        print(f"\n✨ {len(new_results)} yeni özel seçim tur koleksiyonumuza eklendi")
        
        # Sadece ilk 10 yeni turu formatla
        formatted_tours = []
        for i, tour in enumerate(new_results[:10], 1):
            payload = tour.get("payload", {})
            tour_name = payload.get("isim", "İsimsiz Tur")
            formatted_tours.append(f"{i}. {tour_name}")
        
        return {
            "source": "qdrant",
            "reasoning": "Daha fazla tur isteği üzerine yeni sonuçlar getirildi",
            "content": "🌟 Sizin için yeni özel seçimlerimiz:\n" + "\n".join(formatted_tours) + "\n\n💫 Size nasıl yardımcı olabilirim? 😊",
            "has_more": len(new_results) > 10,
            "irrelevant_or": False
        }
    else:
        return {
            "source": "none",
            "reasoning": "Daha fazla tur bulunamadı",
            "content": "Üzgünüm, bu kategoride başka özel seçim turumuz bulunmuyor. Size farklı bir kategori önerebilir miyim? 🌟",
            "irrelevant_or": True
        }
