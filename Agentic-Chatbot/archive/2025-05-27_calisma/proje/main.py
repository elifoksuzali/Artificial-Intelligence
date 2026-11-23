# region Kütüphaneler
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
from pydantic import BaseModel
from utils import system_prompt as custom_rules,tour_detail_analyzer_prompt,determine_agent_and_extract_system_prompt,more_request_analyzer_prompt
# Proje içi importlar
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from qdrantDb import QdrantDatabase
from context import get_context, get_qdrant_client

# endregion

# region Değişkenler

# OpenAI ayarı
openai_api_key = os.getenv("OPENAI_API_KEY")
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
# Qdrant bağlantısı için global değişken
qdrant_client = None

# endregion

# region agent_map
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
    "url": agent.ResearchPaperExtractionUrl,
    "Customer": agent.Customer,
    "PreviousMessage": agent.PreviousMessage,
    "Prompt": agent.Prompt,
    "AISettings": agent.AISettings,
    "ResponseData": agent.ResponseData,
    "SocialResponse": agent.SocialResponse,
    "SocialRequest": agent.SocialRequest
}
# endregion

# region ConversationContext
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
        
        # Yeni filtreleme alanları
        self.destination: Optional[str] = None  # Ziyaret edilecek yer
        self.night_count: Optional[int] = None  # Gece sayısı
        self.night_type: Optional[str] = None   # Gecelik tipi
        self.accommodation: Optional[str] = None  # Konaklama tipi
        self.visa_free: Optional[bool] = None    # Vize durumu
        self.guaranteed_departure: Optional[bool] = None  # Kesin kalkış durumu

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
# endregion

# region Fonksiyonlar
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
            
        # Ziyaret edilecek yerler kontrolü
        if context.destination and context.destination.lower() not in payload.get("ziyaretedilecekyerler", "").lower():
            continue
            
        # Gece sayısı kontrolü
        if context.night_count and str(payload.get("gecesayisi", "")) != str(context.night_count):
            continue
            
        # Gecelik kontrolü
        if context.night_type and payload.get("geceleme") != context.night_type:
            continue
            
        # Konaklama kontrolü
        if context.accommodation and context.accommodation.lower() not in payload.get("konaklama", "").lower():
            continue
            
        # Vize durumu kontrolü
        if context.visa_free is not None and bool(payload.get("vizesiz")) != context.visa_free:
            continue
            
        # Kesin kalkış kontrolü
        if context.guaranteed_departure is not None and bool(payload.get("kesinkalkis")) != context.guaranteed_departure:
            continue
            
        filtered.append(result)
        context.shown_result_ids.add(result_id)
    
    return filtered

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
        analysis_result = await analyze_tour_detail_question(question, conv_context.__dict__)
        detail_field = analysis_result.get("intent", {}).get("type")
        
        # Daha fazla tur isteği analizi
        more_request_result = await analyze_more_request(question, {
            "previous_tours": conv_context.last_shown_tours,
            "shown_count": len(conv_context.shown_result_ids),
            "tour_category": conv_context.tour_category
        })
        is_more_request = more_request_result.get("is_more_request", False)
        
        # Router agent ile kategori analizi
        router_result = await determine_agent_and_extract(question, input_data, chat_history)
        category_analysis = router_result.get("category_analysis", {})
        tour_category = category_analysis.get("primary_category")
        
        # Bağlam oluştur
        context = {
            "detail_field": detail_field,
            "is_more_request": is_more_request,
            "tour_category": tour_category,
            "shown_result_ids": list(conv_context.shown_result_ids),
            "last_shown_tours": conv_context.last_shown_tours,
            "analysis_result": analysis_result,
            "more_request_result": more_request_result,
            "category_analysis": category_analysis
        }
        
        # Eğer tur detayı veya tur listesi isteği varsa Qdrant'tan veri al
        if detail_field or not is_more_request:
            results = await get_context(question, qdrant_client)
            context["results"] = results
        
        return context
        
    except Exception as e:
        print(f"❌ Bağlam oluşturma hatası: {str(e)}")
        return {}

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

async def analyze_tour_detail_question(question: str, context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Kullanıcının tur detayı sorusunu analiz eder ve uygun yanıtı döndürür.
    """
    try:
        # GPT modeline gönderilecek prompt'u hazırla
        messages = [
            {"role": "system", "content": tour_detail_analyzer_prompt},
            {"role": "user", "content": f"""
            Kullanıcı sorusu: {question}
            
            Tur detayları:
            {json.dumps(context, ensure_ascii=False, indent=2)}
            """}
        ]

        # GPT modelinden yanıt al
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            response_format={"type": "json_object"}
        )

        # Yanıtı parse et
        result = json.loads(response.choices[0].message.content)
        return result

    except Exception as e:
        print(f"❌ Tur detayı analiz hatası: {str(e)}")
        return {
            "intent": {"type": "genel", "confidence": 0.0},
            "missing_info": [],
            "available_info": {},
            "response": {
                "message": "Üzgünüm, sorunuzu tam olarak anlayamadım. Biraz daha detay verebilir misiniz?",
                "suggestions": ["Konaklama bilgisi almak için 'konaklama nerede?' diye sorabilirsiniz.",
                              "Program detayları için 'tur programı nedir?' diye sorabilirsiniz."]
            }
        }

async def analyze_more_request(question: str, context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Kullanıcının daha fazla tur isteyip istemediğini analiz eder.
    """
    try:
        # GPT modeline gönderilecek prompt'u hazırla
        messages = [
            {"role": "system", "content": more_request_analyzer_prompt},
            {"role": "user", "content": f"""
            Kullanıcı sorusu: {question}
            
            Mevcut bağlam:
            {json.dumps(context, ensure_ascii=False, indent=2)}
            """}
        ]

        # GPT modelinden yanıt al
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            response_format={"type": "json_object"}
        )

        # Yanıtı parse et
        result = json.loads(response.choices[0].message.content)
        return result

    except Exception as e:
        print(f"❌ Daha fazla tur isteği analiz hatası: {str(e)}")
        return {
            "is_more_request": False,
            "request_type": "unknown",
            "confidence": 0.0,
            "context": {},
            "response": {
                "should_show_more": False,
                "reason": "Analiz sırasında bir hata oluştu",
                "suggestions": ["Lütfen sorunuzu tekrar sorar mısınız?"]
            }
        }

# endregion
