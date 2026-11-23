from typing import Union
import json
from typing import Dict, Any, Optional, List
from core.core_context import ConversationContext
from value.value import input_data, agent_map
from agent.agent import agent_manager


# Filter search import ve kullan
from search.filter_search import create_filter_only_search, search_structured
            
# Global prompt manager'a erişim için import
from prompts.global_prompt_manager import global_prompt_manager
from db_key import KeyManager
key_manager = KeyManager()
api_keys = key_manager.load_api_key()
openai_api_key = api_keys.get('gpt_api_key', '')

from openai import OpenAI

client = OpenAI(api_key=openai_api_key)
from pydantic import ValidationError
import re


def get_tour_detail(tour: Dict[str, Any], detail_field: str) -> str:
    """Tur detayını getir"""
    # Global prompt manager'dan prompt'ları kontrol et
    if not global_prompt_manager.prompts:
        raise Exception("Prompt'lar henüz yüklenmemiş! Önce parametre ile prompt yüklenmelidir.")

    if not tour:
        return "Tur bulunamadı."

    payload = tour.get("payload", {})
    return str(payload.get(detail_field, "Bu bilgi mevcut değil."))


def filter_tours_by_context(results: List[Dict], context: ConversationContext) -> List[Dict]:
    """Sonuçları bağlama göre filtrele"""
    if not global_prompt_manager.prompts:
        raise Exception("Prompt'lar henüz yüklenmemiş! Önce parametre ile prompt yüklenmelidir.")

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

        # Konaklama tipi kontrolü
        if context.accommodation_type and payload.get("konaklama") != context.accommodation_type:
            continue

        # Gece sayısı kontrolü
        if context.night_count and payload.get("gece") != context.night_count:
            continue

        filtered.append(result)
        context.shown_result_ids.add(result_id)

    return filtered


# Qdrant client fonksiyonu ekle
def get_qdrant_client():
    """Qdrant client'ı döndür"""
    from qdrant.qdrantDb import QdrantDatabase
    qdrant_db = QdrantDatabase("mng")
    qdrant_db.connect()
    return qdrant_db.get_client()


# Bu fonksiyon artık kullanılmıyor, logic.py'de doğrudan filter_search ve multivector_search kullanılıyor
async def get_context(question: str, q_client=None, num_results: int = 1, tour_analysis_data: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    """Bu fonksiyon artık kullanılmıyor, logic.py'de doğrudan filter_search ve multivector_search kullanılıyor"""
    print("⚠️ Bu fonksiyon artık kullanılmıyor, logic.py'de doğrudan filter_search ve multivector_search kullanılıyor")
    return []

def clean_json(response_content: Optional[str]) -> Optional[Dict[str, Any]]:
    if not response_content:
        return None
    try:
        response_content = re.sub(r"```json\s*|\s*```", "", response_content).strip()
        return json.loads(response_content)
    except json.JSONDecodeError:
        return None


async def determine_agent_and_extract(user_query: str, input_data: list,
                                      chat_history: Optional[List[Dict[str, str]]] = None) -> tuple[str, List[Dict[str, Any]]]:
    if not global_prompt_manager.prompts:
        raise Exception("Prompt'lar henüz yüklenmemiş! Önce parametre ile prompt yüklenmelidir.")

    system_message = {
        "role": "system",
        "content": f"""
        {global_prompt_manager.get("determine_agent_and_extract_system_prompt")}
        {global_prompt_manager.get("system_prompt")}

        Tour context:
        {json.dumps(input_data, ensure_ascii=False)}

        Previous conversation:
        {json.dumps(chat_history, ensure_ascii=False) if chat_history else "[]"}

        Lütfen yanıtınızı JSON formatında verin. Yanıt şu yapıda olmalı:
        {{
            "agent": "agent_adı",
            "items": ["çıkarılacak_öğeler"]
        }}

        Kullanabileceğiniz agent adları:
        - "isim": Tur isimleri için
        - "konaklama": Konaklama bilgisi için  
        - "vizesiz": Vize durumu için
        - "ulasim": Ulaşım bilgisi için
        - "ziyaretedilecekyerler": Ziyaret yerleri için
        - "geceleme": Gece sayısı için
        - "turkodu": Tur kodu için
        - "url": Tur URL'i için

        Örnek: Kullanıcı "2. turun konaklama bilgisi nedir?" dediğinde:
        {{
            "agent": "konaklama",
            "items": ["2"]
        }}

        Örnek: Kullanıcı "Norveç turunun vize durumu nedir?" dediğinde:
        {{
            "agent": "vizesiz", 
            "items": ["Norveç"]
        }}
        """
    }

    try:
        messages = chat_history.copy() if chat_history else []
        messages.insert(0, system_message)
        messages.append({"role": "user", "content": user_query})

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,  # type: ignore
            temperature=0.1  # Daha tutarlı JSON için
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


def safe_parse_json(model_class, json_data):
    if not global_prompt_manager.prompts:
        raise Exception("Prompt'lar henüz yüklenmemiş! Önce parametre ile prompt yüklenmelidir.")
    try:
        return model_class.model_validate(json_data)
    except ValidationError as e:
        return f"Veri eksik veya hatalı: {e}"


async def get_more_tours(conv_context: ConversationContext) -> Dict[str, Any]:
    print("\n🌟 Sizin için yeni özel seçimler hazırlıyorum...")

    existing_ids = {str(tour.get("id", "")) for tour in input_data}
    print(f"\n📊 Koleksiyonumuzda {len(existing_ids)} özel seçim bulunuyor")

    if conv_context.last_query:
        tour_analysis_result = await agent_manager.tour_analysis_agent.run([{
            "role": "user",
            "content": conv_context.last_query
        }])

        tour_analysis_answer = tour_analysis_result.get("answer", "{}")

        # Tour Analysis Agent sonucu zaten dictionary olarak geliyor
        if isinstance(tour_analysis_answer, dict):
            tour_analysis_data = tour_analysis_answer
            print(f"✅ Tour Analysis Agent sonucu (dict): {tour_analysis_data}")
        elif isinstance(tour_analysis_answer, str):
            try:
                tour_analysis_data = json.loads(tour_analysis_answer)
                print(f"✅ Tour Analysis Agent sonucu (string->dict): {tour_analysis_data}")
            except (json.JSONDecodeError, TypeError):
                print(f"❌ Tour Analysis JSON parse hatası: {tour_analysis_answer}")
                tour_analysis_data = {}
        else:
            print(f"⚠️ Tour Analysis Agent sonucu beklenmeyen tip: {type(tour_analysis_answer)}")
            tour_analysis_data = {}

        # tour_analysis_data'yı filter search için hazırla
        filter_fields = []
        for key, value in tour_analysis_data.items():
            if key in ["turtipi", "vizesiz", "ulasimtipi", "turkodu", "gecesayisi", "kesinkalkis"]:
                filter_fields.append({"field": key, "value": value})
        
        if filter_fields:
            print(f"✅ Filter search kullanılıyor - Alanlar: {[f['field'] for f in filter_fields]}")
            # Filter search yap
            from search.filter_search import create_filter_only_search, search_structured
            
            filter_conditions = create_filter_only_search(filter_fields)
            qdrant_params = {
                "filter": filter_conditions,
                "limit": 5,
                "with_payload": True,
                "with_vector": False
            }
            
            search_result = search_structured(qdrant_params, False)
            
            if search_result.get("success") and search_result.get("data"):
                results = search_result["data"]
            else:
                print("❌ Filter search sonuç vermedi")
                results = []
        else:
            print("✅ Multivector search kullanılıyor")
            # Multivector search yap
            from search.multivector_search import search_tours
            search_result = search_tours(conv_context.last_query, top_k=2)
            
            if search_result.get("success") and search_result.get("data"):
                results = search_result["data"]
            else:
                print("❌ Multivector search sonuç vermedi")
                results = []
        
    else:
        results = []

    new_results = []
    for result in results:
        result_id = str(result.get("id", ""))
        if result_id and result_id not in existing_ids:
            new_results.append(result)
            existing_ids.add(result_id)

    if new_results:
        input_data.extend(new_results)
        print(f"\n✨ {len(new_results)} yeni özel seçim tur koleksiyonumuza eklendi")

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


def get_tour_details_prompt(user_input: str, last_shown_tours: list) -> str:
    if not global_prompt_manager.prompts:
        raise Exception("Prompt'lar henüz yüklenmemiş! Önce parametre ile prompt yüklenmelidir.")

    formatted_tours = []
    for i, tour in enumerate(last_shown_tours, 1):
        tour_name = tour.get('payload', {}).get('isim', 'İsimsiz Tur')
        formatted_tours.append(f"{i}. {tour_name}")

    prompt = f"""
    Kullanıcı şu soruyu sordu: "{user_input}"
    
    Son gösterilen turlar:
    {chr(10).join(formatted_tours)}
    
    {global_prompt_manager.get("get_tour_details_prompt")}
    """
    return prompt


def get_agent_response(prompt: str) -> str:
    try:
        if not global_prompt_manager.prompts:
            raise Exception("Prompt'lar henüz yüklenmemiş! Önce parametre ile prompt yüklenmelidir.")

        system_prompt = global_prompt_manager.get("get_agent_response")
        if not system_prompt:
            system_prompt = "Sen bir tur asistanısın. Kullanıcının hangi turun detaylarını istediğini belirleyip, o turun ID'sini bulmalısın."

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.8
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"❌ Agent yanıtı alınamadı: {str(e)}")
        return "{}"


def find_tour_id_by_name_or_number(user_input: str, input_data: list, last_shown_tours: list) -> Union[str, None]:
    if not global_prompt_manager.prompts:
        raise Exception("Prompt'lar henüz yüklenmemiş! Önce parametre ile prompt yüklenmelidir.")

    print("\n🔍 Tur ID arama başladı...")
    print(f"Kullanıcı girdisi: {user_input}")
    print(f"Son gösterilen tur sayısı: {len(last_shown_tours)}")
    print(f"Toplam tur sayısı: {len(input_data)}")

    if not input_data or len(input_data) == 0:
        print("⚠️ input_data boş, Qdrant'tan veri alınması gerekiyor")
        return None

    print("\n🤖 Agent ile tur ID'si aranıyor...")
    prompt = get_tour_details_prompt(user_input, last_shown_tours)
    response = get_agent_response(prompt)

    try:
        if isinstance(response, str):
            result = json.loads(response)
            tour_id = result.get("tour_id")
            if tour_id:
                print(f"✅ Agent tur ID'sini buldu: {tour_id}")
                for tour in input_data:
                    if str(tour.get("id", "")) == str(tour_id):
                        return str(tour_id)
                print("❌ Agent'ın bulduğu ID geçersiz, manuel aramaya geçiliyor...")
        else:
            print(f"❌ Agent yanıtı string değil: {type(response)}")
    except json.JSONDecodeError as e:
        print(f"❌ Agent yanıtı geçerli JSON formatında değil: {e}")
    except Exception as e:
        print(f"❌ JSON parsing hatası: {e}")

    print("\n📌 Manuel arama yapılıyor...")
    match = re.search(r'(\d+)\.?\s*tur', user_input.lower())
    if match:
        idx = int(match.group(1)) - 1
        print(f"\n📌 Sıra numarası ile arama yapılıyor...")
        print(f"Bulunan sıra numarası: {idx + 1}")

        if 0 <= idx < len(last_shown_tours):
            tour = last_shown_tours[idx]
            tour_name = tour.get('payload', {}).get('isim', 'İsimsiz Tur')
            print(f"✅ {idx + 1}. tur bulundu!")
            print(f"Tur adı: {tour_name}")

            for input_tour in input_data:
                if input_tour.get('payload', {}).get('isim', '').lower() == tour_name.lower():
                    tour_id = str(input_tour.get('id', ''))
                    print(f"✅ Gerçek tur ID'si bulundu: {tour_id}")
                    return tour_id

    print("\n�� İsim ile arama yapılıyor...")
    for tour in last_shown_tours:
        payload = tour.get("payload", {})
        tour_name = payload.get("isim", "").lower()
        if tour_name and tour_name in user_input.lower():
            print(f"✅ Tur ismi ile bulundu!")
            print(f"Tur adı: {tour_name}")

            for input_tour in input_data:
                if input_tour.get('payload', {}).get('isim', '').lower() == tour_name.lower():
                    tour_id = str(input_tour.get('id', ''))
                    print(f"✅ Gerçek tur ID'si bulundu: {tour_id}")
                    return tour_id

    print("\n📌 Tüm turlar içinde arama yapılıyor...")
    for tour in input_data:
        payload = tour.get("payload", {})
        tour_name = payload.get("isim", "").lower()
        if tour_name and tour_name in user_input.lower():
            tour_id = str(tour.get("id", ""))
            print(f"✅ Tüm turlar içinde bulundu!")
            print(f"Tur adı: {tour_name}")
            print(f"Tur ID: {tour_id}")
            return tour_id

    print("\n❌ Tur bulunamadı!")
    return None

