from typing import Union
import json
from typing import Dict, Any,Optional
from qdrant.context import get_context,get_qdrant_client
from core.context import ConversationContext
from value.value import input_data,agent_map
from agent.agent import agent_manager
from db_key import KeyManager
from prompts.handle_db import PromptManager

key_manager = KeyManager()
api_keys = key_manager.load_api_key()
openai_api_key = api_keys.get('gpt_api_key', '')

from openai import OpenAI

client = OpenAI(api_key=openai_api_key)
from typing import List
from pydantic import ValidationError
import re

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
        
        # Konaklama tipi kontrolü
        if context.accommodation_type and payload.get("konaklama") != context.accommodation_type:
            continue
        
        # Gece sayısı kontrolü
        if context.night_count and payload.get("gece") != context.night_count:
            continue
        
        
            
        filtered.append(result)
        context.shown_result_ids.add(result_id)
    
    return filtered
async def get_context_for_question(question: str, conv_context: ConversationContext) -> Dict[str, Any]:
    try:
        # Mesajı analiz et (LLM üzerinden)
        analysis_result = await agent_manager.question_analyzer_agent.run([{
            "role": "user",
            "content": question
        }])
        
        if isinstance(analysis_result, str):
            analysis_result = json.loads(analysis_result)

        answer = analysis_result.get("answer", {})
        detail_field = answer.get("detail_field")
        tour_category = answer.get("category")
        type_value = answer.get("type")

        context = {
            "detail_field": detail_field,
            "tour_category": tour_category,
            "type": type_value,
            "shown_result_ids": list(conv_context.shown_result_ids),
            "last_shown_tours": conv_context.last_shown_tours
        }

        print(f"\n📊 Bağlam bilgileri:")
        print(f"   Tip: {type_value}")
        print(f"   Kategori: {tour_category}")
        print(f"   Detay alanı: {detail_field}")

        # Tur listesi sorgusu veya tur detayı sorgusu ise Qdrant'tan veri al
        if type_value == "tour_info" or type_value == "tour_list" or "tur" in question.lower():
            print(f"\n🔍 Qdrant sorgusu yapılıyor: {question}")
            results = await get_context(question, get_qdrant_client())
            context["results"] = results
            print(f"✅ Qdrant'tan {len(results)} sonuç alındı")
        else:
            print(f"\n⚠️ Qdrant sorgusu yapılmadı (tip: {type_value})")

        return context

    except Exception as e:
        print(f"❌ Bağlam oluşturma hatası: {str(e)}")
        return {}
def clean_json(response_content: Optional[str]) -> Optional[Dict[str, Any]]:
    if not response_content:
        return None
    try:
        response_content = re.sub(r"```json\s*|\s*```", "", response_content).strip()
        return json.loads(response_content)
    except json.JSONDecodeError:
        return None
async def determine_agent_and_extract(user_query: str, input_data: list, chat_history: Optional[List[Dict[str, str]]] = None) -> tuple[str, List[Dict[str, Any]]]:
    # Varsayılan olarak pro kullan

    system_message = {
        "role": "system",
        "content": f"""
        {local_prompt_manager.get("determine_agent_and_extract_system_prompt")}
        {local_prompt_manager.get("system_prompt")}

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
def safe_parse_json(model_class, json_data):
    
    try:
        return model_class.model_validate(json_data)
    except ValidationError as e:
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
            q_client=get_qdrant_client(),
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
def get_tour_details_prompt(user_input: str, last_shown_tours: list) -> str:
    """
    Agent'a tur detaylarını sormak için prompt hazırla
    """
    # Son gösterilen turları daha okunabilir formatta hazırla
    formatted_tours = []
    for i, tour in enumerate(last_shown_tours, 1):
        tour_name = tour.get('payload', {}).get('isim', 'İsimsiz Tur')
        formatted_tours.append(f"{i}. {tour_name}")
    
    prompt = f"""
    Kullanıcı şu soruyu sordu: "{user_input}"
    
    Son gösterilen turlar:
    {chr(10).join(formatted_tours)}
    
  
    {local_prompt_manager.get("get_tour_details_prompt")}

    """
    return prompt
def get_agent_response(prompt: str) -> str:
    """
    Agent'dan yanıt al
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Sen bir tur asistanısın. Kullanıcının hangi turun detaylarını istediğini belirleyip, o turun ID'sini bulmalısın."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.8
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"❌ Agent yanıtı alınamadı: {str(e)}")
        return "{}"
def find_tour_id_by_name_or_number(user_input: str, input_data: list, last_shown_tours: list) -> Union[str, None]:
    """
    Kullanıcının belirttiği turun id'sini bul.
    Önce agent ile dene, başarısız olursa manuel aramaya geç.
    """
    print("\n🔍 Tur ID arama başladı...")
    print(f"Kullanıcı girdisi: {user_input}")
    print(f"Son gösterilen tur sayısı: {len(last_shown_tours)}")
    print(f"Toplam tur sayısı: {len(input_data)}")
    
    # Önce agent ile dene
    print("\n🤖 Agent ile tur ID'si aranıyor...")
    prompt = get_tour_details_prompt(user_input, last_shown_tours)
    response = get_agent_response(prompt)
    
    try:
        # Response'un string olduğundan emin ol
        if isinstance(response, str):
            result = json.loads(response)
            tour_id = result.get("tour_id")
            if tour_id:
                print(f"✅ Agent tur ID'sini buldu: {tour_id}")
                # ID'nin geçerli olduğunu kontrol et
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
    
    # Agent başarısız olursa manuel aramaya geç
    print("\n📌 Manuel arama yapılıyor...")
    import re
    
    # Önce sıra numarası ile arama (örn: "2. tur", "2 tur", "2.tur")
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
            
            # Tur adını kullanarak input_data içinden gerçek ID'yi bul
            for input_tour in input_data:
                if input_tour.get('payload', {}).get('isim', '').lower() == tour_name.lower():
                    tour_id = str(input_tour.get('id', ''))
                    print(f"✅ Gerçek tur ID'si bulundu: {tour_id}")
                    return tour_id
    
    # Sonra isim ile arama
    print("\n📌 İsim ile arama yapılıyor...")
    print("Son gösterilen turlar içinde aranıyor...")
    for tour in last_shown_tours:
        payload = tour.get("payload", {})
        tour_name = payload.get("isim", "").lower()
        if tour_name and tour_name in user_input.lower():
            print(f"✅ Tur ismi ile bulundu!")
            print(f"Tur adı: {tour_name}")
            
            # Tur adını kullanarak input_data içinden gerçek ID'yi bul
            for input_tour in input_data:
                if input_tour.get('payload', {}).get('isim', '').lower() == tour_name.lower():
                    tour_id = str(input_tour.get('id', ''))
                    print(f"✅ Gerçek tur ID'si bulundu: {tour_id}")
                    return tour_id
    
    # Son gösterilen turlarda bulunamazsa, tüm input_data içinde ara
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
