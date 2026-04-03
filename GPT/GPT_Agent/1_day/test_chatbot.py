from agents import Agent, InputGuardrail, GuardrailFunctionOutput, Runner
from pydantic import BaseModel
import asyncio
import openai
import json
import re
import agent.agent as agent
from prompting.chat_utils import system_prompt as custom_rules, response_format
import os
import sys
from typing import List

# Qdrant Setup
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from database.qdrantDb import QdrantDatabase

qdrant_db = QdrantDatabase(collection_name="mng-cosine")

openai_api_key = "OPENAI_API_KEY"
os.environ["OPENAI_API_KEY"] = openai_api_key
client = openai.OpenAI(api_key=openai_api_key)


def get_embedding(text: str) -> List[float]:
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding


def get_context(query: str, q_client: QdrantDatabase, num_results: int = 5) -> list:
    query_vector = get_embedding(query)
    results = q_client.query_points(
        vector=query_vector,
        top_k=num_results
    )
    return [result.payload or {} for result in results]


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

class TourIntentOutput(BaseModel):
    is_tour_related: bool
    reasoning: str

guardrail_agent = Agent(
    name="Guardrail Check",
    instructions="""
    Determine whether the user's query is about a tour-related topic.
    
    Consider a query tour-related if it:
    1. Contains tour-related keywords like tur, gezi, seyahat, konaklama, uçak, otel, etc.
    2. References a specific tour by number (e.g., '4. turun...', 'birinci tur', etc.)
    3. Asks about tour details like tur kodu, id, fiyat, etc.
    4. Follows up on previously mentioned tours
    
    Chat history is important context - if the user is following up on a previous tour-related query,
    treat it as tour-related even if it doesn't explicitly mention tours.
    """,
    output_type=TourIntentOutput,
)

async def tour_guardrail(ctx, agent, user_input):
    # Check if this is a follow-up question referencing a numbered tour
    chat_history = ctx.context.get("chat_history", [])
    
    # Look for patterns like "X. tur" or "tur X" where X is a number
    is_tour_specific_question = re.search(r'(\d+)\.\s*tur|tur\s*(\d+)', user_input.lower())
    
    # If there's tour history and this appears to be a follow-up
    has_tour_history = False
    for msg in chat_history:
        if msg['role'] == 'assistant' and ('✈️' in msg['content'] or 'tur' in msg['content'].lower()):
            has_tour_history = True
            break
    
    # Additional tour-related keywords to check
    tur_related_keywords = ["tur", "gezi", "seyahat", "konaklama", "uçak", "otel", "id", "kod", "numara", 
                           "fiyat", "tarih", "kalkış", "vizesiz", "vize", "ziyaret"]
    has_keywords = any(keyword in user_input.lower() for keyword in tur_related_keywords)
    
    # Automatically pass if it's a follow-up to a tour question or contains tour keywords
    if is_tour_specific_question or (has_tour_history and len(chat_history) > 1) or has_keywords:
        return GuardrailFunctionOutput(
            output_info=TourIntentOutput(
                is_tour_related=True,
                reasoning="Bu, önceki tur sorgusuyla ilgili bir takip sorusu veya spesifik bir tur sorusu."
            ),
            tripwire_triggered=False,
        )
    
    # If not an obvious tour question, use the guardrail agent to check
    result = await Runner.run(guardrail_agent, user_input, context=ctx.context)
    final_output = result.final_output_as(TourIntentOutput)
    return GuardrailFunctionOutput(
        output_info=final_output,
        tripwire_triggered=not final_output.is_tour_related,
    )

# --- GPT tabanlı veri çıkarıcı ajan ---
gpt_extraction_agent = Agent(
    name="GPT Tour Field Extractor",
    handoff_description="Agent that selects the relevant field and extracts values based on user query.",
)

triage_agent = Agent(
    name="Triage Agent",
    instructions="Determine whether a tour-related question is being asked and hand off to the GPT extraction agent.",
    handoffs=[gpt_extraction_agent],
    input_guardrails=[InputGuardrail(guardrail_function=tour_guardrail)],
)

def clean_json(response_content: str):
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

chat_history = [
    {"role": "system", "content": "Sen MNG'ye ait müşteri desteği sohbet robotusun. İsmini sorarlarsa veya nasılsın, günün nasıl gidiyor gibi sorular sorulursa, kibarca iyi olduğunu belirt ve MNG Asistan Yapay Zeka olduğunu söyle."}
]

def summarize_chat_history():
    user_questions = []
    for msg in chat_history:
        if msg['role'] == 'user':
            user_questions.append(msg['content'])
    
    if not user_questions:
        return "Henüz hiç soru sormadınız."
    
    summary = "Size özetlemek gerekirse, şu ana kadar şu soruları sordunuz:\n\n"
    for i, question in enumerate(user_questions, 1):
        summary += f"{i}. {question}\n"
    
    return summary

async def run_query(user_query: str):
    # Özetleme isteği kontrolü
    if user_query.lower() in ["özetle", "ne sordum", "neler sordum", "az önce ne sordum", "şimdiye kadar ne sordum"]:
        return summarize_chat_history()
    
    print("\nChat History:")
    print("=" * 50)
    for msg in chat_history:
        print(f"{msg['role']}: {msg['content']}")
    print("=" * 50)
    
    # Kullanıcı sorusunu chat history'ye ekle
    chat_history.append({"role": "user", "content": user_query})
    
    # Qdrant'tan içerik çek
    input_data = get_context(user_query, q_client=qdrant_db, num_results=5)
    
    # Tur numarası kontrolü - eğer "X. tur" gibi bir referans varsa, o turu içerikten bul
    specific_tour_match = re.search(r'(\d+)\.\s*tur', user_query.lower())
    specific_tour_index = None
    
    if specific_tour_match:
        specific_tour_index = int(specific_tour_match.group(1)) - 1
        
        # Son yanıtı bul
        last_assistant_msg = None
        for msg in reversed(chat_history[:-1]):  # Son eklenen kullanıcı mesajını hariç tut
            if msg['role'] == 'assistant':
                last_assistant_msg = msg['content']
                break
        
        # Son yanıttan turları çıkar
        if last_assistant_msg:
            tour_lines = re.findall(r'\[\d+\] ✈️ .+', last_assistant_msg)
            if specific_tour_index < len(tour_lines):
                # Spesifik tur ismi
                tour_name = re.sub(r'\[\d+\] ✈️ ', '', tour_lines[specific_tour_index])
                # Kullanıcı sorgusunu zenginleştir
                enhanced_query = f"{user_query} - {tour_name}"
                print(f"Enhanced query: {enhanced_query}")
                
                # Zenginleştirilmiş sorgu ile yeniden içerik çek
                input_data = get_context(enhanced_query, q_client=qdrant_db, num_results=5)
    
    # GPT agent'ın instructions'ını dinamik olarak oluştur
    gpt_extraction_agent.instructions = f"""
        Sen MNG'ye ait müşteri desteği sohbet robotusun. İsmini sorarlarsa veya nasılsın, günün nasıl gidiyor gibi sorular sorulursa, kibarca iyi olduğunu belirt ve MNG Asistan Yapay Zeka olduğunu söyle.

        Yanıtlarında şu kurallara uy:
        1. Her zaman samimi ve sıcak bir ton kullan 😊
        2. Emojiler kullanarak yanıtlarını renklendir:
           - ✈️ uçak turları için
           - 🏨 otel turları için
           - 🌍 yurt dışı turları için
           - 🏖️ deniz turları için
           - 🗺️ gezi turları için
           - 😊 genel ifadeler için
           - 💫 özel turlar için
           - 🎉 eğlenceli turlar için
           - 🏛️ kültür turları için
           - 🍽️ yemek turları için

        3. Yanıt formatı:
           - Önce samimi bir giriş yap
           - Sonra gerekçeli açıklama
           - Turları numaralandırılmış liste şeklinde göster:
             [1] ✈️ Tur Adı
             [2] ✈️ Tur Adı
             [3] ✈️ Tur Adı
           - Sonunda nazik bir kapanış

        4. Veri işleme kuralları:
           - Sadece izin verilen kolonları kullan: [isim], [id], [geceleme], [konaklama], [ulasim], [ziyaretedilecekyerler], [puan], [vizesiz], [turtipi], [turKategori], [kategori_isim], [ulasimtipi]
           - Kullanıcı detay istemedikçe sadece [isim] değerlerini göster
           - En fazla 10 tur göster
           - Büyük/küçük harf duyarlılığına dikkat et
           - Kullanıcının girdiği kelimeleri bölme, bütün halde sorgula

        5. Konuşma geçmişini dikkate al ve ona göre yanıt ver
        6. Tur numarası belirtildiğinde (örn: "2. turun idsi nedir"), o tura ait bilgileri bul
        7. Vize durumu için [vizesiz] kolonunu kontrol et (1: vize gerekmez, 0: vize gerekir)

        Örnek yanıt formatı:
        Merhaba! Size yurt dışı turlarımızdan bazılarını sunmaktan mutluluk duyarım! 😊

        Yurt dışına çıkmak için harika seçeneklerimiz var. Şunları inceledim:

        [1] ✈️ Balkanlar 5 Ülke Rüyası Turu THY İle 7 Gece Ekstra Turlar Dahil
        [2] ✈️ Elit Orta Avrupa 5 Ülke Turu THY ile 7 Gece Ekstra Turlar Dahil Kurban Bayramı Dönemi
        [3] ✈️ Comfort Bali Turu THY ile 5 Gece Ekstra Turlar Dahil
        [4] ✈️ Baştanbaşa Balkanlar Turu THY ile 7 Gece - Tüm Çevre Gezileri, Ekstra Turlar, Akşam Yemekleri Dahil
        [5] ✈️ Elegant Ürdün ve Petra Turu THY ile 4 Gece Ekstra Turlar Dahil

        Dilerseniz daha fazla tur programı sunabilirim. 😊

        Previous conversation:
        {json.dumps(chat_history, ensure_ascii=False)}

        Tour context:
        {json.dumps(input_data, ensure_ascii=False)}

        {custom_rules}
    """
    
    context = {
        "chat_history": chat_history.copy(),
        "input_data": input_data
    }
    
    print("\nSending context to GPT...")
    try:
        result = await Runner.run(triage_agent, user_query, context=context)
        print("\nGPT Response:", result.final_output)
        content = result.final_output
        
        # GPT yanıtını işle
        try:
            # Assistant yanıtını chat history'ye ekle
            chat_history.append({"role": "assistant", "content": content})
            
            # Yanıtı formatla
            try:
                # JSON formatını düzelt - eğer JSON varsa
                json_start = content.find('[')
                json_end = content.rfind(']') + 1
                if json_start != -1 and json_end != -1 and '[' in content and ']' in content:
                    json_str = content[json_start:json_end]
                    try:
                        data = json.loads(json_str)
                        
                        # Yanıtı formatla
                        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict) and "isim" in data[0]:
                            response = []
                            for i, item in enumerate(data, 1):
                                # Tur tipine göre emoji seç
                                emoji = "✈️"  # varsayılan
                                if "ulasimtipi" in item:
                                    if "uçak" in item["ulasimtipi"].lower():
                                        emoji = "✈️"
                                    elif "gemi" in item["ulasimtipi"].lower():
                                        emoji = "🛳️"
                                    elif "otobüs" in item["ulasimtipi"].lower():
                                        emoji = "🚌"
                                
                                # Tur adını ve detaylarını birleştir
                                tur_detay = f"{item['isim']}"
                                if "geceleme" in item:
                                    tur_detay += f" {item['geceleme']} Gece"
                                if "konaklama" in item:
                                    tur_detay += f" - {item['konaklama']}"
                                
                                response.append(f"[{i}] {emoji} {tur_detay}")
                            
                            formatted_response = "\n".join(response)
                            
                            # Giriş ve kapanış cümlelerini ekle
                            if content.split(json_str)[0].strip():
                                formatted_response = content.split(json_str)[0].strip() + "\n\n" + formatted_response
                            if content.split(json_str)[1].strip():
                                formatted_response = formatted_response + "\n\n" + content.split(json_str)[1].strip()
                            
                            return formatted_response
                    except:
                        # JSON işlenemezse orijinal içeriği döndür
                        return content
                else:
                    return content
            except:
                return content
        except Exception as e:
            error_response = f"Üzgünüm, bir hata oluştu: {str(e)}"
            # Hata yanıtını chat history'ye ekle
            chat_history.append({"role": "assistant", "content": error_response})
            return error_response
    except Exception as e:
        error_response = f"Üzgünüm, bir hata oluştu: {str(e)}"
        # Hata yanıtını chat history'ye ekle
        chat_history.append({"role": "assistant", "content": error_response})
        return error_response

async def main():
    try:
        while True:
            user_query = input("Sormak istediğiniz bilgiyi girin (Çıkış için 'q' yazın): ")
            if user_query.lower() == 'q':
                break
            result = await run_query(user_query)
            print(f"Extracted Data: {result}")
    except KeyboardInterrupt:
        print("\nProgram sonlandırılıyor...")
    except Exception as e:
        print(f"Bir hata oluştu: {str(e)}")
    finally:
        # Event loop'u düzgün şekilde kapat
        loop = asyncio.get_event_loop()
        loop.close()

if __name__ == "__main__":
    asyncio.run(main())