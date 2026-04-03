import openai
import json
import re
import agent.agent
from prompting.chat_utils import system_prompt as custom_rules, response_format

openai_api_key = "openai_api_key"
client = openai.OpenAI(api_key=openai_api_key)

agent_map = {
    "id": agent.agent.ResearchPaperExtractionID,
    "isim": agent.agent.ResearchPaperExtractionName,
    "turkodu": agent.agent.ResearchPaperExtractionTurKodu,
    "geceleme": agent.agent.ResearchPaperExtractionGeceSayisi,
    "konaklama": agent.agent.ResearchPaperExtractionKonaklama,
    "ulasim": agent.agent.ResearchPaperExtractionUlasim,
    "ziyaretedilecekyerler": agent.agent.ResearchPaperExtractionZiyaretEdilecekYerler,
    "vizesiz": agent.agent.ResearchPaperExtractionVizeDurumu,
    "kesinkalkis": agent.agent.ResearchPaperExtractionKesinkalkis,
    "url": agent.agent.ResearchPaperExtractionUrl
}

def determine_agent_and_extract(user_query: str, input_data: list, chat_history: list = None):
    system_message = {
        "role": "system",
        "content": f"""
        You are a helpful assistant that receives a user question and a list of travel tours in JSON format.
        Your job is to:
        1. Identify which field (like 'isim', 'id', 'turkodu', etc.) the user is asking about.
        2. If the question refers to a previous tour like “3. tur”, “bu tur”, etc., resolve it using previous user/assistant messages.
        3. Return the result as a JSON response under the 'items' key.
        4. Do not use "items" as the value for "agent". "agent" must be one of the actual field names being extracted.
        5. Return the result strictly in the format:

        {{
            "agent": "<field_name>",
            "items": [
            {{ "<field_name>": "..." }}
            ]
        }}

        {custom_rules}

        Tour context:
        {json.dumps(input_data, ensure_ascii=False)}
        """
            }

    try:
        messages = chat_history.copy() if chat_history else []
        messages.insert(0, system_message)  # system mesajı en başa eklenir
        messages.append({"role": "user", "content": user_query})

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            response_format=response_format
        )

        raw = response.choices[0].message.content
        print(f"\n🌟 GPT Raw Output:\n{raw}")

        data = clean_json(raw)
        agent_name = data.get("agent")
        items = data.get("items", [])

        if agent_name not in agent_map:
            print("⚠️ Bilinmeyen agent adı, varsayılan 'isim' kullanılıyor.")
            agent_name = "isim"

        return agent_name, items

    except Exception as e:
        print(f"❌ GPT agent/cevap hatası: {e}")
        return "isim", []

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


chat_history = [{"role": "system", "content": "You are a helpful assistant for travel tours."}]


def process_query(user_query: str, input_data: list):
    
    chat_history.append({"role": "user", "content": user_query})  # Kullanıcı mesajı geçmişe ekleniyor

    agent_name, items = determine_agent_and_extract(user_query, input_data, chat_history)
    if not items:
        return "Cevap bulunamadı veya GPT boş döndü."

    # GPT cevabını da geçmişe ekliyoruz ki sonraki sorularda hatırlasın
    chat_history.append({
        "role": "assistant",
        "content": json.dumps({"agent": agent_name, "items": items}, ensure_ascii=False)
    })

    parsed = [safe_parse_json(agent_map[agent_name], item) for item in items]
    return parsed

