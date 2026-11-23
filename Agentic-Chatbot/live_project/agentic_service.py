import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from connection_sql import KeyManager  # <-- Python 3.9 için gerekli
from runner_service import Agent
from prompts_ollama import (
    main_instructions,
    fields_instructions,
    qdrant__id_instructions,
    output_instructions,
    qdrant_response_instructions,
    summary_instructions,
    tour_id_resolve_instructions,
    reservation_extract_instructions,
)

key_manager = KeyManager()
api_list = key_manager.load_api_key()
model_llm = api_list.get('model_ollama', '')

# =========================================================
# region AGENTİC
# =========================================================

# intent classifier
main_router_agent = Agent(
    name="MainRouterAgent",
    instructions=main_instructions,
    model_llm=model_llm
)

# question field and search method
question_field_agent = Agent(
    name="QuestionFieldAgent",      
    instructions=fields_instructions,
    model_llm=model_llm
)

# tour information — tur ID çözümleme (geçmişten)
tour_information_agent = Agent(
    name="TourInformationAgent",
    instructions=qdrant__id_instructions,
    model_llm=model_llm
)   

# qdrant response format — tur verilerini doğal dile çevirme
qdrant_response_agent = Agent(
    name="QdrantResponseAgent",
    instructions=qdrant_response_instructions,
    model_llm=model_llm
)

# response agent — genel konuşma / kapsam dışı yanıtlar
response_agent = Agent(
    name="GeneralResponseAgent",   
    instructions=output_instructions,
    model_llm=model_llm
)

# =========================================================
# YENİ AGENT'LAR
# =========================================================

# summary agent — token overflow durumunda geçmişi özetler
summary_agent = Agent(
    name="SummaryAgent",
    instructions=summary_instructions,
    model_llm=model_llm
)

# tour ID resolver — fuzzy tur referansı çözümleme
# "ilk tur", "şu kıbrıs turu", "az önceki" gibi ifadeleri ID'ye çevirir
tour_id_resolver_agent = Agent(
    name="TourIdResolverAgent",
    instructions=tour_id_resolve_instructions,
    model_llm=model_llm
)

# reservation info extractor — konuşmadan tarih, kişi, yaş bilgisi çıkarma
# Adım adım rezervasyon sürecinde kullanıcının verdiği bilgileri toplar
reservation_info_agent = Agent(
    name="ReservationInfoAgent",
    instructions=reservation_extract_instructions,
    model_llm=model_llm
)