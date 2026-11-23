"""
OpenAI modelleri ile çalışan agent tanımları.
agent_parametre="pro" olduğunda bu agent'lar kullanılır.

Tüm agent'lar Ollama versiyonuyla aynı prompt'ları kullanır,
sadece backend farklıdır (OpenAI API vs vLLM).
"""
import sys
import os

# Parent dizini import path'e ekle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gpt_agents.runner_service import Agent, OPENAI_MODEL
from prompts_ollama import (
    main_instructions,
    fields_instructions,
    output_instructions,
    summary_instructions,
    tour_id_resolve_instructions,
    qdrant_response_instructions,
    reservation_extract_instructions,
    qdrant__id_instructions,
)

# =========================================================
# AGENT TANIMLARI
# =========================================================

main_router_agent = Agent(
    name="MainRouterAgent",
    instructions=main_instructions,
)

question_field_agent = Agent(
    name="QuestionFieldAgent",
    instructions=fields_instructions,
)

tour_information_agent = Agent(
    name="TourInformationAgent",
    instructions=qdrant__id_instructions,
)

response_agent = Agent(
    name="ResponseAgent",
    instructions=output_instructions,
)

summary_agent = Agent(
    name="SummaryAgent",
    instructions=summary_instructions,
)

tour_id_resolver_agent = Agent(
    name="TourIdResolverAgent",
    instructions=tour_id_resolve_instructions,
)

qdrant_response_agent = Agent(
    name="QdrantResponseAgent",
    instructions=qdrant_response_instructions,
)

reservation_info_agent = Agent(
    name="ReservationInfoAgent",
    instructions=reservation_extract_instructions,
)
