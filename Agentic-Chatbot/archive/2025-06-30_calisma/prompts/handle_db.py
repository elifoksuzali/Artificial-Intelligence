# prompts/handle_db.py

import mysql.connector as mysql
from mysql.connector import Error
import logging

DB_CONFIG = {
    "host": "db.example.com",
    "user": "demo_user",
    "password": "demoPassw0rd123",
    "database": "demo_db"
}

PROMPT_NAMES = [
    "repl_agent_system_prompt",
    "query_router_agent_system_prompt",
    "question_analyzer_system_prompt",
    "response_validator_agent_system_prompt",
    "determine_agent_and_extract_system_prompt",
    "system_prompt",
    "follow_up_agent_system_prompt",
    "get_agent_response",
    "get_tour_details_prompt",
    "qdrant_search_system_prompt",
    "agent_map_system_prompt",
    "chat_history_system_prompt",
    "mcp_service_agent_system_prompt",
    "greeting_agent_system_prompt"
]

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class PromptManager:
    def __init__(self):
        self.prompts = {}
        self.parametre = None

    async def load_prompts(self, parametre: str) -> dict:
        print(f"[PromptManager] Parametre alındı: {parametre}")
        aitip = "gpt-4.1" if parametre == "pro" else "gpt-4o-mini" if parametre == "free" else None
        if aitip is None:
            logging.warning(f"[PromptManager] Geçersiz parametre: {parametre}")
            return {}

        self.parametre = parametre

        try:
            conn = mysql.connect(**DB_CONFIG, charset="utf8")
            if not conn.is_connected():
                logging.error("[PromptManager] Veritabanına bağlanılamadı.")
                return {}

            cursor = conn.cursor()
            cursor.execute("""
                SELECT tip, prompt FROM kurum_ai_prompt
                WHERE aciklama = 'harbiAI' AND aitip = %s
            """, (aitip,))
            results = cursor.fetchall()

            self.prompts = {
                row[0]: row[1]
                for row in results
                if row[0] in PROMPT_NAMES
            }

            logging.info(f"[PromptManager] {len(self.prompts)} prompt yüklendi.")
            return self.prompts

        except Error as e:
            logging.error(f"[PromptManager] SQL Hatası: {e}")
            return {}

        finally:
            if conn:
                conn.close()

    def get(self, prompt_name: str) -> str:
        return self.prompts.get(prompt_name, "")
